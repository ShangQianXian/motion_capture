"""The 2D input scale is a capture setting, so it must survive the whole hop.

A UI control is only real if the value it stores reaches the worker. This test
follows it across every boundary it crosses: the ``EnumProperty`` default and
its declared values, the review cache key, and the emitted job file -- plus the
validator the worker runs on that file.
"""
import os
import tempfile
import unittest
from pathlib import Path

from ._util import ADDON_ROOT, TempDirCase, preferences, scene_props

from core import job_schema, preview


#: Mirror of ``addon/properties.py``. Kept as data so a mismatch is a failure
#: rather than something that silently drifts; ``tests/blender`` renders the
#: real property.
PROPERTY_VALUES = ('current', 'canonical')
PROPERTY_DEFAULT = 'current'


class SceneDefaultMatchesSchema(TempDirCase):
    def test_helper_defaults_are_the_documented_ones(self):
        """``_util.scene_props`` stands in for the Blender property group."""
        self.assertEqual(scene_props().input_normalisation, PROPERTY_DEFAULT)
        self.assertEqual(PROPERTY_DEFAULT, 'current')

    def test_property_source_declares_exactly_these_values(self):
        source = Path(ADDON_ROOT, 'addon', 'properties.py').read_text(encoding='utf-8')
        start = source.index('input_normalisation: EnumProperty(')
        # Stop at the next top-level assignment, so the block is this property
        # and nothing after it.
        rest = source[start + 1:]
        end = start + 1 + min(
            (rest.index(marker) for marker in ('\n    camera_view', '\n    motion_type')
             if marker in rest), default=len(rest))
        block = source[start:end]
        for value in PROPERTY_VALUES:
            self.assertIn("('{0}'".format(value), block)
        self.assertIn("default='{0}'".format(PROPERTY_DEFAULT), block)
        # It must invalidate: a lifter input change makes an old result wrong.
        self.assertIn('update=_capture_changed', block)


class JobCarriesTheSetting(TempDirCase):
    def _job(self, **overrides):
        source = self.path('source.mp4')
        Path(source).touch()
        return job_schema.build_job(
            scene_props(source_media=source, **overrides),
            preferences(models_root=self.tmp),
            blend_path=self.path('test.blend'))

    def test_default_value_is_emitted_and_accepted(self):
        job = self._job()
        self.assertEqual(job['options']['input_normalisation'], PROPERTY_DEFAULT)
        job_schema.validate_job(job)

    def test_explicit_value_is_emitted_and_accepted(self):
        for value in PROPERTY_VALUES:
            with self.subTest(value=value):
                job = self._job(input_normalisation=value)
                self.assertEqual(job['options']['input_normalisation'], value)
                job_schema.validate_job(job)

    def test_an_unknown_value_never_reaches_the_worker(self):
        from core import errors
        job = self._job()
        job['options']['input_normalisation'] = 'guess'
        with self.assertRaises(errors.MocapError) as caught:
            job_schema.validate_job(job)
        self.assertEqual(caught.exception.code, errors.JOB_SCHEMA_INVALID)


class ReviewCacheKeyIncludesTheSetting(TempDirCase):
    def test_snapshot_carries_the_field(self):
        snapshot = preview.settings_snapshot(scene_props())
        self.assertIn('input_normalisation', preview.CAPTURE_FIELDS)
        self.assertEqual(snapshot['input_normalisation'], PROPERTY_DEFAULT)

    def test_snapshot_reads_the_stored_value(self):
        snapshot = preview.settings_snapshot(scene_props(input_normalisation='canonical'))
        self.assertEqual(snapshot['input_normalisation'], 'canonical')

    def test_missing_field_falls_back_so_old_results_still_load(self):
        """Results written before this setting existed must not be rejected."""
        class Legacy(object):
            source_type = 'video'
            capture_profile = 'quality'
            frame_start = 1
            frame_end = 0
            target_fps = 30
            include_hands = False
            smoothing_strength = 0.65
            foot_lock_strength = 0.7
            root_motion = 'in_place'
            motion_type = 'walk'
            camera_view = 'left_front_45'
            align_initial_facing = True

        snapshot = preview.settings_snapshot(Legacy())
        self.assertEqual(snapshot['input_normalisation'], PROPERTY_DEFAULT)
        self.assertEqual(set(snapshot), set(preview.CAPTURE_FIELDS))

    def test_signature_changes_with_the_setting(self):
        first = preview.signature(preview.settings_snapshot(scene_props(
            input_normalisation='current')))
        second = preview.signature(preview.settings_snapshot(scene_props(
            input_normalisation='canonical')))
        self.assertNotEqual(first, second)

    def test_settings_comparison_detects_a_change(self):
        self.assertTrue(preview.settings_equal(
            preview.settings_snapshot(scene_props(input_normalisation='current')),
            preview.settings_snapshot(scene_props(input_normalisation='current'))))
        self.assertFalse(preview.settings_equal(
            preview.settings_snapshot(scene_props(input_normalisation='current')),
            preview.settings_snapshot(scene_props(input_normalisation='canonical'))))


class WorkerReadsTheSetting(unittest.TestCase):
    def test_pipeline_passes_it_through_and_records_it(self):
        """The worker must forward the option and publish what it used."""
        source = Path(ADDON_ROOT, 'backend_worker', 'pipeline.py').read_text(encoding='utf-8')
        self.assertIn('options.get("input_normalisation")', source)
        self.assertIn('input_normalisation=normalisation', source)
        self.assertIn("diagnostics['lifter_input_normalisation'] = normalisation", source)

    def test_the_lifter_accepts_every_declared_value(self):
        from backend_worker import pose3d_motionbert as adapter
        declared = (adapter.NORMALISATION_CURRENT, adapter.NORMALISATION_CANONICAL)
        self.assertEqual(declared, PROPERTY_VALUES)


if __name__ == '__main__':
    unittest.main()
