"""A fresh worker result must match after reloading an older UI module cache."""
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).parent))
import _harness as h


def main():
    module, used = h.enable_addon()
    from motion_capture.core import camera_alignment, orientations, preview, motion_processing

    current_version = orientations.VERSION
    manifest = {'source': {}, 'processing_version': current_version}
    # Simulate Blender retaining modules from before the files were updated.
    orientations.VERSION = 'previous-version'
    motion_processing.VERSION = 'previous-version'
    camera_alignment.MAX_HEADING_CORRECTION = -1
    state = preview.ReviewState(None, manifest, {}, '')
    h.check(not state.matches({}, ''), 'cached old UI version reproduces rejection of a fresh result')
    h.disable_addon(module, used)
    module._reload_submodules()
    module, used = h.enable_addon()
    h.check(orientations.VERSION == current_version, 'reload refreshes the orientation module')
    h.check(motion_processing.VERSION == current_version, 'processing reload follows its version dependency')
    h.check(camera_alignment.MAX_HEADING_CORRECTION > 0, 'reload refreshes camera alignment')
    state = preview.ReviewState(None, manifest, {}, '')
    h.check(state.matches({}, ''), 'fresh worker output matches after plugin reload')
    h.disable_addon(module, used)
    h.finish()


if __name__ == '__main__':
    main()
