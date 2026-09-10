"""Installer validation must precede mutation, including for existing venvs."""
import contextlib
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools import bootstrap_worker_env as bootstrap


class BootstrapTests(unittest.TestCase):
    def test_rejects_wrong_python_before_installing(self):
        for info in ([[3, 12, 3], 64], [[3, 10, 1], 64], [[3, 10, 0], 32]):
            with patch.object(bootstrap.subprocess, 'check_output', return_value=json.dumps(info)), \
                    patch.object(bootstrap, 'run') as execute:
                with self.assertRaises(RuntimeError):
                    bootstrap.main(['--python-exe', sys.executable])
                execute.assert_not_called()

    def test_existing_venv_is_validated_before_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / '.venv/Scripts/python.exe'
            python.parent.mkdir(parents=True)
            python.touch()
            versions = ['[[3,10,0],64]', '[[3,12,3],64]']
            with patch.object(bootstrap, 'ROOT', root), \
                    patch.object(bootstrap.subprocess, 'check_output', side_effect=versions), \
                    patch.object(bootstrap, 'run') as execute:
                with self.assertRaises(RuntimeError):
                    bootstrap.main(['--environment', 'quality'])
                execute.assert_not_called()

    def test_dry_run_keeps_target_paths_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = StringIO()
            with patch.object(bootstrap, 'ROOT', root), \
                    patch.object(bootstrap, 'inspect_python'), \
                    patch.object(bootstrap.subprocess, 'run') as execute, contextlib.redirect_stdout(output):
                bootstrap.main(['--dry-run'])
            execute.assert_not_called()
            self.assertFalse((root / '.venv').exists())
            self.assertIn(str(root / '.venv-preview/Scripts/python.exe'), output.getvalue())

    def test_native_install_failure_propagates(self):
        import subprocess
        with patch.object(bootstrap.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, ['pip'])):
            with self.assertRaises(subprocess.CalledProcessError):
                bootstrap.run(['pip', 'install', 'missing'])
