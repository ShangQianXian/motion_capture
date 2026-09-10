"""Build the two Windows worker environments without touching system Python."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = (3, 10, 0)


def clean_env():
    env = dict(os.environ)
    for key in list(env):
        if key.upper() in {"PYTHONHOME", "PYTHONPATH"}:
            del env[key]
    env.update(PYTHONNOUSERSITE="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    # Windows' registry proxy shorthand may label an HTTP CONNECT endpoint
    # "https://" for HTTPS destinations. urllib handles it, requests does not.
    proxies = urllib.request.getproxies()
    http = proxies.get("http", "")
    https = proxies.get("https", "")
    if (http and https and "HTTPS_PROXY" not in env and "https_proxy" not in env
            and urlsplit(http).scheme == "http" and urlsplit(https).scheme == "https"
            and urlsplit(http).netloc == urlsplit(https).netloc):
        env["HTTPS_PROXY"] = http
    return env


def inspect_python(executable):
    code = "import sys,struct,json;print(json.dumps([list(sys.version_info[:3]),struct.calcsize('P')*8]))"
    value = subprocess.check_output([str(executable), "-I", "-c", code], env=clean_env(), text=True)
    version, bits = json.loads(value)
    if tuple(version) != EXPECTED or bits != 64:
        raise RuntimeError(f"Expected Python 3.10.0 x64; {executable} reports {version}, {bits} bit. Use a new environment directory.")


def run(args, dry_run=False):
    args = [str(arg) for arg in args]
    print(("[dry-run] " if dry_run else "> ") + subprocess.list2cmdline(args), flush=True)
    if not dry_run:
        subprocess.run(args, cwd=ROOT, env=clean_env(), check=True)


def bootstrap_pip(python, dry_run=False):
    """Use a verified pip wheel to upgrade the old pip bundled with Python 3.10.0.

    pip 21.2.3 has a TLS-proxy bug on this Python release. stdlib urllib does
    not use that vendored TLS stack. Certificate verification stays enabled.
    """
    if dry_run:
        print("[dry-run] bootstrap pip 24.3.1 from its SHA256-verified PyPI wheel")
        return
    installed = subprocess.run([str(python), "-I", "-c", "import pip;print(pip.__version__)"],
                               capture_output=True, text=True, env=clean_env())
    if installed.returncode == 0 and installed.stdout.strip() == "24.3.1":
        return
    with urllib.request.urlopen("https://pypi.org/pypi/pip/24.3.1/json", timeout=60) as response:
        metadata = json.load(response)
    wheel = next(item for item in metadata["urls"] if item["filename"].endswith(".whl"))
    cache = ROOT / ".cache" / "bootstrap"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / wheel["filename"]
    if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != wheel["digests"]["sha256"]:
        with urllib.request.urlopen(wheel["url"], timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != wheel["digests"]["sha256"]:
            raise RuntimeError("pip wheel SHA256 mismatch")
        target.write_bytes(data)
    code = "import sys,runpy;sys.path.insert(0,sys.argv.pop(1));runpy.run_module('pip',run_name='__main__')"
    run([python, "-I", "-c", code, target, "--isolated", "install", "--no-index", target])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--environment", choices=("all", "quality", "preview"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    base = Path(args.python_exe).resolve()
    inspect_python(base)
    targets = ("quality", "preview") if args.environment == "all" else (args.environment,)
    for target in targets:
        directory = ROOT / (".venv" if target == "quality" else ".venv-preview")
        python = directory / "Scripts" / "python.exe"
        if python.exists():
            inspect_python(python)
        elif directory.exists():
            raise RuntimeError(f"Incomplete environment: {directory}. Choose a clean directory before retrying.")
        else:
            run([base, "-m", "venv", directory], args.dry_run)
        bootstrap_pip(python, args.dry_run)
        pip = [python, "-m", "pip", "--isolated"]
        run(pip + ["install", "-r", ROOT / "requirements/build.txt"], args.dry_run)
        if target == "quality":
            # Chumpy's sdist imports numpy and pip during setup. It has no native extension.
            run(pip + ["install", "--only-binary=:all:", "numpy==1.23.5", "scipy==1.10.1", "six==1.17.0"], args.dry_run)
            run(pip + ["install", "--no-build-isolation", "--no-deps", "chumpy==0.70"], args.dry_run)
        lock = ROOT / "requirements" / f"{target}.txt"
        requirements = lock if lock.exists() else lock.with_suffix(".in")
        run(pip + ["install", "--only-binary=:all:", "-r", requirements], args.dry_run)
        run(pip + ["check"], args.dry_run)
        run([python, "-m", "backend_worker.cli", "--check-env", "--profile", target], args.dry_run)
        if target == "quality":
            run([python, "-m", "backend_worker.cli", "--check-cuda"], args.dry_run)
        print(f"{target} Worker Python: {python}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Environment setup FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
