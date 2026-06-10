#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
ENTRY_SCRIPT = REPO_ROOT / "core_py" / "psai" / "api_server.py"
DIST_DIR = REPO_ROOT / "build" / "pyinstaller-dist"
WORK_DIR = REPO_ROOT / "build" / "pyinstaller-work"
OUTPUT_DIR = REPO_ROOT / "src-tauri" / "binaries"


def detect_target_triple() -> str:
    try:
        proc = subprocess.run(
            ["rustc", "-Vv"],
            check=True,
            capture_output=True,
            text=True,
        )
        for line in proc.stdout.splitlines():
            if line.startswith("host:"):
                return line.split("host:", 1)[1].strip()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not detect Rust target triple: {exc}") from exc

    raise RuntimeError("Could not detect Rust target triple")


def build_pyinstaller_onefile() -> Path:
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--name",
        "core_py",
        "--paths",
        str(REPO_ROOT / "core_py"),
        "--collect-binaries",
        "z3",
        "--distpath",
        str(DIST_DIR),
        "--workpath",
        str(WORK_DIR),
        str(ENTRY_SCRIPT),
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    exe_name = "core_py.exe" if os.name == "nt" else "core_py"
    built = DIST_DIR / exe_name
    if not built.exists():
        raise FileNotFoundError(f"Expected PyInstaller output not found: {built}")

    return built


def install_sidecar(built_binary: Path, target_triple: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    suffix = ".exe" if os.name == "nt" else ""
    final_name = f"core_py-{target_triple}{suffix}"
    final_path = OUTPUT_DIR / final_name

    for existing in OUTPUT_DIR.glob(f"core_py-{target_triple}*"):
        if existing.is_file():
            existing.unlink()

    shutil.copy2(built_binary, final_path)

    if os.name != "nt":
        final_path.chmod(0o755)

    return final_path


def main() -> None:
    target_triple = detect_target_triple()
    built_binary = build_pyinstaller_onefile()
    final_path = install_sidecar(built_binary, target_triple)
    print(f"Sidecar installed: {final_path}")


if __name__ == "__main__":
    main()
