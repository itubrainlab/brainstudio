"""Build a standalone BrainStudio app with PyInstaller for the system it runs on.

    uv sync --group build
    uv run --group build python packaging/build.py

Writes ``dist/BrainStudio-<version>-<system>-<arch>.<ext>``:

* macOS: ``BrainStudio.app`` in a ``.dmg``
* Windows: the ``BrainStudio`` folder (``BrainStudio.exe``) in a ``.zip``
* Linux: the ``BrainStudio`` folder in a ``.tar.gz``

and prints the path of the app executable, which accepts ``--self-test``.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "pyinstaller"
DIST = ROOT / "dist"
NAME = "BrainStudio"


def _system() -> tuple[str, str]:
    system = {"Darwin": "macos", "Windows": "windows", "Linux": "linux"}[platform.system()]
    arch = platform.machine().lower()
    arch = {"amd64": "x86_64", "x64": "x86_64", "aarch64": "arm64"}.get(arch, arch)
    return system, arch


def _icon_png(path: Path, size: int = 512) -> Path:
    """Render the SVG app icon to PNG (PyInstaller turns it into .icns / .ico)."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtGui, QtSvg
    app = QtGui.QGuiApplication.instance() or QtGui.QGuiApplication([])  # noqa: F841
    svg = ROOT / "src" / "brainstudio" / "resources" / "brainstudio.svg"
    img = QtGui.QImage(size, size, QtGui.QImage.Format_ARGB32)
    img.fill(0)
    painter = QtGui.QPainter(img)
    QtSvg.QSvgRenderer(str(svg)).render(painter)
    painter.end()
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path))
    return path


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from brainstudio import __version__
    system, arch = _system()
    shutil.rmtree(BUILD, ignore_errors=True)
    out_dir = BUILD / "dist"
    icon = _icon_png(BUILD / "icon.png")
    cmd = [
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--windowed",
        "--name", NAME, "--icon", str(icon),
        "--distpath", str(out_dir), "--workpath", str(BUILD / "work"),
        "--specpath", str(BUILD),
        # MNE loads its submodules lazily from .pyi stubs and ships data files
        "--collect-all", "mne", "--collect-all", "mne_bids",
        "--collect-all", "pyqtgraph", "--collect-all", "pygaze",
        "--collect-all", "brainstudio", "--collect-all", "picard",
        "--hidden-import", "pyxdf", "--hidden-import", "pybv",
        "--exclude-module", "PyQt5", "--exclude-module", "PyQt6",
        "--exclude-module", "tkinter", "--exclude-module", "pytest",
        "--exclude-module", "IPython",
        str(ROOT / "packaging" / "entry.py"),
    ]
    if system == "macos":
        cmd[5:5] = ["--osx-bundle-identifier", "dk.itu.brainlab.brainstudio"]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)

    DIST.mkdir(exist_ok=True)
    stem = f"{NAME}-{__version__}-{system}-{arch}"
    if system == "macos":
        app = out_dir / f"{NAME}.app"
        exe = app / "Contents" / "MacOS" / NAME
        target = DIST / f"{stem}.dmg"
        target.unlink(missing_ok=True)
        subprocess.run(["hdiutil", "create", "-volname", NAME, "-srcfolder", str(app),
                        "-ov", "-format", "UDZO", str(target)], check=True)
    elif system == "windows":
        folder = out_dir / NAME
        exe = folder / f"{NAME}.exe"
        target = DIST / f"{stem}.zip"
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
            for f in folder.rglob("*"):
                z.write(f, Path(NAME) / f.relative_to(folder))
    else:
        folder = out_dir / NAME
        exe = folder / NAME
        target = DIST / f"{stem}.tar.gz"
        with tarfile.open(target, "w:gz") as t:
            t.add(folder, arcname=NAME)
    print(f"package: {target}")
    print(f"executable: {exe}")
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:  # for the release workflow
        with open(gh_out, "a") as f:
            f.write(f"package={target}\nexecutable={exe}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
