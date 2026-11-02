# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

hiddenimports = []
hiddenimports += collect_submodules("geopandas")
hiddenimports += collect_submodules("fiona")
hiddenimports += collect_submodules("pyproj")
hiddenimports += collect_submodules("shapely")
hiddenimports += collect_submodules("matplotlib.backends")

datas = []
datas += collect_data_files("matplotlib")
datas += collect_data_files("geopandas")
datas += collect_data_files("fiona")
datas += collect_data_files("pyproj")


a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="InSARStrainGUI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

app = BUNDLE(
    exe,
    name="InSARStrainGUI.app",
    icon=None,
    bundle_identifier="org.insar.strain.gui",
)
