# InSAR LOS Conversion Desktop GUI

将 `Claremont_final_software_development.ipynb` 重构为可双击运行的桌面 GUI（PySide6 + Matplotlib）。

## 1) 运行方式

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python main.py
```

## 2) 打包方式

### Windows `.exe`

```bash
pyinstaller --noconfirm --clean --windowed --name InSARStrainGUI --specpath build build/pyinstaller.spec
```

产物在 `dist/InSARStrainGUI/`（或单文件模式可改为 `--onefile`，但 GIS 依赖通常目录模式更稳）。

### macOS `.app`

```bash
pyinstaller --noconfirm --clean --windowed build/pyinstaller.spec
```

产物为 `dist/InSARStrainGUI.app`。

> 注：`geopandas/fiona/pyproj/shapely` 对底层动态库敏感。建议在目标平台原生环境中打包，不要跨平台打包。

## 3) 输入数据格式约定

- **InSAR shapefile**：必须含 `geometry(Point)` 和日期列，日期列名规则为 `DYYYYMMDD`（例如 `D20240131`）。
- **坐标系**：建议采用投影坐标（米）并与 fault/pipe 保持一致。
- **位移单位**：按 notebook 逻辑默认使用 `mm`；应变计算时会转换到 `m` 进行梯度计算。
- **Fault 数据**：
  - 可选 `fault line shapefile`（LineString）；
  - 可选 `bound_rep.npy`（N x 2 fault polyline），若启用 `auto_use_ml_fault_polyline=True` 且存在则优先使用。
- **Pipe 数据**：
  - 支持 `npy/csv/txt/shp`；
  - 要求可提取 `x,y` 坐标序列。

## 4) 输出目录结构

每次导出在 `output_dir/run_YYYYMMDD_HHMMSS/` 写入：

- `config.json`
- 关键数组：`NEZ.npy`、`matched_coords.npy`、`signed_distance.npy`、`pipe_motion_xy_mm.npy`、`axial_strain.npy`、`bending_strain.npy`、`day_diffs.npy`
- 表格：`matched_dates.csv`、`pipe_axial_latest.csv`、`pipe_bending_latest.csv`
- 图：`map.png`、`fit.png`、`axial.png`、`bending.png`

## 5) 目录

```text
main.py
src/
  config.py
  core/
    insar_3d.py
    fault_model.py
    pipeline_strain.py
  io/
    readers.py
    writers.py
  ui/
    main_window.py
build/
  pyinstaller.spec
requirements.txt
README.md
```
