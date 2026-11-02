from __future__ import annotations

import os
import pickle
import traceback
import copy
import json
from datetime import datetime
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Optional, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import contextily as cx
import geopandas as gpd
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point
from shapely.ops import linemerge, unary_union
from shapely.strtree import STRtree
from PySide6.QtCore import QObject, QThread, Qt, Signal, QTimer
from PySide6.QtGui import QAction, QFont, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QSizePolicy,
    QDialog,
    QDialogButtonBox,
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
)

from src.config import AppConfig
from src.core.fault_model import sigmoid
from src.core.fiber_calibration import run_fiber_b_calibration
from src.core.pipeline_strain import (
    AGGREGATION_METRIC_ORDER,
    _run_dual_aggregation,
    compute_unit_tangent_and_normal,
    default_aggregation_metric_weights,
    predict_pipe_motion_and_strain,
    run_pipeline,
    run_step_displacement,
    run_step_fit,
    run_step_match_3d,
    run_step_sigmoid_b_enhancement,
    run_step_strain,
)
from src.io.writers import export_pipeline_results

matplotlib.use("QtAgg")


class PipelineWorker(QObject):
    progress = Signal(int, str)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, config: AppConfig, task_name: str, step_fn, state: Optional[Dict] = None):
        super().__init__()
        self.config = config
        self.task_name = task_name
        self.step_fn = step_fn
        self.state = state or {}
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            def _progress(pct, msg):
                if self._stop:
                    raise InterruptedError("Computation stopped by user.")
                self.progress.emit(pct, msg)

            result = self.step_fn(self.config, self.state, progress=_progress)
            self.finished.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())


class MplTab(QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(5, 4), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = self.figure.add_subplot(111)
        self.title = title
        self._popup_windows = []
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.btn_popup = QPushButton("Open in New Window")
        self.btn_popup.clicked.connect(self.open_popup)

        self.lay = QVBoxLayout(self)
        self.lay.addWidget(self.toolbar)
        self.lay.addWidget(self.canvas)
        self.lay.addWidget(self.btn_popup)

    def open_popup(self):
        w = QMainWindow(self)
        w.setWindowTitle(self.title)
        try:
            fig = pickle.loads(pickle.dumps(self.figure))
        except Exception:
            fig = Figure(figsize=(7, 5), tight_layout=True)
            ax = fig.add_subplot(111)
            for line in self.ax.lines:
                ax.plot(line.get_xdata(), line.get_ydata(), label=line.get_label())
            ax.set_title(self.ax.get_title())
            ax.set_xlabel(self.ax.get_xlabel())
            ax.set_ylabel(self.ax.get_ylabel())
            if self.ax.get_legend():
                ax.legend()
        try:
            fig.tight_layout()
        except Exception:
            pass
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, w)
        cw = QWidget()
        cl = QVBoxLayout(cw)
        cl.addWidget(toolbar)
        cl.addWidget(canvas)
        w.setCentralWidget(cw)
        w.resize(900, 650)
        w.show()
        w._hold_ref = (canvas, toolbar)
        self._popup_windows.append(w)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("InSAR-Fault-Pipeline Strain GUI")
        self.resize(1600, 950)
        self.config = AppConfig()
        self.last_result: Optional[Dict] = None
        self.pipeline_state: Optional[Dict] = None
        self.current_task: str = ""
        self.worker = None
        self.thread = None

        self.inputs: Dict[str, QLineEdit] = {}
        self.params: Dict[str, QWidget] = {}
        self.fit_param_labels: Dict[str, QLineEdit] = {}
        self.map_colorbar = None
        self._last_fit_log_key: Optional[str] = None
        self._last_strain_peak_log_key: Optional[str] = None
        self._last_mesh_log_key: Optional[str] = None
        self._warned_log_keys = set()
        self._mesh_cache: Dict[str, object] = {}
        self._fit_state_guard: Optional[Dict[str, Any]] = None
        self._step1_selected_fault_idxs = []
        self._step1_selected_pipeline_idxs = []
        self._basemap_watch: Dict[str, Dict[str, object]] = {}
        self._basemap_timers: Dict[str, QTimer] = {}
        self._basemap_suppress_keys = set()
        self._ssi_popup_values: Dict[str, Any] = {}
        self._b_enhancement_popup_values: Dict[str, Any] = {
            "offset_min_m": -300.0,
            "offset_max_m": 400.0,
            "offset_step_m": 10.0,
            "window_half_length_m": 600.0,
            "window_half_width_m": 120.0,
            "x_half_for_c_window": 100.0,
            "min_points_per_window": 30,
            "n_bins": 20,
            "peak_enabled": True,
            "peak_prominence": 0.5,
            "peak_distance": 20,
            "peak_height": 1.3,
            "peak_zscore_threshold": 6.0,
            "remove_outliers_for_aggregation": True,
            "top_percent_for_aggregation": 10.0,
        }
        self._abaqus_inp_popup_values: Dict[str, Any] = {
            "model_type": "PIPE21+PSI",
            "curve_source": "From SSI parameters",
            "temperature_load": 0.0,
            "pipe_internal_pressure": 0.0,
            "unit_system": "kN-m",
            "material_name": "GradeBSteel",
            "poisson_ratio": 0.3,
            "analysis_step_name": "Step-1",
            "axial_curve_points": "",
            "lateral_center_curve_points": "",
            "lateral_outer_curve_points": "",
        }

        self._build_ui()
        self._fill_from_config(self.config)

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        v = QVBoxLayout(root)

        self.btn_run_step1 = QPushButton("Step 1: Solve ground motion")
        self.btn_run_step2 = QPushButton("Step 2: Fit shape function")
        self.btn_run_step3 = QPushButton("Step 3: Estimate pipeline displacement")
        self.btn_run_step4 = QPushButton("Step 4: Estimate pipeline strain")
        self.btn_stop = QPushButton("Stop")
        self.btn_save_state = QPushButton("Save State")
        self.btn_load_state = QPushButton("Load State")
        self.btn_export = QPushButton("Export All Results")
        self.btn_export_opensra = QPushButton("Export Results for OpenSRA")
        self.btn_open = QPushButton("Open Output Folder")
        self.btn_stop.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.btn_export_opensra.setEnabled(False)
        self.btn_open.setEnabled(False)
        self.btn_run_step1.clicked.connect(self.on_run_step1)
        self.btn_run_step2.clicked.connect(self.on_run_step2)
        self.btn_run_step3.clicked.connect(self.on_run_step3_displacement)
        self.btn_run_step4.clicked.connect(self.on_run_step4_strain)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_save_state.clicked.connect(self.on_save_state)
        self.btn_load_state.clicked.connect(self.on_load_state)
        self.btn_export.clicked.connect(self.on_export)
        self.btn_export_opensra.clicked.connect(self.on_export_opensra)
        self.btn_open.clicked.connect(self.on_open_output)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        v.addWidget(self.progress)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        v.addWidget(splitter, stretch=1)

        # Left: file inputs
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.addWidget(self._build_file_group())
        left_lay.addStretch(1)
        run_box = QGroupBox("Run Analysis")
        run_lay = QVBoxLayout(run_box)
        run_lay.setContentsMargins(0, 0, 0, 0)
        run_lay.setSpacing(6)
        run_lay.addWidget(self.btn_run_step1)
        run_lay.addWidget(self.btn_run_step2)
        run_lay.addWidget(self.btn_run_step3)
        run_lay.addWidget(self.btn_run_step4)
        run_lay.addWidget(self.btn_save_state)
        run_lay.addWidget(self.btn_load_state)
        run_lay.addWidget(self.btn_stop)
        run_lay.addWidget(self.btn_export)
        run_lay.addWidget(self.btn_export_opensra)
        run_lay.addWidget(self.btn_open)
        self._style_module_title(run_box)
        left_lay.addWidget(run_box)
        left_lay.addStretch(1)
        left_lay.addWidget(self._build_logo_panel())
        left.setMinimumWidth(260)
        splitter.addWidget(left)

        # Middle: parameters
        middle_scroll = QScrollArea()
        middle_scroll.setWidgetResizable(True)
        middle_holder = QWidget()
        middle_lay = QVBoxLayout(middle_holder)
        middle_lay.addWidget(self._build_param_groups())
        middle_lay.addStretch(1)
        middle_scroll.setWidget(middle_holder)
        middle_scroll.setMinimumWidth(340)
        splitter.addWidget(middle_scroll)

        # Right: plots + log
        right = QWidget()
        right_lay = QVBoxLayout(right)
        viz_panel = QGroupBox("Visualization")
        viz_panel_lay = QVBoxLayout(viz_panel)
        viz_panel_lay.setContentsMargins(6, 6, 6, 4)
        viz_panel_lay.setSpacing(2)
        viz_ctrl = QHBoxLayout()
        self.lbl_date = QLabel("Date")
        viz_ctrl.addWidget(self.lbl_date)
        self.cmb_step1_epoch = QComboBox()
        self.cmb_step1_epoch.addItem("Run Step 1 to load epochs", -1)
        self.cmb_step1_epoch.setEnabled(False)
        self.cmb_step1_epoch.currentIndexChanged.connect(self.on_step1_map_controls_changed)
        viz_ctrl.addWidget(self.cmb_step1_epoch)
        self.lbl_fiber_csv = QLabel("Fiber file")
        viz_ctrl.addWidget(self.lbl_fiber_csv)
        self.le_fiber_csv = QLineEdit()
        self.le_fiber_csv.setText(str(getattr(self.config, "mma_npy_path", "mma.npy")))
        self.le_fiber_csv.setMinimumWidth(220)
        self.le_fiber_csv.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        viz_ctrl.addWidget(self.le_fiber_csv, 1)
        self.btn_fiber_csv = QPushButton("...")
        self.btn_fiber_csv.setFixedWidth(28)
        self.btn_fiber_csv.clicked.connect(self.on_browse_fiber_csv)
        viz_ctrl.addWidget(self.btn_fiber_csv)
        self.lbl_fiber_origin = QLabel("origin (m)")
        viz_ctrl.addWidget(self.lbl_fiber_origin)
        self.le_fiber_origin = QLineEdit("-119")
        self.le_fiber_origin.setFixedWidth(56)
        viz_ctrl.addWidget(self.le_fiber_origin)
        self.chk_fiber_auto_origin = QCheckBox("auto origin")
        self.chk_fiber_auto_origin.setChecked(True)
        self.chk_fiber_auto_origin.toggled.connect(self._on_fiber_auto_origin_toggled)
        viz_ctrl.addWidget(self.chk_fiber_auto_origin)
        self.le_fiber_origin.setReadOnly(True)
        self.chk_fiber_reverse = QCheckBox("reverse")
        self.chk_fiber_reverse.setChecked(True)
        viz_ctrl.addWidget(self.chk_fiber_reverse)
        self.lbl_fiber_spacing = QLabel("spacing (m)")
        viz_ctrl.addWidget(self.lbl_fiber_spacing)
        self.le_fiber_spacing = QLineEdit("0.1")
        self.le_fiber_spacing.setFixedWidth(48)
        viz_ctrl.addWidget(self.le_fiber_spacing)
        self.lbl_fit_data = QLabel("Fit data")
        viz_ctrl.addWidget(self.lbl_fit_data)
        self.cmb_fit_data = QComboBox()
        self.cmb_fit_data.addItems(["Fault-parallel/perpendicular", "N&E"])
        self.cmb_fit_data.setCurrentText("Fault-parallel/perpendicular")
        self.cmb_fit_data.currentIndexChanged.connect(self.on_step1_map_controls_changed)
        self.cmb_fit_data.currentIndexChanged.connect(self._update_fit_override_control_visibility)
        viz_ctrl.addWidget(self.cmb_fit_data)
        self.lbl_disp_coord = QLabel("Coordinate system")
        viz_ctrl.addWidget(self.lbl_disp_coord)
        self.cmb_disp_coord = QComboBox()
        self.cmb_disp_coord.addItems(["Fault-parallel/perpendicular", "N&E"])
        self.cmb_disp_coord.setCurrentText("Fault-parallel/perpendicular")
        self.cmb_disp_coord.currentIndexChanged.connect(self.on_step1_map_controls_changed)
        self.cmb_disp_coord.currentIndexChanged.connect(self._refresh_disp_field_direction_options)
        viz_ctrl.addWidget(self.cmb_disp_coord)
        self.btn_export_abaqus_inp = QPushButton("Export Abaqus Input File")
        self.btn_export_abaqus_inp.clicked.connect(self.on_export_abaqus_inp)
        viz_ctrl.addWidget(self.btn_export_abaqus_inp)
        self.lbl_disp_field_dir = QLabel("Displacement direction")
        viz_ctrl.addWidget(self.lbl_disp_field_dir)
        self.cmb_disp_field_dir = QComboBox()
        self.cmb_disp_field_dir.currentIndexChanged.connect(self.on_step1_map_controls_changed)
        viz_ctrl.addWidget(self.cmb_disp_field_dir)
        self.lbl_disp_field_target = QLabel("Displacement field target")
        viz_ctrl.addWidget(self.lbl_disp_field_target)
        self.cmb_disp_field_target = QComboBox()
        self.cmb_disp_field_target.addItems(["Node", "Element"])
        self.cmb_disp_field_target.setCurrentText("Node")
        self.cmb_disp_field_target.currentIndexChanged.connect(self.on_step1_map_controls_changed)
        viz_ctrl.addWidget(self.cmb_disp_field_target)
        self.lbl_fit_dir = QLabel("Overide fit dir")
        viz_ctrl.addWidget(self.lbl_fit_dir)
        self.cmb_fit_dir = QComboBox()
        self.cmb_fit_dir.addItems(["Fault-parallel", "Fault-perpendicular"])
        self.cmb_fit_dir.currentIndexChanged.connect(self._sync_fit_override_editor_from_state)
        viz_ctrl.addWidget(self.cmb_fit_dir)
        self.btn_export_fit_scatter = QPushButton("Export fit scatter")
        self.btn_export_fit_scatter.clicked.connect(self.on_export_fault_parallel_scatter)
        viz_ctrl.addWidget(self.btn_export_fit_scatter)
        viz_ctrl_2 = QHBoxLayout()
        self.lbl_fitting_parameters = QLabel("Fitting parameters: ")
        viz_ctrl_2.addWidget(self.lbl_fitting_parameters)
        self.lbl_fit_a = QLabel("a")
        viz_ctrl_2.addWidget(self.lbl_fit_a)
        self.le_fit_a = QLineEdit()
        self.le_fit_a.setFixedWidth(48)
        viz_ctrl_2.addWidget(self.le_fit_a)
        self.lbl_fit_b = QLabel("b")
        viz_ctrl_2.addWidget(self.lbl_fit_b)
        self.le_fit_b = QLineEdit()
        self.le_fit_b.setFixedWidth(48)
        viz_ctrl_2.addWidget(self.le_fit_b)
        self.lbl_fit_c = QLabel("c")
        viz_ctrl_2.addWidget(self.lbl_fit_c)
        self.le_fit_c = QLineEdit()
        self.le_fit_c.setFixedWidth(48)
        viz_ctrl_2.addWidget(self.le_fit_c)
        self.lbl_fit_d = QLabel("d")
        viz_ctrl_2.addWidget(self.lbl_fit_d)
        self.le_fit_d = QLineEdit()
        self.le_fit_d.setFixedWidth(48)
        viz_ctrl_2.addWidget(self.le_fit_d)
        self.btn_fit_info = QPushButton("Info")
        self.btn_fit_info.setToolTip("Show sigmoid example")
        self.btn_fit_info.clicked.connect(self.on_show_sigmoid_example_dialog)
        viz_ctrl_2.addWidget(self.btn_fit_info)
        self.btn_apply_fit_override = QPushButton("Apply fit override")
        self.btn_apply_fit_override.clicked.connect(self.on_apply_fit_override)
        viz_ctrl_2.addWidget(self.btn_apply_fit_override)
        self.btn_reset_fit_override = QPushButton("Reset fit override")
        self.btn_reset_fit_override.clicked.connect(self.on_reset_fit_override)
        viz_ctrl_2.addWidget(self.btn_reset_fit_override)
        self.btn_sigmoid_b_enhancement = QPushButton("Sigmoid b enhancement")
        self.btn_sigmoid_b_enhancement.clicked.connect(self.on_sigmoid_b_enhancement)
        viz_ctrl_2.addWidget(self.btn_sigmoid_b_enhancement)
        self.btn_para_time_hist = QPushButton("Para time hist")
        self.btn_para_time_hist.clicked.connect(self.on_para_time_hist)
        viz_ctrl_2.addWidget(self.btn_para_time_hist)
        self.lbl_strain_type = QLabel("Strain type")
        viz_ctrl.addWidget(self.lbl_strain_type)
        self.cmb_strain_type = QComboBox()
        self.cmb_strain_type.addItems(["Axial", "Bending"])
        self.cmb_strain_type.setCurrentText("Axial")
        self.cmb_strain_type.currentIndexChanged.connect(self.on_step1_map_controls_changed)
        viz_ctrl.addWidget(self.cmb_strain_type)
        self.lbl_strain_field_view = QLabel("Strain field view")
        viz_ctrl.addWidget(self.lbl_strain_field_view)
        self.cmb_strain_field_view = QComboBox()
        self.cmb_strain_field_view.addItems(["Signed", "Absolute"])
        self.cmb_strain_field_view.setCurrentText("Signed")
        self.cmb_strain_field_view.currentIndexChanged.connect(self.on_step1_map_controls_changed)
        viz_ctrl.addWidget(self.cmb_strain_field_view)
        self.lbl_strain_field_target = QLabel("Strain field target")
        viz_ctrl.addWidget(self.lbl_strain_field_target)
        self.cmb_strain_field_target = QComboBox()
        self.cmb_strain_field_target.addItems(["Node", "Element"])
        self.cmb_strain_field_target.setCurrentText("Node")
        self.cmb_strain_field_target.currentIndexChanged.connect(self.on_step1_map_controls_changed)
        viz_ctrl.addWidget(self.cmb_strain_field_target)
        self.lbl_nez = QLabel("NEZ")
        viz_ctrl.addWidget(self.lbl_nez)
        self.cmb_step1_nez = QComboBox()
        self.cmb_step1_nez.addItems(["N", "E", "Z"])
        self.cmb_step1_nez.setCurrentText("E")
        self.cmb_step1_nez.currentIndexChanged.connect(self.on_step1_map_controls_changed)
        viz_ctrl.addWidget(self.cmb_step1_nez)
        self.lbl_vmin = QLabel("vmin")
        viz_ctrl.addWidget(self.lbl_vmin)
        self.le_step1_vmin = QLineEdit()
        self.le_step1_vmin.setText("-20")
        self.le_step1_vmin.setPlaceholderText("auto")
        self.le_step1_vmin.setFixedWidth(80)
        self.le_step1_vmin.editingFinished.connect(self.on_step1_map_controls_changed)
        viz_ctrl.addWidget(self.le_step1_vmin)
        self.lbl_vmax = QLabel("vmax")
        viz_ctrl.addWidget(self.lbl_vmax)
        self.le_step1_vmax = QLineEdit()
        self.le_step1_vmax.setText("25")
        self.le_step1_vmax.setPlaceholderText("auto")
        self.le_step1_vmax.setFixedWidth(80)
        self.le_step1_vmax.editingFinished.connect(self.on_step1_map_controls_changed)
        viz_ctrl.addWidget(self.le_step1_vmax)
        viz_ctrl.addStretch(1)

        self.lbl_map_xlim = QLabel("map xlim (m)")
        viz_ctrl_2.addWidget(self.lbl_map_xlim)
        self.le_map_xlim_min = QLineEdit()
        self.le_map_xlim_min.setPlaceholderText("auto")
        self.le_map_xlim_min.setFixedWidth(95)
        self.le_map_xlim_min.editingFinished.connect(self.on_step1_map_controls_changed)
        viz_ctrl_2.addWidget(self.le_map_xlim_min)
        self.le_map_xlim_max = QLineEdit()
        self.le_map_xlim_max.setPlaceholderText("auto")
        self.le_map_xlim_max.setFixedWidth(95)
        self.le_map_xlim_max.editingFinished.connect(self.on_step1_map_controls_changed)
        viz_ctrl_2.addWidget(self.le_map_xlim_max)
        self.lbl_map_ylim = QLabel("map ylim (m)")
        viz_ctrl_2.addWidget(self.lbl_map_ylim)
        self.le_map_ylim_min = QLineEdit()
        self.le_map_ylim_min.setPlaceholderText("auto")
        self.le_map_ylim_min.setFixedWidth(95)
        self.le_map_ylim_min.editingFinished.connect(self.on_step1_map_controls_changed)
        viz_ctrl_2.addWidget(self.le_map_ylim_min)
        self.le_map_ylim_max = QLineEdit()
        self.le_map_ylim_max.setPlaceholderText("auto")
        self.le_map_ylim_max.setFixedWidth(95)
        self.le_map_ylim_max.editingFinished.connect(self.on_step1_map_controls_changed)
        viz_ctrl_2.addWidget(self.le_map_ylim_max)
        viz_ctrl_2.addStretch(1)
        viz_ctrl_3 = QHBoxLayout()
        self.chk_fiber_is_microstrain = QCheckBox("fiber με/year")
        self.chk_fiber_is_microstrain.setChecked(True)
        viz_ctrl_3.addWidget(self.chk_fiber_is_microstrain)
        self.lbl_fiber_b_min = QLabel("b min")
        viz_ctrl_3.addWidget(self.lbl_fiber_b_min)
        self.le_fiber_b_min = QLineEdit("0.05")
        self.le_fiber_b_min.setFixedWidth(48)
        viz_ctrl_3.addWidget(self.le_fiber_b_min)
        self.lbl_fiber_b_max = QLabel("b max")
        viz_ctrl_3.addWidget(self.lbl_fiber_b_max)
        self.le_fiber_b_max = QLineEdit("2.0")
        self.le_fiber_b_max.setFixedWidth(48)
        viz_ctrl_3.addWidget(self.le_fiber_b_max)
        self.lbl_fiber_b_step = QLabel("step")
        viz_ctrl_3.addWidget(self.lbl_fiber_b_step)
        self.le_fiber_b_step = QLineEdit("0.1")
        self.le_fiber_b_step.setFixedWidth(48)
        viz_ctrl_3.addWidget(self.le_fiber_b_step)
        self.btn_run_fiber_calib = QPushButton("Run fiber calibration")
        self.btn_run_fiber_calib.clicked.connect(self.on_run_fiber_calibration)
        viz_ctrl_3.addWidget(self.btn_run_fiber_calib)
        self.btn_apply_fiber_optimal_b = QPushButton("Apply optimal b")
        self.btn_apply_fiber_optimal_b.setEnabled(False)
        self.btn_apply_fiber_optimal_b.clicked.connect(self.on_apply_fiber_optimal_b)
        viz_ctrl_3.addWidget(self.btn_apply_fiber_optimal_b)
        self.lbl_fiber_b_optimal = QLabel("b_optimal")
        viz_ctrl_3.addWidget(self.lbl_fiber_b_optimal)
        self.le_fiber_b_optimal = QLineEdit()
        self.le_fiber_b_optimal.setReadOnly(True)
        self.le_fiber_b_optimal.setFixedWidth(64)
        viz_ctrl_3.addWidget(self.le_fiber_b_optimal)
        self.lbl_fiber_rmse = QLabel("RMSE")
        viz_ctrl_3.addWidget(self.lbl_fiber_rmse)
        self.le_fiber_rmse = QLineEdit()
        self.le_fiber_rmse.setReadOnly(True)
        self.le_fiber_rmse.setFixedWidth(72)
        viz_ctrl_3.addWidget(self.le_fiber_rmse)
        viz_ctrl_3.addStretch(1)
        self.tabs = QTabWidget()
        self.tab_map = MplTab("Map / Points / Fault / Pipe")
        self.tab_fit = MplTab("Fit Functions")
        self.tab_pipe_mesh = MplTab("Pipeline Mesh")
        self.tab_pipe_disp = MplTab("Pipeline Displacement")
        self.tab_pipe_disp_field = MplTab("Pipeline Displacement Field")
        self.tab_pipe_strain = MplTab("Pipeline Strain")
        self.tab_pipe_strain_field = MplTab("Pipeline Strain Field")
        self.tab_fiber_calib = MplTab("Fiber Calibration")
        self.tabs.addTab(self.tab_map, "Map")
        self.tabs.addTab(self.tab_fit, "Fit")
        self.tabs.addTab(self.tab_pipe_mesh, "Pipeline mesh")
        self.tabs.addTab(self.tab_pipe_disp, "Pipeline displacement")
        self.tabs.addTab(self.tab_pipe_disp_field, "Pipeline displacement field")
        self.tabs.addTab(self.tab_pipe_strain, "Pipeline strain")
        self.tabs.addTab(self.tab_pipe_strain_field, "Pipeline strain field")
        # Public build: hide Fiber calibration tab (keep widget for restore).
        # self.tabs.addTab(self.tab_fiber_calib, "Fiber calibration")
        self.tabs.setUsesScrollButtons(True)
        self.tabs.tabBar().setUsesScrollButtons(True)
        self.tabs.tabBar().setExpanding(False)
        self.tabs.tabBar().setElideMode(Qt.ElideRight)
        self.tabs.setStyleSheet(
            "QTabWidget::pane { border: none; margin-top: 2px; top: 0px; } "
            "QTabWidget::tab-bar { alignment: center; } "
            "QTabBar::tab { "
            "min-width: 120px; min-height: 30px; font-size: 12px; "
            "padding: 0px 8px; margin: 0px; "
            "border: 1px solid #cfcfcf; border-right: none; "
            "background: #f2f2f2; color: #222; "
            "border-top-left-radius: 7px; border-bottom-left-radius: 7px; "
            "border-top-right-radius: 7px; border-bottom-right-radius: 7px; "
            "} "
            "QTabBar::tab:last { border-right: 1px solid #cfcfcf; } "
            "QTabBar::tab:selected { "
            "background: #0a84ff; color: white; border-color: #0a84ff; "
            "} "
            "QTabBar::tab:!selected:hover { background: #e9e9e9; }"
        )
        self.tabs.currentChanged.connect(self.on_plot_tab_changed)
        self._refresh_disp_field_direction_options()
        self._update_plot_control_visibility()
        self._update_fit_override_control_visibility()
        self.viz_controls_widget = QWidget()
        self.viz_controls_lay = QVBoxLayout(self.viz_controls_widget)
        self.viz_controls_lay.setContentsMargins(0, 0, 0, 0)
        self.viz_controls_lay.setSpacing(4)
        self.viz_controls_lay.addLayout(viz_ctrl)
        self.viz_controls_lay.addLayout(viz_ctrl_2)
        self.viz_controls_lay.addLayout(viz_ctrl_3)
        self._mount_viz_controls_into_current_tab()
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(2000)
        # Swap order: tabs first, then parameter rows.
        viz_panel_lay.addWidget(self.tabs, stretch=3)
        self._style_module_title(viz_panel)
        log_header = QHBoxLayout()
        log_title = QLabel("Log")
        self._style_module_title(log_title)
        log_header.addWidget(log_title)
        log_header.addStretch(1)
        self.btn_clear_log = QPushButton("Clear")
        self.btn_clear_log.clicked.connect(self.log_text.clear)
        log_header.addWidget(self.btn_clear_log)
        log_box = QWidget()
        log_lay = QVBoxLayout(log_box)
        log_lay.setContentsMargins(0, 0, 0, 0)
        log_lay.setSpacing(4)
        log_lay.addLayout(log_header)
        log_lay.addWidget(self.log_text)
        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.setChildrenCollapsible(False)
        right_splitter.setHandleWidth(4)
        right_splitter.addWidget(viz_panel)
        right_splitter.addWidget(log_box)
        right_splitter.setStretchFactor(0, 6)
        right_splitter.setStretchFactor(1, 1)
        right_splitter.setSizes([760, 180])
        self._lock_visualization_panel_width()
        right_lay.addWidget(right_splitter, stretch=1)
        right.setMinimumWidth(700)
        splitter.addWidget(right)

        splitter.setSizes([300, 430, 870])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 2)

        act_default = QAction("Reset All Defaults", self)
        act_default.triggered.connect(self.reset_defaults)
        self.menuBar().addAction(act_default)
        act_save_state = QAction("Save State", self)
        act_save_state.triggered.connect(self.on_save_state)
        self.menuBar().addAction(act_save_state)
        act_load_state = QAction("Load State", self)
        act_load_state.triggered.connect(self.on_load_state)
        self.menuBar().addAction(act_load_state)

    def _build_logo_panel(self):
        box = QGroupBox("")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        logo_files = [
            "image-4d3ad243-4b77-4f4d-a4ef-0c9612059c0a.png",
            "image-df997310-f5bd-489b-821a-30e464d48bda.png",
        ]

        for fn in logo_files:
            p = self._resolve_ui_asset_path(fn)
            lbl = QLabel()
            lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            pix = QPixmap(str(p)) if p is not None else QPixmap()
            if pix.isNull():
                lbl.setText(f"Missing logo: {fn}")
            else:
                lbl.setPixmap(pix.scaledToWidth(260, Qt.SmoothTransformation))
            lay.addWidget(lbl)
        return box

    def _ui_assets_dir(self) -> Path:
        return Path(__file__).resolve().parent / "assets"

    def _resolve_ui_asset_path(self, filename: str) -> Optional[Path]:
        candidates = [
            self._ui_assets_dir() / filename,
            Path.cwd() / filename,
        ]
        for p in candidates:
            try:
                if p.exists():
                    return p
            except Exception:
                continue
        return None

    def _build_file_group(self):
        box = QGroupBox("Input Files")
        form = QFormLayout(box)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form.setHorizontalSpacing(0)
        form.setVerticalSpacing(6)
        file_fields = [
            ("asc_shp_path", "Ascending InSAR shapefile"),
            ("desc_shp_path", "Descending InSAR shapefile"),
            ("fault_line_shp_path", "Fault line shapefile"),
            ("pipe_axis_path", "Pipe axis geometry"),
            ("output_dir", "Output directory"),
        ]
        for key, label in file_fields:
            item = QWidget()
            item_lay = QVBoxLayout(item)
            item_lay.setContentsMargins(0, 0, 0, 0)
            item_lay.setSpacing(3)
            item_lay.addWidget(QLabel(label))
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(8)
            le = QLineEdit()
            le.setMinimumWidth(300)
            btn = QPushButton("...")
            btn.setFixedWidth(52)
            btn.clicked.connect(lambda _, k=key: self.pick_path(k))
            h.addWidget(le, stretch=1)
            h.addWidget(btn)
            self.inputs[key] = le
            item_lay.addWidget(row)
            form.addRow(item)

        btn_reset = QPushButton("Reset Default File Paths")
        btn_reset.clicked.connect(self.reset_defaults)
        form.addRow(btn_reset)
        self._style_module_title(box)
        return box

    def _build_param_groups(self):
        holder = QWidget()
        lay = QVBoxLayout(holder)
        # Formal display labels (no underscores)
        label_map = {
            "date_match_max_days": "Date Match Max Days",
            "spatial_match_max_m": "Spatial Match Max (m)",
            "processing_crs": "Processing CRS",
            "pipeline_input_crs": "Pipeline Input CRS",
            "asc_heading_deg": "Asc Heading (°)",
            "asc_inc_deg": "Asc Incidence (°)",
            "desc_heading_deg": "Desc Heading (°)",
            "desc_inc_deg": "Desc Incidence (°)",
            "constraint_aspect_deg": "Constraint Aspect (°)",
            "constraint_aspect_source": "Constraint Aspect Source",
            "constraint_b16_floor": "b16 Floor",
            "flip_signed_distance": "Flip Signed Distance",
            "fit_band_offset_m": "Fault Perpendicular Offset Lines (m)",
            "transect_half_length_m": "Fault Perpendicular Half Length (m)",
            "bending_sample_step": "Bending Sample Step",
            "axial_smoothing_window": "Axial Smoothing Window",
            "mesh_size_m": "Mesh Size (m)",
            "fault_trace_type": "Fault trace type",
            "pipe_clip_half_length_m": "Pipe clip half-length (m)",
            "ssi_enabled": "SSI Enabled",
            "ssi_D_out_m": "Pipe Outsider Diameter (m)",
            "ssi_t_wall_m": "Pipe Wall Thickness (m)",
            "ssi_E_pipe_kN_m2": "Pipe Elastic Modulus (kN/m²)",
            "ssi_popup_button": "Soil Spring Parameters",
        }

        groups = [
            ("InSAR data matching (Step 1)", ["date_match_max_days", "spatial_match_max_m", "processing_crs", "pipeline_input_crs"]),
            (
                "LOS + Constraint (Step 1)",
                [
                    "asc_heading_deg",
                    "asc_inc_deg",
                    "desc_heading_deg",
                    "desc_inc_deg",
                    "constraint_aspect_source",
                    "constraint_aspect_deg",
                    "constraint_b16_floor",
                    "flip_signed_distance",
                ],
            ),
            (
                "Area Selection for Fitting (Step 2)",
                [
                    "transect_half_length_m",
                    "fit_band_offset_m",
                ],
            ),
            (
                "Fault Trace Processing (Step 2)",
                [
                    "fault_trace_type",
                    "pipe_clip_half_length_m",
                ],
            ),
            (
                "SSI (Step 3)",
                ["ssi_enabled", "ssi_D_out_m", "ssi_t_wall_m", "ssi_E_pipe_kN_m2", "ssi_popup_button"],
            ),
            ("Strain Calculation (Steps 3 && 4)", ["mesh_size_m", "axial_smoothing_window", "bending_sample_step"]),
        ]

        for title, keys in groups:
            gb = QGroupBox(title)
            form = QFormLayout(gb)
            form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
            form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
            form.setHorizontalSpacing(12)
            form.setVerticalSpacing(8)
            form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
            for k in keys:
                w = self._make_editor(k)
                if isinstance(w, (QLineEdit, QComboBox)):
                    w.setFixedWidth(184)
                if k == "ssi_enabled" and isinstance(w, QComboBox):
                    w.currentIndexChanged.connect(self._update_plot_control_visibility)
                self.params[k] = w
                form.addRow(label_map.get(k, k), w)
            self._style_module_title(gb)
            lay.addWidget(gb)

        self.btn_reset_default_params = QPushButton("Reset Default Parameters")
        self.btn_reset_default_params.clicked.connect(self.on_reset_visible_parameters_to_default)
        lay.addWidget(self.btn_reset_default_params)

        return holder

    def _make_editor(self, key: str):
        if key in {"auto_use_ml_fault_polyline", "flip_signed_distance", "ssi_enabled"}:
            cb = QComboBox()
            cb.addItems(["True", "False"])
            return cb
        if key == "processing_crs":
            cb = QComboBox()
            cb.addItems(["EPSG:32610", "EPSG:3857"])
            cb.currentTextChanged.connect(lambda _text: self._update_map_extent_labels())
            return cb
        if key == "pipeline_input_crs":
            cb = QComboBox()
            cb.addItems(["EPSG:32610", "EPSG:3857"])
            return cb
        if key == "fault_trace_type":
            cb = QComboBox()
            cb.addItems(["selected_geometry", "linear_fit", "quadratic_fit"])
            return cb
        if key == "constraint_aspect_source":
            cb = QComboBox()
            cb.addItems(["manual", "from_selected_fault_linear_fit"])
            return cb
        if key == "ssi_popup_button":
            btn = QPushButton("Soil Spring Parameters...")
            btn.clicked.connect(self.on_open_ssi_params_dialog)
            return btn
        return QLineEdit()

    def _fill_from_config(self, cfg: AppConfig):
        for f in fields(cfg):
            k = f.name
            v = getattr(cfg, k)
            if k in self.inputs:
                self.inputs[k].setText(str(v))
            if k in self.params:
                w = self.params[k]
                if isinstance(w, QComboBox):
                    w.setCurrentText(str(v))
                else:
                    w.setText(str(v))
        if hasattr(self, "le_map_xlim_min"):
            self.le_map_xlim_min.setText(str(cfg.map_xlim_min))
            self.le_map_xlim_max.setText(str(cfg.map_xlim_max))
            self.le_map_ylim_min.setText(str(cfg.map_ylim_min))
            self.le_map_ylim_max.setText(str(cfg.map_ylim_max))
        if hasattr(self, "cmb_disp_coord"):
            self.cmb_disp_coord.setCurrentText(getattr(cfg, "prediction_coordinate_system", "N&E"))
        self._ssi_popup_values = {
            "ssi_bc_type": str(getattr(cfg, "ssi_bc_type", "pinned-pinned")),
            "ssi_E_soil_outer_kN_m2": float(getattr(cfg, "ssi_E_soil_outer_kN_m2", 10000.0)),
            "ssi_E_soil_center_kN_m2": float(getattr(cfg, "ssi_E_soil_center_kN_m2", 2000.0)),
            "ssi_k2_ratio_outer": float(getattr(cfg, "ssi_k2_ratio_outer", 0.325)),
            "ssi_k2_ratio_center": float(getattr(cfg, "ssi_k2_ratio_center", 0.325)),
            "ssi_mu_friction": float(getattr(cfg, "ssi_mu_friction", 0.678)),
            "ssi_delta_y_m": float(getattr(cfg, "ssi_delta_y_m", 0.001)),
            "ssi_delta_max_m": float(getattr(cfg, "ssi_delta_max_m", 0.023)),
            "ssi_k_axial_kN_m": float(getattr(cfg, "ssi_k_axial_kN_m", 10000.0)),
            "ssi_depth_burial_m": float(getattr(cfg, "ssi_depth_burial_m", 1.5)),
            "ssi_gamma_soil_kN_m3": float(getattr(cfg, "ssi_gamma_soil_kN_m3", 18.0)),
            "ssi_k_gap_penalty_kN_m": float(getattr(cfg, "ssi_k_gap_penalty_kN_m", 5.0e5)),
            "ssi_n_increments": int(getattr(cfg, "ssi_n_increments", 20)),
            "ssi_max_iter": int(getattr(cfg, "ssi_max_iter", 50)),
            "ssi_tol": float(getattr(cfg, "ssi_tol", 1e-6)),
        }
        self._update_map_extent_labels()

    def _read_config_from_ui(self) -> AppConfig:
        data = {}
        default = AppConfig()
        for f in fields(default):
            k = f.name
            raw = None
            if k in self.inputs:
                raw = self.inputs[k].text().strip()
            elif k in self.params:
                w = self.params[k]
                raw = w.currentText().strip() if isinstance(w, QComboBox) else w.text().strip()
            else:
                continue

            default_v = getattr(default, k)
            if isinstance(default_v, bool):
                data[k] = raw.lower() == "true"
            elif isinstance(default_v, int) and not isinstance(default_v, bool):
                data[k] = int(raw)
            elif isinstance(default_v, float):
                data[k] = float(raw)
            else:
                data[k] = raw
        cfg = AppConfig.from_dict(data)
        for k, v in self._ssi_popup_values.items():
            setattr(cfg, k, v)
        cfg.prediction_coordinate_system = (
            self.cmb_disp_coord.currentText() if hasattr(self, "cmb_disp_coord") else "N&E"
        )
        return cfg

    def on_open_ssi_params_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Soil Spring Parameters")
        lay = QVBoxLayout(dlg)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        keys = [
            ("ssi_bc_type", "bc_type"),
            ("ssi_E_soil_outer_kN_m2", "E_soil_outer (kN/m²)"),
            ("ssi_E_soil_center_kN_m2", "E_soil_center (kN/m²)"),
            ("ssi_k2_ratio_outer", "k2_ratio_outer (delta_y~delta_max)"),
            ("ssi_k2_ratio_center", "k2_ratio_center (delta_y~delta_max)"),
            ("ssi_mu_friction", "mu_friction"),
            ("ssi_delta_y_m", "delta_y (m)"),
            ("ssi_delta_max_m", "delta_max (m)"),
            ("ssi_k_axial_kN_m", "k_axial (kN/m)"),
            ("ssi_depth_burial_m", "depth_burial (m)"),
            ("ssi_gamma_soil_kN_m3", "gamma_soil (kN/m³)"),
            ("ssi_k_gap_penalty_kN_m", "k_gap_penalty (kN/m)"),
            ("ssi_n_increments", "n_increments"),
            ("ssi_max_iter", "max_iter"),
            ("ssi_tol", "tol"),
        ]
        editors = {}
        bc = QComboBox()
        bc.addItems(["fixed-fixed", "pinned-pinned", "fixed-free", "pinned-free"])
        bc.setCurrentText(str(self._ssi_popup_values.get("ssi_bc_type", "pinned-pinned")))
        form.addRow("bc_type", bc)
        editors["ssi_bc_type"] = bc
        for key, label in keys[1:]:
            le = QLineEdit(str(self._ssi_popup_values.get(key, "")))
            form.addRow(label, le)
            editors[key] = le
        lay.addLayout(form)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dlg)
        lay.addWidget(btns)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        if dlg.exec() != QDialog.Accepted:
            return
        try:
            vals = dict(self._ssi_popup_values)
            vals["ssi_bc_type"] = editors["ssi_bc_type"].currentText().strip().lower()
            float_keys = [
                "ssi_E_soil_outer_kN_m2",
                "ssi_E_soil_center_kN_m2",
                "ssi_k2_ratio_outer",
                "ssi_k2_ratio_center",
                "ssi_mu_friction",
                "ssi_delta_y_m",
                "ssi_delta_max_m",
                "ssi_k_axial_kN_m",
                "ssi_depth_burial_m",
                "ssi_gamma_soil_kN_m3",
                "ssi_k_gap_penalty_kN_m",
                "ssi_tol",
            ]
            int_keys = ["ssi_n_increments", "ssi_max_iter"]
            for key in float_keys:
                vals[key] = float(editors[key].text().strip())
            for key in int_keys:
                vals[key] = int(editors[key].text().strip())
            self._ssi_popup_values = vals
            for k, v in vals.items():
                setattr(self.config, k, v)
            self.append_log("[SSI] Soil spring parameters updated.")
        except Exception:
            self.append_log(traceback.format_exc())
            QMessageBox.critical(self, "SSI Parameters Error", "Invalid SSI soil spring parameters.")

    def reset_defaults(self):
        self.config = AppConfig()
        self._fill_from_config(self.config)
        self.append_log("Parameters have been reset to notebook defaults.")

    def on_reset_visible_parameters_to_default(self):
        default_cfg = AppConfig()
        for f in fields(default_cfg):
            key = f.name
            if key not in self.params:
                continue
            w = self.params[key]
            val = getattr(default_cfg, key)
            if isinstance(w, QComboBox):
                if isinstance(val, bool):
                    w.setCurrentText("True" if val else "False")
                else:
                    w.setCurrentText(str(val))
            elif isinstance(w, QLineEdit):
                w.setText(str(val))

        self._ssi_popup_values = {
            "ssi_bc_type": str(getattr(default_cfg, "ssi_bc_type", "pinned-pinned")),
            "ssi_E_soil_outer_kN_m2": float(getattr(default_cfg, "ssi_E_soil_outer_kN_m2", 10000.0)),
            "ssi_E_soil_center_kN_m2": float(getattr(default_cfg, "ssi_E_soil_center_kN_m2", 2000.0)),
            "ssi_k2_ratio_outer": float(getattr(default_cfg, "ssi_k2_ratio_outer", 0.325)),
            "ssi_k2_ratio_center": float(getattr(default_cfg, "ssi_k2_ratio_center", 0.325)),
            "ssi_mu_friction": float(getattr(default_cfg, "ssi_mu_friction", 0.678)),
            "ssi_delta_y_m": float(getattr(default_cfg, "ssi_delta_y_m", 0.001)),
            "ssi_delta_max_m": float(getattr(default_cfg, "ssi_delta_max_m", 0.023)),
            "ssi_k_axial_kN_m": float(getattr(default_cfg, "ssi_k_axial_kN_m", 10000.0)),
            "ssi_depth_burial_m": float(getattr(default_cfg, "ssi_depth_burial_m", 1.5)),
            "ssi_gamma_soil_kN_m3": float(getattr(default_cfg, "ssi_gamma_soil_kN_m3", 18.0)),
            "ssi_k_gap_penalty_kN_m": float(getattr(default_cfg, "ssi_k_gap_penalty_kN_m", 5.0e5)),
            "ssi_n_increments": int(getattr(default_cfg, "ssi_n_increments", 20)),
            "ssi_max_iter": int(getattr(default_cfg, "ssi_max_iter", 50)),
            "ssi_tol": float(getattr(default_cfg, "ssi_tol", 1e-6)),
        }
        self.append_log("Middle parameter panel has been reset to default values.")

    def on_show_sigmoid_example_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Sigmoid fitting example")
        lay = QVBoxLayout(dlg)
        lbl = QLabel()
        lbl.setAlignment(Qt.AlignCenter)

        p = self._resolve_ui_asset_path("sigmoid_example.jpg")
        if p is None:
            lbl.setText("Missing image: sigmoid_example.jpg")
        else:
            pix = QPixmap(str(p))
            if pix.isNull():
                lbl.setText("Failed to load image: sigmoid_example.jpg")
            else:
                screen = self.screen()
                if screen is not None:
                    geo = screen.availableGeometry()
                    max_w = int(max(600, geo.width() * 0.9))
                    max_h = int(max(420, geo.height() * 0.85))
                else:
                    max_w = 1400
                    max_h = 900
                frame_w = 48
                frame_h = 120
                draw_w = max(320, max_w - frame_w)
                draw_h = max(220, max_h - frame_h)
                scaled = pix.scaled(draw_w, draw_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                lbl.setPixmap(scaled)
                dlg.resize(int(scaled.width() + frame_w), int(scaled.height() + frame_h))
        lay.addWidget(lbl)
        btns = QDialogButtonBox(QDialogButtonBox.Close, parent=dlg)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)
        dlg.exec()

    def pick_path(self, key: str):
        if key == "output_dir":
            p = QFileDialog.getExistingDirectory(self, "Select Output Directory", str(Path.cwd()))
            if p:
                self.inputs[key].setText(p)
            return
        p, _ = QFileDialog.getOpenFileName(self, "Select File", str(Path.cwd()), "All Files (*.*)")
        if p:
            self.inputs[key].setText(p)

    def append_log(self, msg: str):
        self.log_text.appendPlainText(msg)

    def _module_title_font(self, widget) -> QFont:
        font = QFont(widget.font())
        pt = int(font.pointSize())
        if pt <= 0:
            px = int(font.pixelSize())
            pt = max(int(round(px * 0.75)), 13) if px > 0 else 13
        font.setPointSize(pt + 1)
        font.setBold(True)
        return font

    def _style_module_title(self, widget):
        # macOS native QGroupBox ignores QGroupBox::title stylesheet; use a QLabel like Log.
        if isinstance(widget, QLabel):
            widget.setFont(self._module_title_font(widget))
            return
        if not isinstance(widget, QGroupBox):
            return
        title = widget.title()
        lay = widget.layout()
        if not title or lay is None:
            return
        widget.setTitle("")
        lbl = QLabel(title)
        lbl.setFont(self._module_title_font(lbl))
        if isinstance(lay, QFormLayout):
            lay.insertRow(0, lbl)
        else:
            lay.insertWidget(0, lbl)

    def _append_log_once(self, key: str, msg: str):
        if key in self._warned_log_keys:
            return
        self._warned_log_keys.add(key)
        self.append_log(msg)

    def _processing_crs_text(self) -> str:
        raw = str(getattr(self.config, "processing_crs", "EPSG:3857")).strip().upper()
        w = self.params.get("processing_crs") if hasattr(self, "params") else None
        if isinstance(w, QComboBox):
            txt = w.currentText().strip().upper()
            if txt:
                raw = txt
        if raw.startswith("EPSG:"):
            return raw
        return f"EPSG:{raw}" if raw else "EPSG:3857"

    def _processing_epsg(self) -> int:
        crs_txt = self._processing_crs_text()
        try:
            return int(crs_txt.split(":", 1)[1])
        except Exception:
            return 3857

    def _pipeline_input_crs_text(self) -> str:
        raw = str(getattr(self.config, "pipeline_input_crs", "EPSG:3857")).strip().upper()
        w = self.params.get("pipeline_input_crs") if hasattr(self, "params") else None
        if isinstance(w, QComboBox):
            txt = w.currentText().strip().upper()
            if txt:
                raw = txt
        if raw.startswith("EPSG:"):
            return raw
        return f"EPSG:{raw}" if raw else "EPSG:3857"

    def _update_map_extent_labels(self):
        crs_txt = self._processing_crs_text()
        if hasattr(self, "lbl_map_xlim"):
            self.lbl_map_xlim.setText(f"map xlim (m)({crs_txt})")
        if hasattr(self, "lbl_map_ylim"):
            self.lbl_map_ylim.setText(f"map ylim (m)({crs_txt})")

    def _prepare_dynamic_basemap(self, key: str, ax, alpha: float):
        info = self._basemap_watch.get(key)
        if info is not None and info.get("ax") is not ax:
            for cid in info.get("cids", []):
                try:
                    info["ax"].callbacks.disconnect(cid)
                except Exception:
                    pass
        if info is None or info.get("ax") is not ax:
            def _on_lim_change(_evt_ax):
                watch = self._basemap_watch.get(key)
                if not watch:
                    return
                if key in self._basemap_suppress_keys:
                    return
                if not bool(watch.get("armed", False)):
                    return
                timer = self._basemap_timers.get(key)
                if timer is None:
                    timer = QTimer(self)
                    timer.setSingleShot(True)
                    timer.timeout.connect(lambda k=key: self._reload_dynamic_basemap(k))
                    self._basemap_timers[key] = timer
                timer.start(300)

            cidx = ax.callbacks.connect("xlim_changed", _on_lim_change)
            cidy = ax.callbacks.connect("ylim_changed", _on_lim_change)
            info = {"ax": ax, "cids": [cidx, cidy], "armed": False, "alpha": float(alpha)}
            self._basemap_watch[key] = info
        else:
            info["alpha"] = float(alpha)
        self._basemap_suppress_keys.add(key)

    def _finalize_dynamic_basemap(self, key: str):
        info = self._basemap_watch.get(key)
        if info is not None:
            info["armed"] = True
        self._basemap_suppress_keys.discard(key)

    def _basemap_headers(self) -> Dict[str, str]:
        # Use a stable, app-identifiable User-Agent for OSM tile policy compliance.
        return {"User-Agent": "InSAR_LOS_Conversion_GUI/1.0 (desktop-app)"}

    def _add_basemap(self, ax, alpha: float = 0.35, key: Optional[str] = None):
        if key:
            self._prepare_dynamic_basemap(key, ax, alpha)
        try:
            cx.add_basemap(
                ax,
                crs=self._processing_crs_text(),
                source=cx.providers.OpenStreetMap.Mapnik,
                headers=self._basemap_headers(),
                alpha=float(alpha),
            )
        except Exception:
            pass

    def _reload_dynamic_basemap(self, key: str):
        info = self._basemap_watch.get(key)
        if not info:
            return
        ax = info.get("ax")
        if ax is None:
            return
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        self._basemap_suppress_keys.add(key)
        try:
            for im in list(ax.images):
                try:
                    im.remove()
                except Exception:
                    pass
            try:
                cx.add_basemap(
                    ax,
                    crs=self._processing_crs_text(),
                    source=cx.providers.OpenStreetMap.Mapnik,
                    headers=self._basemap_headers(),
                    alpha=float(info.get("alpha", 0.35)),
                )
            except Exception:
                pass
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.figure.canvas.draw_idle()
        finally:
            self._basemap_suppress_keys.discard(key)

    def _attach_picker_dynamic_basemap(self, fig, ax, alpha: float = 0.30, map_crs: Optional[str] = None):
        """
        Picker-only basemap reloader. It is intentionally isolated from the
        tab-level watcher cache to avoid cross-window callback leakage.
        """
        state = {"suppress": True}
        timer = fig.canvas.new_timer(interval=300)
        timer.single_shot = True

        def _reload():
            if not plt.fignum_exists(fig.number):
                return
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            state["suppress"] = True
            try:
                for im in list(ax.images):
                    try:
                        im.remove()
                    except Exception:
                        pass
                try:
                    cx.add_basemap(
                        ax,
                        crs=str(map_crs or self._processing_crs_text()),
                        source=cx.providers.OpenStreetMap.Mapnik,
                        headers=self._basemap_headers(),
                        alpha=float(alpha),
                    )
                except Exception:
                    pass
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                fig.canvas.draw_idle()
            finally:
                state["suppress"] = False

        def _on_lim_change(_evt_ax):
            if state["suppress"]:
                return
            timer.stop()
            timer.start()

        def _on_close_cleanup(_evt):
            try:
                timer.stop()
            except Exception:
                pass
            try:
                ax.callbacks.disconnect(cid_x)
            except Exception:
                pass
            try:
                ax.callbacks.disconnect(cid_y)
            except Exception:
                pass

        cid_x = ax.callbacks.connect("xlim_changed", _on_lim_change)
        cid_y = ax.callbacks.connect("ylim_changed", _on_lim_change)
        fig.canvas.mpl_connect("close_event", _on_close_cleanup)
        timer.add_callback(_reload)
        # Keep strong references tied to figure lifecycle.
        refs = getattr(fig, "_picker_basemap_refs", [])
        refs.append({"timer": timer, "cid_x": cid_x, "cid_y": cid_y})
        fig._picker_basemap_refs = refs

        # Initial basemap load.
        try:
            cx.add_basemap(
                ax,
                crs=str(map_crs or self._processing_crs_text()),
                source=cx.providers.OpenStreetMap.Mapnik,
                headers=self._basemap_headers(),
                alpha=float(alpha),
            )
        except Exception:
            pass
        state["suppress"] = False

    def _reproject_lines_for_picker(self, lines, src_crs: str, dst_crs: str = "EPSG:3857"):
        if lines is None:
            return []
        try:
            seq = list(lines)
        except Exception:
            return []
        if len(seq) == 0:
            return []
        src = str(src_crs).upper() if src_crs else ""
        dst = str(dst_crs).upper() if dst_crs else ""
        if not src or src == dst:
            return seq
        try:
            gs = gpd.GeoSeries(seq, crs=src)
            out = gs.to_crs(dst)
            projected = []
            vals = list(out.values)
            for i, geom in enumerate(vals):
                if geom is None or getattr(geom, "is_empty", False):
                    projected.append(seq[i] if i < len(seq) else geom)
                else:
                    projected.append(geom)
            return projected
        except Exception:
            # Keep original geometry if display-only reprojection fails.
            return seq

    def _flatten_fault_lines_for_picker(self, fault_gdf: gpd.GeoDataFrame):
        lines = []
        for geom in fault_gdf.geometry.values:
            if geom is None or geom.is_empty:
                continue
            if isinstance(geom, LineString):
                lines.append(geom)
            elif isinstance(geom, MultiLineString):
                for ln in geom.geoms:
                    if ln is not None and not ln.is_empty:
                        lines.append(ln)
        return lines

    def _load_pipeline_lines_for_picker(self, pipe_path: Path, target_crs=None):
        lines = []
        if not pipe_path.exists():
            return lines
        ext = pipe_path.suffix.lower()
        if ext == ".shp":
            gdf = gpd.read_file(pipe_path)
            if gdf.crs is None:
                # Some pipeline shapefiles are missing CRS metadata. In Step1 picker,
                # align with the same pipeline_input_crs policy used by readers.
                try:
                    gdf = gdf.set_crs(self._pipeline_input_crs_text(), allow_override=True)
                except Exception:
                    pass
            if target_crs is not None and gdf.crs is not None and str(gdf.crs).upper() != str(target_crs).upper():
                try:
                    gdf = gdf.to_crs(target_crs)
                except Exception:
                    pass
            for geom in gdf.geometry.values:
                if geom is None or geom.is_empty:
                    continue
                if isinstance(geom, LineString):
                    lines.append(geom)
                elif isinstance(geom, MultiLineString):
                    for ln in geom.geoms:
                        if ln is not None and not ln.is_empty:
                            lines.append(ln)
            return lines
        if ext == ".npy":
            arr = np.load(pipe_path)
        elif ext in {".csv", ".txt"}:
            arr = np.loadtxt(pipe_path, delimiter="," if ext == ".csv" else None)
        else:
            return lines
        axis = np.asarray(arr, dtype=float)
        if axis.ndim == 2 and axis.shape[0] > 1 and axis.shape[1] >= 2:
            ln = LineString(axis[:, :2])
            src_crs = self._pipeline_input_crs_text()
            dst_crs = str(target_crs).upper() if target_crs is not None else src_crs
            if src_crs != dst_crs:
                try:
                    gs = gpd.GeoSeries([ln], crs=src_crs)
                    ln = gs.to_crs(dst_crs).iloc[0]
                except Exception:
                    pass
            lines.append(ln)
        return lines

    def _pipeline_lines_from_input_data(self, data: Dict):
        lines = []
        for geom in data.get("pipe_lines_all", []) if isinstance(data, dict) else []:
            if geom is None or getattr(geom, "is_empty", True):
                continue
            if isinstance(geom, LineString):
                lines.append(geom)
            elif isinstance(geom, MultiLineString):
                for ln in geom.geoms:
                    if ln is not None and not ln.is_empty:
                        lines.append(ln)
        if lines:
            return lines
        pipe_axis = np.asarray(data.get("pipe_axis", []), dtype=float) if isinstance(data, dict) else np.array([])
        if pipe_axis.ndim == 2 and pipe_axis.shape[0] > 1 and pipe_axis.shape[1] >= 2:
            return [LineString(pipe_axis[:, :2])]
        return []

    def _interactive_pick_fault_segments(self, fault_lines, pipe_axis=None, pipe_lines=None, map_crs: Optional[str] = None):
        if fault_lines is None or len(fault_lines) == 0:
            return []
        # Always show all segments from the selected fault shapefile.
        # Previous bbox filtering could hide valid trace portions.
        plot_ids = list(range(len(fault_lines)))
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_aspect("equal")
        ax.set_title("Pick fault segments | click=toggle, q=done, r=reset")
        # Build combined bounds from all fault segments + pipeline.
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        for idx in plot_ids:
            ln = fault_lines[idx]
            if ln is None or ln.is_empty:
                continue
            bx0, by0, bx1, by1 = ln.bounds
            xmins.append(float(bx0))
            xmaxs.append(float(bx1))
            ymins.append(float(by0))
            ymaxs.append(float(by1))
        if pipe_lines is not None and len(pipe_lines) > 0:
            for ln in pipe_lines:
                if ln is None or ln.is_empty:
                    continue
                bx0, by0, bx1, by1 = ln.bounds
                xmins.append(float(bx0))
                xmaxs.append(float(bx1))
                ymins.append(float(by0))
                ymaxs.append(float(by1))
        elif pipe_axis is not None:
            pa = np.asarray(pipe_axis, dtype=float)
            if pa.ndim == 2 and pa.shape[0] > 1 and pa.shape[1] >= 2:
                xmins.append(float(np.nanmin(pa[:, 0])))
                xmaxs.append(float(np.nanmax(pa[:, 0])))
                ymins.append(float(np.nanmin(pa[:, 1])))
                ymaxs.append(float(np.nanmax(pa[:, 1])))
        if not xmins or not xmaxs or not ymins or not ymaxs:
            # Fallback bounds when geometry extents are unavailable.
            minx = float(getattr(self.config, "fault_picker_bbox_minx", -13622000.0))
            maxx = float(getattr(self.config, "fault_picker_bbox_maxx", -13617000.0))
            miny = float(getattr(self.config, "fault_picker_bbox_miny", 4573000.0))
            maxy = float(getattr(self.config, "fault_picker_bbox_maxy", 4581000.0))
            xmins = [minx]
            xmaxs = [maxx]
            ymins = [miny]
            ymaxs = [maxy]
        x0 = float(np.nanmin(xmins))
        x1 = float(np.nanmax(xmaxs))
        y0 = float(np.nanmin(ymins))
        y1 = float(np.nanmax(ymaxs))
        xpad = max(50.0, 0.08 * max(x1 - x0, 1.0))
        ypad = max(50.0, 0.08 * max(y1 - y0, 1.0))
        ax.set_xlim(x0 - xpad, x1 + xpad)
        ax.set_ylim(y0 - ypad, y1 + ypad)
        self._attach_picker_dynamic_basemap(fig, ax, alpha=0.30, map_crs=map_crs)
        if pipe_lines is not None and len(pipe_lines) > 0:
            shown_label = False
            for ln in pipe_lines:
                if ln is None or ln.is_empty:
                    continue
                x, y = ln.xy
                ax.plot(
                    x,
                    y,
                    color="C0",
                    linewidth=3.0,
                    alpha=0.98,
                    zorder=4,
                    label="Pipeline (not selectable)" if not shown_label else None,
                )
                shown_label = True
        elif pipe_axis is not None:
            pa = np.asarray(pipe_axis, dtype=float)
            if pa.ndim == 2 and pa.shape[0] > 1 and pa.shape[1] >= 2:
                ax.plot(
                    pa[:, 0],
                    pa[:, 1],
                    color="C0",
                    linewidth=3.0,
                    alpha=0.98,
                    zorder=4,
                    label="Pipeline (not selectable)",
                )
        shown_fault_selectable = False
        for idx in plot_ids:
            x, y = fault_lines[idx].xy
            ax.plot(
                x,
                y,
                color="0.75",
                linewidth=1.0,
                zorder=5,
                label="Fault trace (selectable)" if not shown_fault_selectable else None,
            )
            shown_fault_selectable = True
        selected = set()
        click_tol = float(getattr(self.config, "fault_picker_click_max_dist_m", 200.0))
        sel_lines = []
        done = {"closed": False}

        def _redraw():
            for ln in sel_lines:
                try:
                    ln.remove()
                except Exception:
                    pass
            sel_lines.clear()
            for idx in sorted(selected):
                x, y = fault_lines[idx].xy
                (ln,) = ax.plot(x, y, color="red", linewidth=2.5, zorder=6)
                sel_lines.append(ln)
            ax.set_title(
                f"Pick fault segments | selected={sorted(selected)} | click=toggle, q=done, r=reset"
            )
            fig.canvas.draw_idle()

        def _on_click(event):
            if event.inaxes != ax or event.button != 1 or event.xdata is None or event.ydata is None:
                return
            pt = Point(float(event.xdata), float(event.ydata))
            best_idx = None
            best_dist = np.inf
            for idx in plot_ids:
                d = float(fault_lines[idx].distance(pt))
                if d < best_dist:
                    best_dist = d
                    best_idx = idx
            if best_idx is None or best_dist > click_tol:
                return
            if best_idx in selected:
                selected.remove(best_idx)
            else:
                selected.add(best_idx)
            _redraw()

        def _on_key(event):
            if event.key == "r":
                selected.clear()
                _redraw()
            elif event.key == "q":
                done["closed"] = True
                plt.close(fig)

        def _on_close(_event):
            done["closed"] = True

        fig.canvas.mpl_connect("button_press_event", _on_click)
        fig.canvas.mpl_connect("key_press_event", _on_key)
        fig.canvas.mpl_connect("close_event", _on_close)
        ax.legend(loc="best")
        plt.show(block=False)
        # In Qt GUI context, show(block=True) may return immediately.
        # Keep polling until user actually closes the picker window.
        while not done["closed"] and plt.fignum_exists(fig.number):
            plt.pause(0.05)
        return sorted(selected)

    def _interactive_pick_pipeline_segments(self, pipeline_lines, fault_lines=None, map_crs: Optional[str] = None):
        if pipeline_lines is None or len(pipeline_lines) == 0:
            return []
        plot_ids = list(range(len(pipeline_lines)))
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_aspect("equal")
        ax.set_title("Pick pipeline segments | click=toggle, q=done, r=reset")
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        for ln in pipeline_lines:
            if ln is None or ln.is_empty:
                continue
            bx0, by0, bx1, by1 = ln.bounds
            xmins.append(float(bx0))
            xmaxs.append(float(bx1))
            ymins.append(float(by0))
            ymaxs.append(float(by1))
        if fault_lines is not None and len(fault_lines) > 0:
            fault_iter = fault_lines
        else:
            fault_iter = []
        for ln in fault_iter:
            if ln is None or ln.is_empty:
                continue
            bx0, by0, bx1, by1 = ln.bounds
            xmins.append(float(bx0))
            xmaxs.append(float(bx1))
            ymins.append(float(by0))
            ymaxs.append(float(by1))
        if not xmins:
            return []
        x0, x1 = float(np.nanmin(xmins)), float(np.nanmax(xmaxs))
        y0, y1 = float(np.nanmin(ymins)), float(np.nanmax(ymaxs))
        xpad = max(50.0, 0.08 * max(x1 - x0, 1.0))
        ypad = max(50.0, 0.08 * max(y1 - y0, 1.0))
        ax.set_xlim(x0 - xpad, x1 + xpad)
        ax.set_ylim(y0 - ypad, y1 + ypad)
        self._attach_picker_dynamic_basemap(fig, ax, alpha=0.30, map_crs=map_crs)
        if fault_lines is not None and len(fault_lines) > 0:
            shown_fault = False
            for ln in fault_lines:
                if ln is None or ln.is_empty:
                    continue
                x, y = ln.xy
                ax.plot(
                    x,
                    y,
                    color="0.55",
                    linewidth=1.1,
                    alpha=0.9,
                    zorder=4,
                    label="Fault trace (not selectable)" if not shown_fault else None,
                )
                shown_fault = True
        shown_pipeline_selectable = False
        for ln in pipeline_lines:
            if ln is None or ln.is_empty:
                continue
            x, y = ln.xy
            ax.plot(
                x,
                y,
                color="C0",
                linewidth=2.6,
                alpha=0.95,
                zorder=5,
                label="Pipeline (selectable)" if not shown_pipeline_selectable else None,
            )
            shown_pipeline_selectable = True
        selected = set()
        click_tol = float(getattr(self.config, "fault_picker_click_max_dist_m", 200.0))
        sel_lines = []
        done = {"closed": False}

        def _redraw():
            for ln in sel_lines:
                try:
                    ln.remove()
                except Exception:
                    pass
            sel_lines.clear()
            for idx in sorted(selected):
                x, y = pipeline_lines[idx].xy
                (ln,) = ax.plot(x, y, color="red", linewidth=2.8, zorder=6)
                sel_lines.append(ln)
            ax.set_title(
                f"Pick pipeline segments | selected={sorted(selected)} | click=toggle, q=done, r=reset"
            )
            fig.canvas.draw_idle()

        def _on_click(event):
            if event.inaxes != ax or event.button != 1 or event.xdata is None or event.ydata is None:
                return
            pt = Point(float(event.xdata), float(event.ydata))
            best_idx = None
            best_dist = np.inf
            for idx in plot_ids:
                d = float(pipeline_lines[idx].distance(pt))
                if d < best_dist:
                    best_dist = d
                    best_idx = idx
            if best_idx is None or best_dist > click_tol:
                return
            if best_idx in selected:
                selected.remove(best_idx)
            else:
                selected.add(best_idx)
            _redraw()

        def _on_key(event):
            if event.key == "r":
                selected.clear()
                _redraw()
            elif event.key == "q":
                done["closed"] = True
                plt.close(fig)

        def _on_close(_event):
            done["closed"] = True

        fig.canvas.mpl_connect("button_press_event", _on_click)
        fig.canvas.mpl_connect("key_press_event", _on_key)
        fig.canvas.mpl_connect("close_event", _on_close)
        ax.legend(loc="best")
        plt.show(block=False)
        while not done["closed"] and plt.fignum_exists(fig.number):
            plt.pause(0.05)
        return sorted(selected)

    def _set_combo_text_if_available(self, combo: Optional[QComboBox], text: Optional[str]):
        if combo is None or text is None:
            return
        idx = combo.findText(str(text))
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _build_ui_selection_snapshot(self) -> Dict[str, Any]:
        epoch_idx = self.cmb_step1_epoch.currentData() if hasattr(self, "cmb_step1_epoch") else -1
        return {
            "tab_index": int(self.tabs.currentIndex()) if hasattr(self, "tabs") else 0,
            "epoch_index": int(epoch_idx) if isinstance(epoch_idx, (int, np.integer)) else -1,
            "fit_data": self.cmb_fit_data.currentText() if hasattr(self, "cmb_fit_data") else "",
            "fit_dir": self.cmb_fit_dir.currentText() if hasattr(self, "cmb_fit_dir") else "",
            "disp_coord": self.cmb_disp_coord.currentText() if hasattr(self, "cmb_disp_coord") else "",
            "disp_field_dir": self.cmb_disp_field_dir.currentText() if hasattr(self, "cmb_disp_field_dir") else "",
            "disp_field_target": (
                self.cmb_disp_field_target.currentText() if hasattr(self, "cmb_disp_field_target") else ""
            ),
            "strain_type": self.cmb_strain_type.currentText() if hasattr(self, "cmb_strain_type") else "",
            "strain_field_view": (
                self.cmb_strain_field_view.currentText() if hasattr(self, "cmb_strain_field_view") else ""
            ),
            "strain_field_target": (
                self.cmb_strain_field_target.currentText() if hasattr(self, "cmb_strain_field_target") else ""
            ),
            "fit_override_a": self.le_fit_a.text() if hasattr(self, "le_fit_a") else "",
            "fit_override_b": self.le_fit_b.text() if hasattr(self, "le_fit_b") else "",
            "fit_override_c": self.le_fit_c.text() if hasattr(self, "le_fit_c") else "",
            "fit_override_d": self.le_fit_d.text() if hasattr(self, "le_fit_d") else "",
            "b_enhancement_params": copy.deepcopy(self._b_enhancement_popup_values),
        }

    def _build_state_snapshot(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "config_dict": self.config.to_dict(),
            "pipeline_state": self.pipeline_state,
            "last_result": self.last_result,
            "ui_selection": self._build_ui_selection_snapshot(),
        }

    def _restore_ui_selection(self, selection: Dict[str, Any]):
        if not isinstance(selection, dict):
            return
        controls = []
        for name in (
            "cmb_fit_data",
            "cmb_fit_dir",
            "cmb_disp_coord",
            "cmb_disp_field_dir",
            "cmb_disp_field_target",
            "cmb_strain_type",
            "cmb_strain_field_view",
            "cmb_strain_field_target",
            "cmb_step1_epoch",
            "tabs",
        ):
            if hasattr(self, name):
                controls.append(getattr(self, name))
        for c in controls:
            c.blockSignals(True)
        try:
            self._set_combo_text_if_available(
                self.cmb_fit_data if hasattr(self, "cmb_fit_data") else None,
                selection.get("fit_data"),
            )
            self._set_combo_text_if_available(
                self.cmb_fit_dir if hasattr(self, "cmb_fit_dir") else None,
                selection.get("fit_dir"),
            )
            self._set_combo_text_if_available(
                self.cmb_disp_coord if hasattr(self, "cmb_disp_coord") else None,
                selection.get("disp_coord"),
            )
            self._refresh_disp_field_direction_options()
            self._set_combo_text_if_available(
                self.cmb_disp_field_dir if hasattr(self, "cmb_disp_field_dir") else None,
                selection.get("disp_field_dir"),
            )
            self._set_combo_text_if_available(
                self.cmb_disp_field_target if hasattr(self, "cmb_disp_field_target") else None,
                selection.get("disp_field_target"),
            )
            self._set_combo_text_if_available(
                self.cmb_strain_type if hasattr(self, "cmb_strain_type") else None,
                selection.get("strain_type"),
            )
            self._set_combo_text_if_available(
                self.cmb_strain_field_view if hasattr(self, "cmb_strain_field_view") else None,
                selection.get("strain_field_view"),
            )
            self._set_combo_text_if_available(
                self.cmb_strain_field_target if hasattr(self, "cmb_strain_field_target") else None,
                selection.get("strain_field_target"),
            )
            target_epoch = selection.get("epoch_index")
            if hasattr(self, "cmb_step1_epoch") and isinstance(target_epoch, (int, np.integer)):
                for i in range(self.cmb_step1_epoch.count()):
                    data = self.cmb_step1_epoch.itemData(i)
                    if isinstance(data, (int, np.integer)) and int(data) == int(target_epoch):
                        self.cmb_step1_epoch.setCurrentIndex(i)
                        break
            if hasattr(self, "tabs"):
                tab_index = selection.get("tab_index", 0)
                if isinstance(tab_index, (int, np.integer)):
                    ti = int(tab_index)
                    if 0 <= ti < self.tabs.count():
                        self.tabs.setCurrentIndex(ti)
            if hasattr(self, "le_fit_a"):
                self.le_fit_a.setText(str(selection.get("fit_override_a", self.le_fit_a.text())))
            if hasattr(self, "le_fit_b"):
                self.le_fit_b.setText(str(selection.get("fit_override_b", self.le_fit_b.text())))
            if hasattr(self, "le_fit_c"):
                self.le_fit_c.setText(str(selection.get("fit_override_c", self.le_fit_c.text())))
            if hasattr(self, "le_fit_d"):
                self.le_fit_d.setText(str(selection.get("fit_override_d", self.le_fit_d.text())))
            b_params = selection.get("b_enhancement_params")
            if isinstance(b_params, dict):
                self._b_enhancement_popup_values.update(copy.deepcopy(b_params))
        finally:
            for c in controls:
                c.blockSignals(False)

    def _restore_state_snapshot(self, snapshot: Dict[str, Any]):
        if not isinstance(snapshot, dict):
            raise ValueError("Invalid state file: snapshot root must be a dict.")
        schema = snapshot.get("schema_version", 0)
        if int(schema) != 1:
            self.append_log(f"[State] Warning: schema_version={schema}, trying best-effort restore.")
        cfg_dict = snapshot.get("config_dict", {})
        if isinstance(cfg_dict, dict):
            self.config = AppConfig.from_dict(cfg_dict)
            self._fill_from_config(self.config)
        pipeline_state = snapshot.get("pipeline_state", None)
        last_result = snapshot.get("last_result", None)
        self.pipeline_state = pipeline_state if isinstance(pipeline_state, dict) else None
        if isinstance(last_result, dict):
            self.last_result = last_result
        elif isinstance(self.pipeline_state, dict) and "arrays" in self.pipeline_state and "tables" in self.pipeline_state:
            self.last_result = self.pipeline_state
        else:
            self.last_result = None
        self._update_step1_epoch_selector(self.pipeline_state if isinstance(self.pipeline_state, dict) else {})
        ui_sel = snapshot.get("ui_selection", {})
        self._restore_ui_selection(ui_sel if isinstance(ui_sel, dict) else {})
        self._set_run_buttons_enabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_export.setEnabled(self.last_result is not None)
        self.btn_export_opensra.setEnabled(self.last_result is not None)
        self.btn_open.setEnabled(self.last_result is not None)
        self._mount_viz_controls_into_current_tab()
        self._update_plot_control_visibility()
        if isinstance(self.pipeline_state, dict):
            self.render_plots(self.pipeline_state)
            self._sync_fit_override_editor_from_state()

    def on_save_state(self):
        if not self.pipeline_state:
            QMessageBox.information(self, "Info", "No pipeline state to save yet.")
            return
        suggested = Path(self.config.output_dir).expanduser().resolve() / "pipeline_state.opensra_state.pkl"
        fp, _ = QFileDialog.getSaveFileName(
            self,
            "Save Pipeline State",
            str(suggested),
            "OpenSRA State (*.opensra_state.pkl);;Pickle Files (*.pkl);;All Files (*.*)",
        )
        if not fp:
            return
        try:
            snap = self._build_state_snapshot()
            with open(fp, "wb") as f:
                pickle.dump(snap, f, protocol=pickle.HIGHEST_PROTOCOL)
            self.append_log(f"[State] Saved: {fp}")
            QMessageBox.information(self, "State Saved", f"State has been saved successfully.\n\n{fp}")
        except Exception as e:
            self.append_log(traceback.format_exc())
            QMessageBox.critical(self, "Save State Failed", str(e))

    def on_load_state(self):
        fp, _ = QFileDialog.getOpenFileName(
            self,
            "Load Pipeline State",
            str(Path.cwd()),
            "OpenSRA State (*.opensra_state.pkl);;Pickle Files (*.pkl);;All Files (*.*)",
        )
        if not fp:
            return
        try:
            with open(fp, "rb") as f:
                snap = pickle.load(f)
            self._restore_state_snapshot(snap)
            self.append_log(f"[State] Loaded: {fp}")
            QMessageBox.information(self, "State Loaded", f"State has been successfully loaded.\n\n{fp}")
        except Exception as e:
            self.append_log(traceback.format_exc())
            QMessageBox.critical(self, "Load State Failed", str(e))

    def _set_run_buttons_enabled(self, enabled: bool):
        self.btn_run_step1.setEnabled(enabled)
        self.btn_run_step2.setEnabled(enabled)
        self.btn_run_step3.setEnabled(enabled)
        self.btn_run_step4.setEnabled(enabled)
        if hasattr(self, "btn_run_fiber_calib"):
            self.btn_run_fiber_calib.setEnabled(enabled)

    def _start_worker(self, task_name: str, step_fn, state: Optional[Dict]):
        self.current_task = task_name
        if task_name in {"step3_displacement", "step4_strain"} and isinstance(state, dict):
            guard = {}
            for k in (
                "tangential_fit_by_epoch",
                "normal_fit_by_epoch",
                "epistemic_b_by_epoch",
                "epistemic_b_ne_by_epoch",
                "fault_fit_override_by_epoch",
            ):
                if k in state:
                    guard[k] = copy.deepcopy(state.get(k))
            self._fit_state_guard = guard if guard else None
        else:
            self._fit_state_guard = None
        self._set_run_buttons_enabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setValue(0)
        self.thread = QThread(self)
        self.worker = PipelineWorker(self.config, task_name=task_name, step_fn=step_fn, state=state)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_run_step1(self):
        try:
            self.config = self._read_config_from_ui()
        except Exception as e:
            QMessageBox.critical(self, "Parameter Error", str(e))
            return
        try:
            # Keep picker stage lightweight: only read fault/pipe geometries.
            fault_ui_text = self.inputs["fault_line_shp_path"].text().strip() if "fault_line_shp_path" in self.inputs else ""
            if not fault_ui_text:
                raise ValueError("Fault line shapefile path is empty in Input Files.")
            fault_path = self.config.resolve_path(fault_ui_text)
            if not fault_path.exists():
                raise FileNotFoundError(f"Fault line shapefile does not exist: {fault_path}")
            fault_gdf_picker = gpd.read_file(fault_path)
            if fault_gdf_picker.crs is not None:
                try:
                    fault_gdf_picker = fault_gdf_picker.to_crs(epsg=self._processing_epsg())
                except Exception:
                    pass
            fault_lines = self._flatten_fault_lines_for_picker(fault_gdf_picker)
            if fault_lines is None or len(fault_lines) == 0:
                raise ValueError(f"No valid fault segments found in: {fault_path}")

            pipe_path = self.config.resolve_path(self.config.pipe_axis_path)
            pipe_lines = self._load_pipeline_lines_for_picker(pipe_path, target_crs=self._processing_crs_text())
            if pipe_lines is None or len(pipe_lines) == 0:
                raise ValueError(f"No valid pipeline segments found in: {pipe_path}")
            picker_crs = str(self._processing_crs_text()).upper()
            fault_lines_picker = fault_lines
            pipe_lines_picker = pipe_lines
            # Use EPSG:3857 for picker display to keep basemap loading stable in packaged apps.
            if picker_crs != "EPSG:3857":
                fault_lines_picker = self._reproject_lines_for_picker(fault_lines, picker_crs, "EPSG:3857")
                pipe_lines_picker = self._reproject_lines_for_picker(pipe_lines, picker_crs, "EPSG:3857")
                picker_crs = "EPSG:3857"
            pipe_axis_preview = (
                np.asarray(pipe_lines_picker[0].coords, dtype=float)
                if (pipe_lines_picker is not None and len(pipe_lines_picker) > 0)
                else None
            )
            self.append_log(f"[Fault Picker] fault_shp(UI)={fault_path}")
            self.append_log(f"[Fault Picker] fault_segments={len(fault_lines)}")
            self.append_log(f"[Fault Picker] pipe_path={pipe_path}")
            self.append_log(
                f"[Fault Picker] processing_crs={self._processing_crs_text()}, "
                f"pipeline_input_crs={self._pipeline_input_crs_text()}"
            )
            self.append_log(f"[Fault Picker] display_crs={picker_crs}")
            selected = self._interactive_pick_fault_segments(
                fault_lines_picker,
                pipe_axis_preview,
                pipe_lines=pipe_lines_picker,
                map_crs=picker_crs,
            )
            if selected is None or len(selected) == 0:
                QMessageBox.information(self, "Info", "No fault segments selected. Step 1 canceled.")
                self.append_log("Step 1 canceled: no fault segments selected.")
                return
            self._step1_selected_fault_idxs = selected
            self.append_log(f"[Fault Picker] selected_fault_idxs={selected}")
            selected_pipe = self._interactive_pick_pipeline_segments(
                pipe_lines_picker,
                fault_lines=fault_lines_picker,
                map_crs=picker_crs,
            )
            if selected_pipe is None or len(selected_pipe) == 0:
                QMessageBox.information(self, "Info", "No pipeline segments selected. Step 1 canceled.")
                self.append_log("Step 1 canceled: no pipeline segments selected.")
                return
            self._step1_selected_pipeline_idxs = selected_pipe
            self.append_log(f"[Pipeline Picker] selected_pipeline_idxs={selected_pipe}")
        except Exception as e:
            QMessageBox.critical(self, "Fault Picker Error", str(e))
            self.append_log(traceback.format_exc())
            return
        self.btn_export.setEnabled(False)
        self.btn_export_opensra.setEnabled(False)
        self.append_log("Step 1 started...")
        step1_state = {
            "selected_fault_idxs": list(self._step1_selected_fault_idxs),
            "selected_pipeline_idxs": list(self._step1_selected_pipeline_idxs),
        }
        self._start_worker("step1", run_step_match_3d, step1_state)

    def on_run_step2(self):
        if not self.pipeline_state:
            QMessageBox.information(self, "Info", "Please run Step 1 first.")
            return
        try:
            self.config = self._read_config_from_ui()
        except Exception as e:
            QMessageBox.critical(self, "Parameter Error", str(e))
            return
        self.config.model_type = "sigmoid"
        self.append_log("Step 2 started...")
        self._start_worker("step2", run_step_fit, self.pipeline_state)

    def on_run_step3_displacement(self):
        if not self.pipeline_state or "tangential_fit_by_epoch" not in self.pipeline_state:
            QMessageBox.information(self, "Info", "Please run Step 2 first.")
            return
        try:
            self.config = self._read_config_from_ui()
        except Exception as e:
            QMessageBox.critical(self, "Parameter Error", str(e))
            return
        self.config.model_type = "sigmoid"
        self.config.prediction_coordinate_system = self.cmb_disp_coord.currentText()
        self.append_log(
            f"Step 3 (Displacement) started... Prediction mode: {self.config.prediction_coordinate_system}"
        )
        self._start_worker("step3_displacement", run_step_displacement, self.pipeline_state)

    def on_run_step4_strain(self):
        if (
            not self.pipeline_state
            or "tangential_fit_by_epoch" not in self.pipeline_state
            or "pipe" not in self.pipeline_state
            or "pipe_motion_xy_mm" not in self.pipeline_state["pipe"]
        ):
            QMessageBox.information(self, "Info", "Please run Step 3 (Displacement) first.")
            return
        try:
            self.config = self._read_config_from_ui()
        except Exception as e:
            QMessageBox.critical(self, "Parameter Error", str(e))
            return
        self.config.model_type = "sigmoid"
        self.config.prediction_coordinate_system = self.cmb_disp_coord.currentText()
        self.append_log(f"Step 4 (Strain) started... Prediction mode: {self.config.prediction_coordinate_system}")
        self._start_worker("step4_strain", run_step_strain, self.pipeline_state)

    def on_stop(self):
        if self.worker:
            self.worker.stop()
            self.append_log("Stop requested.")
        self.btn_stop.setEnabled(False)

    def on_progress(self, pct: int, msg: str):
        self.progress.setValue(int(pct))
        self.append_log(f"[{pct:3d}%] {msg}")

    def on_failed(self, tb: str):
        self._fit_state_guard = None
        self._set_run_buttons_enabled(True)
        self.btn_stop.setEnabled(False)
        self.append_log("Run failed:")
        self.append_log(tb)
        msg = tb.splitlines()[-1] if tb else "Unknown error"
        if "Insufficient fit points in fitting band" in (tb or ""):
            hint = (
                "Insufficient fit points in fitting band. "
                "Please increase Transect Half Length or Perpendicular Offset Lines, then run Step 2 again."
            )
            self.append_log(f"[Step2 Hint] {hint}")
            msg = hint
        QMessageBox.critical(self, "Run Failed", msg)

    def on_finished(self, result: Dict):
        if (
            self.current_task in {"step3_displacement", "step4_strain"}
            and isinstance(result, dict)
            and isinstance(self._fit_state_guard, dict)
        ):
            for k, v in self._fit_state_guard.items():
                result[k] = copy.deepcopy(v)
        self._fit_state_guard = None
        self.pipeline_state = result
        if self.current_task == "fiber_calibration":
            if isinstance(result, dict) and "arrays" in result and "tables" in result:
                self.last_result = result
        else:
            self.last_result = result if "arrays" in result and "tables" in result else None
        self._update_step1_epoch_selector(result)
        self._set_run_buttons_enabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_export.setEnabled(self.last_result is not None)
        self.btn_export_opensra.setEnabled(self.last_result is not None)
        self.btn_open.setEnabled(self.last_result is not None)
        self.append_log(f"{self.current_task} completed.")
        if self.current_task == "step1" and isinstance(result, dict):
            fsum = result.get("fault_trace_summary", {})
            if isinstance(fsum, dict) and fsum:
                msg = (
                    f"[Fault Trace] type={fsum.get('trace_type', 'N/A')}, "
                    f"selected_count={fsum.get('selected_fault_count', 0)}, "
                    f"selected_idxs={fsum.get('selected_fault_idxs', [])}, "
                    f"pipeline_selected_count={fsum.get('selected_pipeline_count', 0)}, "
                    f"pipeline_selected_idxs={fsum.get('selected_pipeline_idxs', [])}"
                )
                if "fit_degree" in fsum:
                    msg += f", fit_degree={fsum.get('fit_degree')}, coef={fsum.get('fit_coefficients', [])}"
                if "constraint_aspect_deg_effective" in fsum:
                    try:
                        aspect_val = float(fsum.get("constraint_aspect_deg_effective"))
                    except Exception:
                        aspect_val = np.nan
                    msg += (
                        f", constraint_aspect_effective={aspect_val:.3f} "
                        f"({fsum.get('constraint_aspect_source_used', 'manual')})"
                    )
                if "step1_cache_data_hit" in fsum:
                    msg += (
                        f", cache[data={fsum.get('step1_cache_data_hit')}, "
                        f"match={fsum.get('step1_cache_match_hit')}, nez={fsum.get('step1_cache_nez_hit')}]"
                    )
                self.append_log(msg)
        if self.current_task == "step1" and isinstance(result, dict):
            auto_view = result.get("step1_auto_view", {})
            if isinstance(auto_view, dict):
                xlim = auto_view.get("xlim")
                ylim = auto_view.get("ylim")
                if (
                    isinstance(xlim, (list, tuple))
                    and isinstance(ylim, (list, tuple))
                    and len(xlim) == 2
                    and len(ylim) == 2
                    and np.isfinite(float(xlim[0]))
                    and np.isfinite(float(xlim[1]))
                    and np.isfinite(float(ylim[0]))
                    and np.isfinite(float(ylim[1]))
                    and float(xlim[0]) < float(xlim[1])
                    and float(ylim[0]) < float(ylim[1])
                ):
                    self.config.map_xlim_min = float(xlim[0])
                    self.config.map_xlim_max = float(xlim[1])
                    self.config.map_ylim_min = float(ylim[0])
                    self.config.map_ylim_max = float(ylim[1])
                    if hasattr(self, "le_map_xlim_min"):
                        self.le_map_xlim_min.setText(f"{self.config.map_xlim_min:.6f}")
                        self.le_map_xlim_max.setText(f"{self.config.map_xlim_max:.6f}")
                        self.le_map_ylim_min.setText(f"{self.config.map_ylim_min:.6f}")
                        self.le_map_ylim_max.setText(f"{self.config.map_ylim_max:.6f}")
                    self.append_log(
                        "[Step1 Map] Auto view updated around pipe-fault intersection "
                        f"(x:[{self.config.map_xlim_min:.2f}, {self.config.map_xlim_max:.2f}], "
                        f"y:[{self.config.map_ylim_min:.2f}, {self.config.map_ylim_max:.2f}])"
                    )
        if self.current_task in {"step2", "step3_displacement", "step4_strain"} and isinstance(result, dict):
            fit_counts = result.get("fit_sample_count_by_mode", {})
            if isinstance(fit_counts, dict) and fit_counts:
                ne_raw = fit_counts.get("fit_samples_ne_raw", None)
                ne_bins = fit_counts.get("fit_bins_ne", None)
                ne_txt = f"{fit_counts.get('fit_samples_ne', 'N/A')}"
                if ne_raw is not None:
                    ne_txt = f"{ne_txt} (raw={ne_raw})"
                if ne_bins is not None:
                    ne_txt = f"{ne_txt}, bins={ne_bins}"
                self.append_log(
                    "[Fit Samples] "
                    f"total={fit_counts.get('total_samples', 'N/A')}, "
                    f"N&E={ne_txt}, "
                    f"Fault={fit_counts.get('fit_samples_fault', 'N/A')}"
                )
            diff_arr = np.asarray(result.get("max_abs_xy_diff_between_modes_mm", []), dtype=float)
            if diff_arr.size > 0 and np.isfinite(diff_arr).any():
                self.append_log(
                    "[Mode Consistency] max |u_NE-u_Fault| by epoch (mm): "
                    + ", ".join(f"{v:.3f}" for v in diff_arr[:8])
                    + (" ..." if diff_arr.size > 8 else "")
                )
        if self.current_task == "step2" and isinstance(result, dict):
            selected = int(result.get("fit_subset_count_selected", 0))
            total = int(result.get("fit_subset_count_total", 0))
            recomputed = bool(result.get("fit_band_recomputed_in_step2", False))
            self.append_log(
                "[Step2 Fit Subset] "
                f"selected={selected}/{total}, recomputed_in_step2={recomputed} "
                "(strict polygon-only selection)"
            )
            if bool(result.get("fit_band", {}).get("quad_fallback", False)):
                self.append_log(
                    "[Step2 Fit Subset] Quadrilateral intersection fallback activated (near-parallel edges)."
                )
        if self.current_task == "step2_b_enhancement" and isinstance(result, dict):
            enh = result.get("b_enhancement_last", {})
            if isinstance(enh, dict):
                summary = enh.get("summary", {}) if isinstance(enh.get("summary", {}), dict) else {}
                rec_b = summary.get("b_recommended", np.nan)
                cur_b = summary.get("b_current_step2", np.nan)
                self.append_log(
                    "[Sigmoid b Enhancement] "
                    f"epoch={int(summary.get('epoch_index', -1)) + 1}, "
                    f"windows={summary.get('offset_count', 'N/A')}, "
                    f"success={summary.get('success_window_count', 'N/A')}, "
                    f"outliers={summary.get('outlier_count', 'N/A')}, "
                    f"b_current={cur_b:.6g} -> b_recommended={rec_b:.6g}"
                    if np.isfinite(float(rec_b)) and np.isfinite(float(cur_b))
                    else "[Sigmoid b Enhancement] completed"
                )
        self.render_plots(result)
        self._sync_fit_override_editor_from_state()
        if self.current_task == "step2_b_enhancement" and isinstance(result, dict):
            enh = result.get("b_enhancement_last", {})
            if isinstance(enh, dict) and enh:
                self._show_b_enhancement_results_dialog(enh)
        if self.current_task == "fiber_calibration" and isinstance(result, dict):
            payload = result.get("fiber_calibration", {})
            if isinstance(payload, dict) and payload:
                self._sync_fiber_calibration_labels(payload)
                b_opt = payload.get("b_optimal", np.nan)
                rmse = payload.get("rmse_optimal", np.nan)
                self.append_log(
                    "[Fiber calibration] "
                    f"epoch={int(payload.get('epoch_idx', -1)) + 1}, "
                    f"b_optimal={float(b_opt):.6g}, RMSE={float(rmse):.6g} με/year. "
                    "Fit override not written until Apply optimal b. Re-run Step 3/4 after apply."
                )
                # Public build: Fiber calibration tab is hidden.
                # if hasattr(self, "tab_fiber_calib"):
                #     self.tabs.setCurrentWidget(self.tab_fiber_calib)

    def on_step1_map_controls_changed(self):
        if self.pipeline_state:
            self.render_plots(self.pipeline_state)
        self._sync_fit_override_editor_from_state()

    def on_plot_tab_changed(self, _index: int):
        self._mount_viz_controls_into_current_tab()
        self._update_plot_control_visibility()
        if self.pipeline_state:
            self.render_plots(self.pipeline_state)

    def _mount_viz_controls_into_current_tab(self):
        if not hasattr(self, "viz_controls_widget") or not hasattr(self, "tabs"):
            return
        current_tab = self.tabs.currentWidget()
        if current_tab is None:
            return
        if self.viz_controls_widget.parent() is current_tab:
            return
        self.viz_controls_widget.setParent(None)
        if hasattr(current_tab, "lay"):
            current_tab.lay.insertWidget(0, self.viz_controls_widget)

    def _lock_visualization_panel_width(self):
        if not hasattr(self, "tabs") or not hasattr(self, "viz_controls_widget"):
            return
        count = int(self.tabs.count())
        if count <= 0:
            return
        current_idx = self.tabs.currentIndex()
        widths = []
        self.tabs.blockSignals(True)
        try:
            for idx in range(count):
                self.tabs.setCurrentIndex(idx)
                self._mount_viz_controls_into_current_tab()
                self._update_plot_control_visibility()
                self.viz_controls_widget.adjustSize()
                widths.append(int(self.viz_controls_widget.sizeHint().width()))
                tab = self.tabs.currentWidget()
                if tab is not None:
                    widths.append(int(tab.sizeHint().width()))
            self.tabs.setCurrentIndex(current_idx)
            self._mount_viz_controls_into_current_tab()
            self._update_plot_control_visibility()
        finally:
            self.tabs.blockSignals(False)
        max_width = max(widths) if widths else 0
        if max_width > 0:
            # Keep width stable, but cap minimum to avoid squeezing the parameter panel.
            hard_cap = 760
            min_w = int(min(max_width, hard_cap))
            self.viz_controls_widget.setMinimumWidth(min_w)
            self.tabs.setMinimumWidth(0)

    def _set_controls_visible(self, controls, visible: bool):
        for w in controls:
            w.setVisible(visible)

    def _refresh_disp_field_direction_options(self):
        if not hasattr(self, "cmb_disp_field_dir"):
            return
        mode = self.cmb_disp_coord.currentText() if hasattr(self, "cmb_disp_coord") else "Fault-parallel/perpendicular"
        if mode == "N&E":
            options = ["North", "East"]
            default = "North"
        else:
            options = ["Fault-parallel", "Fault-perpendicular"]
            default = "Fault-parallel"
        old = self.cmb_disp_field_dir.currentText()
        self.cmb_disp_field_dir.blockSignals(True)
        self.cmb_disp_field_dir.clear()
        self.cmb_disp_field_dir.addItems(options)
        self.cmb_disp_field_dir.setCurrentText(old if old in options else default)
        self.cmb_disp_field_dir.blockSignals(False)

    def _update_plot_control_visibility(self):
        # 0: Map, 1: Fit, 2: Pipeline mesh, 3: Pipeline displacement,
        # 4: Pipeline displacement field, 5: Pipeline strain, 6: Pipeline strain field,
        # 7: Fiber calibration
        idx = self.tabs.currentIndex() if hasattr(self, "tabs") else 0
        map_controls = [
            self.lbl_nez,
            self.cmb_step1_nez,
            self.lbl_vmin,
            self.le_step1_vmin,
            self.lbl_vmax,
            self.le_step1_vmax,
            self.lbl_map_xlim,
            self.le_map_xlim_min,
            self.le_map_xlim_max,
            self.lbl_map_ylim,
            self.le_map_ylim_min,
            self.le_map_ylim_max,
        ]
        fit_controls = [
            self.lbl_fit_data,
            self.cmb_fit_data,
            self.lbl_fit_dir,
            self.cmb_fit_dir,
            self.btn_export_fit_scatter,
            self.lbl_fitting_parameters,
            self.lbl_fit_a,
            self.le_fit_a,
            self.lbl_fit_b,
            self.le_fit_b,
            self.lbl_fit_c,
            self.le_fit_c,
            self.lbl_fit_d,
            self.le_fit_d,
            self.btn_fit_info,
            self.btn_apply_fit_override,
            self.btn_reset_fit_override,
            self.btn_sigmoid_b_enhancement,
            self.btn_para_time_hist,
        ]
        coord_controls = [
            self.lbl_disp_coord,
            self.cmb_disp_coord,
        ]
        abaqus_export_controls = [
            self.btn_export_abaqus_inp,
        ]
        strain_controls = [
            self.lbl_strain_type,
            self.cmb_strain_type,
        ]
        strain_field_controls = [
            self.lbl_strain_field_view,
            self.cmb_strain_field_view,
            self.lbl_strain_field_target,
            self.cmb_strain_field_target,
        ]
        disp_field_controls = [
            self.lbl_disp_field_dir,
            self.cmb_disp_field_dir,
            self.lbl_disp_field_target,
            self.cmb_disp_field_target,
        ]
        fiber_controls = [
            self.lbl_fiber_csv,
            self.le_fiber_csv,
            self.btn_fiber_csv,
            self.lbl_fiber_origin,
            self.le_fiber_origin,
            self.chk_fiber_auto_origin,
            self.chk_fiber_reverse,
            self.lbl_fiber_spacing,
            self.le_fiber_spacing,
            self.chk_fiber_is_microstrain,
            self.lbl_fiber_b_min,
            self.le_fiber_b_min,
            self.lbl_fiber_b_max,
            self.le_fiber_b_max,
            self.lbl_fiber_b_step,
            self.le_fiber_b_step,
            self.btn_run_fiber_calib,
            self.btn_apply_fiber_optimal_b,
            self.lbl_fiber_b_optimal,
            self.le_fiber_b_optimal,
            self.lbl_fiber_rmse,
            self.le_fiber_rmse,
        ] if hasattr(self, "le_fiber_csv") else []
        self._set_controls_visible(map_controls, idx == 0)
        self._set_controls_visible(fit_controls, idx == 1)
        self._set_controls_visible(coord_controls, idx in (2, 3, 4, 5, 6))
        self._set_controls_visible(strain_controls, idx in (5, 6))
        self._set_controls_visible(disp_field_controls, idx == 4)
        self._set_controls_visible(strain_field_controls, idx == 6)
        if fiber_controls:
            self._set_controls_visible(fiber_controls, idx == 7)
        # show_abaqus = (idx == 2) and self._is_ssi_enabled_from_ui()
        show_abaqus = False
        self._set_controls_visible(abaqus_export_controls, show_abaqus)
        self.btn_export_abaqus_inp.setEnabled(show_abaqus)
        if idx == 1:
            self._update_fit_override_control_visibility()

    def _is_ssi_enabled_from_ui(self) -> bool:
        w = self.params.get("ssi_enabled")
        if isinstance(w, QComboBox):
            return w.currentText().strip().lower() == "true"
        return bool(getattr(self.config, "ssi_enabled", False))

    def _update_fit_override_control_visibility(self):
        fit_mode = self.cmb_fit_data.currentText() if hasattr(self, "cmb_fit_data") else "Fault-parallel/perpendicular"
        show_override = fit_mode == "Fault-parallel/perpendicular"
        override_controls = [
            self.lbl_fit_dir,
            self.cmb_fit_dir,
            self.lbl_fit_a,
            self.le_fit_a,
            self.lbl_fit_b,
            self.le_fit_b,
            self.lbl_fit_c,
            self.le_fit_c,
            self.lbl_fit_d,
            self.le_fit_d,
            self.btn_fit_info,
            self.btn_apply_fit_override,
            self.btn_reset_fit_override,
            self.btn_sigmoid_b_enhancement,
            self.btn_para_time_hist,
        ]
        for w in override_controls:
            w.setVisible(show_override and self.tabs.currentIndex() == 1)

    def _get_fit_result_for_direction(self, result: Dict, epoch_idx: int, direction: str):
        if direction == "Fault-parallel":
            return result.get("tangential_fit_by_epoch", {}).get(epoch_idx)
        if direction == "Fault-perpendicular":
            return result.get("normal_fit_by_epoch", {}).get(epoch_idx)
        return None

    def _get_fit_params_for_direction(self, result: Dict, epoch_idx: int, direction: str):
        def _normalize(arr_like):
            arr = np.asarray(arr_like, dtype=float).reshape(-1)
            if arr.size >= 4:
                return arr[:4]
            if arr.size == 3:
                return np.array([arr[0], arr[1], 0.0, arr[2]], dtype=float)
            return None

        fit_obj = self._get_fit_result_for_direction(result, epoch_idx, direction)
        if fit_obj is None:
            return None
        base = _normalize(fit_obj.params)
        if base is None:
            return None
        overrides = result.get("fault_fit_override_by_epoch", {})
        epoch_override = overrides.get(epoch_idx, {}) if isinstance(overrides, dict) else {}
        custom = epoch_override.get(direction)
        if custom is not None:
            arr = _normalize(custom)
            if arr is not None:
                return arr
        return base

    def _has_fit_override_for_epoch(self, result: Dict, epoch_idx: int) -> bool:
        if not isinstance(result, dict) or epoch_idx < 0:
            return False
        overrides = result.get("fault_fit_override_by_epoch", {})
        if not isinstance(overrides, dict):
            return False
        epoch_override = overrides.get(epoch_idx, {})
        if not isinstance(epoch_override, dict):
            return False
        return len(epoch_override) > 0

    def _sync_fit_override_editor_from_state(self):
        if not hasattr(self, "le_fit_a"):
            return
        if not self.pipeline_state:
            return
        epoch_idx = self._selected_epoch_index(self.pipeline_state)
        if epoch_idx < 0:
            return
        direction = self.cmb_fit_dir.currentText() if hasattr(self, "cmb_fit_dir") else "Fault-parallel"
        params = self._get_fit_params_for_direction(self.pipeline_state, epoch_idx, direction)
        if params is None:
            return
        self.le_fit_a.setText(self._format_fit_param_display(params[0]))
        self.le_fit_b.setText(self._format_fit_param_display(params[1]))
        self.le_fit_c.setText(self._format_fit_param_display(params[2]))
        self.le_fit_d.setText(self._format_fit_param_display(params[3]))

    def _format_fit_param_display(self, value: float) -> str:
        try:
            v = float(value)
        except Exception:
            return ""
        if not np.isfinite(v):
            return ""
        # Rule requested by UI: <=1e5 -> first 5 characters; >1e5 -> scientific notation.
        if abs(v) > 1e5:
            return f"{v:.1e}"
        s = f"{v:.12g}"
        if "e" in s or "E" in s:
            s = f"{v:.12f}".rstrip("0").rstrip(".")
        return s[:5]

    def on_apply_fit_override(self):
        if not self.pipeline_state:
            QMessageBox.information(self, "Info", "Please run Step 2 first.")
            return
        epoch_idx = self._selected_epoch_index(self.pipeline_state)
        if epoch_idx < 0:
            QMessageBox.information(self, "Info", "Please select a valid date.")
            return
        try:
            a = float(self.le_fit_a.text().strip())
            b = float(self.le_fit_b.text().strip())
            c = float(self.le_fit_c.text().strip())
            d = float(self.le_fit_d.text().strip())
        except Exception:
            QMessageBox.critical(self, "Parameter Error", "a, b, c, d must be valid numbers.")
            return
        direction = self.cmb_fit_dir.currentText() if hasattr(self, "cmb_fit_dir") else "Fault-parallel"
        override_params = np.array([a, b, c, d], dtype=float)
        fit_obj = self._get_fit_result_for_direction(self.pipeline_state, epoch_idx, direction)
        orig_params = self._normalize_fit_params4(getattr(fit_obj, "params", None) if fit_obj is not None else None)
        if orig_params is not None:
            orig_params = np.array(orig_params, dtype=float, copy=True)
        x_data, y_data = self._fit_xy_for_direction(self.pipeline_state, epoch_idx, direction)
        overrides = self.pipeline_state.setdefault("fault_fit_override_by_epoch", {})
        epoch_override = overrides.setdefault(epoch_idx, {})
        epoch_override[direction] = [a, b, c, d]
        # Fit overrides are defined for fault-parallel/perpendicular fits.
        # Switch fit mode so the updated curve is immediately visible on the Fit tab.
        if hasattr(self, "cmb_fit_data") and self.cmb_fit_data.currentText() != "Fault-parallel/perpendicular":
            self.cmb_fit_data.blockSignals(True)
            self.cmb_fit_data.setCurrentText("Fault-parallel/perpendicular")
            self.cmb_fit_data.blockSignals(False)
        self.append_log(f"[Fit Override] Date {epoch_idx + 1}, {direction}: a={a:.4f}, b={b:.4f}, c={c:.4f}, d={d:.4f}")
        self._log_fit_override_mse(x_data, y_data, orig_params, override_params)
        self.render_plots(self.pipeline_state)

    def _normalize_fit_params4(self, arr_like):
        arr = np.asarray(arr_like, dtype=float).reshape(-1)
        if arr.size >= 4:
            return arr[:4]
        if arr.size == 3:
            return np.array([arr[0], arr[1], 0.0, arr[2]], dtype=float)
        return None

    def _fit_xy_for_direction(self, result: Dict, epoch_idx: int, direction: str):
        payloads = self._fit_payload_for_selected_mode(result, epoch_idx)
        if not payloads:
            return None, None
        for payload in payloads:
            if str(payload.get("direction", "")) != str(direction):
                continue
            x = np.asarray(payload.get("x_data", []), dtype=float).reshape(-1)
            y = np.asarray(payload.get("y_data", []), dtype=float).reshape(-1)
            n = int(min(x.size, y.size))
            if n <= 0:
                return None, None
            x = x[:n]
            y = y[:n]
            good = np.isfinite(x) & np.isfinite(y)
            if not np.any(good):
                return None, None
            return x[good], y[good]
        return None, None

    def _sigmoid_mse(self, x: np.ndarray, y: np.ndarray, params, half_width_m: Optional[float] = None) -> float:
        p = self._normalize_fit_params4(params)
        if p is None:
            return float("nan")
        xx = np.asarray(x, dtype=float).reshape(-1)
        yy = np.asarray(y, dtype=float).reshape(-1)
        n = int(min(xx.size, yy.size))
        if n <= 0:
            return float("nan")
        xx = xx[:n]
        yy = yy[:n]
        if half_width_m is not None:
            local = np.abs(xx - float(p[2])) <= float(half_width_m)
            if not np.any(local):
                return float("nan")
            xx = xx[local]
            yy = yy[local]
        yhat = sigmoid(xx, float(p[0]), float(p[1]), float(p[2]), float(p[3]))
        resid = yy - yhat
        keep = np.isfinite(resid)
        if not np.any(keep):
            return float("nan")
        return float(np.mean(resid[keep] ** 2))

    def _count_local_fit_points(self, x: np.ndarray, c_value: float, half_width_m: float = 50.0) -> int:
        xx = np.asarray(x, dtype=float).reshape(-1)
        if xx.size == 0 or not np.isfinite(c_value):
            return 0
        return int(np.count_nonzero(np.abs(xx - float(c_value)) <= float(half_width_m)))

    def _log_fit_override_mse(self, x, y, orig_params, override_params):
        if x is None or y is None:
            self.append_log("[Fit Override MSE] no fit samples available.")
            return
        orig = self._normalize_fit_params4(orig_params)
        over = self._normalize_fit_params4(override_params)
        orig_all = self._sigmoid_mse(x, y, orig) if orig is not None else float("nan")
        orig_local = self._sigmoid_mse(x, y, orig, half_width_m=50.0) if orig is not None else float("nan")
        over_all = self._sigmoid_mse(x, y, over) if over is not None else float("nan")
        over_local = self._sigmoid_mse(x, y, over, half_width_m=50.0) if over is not None else float("nan")
        n_all = int(np.asarray(x).size)
        n_orig_local = self._count_local_fit_points(x, float(orig[2])) if orig is not None else 0
        n_over_local = self._count_local_fit_points(x, float(over[2])) if over is not None else 0

        def _fmt(v: float) -> str:
            return f"{v:.6g}" if np.isfinite(v) else "nan"

        def _fmt_params(p) -> str:
            if p is None:
                return "nan"
            return f"a={p[0]:.6g}, b={p[1]:.6g}, c={p[2]:.6g}, d={p[3]:.6g}"

        self.append_log(f"[Fit Override MSE] original params {_fmt_params(orig)}")
        self.append_log(f"[Fit Override MSE] override params {_fmt_params(over)}")
        self.append_log(
            f"[Fit Override MSE] n_all={n_all}  n_|x-c_orig|<=50m={n_orig_local}  n_|x-c_over|<=50m={n_over_local}"
        )
        self.append_log(
            "[Fit Override MSE] "
            f"original all={_fmt(orig_all)}  original |x-c|<=50m={_fmt(orig_local)}  "
            f"override all={_fmt(over_all)}  override |x-c|<=50m={_fmt(over_local)}  "
            f"delta all={_fmt(over_all - orig_all)}  delta |x-c|<=50m={_fmt(over_local - orig_local)}"
        )

    def on_reset_fit_override(self):
        if not self.pipeline_state:
            return
        epoch_idx = self._selected_epoch_index(self.pipeline_state)
        if epoch_idx < 0:
            return
        direction = self.cmb_fit_dir.currentText() if hasattr(self, "cmb_fit_dir") else "Fault-parallel"
        overrides = self.pipeline_state.get("fault_fit_override_by_epoch", {})
        if isinstance(overrides, dict) and epoch_idx in overrides and direction in overrides.get(epoch_idx, {}):
            overrides[epoch_idx].pop(direction, None)
            if not overrides[epoch_idx]:
                overrides.pop(epoch_idx, None)
            self.append_log(f"[Fit Override] Reset Date {epoch_idx + 1}, {direction}.")
        self._sync_fit_override_editor_from_state()
        self.render_plots(self.pipeline_state)

    def on_export_fault_parallel_scatter(self):
        if not isinstance(self.pipeline_state, dict):
            QMessageBox.information(self, "Info", "No pipeline result found. Run Step 1-2 first.")
            return
        fit_mode = self.cmb_fit_data.currentText() if hasattr(self, "cmb_fit_data") else "Fault-parallel/perpendicular"
        if fit_mode != "Fault-parallel/perpendicular":
            QMessageBox.information(
                self,
                "Info",
                "Please switch Fit data to Fault-parallel/perpendicular before exporting.",
            )
            return
        epoch_idx = self._selected_epoch_index(self.pipeline_state)
        if epoch_idx < 0:
            QMessageBox.information(self, "Info", "Please select a valid date before exporting.")
            return
        payloads = self._fit_payload_for_selected_mode(self.pipeline_state, epoch_idx)
        if not payloads:
            QMessageBox.information(self, "Info", "No fit scatter data available. Run Step 2 first.")
            return
        payload = payloads[0]
        if str(payload.get("direction", "")) != "Fault-parallel":
            for p in payloads:
                if str(p.get("direction", "")) == "Fault-parallel":
                    payload = p
                    break
        x_data = np.asarray(payload.get("x_data", []), dtype=float).reshape(-1)
        y_data = np.asarray(payload.get("y_data", []), dtype=float).reshape(-1)
        n = int(min(x_data.size, y_data.size))
        if n <= 0:
            QMessageBox.information(self, "Info", "Fault-parallel scatter is empty.")
            return
        arr = np.column_stack([x_data[:n], y_data[:n]])
        out_dir = Path(self.config.output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        suggested = out_dir / f"fit_scatter_fault_parallel_epoch_{epoch_idx + 1}.csv"
        fp, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Fault-parallel Fit Scatter",
            str(suggested),
            "CSV (*.csv);;Text (*.txt);;NumPy (*.npy)",
        )
        if not fp:
            return
        try:
            ext = Path(fp).suffix.lower()
            if not ext:
                if "NumPy" in selected_filter:
                    ext = ".npy"
                elif "Text" in selected_filter:
                    ext = ".txt"
                else:
                    ext = ".csv"
                fp = f"{fp}{ext}"
            if ext == ".npy":
                np.save(fp, arr)
            elif ext == ".txt":
                np.savetxt(fp, arr, fmt="%.10g", delimiter="\t", header="x_m\ty_mm", comments="")
            else:
                np.savetxt(fp, arr, fmt="%.10g", delimiter=",", header="x_m,y_mm", comments="")
            self.append_log(
                f"[Fit Export] Fault-parallel scatter exported: {fp} (points={n}, epoch={epoch_idx + 1})"
            )
            QMessageBox.information(self, "Export Completed", fp)
        except Exception:
            self.append_log(traceback.format_exc())
            QMessageBox.critical(self, "Export Failed", "Failed to export fault-parallel fit scatter.")

    def on_para_time_hist(self):
        if not isinstance(self.pipeline_state, dict) or "tangential_fit_by_epoch" not in self.pipeline_state:
            QMessageBox.information(self, "Info", "Please run Step 2 first.")
            return
        result = self.pipeline_state
        matched = result.get("matched", {}) if isinstance(result.get("matched", {}), dict) else {}
        pairs = matched.get("matched_pairs", [])
        if not pairs:
            QMessageBox.information(self, "Info", "No matched dates available.")
            return
        direction = self.cmb_fit_dir.currentText() if hasattr(self, "cmb_fit_dir") else "Fault-parallel"
        n = len(pairs)
        params = np.full((n, 4), np.nan, dtype=float)
        for i in range(n):
            vals = self._get_fit_params_for_direction(result, i, direction)
            if vals is not None:
                params[i, :] = np.asarray(vals, dtype=float)[:4]
        day_diffs = np.asarray(matched.get("day_diffs", []), dtype=float).reshape(-1)
        if day_diffs.size == n:
            x = day_diffs
        else:
            x = np.arange(n, dtype=float)

        def _iqr_keep(y: np.ndarray) -> np.ndarray:
            if y.size < 4:
                return np.ones(y.shape, dtype=bool)
            q1, q3 = np.nanpercentile(y, [25.0, 75.0])
            iqr = float(q3 - q1)
            if not np.isfinite(iqr) or iqr <= 0:
                return np.isfinite(y)
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            return np.isfinite(y) & (y >= lo) & (y <= hi)

        fig = Figure(figsize=(11, 8), tight_layout=False)
        axes = fig.subplots(2, 2)
        names = ["a", "b", "c", "d"]
        fit_start = 30
        for ax, name, col in zip(axes.ravel(), names, range(4)):
            y = params[:, col]
            ax.scatter(x, y, s=28, color="orange", label="Scatter Points", zorder=3)
            fit_x = x[fit_start:]
            fit_y = y[fit_start:]
            finite = np.isfinite(fit_x) & np.isfinite(fit_y)
            fit_x = fit_x[finite]
            fit_y = fit_y[finite]
            keep = _iqr_keep(fit_y)
            fit_x = fit_x[keep]
            fit_y = fit_y[keep]
            if fit_x.size >= 2:
                slope, intercept = np.polyfit(fit_x.astype(float), fit_y.astype(float), 1)
                x_line = np.linspace(float(np.min(fit_x)), float(np.max(fit_x)), 100)
                y_line = slope * x_line + intercept
                ax.plot(
                    x_line,
                    y_line,
                    color="tab:blue",
                    linewidth=1.8,
                    label=f"Fit Line: y = {slope:.6f}x + {intercept:.4f}",
                    zorder=2,
                )
            ax.set_title(f"{direction}: {name}")
            ax.set_xlabel("Day")
            ax.set_ylabel(name)
            ax.grid(alpha=0.3)
            ax.legend(loc="best", fontsize=8)
        fig.suptitle(f"Sigmoid parameter time history ({direction})")
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95], pad=0.7)

        dlg = QDialog(self)
        dlg.setWindowTitle("Para time hist")
        dlg.resize(1100, 850)
        lay = QVBoxLayout(dlg)
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, dlg)
        lay.addWidget(toolbar)
        lay.addWidget(canvas, stretch=1)
        btns = QDialogButtonBox(QDialogButtonBox.Close, parent=dlg)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)
        dlg.exec()

    def on_browse_fiber_csv(self):
        start = self.le_fiber_csv.text().strip() if hasattr(self, "le_fiber_csv") else ""
        try:
            start_path = str(self.config.resolve_path(start)) if start else str(Path.cwd())
        except Exception:
            start_path = str(Path.cwd())
        p, _ = QFileDialog.getOpenFileName(
            self,
            "Select fiber file",
            start_path,
            "Fiber files (*.npy *.csv);;NumPy (*.npy);;CSV Files (*.csv);;All Files (*.*)",
        )
        if p:
            self.le_fiber_csv.setText(p)

    def _read_fiber_calibration_request(self) -> Dict[str, Any]:
        def _f(widget, default: float) -> float:
            txt = widget.text().strip() if widget is not None else ""
            if not txt:
                return float(default)
            return float(txt)

        return {
            "fiber_path": self.le_fiber_csv.text().strip(),
            "spacing_m": _f(self.le_fiber_spacing, 0.1),
            "origin_offset_m": _f(self.le_fiber_origin, -119.0),
            "auto_origin": bool(self.chk_fiber_auto_origin.isChecked()) if hasattr(self, "chk_fiber_auto_origin") else True,
            "reverse": bool(self.chk_fiber_reverse.isChecked()),
            "fiber_is_microstrain": bool(self.chk_fiber_is_microstrain.isChecked()),
            "b_min": _f(self.le_fiber_b_min, 0.05),
            "b_max": _f(self.le_fiber_b_max, 2.0),
            "b_step": _f(self.le_fiber_b_step, 0.1),
        }

    def _sync_fiber_calibration_labels(self, payload: Optional[Dict] = None):
        if not hasattr(self, "le_fiber_b_optimal"):
            return
        if not isinstance(payload, dict):
            payload = (self.pipeline_state or {}).get("fiber_calibration", {}) if isinstance(self.pipeline_state, dict) else {}
        b_opt = payload.get("b_optimal", np.nan) if isinstance(payload, dict) else np.nan
        rmse = payload.get("rmse_optimal", np.nan) if isinstance(payload, dict) else np.nan
        try:
            b_f = float(b_opt)
        except Exception:
            b_f = float("nan")
        try:
            rmse_f = float(rmse)
        except Exception:
            rmse_f = float("nan")
        self.le_fiber_b_optimal.setText(f"{b_f:.6g}" if np.isfinite(b_f) else "")
        self.le_fiber_rmse.setText(f"{rmse_f:.6g}" if np.isfinite(rmse_f) else "")
        if hasattr(self, "le_fiber_origin") and isinstance(payload, dict) and payload.get("auto_origin", False):
            try:
                origin_f = float(payload.get("origin_offset_m", np.nan))
            except Exception:
                origin_f = float("nan")
            if np.isfinite(origin_f):
                self.le_fiber_origin.setText(f"{origin_f:.4g}")
        if hasattr(self, "btn_apply_fiber_optimal_b"):
            self.btn_apply_fiber_optimal_b.setEnabled(bool(np.isfinite(b_f)))

    def _on_fiber_auto_origin_toggled(self, checked: bool):
        if hasattr(self, "le_fiber_origin"):
            self.le_fiber_origin.setReadOnly(bool(checked))

    def on_run_fiber_calibration(self):
        if not isinstance(self.pipeline_state, dict) or "tangential_fit_by_epoch" not in self.pipeline_state:
            QMessageBox.information(self, "Info", "Please run Step 2 first.")
            return
        epoch_idx = self._selected_epoch_index(self.pipeline_state)
        if epoch_idx < 0:
            QMessageBox.information(self, "Info", "Please select a valid date.")
            return
        try:
            self.config = self._read_config_from_ui()
        except Exception as e:
            QMessageBox.critical(self, "Parameter Error", str(e))
            return
        try:
            req = self._read_fiber_calibration_request()
        except Exception as e:
            QMessageBox.critical(self, "Parameter Error", f"Fiber calibration inputs must be valid numbers. {e}")
            return
        if float(req["b_step"]) <= 0:
            QMessageBox.critical(self, "Parameter Error", "step must be > 0.")
            return
        if not req["fiber_path"]:
            QMessageBox.critical(self, "Parameter Error", "Fiber file path is empty.")
            return
        matched = self.pipeline_state.get("matched", {}) if isinstance(self.pipeline_state, dict) else {}
        day_diffs = np.asarray(matched.get("day_diffs", []), dtype=float).reshape(-1)
        if epoch_idx >= day_diffs.size or abs(float(day_diffs[epoch_idx])) < 1e-12:
            QMessageBox.information(
                self,
                "Info",
                "Selected date has zero elapsed time from the baseline. Please choose a later date.",
            )
            return
        req["epoch_idx"] = int(epoch_idx)
        self.config.model_type = "sigmoid"
        self.pipeline_state["fiber_calibration_request"] = req
        b_step = float(req["b_step"])
        ssi_on = self._is_ssi_enabled_from_ui()
        auto_origin = bool(req.get("auto_origin", True))
        self.append_log(
            "[Fiber calibration] started "
            f"(epoch={epoch_idx + 1}, b=[{req['b_min']}, {req['b_max']}], step={b_step}, "
            f"origin={'auto |strain| peak' if auto_origin else req['origin_offset_m']}, "
            f"SSI={'on' if ssi_on else 'off'}). "
            "Each trial uses the Step 3+4 kernel only (no b16/b50/b84)."
        )
        self._start_worker("fiber_calibration", run_fiber_b_calibration, self.pipeline_state)

    def on_apply_fiber_optimal_b(self):
        if not isinstance(self.pipeline_state, dict):
            QMessageBox.information(self, "Info", "Please run Step 2 first.")
            return
        payload = self.pipeline_state.get("fiber_calibration", {})
        if not isinstance(payload, dict) or not payload:
            QMessageBox.information(self, "Info", "Run fiber calibration first.")
            return
        try:
            b_opt = float(payload.get("b_optimal", np.nan))
        except Exception:
            b_opt = float("nan")
        if not np.isfinite(b_opt):
            QMessageBox.information(self, "Info", "No finite optimal b to apply.")
            return
        epoch_idx = int(payload.get("epoch_idx", self._selected_epoch_index(self.pipeline_state)))
        if epoch_idx < 0:
            QMessageBox.information(self, "Info", "Please select a valid date.")
            return
        params = self._get_fit_params_for_direction(self.pipeline_state, epoch_idx, "Fault-parallel")
        if params is None:
            QMessageBox.information(self, "Info", "Fault-parallel fit is missing for the calibrated date.")
            return
        a = float(params[0])
        c = float(params[2])
        d = float(params[3])
        overrides = self.pipeline_state.setdefault("fault_fit_override_by_epoch", {})
        epoch_override = overrides.setdefault(epoch_idx, {})
        epoch_override["Fault-parallel"] = [a, b_opt, c, d]
        if hasattr(self, "cmb_fit_data") and self.cmb_fit_data.currentText() != "Fault-parallel/perpendicular":
            self.cmb_fit_data.setCurrentText("Fault-parallel/perpendicular")
        if hasattr(self, "cmb_step1_epoch"):
            for i in range(self.cmb_step1_epoch.count()):
                if int(self.cmb_step1_epoch.itemData(i) or -1) == epoch_idx:
                    self.cmb_step1_epoch.setCurrentIndex(i)
                    break
        self._sync_fit_override_editor_from_state()
        self.append_log(
            f"[Fiber calibration] Applied optimal b to Date {epoch_idx + 1} Fault-parallel: "
            f"a={a:.4f}, b={b_opt:.4f}, c={c:.4f}, d={d:.4f}. "
            "Re-run Step 3 and Step 4 to refresh Displacement/Strain tabs."
        )
        self.render_plots(self.pipeline_state)

    def _render_fiber_calibration_plot(self, result: Dict):
        if not hasattr(self, "tab_fiber_calib"):
            return
        ax = self.tab_fiber_calib.ax
        ax.clear()
        payload = result.get("fiber_calibration", {}) if isinstance(result, dict) else {}
        if not isinstance(payload, dict) or not payload:
            ax.text(
                0.5,
                0.5,
                "Run fiber calibration to compare yearly axial strain.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xlabel("Location (m)")
            ax.set_ylabel("Yearly axial strain (με/year)")
            self.tab_fiber_calib.canvas.draw_idle()
            return
        fiber_x = np.asarray(payload.get("fiber_x", []), dtype=float).reshape(-1)
        fiber_y = np.asarray(payload.get("fiber_y", []), dtype=float).reshape(-1)
        model_x = np.asarray(payload.get("model_x", []), dtype=float).reshape(-1)
        model_y = np.asarray(payload.get("model_y_optimal", []), dtype=float).reshape(-1)
        cur_x = np.asarray(payload.get("model_x_current", []), dtype=float).reshape(-1)
        cur_y = np.asarray(payload.get("model_y_current", []), dtype=float).reshape(-1)
        if fiber_x.size and fiber_y.size and fiber_x.size == fiber_y.size:
            ax.plot(fiber_x, fiber_y, color="0.35", lw=1.4, alpha=0.9, label="Fiber")
        b_cur = payload.get("b_current", np.nan)
        b_opt = payload.get("b_optimal", np.nan)
        try:
            b_cur_f = float(b_cur)
        except Exception:
            b_cur_f = float("nan")
        try:
            b_opt_f = float(b_opt)
        except Exception:
            b_opt_f = float("nan")
        plot_current = (
            cur_x.size
            and cur_y.size
            and cur_x.size == cur_y.size
            and np.isfinite(b_cur_f)
            and np.isfinite(b_opt_f)
            and abs(b_cur_f - b_opt_f) > 1e-8
        )
        if plot_current:
            ax.plot(
                cur_x,
                cur_y,
                color="#9bb7d4",
                lw=1.6,
                alpha=0.85,
                label=f"Current b={b_cur_f:.4g}",
            )
        if model_x.size and model_y.size and model_x.size == model_y.size:
            ax.plot(
                model_x,
                model_y,
                color="#1f77b4",
                lw=2.2,
                label=f"Optimal b={b_opt_f:.4g}" if np.isfinite(b_opt_f) else "Optimal b",
            )
        epoch_idx = int(payload.get("epoch_idx", -1))
        rmse = payload.get("rmse_optimal", np.nan)
        ssi_txt = "SSI on" if payload.get("ssi_enabled", False) else "SSI off"
        ax.set_xlabel("Location (m)")
        ax.set_ylabel("Yearly axial strain (με/year)")
        ax.set_title(
            f"Fiber calibration  Date {epoch_idx + 1}  RMSE={float(rmse):.4g} με/year  ({ssi_txt})",
            pad=6,
        )
        ax.grid(ls="--", alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        self.tab_fiber_calib.canvas.draw_idle()

    def on_sigmoid_b_enhancement(self):
        if not isinstance(self.pipeline_state, dict) or "tangential_fit_by_epoch" not in self.pipeline_state:
            QMessageBox.information(self, "Info", "Please run Step 2 first.")
            return
        epoch_idx = self._selected_epoch_index(self.pipeline_state)
        if epoch_idx < 0:
            QMessageBox.information(self, "Info", "Please select a valid date.")
            return
        try:
            self.config = self._read_config_from_ui()
        except Exception as e:
            QMessageBox.critical(self, "Parameter Error", str(e))
            return
        params = self._open_b_enhancement_params_dialog()
        if params is None:
            return
        self._b_enhancement_popup_values.update(copy.deepcopy(params))
        req = dict(params)
        req["epoch_idx"] = int(epoch_idx)
        self.pipeline_state["b_enhancement_request"] = req
        self.append_log(
            "[Sigmoid b Enhancement] started "
            f"(epoch={epoch_idx + 1}, offsets=[{req['offset_min_m']}, {req['offset_max_m']}], step={req['offset_step_m']})"
        )
        self._start_worker("step2_b_enhancement", run_step_sigmoid_b_enhancement, self.pipeline_state)

    def _open_b_enhancement_params_dialog(self) -> Optional[Dict[str, Any]]:
        vals = dict(self._b_enhancement_popup_values)
        dlg = QDialog(self)
        dlg.setWindowTitle("Sigmoid b Enhancement Parameters")
        lay = QVBoxLayout(dlg)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        num_keys = [
            ("offset_min_m", "offset_min (m)"),
            ("offset_max_m", "offset_max (m)"),
            ("offset_step_m", "offset_step (m)"),
            ("window_half_length_m", "window_half_length (m)"),
            ("window_half_width_m", "window_half_width (m)"),
            ("x_half_for_c_window", "x_half_for_c_window (m)"),
            ("min_points_per_window", "min_points_per_window"),
            ("n_bins", "n_bins"),
            # Public build: keep only peak_height; other peak filters stay at defaults.
            # ("peak_prominence", "peak_prominence"),
            # ("peak_distance", "peak_distance"),
            ("peak_height", "peak_height (optional)"),
            # ("peak_zscore_threshold", "peak_zscore_threshold"),
        ]
        editors: Dict[str, QWidget] = {}
        for key, label in num_keys:
            le = QLineEdit(str(vals.get(key, "")))
            form.addRow(label, le)
            editors[key] = le

        cmb_peak = QComboBox()
        cmb_peak.addItems(["true", "false"])
        cmb_peak.setCurrentText("true" if bool(vals.get("peak_enabled", True)) else "false")
        form.addRow("peak_enabled", cmb_peak)
        editors["peak_enabled"] = cmb_peak

        cmb_remove = QComboBox()
        cmb_remove.addItems(["true", "false"])
        cmb_remove.setCurrentText("true" if bool(vals.get("remove_outliers_for_aggregation", True)) else "false")
        form.addRow("remove_outliers_for_aggregation", cmb_remove)
        editors["remove_outliers_for_aggregation"] = cmb_remove

        le_top_pct = QLineEdit(str(vals.get("top_percent_for_aggregation", 10.0)))
        form.addRow("top_percent_for_aggregation", le_top_pct)
        editors["top_percent_for_aggregation"] = le_top_pct

        lay.addLayout(form)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dlg)
        lay.addWidget(btns)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        if dlg.exec() != QDialog.Accepted:
            return None
        try:
            out = dict(vals)
            for key, _label in num_keys:
                text = editors[key].text().strip()
                if key in {"min_points_per_window", "n_bins", "peak_distance"}:
                    out[key] = int(float(text))
                elif key == "peak_height":
                    out[key] = text
                else:
                    out[key] = float(text)
            out["peak_enabled"] = editors["peak_enabled"].currentText().strip().lower() == "true"
            out["remove_outliers_for_aggregation"] = (
                editors["remove_outliers_for_aggregation"].currentText().strip().lower() == "true"
            )
            out["top_percent_for_aggregation"] = float(editors["top_percent_for_aggregation"].text().strip())
            if out["offset_step_m"] <= 0:
                raise ValueError("offset_step_m must be > 0.")
            if out["offset_max_m"] < out["offset_min_m"]:
                raise ValueError("offset_max_m must be >= offset_min_m.")
            if out["window_half_length_m"] <= 0 or out["window_half_width_m"] <= 0:
                raise ValueError("window_half_length_m and window_half_width_m must be > 0.")
            if out["min_points_per_window"] < 5:
                raise ValueError("min_points_per_window must be >= 5.")
            if out["n_bins"] < 5:
                raise ValueError("n_bins must be >= 5.")
            if not (0 < out["top_percent_for_aggregation"] <= 100):
                raise ValueError("top_percent_for_aggregation must be in (0, 100].")
            return out
        except Exception as e:
            QMessageBox.critical(self, "Parameter Error", str(e))
            return None

    def _attach_figure_tab(self, tabs: QTabWidget, title: str, fig: Figure):
        w = QWidget()
        l = QVBoxLayout(w)
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, w)
        l.addWidget(toolbar)
        l.addWidget(canvas)
        tabs.addTab(w, title)
        return canvas

    def _pick_sparse_label_indices(self, n: int, max_labels: int = 24) -> np.ndarray:
        if n <= 0:
            return np.array([], dtype=int)
        if n <= max_labels:
            return np.arange(n, dtype=int)
        idx = np.linspace(0, n - 1, max_labels).round().astype(int)
        return np.unique(np.clip(idx, 0, n - 1))

    def _build_b_enhancement_map_overview_figure(self, enh: Dict) -> Figure:
        fig = Figure(figsize=(10, 9), tight_layout=False)
        ax = fig.add_subplot(111)
        m = enh.get("map_overview", {}) if isinstance(enh.get("map_overview", {}), dict) else {}
        coords = np.asarray(m.get("matched_coords", []), dtype=float)
        color_values = np.asarray(m.get("color_values", []), dtype=float).reshape(-1)
        pipe_axis = np.asarray(m.get("pipe_axis", []), dtype=float)
        fault_coords = np.asarray(m.get("fault_coords", []), dtype=float)
        inter = np.asarray(m.get("intersection_xy", []), dtype=float).reshape(-1)
        ex_windows = m.get("example_windows", {}) if isinstance(m.get("example_windows", {}), dict) else {}
        if coords.ndim != 2 or coords.shape[0] <= 0 or coords.shape[1] < 2:
            ax.text(0.5, 0.5, "No map overview data", ha="center", va="center", transform=ax.transAxes)
            return fig
        n = min(coords.shape[0], color_values.size if color_values.size > 0 else coords.shape[0])
        clr = color_values[:n] if color_values.size > 0 else np.zeros(n, dtype=float)
        sc = ax.scatter(
            coords[:n, 0],
            coords[:n, 1],
            c=clr,
            cmap="coolwarm",
            s=8,
            vmin=-40,
            vmax=20,
            alpha=0.9,
            label="InSAR data",
        )
        if pipe_axis.ndim == 2 and pipe_axis.shape[0] >= 2:
            ax.plot(pipe_axis[:, 0], pipe_axis[:, 1], color="tab:blue", linewidth=2.5, label="Pipe line")
        if fault_coords.ndim == 2 and fault_coords.shape[0] >= 2:
            ax.plot(fault_coords[:, 0], fault_coords[:, 1], color="red", linestyle="--", linewidth=1.8, label="Fault trace")
        if inter.size >= 2 and np.isfinite(inter[:2]).all():
            ax.scatter([inter[0]], [inter[1]], color="limegreen", s=40, zorder=5, label="Fault-pipe intersection")
        styles = [(0.0, "-", "black"), (float(enh.get("params", {}).get("offset_min_m", -300.0)), "--", "purple"), (float(enh.get("params", {}).get("offset_max_m", 400.0)), "--", "darkgreen")]
        for s_ref, ls, color in styles:
            corners = np.asarray(ex_windows.get(str(float(s_ref)), []), dtype=float)
            if corners.ndim == 2 and corners.shape[0] >= 2 and corners.shape[1] >= 2:
                ax.plot(corners[:, 0], corners[:, 1], linestyle=ls, color=color, linewidth=1.8, label=f"window s={int(s_ref)} m")
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend(loc="best")
        fig.colorbar(sc, ax=ax, label="Displacement-N (mm)", shrink=0.85, fraction=0.045, pad=0.006)
        try:
            cx.add_basemap(
                ax,
                crs="EPSG:3857",
                source=cx.providers.OpenStreetMap.Mapnik,
                headers=self._basemap_headers(),
                alpha=0.5,
            )
        except Exception:
            pass
        ax.set_anchor("C")
        ax.set_title("InSAR points and bounded offset windows (L=1200 m, W=240 m)")
        fig.subplots_adjust(left=0.08, right=0.92, bottom=0.07, top=0.95)
        return fig

    def _mean_b_from_enhancement_fit_df(self, fit_df: pd.DataFrame) -> float:
        if fit_df is None or not isinstance(fit_df, pd.DataFrame) or fit_df.empty or "b" not in fit_df.columns:
            return float("nan")
        ok = fit_df.loc[np.isfinite(pd.to_numeric(fit_df["b"], errors="coerce"))].copy()
        if ok.empty:
            return float("nan")
        if "is_outlier" in ok.columns:
            keep = ok[~ok["is_outlier"].fillna(False).astype(bool)]
        else:
            keep = ok
        keep_b = pd.to_numeric(keep["b"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(keep_b).any():
            return float("nan")
        return float(np.nanmean(keep_b))

    def _build_b_vs_offset_figure(self, fit_df: pd.DataFrame) -> Figure:
        fig = Figure(figsize=(9, 6), tight_layout=True)
        ax = fig.add_subplot(111)
        if fit_df is None or fit_df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return fig
        ok = fit_df.loc[np.isfinite(pd.to_numeric(fit_df["b"], errors="coerce"))].copy()
        if ok.empty:
            ax.text(0.5, 0.5, "No finite b values", ha="center", va="center", transform=ax.transAxes)
            return fig
        out = ok[ok.get("is_outlier", False).astype(bool)]
        keep = ok[~ok.get("is_outlier", False).astype(bool)]
        ax.scatter(keep["offset"], keep["b"], s=28, color="tab:blue", label="b")
        if not out.empty:
            ax.scatter(out["offset"], out["b"], s=36, color="crimson", marker="x", label="outlier")
        mean_b = self._mean_b_from_enhancement_fit_df(ok)
        if np.isfinite(mean_b):
            ax.axhline(mean_b, color="tab:red", linestyle="--", linewidth=1.6, label="mean b")
        cand = ok.copy()
        if len(cand) > 0:
            idx_sparse = self._pick_sparse_label_indices(len(cand), max_labels=22)
            mark = np.zeros(len(cand), dtype=bool)
            mark[idx_sparse] = True
            if "is_outlier" in cand.columns:
                mark = mark | cand["is_outlier"].to_numpy(dtype=bool)
            if len(cand) >= 2:
                vals = cand["b"].to_numpy(dtype=float)
                if np.isfinite(vals).any():
                    i_max = int(np.nanargmax(vals))
                    i_min = int(np.nanargmin(vals))
                    mark[i_max] = True
                    mark[i_min] = True
            cand = cand.iloc[np.where(mark)[0]]
        for _, rr in cand.iterrows():
            ax.annotate(f"s={int(rr['offset'])}", (rr["offset"], rr["b"]), textcoords="offset points", xytext=(3, 3), fontsize=7)
        ax.set_xlabel("Offset along fault (m)")
        ax.set_ylabel("Fitted sigmoid parameter b")
        ax.set_title("b vs offset (outliers marked)")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        return fig

    def _build_fit_matrix_figure(self, fit_df: pd.DataFrame) -> Figure:
        fig = Figure(figsize=(30, 42), tight_layout=False)
        n_rows, n_cols = 12, 6
        axes = fig.subplots(n_rows, n_cols, sharex=True, sharey=True).flatten()
        off_list = fit_df["offset"].tolist() if isinstance(fit_df, pd.DataFrame) and not fit_df.empty else []
        for i, s in enumerate(off_list):
            ax = axes[i]
            rr = fit_df.iloc[i]
            x = np.asarray(rr.get("_x_window", []), dtype=float).reshape(-1)
            y = np.asarray(rr.get("_y_window", []), dtype=float).reshape(-1)
            row, col = divmod(i, n_cols)
            if (not bool(rr.get("success", False))) or x.size <= 0:
                ax.text(0.5, 0.5, f"offset={int(s)} m\n{rr.get('message', 'failed')}", ha="center", va="center", transform=ax.transAxes, fontsize=8)
                ax.text(0.5, 0.97, f"offset={int(s)} m", transform=ax.transAxes, ha="center", va="top", fontsize=7)
                ax.grid(alpha=0.25)
                if row < (n_rows - 1):
                    ax.tick_params(labelbottom=False, direction="in")
                if col > 0:
                    ax.tick_params(labelleft=False, direction="in")
                continue
            ax.scatter(x, y, s=7, alpha=0.25, color="tab:blue")
            try:
                a, b, c, d = float(rr["a"]), float(rr["b"]), float(rr["c"]), float(rr["d"])
                x_fit = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 500)
                y_fit = a * (1.0 / (1.0 + np.exp(-b * (x_fit - c)))) + d
                ax.plot(x_fit, y_fit, color="tab:orange", linewidth=1.8)
                sigma_b = rr.get("std_b", np.nan)
                txt = f"n={int(rr.get('n_points', 0))}\nb={b:.4g}"
                if np.isfinite(float(sigma_b)):
                    txt += f"\nstd_b={float(sigma_b):.3g}"
                if bool(rr.get("b_at_lower_bound", False)):
                    txt += "\nb~lower-bound"
                ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=6.5)
            except Exception:
                pass
            ax.text(0.5, 0.97, f"offset={int(s)} m", transform=ax.transAxes, ha="center", va="top", fontsize=7)
            ax.grid(alpha=0.25)
            ax.tick_params(labelsize=5, direction="in")
            if row < (n_rows - 1):
                ax.tick_params(labelbottom=False, direction="in")
            if col > 0:
                ax.tick_params(labelleft=False, direction="in")
        for j in range(len(off_list), len(axes)):
            axes[j].axis("off")
        fig.suptitle("Sliding-window fitting results", fontsize=16, y=0.992)
        fig.subplots_adjust(left=0.015, right=0.998, bottom=0.015, top=0.972, hspace=0.08, wspace=0.05)
        return fig

    def _build_coverage_matrix_figure(self, coverage_df: pd.DataFrame) -> Figure:
        fig = Figure(figsize=(30, 42), tight_layout=False)
        n_rows, n_cols = 12, 6
        axes = fig.subplots(n_rows, n_cols, sharex=True, sharey=True).flatten()
        off_list = coverage_df["offset"].tolist() if isinstance(coverage_df, pd.DataFrame) and not coverage_df.empty else []
        for i, s in enumerate(off_list):
            ax = axes[i]
            rr = coverage_df.iloc[i]
            centers = np.asarray(rr.get("_bin_centers", []), dtype=float).reshape(-1)
            counts = np.asarray(rr.get("_bin_counts", []), dtype=float).reshape(-1)
            row, col = divmod(i, n_cols)
            if (not bool(rr.get("success", False))) or centers.size <= 0 or counts.size <= 0:
                ax.text(0.5, 0.5, f"offset={int(s)} m\n{rr.get('message', 'no bins')}", ha="center", va="center", transform=ax.transAxes, fontsize=8)
                ax.text(0.5, 0.97, f"offset={int(s)} m", transform=ax.transAxes, ha="center", va="top", fontsize=7)
                ax.grid(alpha=0.25)
                if row < (n_rows - 1):
                    ax.tick_params(labelbottom=False, direction="in")
                if col > 0:
                    ax.tick_params(labelleft=False, direction="in")
                continue
            bar_w = (centers[1] - centers[0]) * 0.85 if centers.size > 1 else 8.0
            ax.bar(centers, counts, width=bar_w, color="tab:blue", alpha=0.8)
            ue = rr.get("uniform_expectation", np.nan)
            if np.isfinite(float(ue)):
                ax.axhline(float(ue), color="tab:red", linestyle="--", linewidth=1.1)
            txt = f"n={int(rr.get('n_local', 0))}\nVar={float(rr.get('bin_count_var', np.nan)):.1f}\nempty={int(rr.get('empty_bin_count', 0))}"
            ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=6.5)
            ax.text(0.5, 0.97, f"offset={int(s)} m", transform=ax.transAxes, ha="center", va="top", fontsize=7)
            ax.grid(alpha=0.25)
            ax.tick_params(labelsize=5, direction="in")
            if row < (n_rows - 1):
                ax.tick_params(labelbottom=False, direction="in")
            if col > 0:
                ax.tick_params(labelleft=False, direction="in")
        for j in range(len(off_list), len(axes)):
            axes[j].axis("off")
        fig.suptitle("All offsets: c±100m bin count coverage", fontsize=16, y=0.992)
        fig.subplots_adjust(left=0.015, right=0.998, bottom=0.015, top=0.972, hspace=0.08, wspace=0.05)
        return fig

    def _build_metric_pair_figure(self, df: pd.DataFrame, x1: str, t1: str, x2: str, t2: str, c1: str, c2: str) -> Figure:
        fig = Figure(figsize=(14, 6), tight_layout=False)
        ax1, ax2 = fig.subplots(1, 2)
        has_df = isinstance(df, pd.DataFrame) and (not df.empty) and ("b" in df.columns)
        p1 = df.dropna(subset=[x1, "b"]).copy() if has_df and x1 in df.columns else pd.DataFrame()
        p2 = df.dropna(subset=[x2, "b"]).copy() if has_df and x2 in df.columns else pd.DataFrame()
        if not p1.empty:
            ax1.scatter(p1[x1], p1["b"], color=c1, s=30)
            p1_lab = p1.iloc[self._pick_sparse_label_indices(len(p1), max_labels=18)]
            for _, rr in p1_lab.iterrows():
                ax1.annotate(f"s={int(rr['offset'])}", (rr[x1], rr["b"]), textcoords="offset points", xytext=(3, 3), fontsize=7)
        else:
            ax1.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax1.transAxes)
        if not p2.empty:
            ax2.scatter(p2[x2], p2["b"], color=c2, s=30)
            p2_lab = p2.iloc[self._pick_sparse_label_indices(len(p2), max_labels=18)]
            for _, rr in p2_lab.iterrows():
                ax2.annotate(f"s={int(rr['offset'])}", (rr[x2], rr["b"]), textcoords="offset points", xytext=(3, 3), fontsize=7)
        else:
            ax2.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax2.transAxes)
        ax1.set_xlabel(x1)
        ax1.set_ylabel("Fitted sigmoid parameter b")
        ax1.set_title(t1)
        ax1.grid(alpha=0.3)
        ax2.set_xlabel(x2)
        ax2.set_ylabel("Fitted sigmoid parameter b")
        ax2.set_title(t2)
        ax2.grid(alpha=0.3)
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.98], pad=0.7)
        return fig

    def _build_aggregation_figure(self, method_a_df: pd.DataFrame, method_b_df: pd.DataFrame, summary: Dict, fig: Optional[Figure] = None) -> Figure:
        if fig is None:
            fig = Figure(figsize=(16, 7), tight_layout=False)
        else:
            fig.clear()
        ax1, ax2 = fig.subplots(1, 2)
        cons_a = float(summary.get("rank_top10_mean_b", np.nan))
        cons_b = float(summary.get("score_top10_mean_b", np.nan))
        head_a = float(summary.get("rank_head_mean_b", np.nan))
        head_b = float(summary.get("score_head_mean_b", np.nan))
        if isinstance(method_a_df, pd.DataFrame) and not method_a_df.empty and "final_rank" in method_a_df.columns:
            ax1.scatter(method_a_df["final_rank"], method_a_df["b"], s=30, color="tab:blue")
            m1_lab = method_a_df.iloc[self._pick_sparse_label_indices(len(method_a_df), max_labels=20)]
            for _, rr in m1_lab.iterrows():
                ax1.annotate(f"s={int(rr['offset'])}", (rr["final_rank"], rr["b"]), textcoords="offset points", xytext=(3, 3), fontsize=7)
            if np.isfinite(cons_a):
                ax1.axhline(cons_a, color="tab:red", linestyle="--", linewidth=1.4, label=f"Weighted-median consensus b={cons_a:.4g}")
            if np.isfinite(head_a):
                ax1.axhline(head_a, color="tab:green", linestyle=":", linewidth=1.4, label=f"Head (median) b={head_a:.4g}")
            ax1.legend(loc="best", fontsize=8)
        else:
            ax1.text(0.5, 0.5, "No aggregation data", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_xlabel("Constraint rank (1/w, smaller = better)")
        ax1.set_ylabel("Fitted sigmoid parameter b")
        ax1.set_title("Method A: constraint-weighted consensus")
        ax1.grid(alpha=0.3)

        if isinstance(method_b_df, pd.DataFrame) and not method_b_df.empty and "score_mean" in method_b_df.columns:
            ax2.scatter(method_b_df["score_mean"], method_b_df["b"], s=30, color="tab:orange")
            m2_lab = method_b_df.iloc[self._pick_sparse_label_indices(len(method_b_df), max_labels=20)]
            for _, rr in m2_lab.iterrows():
                ax2.annotate(f"s={int(rr['offset'])}", (rr["score_mean"], rr["b"]), textcoords="offset points", xytext=(3, 3), fontsize=7)
            if np.isfinite(cons_b):
                ax2.axhline(cons_b, color="tab:red", linestyle="--", linewidth=1.4, label=f"Weighted-median consensus b={cons_b:.4g}")
            if np.isfinite(head_b):
                ax2.axhline(head_b, color="tab:green", linestyle=":", linewidth=1.4, label=f"Head (median) b={head_b:.4g}")
            ax2.legend(loc="best", fontsize=8)
        else:
            ax2.text(0.5, 0.5, "No aggregation data", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_xlabel("Quality score q (larger = better)")
        ax2.set_ylabel("Fitted sigmoid parameter b")
        ax2.set_title("Method B: quality-score consensus")
        ax2.grid(alpha=0.3)
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.98], pad=0.7)
        return fig

    @staticmethod
    def _symmetric_trimmed_mean(values, trim_percent: float) -> float:
        v = np.asarray(values, dtype=float).reshape(-1)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return np.nan
        p = float(np.clip(trim_percent, 0.0, 49.9)) / 100.0
        n = int(v.size)
        k = int(np.floor(p * n))
        if (2 * k) >= n:
            k = max(0, (n - 1) // 2)
        v = np.sort(v)
        core = v[k : n - k] if k > 0 else v
        if core.size == 0:
            return np.nan
        return float(np.mean(core))

    def _build_top_percent_sweep_df(self, method_a_df: pd.DataFrame, method_b_df: pd.DataFrame) -> pd.DataFrame:
        percents = np.arange(5.0, 40.01, 5.0)

        def _sweep_trim_b(df: pd.DataFrame) -> list[float]:
            if not isinstance(df, pd.DataFrame) or df.empty or "b" not in df.columns:
                return [np.nan] * len(percents)
            bvals = pd.to_numeric(df["b"], errors="coerce").to_numpy(dtype=float)
            return [self._symmetric_trimmed_mean(bvals, float(pct)) for pct in percents]

        return pd.DataFrame(
            {
                "method_a_trim_percent": percents,
                "method_a_trimmed_mean_b": _sweep_trim_b(method_a_df),
                "method_b_trim_percent": percents,
                "method_b_trimmed_mean_b": _sweep_trim_b(method_b_df),
            }
        )

    def _build_top_percent_sweep_figure(
        self, method_a_df: pd.DataFrame, method_b_df: pd.DataFrame, summary: Dict, fig: Optional[Figure] = None
    ) -> Figure:
        if fig is None:
            fig = Figure(figsize=(16, 7), tight_layout=False)
        else:
            fig.clear()
        ax1, ax2 = fig.subplots(1, 2)
        sweep_df = self._build_top_percent_sweep_df(method_a_df, method_b_df)
        cons_a = float(summary.get("rank_top10_mean_b", np.nan))
        cons_b = float(summary.get("score_top10_mean_b", np.nan))

        if isinstance(method_a_df, pd.DataFrame) and not method_a_df.empty and "b" in method_a_df.columns:
            ax1.plot(
                sweep_df["method_a_trim_percent"],
                sweep_df["method_a_trimmed_mean_b"],
                color="tab:blue",
                marker="o",
                linewidth=1.6,
                label="Trimmed-mean b",
            )
            if np.isfinite(cons_a):
                ax1.axhline(cons_a, color="tab:red", linestyle="--", linewidth=1.4, label=f"Consensus b={cons_a:.4g}")
            ax1.legend(loc="best", fontsize=8)
        else:
            ax1.text(0.5, 0.5, "No aggregation data", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_xlabel("Trim percent (%)")
        ax1.set_ylabel("Trimmed-mean b")
        ax1.set_title("Method A: trimmed-mean sweep")
        ax1.set_xlim(5, 40)
        ax1.grid(alpha=0.3)

        if isinstance(method_b_df, pd.DataFrame) and not method_b_df.empty and "b" in method_b_df.columns:
            ax2.plot(
                sweep_df["method_b_trim_percent"],
                sweep_df["method_b_trimmed_mean_b"],
                color="tab:orange",
                marker="o",
                linewidth=1.6,
                label="Trimmed-mean b",
            )
            if np.isfinite(cons_b):
                ax2.axhline(cons_b, color="tab:red", linestyle="--", linewidth=1.4, label=f"Consensus b={cons_b:.4g}")
            ax2.legend(loc="best", fontsize=8)
        else:
            ax2.text(0.5, 0.5, "No aggregation data", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_xlabel("Trim percent (%)")
        ax2.set_ylabel("Trimmed-mean b")
        ax2.set_title("Method B: trimmed-mean sweep")
        ax2.set_xlim(5, 40)
        ax2.grid(alpha=0.3)
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.98], pad=0.7)
        return fig

    def _export_b_enhancement_results(self, enh: Dict, figures: Dict[str, Figure]):
        out_dir = QFileDialog.getExistingDirectory(self, "Select folder for b enhancement export", str(Path(self.config.output_dir).resolve()))
        if not out_dir:
            return
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        prefix = f"b_enhancement_epoch_{int(enh.get('epoch_idx', 0)) + 1}"
        tables = {
            "fit_df": enh.get("fit_df"),
            "coverage_residual_df": enh.get("coverage_residual_df"),
            "balance_evenness_df": enh.get("balance_evenness_df"),
            "merged_df": enh.get("merged_df"),
            "method_a_df": enh.get("method_a_df"),
            "method_b_df": enh.get("method_b_df"),
        }
        try:
            for name, table in tables.items():
                if isinstance(table, pd.DataFrame):
                    csv_fp = out_path / f"{prefix}_{name}.csv"
                    npy_fp = out_path / f"{prefix}_{name}.npy"
                    table.to_csv(csv_fp, index=False)
                    np.save(npy_fp, table.to_dict(orient="list"), allow_pickle=True)
            sweep_df = self._build_top_percent_sweep_df(
                enh.get("method_a_df") if isinstance(enh.get("method_a_df"), pd.DataFrame) else pd.DataFrame(),
                enh.get("method_b_df") if isinstance(enh.get("method_b_df"), pd.DataFrame) else pd.DataFrame(),
            )
            sweep_df.to_csv(out_path / f"{prefix}_trim_sweep.csv", index=False)
            summary_fp = out_path / f"{prefix}_summary.json"
            with open(summary_fp, "w", encoding="utf-8") as f:
                json.dump(enh.get("summary", {}), f, indent=2, ensure_ascii=False)
            for name, fig in figures.items():
                fig.savefig(out_path / f"{prefix}_{name}.png", dpi=180, bbox_inches="tight")
            self.append_log(f"[Sigmoid b Enhancement] exported to {out_path}")
            QMessageBox.information(self, "Export Completed", str(out_path))
        except Exception:
            self.append_log(traceback.format_exc())
            QMessageBox.critical(self, "Export Failed", "Failed to export b enhancement results.")

    def _swap_tab_figure(self, holder: Dict, fig: Figure):
        layout = holder.get("layout")
        parent = holder.get("parent")
        old_canvas = holder.get("canvas")
        old_toolbar = holder.get("toolbar")
        if layout is None or parent is None or old_canvas is None:
            return
        idx = layout.indexOf(old_toolbar) if old_toolbar is not None else layout.indexOf(old_canvas)
        if idx < 0:
            idx = 0
        if old_toolbar is not None:
            layout.removeWidget(old_toolbar)
            old_toolbar.hide()
            old_toolbar.setParent(None)
            old_toolbar.deleteLater()
        layout.removeWidget(old_canvas)
        old_canvas.hide()
        old_fig = getattr(old_canvas, "figure", None)
        old_canvas.setParent(None)
        old_canvas.deleteLater()
        if old_fig is not None and old_fig is not fig:
            old_fig.clf()
            old_fig.set_canvas(None)
        new_canvas = FigureCanvas(fig)
        new_toolbar = NavigationToolbar(new_canvas, parent)
        layout.insertWidget(idx, new_toolbar)
        layout.insertWidget(idx + 1, new_canvas, 1)
        holder["canvas"] = new_canvas
        holder["toolbar"] = new_toolbar
        new_canvas.draw()

    def _show_b_enhancement_results_dialog(self, enh: Dict):
        fit_df = enh.get("fit_df")
        coverage_df = enh.get("coverage_residual_df")
        merged_df = enh.get("merged_df")
        method_a_df = enh.get("method_a_df")
        method_b_df = enh.get("method_b_df")
        summary = enh.get("summary", {}) if isinstance(enh.get("summary", {}), dict) else {}
        metrics_df = fit_df if isinstance(fit_df, pd.DataFrame) else pd.DataFrame()
        if metrics_df.empty and isinstance(coverage_df, pd.DataFrame):
            metrics_df = coverage_df

        dlg = QDialog(self)
        dlg.setWindowTitle("Sigmoid b Enhancement Results")
        dlg.resize(1500, 950)
        lay = QVBoxLayout(dlg)

        rec_b = float(summary.get("b_recommended", np.nan))
        cur_b = float(summary.get("b_current_step2", np.nan))
        mean_b = self._mean_b_from_enhancement_fit_df(fit_df if isinstance(fit_df, pd.DataFrame) else pd.DataFrame())
        title_lbl = QLabel(
            f"Epoch {int(summary.get('epoch_index', enh.get('epoch_idx', 0))) + 1} | "
            f"current b={cur_b:.6g} | mean b={mean_b:.6g}"
        )
        lay.addWidget(title_lbl)

        tabs = QTabWidget(dlg)
        lay.addWidget(tabs, stretch=1)

        figures = {
            "map_overview": self._build_b_enhancement_map_overview_figure(enh),
            "b_vs_offset": self._build_b_vs_offset_figure(fit_df if isinstance(fit_df, pd.DataFrame) else pd.DataFrame()),
            "fit_windows": self._build_fit_matrix_figure(fit_df if isinstance(fit_df, pd.DataFrame) else pd.DataFrame()),
            "coverage_matrix": self._build_coverage_matrix_figure(coverage_df if isinstance(coverage_df, pd.DataFrame) else pd.DataFrame()),
            "metrics_var_empty": self._build_metric_pair_figure(
                metrics_df,
                "bin_count_var",
                "b vs bin-count variance",
                "empty_bin_count",
                "b vs empty-bin count",
                "tab:blue",
                "tab:orange",
            ),
            "metrics_rmse_vara": self._build_metric_pair_figure(
                metrics_df,
                "rmse_local",
                "b vs RMSE",
                "resid_var_over_a",
                "b vs variance/a",
                "tab:green",
                "tab:purple",
            ),
            "metrics_balance_pielou": self._build_metric_pair_figure(
                metrics_df,
                "s1_balance",
                "b vs balance score",
                "pielou_J",
                "b vs Pielou evenness",
                "tab:blue",
                "tab:orange",
            ),
            "metrics_transition": self._build_metric_pair_figure(
                metrics_df,
                "n_transition",
                "b vs n_transition",
                "rmse_transition",
                "b vs rmse_transition",
                "tab:blue",
                "tab:orange",
            ),
            "metrics_stability": self._build_metric_pair_figure(
                metrics_df,
                "rel_std_b",
                "b vs rel_std_b",
                "a_rel_dev",
                "b vs a_rel_dev",
                "tab:green",
                "tab:purple",
            ),
            "aggregation": self._build_aggregation_figure(
                method_a_df if isinstance(method_a_df, pd.DataFrame) else pd.DataFrame(),
                method_b_df if isinstance(method_b_df, pd.DataFrame) else pd.DataFrame(),
                summary,
            ),
            "trim_sweep": self._build_top_percent_sweep_figure(
                method_a_df if isinstance(method_a_df, pd.DataFrame) else pd.DataFrame(),
                method_b_df if isinstance(method_b_df, pd.DataFrame) else pd.DataFrame(),
                summary,
            ),
        }
        saved_weights = summary.get("aggregation_metric_weights", {}) if isinstance(summary.get("aggregation_metric_weights", {}), dict) else {}
        default_weights = default_aggregation_metric_weights()
        holders = {}
        # Public build: hide metrics/aggregation/trim_sweep tabs (figures still built for export).
        visible_result_tabs = ("map_overview", "b_vs_offset", "fit_windows", "coverage_matrix")
        for key, fig in figures.items():
            if key not in visible_result_tabs:
                continue
            if key == "aggregation":
                agg_tab = QWidget()
                agg_lay = QVBoxLayout(agg_tab)
                agg_canvas = FigureCanvas(fig)
                agg_toolbar = NavigationToolbar(agg_canvas, agg_tab)
                agg_lay.addWidget(agg_toolbar)
                agg_lay.addWidget(agg_canvas, stretch=1)
                metric_grid = QGridLayout()
                metric_editors = {}
                n_cols = 6
                for i, name in enumerate(AGGREGATION_METRIC_ORDER):
                    cell = QWidget()
                    cell_row = QHBoxLayout(cell)
                    cell_row.setContentsMargins(2, 2, 2, 2)
                    chk = QCheckBox(name)
                    chk.setChecked(name in saved_weights if saved_weights else True)
                    chk.setMinimumWidth(118)
                    spin = QDoubleSpinBox()
                    spin.setRange(0.0, 99.0)
                    spin.setDecimals(2)
                    spin.setSingleStep(0.1)
                    spin.setValue(float(saved_weights.get(name, default_weights.get(name, 1.0))))
                    spin.setFixedWidth(70)
                    cell_row.addWidget(chk)
                    cell_row.addWidget(spin)
                    r, c = divmod(i, n_cols)
                    metric_grid.addWidget(cell, r, c)
                    metric_editors[name] = (chk, spin)
                btn_run_agg = QPushButton("Run aggregation")
                run_row = QHBoxLayout()
                run_row.addStretch(1)
                run_row.addWidget(btn_run_agg)
                agg_lay.addLayout(metric_grid)
                agg_lay.addLayout(run_row)
                tabs.addTab(agg_tab, "aggregation")
                holders["aggregation"] = {
                    "layout": agg_lay,
                    "parent": agg_tab,
                    "canvas": agg_canvas,
                    "toolbar": agg_toolbar,
                }
            else:
                canvas = self._attach_figure_tab(tabs, key, fig)
                parent = canvas.parent()
                layout = parent.layout() if parent is not None else None
                toolbar = layout.itemAt(0).widget() if layout is not None and layout.count() > 0 else None
                holders[key] = {
                    "layout": layout,
                    "parent": parent,
                    "canvas": canvas,
                    "toolbar": toolbar,
                }

        row = QHBoxLayout()
        row.addStretch(1)
        epoch_idx = int(summary.get("epoch_index", enh.get("epoch_idx", 0)))

        def _set_epoch_for_apply(target_epoch: int):
            if not hasattr(self, "cmb_step1_epoch"):
                return
            for i in range(self.cmb_step1_epoch.count()):
                data = self.cmb_step1_epoch.itemData(i)
                if isinstance(data, (int, np.integer)) and int(data) == int(target_epoch):
                    self.cmb_step1_epoch.setCurrentIndex(i)
                    break

        def _apply_b_value_to_fit(b_value: float):
            if not np.isfinite(float(b_value)):
                QMessageBox.information(dlg, "Info", "Selected b value is not finite.")
                return
            if hasattr(self, "cmb_fit_dir"):
                self.cmb_fit_dir.setCurrentText("Fault-parallel")
            _set_epoch_for_apply(epoch_idx)
            if hasattr(self, "le_fit_b"):
                self.le_fit_b.setText(self._format_fit_param_display(b_value))
            self.on_apply_fit_override()

        def _update_title_label():
            cur_now = float(summary.get("b_current_step2", np.nan))
            mean_now = self._mean_b_from_enhancement_fit_df(
                fit_df if isinstance(fit_df, pd.DataFrame) else pd.DataFrame()
            )
            title_lbl.setText(
                f"Epoch {int(summary.get('epoch_index', enh.get('epoch_idx', 0))) + 1} | "
                f"current b={cur_now:.6g} | mean b={mean_now:.6g}"
            )

        def _read_metric_weights() -> dict:
            out = {}
            for name, (chk, spin) in metric_editors.items():
                if chk.isChecked() and float(spin.value()) > 0:
                    out[name] = float(spin.value())
            return out

        def _rerun_aggregation():
            weights = _read_metric_weights()
            if not weights:
                QMessageBox.information(dlg, "Info", "Select at least one metric with weight > 0.")
                return
            src = merged_df if isinstance(merged_df, pd.DataFrame) and not merged_df.empty else (
                fit_df if isinstance(fit_df, pd.DataFrame) else pd.DataFrame()
            )
            keep_keys = (
                "b_current_step2",
                "epoch_index",
                "offset_count",
                "success_window_count",
                "outlier_count",
                "b_at_lower_bound_count",
                "computed_at",
            )
            keep = {k: summary.get(k) for k in keep_keys if k in summary}
            params = enh.get("params", {}) if isinstance(enh.get("params", {}), dict) else {}
            a_df, b_df, new_sum = _run_dual_aggregation(
                src.copy() if isinstance(src, pd.DataFrame) else pd.DataFrame(),
                top_percent=float(summary.get("top_percent_for_aggregation", 10.0)),
                n_min_transition=int(params.get("n_min_transition", summary.get("n_min_transition", 8))),
                metric_weights=weights,
            )
            new_sum.update(keep)
            summary.clear()
            summary.update(new_sum)
            enh["method_a_df"] = a_df
            enh["method_b_df"] = b_df
            enh["summary"] = summary
            figures["aggregation"] = self._build_aggregation_figure(
                a_df if isinstance(a_df, pd.DataFrame) else pd.DataFrame(),
                b_df if isinstance(b_df, pd.DataFrame) else pd.DataFrame(),
                summary,
            )
            figures["trim_sweep"] = self._build_top_percent_sweep_figure(
                a_df if isinstance(a_df, pd.DataFrame) else pd.DataFrame(),
                b_df if isinstance(b_df, pd.DataFrame) else pd.DataFrame(),
                summary,
            )
            if "aggregation" in holders:
                self._swap_tab_figure(holders["aggregation"], figures["aggregation"])
            if "trim_sweep" in holders:
                self._swap_tab_figure(holders["trim_sweep"], figures["trim_sweep"])
            _update_title_label()

        # Public build: aggregation tab hidden, so Run aggregation is not connected.
        # btn_run_agg.clicked.connect(_rerun_aggregation)

        btn_mean = QPushButton("Apply mean b")
        btn_mean.clicked.connect(lambda: _apply_b_value_to_fit(mean_b))
        row.addWidget(btn_mean)
        # Public build: replace four apply-b buttons with Apply mean b.
        # btn_recommended = QPushButton("Apply Recommended b")
        # btn_recommended.clicked.connect(lambda: _apply_b_value_to_fit(float(summary.get("b_recommended", np.nan))))
        # row.addWidget(btn_recommended)
        # btn_rank_top = QPushButton("Apply Method A consensus b")
        # btn_rank_top.clicked.connect(lambda: _apply_b_value_to_fit(float(summary.get("rank_top10_mean_b", np.nan))))
        # row.addWidget(btn_rank_top)
        # btn_score_top = QPushButton("Apply Method B consensus b")
        # btn_score_top.clicked.connect(lambda: _apply_b_value_to_fit(float(summary.get("score_top10_mean_b", np.nan))))
        # row.addWidget(btn_score_top)
        # btn_head = QPushButton("Apply Head (median) b")
        # btn_head.clicked.connect(lambda: _apply_b_value_to_fit(float(summary.get("rank_head_mean_b", np.nan))))
        # row.addWidget(btn_head)
        btn_export = QPushButton("Export CSV/NPY/PNG")
        btn_export.clicked.connect(lambda: self._export_b_enhancement_results(enh, figures))
        row.addWidget(btn_export)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        row.addWidget(btn_close)
        lay.addLayout(row)
        dlg.exec()

    def _optional_float(self, line_edit: QLineEdit):
        txt = line_edit.text().strip()
        if not txt:
            return None
        try:
            return float(txt)
        except Exception:
            return None

    def _selected_epoch_index(self, result: Dict) -> int:
        matched = result.get("matched", {}) if isinstance(result, dict) else {}
        pairs = matched.get("matched_pairs", [])
        if not pairs:
            return -1
        idx = self.cmb_step1_epoch.currentData() if hasattr(self, "cmb_step1_epoch") else -1
        idx = int(idx) if isinstance(idx, (int, np.integer)) else -1
        return min(max(idx, 0), len(pairs) - 1)

    def _fit_payload_for_selected_mode(self, result: Dict, epoch_idx: int):
        if epoch_idx < 0 or "NEZ" not in result:
            return None
        signed_distance = np.asarray(result.get("signed_distance", []), dtype=float)
        fit_mask = np.asarray(result.get("fit_mask", []), dtype=bool)
        if signed_distance.size == 0 or fit_mask.size == 0 or fit_mask.sum() < 8:
            return None
        x_data = signed_distance[fit_mask]
        nez = np.asarray(result["NEZ"])
        if nez.ndim != 3 or epoch_idx >= nez.shape[0]:
            return None
        disp_xy = np.column_stack((nez[epoch_idx, :, 0], nez[epoch_idx, :, 1]))
        tangents = np.asarray(result.get("fault_tangents", []), dtype=float)
        normals = np.asarray(result.get("fault_normals", []), dtype=float)
        if tangents.shape != disp_xy.shape or normals.shape != disp_xy.shape:
            return None
        disp_swap = np.column_stack((disp_xy[:, 1], disp_xy[:, 0]))
        v_tangent = np.sum(disp_swap * tangents, axis=1)
        v_normal = np.sum(disp_swap * normals, axis=1)
        fit_key = self.cmb_fit_data.currentText() if hasattr(self, "cmb_fit_data") else "Fault-parallel/perpendicular"
        if fit_key == "Fault-parallel/perpendicular":
            series = [
                ("Fault-parallel", v_tangent, "Fault-parallel displacement (mm)"),
                ("Fault-perpendicular", v_normal, "Fault-perpendicular displacement (mm)"),
            ]
        else:
            series = [
                ("North", nez[epoch_idx, :, 0], "North displacement (mm)"),
                ("East", nez[epoch_idx, :, 1], "East displacement (mm)"),
            ]
        payloads = []
        for direction, y_all, ylabel in series:
            y_data = np.asarray(y_all, dtype=float)[fit_mask]
            epistemic_entry = None
            if fit_key == "Fault-parallel/perpendicular":
                p = self._get_fit_params_for_direction(result, epoch_idx, direction)
                if p is None:
                    continue
                yfit_fn = lambda x, params=np.asarray(p, dtype=float): sigmoid(x, *params[:4])
                params = np.asarray(p, dtype=float)
                if direction == "Fault-parallel":
                    epistemic_entry = result.get("epistemic_b_by_epoch", {}).get(epoch_idx)
            else:
                model_ne = result.get("models_by_epoch", {}).get(epoch_idx)
                if model_ne is None:
                    continue
                comp = "x_component" if direction == "North" else "y_component"
                params = np.asarray(model_ne[comp].params, dtype=float)
                yfit_fn = lambda x, pp=np.asarray(params, dtype=float): sigmoid(x, *pp[:4])
            payloads.append(
                {
                    "x_data": x_data,
                    "y_data": y_data,
                    "yfit_fn": yfit_fn,
                    "ylabel": ylabel,
                    "fit_key": fit_key,
                    "direction": direction,
                    "params": np.asarray(params, dtype=float),
                    "epistemic": epistemic_entry,
                }
            )
        return payloads if payloads else None

    def _fault_parallel_epistemic_fit_curves(self, xfit: np.ndarray, payload: Dict):
        epi = payload.get("epistemic")
        params = np.asarray(payload.get("params", []), dtype=float).reshape(-1)
        if payload.get("direction") != "Fault-parallel" or params.size < 4 or not isinstance(epi, dict):
            return None
        b16 = float(epi.get("b16", np.nan))
        b84 = float(epi.get("b84", np.nan))
        if not np.isfinite(b84):
            return None
        a = float(params[0])
        b50 = float(params[1])
        c = float(params[2])
        d = float(params[3])
        y50 = sigmoid(xfit, a, b50, c, d)
        y16 = sigmoid(xfit, a, b16, c, d) if np.isfinite(b16) else None
        y84 = sigmoid(xfit, a, b84, c, d)
        if y16 is None:
            lo = np.minimum(y50, y84)
            hi = np.maximum(y50, y84)
        else:
            lo = np.minimum(y16, y84)
            hi = np.maximum(y16, y84)
        return {
            "y16": np.asarray(y16, dtype=float) if y16 is not None else None,
            "y84": np.asarray(y84, dtype=float),
            "ylow": np.asarray(lo, dtype=float),
            "yhigh": np.asarray(hi, dtype=float),
            "one_sided": bool(y16 is None),
        }

    def _log_fit_sigmoid_info(self, result: Dict, epoch_idx: int, fit_payload: Dict):
        if epoch_idx < 0:
            return
        matched = result.get("matched", {})
        pairs = matched.get("matched_pairs", [])
        if epoch_idx >= len(pairs):
            return
        params = np.asarray(fit_payload.get("params", []), dtype=float).reshape(-1)
        if params.size < 4:
            return
        a, b, c, d = float(params[0]), float(params[1]), float(params[2]), float(params[3])
        if abs(b) < 1e-12:
            width = float("inf")
        else:
            x_1 = np.log(0.01 / 0.99) / b
            x_99 = np.log(0.99 / 0.01) / b
            width = abs(float(x_99 - x_1))
        asc_col, desc_col = pairs[epoch_idx]
        direction = fit_payload.get("direction", "Unknown")
        epi = fit_payload.get("epistemic")
        sigma_b = np.nan
        b16 = np.nan
        b84 = np.nan
        b16_disabled = False
        if isinstance(epi, dict):
            sigma_b = float(epi.get("sigma_b", np.nan))
            b16 = float(epi.get("b16", np.nan))
            b84 = float(epi.get("b84", np.nan))
            b16_disabled = bool(epi.get("b16_disabled", False))
        log_key = f"{epoch_idx}|{direction}|{a:.8g}|{b:.8g}|{c:.8g}|{d:.8g}|{sigma_b:.8g}|{b16:.8g}|{b84:.8g}"
        if log_key == self._last_fit_log_key:
            return
        self._last_fit_log_key = log_key
        msg = (
            f"[Fit] Date: {asc_col}/{desc_col}, Direction: {direction}, "
            f"sigmoid a={a:.3f}, b={b:.3f}, c={c:.3f}, d={d:.3f}, Width of shear band={width:.3f} m"
        )
        if direction == "Fault-parallel" and np.isfinite(sigma_b):
            msg += f", sigma_b={sigma_b:.3f}"
        if direction == "Fault-parallel" and np.isfinite(b16) and np.isfinite(b84):
            msg += f", b16={b16:.3f}, b84={b84:.3f}"
        elif direction == "Fault-parallel" and b16_disabled and np.isfinite(b84):
            msg += f", b16 disabled (floor), b84={b84:.3f}"
        self.append_log(msg)

    def _pipe_disp_payloads(self, result: Dict, epoch_idx: int):
        if epoch_idx < 0 or "pipe" not in result:
            return None
        mode = self.cmb_disp_coord.currentText() if hasattr(self, "cmb_disp_coord") else "Fault-parallel/perpendicular"
        pipe = self._get_pipe_prediction_for_mode(result, mode, need_strain=False)
        if pipe is None and isinstance(result.get("pipe"), dict):
            pipe = result.get("pipe")
            src_mode = str(pipe.get("prediction_coordinate_system", "unknown"))
            self._append_log_once(
                f"disp-fallback-{mode}-{src_mode}",
                f"[Step3 Render] Requested mode '{mode}' prediction unavailable. Fallback to available mode '{src_mode}'.",
            )
        if not isinstance(pipe, dict):
            return None
        xloc = np.asarray(pipe.get("s_nodes_offset", []), dtype=float)
        if xloc.size == 0:
            return None
        cases = self._get_epistemic_cases(result, pipe, mode)
        case16 = cases.get("b16") if isinstance(cases, dict) else None
        case50 = cases.get("b50") if isinstance(cases, dict) else None
        case84 = cases.get("b84") if isinstance(cases, dict) else None
        out = []
        active_mode = str(pipe.get("prediction_coordinate_system", mode))
        if active_mode == "N&E":
            motions = np.asarray(
                pipe.get("pipe_motion_xy_mm", []),
                dtype=float,
            )
            if motions.ndim != 3 or motions.shape[0] <= epoch_idx:
                return None
            east = np.nan_to_num(motions[epoch_idx, :, 0], nan=0.0, posinf=0.0, neginf=0.0)
            north = np.nan_to_num(motions[epoch_idx, :, 1], nan=0.0, posinf=0.0, neginf=0.0)
            north16 = north84 = east16 = east84 = None
            out.append(("North", north, north16, north84))
            out.append(("East", east, east16, east84))
            return out

        arrays = result.get("arrays", {}) if isinstance(result, dict) else {}
        motions_fault = np.asarray(
            arrays.get(
                "pipe_motion_fault_fault_b50_mm",
                arrays.get("pipe_motion_fault_b50_mm", (case50 or pipe).get("pipe_motion_fault_mm", [])),
            ),
            dtype=float,
        )
        if motions_fault.ndim != 3 or motions_fault.shape[0] <= epoch_idx:
            return None
        v_t = np.nan_to_num(motions_fault[epoch_idx, :, 0], nan=0.0, posinf=0.0, neginf=0.0)
        v_n = np.nan_to_num(motions_fault[epoch_idx, :, 1], nan=0.0, posinf=0.0, neginf=0.0)
        v_t16 = v_t84 = v_n16 = v_n84 = None
        epi_entry = result.get("epistemic_b_by_epoch", {}).get(epoch_idx, {})
        b16_epoch = float(epi_entry.get("b16", np.nan)) if isinstance(epi_entry, dict) else np.nan
        allow_fp16 = np.isfinite(b16_epoch) and (not bool(epi_entry.get("b16_disabled", False)))
        m16_src = arrays.get("pipe_motion_fault_fault_b16_mm", None)
        if m16_src is None:
            m16_src = arrays.get("pipe_motion_fault_b16_mm", None)
        if m16_src is None and isinstance(case16, dict):
            m16_src = case16.get("pipe_motion_fault_mm", [])
        m16 = np.asarray(m16_src if m16_src is not None else [], dtype=float)
        if allow_fp16 and m16.ndim == 3 and m16.shape[0] > epoch_idx:
            v_t16 = np.nan_to_num(m16[epoch_idx, :, 0], nan=0.0, posinf=0.0, neginf=0.0)
            v_n16 = np.nan_to_num(m16[epoch_idx, :, 1], nan=0.0, posinf=0.0, neginf=0.0)
        m84_src = arrays.get("pipe_motion_fault_fault_b84_mm", None)
        if m84_src is None:
            m84_src = arrays.get("pipe_motion_fault_b84_mm", None)
        if m84_src is None and isinstance(case84, dict):
            m84_src = case84.get("pipe_motion_fault_mm", [])
        m84 = np.asarray(m84_src if m84_src is not None else [], dtype=float)
        if m84.ndim == 3 and m84.shape[0] > epoch_idx:
            v_t84 = np.nan_to_num(m84[epoch_idx, :, 0], nan=0.0, posinf=0.0, neginf=0.0)
            v_n84 = np.nan_to_num(m84[epoch_idx, :, 1], nan=0.0, posinf=0.0, neginf=0.0)
        if allow_fp16 and isinstance(v_t16, np.ndarray) and v_t16.size == v_t.size:
            b50_epoch = float(epi_entry.get("b50", np.nan)) if isinstance(epi_entry, dict) else np.nan
            max_diff = float(np.nanmax(np.abs(v_t16 - v_t))) if v_t.size > 0 else np.nan
            if np.isfinite(max_diff) and max_diff < 1e-6 and np.isfinite(b50_epoch) and abs(b16_epoch - b50_epoch) > 1e-3:
                self._append_log_once(
                    f"fp-b16-overlap-{epoch_idx}",
                    (
                        "[Step3 Render] FP 16th curve overlaps b50 while b16!=b50 "
                        f"(epoch={epoch_idx}, b16={b16_epoch:.6g}, b50={b50_epoch:.6g}, "
                        f"max|d16-d50|={max_diff:.3e})."
                    ),
                )
        out.append(("Fault-parallel", v_t, v_t16, v_t84))
        out.append(("Fault-perpendicular", v_n, v_n16, v_n84))
        return out

    def _resample_series_to_target_len(self, arr: Optional[np.ndarray], target_len: int):
        if arr is None:
            return None
        vals = np.asarray(arr, dtype=float).reshape(-1)
        n = int(target_len)
        if n <= 0 or vals.size == 0:
            return None
        if vals.size == n:
            return vals
        x_src = np.linspace(0.0, 1.0, vals.size)
        x_tgt = np.linspace(0.0, 1.0, n)
        return np.interp(x_tgt, x_src, vals)

    def _reference_disp_xloc(self, result: Dict, mode_pipe: Dict, payloads):
        if not payloads:
            return np.asarray(mode_pipe.get("s_nodes_offset", []), dtype=float)
        mode = self.cmb_disp_coord.currentText() if hasattr(self, "cmb_disp_coord") else "Fault-parallel/perpendicular"
        cases = self._get_epistemic_cases(result, mode_pipe, mode)
        case50 = cases.get("b50") if isinstance(cases, dict) else None
        if isinstance(case50, dict):
            x_case = np.asarray(case50.get("s_nodes_offset", []), dtype=float)
            if x_case.size > 0:
                return x_case
        x_cur = np.asarray(mode_pipe.get("s_nodes_offset", []), dtype=float)
        if x_cur.size > 0:
            return x_cur
        n_ref = len(np.asarray(payloads[0][1], dtype=float).reshape(-1))
        return np.arange(max(n_ref, 0), dtype=float)

    def _is_valid_pipe_prediction(self, pipe: Dict, need_strain: bool) -> bool:
        if not isinstance(pipe, dict):
            return False
        xloc = np.asarray(pipe.get("s_nodes_offset", []), dtype=float).reshape(-1)
        if xloc.size == 0:
            return False
        motion_xy = np.asarray(pipe.get("pipe_motion_xy_mm", []), dtype=float)
        if motion_xy.ndim != 3 or motion_xy.shape[1] != xloc.size:
            return False
        motion_fault = np.asarray(pipe.get("pipe_motion_fault_mm", []), dtype=float)
        if motion_fault.ndim == 3 and motion_fault.shape[1] != xloc.size:
            return False
        if need_strain:
            axial = np.asarray(pipe.get("axial_strain_smoothed", []), dtype=float)
            if axial.ndim != 2 or axial.shape[1] != xloc.size:
                return False
        return True

    def _compute_peak_abs_with_location(self, xloc: np.ndarray, vals_micro: np.ndarray):
        x = np.asarray(xloc, dtype=float).reshape(-1)
        y = np.asarray(vals_micro, dtype=float).reshape(-1)
        if x.size == 0 or y.size == 0:
            return None
        n = min(x.size, y.size)
        x = x[:n]
        y = y[:n]
        mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            return None
        xv = x[mask]
        yv = y[mask]
        idx = int(np.argmax(np.abs(yv)))
        return {
            "x_m": float(xv[idx]),
            "abs_microstrain": float(np.abs(yv[idx])),
            "signed_microstrain": float(yv[idx]),
        }

    def _has_step4_outputs(self, result: Dict) -> bool:
        tables = result.get("tables", {}) if isinstance(result, dict) else {}
        if not isinstance(tables, dict):
            return False
        return "pipe_axial_latest" in tables and "pipe_bending_latest" in tables

    def _safe_element_values_from_nodes(self, node_vals: np.ndarray, conn: np.ndarray) -> np.ndarray:
        vals = np.asarray(node_vals, dtype=float).reshape(-1)
        c = np.asarray(conn, dtype=int)
        if vals.size == 0 or c.ndim != 2 or c.shape[1] != 2 or c.size == 0:
            return np.array([], dtype=float)
        out = np.zeros(c.shape[0], dtype=float)
        for i, (n0, n1) in enumerate(c):
            if 0 <= int(n0) < vals.size and 0 <= int(n1) < vals.size:
                out[i] = 0.5 * (float(vals[int(n0)]) + float(vals[int(n1)]))
        return out

    def _compute_element_axial_strain(self, nodes_xy: np.ndarray, deformed_xy: np.ndarray, conn: np.ndarray) -> np.ndarray:
        pts0 = np.asarray(nodes_xy, dtype=float)
        pts1 = np.asarray(deformed_xy, dtype=float)
        c = np.asarray(conn, dtype=int)
        if pts0.ndim != 2 or pts1.ndim != 2 or c.ndim != 2 or c.shape[1] != 2:
            return np.array([], dtype=float)
        out = np.zeros(c.shape[0], dtype=float)
        for i, (n0, n1) in enumerate(c):
            n0i = int(n0)
            n1i = int(n1)
            if not (0 <= n0i < pts0.shape[0] and 0 <= n1i < pts0.shape[0]):
                continue
            l0 = float(np.linalg.norm(pts0[n1i] - pts0[n0i]))
            l1 = float(np.linalg.norm(pts1[n1i] - pts1[n0i]))
            if l0 <= 1e-12:
                l0 = 1e-12
            out[i] = (l1 - l0) / l0
        return out

    def _compute_element_longitudinal_strain_rate(
        self, pipe: Dict, epoch_idx: int, years_elapsed: float
    ) -> tuple[np.ndarray, np.ndarray]:
        nodes = np.asarray(pipe.get("pipe_axis_nodes", []), dtype=float)
        conn = np.asarray(pipe.get("pipe_connectivity", []), dtype=int)
        if nodes.ndim != 2 or nodes.shape[0] < 2 or conn.ndim != 2 or conn.shape[1] != 2 or conn.shape[0] == 0:
            return np.array([], dtype=float), np.array([], dtype=float)

        axial_all = np.asarray(pipe.get("axial_strain_smoothed", []), dtype=float)
        bending_all = np.asarray(pipe.get("bending_strain", []), dtype=float)
        if axial_all.ndim != 2 or axial_all.shape[0] == 0:
            return np.array([], dtype=float), np.array([], dtype=float)
        i_sel = min(max(int(epoch_idx), 0), axial_all.shape[0] - 1)

        axial_nodes_micro = np.nan_to_num(axial_all[i_sel], nan=0.0, posinf=0.0, neginf=0.0) * 1e6
        if axial_nodes_micro.size != nodes.shape[0]:
            if axial_nodes_micro.size > 1 and nodes.shape[0] > 1:
                axial_nodes_micro = np.interp(
                    np.linspace(0, axial_nodes_micro.size - 1, nodes.shape[0]),
                    np.arange(axial_nodes_micro.size, dtype=float),
                    axial_nodes_micro,
                )
            else:
                axial_nodes_micro = np.zeros(nodes.shape[0], dtype=float)
        axial_elem_micro = self._safe_element_values_from_nodes(axial_nodes_micro, conn)

        bending_elem_micro = np.zeros(conn.shape[0], dtype=float)
        if bending_all.ndim == 2 and bending_all.shape[0] > i_sel:
            bending_vals_micro = np.nan_to_num(bending_all[i_sel], nan=0.0, posinf=0.0, neginf=0.0) * 1e6
            if bending_vals_micro.size == nodes.shape[0]:
                bending_elem_micro = self._safe_element_values_from_nodes(bending_vals_micro, conn)
            elif bending_vals_micro.size == conn.shape[0]:
                bending_elem_micro = bending_vals_micro
            else:
                sampled_xy = np.asarray(pipe.get("pipe_axis_sampled", []), dtype=float)
                if sampled_xy.ndim == 2 and sampled_xy.shape[0] == bending_vals_micro.size and sampled_xy.shape[1] >= 2:
                    mid_xy = 0.5 * (nodes[conn[:, 0], :2] + nodes[conn[:, 1], :2])
                    for j, mid in enumerate(mid_xy):
                        d2 = np.sum((sampled_xy[:, :2] - mid[:2]) ** 2, axis=1)
                        if d2.size:
                            bending_elem_micro[j] = float(bending_vals_micro[int(np.argmin(d2))])
                elif bending_vals_micro.size > 1:
                    bending_elem_micro = np.interp(
                        np.linspace(0, bending_vals_micro.size - 1, conn.shape[0]),
                        np.arange(bending_vals_micro.size, dtype=float),
                        bending_vals_micro,
                    )

        axial_sign = np.where(axial_elem_micro >= 0.0, 1.0, -1.0)
        longitudinal_micro = axial_elem_micro + axial_sign * np.abs(bending_elem_micro)
        years_safe = max(float(years_elapsed), 1e-12)
        longitudinal_pct_per_year = longitudinal_micro / (1e4 * years_safe)
        comp_lsr = np.maximum(-longitudinal_pct_per_year, 0.0)
        ext_lsr = np.maximum(longitudinal_pct_per_year, 0.0)
        return comp_lsr, ext_lsr

    def _apply_field_context_layers(self, ax, result: Dict, epoch_idx: int, basemap_key: Optional[str] = None):
        nez = np.asarray(result.get("NEZ", []), dtype=float)
        matched_coords = np.asarray(result.get("matched", {}).get("matched_coords", []), dtype=float)
        if nez.ndim == 3 and matched_coords.ndim == 2 and matched_coords.shape[0] == nez.shape[1]:
            i_sel = min(max(epoch_idx, 0), nez.shape[0] - 1)
            bg = np.asarray(nez[i_sel, :, 0], dtype=float)
            ax.scatter(
                matched_coords[:, 0],
                matched_coords[:, 1],
                c=bg,
                marker="o",
                cmap="coolwarm",
                s=18,
                alpha=0.22,
                label="Asc-Desc",
            )
        nb_ref = result.get("notebook_fault_ref", {})
        fault_selected = nb_ref.get("fault_line_selected")
        if isinstance(fault_selected, LineString):
            xs, ys = fault_selected.xy
            ax.plot(xs, ys, linewidth=2, c="r", linestyle="--", label="Fault (selected)")
        trace_type = str(nb_ref.get("fault_trace_type", "selected_geometry"))
        inter_source = str(nb_ref.get("intersection_trace_source", ""))
        fit_raw = nb_ref.get("fault_line_fitted_raw")
        fit_ext = nb_ref.get("fault_line_fitted_extended")
        if trace_type in {"linear_fit", "quadratic_fit"}:
            if isinstance(fit_ext, LineString):
                x_ext, y_ext = fit_ext.xy
                ax.plot(x_ext, y_ext, linewidth=2.2, c="r", linestyle="-", label="Fault (fitted extended)")
            elif not isinstance(fit_raw, LineString):
                fl2 = np.asarray(nb_ref.get("fl2", []), dtype=float)
                if fl2.ndim == 2 and fl2.shape[1] >= 2 and fl2.shape[0] > 1:
                    ax.plot(fl2[:, 0], fl2[:, 1], linewidth=2.0, c="r", linestyle="-", label="Fault (fitted)")
        else:
            fl2 = np.asarray(nb_ref.get("fl2", []), dtype=float)
            if fl2.ndim == 2 and fl2.shape[1] >= 2 and fl2.shape[0] > 1 and not isinstance(fault_selected, LineString):
                ax.plot(fl2[:, 0], fl2[:, 1], linewidth=2.0, c="C2", linestyle="--", label="Fault")
        fit_band = result.get("fit_band", {})
        if isinstance(fit_band, dict):
            transect = fit_band.get("transect")
            if transect is not None:
                tx, ty = transect.xy
                ax.plot(tx, ty, "C1-", linewidth=0.7, label="Perpendicular to fault line")
            offset_lines = fit_band.get("offset_lines", ())
            if len(offset_lines) == 2:
                l1, l2 = offset_lines
                x1, y1 = l1.xy
                x2, y2 = l2.xy
                ax.plot(x1, y1, "k--", linewidth=0.5, label="Upper Offset")
                ax.plot(x2, y2, "k--", linewidth=0.5, label="Lower Offset")
            cap_lines = fit_band.get("cap_lines", ())
            if len(cap_lines) == 2:
                c1, c2 = cap_lines
                cx1, cy1 = c1.xy
                cx2, cy2 = c2.xy
                ax.plot(cx1, cy1, "k:", linewidth=0.8, label="Transect caps")
                ax.plot(cx2, cy2, "k:", linewidth=0.8)
            inter = fit_band.get("intersection")
            if inter is not None:
                intersect = np.asarray(inter, dtype=float).reshape(-1)
                if intersect.size >= 2:
                    mg = 150.0
                    mg2 = 120.0
                    ax.set_xlim(float(intersect[0]) - mg - 20.0, float(intersect[0]) + mg - 20.0)
                    ax.set_ylim(float(intersect[1]) - mg2 - 20.0, float(intersect[1]) + mg2 + 20.0)
        self._add_basemap(ax, alpha=0.3, key=basemap_key)

    def _get_pipe_prediction_for_mode(self, result: Dict, mode: str, need_strain: bool):
        if not isinstance(result, dict):
            return None
        base_pipe = result.get("pipe")
        if isinstance(base_pipe, dict):
            has_strain = "axial_strain_smoothed" in base_pipe and "bending_strain" in base_pipe
            if base_pipe.get("prediction_coordinate_system") == mode and (has_strain or not need_strain):
                return base_pipe
        if need_strain and not self._has_step4_outputs(result):
            return None

        cache = result.setdefault("_pipe_mode_cache", {})
        key = f"{mode}|{'strain' if need_strain else 'disp'}"
        if key in cache:
            cached = cache.get(key)
            if self._is_valid_pipe_prediction(cached, need_strain=need_strain):
                return cached
            cache.pop(key, None)

        if need_strain:
            by_mode = result.get("pipe_epistemic_cases_by_mode", {})
            mode_cases = by_mode.get(mode, {}) if isinstance(by_mode, dict) else {}
            mid = mode_cases.get("b50") if isinstance(mode_cases, dict) else None
            if isinstance(mid, dict) and np.asarray(mid.get("axial_strain_smoothed", [])).ndim == 2:
                pred = copy.deepcopy(mid)
                ref_pipe = result.get("pipe", {}) if isinstance(result.get("pipe", {}), dict) else {}
                if "s_nodes_offset" not in pred and "s_nodes_offset" in ref_pipe:
                    pred["s_nodes_offset"] = ref_pipe["s_nodes_offset"]
                pred["prediction_coordinate_system"] = mode
                pred["epistemic_cases"] = mode_cases if isinstance(mode_cases, dict) else {}
                pred["epistemic_cases_by_mode"] = by_mode if isinstance(by_mode, dict) else {}
                if self._is_valid_pipe_prediction(pred, need_strain=True):
                    cache[key] = pred
                    return pred
            return None

        required = ["data", "notebook_fault_ref", "models_by_epoch", "NEZ", "fit_band"]
        if any(k not in result for k in required):
            if isinstance(base_pipe, dict) and not need_strain:
                return base_pipe
            return None
        cfg = copy.deepcopy(self.config)
        cfg.model_type = str(result.get("model_type", getattr(self.config, "model_type", "sigmoid")))
        cfg.prediction_coordinate_system = mode
        try:
            pred = predict_pipe_motion_and_strain(
                config=cfg,
                pipe_axis=result["data"]["pipe_axis"],
                pipe_axis_segments=result.get("data", {}).get("pipe_axis_segments"),
                fault_line=result["notebook_fault_ref"]["fault_line_truth"],
                left_area=result["notebook_fault_ref"]["left_area"],
                fault_polyline_for_projection=np.asarray(result["notebook_fault_ref"]["fl2"]),
                models_by_epoch=result["models_by_epoch"],
                tangential_fit_by_epoch=result.get("tangential_fit_by_epoch", {}),
                normal_fit_by_epoch=result.get("normal_fit_by_epoch", {}),
                fit_overrides_by_epoch=result.get("fault_fit_override_by_epoch", {}),
                pair_count=np.asarray(result["NEZ"]).shape[0],
                intersection_point=result["fit_band"]["intersection"],
                compute_strain=need_strain,
            )
        except Exception as exc:
            if isinstance(base_pipe, dict) and not need_strain:
                self._append_log_once(
                    f"mode-predict-fallback-{mode}",
                    f"[Step3 Render] Failed to rebuild mode '{mode}' prediction: {exc}. Using Step3 base output.",
                )
                return base_pipe
            raise
        if self._is_valid_pipe_prediction(pred, need_strain=need_strain):
            cache[key] = pred
            return pred
        if isinstance(base_pipe, dict) and not need_strain:
            self._append_log_once(
                f"mode-predict-shape-fallback-{mode}",
                f"[Step3 Render] Mode '{mode}' prediction shape mismatch; fallback to base output.",
            )
            return base_pipe
        return pred

    def _get_epistemic_cases(self, result: Dict, pipe: Dict, mode: Optional[str] = None):
        target_mode = mode or (self.cmb_disp_coord.currentText() if hasattr(self, "cmb_disp_coord") else None)
        if isinstance(pipe, dict):
            by_mode = pipe.get("epistemic_cases_by_mode", {})
            if isinstance(by_mode, dict) and target_mode in by_mode and isinstance(by_mode[target_mode], dict):
                return by_mode[target_mode]
            cases = pipe.get("epistemic_cases", {})
            if isinstance(cases, dict) and cases:
                return cases
        if isinstance(result, dict):
            top_by_mode = result.get("pipe_epistemic_cases_by_mode", {})
            if isinstance(top_by_mode, dict) and target_mode in top_by_mode and isinstance(top_by_mode[target_mode], dict):
                return top_by_mode[target_mode]
        top = result.get("pipe_epistemic_cases", {}) if isinstance(result, dict) else {}
        return top if isinstance(top, dict) else {}

    def _as_lines_list(self, geom):
        if geom is None:
            return []
        if isinstance(geom, LineString):
            return [geom]
        if isinstance(geom, MultiLineString):
            return list(geom.geoms)
        if isinstance(geom, GeometryCollection):
            out = []
            for g in geom.geoms:
                out.extend(self._as_lines_list(g))
            return out
        return []

    def _sample_line_with_mesh_size(self, line: LineString, mesh_size: float):
        coords = np.asarray(line.coords, dtype=float)
        if coords.ndim != 2 or coords.shape[0] < 2:
            return np.empty((0, 2), dtype=float), np.empty((0, 2), dtype=int)
        seg_vecs = np.diff(coords, axis=0)
        seg_lengths = np.linalg.norm(seg_vecs, axis=1)
        cum_lengths = np.insert(np.cumsum(seg_lengths), 0, 0.0)
        total_len = float(cum_lengths[-1])
        if total_len <= 0:
            return np.empty((0, 2), dtype=float), np.empty((0, 2), dtype=int)
        spacing = max(float(mesh_size), 0.1)
        n_steps = max(2, int(np.ceil(total_len / spacing)) + 1)
        target_d = np.linspace(0.0, total_len, n_steps)
        sampled = []
        for td in target_d:
            si = int(np.clip(np.searchsorted(cum_lengths, td) - 1, 0, len(seg_lengths) - 1))
            seg_len = float(seg_lengths[si])
            t = (td - float(cum_lengths[si])) / seg_len if seg_len > 0 else 0.0
            p = (1.0 - t) * coords[si] + t * coords[si + 1]
            sampled.append(p[:2])
        sampled = np.asarray(sampled, dtype=float)
        conn = np.column_stack([np.arange(0, len(sampled) - 1), np.arange(1, len(sampled))]).astype(int)
        return sampled, conn

    def _build_pipeline_mesh(self, result: Dict):
        mesh_size = max(float(getattr(self.config, "mesh_size_m", 8.0)), 0.1)
        pipe_pred = np.asarray(result.get("pipe", {}).get("pipe_axis_nodes", []), dtype=float)
        pipe_pred_conn = np.asarray(result.get("pipe", {}).get("pipe_connectivity", []), dtype=int)
        pipe_pred_segs = result.get("pipe", {}).get("pipe_axis_segments", [])
        if (
            pipe_pred.ndim == 2
            and pipe_pred.shape[0] >= 2
            and pipe_pred.shape[1] >= 2
            and pipe_pred_conn.ndim == 2
            and pipe_pred_conn.shape[1] == 2
            and pipe_pred_conn.shape[0] > 0
        ):
            out = {
                "points": pipe_pred[:, :2].astype(float),
                "connectivity": pipe_pred_conn.astype(int),
                "lines": [LineString(np.asarray(seg, dtype=float)[:, :2]) for seg in pipe_pred_segs if np.asarray(seg).ndim == 2],
            }
            return out
        if pipe_pred.ndim == 2 and pipe_pred.shape[0] >= 2 and pipe_pred.shape[1] >= 2:
            head = pipe_pred[0, :2]
            tail = pipe_pred[-1, :2]
            cache_key = (
                f"step3_axis|{mesh_size:.6f}|{pipe_pred.shape[0]}|"
                f"{head[0]:.3f}|{head[1]:.3f}|{tail[0]:.3f}|{tail[1]:.3f}"
            )
        else:
            cache_key = f"{self.config.pipe_axis_path}|{mesh_size:.6f}|{self._processing_crs_text()}"
        cached = self._mesh_cache.get(cache_key)
        if isinstance(cached, dict):
            return cached

        lines = []
        if pipe_pred.ndim == 2 and pipe_pred.shape[0] >= 2 and pipe_pred.shape[1] >= 2:
            lines = [LineString(pipe_pred[:, :2])]
        else:
            path = self.config.resolve_path(self.config.pipe_axis_path)
            if path.suffix.lower() == ".shp" and path.exists():
                gdf = gpd.read_file(path)
                gdf = gdf[gdf.geometry.notnull()].copy()
                try:
                    if gdf.crs is not None:
                        gdf = gdf.to_crs(self._processing_crs_text())
                except Exception:
                    pass
                if hasattr(gdf, "explode"):
                    gdf = gdf.explode(index_parts=False, ignore_index=True)
                raw_lines = []
                for geom in gdf.geometry.values:
                    raw_lines.extend(self._as_lines_list(geom))
                if raw_lines:
                    # Build connected components for complex networks.
                    tree = STRtree(raw_lines)
                    parent = list(range(len(raw_lines)))

                    def find(a):
                        while parent[a] != a:
                            parent[a] = parent[parent[a]]
                            a = parent[a]
                        return a

                    def union(a, b):
                        ra, rb = find(a), find(b)
                        if ra != rb:
                            parent[rb] = ra

                    for i, gi in enumerate(raw_lines):
                        for j_raw in tree.query(gi):
                            j = int(j_raw)
                            if j <= i:
                                continue
                            gj = raw_lines[j]
                            try:
                                if gi.intersects(gj) or gi.touches(gj):
                                    union(i, j)
                            except Exception:
                                continue

                    components = {}
                    for i in range(len(raw_lines)):
                        r = find(i)
                        components.setdefault(r, []).append(i)
                    for _, idxs in components.items():
                        comp_geom = unary_union([raw_lines[ii] for ii in idxs])
                        merged = linemerge(comp_geom)
                        lines.extend(self._as_lines_list(merged))
        if not lines:
            pipe_axis = np.asarray(result.get("data", {}).get("pipe_axis", []), dtype=float)
            if pipe_axis.ndim == 2 and pipe_axis.shape[0] >= 2:
                lines = [LineString(pipe_axis[:, :2])]

        all_points = []
        all_connectivity = []
        point_offset = 0
        for ln in lines:
            pts, conn = self._sample_line_with_mesh_size(ln, mesh_size)
            if pts.size == 0:
                continue
            conn_shift = conn + point_offset if conn.size > 0 else conn
            all_points.append(pts)
            if conn_shift.size > 0:
                all_connectivity.append(conn_shift)
            point_offset += int(pts.shape[0])

        if all_points:
            points = np.vstack(all_points).astype(float)
            connectivity = np.vstack(all_connectivity).astype(int) if all_connectivity else np.empty((0, 2), dtype=int)
        else:
            points = np.empty((0, 2), dtype=float)
            connectivity = np.empty((0, 2), dtype=int)
        out = {"points": points, "connectivity": connectivity, "lines": lines}
        self._mesh_cache[cache_key] = out
        return out

    def _update_step1_epoch_selector(self, result: Dict):
        matched = result.get("matched", {}) if isinstance(result, dict) else {}
        pairs = matched.get("matched_pairs", [])
        prev_data = self.cmb_step1_epoch.currentData() if hasattr(self, "cmb_step1_epoch") else None
        if not hasattr(self, "cmb_step1_epoch"):
            return

        self.cmb_step1_epoch.blockSignals(True)
        self.cmb_step1_epoch.clear()
        if pairs:
            for i, (asc_col, desc_col) in enumerate(pairs):
                self.cmb_step1_epoch.addItem(f"Date {i + 1}: {asc_col} | {desc_col}", i)
            idx = self.cmb_step1_epoch.findData(prev_data)
            if idx < 0:
                idx = len(pairs) - 1
            self.cmb_step1_epoch.setCurrentIndex(idx)
            self.cmb_step1_epoch.setEnabled(True)
        else:
            self.cmb_step1_epoch.addItem("Run Step 1 to load epochs", -1)
            self.cmb_step1_epoch.setEnabled(False)
        self.cmb_step1_epoch.blockSignals(False)

    def render_plots(self, result: Dict):
        if not result:
            return
        data = result["data"]
        matched = result["matched"]

        # Tab1 map
        fig_map = self.tab_map.figure
        fig_map.clear()
        gs_map = fig_map.add_gridspec(
            1,
            3,
            width_ratios=[1.0, 0.050, 0.48],
            left=0.12,
            right=0.995,
            bottom=0.10,
            top=0.90,
            wspace=0.18,
        )
        ax = fig_map.add_subplot(gs_map[0, 0])
        cax = fig_map.add_subplot(gs_map[0, 1])
        lax = fig_map.add_subplot(gs_map[0, 2])
        self.tab_map.ax = ax
        self.map_colorbar = None
        asc = data["asc_data"]
        desc = data["desc_data"]
        if getattr(asc, "crs", None) is not None:
            try:
                epsg = self._processing_epsg()
                asc = asc.to_crs(epsg=epsg)
                desc = desc.to_crs(epsg=epsg)
            except Exception:
                pass
        matched_pairs = matched.get("matched_pairs", [])
        epoch_idx = self._selected_epoch_index(result)

        day_label = "N/A"
        if epoch_idx >= 0 and "NEZ" in result:
            asc_col, desc_col = matched_pairs[epoch_idx]
            if "day_diffs" in matched and len(matched["day_diffs"]) > epoch_idx:
                day_label = f"{int(matched['day_diffs'][epoch_idx]):+d}"
            nez = np.asarray(result["NEZ"])
            comp = self.cmb_step1_nez.currentText() if hasattr(self, "cmb_step1_nez") else "E"
            comp_map = {"N": 0, "E": 1, "Z": 2}
            comp_idx = comp_map.get(comp, 1)
            vals = np.asarray(nez[epoch_idx, :, comp_idx], dtype=float)
            vmin = self._optional_float(self.le_step1_vmin) if hasattr(self, "le_step1_vmin") else None
            vmax = self._optional_float(self.le_step1_vmax) if hasattr(self, "le_step1_vmax") else None
            if vmin is not None and vmax is not None and vmin >= vmax:
                vmin, vmax = None, None
            sc = ax.scatter(
                matched["matched_coords"][:, 0],
                matched["matched_coords"][:, 1],
                s=12,
                c=vals,
                cmap="coolwarm_r",
                vmin=vmin,
                vmax=vmax,
                alpha=0.9,
                label="InSAR data",
            )
            fit_mask = np.asarray(result.get("fit_mask", []), dtype=bool).reshape(-1)
            fit_pts = np.asarray(matched.get("matched_coords", []), dtype=float)
            n_fit = int(min(fit_mask.size, fit_pts.shape[0])) if fit_pts.ndim == 2 else 0
            if n_fit > 0:
                sel = fit_mask[:n_fit]
                if np.any(sel):
                    fit_sel_pts = fit_pts[:n_fit][sel]
                    ax.scatter(
                        fit_sel_pts[:, 0],
                        fit_sel_pts[:, 1],
                        s=18,
                        facecolors="none",
                        edgecolors="black",
                        linewidths=0.0,
                        alpha=0.8,
                        
                    )
            self.map_colorbar = self.tab_map.figure.colorbar(sc, cax=cax)
            self.map_colorbar.set_label(f"{comp} displacement (mm)")
        else:
            ax.scatter(asc.geometry.x, asc.geometry.y, s=8, c="tab:blue", alpha=0.4, label="Ascending points")
            ax.scatter(desc.geometry.x, desc.geometry.y, s=8, c="tab:orange", alpha=0.4, label="Descending points")
            cax.axis("off")
        pipe_for_map = np.asarray(data.get("pipe_axis", []), dtype=float)
        if pipe_for_map.ndim == 2 and pipe_for_map.shape[1] >= 2 and pipe_for_map.shape[0] >= 2:
            ax.plot(pipe_for_map[:, 0], pipe_for_map[:, 1], "k-", lw=2, label="Pipe")
        nb_ref = result.get("notebook_fault_ref", {})
        fault_selected = nb_ref.get("fault_line_selected")
        if isinstance(fault_selected, LineString):
            sx, sy = fault_selected.xy
            ax.plot(sx, sy, color="C2", linestyle="--", lw=1.2, label="Fault (selected)")
        trace_type = str(nb_ref.get("fault_trace_type", "selected_geometry"))
        inter_source = str(nb_ref.get("intersection_trace_source", ""))
        fit_raw = nb_ref.get("fault_line_fitted_raw")
        fit_ext = nb_ref.get("fault_line_fitted_extended")
        if trace_type in {"linear_fit", "quadratic_fit"}:
            if isinstance(fit_raw, LineString):
                xr, yr = fit_raw.xy
                ax.plot(xr, yr, "r--", lw=1.4, label="_nolegend_")
            if inter_source == "extended_fit" and isinstance(fit_ext, LineString):
                xe, ye = fit_ext.xy
                ax.plot(xe, ye, "r-", lw=1.9, label="Fault (fitted extended)")
            elif isinstance(result["notebook_fault_ref"].get("fault_line_truth"), LineString):
                fx, fy = result["notebook_fault_ref"]["fault_line_truth"].xy
                ax.plot(fx, fy, "r-", lw=1.8, label="Fault (fitted)")
        elif not isinstance(fault_selected, LineString):
            fx, fy = result["notebook_fault_ref"]["fault_line_truth"].xy
            ax.plot(fx, fy, "r--", lw=1.5, label="Fault")
        if "fit_band" in result:
            tx, ty = result["fit_band"]["transect"].xy
            ax.plot(tx, ty, color="gray", linestyle="-", lw=1, alpha=0.9, label="Fault perpendicular line")
            # Show +/- half-length endpoints so transect_half_length is explicit on map.
            ax.scatter([tx[0], tx[-1]], [ty[0], ty[-1]], s=36, c="k", marker="x", label="Transect endpoint")
            l1, l2 = result["fit_band"]["offset_lines"]
            x1, y1 = l1.xy
            x2, y2 = l2.xy
            ax.plot(x1, y1, "k-.", lw=1.0, alpha=0.9, label="Fitting boundary")
            ax.plot(x2, y2, "k-.", lw=1.0, alpha=0.9)
            cap_lines = result["fit_band"].get("cap_lines", ())
            if len(cap_lines) == 2:
                c1, c2 = cap_lines
                cx1, cy1 = c1.xy
                cx2, cy2 = c2.xy
                ax.plot(cx1, cy1, "k:", lw=1.1, alpha=0.95, label="Fault parallel line")
                ax.plot(cx2, cy2, "k:", lw=1.1, alpha=0.95)
        self._add_basemap(ax, alpha=0.6, key="map")
        ax.set_title(f"InSAR, Fault, and Pipeline (Day {day_label})")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        mx_min = self._optional_float(self.le_map_xlim_min) if hasattr(self, "le_map_xlim_min") else None
        mx_max = self._optional_float(self.le_map_xlim_max) if hasattr(self, "le_map_xlim_max") else None
        my_min = self._optional_float(self.le_map_ylim_min) if hasattr(self, "le_map_ylim_min") else None
        my_max = self._optional_float(self.le_map_ylim_max) if hasattr(self, "le_map_ylim_max") else None
        if mx_min is None:
            mx_min = self.config.map_xlim_min
        if mx_max is None:
            mx_max = self.config.map_xlim_max
        if my_min is None:
            my_min = self.config.map_ylim_min
        if my_max is None:
            my_max = self.config.map_ylim_max
        if mx_min < mx_max:
            ax.set_xlim(mx_min, mx_max)
        else:
            ax.set_xlim(self.config.map_xlim_min, self.config.map_xlim_max)
        if my_min < my_max:
            ax.set_ylim(my_min, my_max)
        else:
            ax.set_ylim(self.config.map_ylim_min, self.config.map_ylim_max)
        handles, labels = ax.get_legend_handles_labels()
        lax.axis("off")
        lax.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(0.10, 1.02),
            frameon=True,
            borderaxespad=0.0,
            fontsize=9,
        )
        ax.set_aspect("equal", adjustable="box")
        self.tab_map.canvas.draw_idle()
        self._finalize_dynamic_basemap("map")

        # Tab2 fit
        self.tab_fit.figure.clear()
        fit_axes = self.tab_fit.figure.subplots(1, 2)
        self.tab_fit.ax = fit_axes[0]
        fit_ready = "tangential_fit_by_epoch" in result and "normal_fit_by_epoch" in result
        fit_payloads = self._fit_payload_for_selected_mode(result, epoch_idx) if fit_ready else None
        if fit_payloads is not None and len(fit_payloads) > 0:
            for i, payload in enumerate(fit_payloads[:2]):
                ax = fit_axes[i]
                x_data = payload["x_data"]
                y_data = payload["y_data"]
                show_mask = np.abs(x_data) <= 200
                ax.scatter(x_data[show_mask], y_data[show_mask], s=10, alpha=0.25, c="#d9a066", label="Data points")
                xfit = np.linspace(-200, 200, 600)
                yfit = payload["yfit_fn"](xfit)
                epi_curves = self._fault_parallel_epistemic_fit_curves(xfit, payload)
                if yfit is not None and np.isfinite(yfit).any():
                    ax.plot(xfit, yfit, color="#1f77b4", lw=2.5, label="Median fit", zorder=4)
                if epi_curves is not None:
                    y84 = np.asarray(epi_curves["y84"], dtype=float)
                    lo = np.asarray(epi_curves["ylow"], dtype=float)
                    hi = np.asarray(epi_curves["yhigh"], dtype=float)
                    y16_raw = epi_curves.get("y16")
                    if y16_raw is not None:
                        y16 = np.asarray(y16_raw, dtype=float)
                        ax.plot(xfit, y16, color="#1f77b4", linestyle="--", lw=1.6, alpha=0.8, label="16th / 84th percentile")
                        ax.plot(xfit, y84, color="#1f77b4", linestyle="--", lw=1.6, alpha=0.8)
                    else:
                        ax.plot(xfit, y84, color="#1f77b4", linestyle="--", lw=1.6, alpha=0.8, label="84th percentile (one-sided)")
                    ax.fill_between(xfit, lo, hi, color="#9ecae1", alpha=0.30, label="Epistemic envelope", zorder=1)
                ax.set_title(payload["direction"])
                ax.set_xlabel("Signed distance to fault (m)")
                ax.set_ylabel(payload["ylabel"])
                ax.legend()
                ax.grid(ls="--", alpha=0.3)
                if self.tabs.currentIndex() == 1:
                    self._log_fit_sigmoid_info(result, epoch_idx, payload)
            self.tab_fit.figure.suptitle(self.cmb_fit_data.currentText(), fontsize=11)
        else:
            for ax in fit_axes:
                ax.text(0.5, 0.5, "Run Step 2 to see fit.", ha="center", va="center", transform=ax.transAxes)
                ax.set_xlabel("Signed distance to fault (m)")
                ax.set_ylabel("Displacement (mm)")
                ax.grid(ls="--", alpha=0.3)
            self.tab_fit.figure.suptitle("Fit functions", fontsize=11)
        for ax in fit_axes:
            ax.set_xlim(-200, 200)
        self.tab_fit.figure.tight_layout()
        self.tab_fit.canvas.draw_idle()

        # Tab3 pipeline mesh
        ax_mesh = self.tab_pipe_mesh.ax
        ax_mesh.clear()
        # Mesh display is tied to Step 3/4 outputs.
        if "pipe" in result:
            mesh = self._build_pipeline_mesh(result)
            mesh_pts = np.asarray(mesh.get("points", []), dtype=float)
            mesh_conn = np.asarray(mesh.get("connectivity", []), dtype=int)
        else:
            mesh_pts = np.empty((0, 2), dtype=float)
            mesh_conn = np.empty((0, 2), dtype=int)
        if mesh_pts.ndim == 2 and mesh_pts.shape[0] > 1 and mesh_pts.shape[1] >= 2:
            shown_beam_elements = False
            for e0, e1 in mesh_conn:
                if 0 <= int(e0) < mesh_pts.shape[0] and 0 <= int(e1) < mesh_pts.shape[0]:
                    ax_mesh.plot(
                        [mesh_pts[int(e0), 0], mesh_pts[int(e1), 0]],
                        [mesh_pts[int(e0), 1], mesh_pts[int(e1), 1]],
                        "-",
                        color="#1f77b4",
                        lw=0.8,
                        alpha=0.75,
                        label="Beam elements" if not shown_beam_elements else None,
                    )
                    shown_beam_elements = True
            ax_mesh.scatter(mesh_pts[:, 0], mesh_pts[:, 1], s=8, c="#d62728", alpha=0.9, label="Nodes")
            n_nodes = int(mesh_pts.shape[0])
            n_elem = int(mesh_conn.shape[0]) if mesh_conn.ndim == 2 else 0
            mesh_log_key = f"{n_nodes}|{n_elem}"
            if self.tabs.currentWidget() is self.tab_pipe_mesh and mesh_log_key != self._last_mesh_log_key:
                self._last_mesh_log_key = mesh_log_key
                self.append_log(f"[Pipeline Mesh] Nodes={n_nodes}, Elements={n_elem}")
            ax_mesh.set_title("Pipeline mesh")
            ax_mesh.set_xlabel("x (m)", labelpad=10)
            ax_mesh.set_ylabel("y (m)")
            ax_mesh.legend()
            ax_mesh.grid(ls="--", alpha=0.3)
            ax_mesh.set_aspect("equal", adjustable="box")
        else:
            mesh_msg = "Run Step 3 to see pipeline mesh." if "pipe" not in result else "Mesh generation produced no valid nodes."
            ax_mesh.text(0.5, 0.5, mesh_msg, ha="center", va="center", transform=ax_mesh.transAxes)
            ax_mesh.grid(ls="--", alpha=0.3)
        self.tab_pipe_mesh.figure.tight_layout(rect=(0.02, 0.04, 0.98, 0.98), pad=0.9)
        self.tab_pipe_mesh.canvas.draw_idle()

        # Tab4 pipeline displacement
        self.tab_pipe_disp.figure.clear()
        disp_axes = self.tab_pipe_disp.figure.subplots(1, 2)
        self.tab_pipe_disp.ax = disp_axes[0]
        disp_payloads = self._pipe_disp_payloads(result, epoch_idx)
        if disp_payloads:
            mode = self.cmb_disp_coord.currentText() if hasattr(self, "cmb_disp_coord") else "Fault-parallel/perpendicular"
            has_override = self._has_fit_override_for_epoch(result, epoch_idx)
            label_16 = "16th Percentiles (original fit)" if has_override else "16th Percentiles"
            label_84 = "84th Percentiles (original fit)" if has_override else "84th Percentiles"
            pipe = self._get_pipe_prediction_for_mode(result, mode, need_strain=False) or {}
            ssi_on = bool(pipe.get("ssi_enabled", False)) or bool((result.get("pipe") or {}).get("ssi_enabled", False))
            cases = self._get_epistemic_cases(result, pipe, mode)
            case50 = cases.get("b50") if isinstance(cases, dict) else None
            source_for_main = pipe if mode == "N&E" else (case50 if isinstance(case50, dict) else pipe)
            soil_fault = np.asarray(source_for_main.get("soil_motion_fault_mm", []), dtype=float)
            soil_xy = np.asarray(source_for_main.get("soil_motion_xy_mm", []), dtype=float)
            xloc = self._reference_disp_xloc(result, pipe, disp_payloads)
            for i, (name, yv, y16, y84) in enumerate(disp_payloads[:2]):
                ax = disp_axes[i]
                yy_raw = np.asarray(yv, dtype=float).reshape(-1)
                x_plot = np.asarray(xloc, dtype=float).reshape(-1)
                if mode == "N&E":
                    n_aligned = int(min(x_plot.size, yy_raw.size))
                    if n_aligned <= 0:
                        ax.text(0.5, 0.5, "No valid N&E displacement series.", ha="center", va="center", transform=ax.transAxes)
                        ax.grid(ls="--", alpha=0.3)
                        continue
                    if yy_raw.size != x_plot.size:
                        self._append_log_once(
                            f"disp-xalign-{mode}-{name}",
                            f"[Step3 Render] {mode}/{name} node count {yy_raw.size} != x count {x_plot.size}; truncating to common length.",
                        )
                    x_plot = x_plot[:n_aligned]
                    yy = np.nan_to_num(yy_raw[:n_aligned], nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    if yy_raw.size != x_plot.size:
                        self._append_log_once(
                            f"disp-xalign-{mode}-{name}",
                            f"[Step3 Render] {mode}/{name} node count {yy_raw.size} != x count {x_plot.size}; resampling curve to match N&E node count.",
                        )
                    yy = np.nan_to_num(
                        self._resample_series_to_target_len(yy_raw, x_plot.size),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                soil_raw = None
                if mode == "Fault-parallel/perpendicular":
                    if soil_fault.ndim == 3 and soil_fault.shape[0] > epoch_idx:
                        idx = 0 if name == "Fault-parallel" else 1
                        soil_raw = np.asarray(soil_fault[epoch_idx, :, idx], dtype=float).reshape(-1)
                else:
                    if soil_xy.ndim == 3 and soil_xy.shape[0] > epoch_idx:
                        idx = 1 if name == "North" else 0
                        soil_raw = np.asarray(soil_xy[epoch_idx, :, idx], dtype=float).reshape(-1)
                if mode == "Fault-parallel/perpendicular" and soil_raw is not None and soil_raw.size > 0:
                    if mode == "N&E":
                        n_soil = int(min(x_plot.size, soil_raw.size))
                        soil_curve = np.nan_to_num(soil_raw[:n_soil], nan=0.0, posinf=0.0, neginf=0.0)
                        x_soil = x_plot[:n_soil]
                    else:
                        soil_curve = np.nan_to_num(
                            self._resample_series_to_target_len(soil_raw, x_plot.size),
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        )
                        x_soil = x_plot
                    if soil_curve.size > 0 and x_soil.size == soil_curve.size:
                        ax.plot(x_soil, soil_curve, "--", color="#3b5bff", lw=1.8, alpha=0.9, label=f"Soil {name}")
                ax.plot(x_plot, yy, "-", color="#e03131", lw=2.0, label=f"Pipe {name}")
                if (not ssi_on) and mode == "Fault-parallel/perpendicular" and y84 is not None:
                    y84_raw = np.asarray(y84, dtype=float).reshape(-1)
                    if mode == "N&E":
                        n_band = int(min(x_plot.size, yy.size, y84_raw.size))
                        x_band = x_plot[:n_band]
                        yy_band = yy[:n_band]
                        y84_arr = np.nan_to_num(y84_raw[:n_band], nan=0.0, posinf=0.0, neginf=0.0)
                    else:
                        x_band = x_plot
                        yy_band = yy
                        y84_arr = np.nan_to_num(
                            self._resample_series_to_target_len(y84_raw, x_plot.size),
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        )
                    if y16 is not None:
                        y16_raw = np.asarray(y16, dtype=float).reshape(-1)
                        if mode == "N&E":
                            n_band16 = int(min(x_band.size, y16_raw.size, y84_arr.size))
                            x_fill = x_band[:n_band16]
                            yy_fill = yy_band[:n_band16]
                            y84_fill = y84_arr[:n_band16]
                            y16_arr = np.nan_to_num(y16_raw[:n_band16], nan=0.0, posinf=0.0, neginf=0.0)
                        else:
                            x_fill = x_band
                            yy_fill = yy_band
                            y84_fill = y84_arr
                            y16_arr = np.nan_to_num(
                                self._resample_series_to_target_len(y16_raw, x_plot.size),
                                nan=0.0,
                                posinf=0.0,
                                neginf=0.0,
                            )
                        lo = np.minimum(y16_arr, y84_fill)
                        hi = np.maximum(y16_arr, y84_fill)
                        ax.plot(x_fill, y16_arr, "--", color="#1f77b4", alpha=0.65, lw=1.4, label=label_16)
                        ax.plot(x_fill, y84_fill, "--", color="#2ca02c", alpha=0.65, lw=1.4, label=label_84)
                    else:
                        x_fill = x_band
                        yy_fill = yy_band
                        lo = np.minimum(yy_fill, y84_arr)
                        hi = np.maximum(yy_fill, y84_arr)
                        ax.plot(x_fill, y84_arr, "--", color="#2ca02c", alpha=0.65, lw=1.4, label="84th percentile (one-sided)")
                    ax.fill_between(x_fill, lo, hi, color="#9ecae1", alpha=0.18, label="Epistemic envelope")
                ax.set_title(name)
                ax.set_xlabel("Location (m)")
                ax.set_ylabel("Displacement (mm)")
                if mode == "N&E":
                    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
                ax.grid(ls="--", alpha=0.3)
                ax.legend()
            self.tab_pipe_disp.figure.suptitle(self.cmb_disp_coord.currentText(), fontsize=11)
        else:
            mode = self.cmb_disp_coord.currentText() if hasattr(self, "cmb_disp_coord") else "Fault-parallel/perpendicular"
            if "pipe" not in result:
                empty_msg = "Run Step 3 to see pipeline displacement."
            else:
                empty_msg = f"No valid displacement payload for mode '{mode}' and current epoch."
            for ax in disp_axes:
                ax.text(0.5, 0.5, empty_msg, ha="center", va="center", transform=ax.transAxes)
                ax.grid(ls="--", alpha=0.3)
            self.tab_pipe_disp.figure.suptitle("Pipeline displacement", fontsize=11)
        self.tab_pipe_disp.figure.tight_layout()
        self.tab_pipe_disp.canvas.draw_idle()
        # Run one deferred relayout after Qt finishes this paint cycle.
        # This avoids first-render spacing artifacts when canvas size settles late.
        QTimer.singleShot(
            0,
            lambda: (self.tab_pipe_disp.figure.tight_layout(), self.tab_pipe_disp.canvas.draw_idle()),
        )

        # Tab5 pipeline displacement field
        fig_df = self.tab_pipe_disp_field.figure
        fig_df.clear()
        gs_df = fig_df.add_gridspec(
            1,
            3,
            width_ratios=[1.0, 0.050, 0.48],
            left=0.12,
            right=0.995,
            bottom=0.10,
            top=0.90,
            wspace=0.18,
        )
        ax_df = fig_df.add_subplot(gs_df[0, 0])
        cax_df = fig_df.add_subplot(gs_df[0, 1])
        lax_df = fig_df.add_subplot(gs_df[0, 2])
        self.tab_pipe_disp_field.ax = ax_df
        has_colorbar_df = False
        if "pipe" in result:
            mode = self.cmb_disp_coord.currentText() if hasattr(self, "cmb_disp_coord") else "Fault-parallel/perpendicular"
            pipe = self._get_pipe_prediction_for_mode(result, mode, need_strain=False) or {}
            motion_xy = np.asarray(pipe.get("pipe_motion_xy_mm", []), dtype=float)
            motion_fault = np.asarray(pipe.get("pipe_motion_fault_mm", []), dtype=float)
            pipe_axis = np.asarray(pipe.get("pipe_axis_nodes", data.get("pipe_axis", [])), dtype=float)
            if motion_xy.ndim == 3 and motion_xy.shape[0] > epoch_idx and pipe_axis.ndim == 2 and pipe_axis.shape[0] > 1:
                i_sel = min(max(epoch_idx, 0), motion_xy.shape[0] - 1)
                deformed = pipe_axis + motion_xy[i_sel] * 1e-3
                disp_dir = self.cmb_disp_field_dir.currentText() if hasattr(self, "cmb_disp_field_dir") else ""
                disp_target = (
                    self.cmb_disp_field_target.currentText()
                    if hasattr(self, "cmb_disp_field_target")
                    else "Node"
                )
                if mode == "N&E":
                    if disp_dir == "North":
                        cvals = np.asarray(motion_xy[i_sel, :, 1], dtype=float)
                    else:
                        cvals = np.asarray(motion_xy[i_sel, :, 0], dtype=float)
                else:
                    if motion_fault.ndim == 3 and motion_fault.shape[0] > i_sel:
                        if disp_dir == "Fault-perpendicular":
                            cvals = np.asarray(motion_fault[i_sel, :, 1], dtype=float)
                        else:
                            cvals = np.asarray(motion_fault[i_sel, :, 0], dtype=float)
                    else:
                        cvals = np.array([])
                if cvals.size > 0:
                    cvals = np.nan_to_num(cvals, nan=0.0, posinf=0.0, neginf=0.0)
                    self._apply_field_context_layers(ax_df, result, epoch_idx, basemap_key="disp_field")
                    if disp_target == "Element" and deformed.shape[0] > 1:
                        conn = np.asarray(pipe.get("pipe_connectivity", []), dtype=int)
                        if conn.ndim != 2 or conn.shape[1] != 2 or conn.shape[0] == 0:
                            conn = np.column_stack(
                                [np.arange(0, deformed.shape[0] - 1), np.arange(1, deformed.shape[0])]
                            ).astype(int)
                        elem_vals = self._safe_element_values_from_nodes(cvals, conn)
                        segments = np.stack([deformed[conn[:, 0]], deformed[conn[:, 1]]], axis=1)
                        lc = LineCollection(segments, cmap="coolwarm", linewidths=4.0, alpha=0.95)
                        lc.set_array(elem_vals)
                        ax_df.add_collection(lc)
                        fig_df.colorbar(lc, cax=cax_df, label=f"{disp_dir} displacement (mm)")
                        has_colorbar_df = True
                    else:
                        scat = ax_df.scatter(
                            deformed[:, 0],
                            deformed[:, 1],
                            c=cvals,
                            marker="o",
                            cmap="coolwarm",
                            s=20,
                            label=f"{disp_dir} displacement",
                        )
                        fig_df.colorbar(scat, cax=cax_df, label=f"{disp_dir} displacement (mm)")
                        has_colorbar_df = True
                    ax_df.set_xlabel("x (m)")
                    ax_df.set_ylabel("y (m)")
                    ax_df.set_title(f"Displacement field - {disp_dir} ({disp_target.lower()})", pad=6)
                    ax_df.grid(ls="--", alpha=0.3)
                    ax_df.set_aspect("equal", adjustable="box")
                else:
                    ax_df.text(
                        0.5,
                        0.5,
                        "No valid displacement component for selected mode/direction.",
                        ha="center",
                        va="center",
                        transform=ax_df.transAxes,
                    )
            else:
                ax_df.text(
                    0.5,
                    0.5,
                    "No valid displacement field payload for current epoch.",
                    ha="center",
                    va="center",
                    transform=ax_df.transAxes,
                )
        else:
            ax_df.text(0.5, 0.5, "Run Step 3 to see displacement field.", ha="center", va="center", transform=ax_df.transAxes)
        if not has_colorbar_df:
            cax_df.axis("off")
        handles_df, labels_df = ax_df.get_legend_handles_labels()
        lax_df.axis("off")
        if handles_df:
            lax_df.legend(
                handles_df,
                labels_df,
                loc="upper left",
                bbox_to_anchor=(0.10, 1.02),
                frameon=True,
                borderaxespad=0.0,
                fontsize=9,
            )
        self.tab_pipe_disp_field.canvas.draw_idle()
        if "disp_field" in self._basemap_watch:
            self._finalize_dynamic_basemap("disp_field")

        # Tab6 pipeline strain
        ax = self.tab_pipe_strain.ax
        ax.clear()
        strain_type = self.cmb_strain_type.currentText() if hasattr(self, "cmb_strain_type") else "Axial"
        if "pipe" in result and self._has_step4_outputs(result):
            mode = self.cmb_disp_coord.currentText() if hasattr(self, "cmb_disp_coord") else "Fault-parallel/perpendicular"
            has_override = self._has_fit_override_for_epoch(result, epoch_idx)
            label_16 = "16th Percentiles (original fit)" if has_override else "16th Percentiles"
            label_84 = "84th Percentiles (original fit)" if has_override else "84th Percentiles"
            pipe = self._get_pipe_prediction_for_mode(result, mode, need_strain=True) or {}
            ssi_on = bool(pipe.get("ssi_enabled", False)) or bool((result.get("pipe") or {}).get("ssi_enabled", False))
            cases = self._get_epistemic_cases(result, pipe, mode)
            # Restore: remove the SSI guard to show 16/84 envelopes again.
            case16 = cases.get("b16") if (isinstance(cases, dict) and not ssi_on) else None
            case84 = cases.get("b84") if (isinstance(cases, dict) and not ssi_on) else None
            peak_info = {"Axial": None, "Bending": None}
            # Pre-compute both strain peaks before type selection branch.
            axial_arr_all = np.asarray(pipe.get("axial_strain_smoothed", []), dtype=float)
            if axial_arr_all.ndim == 2 and axial_arr_all.shape[0] > 0:
                idx_ax_all = min(max(epoch_idx, 0), axial_arr_all.shape[0] - 1)
                xloc_ax_all = np.asarray(pipe.get("s_nodes_offset", np.arange(len(axial_arr_all[idx_ax_all]))))
                vals_ax_micro = np.nan_to_num(axial_arr_all[idx_ax_all], nan=0.0, posinf=0.0, neginf=0.0) * 1e6
                peak_info["Axial"] = self._compute_peak_abs_with_location(xloc_ax_all, vals_ax_micro)
            bending_arr_all = np.asarray(pipe.get("bending_strain", []), dtype=float)
            if bending_arr_all.ndim == 2 and bending_arr_all.shape[0] > 0:
                idx_b_all = min(max(epoch_idx, 0), bending_arr_all.shape[0] - 1)
                vals_b_micro = np.nan_to_num(bending_arr_all[idx_b_all], nan=0.0, posinf=0.0, neginf=0.0) * 1e6
                step_all = max(int(self.config.bending_sample_step), 1)
                xloc_b_all = np.asarray(pipe.get("s_nodes_offset", np.arange(len(vals_b_micro))))[::step_all]
                if len(xloc_b_all) != len(vals_b_micro):
                    xloc_b_all = np.linspace(0, len(vals_b_micro) - 1, len(vals_b_micro))
                peak_info["Bending"] = self._compute_peak_abs_with_location(xloc_b_all, vals_b_micro)
            if strain_type == "Axial":
                axial_arr = np.asarray(pipe.get("axial_strain_smoothed", []), dtype=float)
                if axial_arr.ndim == 2 and axial_arr.shape[0] > 0:
                    idx_ax = min(max(epoch_idx, 0), axial_arr.shape[0] - 1)
                    xloc = np.asarray(pipe.get("s_nodes_offset", np.arange(len(axial_arr[idx_ax]))))
                    axial_sel = np.nan_to_num(axial_arr[idx_ax], nan=0.0, posinf=0.0, neginf=0.0) * 1e6
                    ax16 = ax84 = None
                    if isinstance(case16, dict):
                        a16 = np.asarray(case16.get("axial_strain_smoothed", []), dtype=float)
                        if a16.ndim == 2 and a16.shape[0] > idx_ax:
                            ax16 = np.nan_to_num(a16[idx_ax], nan=0.0, posinf=0.0, neginf=0.0) * 1e6
                    if isinstance(case84, dict):
                        a84 = np.asarray(case84.get("axial_strain_smoothed", []), dtype=float)
                        if a84.ndim == 2 and a84.shape[0] > idx_ax:
                            ax84 = np.nan_to_num(a84[idx_ax], nan=0.0, posinf=0.0, neginf=0.0) * 1e6
                    if ax84 is not None:
                        if ax16 is not None:
                            lo = np.minimum(ax16, ax84)
                            hi = np.maximum(ax16, ax84)
                            ax.plot(xloc, ax16, ":", color="#7fb3d5", alpha=0.8, label=label_16)
                            ax.plot(xloc, ax84, "--", color="#7fb3d5", alpha=0.8, label=label_84)
                        else:
                            lo = np.minimum(axial_sel, ax84)
                            hi = np.maximum(axial_sel, ax84)
                            ax.plot(xloc, ax84, "--", color="#7fb3d5", alpha=0.8, label="84th percentile (one-sided)")
                        ax.fill_between(xloc, lo, hi, color="#7fb3d5", alpha=0.22, label="Epistemic envelope")
                    ax.plot(xloc, axial_sel, color="#7fb3d5", lw=2.2, label="Axial strain")
                    ax.set_xlabel("Location (m)")
                    ax.set_ylabel("Axial strain (microstrain)")
            else:
                bending_arr = np.asarray(pipe.get("bending_strain", []), dtype=float)
                if bending_arr.ndim == 2 and bending_arr.shape[0] > 0:
                    idx_b = min(max(epoch_idx, 0), bending_arr.shape[0] - 1)
                    bending_sel = np.nan_to_num(bending_arr[idx_b], nan=0.0, posinf=0.0, neginf=0.0) * 1e6
                    step = max(int(self.config.bending_sample_step), 1)
                    xloc_b = np.asarray(pipe.get("s_nodes_offset", np.arange(len(bending_sel))))[::step]
                    if len(xloc_b) != len(bending_sel):
                        xloc_b = np.linspace(0, len(bending_sel) - 1, len(bending_sel))
                    bd16 = bd84 = None
                    if isinstance(case16, dict):
                        b16 = np.asarray(case16.get("bending_strain", []), dtype=float)
                        if b16.ndim == 2 and b16.shape[0] > idx_b:
                            bd16 = np.nan_to_num(b16[idx_b], nan=0.0, posinf=0.0, neginf=0.0) * 1e6
                    if isinstance(case84, dict):
                        b84 = np.asarray(case84.get("bending_strain", []), dtype=float)
                        if b84.ndim == 2 and b84.shape[0] > idx_b:
                            bd84 = np.nan_to_num(b84[idx_b], nan=0.0, posinf=0.0, neginf=0.0) * 1e6
                    if bd84 is not None:
                        if bd16 is not None:
                            lo = np.minimum(bd16, bd84)
                            hi = np.maximum(bd16, bd84)
                            ax.plot(xloc_b, bd16, ":", color="#d291ff", alpha=0.8, label=label_16)
                            ax.plot(xloc_b, bd84, "--", color="#d291ff", alpha=0.8, label=label_84)
                        else:
                            lo = np.minimum(bending_sel, bd84)
                            hi = np.maximum(bending_sel, bd84)
                            ax.plot(xloc_b, bd84, "--", color="#d291ff", alpha=0.8, label="84th percentile (one-sided)")
                        ax.fill_between(xloc_b, lo, hi, color="#d291ff", alpha=0.20, label="Epistemic envelope")
                    ax.plot(xloc_b, bending_sel, "m-", lw=1.5, label="Bending strain")
                    ax.set_xlabel("Location (m)")
                    ax.set_ylabel("Bending strain (microstrain)")
            peak = peak_info.get(strain_type)
            if peak is not None:
                x_peak = float(peak["x_m"])
                y_peak = float(peak["signed_microstrain"])
                txt = f"{peak['abs_microstrain']:.2f} με"
                ax.scatter([x_peak], [y_peak], s=46, c="red", edgecolors="white", linewidths=0.8, zorder=6)
                x_min, x_max = ax.get_xlim()
                y_min, y_max = ax.get_ylim()
                x_pad = 0.02 * (x_max - x_min) if np.isfinite(x_max - x_min) else 1.0
                y_pad = 0.04 * (y_max - y_min) if np.isfinite(y_max - y_min) else 1.0
                x_txt_try = x_peak + x_pad
                y_txt_try = y_peak + y_pad
                x_txt = min(max(x_txt_try, x_min + x_pad), x_max - x_pad)
                y_txt = min(max(y_txt_try, y_min + y_pad), y_max - y_pad)
                ha = "left" if x_txt >= x_peak else "right"
                va = "bottom" if y_txt >= y_peak else "top"
                ax.text(
                    x_txt,
                    y_txt,
                    txt,
                    color="red",
                    fontsize=10,
                    ha=ha,
                    va=va,
                    clip_on=True,
                    zorder=7,
                )
                if self.tabs.currentWidget() is self.tab_pipe_strain:
                    pair_txt = "N/A"
                    pairs = result.get("matched", {}).get("matched_pairs", [])
                    if 0 <= epoch_idx < len(pairs):
                        pair_txt = f"{pairs[epoch_idx][0]}/{pairs[epoch_idx][1]}"
                    peak_ax = peak_info.get("Axial")
                    peak_bd = peak_info.get("Bending")
                    if peak_ax is None or peak_bd is None:
                        peak_ax = peak_ax or {"abs_microstrain": np.nan, "x_m": np.nan}
                        peak_bd = peak_bd or {"abs_microstrain": np.nan, "x_m": np.nan}
                    log_key = (
                        f"{epoch_idx}|{mode}|{peak_info.get('Axial')}|{peak_info.get('Bending')}"
                    )
                    if log_key != self._last_strain_peak_log_key:
                        self._last_strain_peak_log_key = log_key
                        sigma_arr = np.asarray(result.get("epistemic_final_sigma_by_epoch", []), dtype=float)
                        sigma_txt = ""
                        if sigma_arr.ndim == 1 and sigma_arr.size > epoch_idx and np.isfinite(sigma_arr[epoch_idx]):
                            sigma_txt = f", epistemic_sigma={float(sigma_arr[epoch_idx]):.3f}"
                        self.append_log(
                            f"[Strain Peak] Date: {pair_txt}, Mode: {mode}, "
                            f"Axial max|strain|={peak_ax['abs_microstrain']:.2f} με @ x={peak_ax['x_m']:.2f} m, "
                            f"Bending max|strain|={peak_bd['abs_microstrain']:.2f} με @ x={peak_bd['x_m']:.2f} m"
                            f"{sigma_txt}"
                        )
        else:
            ax.text(0.5, 0.5, "Run Step 4 to see pipeline strain.", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"Pipeline Strain - {strain_type}")
        ax.grid(ls="--", alpha=0.3)
        ax.legend()
        self.tab_pipe_strain.figure.tight_layout()
        self.tab_pipe_strain.canvas.draw_idle()

        # Tab7 pipeline strain field
        fig_sf = self.tab_pipe_strain_field.figure
        fig_sf.clear()
        gs_sf = fig_sf.add_gridspec(
            1,
            3,
            width_ratios=[1.0, 0.050, 0.48],
            left=0.12,
            right=0.995,
            bottom=0.10,
            top=0.90,
            wspace=0.18,
        )
        ax_sf = fig_sf.add_subplot(gs_sf[0, 0])
        cax_sf = fig_sf.add_subplot(gs_sf[0, 1])
        lax_sf = fig_sf.add_subplot(gs_sf[0, 2])
        self.tab_pipe_strain_field.ax = ax_sf
        has_colorbar_sf = False
        if "pipe" in result and self._has_step4_outputs(result):
            mode = self.cmb_disp_coord.currentText() if hasattr(self, "cmb_disp_coord") else "Fault-parallel/perpendicular"
            field_view = (
                self.cmb_strain_field_view.currentText()
                if hasattr(self, "cmb_strain_field_view")
                else "Signed"
            )
            field_target = (
                self.cmb_strain_field_target.currentText()
                if hasattr(self, "cmb_strain_field_target")
                else "Node"
            )
            signed_view = field_view == "Signed"
            pipe = self._get_pipe_prediction_for_mode(result, mode, need_strain=True) or {}
            motion_xy = np.asarray(pipe.get("pipe_motion_xy_mm", []), dtype=float)
            pipe_axis = np.asarray(pipe.get("pipe_axis_nodes", data.get("pipe_axis", [])), dtype=float)
            if motion_xy.ndim == 3 and motion_xy.shape[0] > epoch_idx and pipe_axis.ndim == 2 and pipe_axis.shape[0] > 1:
                i_sel = min(max(epoch_idx, 0), motion_xy.shape[0] - 1)
                deformed = pipe_axis + motion_xy[i_sel] * 1e-3
                cvals = np.array([])
                cbar_label = ""
                title_txt = ""
                draw_element = field_target == "Element"
                if strain_type == "Axial":
                    vals = np.asarray(pipe.get("axial_strain_smoothed", []), dtype=float)
                    if draw_element:
                        conn = np.asarray(pipe.get("pipe_connectivity", []), dtype=int)
                        if conn.ndim != 2 or conn.shape[1] != 2 or conn.shape[0] == 0:
                            conn = np.column_stack(
                                [np.arange(0, max(pipe_axis.shape[0] - 1, 0)), np.arange(1, max(pipe_axis.shape[0], 1))]
                            ).astype(int)
                        cvals = self._compute_element_axial_strain(pipe_axis, deformed, conn) * 1e6
                        cvals = cvals if signed_view else np.abs(cvals)
                        segments = np.stack([pipe_axis[conn[:, 0]], pipe_axis[conn[:, 1]]], axis=1) if conn.size else np.array([])
                        cbar_label = "Axial strain (microstrain)"
                        title_txt = (
                            "Signed axial strain (element)"
                            if signed_view
                            else "Absolute axial strain (element)"
                        )
                    elif vals.ndim == 2 and vals.shape[0] > i_sel:
                        base_vals = np.asarray(vals[i_sel], dtype=float) * 1e6
                        cvals = base_vals if signed_view else np.abs(base_vals)
                        xy_field = deformed
                        cbar_label = "Axial strain (microstrain)"
                        title_txt = "Signed axial strain (node)" if signed_view else "Absolute axial strain (node)"
                else:
                    vals = np.asarray(pipe.get("bending_strain", []), dtype=float)
                    if vals.ndim == 2 and vals.shape[0] > i_sel:
                        base_vals = np.asarray(vals[i_sel], dtype=float) * 1e6
                        step = max(int(self.config.bending_sample_step), 1)
                        xy_b = deformed[::step]
                        if xy_b.shape[0] != base_vals.shape[0]:
                            idxs = np.linspace(0, max(deformed.shape[0] - 1, 0), num=base_vals.shape[0], dtype=int)
                            xy_b = deformed[idxs]
                        if draw_element and xy_b.shape[0] > 1:
                            seg_ids = np.asarray(pipe.get("pipe_axis_sampled_segment_ids", []), dtype=int).reshape(-1)
                            if seg_ids.size == xy_b.shape[0]:
                                idx0 = np.arange(0, xy_b.shape[0] - 1, dtype=int)
                                idx1 = np.arange(1, xy_b.shape[0], dtype=int)
                                keep = seg_ids[idx0] == seg_ids[idx1]
                                conn = np.column_stack([idx0[keep], idx1[keep]]).astype(int)
                            else:
                                conn = np.column_stack(
                                    [np.arange(0, xy_b.shape[0] - 1), np.arange(1, xy_b.shape[0])]
                                ).astype(int)
                            cvals = self._safe_element_values_from_nodes(base_vals, conn)
                            cvals = cvals if signed_view else np.abs(cvals)
                            segments = np.stack([xy_b[conn[:, 0]], xy_b[conn[:, 1]]], axis=1)
                            cbar_label = "Bending strain (microstrain)"
                            title_txt = (
                                "Signed bending strain (element)"
                                if signed_view
                                else "Absolute bending strain (element)"
                            )
                        else:
                            cvals = base_vals if signed_view else np.abs(base_vals)
                            xy_field = xy_b
                            cbar_label = "Bending strain (microstrain)"
                            title_txt = "Signed bending strain (node)" if signed_view else "Absolute bending strain (node)"

                if np.asarray(cvals, dtype=float).size > 0:
                    cvals = np.nan_to_num(np.asarray(cvals, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
                    self._apply_field_context_layers(ax_sf, result, epoch_idx, basemap_key="strain_field")
                    if signed_view:
                        vmax = float(np.nanmax(np.abs(cvals)))
                        if not np.isfinite(vmax) or vmax <= 0:
                            vmax = 1.0
                        vmin = -vmax
                    else:
                        vmin = None
                        vmax = None
                    if draw_element:
                        lc = LineCollection(segments, cmap="coolwarm", linewidths=4.0, alpha=0.95)
                        lc.set_array(cvals)
                        if vmin is not None and vmax is not None:
                            lc.set_clim(vmin, vmax)
                        ax_sf.add_collection(lc)
                        fig_sf.colorbar(lc, cax=cax_sf, label=cbar_label)
                        has_colorbar_sf = True
                    else:
                        scatter_kwargs = {}
                        if vmin is not None and vmax is not None:
                            scatter_kwargs["vmin"] = vmin
                            scatter_kwargs["vmax"] = vmax
                        scat = ax_sf.scatter(
                            xy_field[:, 0],
                            xy_field[:, 1],
                            c=cvals,
                            marker="o",
                            cmap="coolwarm",
                            s=20,
                            label=f"{strain_type} strain",
                            **scatter_kwargs,
                        )
                        fig_sf.colorbar(scat, cax=cax_sf, label=cbar_label)
                        has_colorbar_sf = True
                    ax_sf.set_xlabel("x (m)")
                    ax_sf.set_ylabel("y (m)")
                    ax_sf.set_title(title_txt, pad=6)
                    ax_sf.grid(ls="--", alpha=0.3)
                    ax_sf.set_aspect("equal", adjustable="box")
                else:
                    ax_sf.text(
                        0.5,
                        0.5,
                        "Run Step 4 to see strain field.",
                        ha="center",
                        va="center",
                        transform=ax_sf.transAxes,
                    )
            else:
                ax_sf.text(0.5, 0.5, "Run Step 4 to see strain field.", ha="center", va="center", transform=ax_sf.transAxes)
        else:
            ax_sf.text(0.5, 0.5, "Run Step 4 to see strain field.", ha="center", va="center", transform=ax_sf.transAxes)
        if not has_colorbar_sf:
            cax_sf.axis("off")
        handles_sf, labels_sf = ax_sf.get_legend_handles_labels()
        lax_sf.axis("off")
        if handles_sf:
            lax_sf.legend(
                handles_sf,
                labels_sf,
                loc="upper left",
                bbox_to_anchor=(0.10, 1.02),
                frameon=True,
                borderaxespad=0.0,
                fontsize=9,
            )
        self.tab_pipe_strain_field.canvas.draw_idle()
        if "strain_field" in self._basemap_watch:
            self._finalize_dynamic_basemap("strain_field")
        self._render_fiber_calibration_plot(result)

    def on_export(self):
        if not self.last_result:
            QMessageBox.information(self, "Info", "Please run the pipeline first.")
            return
        try:
            figs = {
                "map": self.tab_map.figure,
                "fit": self.tab_fit.figure,
                "pipeline_mesh": self.tab_pipe_mesh.figure,
                "pipeline_displacement": self.tab_pipe_disp.figure,
                "pipeline_displacement_field": self.tab_pipe_disp_field.figure,
                "pipeline_strain": self.tab_pipe_strain.figure,
                "pipeline_strain_field": self.tab_pipe_strain_field.figure,
            }
            run_dir = export_pipeline_results(
                self.config,
                arrays=self.last_result["arrays"],
                tables=self.last_result["tables"],
                figures=figs,
            )
            self.append_log(f"Export completed: {run_dir}")
            QMessageBox.information(self, "Export Completed", str(run_dir))
        except Exception as e:
            self.append_log(traceback.format_exc())
            QMessageBox.critical(self, "Export Failed", str(e))

    def on_export_opensra(self):
        if not self.last_result:
            QMessageBox.information(self, "Info", "Please run the pipeline first.")
            return
        if not self._has_step4_outputs(self.last_result):
            QMessageBox.information(self, "Info", "Please run Step 4 first.")
            return
        mode = self.cmb_disp_coord.currentText() if hasattr(self, "cmb_disp_coord") else "Fault-parallel/perpendicular"
        pipe = self._get_pipe_prediction_for_mode(self.last_result, mode, need_strain=True) or {}
        epoch_idx = self._selected_epoch_index(self.last_result)
        if epoch_idx < 0:
            QMessageBox.information(self, "Info", "Please select a valid date before exporting.")
            return

        nodes = np.asarray(pipe.get("pipe_axis_nodes", []), dtype=float)
        conn = np.asarray(pipe.get("pipe_connectivity", []), dtype=int)
        if nodes.ndim != 2 or nodes.shape[0] < 2 or conn.ndim != 2 or conn.shape[1] != 2 or conn.shape[0] == 0:
            QMessageBox.information(self, "Info", "Pipe mesh is unavailable. Please run Step 4 again.")
            return

        matched = self.last_result.get("matched", {}) if isinstance(self.last_result, dict) else {}
        day_diffs = np.asarray(matched.get("day_diffs", []), dtype=float).reshape(-1)
        if day_diffs.size == 0:
            QMessageBox.critical(self, "Export Failed", "Missing day_diffs for yearly strain-rate conversion.")
            return
        i_day = min(max(int(epoch_idx), 0), day_diffs.size - 1)
        years_elapsed = float(day_diffs[i_day]) / 365.25
        if years_elapsed <= 1e-12:
            QMessageBox.information(
                self,
                "Info",
                "Selected epoch has zero elapsed time from the baseline. Please choose a later date for yearly strain-rate export.",
            )
            return

        comp_lsr, ext_lsr = self._compute_element_longitudinal_strain_rate(pipe, epoch_idx, years_elapsed)
        if comp_lsr.size != conn.shape[0] or ext_lsr.size != conn.shape[0]:
            QMessageBox.critical(self, "Export Failed", "Failed to compute element strain-rate fields.")
            return

        def _load_source_pipeline_attributes():
            try:
                result_cfg = (
                    self.last_result.get("config", self.config)
                    if isinstance(self.last_result, dict)
                    else self.config
                )
                pipe_path_raw = str(getattr(result_cfg, "pipe_axis_path", "")).strip()
                if not pipe_path_raw:
                    return [], [], []
                resolver = getattr(result_cfg, "resolve_path", None)
                if callable(resolver):
                    pipe_path = Path(resolver(pipe_path_raw))
                else:
                    pipe_path = self.config.resolve_path(pipe_path_raw)
                if pipe_path.suffix.lower() != ".shp" or not pipe_path.exists():
                    return [], [], []

                gdf_src = gpd.read_file(pipe_path)
                if gdf_src.empty:
                    return [], [], []
                if gdf_src.crs is None:
                    src_crs_txt = str(
                        getattr(result_cfg, "pipeline_input_crs", self._pipeline_input_crs_text())
                    ).strip()
                    if src_crs_txt:
                        if not src_crs_txt.upper().startswith("EPSG:"):
                            src_crs_txt = f"EPSG:{src_crs_txt}"
                        gdf_src = gdf_src.set_crs(src_crs_txt, allow_override=True)
                try:
                    if gdf_src.crs is not None:
                        gdf_src = gdf_src.to_crs(self._processing_crs_text())
                except Exception:
                    pass

                attr_cols = [c for c in list(gdf_src.columns) if c != gdf_src.geometry.name]
                flattened_lines = []
                flattened_attrs = []
                for _, row in gdf_src.iterrows():
                    geom = row.geometry
                    attrs = {c: row.get(c) for c in attr_cols}
                    if geom is None or getattr(geom, "is_empty", True):
                        continue
                    if isinstance(geom, LineString):
                        flattened_lines.append(geom)
                        flattened_attrs.append(attrs)
                    elif isinstance(geom, MultiLineString):
                        for ln in geom.geoms:
                            if ln is None or ln.is_empty:
                                continue
                            flattened_lines.append(ln)
                            flattened_attrs.append(attrs)

                if not flattened_lines:
                    return [], [], []

                fsum = self.last_result.get("fault_trace_summary", {}) if isinstance(self.last_result, dict) else {}
                selected_idxs_raw = fsum.get("selected_pipeline_idxs", [])
                selected_idxs = []
                for v in selected_idxs_raw if isinstance(selected_idxs_raw, (list, tuple, np.ndarray)) else []:
                    try:
                        vi = int(v)
                    except Exception:
                        continue
                    if 0 <= vi < len(flattened_lines):
                        selected_idxs.append(vi)
                selected_idxs = sorted(set(selected_idxs))
                if not selected_idxs:
                    return [], [], []

                selected_lines = [flattened_lines[i] for i in selected_idxs]
                selected_attrs = [flattened_attrs[i] for i in selected_idxs]
                return selected_lines, selected_attrs, attr_cols
            except Exception:
                self.append_log(traceback.format_exc())
                return [], [], []

        def _nearest_attr_for_geom(line_geom: LineString, src_lines: List[LineString], src_attrs: List[Dict[str, Any]]):
            if not src_lines or not src_attrs:
                return None
            try:
                mid = line_geom.interpolate(0.5, normalized=True)
            except Exception:
                mid = Point(np.asarray(line_geom.coords, dtype=float).mean(axis=0)[:2])
            best_i = -1
            best_d = float("inf")
            for k, s_ln in enumerate(src_lines):
                try:
                    d = float(s_ln.distance(mid))
                except Exception:
                    continue
                if d < best_d:
                    best_d = d
                    best_i = k
            if best_i < 0 or best_i >= len(src_attrs):
                return None
            return src_attrs[best_i]

        def _safe_shp_field_map(columns: List[str]) -> Dict[str, str]:
            fmap: Dict[str, str] = {}
            used = set()
            for col in columns:
                if col == "geometry":
                    continue
                if col == "epsrate%c":
                    base = "epsrate_c"
                elif col == "epsrate%t":
                    base = "epsrate_t"
                else:
                    base = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in str(col))
                    if not base:
                        base = "field"
                    if base[0].isdigit():
                        base = f"f_{base}"
                cand = base[:10]
                if not cand:
                    cand = "field"
                if cand not in used:
                    fmap[col] = cand
                    used.add(cand)
                    continue
                i = 1
                while True:
                    suf = str(i)
                    trimmed = cand[: max(1, 10 - len(suf))]
                    new_name = f"{trimmed}{suf}"
                    if new_name not in used:
                        fmap[col] = new_name
                        used.add(new_name)
                        break
                    i += 1
            return fmap

        def _diameter_inch_to_mm(v: Any) -> Optional[float]:
            if v is None:
                return None
            try:
                if isinstance(v, (int, float, np.integer, np.floating)):
                    if not np.isfinite(float(v)):
                        return None
                    return float(v) * 25.4
            except Exception:
                return None
            txt = str(v).strip().replace(",", "")
            if not txt:
                return None
            num = pd.to_numeric(txt, errors="coerce")
            if pd.isna(num):
                return None
            try:
                return float(num) * 25.4
            except Exception:
                return None

        src_lines, src_attrs, src_cols = _load_source_pipeline_attributes()
        node_seg_ids = np.asarray(pipe.get("pipe_segment_ids", []), dtype=int).reshape(-1)

        elem_geoms = []
        comp_out = []
        ext_out = []
        elem_seg_ids = []
        for i, (n0, n1) in enumerate(conn):
            n0i = int(n0)
            n1i = int(n1)
            if not (0 <= n0i < nodes.shape[0] and 0 <= n1i < nodes.shape[0]) or n0i == n1i:
                continue
            line_geom = LineString([nodes[n0i, :2], nodes[n1i, :2]])
            elem_geoms.append(line_geom)
            comp_out.append(float(max(comp_lsr[i], 0.0)))
            ext_out.append(float(max(ext_lsr[i], 0.0)))
            seg_id = -1
            if node_seg_ids.size == nodes.shape[0]:
                seg_id = int(node_seg_ids[n0i])
            elem_seg_ids.append(seg_id)
        if not elem_geoms:
            QMessageBox.critical(self, "Export Failed", "No valid mesh elements available for shapefile export.")
            return

        out_dir = Path(self.config.output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        suggested = out_dir / f"opensra_pipe_strain_rate_epoch_{epoch_idx + 1}"
        fp, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results for OpenSRA",
            str(suggested),
            "Geo Files (*.shp *.gpkg)",
        )
        if not fp:
            return
        base = Path(fp)
        if base.suffix.lower() in {".shp", ".gpkg"}:
            base = base.with_suffix("")
        shp_fp = str(base.with_suffix(".shp"))
        gpkg_fp = str(base.with_suffix(".gpkg"))

        try:
            gpkg_data: Dict[str, Any] = {
                "epsrate%c": np.asarray(comp_out, dtype=float),
                "epsrate%t": np.asarray(ext_out, dtype=float),
                "Diam_mm": [None] * len(elem_geoms),
            }

            if src_lines and src_attrs and src_cols:
                seg_attr_map = {i: src_attrs[i] for i in range(min(len(src_lines), len(src_attrs)))}
                attr_values = {c: [] for c in src_cols}
                diam_mm_values = []
                for j, geom in enumerate(elem_geoms):
                    sid = int(elem_seg_ids[j]) if j < len(elem_seg_ids) else -1
                    attrs = seg_attr_map.get(sid)
                    if attrs is None:
                        attrs = _nearest_attr_for_geom(geom, src_lines, src_attrs)
                    diam_src = None
                    if isinstance(attrs, dict):
                        diam_src = attrs.get("Diameter")
                        if diam_src is None:
                            for k in attrs.keys():
                                if str(k).strip().lower() == "diameter":
                                    diam_src = attrs.get(k)
                                    break
                    diam_mm_values.append(_diameter_inch_to_mm(diam_src))
                    for col in src_cols:
                        v = attrs.get(col) if isinstance(attrs, dict) else None
                        if isinstance(v, (list, dict, tuple, np.ndarray)):
                            v = json.dumps(v, ensure_ascii=False)
                        if pd.isna(v):
                            v = None
                        attr_values[col].append(v)
                for col, vals in attr_values.items():
                    gpkg_data[col] = vals
                gpkg_data["Diam_mm"] = diam_mm_values

            gdf_gpkg = gpd.GeoDataFrame(gpkg_data, geometry=elem_geoms, crs=self._processing_crs_text())
            gdf_export = gdf_gpkg.to_crs("EPSG:4326")

            shp_field_map = _safe_shp_field_map([c for c in gdf_export.columns if c != "geometry"])
            gdf_shp = gdf_export.rename(columns=shp_field_map)
            try:
                gdf_shp.to_file(shp_fp, driver="ESRI Shapefile")
            except Exception as e:
                raise RuntimeError(f"SHP export failed: {e}") from e
            try:
                gdf_export.to_file(gpkg_fp, layer="opensra_pipe_strain_rate", driver="GPKG")
            except Exception as e:
                raise RuntimeError(f"GPKG export failed: {e}") from e
            self.append_log(
                f"[OpenSRA Export] Exported {len(gdf_gpkg)} mesh elements to SHP={shp_fp}, GPKG={gpkg_fp} "
                f"(epoch={epoch_idx + 1}, mode={mode})."
            )
            QMessageBox.information(
                self,
                "Export Completed",
                f"OpenSRA files exported successfully.\n\nSHP: {shp_fp}\nGPKG: {gpkg_fp}",
            )
        except Exception as e:
            self.append_log(traceback.format_exc())
            QMessageBox.critical(self, "Export Failed", f"Failed to export OpenSRA files.\n\n{e}")

    def _open_abaqus_inp_params_dialog(self) -> Optional[Dict[str, Any]]:
        vals = dict(self._abaqus_inp_popup_values)
        if "model_type" not in vals:
            et_old = str(vals.get("element_type", "PIPE21")).upper()
            st_old = str(vals.get("ssi_type", "PSI")).upper()
            if st_old == "PSI":
                vals["model_type"] = "PIPE21+PSI"
            elif et_old in {"ELBOW31", "BEAM21"}:
                vals["model_type"] = "ELBOW31+SPRING"
            else:
                vals["model_type"] = "PIPE21+PSI"
        vals.setdefault("curve_source", "From SSI parameters")
        vals.setdefault("axial_curve_points", "")
        vals.setdefault("lateral_center_curve_points", "")
        vals.setdefault("lateral_outer_curve_points", "")
        dlg = QDialog(self)
        dlg.setWindowTitle("Export Abaqus Input File Parameters")
        lay = QVBoxLayout(dlg)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        editors: Dict[str, QWidget] = {}

        cmb_model = QComboBox()
        cmb_model.addItems(["PIPE21+PSI", "ELBOW31+SPRING"])
        model_cur = str(vals.get("model_type", "PIPE21+PSI"))
        if model_cur not in {"PIPE21+PSI", "ELBOW31+SPRING"}:
            model_cur = "PIPE21+PSI"
        cmb_model.setCurrentText(model_cur)
        form.addRow("model_type", cmb_model)
        editors["model_type"] = cmb_model

        cmb_curve = QComboBox()
        cmb_curve.addItems(["From SSI parameters", "From point lists"])
        curve_cur = str(vals.get("curve_source", "From SSI parameters"))
        if curve_cur not in {"From SSI parameters", "From point lists"}:
            curve_cur = "From SSI parameters"
        cmb_curve.setCurrentText(curve_cur)
        form.addRow("curve_source", cmb_curve)
        editors["curve_source"] = cmb_curve

        float_keys = [
            ("temperature_load", "temperature_load"),
            ("pipe_internal_pressure", "pipe_internal_pressure"),
            ("poisson_ratio", "poisson_ratio"),
        ]
        for k, label in float_keys:
            le = QLineEdit(str(vals.get(k, "")))
            form.addRow(label, le)
            editors[k] = le

        text_keys = [
            ("unit_system", "unit_system"),
            ("material_name", "material_name"),
            ("analysis_step_name", "analysis_step_name"),
        ]
        for k, label in text_keys:
            le = QLineEdit(str(vals.get(k, "")))
            form.addRow(label, le)
            editors[k] = le

        axial_pts = QPlainTextEdit(str(vals.get("axial_curve_points", "")))
        axial_pts.setPlaceholderText("axial curve points: force,disp per line")
        axial_pts.setFixedHeight(72)
        form.addRow("axial_curve_points", axial_pts)
        editors["axial_curve_points"] = axial_pts

        center_pts = QPlainTextEdit(str(vals.get("lateral_center_curve_points", "")))
        center_pts.setPlaceholderText("lateral center points: force,disp per line")
        center_pts.setFixedHeight(72)
        form.addRow("lateral_center_curve_points", center_pts)
        editors["lateral_center_curve_points"] = center_pts

        outer_pts = QPlainTextEdit(str(vals.get("lateral_outer_curve_points", "")))
        outer_pts.setPlaceholderText("lateral outer points: force,disp per line")
        outer_pts.setFixedHeight(72)
        form.addRow("lateral_outer_curve_points", outer_pts)
        editors["lateral_outer_curve_points"] = outer_pts

        def _sync_curve_inputs():
            use_points = cmb_curve.currentText() == "From point lists"
            axial_pts.setVisible(use_points)
            center_pts.setVisible(use_points)
            outer_pts.setVisible(use_points)
            form.labelForField(axial_pts).setVisible(use_points)
            form.labelForField(center_pts).setVisible(use_points)
            form.labelForField(outer_pts).setVisible(use_points)
            dlg.adjustSize()

        cmb_curve.currentIndexChanged.connect(lambda _i: _sync_curve_inputs())
        _sync_curve_inputs()

        lay.addLayout(form)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dlg)
        ok_btn = btns.button(QDialogButtonBox.Ok)
        if ok_btn is not None:
            ok_btn.setText("Export")
        lay.addWidget(btns)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)

        if dlg.exec() != QDialog.Accepted:
            return None

        try:
            out = {
                "model_type": cmb_model.currentText().strip(),
                "curve_source": cmb_curve.currentText().strip(),
                "temperature_load": float(editors["temperature_load"].text().strip()),
                "pipe_internal_pressure": float(editors["pipe_internal_pressure"].text().strip()),
                "poisson_ratio": float(editors["poisson_ratio"].text().strip()),
                "unit_system": str(editors["unit_system"].text().strip() or "kN-m"),
                "material_name": str(editors["material_name"].text().strip() or "GradeBSteel"),
                "analysis_step_name": str(editors["analysis_step_name"].text().strip() or "Step-1"),
                "axial_curve_points": str(axial_pts.toPlainText().strip()),
                "lateral_center_curve_points": str(center_pts.toPlainText().strip()),
                "lateral_outer_curve_points": str(outer_pts.toPlainText().strip()),
            }
        except Exception:
            self.append_log(traceback.format_exc())
            QMessageBox.critical(self, "Parameter Error", "Invalid Abaqus export parameters.")
            return None

        self._abaqus_inp_popup_values.update(out)
        return out

    def _parse_curve_points_text(self, text: str) -> List[tuple[float, float]]:
        pts: List[tuple[float, float]] = []
        for raw in str(text or "").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            line = line.replace(";", ",").replace("\t", ",")
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) != 2:
                raise ValueError(f"Invalid point line: '{raw}'. Use 'force,disp'.")
            f = float(parts[0])
            d = float(parts[1])
            if not np.isfinite(f) or not np.isfinite(d):
                raise ValueError(f"Non-finite point in line: '{raw}'.")
            pts.append((f, d))
        pts.sort(key=lambda x: x[1])
        return pts

    def _curve_from_ssi_params(
        self,
        *,
        e_soil_kN_m2: float,
        d_out_m: float,
        spacing_m: float,
        delta_y_m: float,
        delta_max_m: float,
        k2_ratio: float,
    ) -> List[tuple[float, float]]:
        k0 = max(float(e_soil_kN_m2), 1e-6) * max(float(d_out_m), 1e-6) * max(float(spacing_m), 1e-6)
        dy = max(float(delta_y_m), 1e-7)
        dm = max(float(delta_max_m), dy + 1e-7)
        k2 = max(float(k2_ratio), 0.0) * k0
        fy = k0 * dy
        fmax = fy + k2 * max(dm - dy, 0.0)
        d_far = max(10.0, dm * 5.0)
        return [
            (-fmax, -d_far),
            (-fmax, -dm),
            (-fy, -dy),
            (0.0, 0.0),
            (fy, dy),
            (fmax, dm),
            (fmax, d_far),
        ]

    def _axial_curve_from_ssi_params(self, *, spacing_m: float, d_out_m: float) -> List[tuple[float, float]]:
        k_axial = max(float(getattr(self.config, "ssi_k_axial_kN_m", 10000.0)), 1e-9)
        mu = max(float(getattr(self.config, "ssi_mu_friction", 0.678)), 0.0)
        gamma = max(float(getattr(self.config, "ssi_gamma_soil_kN_m3", 18.0)), 0.0)
        depth = max(float(getattr(self.config, "ssi_depth_burial_m", 1.5)), 0.0)
        n_over = gamma * depth * (np.pi * max(float(d_out_m), 1e-6)) * max(float(spacing_m), 1e-6)
        fcap = max(mu * n_over, 1e-9)
        dy = fcap / k_axial
        d_far = max(10.0, 5.0 * dy)
        return [
            (-fcap, -d_far),
            (-fcap, -dy),
            (0.0, 0.0),
            (fcap, dy),
            (fcap, d_far),
        ]

    def _append_curve_points(self, lines: List[str], pts: List[tuple[float, float]]):
        for f, d in pts:
            lines.append(f"{float(f):.9g}, {float(d):.9g}")

    def _append_elset_ids(self, lines: List[str], name: str, ids: List[int]):
        if not ids:
            return
        lines.append(f"*ELSET, ELSET={name}")
        for i in range(0, len(ids), 16):
            lines.append(", ".join(str(v) for v in ids[i : i + 16]))

    def _build_abaqus_inp_text(
        self,
        nodes_xy: np.ndarray,
        connectivity: np.ndarray,
        soil_fault_mm: np.ndarray,
        soil_xy_mm: Optional[np.ndarray],
        epoch_idx: int,
        params: Dict[str, Any],
        in_center_zone_nodes: Optional[np.ndarray] = None,
        fault_polyline_xy: Optional[np.ndarray] = None,
    ) -> str:
        nodes = np.asarray(nodes_xy, dtype=float)
        conn = np.asarray(connectivity, dtype=int)
        soil_fault_in = np.asarray(soil_fault_mm, dtype=float)
        soil_xy_in = np.asarray(soil_xy_mm, dtype=float) if soil_xy_mm is not None else np.empty((0, 2), dtype=float)
        if nodes.ndim != 2 or nodes.shape[0] < 2 or nodes.shape[1] < 2:
            raise ValueError("Invalid mesh nodes for Abaqus export.")
        if conn.ndim != 2 or conn.shape[1] != 2 or conn.shape[0] == 0:
            raise ValueError("Invalid mesh connectivity for Abaqus export.")

        n_nodes = int(nodes.shape[0])
        model_type = str(params.get("model_type", "PIPE21+PSI")).strip()
        if model_type not in {"PIPE21+PSI", "ELBOW31+SPRING"}:
            # backward compatibility
            et_old = str(params.get("element_type", "PIPE21")).upper()
            st_old = str(params.get("ssi_type", "PSI")).upper()
            if st_old == "PSI":
                model_type = "PIPE21+PSI"
            elif et_old in {"ELBOW31", "BEAM21"}:
                model_type = "ELBOW31+SPRING"
            else:
                model_type = "PIPE21+PSI"
        curve_source = str(params.get("curve_source", "From SSI parameters")).strip()
        if curve_source not in {"From SSI parameters", "From point lists"}:
            curve_source = "From SSI parameters"

        if model_type == "PIPE21+PSI":
            etype = "PIPE21"
            ssi_type = "PSI"
        else:
            etype = "ELBOW31"
            ssi_type = "spring"

        e_pipe = float(getattr(self.config, "ssi_E_pipe_kN_m2", 1000000.0))
        d_out = float(getattr(self.config, "ssi_D_out_m", 0.9144))
        t_wall = float(getattr(self.config, "ssi_t_wall_m", 0.08313))
        nu = float(params.get("poisson_ratio", 0.3))
        pressure = float(params.get("pipe_internal_pressure", 0.0))
        temp_load = float(params.get("temperature_load", 0.0))
        material_name = str(params.get("material_name", "GradeBSteel"))
        step_name = str(params.get("analysis_step_name", "Step-1"))
        unit_system = str(params.get("unit_system", "kN-m"))

        def _resize_vector_field_2d(arr: np.ndarray, target_n: int) -> np.ndarray:
            out = np.zeros((target_n, 2), dtype=float)
            a = np.asarray(arr, dtype=float)
            if a.ndim != 2 or a.shape[1] < 2 or target_n <= 0:
                return out
            m = min(target_n, a.shape[0])
            out[:m, :2] = a[:m, :2]
            if m < n_nodes and m > 1:
                x_src = np.linspace(0.0, 1.0, m)
                x_tgt = np.linspace(0.0, 1.0, target_n)
                out[:, 0] = np.interp(x_tgt, x_src, out[:m, 0])
                out[:, 1] = np.interp(x_tgt, x_src, out[:m, 1])
            return out

        soil_u_fault = _resize_vector_field_2d(soil_fault_in, n_nodes)
        soil_u_xy_direct = _resize_vector_field_2d(soil_xy_in, n_nodes)

        def _node_tangent_normal(path_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            pts = np.asarray(path_xy, dtype=float)
            tan = np.zeros((n_nodes, 2), dtype=float)
            tan[0] = pts[min(1, n_nodes - 1), :2] - pts[0, :2]
            tan[-1] = pts[-1, :2] - pts[max(n_nodes - 2, 0), :2]
            if n_nodes > 2:
                tan[1:-1] = pts[2:, :2] - pts[:-2, :2]
            tnorm = np.linalg.norm(tan, axis=1, keepdims=True)
            tnorm = np.where(tnorm <= 1e-12, 1.0, tnorm)
            tan = tan / tnorm
            nrm = np.column_stack([-tan[:, 1], tan[:, 0]])
            return tan, nrm

        pipe_tangent, pipe_normal = _node_tangent_normal(nodes[:, :2])
        x_shift = float(np.nanmin(nodes[:, 0])) if n_nodes > 0 else 0.0
        y_shift = float(np.nanmin(nodes[:, 1])) if n_nodes > 0 else 0.0
        nodes_local = np.asarray(nodes[:, :2], dtype=float).copy()
        nodes_local[:, 0] -= x_shift
        nodes_local[:, 1] -= y_shift
        fault_tangent = pipe_tangent.copy()
        fault_normal = pipe_normal.copy()
        fp = np.asarray(fault_polyline_xy if fault_polyline_xy is not None else [], dtype=float)
        if fp.ndim == 2 and fp.shape[0] >= 2 and fp.shape[1] >= 2:
            try:
                ft, fn = compute_unit_tangent_and_normal(np.asarray(nodes[:, :2], dtype=float), fp[:, :2])
                ft = np.asarray(ft, dtype=float)
                fn = np.asarray(fn, dtype=float)
                if ft.shape == pipe_tangent.shape and fn.shape == pipe_normal.shape:
                    fault_tangent = np.nan_to_num(ft, nan=0.0, posinf=0.0, neginf=0.0)
                    fault_normal = np.nan_to_num(fn, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception:
                pass

        # Convert fault-parallel/perpendicular components to global XY (reconstructed path).
        soil_u_xy_from_fault = np.column_stack(
            [
                soil_u_fault[:, 0] * fault_tangent[:, 0] + soil_u_fault[:, 1] * fault_normal[:, 0],
                soil_u_fault[:, 0] * fault_tangent[:, 1] + soil_u_fault[:, 1] * fault_normal[:, 1],
            ]
        )
        has_direct_xy = bool(
            soil_xy_in.ndim == 2 and soil_xy_in.shape[0] > 0 and soil_xy_in.shape[1] >= 2
        )
        if has_direct_xy:
            soil_u_xy = soil_u_xy_direct
        else:
            soil_u_xy = soil_u_xy_from_fault

        # Export-level diagnostic: compare direct-XY and reconstructed-fault paths.
        if has_direct_xy:
            dx = np.nan_to_num(soil_u_xy_direct[:, 0] - soil_u_xy_from_fault[:, 0], nan=0.0, posinf=0.0, neginf=0.0)
            dy = np.nan_to_num(soil_u_xy_direct[:, 1] - soil_u_xy_from_fault[:, 1], nan=0.0, posinf=0.0, neginf=0.0)
            ax_direct = np.sum(soil_u_xy_direct * pipe_tangent, axis=1)
            nm_direct = np.sum(soil_u_xy_direct * pipe_normal, axis=1)
            ax_fault = np.sum(soil_u_xy_from_fault * pipe_tangent, axis=1)
            nm_fault = np.sum(soil_u_xy_from_fault * pipe_normal, axis=1)
            dax = np.nan_to_num(ax_direct - ax_fault, nan=0.0, posinf=0.0, neginf=0.0)
            dnm = np.nan_to_num(nm_direct - nm_fault, nan=0.0, posinf=0.0, neginf=0.0)
            self.append_log(
                "[Abaqus Export] soil motion chain check: using direct soil_motion_xy_mm; "
                f"max|dX|={float(np.max(np.abs(dx))):.6g} mm, "
                f"max|dY|={float(np.max(np.abs(dy))):.6g} mm, "
                f"max|dU_axial|={float(np.max(np.abs(dax))):.6g} mm, "
                f"max|dU_normal|={float(np.max(np.abs(dnm))):.6g} mm."
            )
        else:
            self.append_log(
                "[Abaqus Export] soil_motion_xy_mm unavailable; fallback to "
                "soil_motion_fault_mm + fault-basis reconstruction."
            )

        # Then project global XY into pipe local axial/normal components.
        soil_pipe_ax_mm = np.sum(soil_u_xy * pipe_tangent, axis=1)
        soil_pipe_nm_mm = np.sum(soil_u_xy * pipe_normal, axis=1)
        soil_u_pipe_local_m = np.column_stack([soil_pipe_ax_mm, soil_pipe_nm_mm]) * 1e-3
        # For PSI/global DOF boundary writing, rotate local pipe components back to global XY.
        soil_u_global_m = (
            soil_u_pipe_local_m[:, 0:1] * pipe_tangent + soil_u_pipe_local_m[:, 1:2] * pipe_normal
        )

        spacing_mean = 1.0
        if conn.shape[0] > 0:
            seg_len = []
            for n0, n1 in conn:
                n0i = int(n0)
                n1i = int(n1)
                if 0 <= n0i < n_nodes and 0 <= n1i < n_nodes and n0i != n1i:
                    seg_len.append(float(np.linalg.norm(nodes[n1i, :2] - nodes[n0i, :2])))
            if seg_len:
                spacing_mean = max(float(np.nanmean(seg_len)), 1e-6)

        # Build nonlinear curves
        if curve_source == "From point lists":
            axial_curve = self._parse_curve_points_text(str(params.get("axial_curve_points", "")))
            lateral_center_curve = self._parse_curve_points_text(str(params.get("lateral_center_curve_points", "")))
            lateral_outer_curve = self._parse_curve_points_text(str(params.get("lateral_outer_curve_points", "")))
            if len(axial_curve) < 2 or len(lateral_center_curve) < 2 or len(lateral_outer_curve) < 2:
                raise ValueError("From point lists requires >=2 points for axial/center/outer curves.")
        else:
            axial_curve = self._axial_curve_from_ssi_params(spacing_m=spacing_mean, d_out_m=d_out)
            lateral_center_curve = self._curve_from_ssi_params(
                e_soil_kN_m2=float(getattr(self.config, "ssi_E_soil_center_kN_m2", 2000.0)),
                d_out_m=d_out,
                spacing_m=spacing_mean,
                delta_y_m=float(getattr(self.config, "ssi_delta_y_m", 0.001)),
                delta_max_m=float(getattr(self.config, "ssi_delta_max_m", 0.023)),
                k2_ratio=float(getattr(self.config, "ssi_k2_ratio_center", 0.325)),
            )
            lateral_outer_curve = self._curve_from_ssi_params(
                e_soil_kN_m2=float(getattr(self.config, "ssi_E_soil_outer_kN_m2", 10000.0)),
                d_out_m=d_out,
                spacing_m=spacing_mean,
                delta_y_m=float(getattr(self.config, "ssi_delta_y_m", 0.001)),
                delta_max_m=float(getattr(self.config, "ssi_delta_max_m", 0.023)),
                k2_ratio=float(getattr(self.config, "ssi_k2_ratio_outer", 0.325)),
            )

        lines: List[str] = []
        lines.append("*HEADING")
        lines.append(" InSAR Fault-Pipeline: Abaqus input exported from GUI")
        lines.append(f" ** Epoch index: {int(epoch_idx) + 1}")
        lines.append(f" ** Unit system: {unit_system}")
        lines.append(f" ** model_type: {model_type}")
        lines.append(f" ** curve_source: {curve_source}")
        lines.append("*PREPRINT, ECHO=NO, MODEL=YES, HISTORY=YES")

        lines.append("*NODE, NSET=PIPE_NODES")
        for i in range(n_nodes):
            lines.append(f"{i + 1}, {nodes_local[i, 0]:.3f}, {nodes_local[i, 1]:.3f}, 0.")
        lines.append("*NSET, NSET=PIPE_ENDS")
        lines.append(f"1, {n_nodes}")

        lines.append(f"*ELEMENT, TYPE={etype}, ELSET=PIPE_ELEMS")
        valid_conn: List[tuple[int, int, int]] = []
        for eidx, (n0, n1) in enumerate(conn, start=1):
            n0i = int(n0) + 1
            n1i = int(n1) + 1
            if 1 <= n0i <= n_nodes and 1 <= n1i <= n_nodes and n0i != n1i:
                lines.append(f"{eidx}, {n0i}, {n1i}")
                valid_conn.append((eidx, n0i, n1i))

        lines.append(f"*MATERIAL, NAME={material_name}")
        lines.append("*ELASTIC")
        lines.append(f"{e_pipe:.9g}, {nu:.9g}")
        if etype == "ELBOW31":
            lines.append(f"*BEAM SECTION, ELSET=PIPE_ELEMS, SECTION=ELBOW, MATERIAL={material_name}")
            lines.append(f"{0.5 * d_out:.9g}, {t_wall:.9g}, 0")
            lines.append("0,50,50")
            lines.append("5,12,4")
        else:
            lines.append(f"*BEAM SECTION, SECTION=PIPE, ELSET=PIPE_ELEMS, MATERIAL={material_name}")
            lines.append(f"{0.5 * d_out:.9g}, {t_wall:.9g}")
            lines.append("0., 0., -1.")

        soil_base = 100000
        soil_nodes_xy = np.asarray(nodes[:, :2], dtype=float).copy()
        if ssi_type == "PSI" and n_nodes >= 2:
            offset_m = max(2.0 * max(float(d_out), 1e-6), 1.0)
            soil_nodes_xy = soil_nodes_xy + pipe_normal * offset_m
        soil_nodes_local = np.asarray(soil_nodes_xy, dtype=float).copy()
        soil_nodes_local[:, 0] -= x_shift
        soil_nodes_local[:, 1] -= y_shift
        lines.append("*NODE, NSET=SOIL_REF_NODES")
        for i in range(n_nodes):
            sid = soil_base + i + 1
            lines.append(f"{sid}, {soil_nodes_local[i, 0]:.3f}, {soil_nodes_local[i, 1]:.3f}, 0.")

        if ssi_type == "spring":
            ax_base = 500000
            nm_base = 600000
            lines.append("*ELEMENT, TYPE=SPRING2, ELSET=SSI_AXIAL")
            for i in range(n_nodes):
                lines.append(f"{ax_base + i + 1}, {i + 1}, {soil_base + i + 1}")
            lines.append("*SPRING, ELSET=SSI_AXIAL, NONLINEAR")
            lines.append("1,1")
            self._append_curve_points(lines, axial_curve)

            lines.append("*ELEMENT, TYPE=SPRING2, ELSET=SSI_NORMAL")
            for i in range(n_nodes):
                lines.append(f"{nm_base + i + 1}, {i + 1}, {soil_base + i + 1}")
            lines.append("*SPRING, ELSET=SSI_NORMAL, NONLINEAR")
            lines.append("2,2")
            # spring model uses a single normal set; prefer center curve
            self._append_curve_points(lines, lateral_center_curve)

            # Define node-wise local transforms so spring U1/U2 align with
            # local tangent/normal directions along curved pipe geometry.
            tangent_loc = pipe_tangent
            normal_loc = pipe_normal
            lines.append("** Node-wise local transforms: U1=tangent, U2=normal")
            for i in range(n_nodes):
                pid = i + 1
                sid = soil_base + i + 1
                px = float(nodes_local[i, 0])
                py = float(nodes_local[i, 1])
                tx = float(tangent_loc[i, 0])
                ty = float(tangent_loc[i, 1])
                nx = float(normal_loc[i, 0])
                ny = float(normal_loc[i, 1])
                lines.append(f"*NSET, NSET=TR_SPR_{pid}")
                lines.append(f"{pid}, {sid}")
                lines.append(f"*TRANSFORM, NSET=TR_SPR_{pid}, TYPE=R")
                lines.append(f"{px + tx:.3f}, {py + ty:.3f}, 0., {px + nx:.3f}, {py + ny:.3f}, 0.")
        else:
            psi_base = 700000
            center_ids: List[int] = []
            outer_ids: List[int] = []
            zone_nodes = np.asarray(in_center_zone_nodes, dtype=bool).reshape(-1) if in_center_zone_nodes is not None else np.zeros(n_nodes, dtype=bool)
            if zone_nodes.size != n_nodes:
                zone_nodes = np.zeros(n_nodes, dtype=bool)

            lines.append("*ELEMENT, TYPE=PSI24, ELSET=PSI_SET")
            for local_idx, (_eid, n0i, n1i) in enumerate(valid_conn, start=1):
                s0i = soil_base + n0i
                s1i = soil_base + n1i
                eid = psi_base + local_idx
                lines.append(f"{eid}, {n0i}, {n1i}, {s1i}, {s0i}")
                is_center = bool(zone_nodes[n0i - 1] and zone_nodes[n1i - 1])
                if is_center:
                    center_ids.append(eid)
                else:
                    outer_ids.append(eid)

            self._append_elset_ids(lines, "PSI_CENTER", center_ids)
            self._append_elset_ids(lines, "PSI_OUTER", outer_ids)

            def _append_psi_block(elset_name: str, lat_curve: List[tuple[float, float]]):
                lines.append(f"*PIPE-SOIL INTERACTION, ELSET={elset_name}")
                lines.append("*PIPE-SOIL STIFFNESS, TYPE=NONLINEAR, DIR=AXIAL")
                self._append_curve_points(lines, axial_curve)
                lines.append("*PIPE-SOIL STIFFNESS, TYPE=NONLINEAR, DIR=VERTICAL")
                self._append_curve_points(lines, lat_curve)
                lines.append("*PIPE-SOIL STIFFNESS, TYPE=NONLINEAR, DIR=HORIZONTAL")
                self._append_curve_points(lines, lat_curve)

            if center_ids:
                _append_psi_block("PSI_CENTER", lateral_center_curve)
            if outer_ids:
                _append_psi_block("PSI_OUTER", lateral_outer_curve)
            if (not center_ids) and (not outer_ids):
                _append_psi_block("PSI_SET", lateral_center_curve)

        lines.append(f"*STEP, NAME={step_name}, NLGEOM=NO")
        lines.append("*STATIC")
        lines.append("1., 1., 1e-05, 1.")
        lines.append("*BOUNDARY")
        lines.append("1, 1, 6, 0.")
        lines.append(f"{n_nodes}, 1, 6, 0.")
        lines.append("PIPE_NODES, 4, 5, 0.")
        if ssi_type == "PSI":
            lines.append("SOIL_REF_NODES, 3, 6, 0.")
        for i in range(n_nodes):
            sid = soil_base + i + 1
            if ssi_type == "spring":
                ut = float(soil_u_pipe_local_m[i, 0])
                un = float(soil_u_pipe_local_m[i, 1])
            else:
                ut = float(soil_u_global_m[i, 0])
                un = float(soil_u_global_m[i, 1])
            lines.append(f"{sid}, 1, 1, {ut:.9g}")
            lines.append(f"{sid}, 2, 2, {un:.9g}")

        if np.isfinite(pressure) and abs(pressure) > 0.0:
            lines.append("*DLOAD")
            lines.append(f"PIPE_ELEMS, PI, {pressure:.9g}")
        if np.isfinite(temp_load) and abs(temp_load) > 0.0:
            lines.append("*TEMPERATURE")
            lines.append(f"PIPE_NODES, {temp_load:.9g}")

        lines.append("*OUTPUT, FIELD, VARIABLE=PRESELECT")
        lines.append("*NODE OUTPUT, NSET=PIPE_NODES")
        lines.append("U, RF, COORD")
        if etype == "ELBOW31":
            lines.append("*ELEMENT OUTPUT, ELSET=PIPE_ELEMS")
            lines.append("E, S")
        lines.append("*END STEP")
        lines.append("")
        return "\n".join(lines)

    def on_export_abaqus_inp(self):
        if not self._is_ssi_enabled_from_ui():
            QMessageBox.information(self, "Info", "Enable SSI first to export Abaqus input.")
            return
        if not isinstance(self.last_result, dict):
            QMessageBox.information(self, "Info", "Please run Step 3/4 first.")
            return

        result = self.last_result
        pipe = result.get("pipe", {}) if isinstance(result.get("pipe", {}), dict) else {}
        nodes = np.asarray(pipe.get("pipe_axis_nodes", []), dtype=float)
        conn = np.asarray(pipe.get("pipe_connectivity", []), dtype=int)
        if nodes.ndim != 2 or nodes.shape[0] < 2 or conn.ndim != 2 or conn.shape[1] != 2 or conn.shape[0] == 0:
            mesh = self._build_pipeline_mesh(result)
            nodes = np.asarray(mesh.get("points", []), dtype=float)
            conn = np.asarray(mesh.get("connectivity", []), dtype=int)
        if nodes.ndim != 2 or nodes.shape[0] < 2 or conn.ndim != 2 or conn.shape[1] != 2 or conn.shape[0] == 0:
            QMessageBox.information(self, "Info", "No valid pipeline mesh available for Abaqus export.")
            return

        epoch_idx = self._selected_epoch_index(result)
        soil_fault_all = np.asarray(pipe.get("soil_motion_fault_mm", []), dtype=float)
        if soil_fault_all.ndim != 3:
            soil_fault_all = np.asarray(result.get("arrays", {}).get("soil_motion_fault_mm", []), dtype=float)
        soil_xy_all = np.asarray(pipe.get("soil_motion_xy_mm", []), dtype=float)
        if soil_xy_all.ndim != 3:
            soil_xy_all = np.asarray(result.get("arrays", {}).get("soil_motion_xy_mm", []), dtype=float)

        epoch_counts = []
        if soil_fault_all.ndim == 3 and soil_fault_all.shape[0] > 0:
            epoch_counts.append(int(soil_fault_all.shape[0]))
        if soil_xy_all.ndim == 3 and soil_xy_all.shape[0] > 0:
            epoch_counts.append(int(soil_xy_all.shape[0]))
        if epoch_counts:
            max_epoch_count = max(epoch_counts)
            i_sel = int(np.clip(epoch_idx if epoch_idx >= 0 else max_epoch_count - 1, 0, max_epoch_count - 1))
        else:
            i_sel = 0

        def _slice_epoch_xy(arr3: np.ndarray, i_target: int) -> np.ndarray:
            if arr3.ndim == 3 and arr3.shape[0] > 0 and arr3.shape[2] >= 2:
                ii = int(np.clip(i_target, 0, arr3.shape[0] - 1))
                return np.asarray(arr3[ii, :, :2], dtype=float)
            return np.zeros((nodes.shape[0], 2), dtype=float)

        soil_epoch_fault = _slice_epoch_xy(soil_fault_all, i_sel)
        soil_epoch_xy = _slice_epoch_xy(soil_xy_all, i_sel)
        zone_all = np.asarray(pipe.get("ssi_zone_by_epoch", []), dtype=bool)
        if zone_all.ndim == 2 and zone_all.shape[0] > 0:
            z_sel = int(np.clip(i_sel, 0, zone_all.shape[0] - 1))
            zone_epoch = np.asarray(zone_all[z_sel], dtype=bool).reshape(-1)
        else:
            zone_epoch = np.zeros(nodes.shape[0], dtype=bool)
        fault_polyline_xy = np.array([], dtype=float)
        nb_ref = result.get("notebook_fault_ref", {}) if isinstance(result, dict) else {}
        fl2 = np.asarray(nb_ref.get("fl2", []), dtype=float) if isinstance(nb_ref, dict) else np.array([], dtype=float)
        if fl2.ndim == 2 and fl2.shape[0] >= 2 and fl2.shape[1] >= 2:
            fault_polyline_xy = fl2[:, :2]
        else:
            proj = np.asarray(result.get("fault_coords_for_projection", []), dtype=float)
            if proj.ndim == 2 and proj.shape[0] >= 2 and proj.shape[1] >= 2:
                fault_polyline_xy = proj[:, :2]

        params = self._open_abaqus_inp_params_dialog()
        if params is None:
            return

        try:
            inp_text = self._build_abaqus_inp_text(
                nodes_xy=nodes[:, :2],
                connectivity=conn,
                soil_fault_mm=soil_epoch_fault,
                soil_xy_mm=soil_epoch_xy,
                epoch_idx=i_sel,
                params=params,
                in_center_zone_nodes=zone_epoch,
                fault_polyline_xy=fault_polyline_xy,
            )
        except Exception as e:
            self.append_log(traceback.format_exc())
            QMessageBox.critical(self, "Export Failed", f"Failed to build Abaqus input text: {e}")
            return

        out_dir = Path(self.config.output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        model_slug = str(params.get("model_type", "PIPE21+PSI")).replace("+", "_").lower()
        suggested = out_dir / f"abaqus_export_epoch_{i_sel + 1}_{model_slug}.inp"
        fp, _ = QFileDialog.getSaveFileName(
            self,
            "Export Abaqus Input File",
            str(suggested),
            "Abaqus Input (*.inp)",
        )
        if not fp:
            return
        if Path(fp).suffix.lower() != ".inp":
            fp = f"{fp}.inp"

        try:
            Path(fp).write_text(inp_text, encoding="utf-8")
            self.append_log(
                f"[Abaqus Export] Exported INP file: {fp} "
                f"(epoch={i_sel + 1}, model={params.get('model_type')}, curve_source={params.get('curve_source')})"
            )
            QMessageBox.information(self, "Export Completed", fp)
        except Exception:
            self.append_log(traceback.format_exc())
            QMessageBox.critical(self, "Export Failed", "Failed to write Abaqus input file.")

    def on_open_output(self):
        out = Path(self.config.output_dir).expanduser().resolve()
        if not out.exists():
            QMessageBox.warning(self, "Info", f"Directory does not exist: {out}")
            return
        if os.name == "nt":
            os.startfile(str(out))
        elif os.uname().sysname == "Darwin":
            os.system(f'open "{out}"')
        else:
            os.system(f'xdg-open "{out}"')
