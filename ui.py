# -*- coding: utf-8 -*-
"""ui.py

PyQt6 tabanlı "Orbit PINN Dashboard".

Amaç:
- Dataset üret (generate_dataset.py)  [şimdilik CLI]
- PINN eğit (train_pinn.py)          [library-style: train(cfg)]
- Model değerlendir (evaluate_pinn.py) [library-style: evaluate(cfg)]

Not:
- Script'ler QProcess ile ayrı süreçte çalışır (UI donmaz).
- Train/Eval için artık CLI argümanları yok: UI geçici JSON config yazar,
  subprocess python -c ile modülü import edip fonksiyonu çağırır.
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
from pathlib import Path
from typing import Optional, Callable, Dict, Any

from PyQt6.QtCore import Qt, QProcess, QTimer, QProcessEnvironment
from PyQt6.QtGui import QFont, QPalette, QColor, QGuiApplication
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QProgressBar,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR / "_ui_runs"
RUNS_DIR.mkdir(exist_ok=True)


# =============================================================================
# Small helpers
# =============================================================================

def _norm_path(p: str) -> str:
    return str(Path(p).expanduser().resolve()) if p else ""


def _mono_font() -> QFont:
    f = QFont("Consolas")
    if not f.exactMatch():
        f = QFont("Courier New")
    f.setPointSize(10)
    return f


def _tune_form(form: QFormLayout) -> None:
    form.setContentsMargins(18, 18, 18, 18)
    form.setHorizontalSpacing(18)
    form.setVerticalSpacing(14)
    form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
    form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)


def _tune_inputs(root: QWidget, h: int = 40) -> None:
    for w in root.findChildren((QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox)):
        w.setMinimumHeight(h)
        w.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    for sb in root.findChildren((QSpinBox, QDoubleSpinBox)):
        sb.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
        sb.setCorrectionMode(QAbstractSpinBox.CorrectionMode.CorrectToNearestValue)


def _row_lineedit_with_button(edit: QLineEdit, button: QPushButton) -> QWidget:
    edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    button.setMinimumHeight(edit.minimumHeight())

    wrap = QWidget()
    h = QHBoxLayout()
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(10)
    h.addWidget(edit, 1)
    h.addWidget(button, 0)
    wrap.setLayout(h)
    return wrap


def _scroll_wrap(widget: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QScrollArea.Shape.NoFrame)
    area.setWidget(widget)
    return area


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _python_call_train_with_json(cfg_json_path: str) -> str:
    # JSON -> TrainConfig(**filtered) -> train(cfg)
    return (
        "import json, dataclasses\n"
        "from train_pinn import TrainConfig, train\n"
        f"p={cfg_json_path!r}\n"
        "d=json.load(open(p,'r',encoding='utf-8'))\n"
        "fields={f.name for f in dataclasses.fields(TrainConfig)}\n"
        "cfg=TrainConfig(**{k:v for k,v in d.items() if k in fields})\n"
        "train(cfg)\n"
    )


def _python_call_eval_with_json(cfg_json_path: str) -> str:
    # JSON -> EvalConfig(**filtered) -> evaluate(cfg)
    return (
        "import json, dataclasses\n"
        "from evaluate_pinn import EvalConfig, evaluate\n"
        f"p={cfg_json_path!r}\n"
        "d=json.load(open(p,'r',encoding='utf-8'))\n"
        "fields={f.name for f in dataclasses.fields(EvalConfig)}\n"
        "cfg=EvalConfig(**{k:v for k,v in d.items() if k in fields})\n"
        "evaluate(cfg)\n"
    )


# =============================================================================
# Process Pane
# =============================================================================

class ProcessPane(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.proc: Optional[QProcess] = None
        self._on_parse_progress: Optional[Callable[[str], None]] = None

        self.status = QLabel("Hazır")
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(_mono_font())

        self.btn_start = QPushButton("Başlat")
        self.btn_stop = QPushButton("Durdur")
        self.btn_clear = QPushButton("Log temizle")

        self.btn_start.setProperty("kind", "primary")
        self.btn_stop.setProperty("kind", "danger")
        self.btn_clear.setProperty("kind", "ghost")

        self.btn_stop.setEnabled(False)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(10)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_clear)

        layout = QVBoxLayout()
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        layout.addWidget(self.status)
        layout.addWidget(self.progress)
        layout.addLayout(btn_row)
        layout.addWidget(self.log, 1)
        self.setLayout(layout)

        self.btn_clear.clicked.connect(self.log.clear)
        self.btn_stop.clicked.connect(self.stop)

    def set_progress_parser(self, fn: Optional[Callable[[str], None]]) -> None:
        self._on_parse_progress = fn

    def append(self, text: str) -> None:
        self.log.appendPlainText(text.rstrip("\n"))
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())
        if self._on_parse_progress:
            try:
                self._on_parse_progress(text)
            except Exception:
                pass

    def start(self, program: str, args: list[str], workdir: Optional[str] = None) -> None:
        if self.proc and self.proc.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.warning(self, "Çalışıyor", "Zaten bir süreç çalışıyor. Önce durdurun.")
            return

        self.log.clear()
        self.progress.setValue(0)

        self.proc = QProcess(self)

        # Unbuffered stdout/stderr so logs & progress arrive instantly
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        self.proc.setProcessEnvironment(env)

        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        if workdir:
            self.proc.setWorkingDirectory(workdir)

        self.append("> " + " ".join([program] + args) + "\n")

        self.status.setText("Çalışıyor...")
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self.proc.readyReadStandardOutput.connect(self._on_ready_read)
        self.proc.finished.connect(self._on_finished)

        self.proc.setProgram(program)
        self.proc.setArguments(args)
        self.proc.start()

        if not self.proc.waitForStarted(3000):
            self.append("[HATA] Süreç başlatılamadı. Python yolu / bağımlılıklar kontrol edin.")
            self.status.setText("Hata")
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)

    def stop(self) -> None:
        if not self.proc or self.proc.state() == QProcess.ProcessState.NotRunning:
            return

        self.append("\n[UI] Durdurma istendi...\n")
        self.status.setText("Durduruluyor...")
        self.proc.terminate()

        def kill_if_needed() -> None:
            if self.proc and self.proc.state() != QProcess.ProcessState.NotRunning:
                self.append("[UI] Zorla sonlandırılıyor (kill).\n")
                self.proc.kill()

        QTimer.singleShot(2000, kill_if_needed)

    def _on_ready_read(self) -> None:
        if not self.proc:
            return
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="ignore")
        if data:
            for line in data.splitlines():
                self.append(line)

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        st = "Bitti" if exit_status == QProcess.ExitStatus.NormalExit else "Çöktü"
        self.status.setText(f"{st} | exit_code={exit_code}")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

        if exit_status == QProcess.ExitStatus.NormalExit:
            try:
                self.progress.setValue(self.progress.maximum())
            except Exception:
                pass


# =============================================================================
# Dataset Tab (şimdilik CLI)
# =============================================================================

class DatasetTab(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        form = QFormLayout()
        _tune_form(form)

        self.out_dir = QLineEdit(str(SCRIPT_DIR / "dataset"))
        btn_out = QPushButton("Seç")
        btn_out.clicked.connect(self._pick_out_dir)
        out_row = _row_lineedit_with_button(self.out_dir, btn_out)

        self.name = QLineEdit("orbit_j2_dataset_v2")

        self.num_orbits = QSpinBox(); self.num_orbits.setRange(1, 2_000_000); self.num_orbits.setValue(1000)
        self.duration = QDoubleSpinBox(); self.duration.setDecimals(1); self.duration.setRange(1.0, 365.0 * 86400.0); self.duration.setValue(86400.0)
        self.dt = QDoubleSpinBox(); self.dt.setDecimals(1); self.dt.setRange(0.1, 86400.0); self.dt.setValue(600.0)

        self.method = QComboBox()
        self.method.addItems(["DOP853", "RK45", "Radau", "BDF"])
        self.method.setCurrentText("DOP853")

        self.max_step = QDoubleSpinBox(); self.max_step.setDecimals(3); self.max_step.setRange(0.0, 86400.0); self.max_step.setValue(0.0)

        self.rtol = QDoubleSpinBox(); self.rtol.setDecimals(12); self.rtol.setRange(1e-16, 1.0); self.rtol.setValue(1e-9)
        self.atol = QDoubleSpinBox(); self.atol.setDecimals(12); self.atol.setRange(1e-16, 1.0); self.atol.setValue(1e-9)
        self.seed = QSpinBox(); self.seed.setRange(0, 2_147_483_647); self.seed.setValue(42)

        self.include_end = QCheckBox("t_eval içine duration'ı ekle (include_end)")
        self.include_end.setChecked(True)

        form.addRow("Output klasörü", out_row)
        form.addRow("Dataset adı", self.name)
        form.addRow("Orbit sayısı", self.num_orbits)
        form.addRow("Sim süre [s]", self.duration)
        form.addRow("Örnekleme dt [s]", self.dt)
        form.addRow("solve_ivp method", self.method)
        form.addRow("max_step [s] (0=default)", self.max_step)
        form.addRow("RTOL", self.rtol)
        form.addRow("ATOL", self.atol)
        form.addRow("Seed", self.seed)
        form.addRow(self.include_end)

        box = QGroupBox("Dataset Üretimi")
        box.setLayout(form)
        _tune_inputs(box)

        self.runner = ProcessPane()
        self.runner.btn_start.setText("Dataset üret")
        self.runner.btn_start.clicked.connect(self._start)
        self.runner.set_progress_parser(self._parse_progress)

        top = QWidget()
        top_l = QVBoxLayout(); top_l.setContentsMargins(0, 0, 0, 0); top_l.setSpacing(0)
        top_l.addWidget(box)
        top.setLayout(top_l)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(_scroll_wrap(top))
        splitter.addWidget(self.runner)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 520])

        layout = QVBoxLayout()
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)
        layout.addWidget(splitter, 1)
        self.setLayout(layout)

    def _pick_out_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Output klasörü seç", self.out_dir.text() or str(SCRIPT_DIR))
        if d:
            self.out_dir.setText(_norm_path(d))

    def _start(self) -> None:
        script = str(SCRIPT_DIR / "generate_dataset.py")
        if not Path(script).exists():
            QMessageBox.critical(self, "Bulunamadı", "generate_dataset.py aynı klasörde olmalı.")
            return

        args = [
            "-u",
            script,
            "--num-orbits", str(self.num_orbits.value()),
            "--duration", str(self.duration.value()),
            "--dt", str(self.dt.value()),
            "--out", self.out_dir.text().strip(),
            "--name", self.name.text().strip(),
            "--seed", str(self.seed.value()),
            "--rtol", str(self.rtol.value()),
            "--atol", str(self.atol.value()),
            "--method", self.method.currentText().strip(),
        ]
        if self.max_step.value() > 0.0:
            args += ["--max-step", str(self.max_step.value())]
        if self.include_end.isChecked():
            args += ["--include-end"]

        self.runner.progress.setRange(0, self.num_orbits.value())
        self.runner.start(sys.executable, args, workdir=str(SCRIPT_DIR))

    def _parse_progress(self, line: str) -> None:
        m = re.search(r"\.\.\s*(\d+)\/(\d+)", line)
        if m:
            done = int(m.group(1)); total = int(m.group(2))
            self.runner.progress.setRange(0, max(1, total))
            self.runner.progress.setValue(min(done, total))


# =============================================================================
# Train Tab (library-style)
# =============================================================================

class TrainTab(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        form = QFormLayout()
        _tune_form(form)

        self.csv = QLineEdit(str(SCRIPT_DIR / "dataset" / "orbit_j2_dataset_v2.csv"))
        btn_csv = QPushButton("Seç")
        btn_csv.clicked.connect(self._pick_csv)
        csv_row = _row_lineedit_with_button(self.csv, btn_csv)

        self.out_dir = QLineEdit(str(SCRIPT_DIR / "checkpoints"))
        btn_out = QPushButton("Seç")
        btn_out.clicked.connect(self._pick_out_dir)
        out_row = _row_lineedit_with_button(self.out_dir, btn_out)

        self.ckpt_name = QLineEdit("orbit_pinn.pt")

        # --- Core hyperparams ---
        self.epochs = QSpinBox(); self.epochs.setRange(1, 5_000_000); self.epochs.setValue(2000)
        self.batch = QSpinBox(); self.batch.setRange(1, 10_000_000); self.batch.setValue(2048)
        self.lr = QDoubleSpinBox(); self.lr.setDecimals(10); self.lr.setRange(1e-8, 10.0); self.lr.setValue(3e-4)
        self.w_data = QDoubleSpinBox(); self.w_data.setDecimals(6); self.w_data.setRange(0.0, 1e6); self.w_data.setValue(1.0)
        self.w_phys = QDoubleSpinBox(); self.w_phys.setDecimals(6); self.w_phys.setRange(0.0, 1e6); self.w_phys.setValue(0.1)
        self.warmup = QSpinBox(); self.warmup.setRange(0, 5_000_000); self.warmup.setValue(0)

        self.hidden = QSpinBox(); self.hidden.setRange(1, 8192); self.hidden.setValue(128)
        self.depth = QSpinBox(); self.depth.setRange(1, 64); self.depth.setValue(3)
        self.seed = QSpinBox(); self.seed.setRange(0, 2_147_483_647); self.seed.setValue(42)
        self.dtype = QComboBox(); self.dtype.addItems(["float32", "float64"]); self.dtype.setCurrentText("float64")

        self.arch = QComboBox()
        self.arch.addItems([
            "deeponet_phase",
            "deeponet_fourier",
            "deeponet",
            "phase_mlp",
            "phase_siren",
            "fourier_mlp",
            "fourier_siren",
            "mlp",
            "siren",
        ])
        self.arch.setCurrentText("deeponet_phase")

        # --- Fourier ---
        self.fourier_features = QSpinBox(); self.fourier_features.setRange(1, 4096); self.fourier_features.setValue(24)
        self.fourier_min_freq = QDoubleSpinBox(); self.fourier_min_freq.setDecimals(6); self.fourier_min_freq.setRange(1e-6, 1e3); self.fourier_min_freq.setValue(0.005)
        self.fourier_max_freq = QDoubleSpinBox(); self.fourier_max_freq.setDecimals(6); self.fourier_max_freq.setRange(1e-6, 1e6); self.fourier_max_freq.setValue(2.0)
        self.fourier_include_input = QCheckBox("Fourier: ham t'yi de ekle (include_input)")
        self.fourier_include_input.setChecked(True)
        self.fourier_linear = QCheckBox("Fourier: frekansları linear dağıt (log yerine)")
        self.fourier_linear.setChecked(False)

        # --- Phase ---
        self.phase_harmonics = QSpinBox(); self.phase_harmonics.setRange(1, 128); self.phase_harmonics.setValue(8)
        self.phase_include_t = QCheckBox("Phase: t'yi de ekle")
        self.phase_include_t.setChecked(True)
        self.phase_include_phase = QCheckBox("Phase: faz (n0*t) ham değerini ekle")
        self.phase_include_phase.setChecked(False)

        # --- DeepONet ---
        self.deeponet_latent = QSpinBox(); self.deeponet_latent.setRange(8, 1024); self.deeponet_latent.setValue(64)
        self.deeponet_branch_aug = QCheckBox("DeepONet: branch'e IC augment ekle")
        self.deeponet_branch_aug.setChecked(True)

        # --- SIREN ---
        self.siren_w0_initial = QDoubleSpinBox(); self.siren_w0_initial.setDecimals(3); self.siren_w0_initial.setRange(0.01, 1e4); self.siren_w0_initial.setValue(30.0)
        self.siren_w0 = QDoubleSpinBox(); self.siren_w0.setDecimals(3); self.siren_w0.setRange(0.001, 1e4); self.siren_w0.setValue(1.0)

        # --- Loss/constraints (new) ---
        self.hard_constraint = QCheckBox("Hard constraint (IC/BC) kullan")
        self.hard_constraint.setChecked(True)

        self.use_energy_loss = QCheckBox("Energy loss aktif")
        self.use_energy_loss.setChecked(True)
        self.energy_weight = QDoubleSpinBox(); self.energy_weight.setDecimals(6); self.energy_weight.setRange(0.0, 1e6); self.energy_weight.setValue(1.0)

        self.use_deriv_loss = QCheckBox("Derivative data loss (dr/dt ~ v_data) aktif")
        self.use_deriv_loss.setChecked(True)
        self.alpha_v = QDoubleSpinBox(); self.alpha_v.setDecimals(6); self.alpha_v.setRange(0.0, 1e6); self.alpha_v.setValue(1.0)

        def _sync_loss_toggles() -> None:
            self.energy_weight.setEnabled(self.use_energy_loss.isChecked())
            self.alpha_v.setEnabled(self.use_deriv_loss.isChecked())

        self.use_energy_loss.stateChanged.connect(_sync_loss_toggles)
        self.use_deriv_loss.stateChanged.connect(_sync_loss_toggles)
        _sync_loss_toggles()

        # --- Adaptive weights (new) ---
        self.adaptive_weighting = QCheckBox("Adaptive weighting (GradNorm-lite) aktif")
        self.adaptive_weighting.setChecked(True)
        self.aw_update_every = QSpinBox(); self.aw_update_every.setRange(1, 1_000_000); self.aw_update_every.setValue(1)
        self.aw_beta = QDoubleSpinBox(); self.aw_beta.setDecimals(4); self.aw_beta.setRange(0.0, 10.0); self.aw_beta.setValue(0.5)
        self.aw_ema = QDoubleSpinBox(); self.aw_ema.setDecimals(4); self.aw_ema.setRange(0.0, 0.9999); self.aw_ema.setValue(0.9)
        self.aw_min = QDoubleSpinBox(); self.aw_min.setDecimals(8); self.aw_min.setRange(0.0, 1e6); self.aw_min.setValue(1e-4)
        self.aw_max = QDoubleSpinBox(); self.aw_max.setDecimals(2); self.aw_max.setRange(1.0, 1e12); self.aw_max.setValue(1e4)

        # --- Curriculum (new) ---
        self.curriculum = QCheckBox("Curriculum learning aktif (zaman pencereli)")
        self.curriculum.setChecked(True)
        self.curr_e1 = QSpinBox(); self.curr_e1.setRange(0, 10_000_000); self.curr_e1.setValue(500)
        self.curr_e2 = QSpinBox(); self.curr_e2.setRange(0, 10_000_000); self.curr_e2.setValue(1000)
        self.curr_t1_s = QDoubleSpinBox(); self.curr_t1_s.setDecimals(1); self.curr_t1_s.setRange(1.0, 365.0*86400.0); self.curr_t1_s.setValue(3600.0)
        self.curr_t2_s = QDoubleSpinBox(); self.curr_t2_s.setDecimals(1); self.curr_t2_s.setRange(1.0, 365.0*86400.0); self.curr_t2_s.setValue(21600.0)
        self.curr_t3_s = QDoubleSpinBox(); self.curr_t3_s.setDecimals(1); self.curr_t3_s.setRange(1.0, 365.0*86400.0); self.curr_t3_s.setValue(86400.0)

        # --- L-BFGS (new) ---
        self.use_lbfgs = QCheckBox("L-BFGS fine-tune kullan")
        self.use_lbfgs.setChecked(True)
        self.lbfgs_fraction = QDoubleSpinBox(); self.lbfgs_fraction.setDecimals(3); self.lbfgs_fraction.setRange(0.0, 1.0); self.lbfgs_fraction.setValue(0.2)
        self.lbfgs_max_iter = QSpinBox(); self.lbfgs_max_iter.setRange(0, 1_000_000); self.lbfgs_max_iter.setValue(200)
        self.lbfgs_history_size = QSpinBox(); self.lbfgs_history_size.setRange(1, 10_000); self.lbfgs_history_size.setValue(50)
        self.lbfgs_lr = QDoubleSpinBox(); self.lbfgs_lr.setDecimals(6); self.lbfgs_lr.setRange(1e-6, 100.0); self.lbfgs_lr.setValue(1.0)
        self.lbfgs_line_search = QComboBox(); self.lbfgs_line_search.addItems(["strong_wolfe", "None"]); self.lbfgs_line_search.setCurrentText("strong_wolfe")
        self.lbfgs_sample_size = QSpinBox(); self.lbfgs_sample_size.setRange(256, 10_000_000); self.lbfgs_sample_size.setValue(8192)

        # --- Loader/perf ---
        self.num_workers = QSpinBox(); self.num_workers.setRange(0, 64); self.num_workers.setValue(0)

        # --- Optional advanced (existing) ---
        self.val_split = QDoubleSpinBox(); self.val_split.setDecimals(4); self.val_split.setRange(0.0, 0.5); self.val_split.setValue(0.0)
        self.val_batches = QSpinBox(); self.val_batches.setRange(1, 100_000); self.val_batches.setValue(16)
        self.grad_clip = QDoubleSpinBox(); self.grad_clip.setDecimals(4); self.grad_clip.setRange(0.0, 1e6); self.grad_clip.setValue(0.0)
        self.log_every = QSpinBox(); self.log_every.setRange(1, 100_000); self.log_every.setValue(10)
        self.save_every = QSpinBox(); self.save_every.setRange(0, 1_000_000); self.save_every.setValue(200)

        self.resume = QLineEdit("")
        btn_resume = QPushButton("Seç")
        btn_resume.clicked.connect(self._pick_resume_ckpt)
        resume_row = _row_lineedit_with_button(self.resume, btn_resume)

        self.arch.currentTextChanged.connect(self._sync_arch_fields)
        self._sync_arch_fields()

        # --- UI layout ---
        form.addRow("Dataset CSV", csv_row)
        form.addRow("Output klasörü", out_row)
        form.addRow("Checkpoint adı", self.ckpt_name)

        form.addRow("Epoch", self.epochs)
        form.addRow("Batch size", self.batch)
        form.addRow("Learning rate", self.lr)
        form.addRow("w_data", self.w_data)
        form.addRow("w_phys", self.w_phys)
        form.addRow("Phys warmup epoch", self.warmup)

        form.addRow("Hidden", self.hidden)
        form.addRow("Depth", self.depth)
        form.addRow("Seed", self.seed)
        form.addRow("dtype", self.dtype)
        form.addRow("Mimari (arch)", self.arch)

        # Fourier rows
        form.addRow("Fourier features (K)", self.fourier_features)
        form.addRow("Fourier min freq", self.fourier_min_freq)
        form.addRow("Fourier max freq", self.fourier_max_freq)
        form.addRow(self.fourier_include_input)
        form.addRow(self.fourier_linear)

        # Phase rows
        form.addRow("Phase harmonics", self.phase_harmonics)
        form.addRow(self.phase_include_t)
        form.addRow(self.phase_include_phase)

        # DeepONet rows
        form.addRow("DeepONet latent", self.deeponet_latent)
        form.addRow(self.deeponet_branch_aug)

        # SIREN rows
        form.addRow("SIREN w0_initial", self.siren_w0_initial)
        form.addRow("SIREN w0", self.siren_w0)

        # Loss/constraints
        form.addRow(self.hard_constraint)
        form.addRow(self.use_energy_loss)
        form.addRow("Energy weight", self.energy_weight)
        form.addRow(self.use_deriv_loss)
        form.addRow("alpha_v", self.alpha_v)

        # Adaptive weights
        form.addRow(self.adaptive_weighting)
        form.addRow("aw_update_every", self.aw_update_every)
        form.addRow("aw_beta", self.aw_beta)
        form.addRow("aw_ema", self.aw_ema)
        form.addRow("aw_min", self.aw_min)
        form.addRow("aw_max", self.aw_max)

        # Curriculum
        form.addRow(self.curriculum)
        form.addRow("curr_e1", self.curr_e1)
        form.addRow("curr_e2", self.curr_e2)
        form.addRow("curr_t1_s", self.curr_t1_s)
        form.addRow("curr_t2_s", self.curr_t2_s)
        form.addRow("curr_t3_s", self.curr_t3_s)

        # L-BFGS
        form.addRow(self.use_lbfgs)
        form.addRow("lbfgs_fraction", self.lbfgs_fraction)
        form.addRow("lbfgs_max_iter", self.lbfgs_max_iter)
        form.addRow("lbfgs_history_size", self.lbfgs_history_size)
        form.addRow("lbfgs_lr", self.lbfgs_lr)
        form.addRow("lbfgs_line_search", self.lbfgs_line_search)
        form.addRow("lbfgs_sample_size", self.lbfgs_sample_size)

        # perf / misc
        form.addRow("num_workers", self.num_workers)
        form.addRow("val_split (0=kapalı)", self.val_split)
        form.addRow("val_batches", self.val_batches)
        form.addRow("grad_clip (0=kapalı)", self.grad_clip)
        form.addRow("log_every", self.log_every)
        form.addRow("save_every (0=kapalı)", self.save_every)
        form.addRow("resume ckpt (ops.)", resume_row)

        box = QGroupBox("Eğitim Ayarları")
        box.setLayout(form)
        _tune_inputs(box)

        self.runner = ProcessPane()
        self.runner.btn_start.setText("Eğitimi başlat")
        self.runner.btn_start.clicked.connect(self._start)
        self.runner.set_progress_parser(self._parse_progress)

        top = QWidget()
        top_l = QVBoxLayout(); top_l.setContentsMargins(0, 0, 0, 0); top_l.setSpacing(0)
        top_l.addWidget(box)
        top.setLayout(top_l)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(_scroll_wrap(top))
        splitter.addWidget(self.runner)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 520])

        layout = QVBoxLayout()
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)
        layout.addWidget(splitter, 1)
        self.setLayout(layout)

    def _sync_arch_fields(self) -> None:
        arch = self.arch.currentText().strip().lower()
        has_fourier = "fourier" in arch
        has_phase = "phase" in arch
        has_deeponet = "deeponet" in arch
        has_siren = "siren" in arch

        for w in (self.fourier_features, self.fourier_min_freq, self.fourier_max_freq,
                  self.fourier_include_input, self.fourier_linear):
            w.setEnabled(has_fourier)

        for w in (self.phase_harmonics, self.phase_include_t, self.phase_include_phase):
            w.setEnabled(has_phase)

        for w in (self.deeponet_latent, self.deeponet_branch_aug):
            w.setEnabled(has_deeponet)

        for w in (self.siren_w0_initial, self.siren_w0):
            w.setEnabled(has_siren)

    def _pick_resume_ckpt(self) -> None:
        fn, _ = QFileDialog.getOpenFileName(self, "Resume checkpoint seç", self.resume.text() or str(SCRIPT_DIR), "PT Files (*.pt);;All Files (*.*)")
        if fn:
            self.resume.setText(_norm_path(fn))

    def _pick_csv(self) -> None:
        fn, _ = QFileDialog.getOpenFileName(self, "CSV seç", self.csv.text() or str(SCRIPT_DIR), "CSV Files (*.csv);;All Files (*.*)")
        if fn:
            self.csv.setText(_norm_path(fn))

    def _pick_out_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Output klasörü seç", self.out_dir.text() or str(SCRIPT_DIR))
        if d:
            self.out_dir.setText(_norm_path(d))

    def _start(self) -> None:
        # Ensure modules exist (friendly error)
        if not (SCRIPT_DIR / "train_pinn.py").exists():
            QMessageBox.critical(self, "Bulunamadı", "train_pinn.py aynı klasörde olmalı.")
            return

        csv_path = self.csv.text().strip()
        if not csv_path or not Path(csv_path).exists():
            QMessageBox.critical(self, "CSV yok", "Dataset CSV dosyası bulunamadı.")
            return

        out_dir = self.out_dir.text().strip()
        if not out_dir:
            QMessageBox.critical(self, "Klasör yok", "Output klasörü boş olamaz.")
            return

        # Build TrainConfig-like dict (JSON)
        arch = self.arch.currentText().strip()
        cfg: Dict[str, Any] = {
            "dataset_csv": csv_path,
            "out_dir": out_dir,
            "ckpt_name": self.ckpt_name.text().strip() or "orbit_pinn.pt",

            "epochs": int(self.epochs.value()),
            "batch_size": int(self.batch.value()),
            "lr": float(self.lr.value()),
            "w_data": float(self.w_data.value()),
            "w_phys": float(self.w_phys.value()),
            "phys_warmup_epochs": int(self.warmup.value()),

            "hidden": int(self.hidden.value()),
            "depth": int(self.depth.value()),
            "seed": int(self.seed.value()),
            "dtype": self.dtype.currentText().strip(),
            "arch": arch,

            # Fourier
            "fourier_features": int(self.fourier_features.value()),
            "fourier_min_freq": float(self.fourier_min_freq.value()),
            "fourier_max_freq": float(self.fourier_max_freq.value()),
            "fourier_include_input": bool(self.fourier_include_input.isChecked()),
            "fourier_log_sampling": (not bool(self.fourier_linear.isChecked())),

            # Phase
            "phase_harmonics": int(self.phase_harmonics.value()),
            "phase_include_t": bool(self.phase_include_t.isChecked()),
            "phase_include_phase": bool(self.phase_include_phase.isChecked()),

            # DeepONet
            "deeponet_latent": int(self.deeponet_latent.value()),
            "deeponet_branch_aug": bool(self.deeponet_branch_aug.isChecked()),

            # SIREN
            "siren_w0_initial": float(self.siren_w0_initial.value()),
            "siren_w0": float(self.siren_w0.value()),

            # Loss/constraints
            "hard_constraint": bool(self.hard_constraint.isChecked()),
            "use_energy_loss": bool(self.use_energy_loss.isChecked()),
            "energy_weight": float(self.energy_weight.value()),
            "use_derivative_data_loss": bool(self.use_deriv_loss.isChecked()),
            "alpha_v": float(self.alpha_v.value()),

            # Adaptive weights
            "adaptive_weighting": bool(self.adaptive_weighting.isChecked()),
            "aw_update_every": int(self.aw_update_every.value()),
            "aw_beta": float(self.aw_beta.value()),
            "aw_ema": float(self.aw_ema.value()),
            "aw_min": float(self.aw_min.value()),
            "aw_max": float(self.aw_max.value()),

            # Curriculum
            "curriculum": bool(self.curriculum.isChecked()),
            "curr_e1": int(self.curr_e1.value()),
            "curr_e2": int(self.curr_e2.value()),
            "curr_t1_s": float(self.curr_t1_s.value()),
            "curr_t2_s": float(self.curr_t2_s.value()),
            "curr_t3_s": float(self.curr_t3_s.value()),

            # L-BFGS
            "use_lbfgs": bool(self.use_lbfgs.isChecked()),
            "lbfgs_fraction": float(self.lbfgs_fraction.value()),
            "lbfgs_max_iter": int(self.lbfgs_max_iter.value()),
            "lbfgs_history_size": int(self.lbfgs_history_size.value()),
            "lbfgs_lr": float(self.lbfgs_lr.value()),
            "lbfgs_line_search": (None if self.lbfgs_line_search.currentText().strip() == "None" else self.lbfgs_line_search.currentText().strip()),
            "lbfgs_sample_size": int(self.lbfgs_sample_size.value()),

            # Loader/perf
            "num_workers": int(self.num_workers.value()),

            # Optional
            "val_split": float(self.val_split.value()),
            "val_batches": int(self.val_batches.value()),
            "grad_clip_norm": float(self.grad_clip.value()),
            "log_every": int(self.log_every.value()),
            "save_every": int(self.save_every.value()),
        }

        resume_path = self.resume.text().strip()
        cfg["resume_from"] = resume_path if resume_path else None

        # Write JSON config
        ts = time.strftime("%Y%m%d_%H%M%S")
        cfg_path = RUNS_DIR / f"train_{ts}.json"
        _write_json(cfg_path, cfg)

        # Run python -c ...
        code = _python_call_train_with_json(str(cfg_path))
        args = ["-u", "-c", code]

        self.runner.progress.setRange(0, self.epochs.value())
        self.runner.progress.setFormat("Epoch %v / %m")
        self.runner.start(sys.executable, args, workdir=str(SCRIPT_DIR))

    def _parse_progress(self, line: str) -> None:
        m = re.search(r"Epoch\s+(\d+)", line)
        if m:
            ep = int(m.group(1))
            self.runner.progress.setValue(min(ep + 1, self.epochs.value()))


# =============================================================================
# Eval Tab (library-style)
# =============================================================================

class EvalTab(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        form = QFormLayout()
        _tune_form(form)

        self.ckpt = QLineEdit(str(SCRIPT_DIR / "checkpoints" / "orbit_pinn.pt"))
        btn_ckpt = QPushButton("Seç")
        btn_ckpt.clicked.connect(self._pick_ckpt)
        ckpt_row = _row_lineedit_with_button(self.ckpt, btn_ckpt)

        self.out_dir = QLineEdit(str(SCRIPT_DIR / "eval_outputs"))
        btn_out = QPushButton("Seç")
        btn_out.clicked.connect(self._pick_out_dir)
        out_row = _row_lineedit_with_button(self.out_dir, btn_out)

        self.duration = QDoubleSpinBox(); self.duration.setDecimals(1); self.duration.setRange(1.0, 365.0 * 86400.0); self.duration.setValue(86400.0)
        self.dt = QDoubleSpinBox(); self.dt.setDecimals(1); self.dt.setRange(0.1, 86400.0); self.dt.setValue(60.0)
        self.seed = QSpinBox(); self.seed.setRange(0, 2_147_483_647); self.seed.setValue(123)

        self.method = QComboBox()
        self.method.addItems(["DOP853", "RK45", "Radau", "BDF"])
        self.method.setCurrentText("DOP853")

        self.rtol = QDoubleSpinBox(); self.rtol.setDecimals(16); self.rtol.setRange(1e-16, 1.0); self.rtol.setValue(1e-12)
        self.atol = QDoubleSpinBox(); self.atol.setDecimals(16); self.atol.setRange(1e-16, 1.0); self.atol.setValue(1e-12)

        self.include_end = QCheckBox("t_eval içine duration'ı ekle (include_end)")
        self.include_end.setChecked(True)

        # new in EvalConfig
        self.save_plots = QCheckBox("Plotları üret (save_plots)")
        self.save_plots.setChecked(True)
        self.save_json = QCheckBox("metrics.json yaz (save_json)")
        self.save_json.setChecked(True)

        self.perigee = QDoubleSpinBox(); self.perigee.setDecimals(2); self.perigee.setRange(0.0, 100_000.0); self.perigee.setValue(700.0)
        self.ecc = QDoubleSpinBox(); self.ecc.setDecimals(6); self.ecc.setRange(0.0, 0.99); self.ecc.setValue(0.01)
        self.inc = QDoubleSpinBox(); self.inc.setDecimals(3); self.inc.setRange(0.0, 180.0); self.inc.setValue(45.0)
        self.raan = QDoubleSpinBox(); self.raan.setDecimals(3); self.raan.setRange(0.0, 360.0); self.raan.setValue(10.0)
        self.argp = QDoubleSpinBox(); self.argp.setDecimals(3); self.argp.setRange(0.0, 360.0); self.argp.setValue(20.0)
        self.nu = QDoubleSpinBox(); self.nu.setDecimals(3); self.nu.setRange(0.0, 360.0); self.nu.setValue(0.0)

        self.use_y0 = QCheckBox("Direkt y0 gir")
        self.y0_edits = [QLineEdit("") for _ in range(6)]
        for e in self.y0_edits:
            e.setPlaceholderText("float")
            e.setEnabled(False)
        self.use_y0.stateChanged.connect(self._toggle_y0)

        y0_row = QHBoxLayout(); y0_row.setContentsMargins(0, 0, 0, 0); y0_row.setSpacing(10)
        for lab, ed in zip(["x","y","z","vx","vy","vz"], self.y0_edits):
            col = QVBoxLayout(); col.setContentsMargins(0, 0, 0, 0); col.setSpacing(6)
            lbl = QLabel(lab)
            col.addWidget(lbl)
            col.addWidget(ed)
            y0_row.addLayout(col)
        y0_widget = QWidget(); y0_widget.setLayout(y0_row)

        form.addRow("Checkpoint", ckpt_row)
        form.addRow("Output klasörü", out_row)
        form.addRow("Duration [s]", self.duration)
        form.addRow("dt [s]", self.dt)
        form.addRow("Seed", self.seed)
        form.addRow("solve_ivp method", self.method)
        form.addRow("RTOL", self.rtol)
        form.addRow("ATOL", self.atol)
        form.addRow(self.include_end)
        form.addRow(self.save_plots)
        form.addRow(self.save_json)
        form.addRow(self.use_y0)
        form.addRow("y0 (km, km/s)", y0_widget)
        form.addRow("Perigee alt [km]", self.perigee)
        form.addRow("e", self.ecc)
        form.addRow("inc [deg]", self.inc)
        form.addRow("RAAN [deg]", self.raan)
        form.addRow("argp [deg]", self.argp)
        form.addRow("nu [deg]", self.nu)

        box = QGroupBox("Değerlendirme Ayarları")
        box.setLayout(form)
        _tune_inputs(box)

        self.runner = ProcessPane()
        self.runner.btn_start.setText("Değerlendir")
        self.runner.btn_start.clicked.connect(self._start)
        self.runner.set_progress_parser(self._parse_progress)

        top = QWidget()
        top_l = QVBoxLayout(); top_l.setContentsMargins(0, 0, 0, 0); top_l.setSpacing(0)
        top_l.addWidget(box)
        top.setLayout(top_l)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(_scroll_wrap(top))
        splitter.addWidget(self.runner)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 520])

        layout = QVBoxLayout()
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)
        layout.addWidget(splitter, 1)
        self.setLayout(layout)

        self._saw_ok = False

    def _toggle_y0(self) -> None:
        enabled = self.use_y0.isChecked()
        for e in self.y0_edits:
            e.setEnabled(enabled)

    def _pick_ckpt(self) -> None:
        fn, _ = QFileDialog.getOpenFileName(self, "Checkpoint seç", self.ckpt.text() or str(SCRIPT_DIR), "PT Files (*.pt);;All Files (*.*)")
        if fn:
            self.ckpt.setText(_norm_path(fn))

    def _pick_out_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Output klasörü seç", self.out_dir.text() or str(SCRIPT_DIR))
        if d:
            self.out_dir.setText(_norm_path(d))

    def _start(self) -> None:
        if not (SCRIPT_DIR / "evaluate_pinn.py").exists():
            QMessageBox.critical(self, "Bulunamadı", "evaluate_pinn.py aynı klasörde olmalı.")
            return

        ckpt_path = self.ckpt.text().strip()
        if not ckpt_path or not Path(ckpt_path).exists():
            QMessageBox.critical(self, "Checkpoint yok", "Checkpoint dosyası bulunamadı.")
            return

        out_dir = self.out_dir.text().strip()
        if not out_dir:
            QMessageBox.critical(self, "Klasör yok", "Output klasörü boş olamaz.")
            return

        self._saw_ok = False
        self.runner.progress.setRange(0, 100)
        self.runner.progress.setValue(0)

        cfg: Dict[str, Any] = {
            "checkpoint": ckpt_path,
            "out_dir": out_dir,
            "duration_s": float(self.duration.value()),
            "dt_s": float(self.dt.value()),
            "include_end": bool(self.include_end.isChecked()),
            "method": self.method.currentText().strip(),
            "rtol": float(self.rtol.value()),
            "atol": float(self.atol.value()),
            "seed": int(self.seed.value()),
            "save_plots": bool(self.save_plots.isChecked()),
            "save_json": bool(self.save_json.isChecked()),
            "return_arrays": False,
        }

        if self.use_y0.isChecked():
            try:
                y0_vals = [float(e.text().strip()) for e in self.y0_edits]
            except Exception:
                QMessageBox.critical(self, "y0 hatalı", "y0 alanlarına 6 adet sayı gir (x,y,z,vx,vy,vz).")
                return
            cfg["y0"] = y0_vals
            cfg["use_coe"] = False
        else:
            cfg["y0"] = None
            cfg["use_coe"] = True
            cfg["perigee_alt_km"] = float(self.perigee.value())
            cfg["e"] = float(self.ecc.value())
            cfg["inc_deg"] = float(self.inc.value())
            cfg["raan_deg"] = float(self.raan.value())
            cfg["argp_deg"] = float(self.argp.value())
            cfg["nu_deg"] = float(self.nu.value())

        ts = time.strftime("%Y%m%d_%H%M%S")
        cfg_path = RUNS_DIR / f"eval_{ts}.json"
        _write_json(cfg_path, cfg)

        code = _python_call_eval_with_json(str(cfg_path))
        args = ["-u", "-c", code]
        self.runner.start(sys.executable, args, workdir=str(SCRIPT_DIR))

    def _parse_progress(self, line: str) -> None:
        if line.strip():
            self.runner.progress.setValue(max(self.runner.progress.value(), 25))
        if "[OK]" in line or "Evaluation completed" in line:
            self._saw_ok = True
            self.runner.progress.setValue(100)


# =============================================================================
# Main window
# =============================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Orbit PINN – Premium Dashboard")
        self.resize(1220, 860)

        tabs = QTabWidget()
        tabs.addTab(DatasetTab(), "Dataset")
        tabs.addTab(TrainTab(), "Eğitim")
        tabs.addTab(EvalTab(), "Değerlendirme")

        info = QLabel(
            "<b>Not:</b> Bu panel script'leri QProcess ile çalıştırır. "
            "Çalışma dizini: bu dosyanın bulunduğu klasör. "
            "Train/Eval için CLI yok: UI geçici JSON yazar ve python -c ile fonksiyonu çağırır."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #b9c2dd;")

        root = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        layout.addWidget(info)
        layout.addWidget(tabs, 1)
        root.setLayout(layout)
        self.setCentralWidget(root)


def apply_premium_dark_theme(app: QApplication) -> None:
    app.setStyle("Fusion")

    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor("#0b1020"))
    pal.setColor(QPalette.ColorRole.WindowText, QColor("#e8ecf8"))
    pal.setColor(QPalette.ColorRole.Base, QColor("#070b14"))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor("#0f1830"))
    pal.setColor(QPalette.ColorRole.Text, QColor("#e8ecf8"))
    pal.setColor(QPalette.ColorRole.Button, QColor("#121a33"))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor("#e8ecf8"))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor("#0f1830"))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor("#e8ecf8"))
    pal.setColor(QPalette.ColorRole.Highlight, QColor("#7c5cff"))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.Link, QColor("#9aa7ff"))
    app.setPalette(pal)

    app.setStyleSheet(
        """
        QWidget { font-size: 13px; color: #e8ecf8; }
        QMainWindow, QWidget {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #0b1020, stop:1 #070a12);
        }

        QGroupBox {
            background-color: rgba(16, 24, 48, 0.72);
            border: 1px solid rgba(120, 92, 255, 0.22);
            border-radius: 16px;
            margin-top: 14px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 14px;
            padding: 0 10px;
            color: #d7dcff;
        }

        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
            background-color: rgba(7, 11, 20, 0.92);
            border: 1px solid rgba(185, 194, 221, 0.22);
            border-radius: 12px;
            padding: 0px 12px;
            min-height: 40px;
            selection-background-color: #7c5cff;
        }
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
            border: 1px solid rgba(124, 92, 255, 0.85);
        }

        QSpinBox, QDoubleSpinBox { padding-right: 44px; }
        QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {
            subcontrol-origin: border;
            width: 34px;
            background: rgba(18, 26, 51, 0.9);
            border-left: 1px solid rgba(185, 194, 221, 0.18);
        }
        QAbstractSpinBox::up-button { subcontrol-position: top right; border-top-right-radius: 12px; }
        QAbstractSpinBox::down-button { subcontrol-position: bottom right; border-bottom-right-radius: 12px; }

        QPlainTextEdit {
            background-color: rgba(7, 11, 20, 0.92);
            border: 1px solid rgba(185, 194, 221, 0.22);
            border-radius: 14px;
            padding: 10px 12px;
            selection-background-color: #7c5cff;
        }

        QTabWidget::pane {
            border: 1px solid rgba(185, 194, 221, 0.18);
            border-radius: 16px;
            background-color: rgba(15, 24, 48, 0.35);
            top: -1px;
        }
        QTabBar::tab {
            background: rgba(16, 24, 48, 0.65);
            border: 1px solid rgba(185, 194, 221, 0.14);
            border-bottom: none;
            padding: 10px 14px;
            margin-right: 8px;
            border-top-left-radius: 14px;
            border-top-right-radius: 14px;
            color: #cdd6f6;
        }
        QTabBar::tab:selected {
            background: rgba(15, 24, 48, 0.92);
            border-color: rgba(124, 92, 255, 0.55);
            color: #ffffff;
        }

        QProgressBar {
            background-color: rgba(7, 11, 20, 0.92);
            border: 1px solid rgba(185, 194, 221, 0.22);
            border-radius: 10px;
            height: 18px;
            text-align: center;
        }
        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #7c5cff, stop:1 #38bdf8);
            border-radius: 10px;
        }

        QPushButton {
            border-radius: 12px;
            padding: 9px 14px;
            border: 1px solid rgba(185, 194, 221, 0.18);
            background-color: rgba(18, 26, 51, 0.9);
        }
        QPushButton:hover { background-color: rgba(26, 36, 70, 0.95); }
        QPushButton:pressed { background-color: rgba(14, 20, 40, 0.95); }
        QPushButton:disabled { color: rgba(232, 236, 248, 0.35); background-color: rgba(16, 24, 48, 0.35); }

        QPushButton[kind="primary"] {
            border: 1px solid rgba(124, 92, 255, 0.55);
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 rgba(124, 92, 255, 0.95), stop:1 rgba(56, 189, 248, 0.85));
            color: #ffffff;
        }

        QPushButton[kind="danger"] {
            border: 1px solid rgba(248, 113, 113, 0.55);
            background-color: rgba(248, 113, 113, 0.18);
        }

        QPushButton[kind="ghost"] { background-color: rgba(16, 24, 48, 0.35); }

        QCheckBox { spacing: 10px; }
        QCheckBox::indicator {
            width: 18px; height: 18px;
            border-radius: 6px;
            border: 1px solid rgba(185, 194, 221, 0.22);
            background: rgba(7, 11, 20, 0.92);
        }
        QCheckBox::indicator:checked {
            background: rgba(124, 92, 255, 0.9);
            border-color: rgba(124, 92, 255, 0.9);
        }

        QScrollBar:vertical { background: transparent; width: 12px; margin: 0; }
        QScrollBar::handle:vertical {
            background: rgba(185, 194, 221, 0.22);
            min-height: 28px;
            border-radius: 6px;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        """
    )


def main() -> None:
    try:
        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except Exception:
        pass

    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    os.chdir(str(SCRIPT_DIR))

    app = QApplication(sys.argv)
    apply_premium_dark_theme(app)

    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
