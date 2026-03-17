
from __future__ import annotations

import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSlider,
    QSplitter,
    QTabBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QMenu,
    QWidget,
)

if __package__ in (None, ""):
    module_root = Path(__file__).resolve().parents[1]
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))

import config
from core.ai.analyzer import EventAnalyzer
from core.db_ai import (
    AIResultRecord,
    AIResultStore,
    PRIMARY_TYPES,
    ai_status_label,
    event_hash,
    event_key_from_event,
    primary_type_label,
)
from core.grouping import group_events
from core.models import CAMERA_PRIORITY, CAMERAS, CLIP_GROUPS, Event, GroupByMode, VideoClip
from core.scanner import scan_videos
from player.sync import SyncController
from player.vlc_panel import VlcPanel
from ui.filters import AiFilterBar

CAMERA_TITLES = {
    "front": "前视",
    "front_narrow": "前视窄角",
    "left_repeater": "左侧",
    "right_repeater": "右侧",
    "back": "后视",
    "left_pillar": "左A柱",
    "right_pillar": "右A柱",
    "left_b_pillar": "左B柱",
    "right_b_pillar": "右B柱",
}

GROUP_TITLES = {
    "RecentClips": "最近片段",
    "SavedClips": "已保存片段",
    "SentryClips": "哨兵片段",
    "Other": "其他",
}

DISPLAY_TITLES = {"auto": "自动路数", "4": "4路", "6": "6路"}


def camera_title(camera: str) -> str:
    return CAMERA_TITLES.get(camera, camera)


def pick_event_source(event: Event) -> str:
    for group in CLIP_GROUPS:
        if group in event.source_groups:
            return group
    return sorted(event.source_groups)[0] if event.source_groups else "Other"


class AnalyzeWorker(QObject):
    progress = Signal(int, int, str)
    result_ready = Signal(str, object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, events: list[Event], db_path: Path, roi_map: dict[str, dict[str, float]]) -> None:
        super().__init__()
        self._events = events
        self._db_path = db_path
        self._roi_map = roi_map
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        store = AIResultStore(self._db_path)
        try:
            analyzer = EventAnalyzer()
            total = len(self._events)
            for idx, event in enumerate(self._events, start=1):
                if self._cancelled:
                    break

                key = event_key_from_event(event)
                self.progress.emit(idx, total, key)

                expected_hash = event_hash(event, config.AI_RULE_VERSION, analyzer.model_version)
                cached = store.get_result(key, analyzer.version)
                if cached and cached.event_hash == expected_hash:
                    self.result_ready.emit(key, cached)
                    continue

                output = analyzer.analyze_event(event, self._roi_map)
                source = pick_event_source(event)
                store.save_event_analysis(
                    event=event,
                    version=analyzer.version,
                    source_folder=source,
                    record=output.record,
                    durations_ms=output.durations_ms,
                )
                saved = store.get_result(key, analyzer.version) or output.record
                self.result_ready.emit(key, saved)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
        finally:
            store.close()
            self.finished.emit()


class ScanWorker(QObject):
    scanned = Signal(list)
    failed = Signal(str)

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root

    def run(self) -> None:
        try:
            clips = scan_videos(self.root)
            self.scanned.emit(clips)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("TeslaCam 查看器 - 作者：没有清风了")
        icon_path = Path(__file__).resolve().parents[1] / "desktop_icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        self.resize(1620, 940)

        self.all_events: list[Event] = []
        self.events: list[Event] = []
        self.scanned_clips: list[VideoClip] = []
        self.current_event: Event | None = None
        self.current_root: Path | None = None
        self.group_by_mode: GroupByMode = "second"
        self.scan_thread: QThread | None = None
        self.scan_worker: ScanWorker | None = None
        self.display_mode: str = "auto"
        self.visible_cameras: list[str] = []
        self.panels: dict[str, VlcPanel] = {}
        self._shortcuts: list[QShortcut] = []
        self._user_seeking = False

        self.ai_store = AIResultStore(config.AI_DB_PATH)
        self.ai_results: dict[str, AIResultRecord] = {}
        self.analyze_thread: QThread | None = None
        self.analyze_worker: AnalyzeWorker | None = None

        self._build_ui()
        self.sync = SyncController(self.panels, self)
        self._bind_signals()
        self._bind_shortcuts()
        self._notify_vlc_error_if_any()

        self.ui_timer = QTimer(self)
        self.ui_timer.setInterval(200)
        self.ui_timer.timeout.connect(self._update_playback_ui)
        self.ui_timer.start()

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        top_bar = QHBoxLayout()
        self.btn_select_dir = QPushButton("选择 TeslaCam 目录")
        self.btn_rescan = QPushButton("重新扫描")
        self.btn_rescan.setEnabled(False)
        self.btn_analyze = QPushButton("开始分析")
        self.btn_analyze.setEnabled(False)
        self.btn_ai_settings = QPushButton("智能设置")
        self.btn_roi_settings = QPushButton("区域设置")

        self.group_by_combo = QComboBox()
        self.group_by_combo.addItem("按秒分组", userData="second")
        self.group_by_combo.addItem("按分钟分组", userData="minute")

        self.display_mode_combo = QComboBox()
        for key in ("auto", "4", "6"):
            self.display_mode_combo.addItem(DISPLAY_TITLES[key], userData=key)

        self.lbl_path = QLabel("未选择目录")
        self.lbl_path.setWordWrap(True)
        self.lbl_scan_status = QLabel("")
        self.lbl_scan_status.setStyleSheet("color:#1177dd; font-weight: 600;")

        top_bar.addWidget(self.btn_select_dir)
        top_bar.addWidget(self.btn_rescan)
        top_bar.addWidget(self.btn_analyze)
        top_bar.addWidget(self.lbl_scan_status)
        top_bar.addWidget(self.group_by_combo)
        top_bar.addWidget(self.display_mode_combo)
        top_bar.addWidget(self.btn_roi_settings)
        top_bar.addWidget(self.btn_ai_settings)
        top_bar.addWidget(self.lbl_path, 1)
        root_layout.addLayout(top_bar)

        progress_row = QHBoxLayout()
        self.lbl_analyze = QLabel("智能分析: 空闲")
        self.progress_ai = QProgressBar()
        self.progress_ai.setRange(0, 100)
        self.progress_ai.setValue(0)
        progress_row.addWidget(self.lbl_analyze)
        progress_row.addWidget(self.progress_ai, 1)
        root_layout.addLayout(progress_row)

        self.filter_tabs = QTabBar()
        idx = self.filter_tabs.addTab("全部")
        self.filter_tabs.setTabData(idx, "All")
        for group in CLIP_GROUPS:
            tab_idx = self.filter_tabs.addTab(GROUP_TITLES.get(group, group))
            self.filter_tabs.setTabData(tab_idx, group)
        self.filter_tabs.setExpanding(False)
        root_layout.addWidget(self.filter_tabs)

        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)
        left_layout.addWidget(QLabel("事件列表"))
        self.ai_filter_bar = AiFilterBar()
        left_layout.addWidget(self.ai_filter_bar)
        self.event_list = QListWidget()
        self.event_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.event_list.customContextMenuRequested.connect(self._show_event_context_menu)
        left_layout.addWidget(self.event_list, 1)
        self.event_detail = QLabel("事件详情: -")
        self.event_detail.setWordWrap(True)
        self.event_detail.setStyleSheet("color:#444;")
        left_layout.addWidget(self.event_detail)
        splitter.addWidget(left)

        self.video_widget = QWidget()
        self.video_grid = QGridLayout(self.video_widget)
        self.video_grid.setContentsMargins(0, 0, 0, 0)
        self.video_grid.setSpacing(6)
        splitter.addWidget(self.video_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        controls = QHBoxLayout()
        self.btn_play_pause = QPushButton("播放")
        self.btn_play_pause.setEnabled(False)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setEnabled(False)
        self.btn_jump_evidence = QPushButton("跳转证据")
        self.btn_jump_evidence.setEnabled(False)
        self.chk_sync = QCheckBox("同步")
        self.chk_sync.setChecked(True)
        self.master_combo = QComboBox()
        self.rate_combo = QComboBox()
        self.rate_combo.addItem("0.5x", userData=0.5)
        self.rate_combo.addItem("1x", userData=1.0)
        self.rate_combo.addItem("1.5x", userData=1.5)
        self.rate_combo.addItem("2x", userData=2.0)
        self.rate_combo.setCurrentIndex(1)

        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setEnabled(False)
        self.position_slider.setRange(0, 0)
        self.lbl_seek_target = QLabel("")
        self.lbl_seek_target.setMinimumWidth(90)
        self.lbl_seek_target.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_time.setMinimumWidth(160)
        self.lbl_time.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        controls.addWidget(self.btn_play_pause)
        controls.addWidget(self.btn_stop)
        controls.addWidget(self.btn_jump_evidence)
        controls.addWidget(self.chk_sync)
        controls.addWidget(self.master_combo)
        controls.addWidget(self.rate_combo)
        controls.addWidget(self.position_slider, 1)
        controls.addWidget(self.lbl_seek_target)
        controls.addWidget(self.lbl_time)
        root_layout.addLayout(controls)

        self._rebuild_video_grid(list(CAMERAS))
    def _bind_signals(self) -> None:
        self.btn_select_dir.clicked.connect(self._choose_directory)
        self.btn_rescan.clicked.connect(self._rescan)
        self.btn_analyze.clicked.connect(self._toggle_analyze)
        self.btn_ai_settings.clicked.connect(self._open_ai_settings)
        self.btn_roi_settings.clicked.connect(self._open_roi_settings)
        self.group_by_combo.currentIndexChanged.connect(self._on_group_by_changed)
        self.display_mode_combo.currentIndexChanged.connect(self._on_display_mode_changed)
        self.filter_tabs.currentChanged.connect(self._apply_filter)
        self.ai_filter_bar.filter_changed.connect(self._on_ai_filter_changed)
        self.event_list.currentRowChanged.connect(self._on_event_selected)
        self.btn_play_pause.clicked.connect(self._toggle_play_pause)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_jump_evidence.clicked.connect(self._jump_to_evidence)
        self.chk_sync.toggled.connect(self.sync.set_sync_enabled)
        self.master_combo.currentIndexChanged.connect(self._on_master_changed)
        self.rate_combo.currentIndexChanged.connect(self._on_rate_changed)
        self.position_slider.sliderPressed.connect(self._on_seek_press)
        self.position_slider.sliderReleased.connect(self._on_seek_release)
        self.position_slider.valueChanged.connect(self._on_slider_value_changed)

    def _bind_shortcuts(self) -> None:
        mapping = [
            ("Space", self._toggle_play_pause),
            ("Left", lambda: self._seek_relative(-5000)),
            ("Right", lambda: self._seek_relative(5000)),
            ("S", self._stop),
        ]
        for key_text, callback in mapping:
            shortcut = QShortcut(QKeySequence(key_text), self)
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)

    def _notify_vlc_error_if_any(self) -> None:
        errors = {p.unavailable_reason for p in self.panels.values() if p.unavailable_reason}
        if not errors:
            return
        QMessageBox.critical(
            self,
            "VLC 初始化失败",
            "视频播放不可用。\n"
            + "\n".join(sorted(errors))
            + "\n\n请安装 VLC，并确保 VLC 与 Python 位数一致。",
        )

    def _choose_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "选择 TeslaCam 根目录或 clips 目录")
        if selected:
            self._start_scan(Path(selected))

    def _rescan(self) -> None:
        if self.current_root:
            self._start_scan(self.current_root)

    def _start_scan(self, root: Path) -> None:
        if self.scan_thread and self.scan_thread.isRunning():
            QMessageBox.information(self, "扫描进行中", "正在扫描，请稍候。")
            return
        self.current_root = root
        self.lbl_path.setText(str(root))
        self.statusBar().showMessage("正在扫描视频文件，请稍候...", 5000)
        self.btn_rescan.setEnabled(False)
        self.btn_analyze.setEnabled(False)
        self.event_list.setEnabled(False)

        self.scan_thread = QThread(self)
        self.scan_worker = ScanWorker(root)
        self.scan_worker.moveToThread(self.scan_thread)
        self.scan_thread.started.connect(self.scan_worker.run)
        self.scan_worker.scanned.connect(self._on_scan_finished)
        self.scan_worker.failed.connect(self._on_scan_failed)
        self.scan_worker.scanned.connect(self.scan_thread.quit)
        self.scan_worker.failed.connect(self.scan_thread.quit)
        self.scan_thread.finished.connect(self._on_scan_thread_done)
        self.scan_thread.start()
        self.lbl_scan_status.setText("正在加载中...")
        QApplication.setOverrideCursor(Qt.WaitCursor)

    def _on_scan_finished(self, clips: list[VideoClip]) -> None:
        self.scanned_clips = clips
        self.lbl_scan_status.setText("扫描完成")
        if not clips:
            QMessageBox.information(self, "没有视频", "未找到支持的视频文件（mp4/mov/mkv）。")
            self._set_events([])
            self.statusBar().showMessage("扫描完成：未找到视频。", 5000)
            return

        events = group_events(clips, group_by=self.group_by_mode)
        if not events:
            QMessageBox.information(self, "没有事件", "找到视频但无法分组为事件。")
            self._set_events([])
            self.statusBar().showMessage("扫描完成：未找到事件。", 5000)
            return

        self._set_events(events)
        self.statusBar().showMessage(f"已加载 {len(clips)} 个文件，分组得到 {len(events)} 个事件", 5000)

    def _on_scan_failed(self, message: str) -> None:
        QMessageBox.warning(self, "扫描失败", f"扫描目录失败:\n{message}")
        self.lbl_scan_status.setText("扫描失败")
        self._set_events([])

    def _on_scan_thread_done(self) -> None:
        self.scan_worker = None
        self.scan_thread = None
        QApplication.restoreOverrideCursor()
        self.lbl_scan_status.setText("")
        self.btn_rescan.setEnabled(True)
        self.event_list.setEnabled(True)
        self.btn_analyze.setEnabled(bool(self.all_events))

    def _load_directory(self, root: Path) -> None:
        self.current_root = root
        self.lbl_path.setText(str(root))
        self.btn_rescan.setEnabled(True)
        self.btn_analyze.setEnabled(bool(self.all_events))
        self.statusBar().showMessage(f"已加载 {len(self.scanned_clips)} 个文件，分组得到 {len(self.all_events)} 个事件", 5000)

    def _set_events(self, events: list[Event]) -> None:
        self.all_events = events
        self.current_event = None
        self._load_cached_results()
        self._update_visible_cameras()
        self._apply_filter()
        has_data = bool(events)
        self.btn_analyze.setEnabled(has_data)
        self.progress_ai.setValue(0)
        self.lbl_analyze.setText("智能分析: 空闲")

    def _load_cached_results(self) -> None:
        keys = [event_key_from_event(e) for e in self.all_events]
        self.ai_results = self.ai_store.get_many_latest(keys)

    def _on_ai_filter_changed(self, _types: object, _risk: int, _only: bool) -> None:
        self._apply_filter()

    def _apply_filter(self, _index: int | None = None) -> None:
        selected_ts = self.current_event.timestamp if self.current_event else None
        tab_idx = self.filter_tabs.currentIndex()
        selected_group = self.filter_tabs.tabData(tab_idx) or "All"
        selected_types, min_risk, only_ai = self.ai_filter_bar.current_filter()
        all_types = set(PRIMARY_TYPES)

        filtered: list[Event] = []
        for event in self.all_events:
            if selected_group not in {"All", "全部"} and selected_group not in event.source_groups:
                continue

            key = event_key_from_event(event)
            ai = self.ai_results.get(key)
            if ai is None:
                if only_ai:
                    continue
                if selected_types != all_types:
                    continue
                if min_risk > 0:
                    continue
            else:
                if ai.primary_type not in selected_types:
                    continue
                if ai.risk_level < min_risk:
                    continue
            filtered.append(event)

        self.events = filtered
        self.event_list.blockSignals(True)
        self.event_list.clear()
        for idx, event in enumerate(self.events):
            item = QListWidgetItem(self._event_label(event))
            item.setData(Qt.UserRole, idx)
            self.event_list.addItem(item)
        self.event_list.blockSignals(False)

        has_data = bool(self.events)
        self.btn_play_pause.setEnabled(has_data)
        self.btn_stop.setEnabled(has_data)
        self.position_slider.setEnabled(has_data)

        if not has_data:
            self.current_event = None
            self._clear_players()
            self.btn_jump_evidence.setEnabled(False)
            self.event_detail.setText("事件详情: -")
            self.lbl_time.setText("00:00 / 00:00")
            self.lbl_seek_target.setText("")
            self.position_slider.setRange(0, 0)
            self.position_slider.setValue(0)
            return

        row = 0
        if selected_ts is not None:
            for i, event in enumerate(self.events):
                if event.timestamp == selected_ts:
                    row = i
                    break
        self.event_list.setCurrentRow(row)

    def _event_label(self, event: Event) -> str:
        ts = event.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        missing = [cam for cam in self.visible_cameras if cam not in event.clips]
        missing_text = f" 缺{','.join(camera_title(cam) for cam in missing)}" if missing else ""
        flags_text = f" [{','.join(sorted(event.flags))}]" if event.flags else ""

        key = event_key_from_event(event)
        ai = self.ai_results.get(key)
        if ai is None:
            ai_text = "未分析"
        else:
            obj = ai.objects or {}
            ai_text = (
                f"{primary_type_label(ai.primary_type)} 风险{ai.risk_level} "
                f"人{obj.get('person', 0)}/车{obj.get('vehicle', 0)}/动物{obj.get('animal', 0)}"
            )
            if ai.status in {"error", "skipped"}:
                ai_text += f"（{ai_status_label(ai.status)}）"
        return f"{ts} {flags_text} [{ai_text}]{missing_text}"

    def _show_event_context_menu(self, pos) -> None:
        item = self.event_list.itemAt(pos)
        if item is None:
            return
        idx = item.data(Qt.UserRole)
        if idx is None or not isinstance(idx, int) or idx < 0 or idx >= len(self.events):
            return
        menu = QMenu(self)
        export_action = menu.addAction("导出全部画面")
        action = menu.exec(self.event_list.viewport().mapToGlobal(pos))
        if action == export_action:
            self._export_all_screens(self.events[idx])

    def _export_all_screens(self, event: Event) -> None:
        if not event.clips:
            QMessageBox.information(self, "导出失败", "当前事件没有可导出的视频。")
            return

        target_dir = QFileDialog.getExistingDirectory(self, "选择导出文件夹")
        if not target_dir:
            return
        target_path = Path(target_dir)

        copied = 0
        errors: list[str] = []
        for cam in sorted(event.all_cameras()):
            clip = event.clip_for(cam)
            if clip is None or not clip.path.exists():
                errors.append(f"{cam}: 文件不存在")
                continue
            suffix = clip.path.suffix
            out_name = f"{event.timestamp.strftime('%Y%m%d_%H%M%S')}_{cam}{suffix}"
            out_file = target_path / out_name
            count = 1
            while out_file.exists():
                out_file = target_path / f"{event.timestamp.strftime('%Y%m%d_%H%M%S')}_{cam}_{count}{suffix}"
                count += 1
            try:
                shutil.copy2(clip.path, out_file)
                copied += 1
            except Exception as exc:
                errors.append(f"{cam}: {exc}")

        msg = f"已导出 {copied} 个画面到 {target_path}。"
        if errors:
            msg += "\n部分文件导出失败:\n" + "\n".join(errors)
        QMessageBox.information(self, "导出结果", msg)

    def _on_event_selected(self, row: int) -> None:
        if row < 0 or row >= len(self.events):
            return
        event = self.events[row]
        self.current_event = event
        self._load_event(event)
        self._update_event_detail(event)
        self.sync.play()
        self.btn_play_pause.setText("暂停")

    def _load_event(self, event: Event) -> None:
        for cam in self.visible_cameras:
            panel = self.panels[cam]
            clip = event.clip_for(cam)
            if clip is None or not clip.path.exists():
                panel.clear()
                continue
            ok = panel.load_media(clip.path)
            if not ok:
                QMessageBox.warning(self, "播放错误", f"无法加载文件:\n{clip.path}")

        self.sync.select_master("front")
        self._set_master_combo(self.sync.master_camera)
        self.sync.stop()
        self.sync.set_rate(self._selected_rate())
        self.btn_play_pause.setText("播放")
        self.position_slider.setRange(0, 0)
        self.position_slider.setValue(0)
        self.lbl_seek_target.setText("")
        self.lbl_time.setText("00:00 / 00:00")

    def _update_event_detail(self, event: Event) -> None:
        cameras = ", ".join(camera_title(cam) for cam in event.all_cameras()) if event.clips else "-"
        groups = ", ".join(GROUP_TITLES.get(g, g) for g in sorted(event.source_groups)) or "其他"
        flags = ", ".join(sorted(event.flags)) or "无"
        key = event_key_from_event(event)
        ai = self.ai_results.get(key)
        if ai is None:
            ai_text = "智能分析: 未分析"
            self.btn_jump_evidence.setEnabled(False)
        else:
            ai_text = (
                f"智能分析: {primary_type_label(ai.primary_type)} / 风险{ai.risk_level} / "
                f"人{ai.objects.get('person', 0)} 车{ai.objects.get('vehicle', 0)} 动物{ai.objects.get('animal', 0)} / "
                f"{ai.ai_summary}"
            )
            if ai.error_message:
                ai_text += f" / 状态: {ai_status_label(ai.status)} / 备注: {ai.error_message}"
            self.btn_jump_evidence.setEnabled(bool(ai.evidence))
        self.event_detail.setText(f"事件详情: 路数=[{cameras}] 分组=[{groups}] 状态=[{flags}] {ai_text}")
    def _toggle_play_pause(self) -> None:
        if not self.current_event:
            return
        if self.sync.is_playing():
            self.sync.pause()
            self.btn_play_pause.setText("播放")
        else:
            self.sync.play()
            self.btn_play_pause.setText("暂停")

    def _stop(self) -> None:
        if not self.current_event:
            return
        self.sync.stop()
        self.btn_play_pause.setText("播放")

    def _seek_relative(self, delta_ms: int) -> None:
        if not self.current_event:
            return
        base = self.sync.current_time()
        total = max(self.sync.length(), 0)
        target = max(0, base + delta_ms)
        if total > 0:
            target = min(target, total)
        self.sync.seek(target, force_all=True)
        self.position_slider.setValue(target)

    def _jump_to_evidence(self) -> None:
        if not self.current_event:
            return
        key = event_key_from_event(self.current_event)
        ai = self.ai_results.get(key)
        if ai is None or not ai.evidence:
            return
        target: dict[str, Any] | None = None
        for item in ai.evidence:
            if item.get("cam") == self.sync.master_camera:
                target = item
                break
        if target is None:
            target = ai.evidence[0]
        t_ms = int(target.get("t_ms", 0))
        self.sync.seek(t_ms, force_all=True)
        self.position_slider.setValue(t_ms)

    def _on_master_changed(self, index: int) -> None:
        cam = self.master_combo.itemData(index)
        if cam in self.panels:
            self.sync.select_master(cam)

    def _set_master_combo(self, cam: str) -> None:
        idx = self.master_combo.findData(cam)
        if idx >= 0 and idx != self.master_combo.currentIndex():
            self.master_combo.blockSignals(True)
            self.master_combo.setCurrentIndex(idx)
            self.master_combo.blockSignals(False)

    def _on_rate_changed(self, _index: int) -> None:
        self.sync.set_rate(self._selected_rate())

    def _selected_rate(self) -> float:
        return float(self.rate_combo.currentData() or 1.0)

    def _on_seek_press(self) -> None:
        self._user_seeking = True
        self._on_slider_value_changed(self.position_slider.value())

    def _on_seek_release(self) -> None:
        self._user_seeking = False
        target = self.position_slider.value()
        self.lbl_seek_target.setText("")
        self.sync.seek(target, force_all=True)

    def _on_slider_value_changed(self, value: int) -> None:
        if self._user_seeking:
            self.lbl_seek_target.setText(f"目标 {self._format_ms(value)}")

    def _on_group_by_changed(self, _index: int) -> None:
        selected = self.group_by_combo.currentData()
        if selected in {"second", "minute"}:
            self.group_by_mode = selected
        if not self.scanned_clips:
            return
        events = group_events(self.scanned_clips, group_by=self.group_by_mode)
        self._set_events(events)
        mode_text = "秒" if self.group_by_mode == "second" else "分钟"
        self.statusBar().showMessage(f"已按{mode_text}重新分组: {len(events)} 个事件", 2500)

    def _on_display_mode_changed(self, _index: int) -> None:
        selected = self.display_mode_combo.currentData()
        if selected in {"auto", "4", "6"}:
            self.display_mode = selected
        self._update_visible_cameras()
        if self.all_events:
            self._apply_filter()
        self.statusBar().showMessage(f"当前显示模式: {DISPLAY_TITLES.get(self.display_mode, self.display_mode)}", 2500)

    def _rebuild_video_grid(self, cameras: list[str]) -> None:
        if hasattr(self, "sync"):
            self.sync.stop()

        while self.video_grid.count():
            item = self.video_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        self.panels = {}
        self.visible_cameras = list(cameras)
        cols = 3 if len(cameras) > 4 else 2
        for i, cam in enumerate(cameras):
            panel = VlcPanel(cam, camera_title(cam))
            panel.playback_error.connect(self._on_panel_playback_error)
            panel.zoom_requested.connect(self._on_panel_zoom_requested)
            self.panels[cam] = panel
            self.video_grid.addWidget(panel, i // cols, i % cols)

        if hasattr(self, "sync"):
            self.sync.set_panels(self.panels)
        self._refresh_master_combo()

    def _refresh_master_combo(self) -> None:
        prev = self.master_combo.currentData()
        self.master_combo.blockSignals(True)
        self.master_combo.clear()
        for cam in self.visible_cameras:
            self.master_combo.addItem(f"主视角: {camera_title(cam)}", userData=cam)
        self.master_combo.blockSignals(False)
        if self.visible_cameras:
            target = prev if prev in self.visible_cameras else self.visible_cameras[0]
            self._set_master_combo(target)

    def _compute_visible_cameras(self) -> list[str]:
        if not self.all_events:
            target = 6 if self.display_mode == "6" else 4
            return list(CAMERA_PRIORITY[:target])

        counts: Counter[str] = Counter()
        for event in self.all_events:
            counts.update(event.clips.keys())

        ordered: list[str] = []
        for cam in CAMERA_PRIORITY:
            if counts.get(cam, 0) > 0:
                ordered.append(cam)

        extra = [cam for cam in counts if cam not in ordered]
        extra.sort(key=lambda c: (-counts[c], c))
        ordered.extend(extra)

        if self.display_mode == "4":
            target = 4
        elif self.display_mode == "6":
            target = 6
        else:
            target = 6 if len(ordered) >= 6 else 4

        for cam in CAMERA_PRIORITY:
            if len(ordered) >= target:
                break
            if cam not in ordered:
                ordered.append(cam)
        return ordered[:target]

    def _update_visible_cameras(self) -> None:
        next_cameras = self._compute_visible_cameras()
        if next_cameras != self.visible_cameras:
            self._rebuild_video_grid(next_cameras)
            if self.current_event:
                self._load_event(self.current_event)

    def _clear_players(self) -> None:
        for panel in self.panels.values():
            panel.clear()

    def _on_panel_playback_error(self, camera: str, message: str) -> None:
        QMessageBox.warning(self, "VLC 播放错误", f"{camera}: {message}\n该路已跳过。")

    def _on_panel_zoom_requested(self, camera: str, path: Path | None) -> None:
        if path is None or not path.exists():
            QMessageBox.information(self, "放大查看", "该路视频未加载或文件不存在。")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f"放大查看 - {camera_title(camera)}")
        dialog.resize(1024, 640)
        layout = QVBoxLayout(dialog)

        zoom_panel = VlcPanel(camera, camera_title(camera), parent=dialog)
        layout.addWidget(zoom_panel, 1)

        if zoom_panel.load_media(path):
            if self.panels.get(camera) is not None:
                try:
                    t = self.panels[camera].get_time()
                except Exception:
                    t = 0
                zoom_panel.set_time(t)
                zoom_panel.play()

        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(dialog.accept)
        layout.addWidget(btn_close)

        dialog.exec()

    def _update_playback_ui(self) -> None:
        if not self.current_event:
            return

        current = max(0, self.sync.current_time())
        total = max(0, self.sync.length())
        if total > 0 and self.position_slider.maximum() != total:
            self.position_slider.setRange(0, total)
        if not self._user_seeking and total > 0:
            self.position_slider.setValue(min(current, total))
        self.lbl_time.setText(f"{self._format_ms(current)} / {self._format_ms(total)}")
        self.btn_play_pause.setText("暂停" if self.sync.is_playing() else "播放")

    def _toggle_analyze(self) -> None:
        if self.analyze_thread and self.analyze_thread.isRunning():
            if self.analyze_worker:
                self.analyze_worker.cancel()
                self.statusBar().showMessage("正在取消分析...", 2000)
            return
        self._start_analyze_worker()

    def _start_analyze_worker(self) -> None:
        if not self.all_events:
            return
        if self.analyze_thread and self.analyze_thread.isRunning():
            return

        roi_map = self.ai_store.get_all_rois()
        self.analyze_thread = QThread(self)
        self.analyze_worker = AnalyzeWorker(list(self.all_events), config.AI_DB_PATH, roi_map)
        self.analyze_worker.moveToThread(self.analyze_thread)

        self.analyze_thread.started.connect(self.analyze_worker.run)
        self.analyze_worker.progress.connect(self._on_analyze_progress)
        self.analyze_worker.result_ready.connect(self._on_analyze_result)
        self.analyze_worker.failed.connect(self._on_analyze_failed)
        self.analyze_worker.finished.connect(self._on_analyze_finished)
        self.analyze_worker.finished.connect(self.analyze_thread.quit)
        self.analyze_thread.finished.connect(self.analyze_thread.deleteLater)
        self.analyze_thread.start()

        self.btn_analyze.setText("取消分析")
        self.lbl_analyze.setText("智能分析: 分析中")
        self.progress_ai.setValue(0)
        self.statusBar().showMessage("开始分析事件...", 2000)

    def _on_analyze_progress(self, current: int, total: int, event_key: str) -> None:
        if total <= 0:
            self.progress_ai.setValue(0)
        else:
            self.progress_ai.setValue(int((current / total) * 100))
        self.lbl_analyze.setText(f"智能分析: {current}/{total} - {event_key}")

    def _on_analyze_result(self, event_key: str, result_obj: object) -> None:
        if not isinstance(result_obj, AIResultRecord):
            return
        self.ai_results[event_key] = result_obj
        self._apply_filter()
        if self.current_event and event_key_from_event(self.current_event) == event_key:
            self._update_event_detail(self.current_event)

    def _on_analyze_failed(self, message: str) -> None:
        QMessageBox.warning(self, "智能分析异常", message)

    def _on_analyze_finished(self) -> None:
        self.btn_analyze.setText("开始分析")
        self.lbl_analyze.setText("智能分析: 完成")
        self.progress_ai.setValue(100)
        self.statusBar().showMessage("智能分析完成", 3000)
        self.analyze_worker = None
        self.analyze_thread = None
    def _open_ai_settings(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("智能分析设置")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()

        current = config.get_user_settings(mask_key=False)
        key_edit = QLineEdit(current.get("dashscope_api_key", ""))
        key_edit.setPlaceholderText("输入 DashScope 接口密钥")
        key_edit.setEchoMode(QLineEdit.Password)
        key_edit.setClearButtonEnabled(True)

        endpoint_edit = QLineEdit(current.get("dashscope_endpoint", ""))
        endpoint_edit.setPlaceholderText("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
        endpoint_edit.setClearButtonEnabled(True)

        model_edit = QLineEdit(current.get("qwen_model_name", ""))
        model_edit.setPlaceholderText("例如 qwen-vl-max-latest")
        model_edit.setClearButtonEnabled(True)

        form.addRow("接口密钥", key_edit)
        form.addRow("接口地址", endpoint_edit)
        form.addRow("模型名称", model_edit)
        layout.addLayout(form)

        btns = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel, parent=dialog)
        btn_save = btns.button(QDialogButtonBox.Save)
        btn_cancel = btns.button(QDialogButtonBox.Cancel)
        if btn_save:
            btn_save.setText("保存")
        if btn_cancel:
            btn_cancel.setText("取消")
        btns.accepted.connect(dialog.accept)
        btns.rejected.connect(dialog.reject)
        layout.addWidget(btns)

        if dialog.exec() != QDialog.Accepted:
            return
        try:
            config.save_user_settings(
                api_key=key_edit.text(),
                endpoint=endpoint_edit.text(),
                model_name=model_edit.text(),
            )
            self.statusBar().showMessage("智能分析配置已保存。", 3000)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "保存失败", f"无法保存智能分析配置:\n{exc}")

    def _open_roi_settings(self) -> None:
        cams = list(dict.fromkeys([*CAMERA_PRIORITY, *(cam for e in self.all_events for cam in e.clips.keys())]))
        rois = self.ai_store.get_all_rois()

        dialog = QDialog(self)
        dialog.setWindowTitle("区域设置（相对坐标 0-1）")
        layout = QVBoxLayout(dialog)
        table = QTableWidget(len(cams), 5, dialog)
        table.setHorizontalHeaderLabels(["摄像头", "x", "y", "宽", "高"])
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)

        spin_boxes: dict[tuple[int, int], QDoubleSpinBox] = {}
        for row, cam in enumerate(cams):
            item = QTableWidgetItem(cam)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            table.setItem(row, 0, item)
            roi = rois.get(cam, {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0})
            for col, key in enumerate(("x", "y", "w", "h"), start=1):
                spin = QDoubleSpinBox()
                spin.setRange(0.0, 1.0)
                spin.setDecimals(3)
                spin.setSingleStep(0.01)
                spin.setValue(float(roi.get(key, 0.0 if key in {"x", "y"} else 1.0)))
                table.setCellWidget(row, col, spin)
                spin_boxes[(row, col)] = spin

        layout.addWidget(table)
        btns = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel, parent=dialog)
        btn_save = btns.button(QDialogButtonBox.Save)
        btn_cancel = btns.button(QDialogButtonBox.Cancel)
        if btn_save:
            btn_save.setText("保存")
        if btn_cancel:
            btn_cancel.setText("取消")
        btns.accepted.connect(dialog.accept)
        btns.rejected.connect(dialog.reject)
        layout.addWidget(btns)

        if dialog.exec() != QDialog.Accepted:
            return

        try:
            for row, cam in enumerate(cams):
                roi = {
                    "x": spin_boxes[(row, 1)].value(),
                    "y": spin_boxes[(row, 2)].value(),
                    "w": max(0.01, spin_boxes[(row, 3)].value()),
                    "h": max(0.01, spin_boxes[(row, 4)].value()),
                }
                self.ai_store.upsert_roi(cam, roi)
            self.statusBar().showMessage("区域配置已保存。", 3000)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "保存失败", f"无法保存区域配置:\n{exc}")

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.analyze_worker:
            self.analyze_worker.cancel()
        if self.analyze_thread and self.analyze_thread.isRunning():
            self.analyze_thread.quit()
            self.analyze_thread.wait(3000)
        if self.scan_thread and self.scan_thread.isRunning():
            self.scan_thread.quit()
            self.scan_thread.wait(3000)
        self.ai_store.close()
        super().closeEvent(event)

    @staticmethod
    def _format_ms(ms: int) -> str:
        seconds = max(ms // 1000, 0)
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
