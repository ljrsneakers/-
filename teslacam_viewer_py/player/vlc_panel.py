from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QMenu, QVBoxLayout, QWidget

try:
    import vlc
except Exception as exc:  # pragma: no cover
    vlc = None
    VLC_IMPORT_ERROR = str(exc)
else:
    VLC_IMPORT_ERROR = ""


class VlcPanel(QWidget):
    playback_error = Signal(str, str)
    zoom_requested = Signal(str, Path | None)
    _placeholder_signal = Signal(str)
    _shared_instance = None
    _shared_error: str | None = None

    def __init__(self, camera_id: str, camera_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.camera_id = camera_id
        self.camera_name = camera_name
        self._player = None
        self._current_path: Path | None = None
        self._event_manager = None
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)
        self._placeholder_signal.connect(self._show_placeholder)
        self._build_ui()
        self._init_player()

    @classmethod
    def _ensure_shared_instance(cls) -> None:
        if cls._shared_instance is not None or cls._shared_error is not None:
            return
        if vlc is None:
            cls._shared_error = (
                f"libVLC 不可用: {VLC_IMPORT_ERROR}" if VLC_IMPORT_ERROR else "未安装 python-vlc。"
            )
            return
        try:
            cls._shared_instance = vlc.Instance("--quiet", "--no-video-title-show")
        except Exception as exc:  # pragma: no cover
            cls._shared_error = f"libVLC 初始化失败: {exc}"

    def _build_ui(self) -> None:
        self.setObjectName(f"vlc_panel_{self.camera_name}")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.title_label = QLabel(self.camera_name)
        self.title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.title_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(self.title_label)

        self.video_surface = QWidget(self)
        self.video_surface.setStyleSheet("background: #111; border: 1px solid #333;")
        self.video_surface.setMinimumSize(320, 180)
        self.video_surface.setAttribute(Qt.WA_NativeWindow, True)
        self.video_surface.setContextMenuPolicy(Qt.CustomContextMenu)
        self.video_surface.customContextMenuRequested.connect(self._on_context_menu)
        layout.addWidget(self.video_surface, 1)

        self.placeholder = QLabel(self.video_surface)
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setStyleSheet(
            "color: #ddd; background: rgba(0, 0, 0, 120); padding: 8px; font-size: 13px;"
        )
        self._show_placeholder("等待加载")

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self.placeholder.setGeometry(self.video_surface.rect())

    def _init_player(self) -> None:
        self._ensure_shared_instance()
        if self._shared_error:
            self._show_placeholder(self._shared_error)
            return
        try:
            self._player = self._shared_instance.media_player_new()
            self._bind_output_window()
            self._event_manager = self._player.event_manager()
            self._event_manager.event_attach(vlc.EventType.MediaPlayerEncounteredError, self._on_vlc_error)
        except Exception as exc:  # pragma: no cover
            self._shared_error = f"创建播放器失败: {exc}"
            self._player = None
            self._show_placeholder(self._shared_error)

    def _bind_output_window(self) -> None:
        if self._player is None:
            return
        wid = int(self.video_surface.winId())
        if sys.platform.startswith("linux"):
            self._player.set_xwindow(wid)
        elif sys.platform == "win32":
            self._player.set_hwnd(wid)
        elif sys.platform == "darwin":
            self._player.set_nsobject(wid)

    def _on_context_menu(self, pos) -> None:
        menu = QMenu(self)
        zoom = menu.addAction("放大查看")
        action = menu.exec(self.mapToGlobal(pos))
        if action == zoom:
            self.zoom_requested.emit(self.camera_id, self._current_path)

    def _show_placeholder(self, text: str) -> None:
        self.placeholder.setText(text)
        self.placeholder.show()

    def _on_vlc_error(self, _event) -> None:  # pragma: no cover
        self._current_path = None
        self._placeholder_signal.emit("缺失")
        self.playback_error.emit(self.camera_name, "该路视频出现 VLC 播放错误。")

    def load_media(self, path: Path | None) -> bool:
        if self._player is None:
            return False
        if path is None or not path.exists():
            self._current_path = None
            self.stop()
            self._show_placeholder("缺失")
            return False
        try:
            media = self._shared_instance.media_new(str(path))
            self._player.set_media(media)
            self._bind_output_window()
            self._current_path = path
            self.placeholder.hide()
            return True
        except Exception as exc:  # pragma: no cover
            self._current_path = None
            self._show_placeholder(f"无法加载视频: {exc}")
            return False

    def clear(self) -> None:
        self._current_path = None
        self.stop()
        self._show_placeholder("缺失")

    def play(self) -> None:
        if self._player is None or self._current_path is None:
            return
        self._player.play()

    def pause(self) -> None:
        if self._player is not None:
            self._player.pause()

    def stop(self) -> None:
        if self._player is not None:
            self._player.stop()

    def set_time(self, ms: int) -> None:
        if self._player is None or self._current_path is None:
            return
        self._player.set_time(max(0, ms))

    def set_rate(self, rate: float) -> bool:
        if self._player is None or self._current_path is None:
            return False
        return bool(self._player.set_rate(rate))

    def get_time(self) -> int:
        if self._player is None:
            return 0
        return max(self._player.get_time(), 0)

    def get_length(self) -> int:
        if self._player is None:
            return 0
        return max(self._player.get_length(), 0)

    def is_playing(self) -> bool:
        if self._player is None:
            return False
        return bool(self._player.is_playing())

    @property
    def has_media(self) -> bool:
        return self._current_path is not None

    @property
    def unavailable_reason(self) -> str | None:
        return self._shared_error
