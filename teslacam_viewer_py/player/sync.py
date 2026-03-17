from __future__ import annotations

import time
from typing import Dict

from PySide6.QtCore import QObject, QTimer

try:
    from player.vlc_panel import VlcPanel
except ModuleNotFoundError:
    from teslacam_viewer_py.player.vlc_panel import VlcPanel


class SyncController(QObject):
    def __init__(self, panels: Dict[str, VlcPanel], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.panels = panels
        self.master_camera = "front"
        self.sync_enabled = True
        self.drift_threshold_ms = 200
        self.playback_rate = 1.0
        self._calibration_suspended_until = 0.0

        self._calibration_timer = QTimer(self)
        self._calibration_timer.setInterval(250)
        self._calibration_timer.timeout.connect(self.calibrate)
        self._calibration_timer.start()

    def set_sync_enabled(self, enabled: bool) -> None:
        self.sync_enabled = enabled

    def set_panels(self, panels: Dict[str, VlcPanel]) -> None:
        self.panels = panels
        if self.master_camera not in self.panels:
            self.select_master("front")
        self.set_rate(self.playback_rate)

    def select_master(self, preferred_camera: str = "front") -> None:
        if preferred_camera in self.panels and self.panels[preferred_camera].has_media:
            self.master_camera = preferred_camera
            return
        for camera, panel in self.panels.items():
            if panel.has_media:
                self.master_camera = camera
                return
        self.master_camera = preferred_camera

    def master_panel(self) -> VlcPanel:
        panel = self.panels.get(self.master_camera)
        if panel is not None:
            return panel
        if self.panels:
            return next(iter(self.panels.values()))
        raise RuntimeError("No VLC panels available")

    def play(self) -> None:
        self.set_rate(self.playback_rate)
        if self.sync_enabled:
            for panel in self.panels.values():
                panel.play()
            return
        self.master_panel().play()

    def pause(self) -> None:
        if self.sync_enabled:
            for panel in self.panels.values():
                panel.pause()
            return
        self.master_panel().pause()

    def stop(self) -> None:
        if self.sync_enabled:
            for panel in self.panels.values():
                panel.stop()
            return
        self.master_panel().stop()

    def seek(self, ms: int, force_all: bool = False) -> None:
        target = max(ms, 0)
        self._suspend_calibration(500)
        if force_all or self.sync_enabled:
            for panel in self.panels.values():
                panel.set_time(target)
            return
        self.master_panel().set_time(target)

    def set_rate(self, rate: float) -> None:
        self.playback_rate = rate
        for panel in self.panels.values():
            panel.set_rate(rate)

    def current_time(self) -> int:
        if not self.panels:
            return 0
        return self.master_panel().get_time()

    def length(self) -> int:
        if not self.panels:
            return 0
        return self.master_panel().get_length()

    def is_playing(self) -> bool:
        if not self.panels:
            return False
        return self.master_panel().is_playing()

    def calibrate(self) -> None:
        if not self.sync_enabled:
            return
        if time.monotonic() < self._calibration_suspended_until:
            return

        master = self.master_panel()
        if not master.has_media:
            return
        master_time = master.get_time()

        for camera, panel in self.panels.items():
            if camera == self.master_camera or not panel.has_media:
                continue
            delta = abs(panel.get_time() - master_time)
            if delta > self.drift_threshold_ms:
                panel.set_time(master_time)

    def _suspend_calibration(self, delay_ms: int) -> None:
        self._calibration_suspended_until = max(
            self._calibration_suspended_until,
            time.monotonic() + (delay_ms / 1000.0),
        )
