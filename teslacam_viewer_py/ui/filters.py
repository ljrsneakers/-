from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QMenu, QToolButton, QWidget

try:
    from core.db_ai import PRIMARY_TYPES, primary_type_label
except ModuleNotFoundError:
    from teslacam_viewer_py.core.db_ai import PRIMARY_TYPES, primary_type_label


class AiFilterBar(QWidget):
    filter_changed = Signal(object, int, bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._actions: dict[str, QAction] = {}
        self._building = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.type_btn = QToolButton()
        self.type_btn.setPopupMode(QToolButton.InstantPopup)
        self.type_btn.setText("\u7c7b\u578b: \u5168\u90e8")
        self.type_menu = QMenu(self)
        self.type_btn.setMenu(self.type_menu)

        action_all = QAction("\u5168\u9009", self.type_menu)
        action_all.triggered.connect(self._select_all_types)
        self.type_menu.addAction(action_all)

        action_none = QAction("\u6e05\u7a7a", self.type_menu)
        action_none.triggered.connect(self._clear_all_types)
        self.type_menu.addAction(action_none)
        self.type_menu.addSeparator()

        for item in PRIMARY_TYPES:
            act = QAction(primary_type_label(item), self.type_menu)
            act.setCheckable(True)
            act.setChecked(True)
            act.toggled.connect(self._on_type_toggle)
            self.type_menu.addAction(act)
            self._actions[item] = act

        self.risk_combo = QComboBox()
        self.risk_combo.addItem("\u98ce\u9669>=0", userData=0)
        self.risk_combo.addItem("\u98ce\u9669>=1", userData=1)
        self.risk_combo.addItem("\u98ce\u9669>=2", userData=2)
        self.risk_combo.addItem("\u98ce\u9669>=3", userData=3)

        self.only_ai_check = QCheckBox("\u4ec5\u663e\u793a\u5df2\u5206\u6790")

        layout.addWidget(QLabel("\u7b5b\u9009"))
        layout.addWidget(self.type_btn)
        layout.addWidget(self.risk_combo)
        layout.addWidget(self.only_ai_check)
        layout.addStretch(1)

        self.risk_combo.currentIndexChanged.connect(self._emit_filter_changed)
        self.only_ai_check.toggled.connect(self._emit_filter_changed)
        self._refresh_type_text()

    def _select_all_types(self) -> None:
        self._building = True
        for action in self._actions.values():
            action.setChecked(True)
        self._building = False
        self._refresh_type_text()
        self._emit_filter_changed()

    def _clear_all_types(self) -> None:
        self._building = True
        for action in self._actions.values():
            action.setChecked(False)
        self._building = False
        self._refresh_type_text()
        self._emit_filter_changed()

    def _on_type_toggle(self, _checked: bool) -> None:
        if self._building:
            return
        self._refresh_type_text()
        self._emit_filter_changed()

    def _refresh_type_text(self) -> None:
        selected = self.selected_types()
        if len(selected) == len(PRIMARY_TYPES):
            self.type_btn.setText("\u7c7b\u578b: \u5168\u90e8")
        elif not selected:
            self.type_btn.setText("\u7c7b\u578b: \u65e0")
        else:
            self.type_btn.setText(f"\u7c7b\u578b: {len(selected)}\u9879")

    def _emit_filter_changed(self) -> None:
        self.filter_changed.emit(self.selected_types(), self.min_risk(), self.only_ai_check.isChecked())

    def selected_types(self) -> set[str]:
        return {name for name, action in self._actions.items() if action.isChecked()}

    def min_risk(self) -> int:
        return int(self.risk_combo.currentData() or 0)

    def current_filter(self) -> tuple[set[str], int, bool]:
        return self.selected_types(), self.min_risk(), self.only_ai_check.isChecked()
