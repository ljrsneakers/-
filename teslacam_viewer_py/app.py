import sys
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

try:
    from ui.main_window import MainWindow
except ModuleNotFoundError:
    from teslacam_viewer_py.ui.main_window import MainWindow


def _icon_path() -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "teslacam_viewer_py" / "desktop_icon.png"  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent / "desktop_icon.png"


def main() -> int:
    app = QApplication(sys.argv)
    icon_file = _icon_path()
    if icon_file.exists():
        app.setWindowIcon(QIcon(str(icon_file)))
    window = MainWindow()
    if icon_file.exists():
        window.setWindowIcon(QIcon(str(icon_file)))
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
