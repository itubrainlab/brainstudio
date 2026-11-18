"""The look: the brAIn lab's dark style shared with brAInStream.

brAInStream dresses itself in g.Pype's dark theme with the lab's red. This module
recreates that look for brAInStudio with its own palette and style sheet (no g.Pype
files are used): a #353535 window, #2a2a2a fields, #2a82da highlight, square
buttons, gradient title bars on docks and group boxes, tabs underlined in blue, and
the brAIn lab red (#BA0022) for the brand.
"""

from __future__ import annotations

from pathlib import Path

from PySide6 import QtGui, QtWidgets

APP_ICON = Path(__file__).resolve().parent.parent / "resources" / "brainstudio.svg"

WINDOW = "#353535"
BASE = "#2a2a2a"
ALTERNATE = "#424242"
DARK = "#232323"
SHADOW = "#141414"
TEXT = "#ffffff"
MUTED = "#7f7f7f"
HIGHLIGHT = "#2a82da"
HIGHLIGHT_HOVER = "#0078cc"
HOVER = "#828282"
BRAND = "#BA0022"          # the brAIn lab's red
RECORDING = "#e53935"

ICON = TEXT
ICON_DISABLED = MUTED


def dark_palette() -> QtGui.QPalette:
    c = QtGui.QColor
    role, dis = QtGui.QPalette.ColorRole, QtGui.QPalette.ColorGroup.Disabled
    p = QtGui.QPalette()
    for r, v in ((role.Window, WINDOW), (role.WindowText, TEXT), (role.Base, BASE),
                 (role.AlternateBase, ALTERNATE), (role.ToolTipBase, BASE),
                 (role.ToolTipText, TEXT), (role.Text, TEXT), (role.Dark, DARK),
                 (role.Shadow, SHADOW), (role.Button, WINDOW), (role.ButtonText, TEXT),
                 (role.BrightText, "#ff0000"), (role.Link, HIGHLIGHT),
                 (role.Highlight, HIGHLIGHT), (role.HighlightedText, TEXT),
                 (role.PlaceholderText, "#8c8c8c"), (role.Mid, "#a0a0a0")):
        p.setColor(r, c(v))
    for r in (role.WindowText, role.Text, role.ButtonText, role.HighlightedText):
        p.setColor(dis, r, c(MUTED))
    p.setColor(dis, role.Highlight, c("#505050"))
    return p


_GRAD_BAR = ("qlineargradient(x1:0, y1:1, x2:0, y2:0, stop:0 rgba(25,25,25,127), "
             "stop:1 rgba(53,53,53,75))")

STYLESHEET = f"""
QWidget {{ font-size: 12px; }}
QPushButton {{
    background: {WINDOW}; border: 1px solid #4a4a4a; border-radius: 0px;
    padding: 4px 12px;
}}
QPushButton:hover {{ background: {HOVER}; border-color: {HOVER}; }}
QPushButton:pressed, QPushButton:checked {{
    background: {HIGHLIGHT}; border-color: {HIGHLIGHT};
}}
QPushButton:checked:hover {{ background: {HIGHLIGHT_HOVER}; border-color: {HIGHLIGHT_HOVER}; }}
QPushButton:disabled {{ color: {MUTED}; background: {ALTERNATE}; border-color: {ALTERNATE}; }}
QPushButton:flat {{ border: none; background: transparent; }}
QPushButton:flat:hover {{ background: rgba(255,255,255,30); }}
QPushButton#Primary {{ background: {HIGHLIGHT}; border: 1px solid {HIGHLIGHT}; font-weight: 600; }}
QPushButton#Primary:hover {{ background: {HIGHLIGHT_HOVER}; border-color: {HIGHLIGHT_HOVER}; }}
QPushButton#Primary:disabled {{ background: {ALTERNATE}; border-color: {ALTERNATE}; }}
QDialogButtonBox QPushButton {{ padding: 4px 24px; }}
QToolButton {{ border: none; padding: 3px; border-radius: 0px; }}
QToolButton:hover {{ background: rgba(255,255,255,30); }}
QToolButton:pressed, QToolButton:checked {{ background: {HIGHLIGHT}; }}
QToolTip {{ color: {TEXT}; background-color: {BASE}; border: 1px solid {HIGHLIGHT}; }}
QStatusBar {{
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(25,25,25,127), stop:1 rgba(53,53,53,75));
    color: #a0a0a0;
}}
QStatusBar::item {{ border: none; }}
QMenuBar {{ background-color: rgb(25,25,25); }}
QMenuBar::item {{ padding: 3px 6px; background: transparent; }}
QMenuBar::item:selected {{ background: rgba(106,106,106,160); }}
QMenuBar::item:pressed {{ background: {HIGHLIGHT}; }}
QMenu {{ background-color: {WINDOW}; border: 1px solid {SHADOW}; }}
QMenu::item {{ padding: 4px 24px; }}
QMenu::item:selected {{ background: {HIGHLIGHT}; }}
QMenu::item:disabled {{ color: {MUTED}; }}
QMenu::separator {{ height: 1px; background: #4a4a4a; margin: 4px 8px; }}
QMainWindow::separator {{ width: 6px; height: 6px; background: {DARK}; }}
QMainWindow::separator:hover, QSplitter::handle:hover {{ background: {HIGHLIGHT}; }}
QSplitter::handle {{ background: transparent; }}
QSplitter::handle:horizontal {{ width: 8px; }}
QSplitter::handle:vertical {{ height: 8px; }}
QDockWidget {{ color: {TEXT}; }}
QDockWidget::title {{
    padding: 5px 6px; text-align: left;
    background-color: qlineargradient(x1:0, y1:1, x2:0, y2:0,
        stop:0 rgba(10,10,10,220), stop:1 rgba(25,25,25,200));
    border: 1px solid rgba(0,0,0,230); border-bottom: 2px solid rgba(0,0,0,255);
}}
QGroupBox {{
    background-color: {ALTERNATE}; margin-top: 26px; padding-top: 4px;
    border: 1px solid rgba(25,25,25,75); border-radius: 0px;
}}
QGroupBox::title {{
    subcontrol-origin: margin; subcontrol-position: left top; padding: 4px 6px;
    background-color: {_GRAD_BAR}; border: 1px solid rgba(25,25,25,75);
    border-bottom: 2px solid rgb(127,127,127);
}}
QTabWidget::pane {{ background-color: rgba(66,66,66,128); border-top: 1px solid rgba(25,25,25,128); }}
QTabWidget::tab-bar {{ left: 3px; }}
QTabBar {{ qproperty-drawBase: 0; }}
QTabBar::tab {{
    padding: 5px 10px; background-color: {_GRAD_BAR};
    border: 1px solid rgba(25,25,25,75); border-radius: 0px;
}}
QTabBar::tab:selected, QTabBar::tab:hover {{
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(53,53,53,127), stop:1 rgba(66,66,66,128));
}}
QTabBar::tab:selected {{ border-bottom: 2px solid {HIGHLIGHT}; }}
QTabBar::tab:!selected {{ margin-top: 2px; color: #c8c8c8; }}
QTreeView, QTableView, QListView {{
    background: {BASE}; alternate-background-color: {WINDOW};
    border: 1px solid rgba(0,0,0,120); gridline-color: #3a3a3a;
    selection-background-color: {HIGHLIGHT};
}}
QHeaderView::section {{
    background-color: {_GRAD_BAR}; color: {TEXT}; border: none;
    border-bottom: 1px solid {DARK}; padding: 3px 5px;
}}
QTableCornerButton::section {{ background: {DARK}; border: none; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
    background: {BASE}; border: 1px solid #4a4a4a; border-radius: 0px; padding: 2px 4px;
    selection-background-color: {HIGHLIGHT};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
QPlainTextEdit:focus {{ border: 1px solid {HIGHLIGHT}; }}
QComboBox QAbstractItemView {{ background: {BASE}; border: 1px solid {SHADOW};
    selection-background-color: {HIGHLIGHT}; }}
QScrollBar:vertical {{ background: {BASE}; width: 14px; margin: 0; }}
QScrollBar:horizontal {{ background: {BASE}; height: 14px; margin: 0; }}
QScrollBar::handle:vertical {{ background: rgb(115,115,116); min-height: 20px; margin: 2px 3px; }}
QScrollBar::handle:horizontal {{ background: rgb(115,115,116); min-width: 20px; margin: 3px 2px; }}
QScrollBar::handle:hover {{ background: {HIGHLIGHT}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
QProgressBar {{ background: {BASE}; border: 1px solid #4a4a4a; height: 10px; }}
QProgressBar::chunk {{ background: {HIGHLIGHT}; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; }}

/* brAIn lab chrome */
#Toolbar {{
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(25,25,25,255), stop:1 rgba(45,45,45,255));
    border: none; border-bottom: 1px solid rgba(0,0,0,200); spacing: 0px;
}}
#Toolbar QToolButton {{ border: none; padding: 4px; }}
#Toolbar QToolButton:hover {{ background: rgba(255,255,255,30); }}
#Toolbar QLabel {{ color: {MUTED}; background: transparent; }}
#Brand {{ color: {TEXT}; font-size: 16px; font-weight: 700; }}
#BrandAI {{ color: {BRAND}; font-size: 16px; font-weight: 800; }}
#BrandSub {{ color: {MUTED}; font-size: 16px; }}
#Toolbar QLabel#ProjectName {{ color: {TEXT}; font-weight: 600; }}
#Divider {{ background: rgba(255,255,255,40); }}
#SidePanel {{ background: {WINDOW}; }}
QLabel#PanelTitle {{
    padding: 5px 8px; color: {TEXT};
    background-color: qlineargradient(x1:0, y1:1, x2:0, y2:0,
        stop:0 rgba(10,10,10,220), stop:1 rgba(25,25,25,200));
    border: 1px solid rgba(0,0,0,230); border-bottom: 2px solid rgba(0,0,0,255);
}}
#Ribbon {{ background: {DARK}; border-bottom: 1px solid rgba(0,0,0,200); }}
#Ribbon QToolButton {{ color: #d0d0d0; padding: 4px 2px; font-size: 11px; }}
#Ribbon QToolButton:hover {{ background: rgba(255,255,255,24); color: {TEXT}; }}
#Ribbon QToolButton:disabled {{ color: {MUTED}; }}
#ColumnTitle {{ color: {MUTED}; font-size: 11px; font-weight: 600; letter-spacing: 1px; }}
#Hint {{ color: {MUTED}; }}
"""


def apply(app: QtWidgets.QApplication) -> None:
    """Dress the application in the brAIn lab's dark style (idempotent)."""
    if app.property("brainlab_style"):
        return
    app.setStyle("Fusion")  # the native macOS style ignores parts of palettes/sheets
    app.setPalette(dark_palette())
    app.setStyleSheet(STYLESHEET)
    app.setWindowIcon(QtGui.QIcon(str(APP_ICON)))
    app.setProperty("brainlab_style", True)
