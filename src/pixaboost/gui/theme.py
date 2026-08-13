"""Compact dark palette adapted from the miniDS experiment GUI."""

BACKGROUND = "#111318"
SURFACE = "#191d24"
SURFACE_HIGH = "#222832"
BORDER = "#303846"
TEXT = "#e8edf5"
TEXT_DIM = "#9aa6b7"
ACCENT = "#3b73d1"
SUCCESS = "#39c889"
WARNING = "#f1b84b"
DANGER = "#ff6f70"

STYLESHEET = f"""
QWidget {{
    background: {BACKGROUND};
    color: {TEXT};
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}}
QMainWindow {{ background: {BACKGROUND}; }}
QLabel {{ background: transparent; }}
QLabel#title {{ font-size: 25px; font-weight: 650; }}
QLabel#subtitle, QLabel#dim {{ color: {TEXT_DIM}; }}
QLabel#warning {{ color: {WARNING}; }}
QLabel#result {{
    background: {SURFACE_HIGH};
    border: 1px solid {BORDER};
    border-radius: 7px;
    padding: 9px 11px;
    font-weight: 600;
}}
QLabel#result[status="running"] {{ border-color: {ACCENT}; }}
QLabel#result[status="success"] {{ border-color: {SUCCESS}; color: {SUCCESS}; }}
QLabel#result[status="failure"] {{ border-color: {DANGER}; color: {DANGER}; }}
QLabel#result[status="cancelled"] {{ border-color: {WARNING}; color: {WARNING}; }}
QGroupBox {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 9px;
    margin-top: 14px;
    padding: 15px 12px 12px 12px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {TEXT_DIM};
}}
QComboBox, QLineEdit, QPlainTextEdit, QTreeWidget {{
    background: {SURFACE_HIGH};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 7px;
    selection-background-color: #315f9f;
}}
QComboBox:focus, QLineEdit:focus, QPlainTextEdit:focus, QTreeWidget:focus {{
    border: 2px solid {ACCENT};
    padding: 6px;
}}
QPushButton {{
    background: {SURFACE_HIGH};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 8px 15px;
}}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:focus {{ border: 2px solid {ACCENT}; padding: 7px 14px; }}
QPushButton:disabled {{ color: #657184; background: {SURFACE}; }}
QPushButton#primary {{
    background: {ACCENT}; border-color: {ACCENT}; color: white; font-weight: 600;
}}
QPushButton#primary:disabled {{
    background: {SURFACE}; border-color: {BORDER}; color: #657184;
}}
QPushButton#danger {{ border-color: {DANGER}; color: {DANGER}; }}
QPushButton#danger:disabled {{ border-color: {BORDER}; color: #657184; }}
QProgressBar {{
    background: {SURFACE_HIGH};
    border: 1px solid {BORDER};
    border-radius: 6px;
    height: 20px;
    text-align: center;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 5px; }}
QPlainTextEdit#log {{
    background: #0c0f14;
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 12px;
}}
QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 8px; background: {SURFACE}; }}
QTabBar::tab {{ color: {TEXT_DIM}; padding: 9px 17px; border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom-color: {ACCENT}; }}
QHeaderView::section {{
    background: {SURFACE_HIGH}; color: {TEXT_DIM}; border: none;
    border-bottom: 1px solid {BORDER}; padding: 7px;
}}
QStatusBar {{ background: {SURFACE}; color: {TEXT_DIM}; border-top: 1px solid {BORDER}; }}
QSplitter::handle {{ background: {BORDER}; width: 2px; }}
"""
