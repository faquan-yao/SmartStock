"""同花顺配色，白底。阳线红边空心，阴线青色实心。"""

from __future__ import annotations

import matplotlib
from matplotlib import font_manager
from matplotlib.axes import Axes

BG = "#ffffff"
GRID = "#e6e6e6"
TEXT = "#222222"
SPINE = "#b0b0b0"
GUIDE = "#c8c8c8"
UP = "#ff3232"
DOWN = "#00a3a3"
SEGMENT = "#e69500"
PREDICT_NEGATIVE = "#1a9a1a"
CLOSE = "#222222"

MA_COLORS = {
    "ma5": "#222222",
    "ma10": "#c4a000",
    "ma20": "#c000c0",
    "ma60": "#1a9a1a",
    "ma120": "#2a7fd4",
    "ma250": "#666666",
}
VOL_COLORS = {
    "vol_ma5": "#222222",
    "vol_ma10": "#c4a000",
    "vol_ma20": "#c000c0",
}
LINE_COLORS = ("#222222", "#c4a000", "#c000c0", "#1a9a1a", "#2a7fd4", "#e07000")


def configure_font() -> None:
    """优先使用系统里的中文字体，避免标题变成方框。"""
    available = {font.name for font in font_manager.fontManager.ttflist}
    chosen = "DejaVu Sans"
    for name in ("Microsoft YaHei", "SimHei", "SimSun", "PingFang SC", "Noto Sans CJK SC"):
        if name in available:
            chosen = name
            break
    matplotlib.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def apply_axes(ax: Axes) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=8, labelright=True, labelleft=False)
    ax.yaxis.tick_right()
    for spine in ax.spines.values():
        spine.set_color(SPINE)
    ax.grid(True, color=GRID, linestyle=":", linewidth=0.5)
    ax.yaxis.get_major_formatter().set_useOffset(False)
    ax.title.set_color(TEXT)
