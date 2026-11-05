"""Zoom helpers shared by the Position monomer and Position coordinates canvases.

Mouse wheel zooms around the cursor, buttons zoom around the centre, and Reset view
always returns to the whole image. The matplotlib toolbar's Home button is also
anchored to the whole image, so it cannot get stuck on a zoomed view.
"""
from __future__ import annotations

from ..._shared import QtWidgets


class CanvasZoom:
    WHEEL_STEP = 1.25          # zoom factor per wheel notch
    MAX_OUT = 1.15             # never show more than the image plus a small margin
    MIN_SPAN_PX = 3.0          # never zoom in past a few pixels

    def __init__(self, canvas, ax, image_shape, toolbar=None):
        self.canvas, self.ax, self.toolbar = canvas, ax, toolbar
        self.shape = tuple(image_shape)
        if toolbar is not None:
            # Make the toolbar's Home the full image even if the first zoom is ours.
            toolbar.push_current()
        self._cid = canvas.mpl_connect("scroll_event", self._on_scroll)

    def disconnect(self):
        if self._cid is not None:
            self.canvas.mpl_disconnect(self._cid)
            self._cid = None

    def full_limits(self):
        h, w = self.shape
        return (-0.5, w - 0.5), (-0.5, h - 0.5)

    def reset(self):
        (x0, x1), (y0, y1) = self.full_limits()
        self._apply(x0, x1, y0, y1)

    def zoom(self, factor, center=None):
        """factor < 1 zooms in, factor > 1 zooms out; ``center`` is (col, row) data."""
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        cx = (x0 + x1) / 2.0 if center is None else center[0]
        cy = (y0 + y1) / 2.0 if center is None else center[1]
        (fx0, fx1), (fy0, fy1) = self.full_limits()
        full_w, full_h = fx1 - fx0, fy1 - fy0
        # Keep the point under the cursor fixed while scaling.
        nx0, nx1 = cx + (x0 - cx) * factor, cx + (x1 - cx) * factor
        ny0, ny1 = cy + (y0 - cy) * factor, cy + (y1 - cy) * factor
        width, height = abs(nx1 - nx0), abs(ny1 - ny0)
        if width > full_w * self.MAX_OUT or height > full_h * self.MAX_OUT:
            self.reset()
            return
        if min(width, height) < self.MIN_SPAN_PX:
            return
        self._apply(nx0, nx1, ny0, ny1)

    def _apply(self, x0, x1, y0, y1):
        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y0, y1)
        if self.toolbar is not None:
            self.toolbar.push_current()
        self.canvas.draw_idle()

    def _on_scroll(self, event):
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return
        steps = getattr(event, "step", 0) or (1 if event.button == "up" else -1)
        factor = (1.0 / self.WHEEL_STEP) ** steps if steps else 1.0
        self.zoom(factor, (event.xdata, event.ydata))


def zoom_buttons(layout, zoom: CanvasZoom, parent=None):
    """Reset view / Zoom out / Zoom in row, with a hint about the wheel and Home."""
    row = QtWidgets.QHBoxLayout()
    for label, slot, tip in (
            ("Reset view", zoom.reset, "Show the whole image again."),
            ("Zoom out", lambda: zoom.zoom(2.0), "Zoom out around the centre."),
            ("Zoom in", lambda: zoom.zoom(0.5), "Zoom in around the centre.")):
        button = QtWidgets.QPushButton(label, parent)
        button.setToolTip(tip)
        button.clicked.connect(lambda _checked=False, slot=slot: slot())
        row.addWidget(button)
    hint = QtWidgets.QLabel("Mouse wheel zooms at the cursor. The house icon also resets.")
    hint.setStyleSheet("color: gray;")
    row.addWidget(hint, 1)
    layout.addLayout(row)
