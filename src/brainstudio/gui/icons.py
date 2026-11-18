"""Icons drawn with QPainter (no image files), in the brAIn lab style: white line
glyphs on the dark chrome, greyed when disabled."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF

from . import style

SIZE = 64


def _icon(draw, color: str = style.ICON) -> QIcon:
    icon = QIcon()
    for mode, tint in ((QIcon.Mode.Normal, color), (QIcon.Mode.Disabled, style.ICON_DISABLED)):
        pm = QPixmap(SIZE, SIZE)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        draw(p, QColor(tint), SIZE)
        p.end()
        icon.addPixmap(pm, mode)
    return icon


def _pen(p, c, s, w=0.07):
    p.setPen(QPen(c, s * w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                  Qt.PenJoinStyle.RoundJoin))
    p.setBrush(Qt.BrushStyle.NoBrush)


def app() -> QIcon:
    return QIcon(str(style.APP_ICON))


def folder(color=style.ICON) -> QIcon:
    """A dataset folder (BIDS)."""
    def draw(p, c, s):
        _pen(p, c, s)
        path = QPainterPath()
        path.moveTo(s * .08, s * .25)
        path.lineTo(s * .38, s * .25)
        path.lineTo(s * .46, s * .34)
        path.lineTo(s * .92, s * .34)
        path.lineTo(s * .92, s * .82)
        path.lineTo(s * .08, s * .82)
        path.closeSubpath()
        p.drawPath(path)
        for k, y in enumerate((.48, .58, .68)):
            p.drawLine(QPointF(s * .22, s * y), QPointF(s * (.78 - .1 * k), s * y))
    return _icon(draw, color)


def file(color=style.ICON) -> QIcon:
    """A recording file with a signal."""
    def draw(p, c, s):
        _pen(p, c, s)
        path = QPainterPath()
        path.moveTo(s * .2, s * .08)
        path.lineTo(s * .62, s * .08)
        path.lineTo(s * .82, s * .28)
        path.lineTo(s * .82, s * .92)
        path.lineTo(s * .2, s * .92)
        path.closeSubpath()
        p.drawPath(path)
        p.drawPolyline(QPolygonF([QPointF(s * .62, s * .08), QPointF(s * .62, s * .28),
                                  QPointF(s * .82, s * .28)]))
        p.drawPolyline(QPolygonF([QPointF(s * x, s * y) for x, y in (
            (.3, .6), (.4, .6), (.45, .45), (.52, .75), (.58, .52), (.63, .6), (.72, .6))]))
    return _icon(draw, color)


def eye_plus(color=style.ICON) -> QIcon:
    """Attach eye-tracking data."""
    def draw(p, c, s):
        _pen(p, c, s)
        path = QPainterPath()
        path.moveTo(s * .06, s * .5)
        path.quadTo(s * .42, s * .12, s * .78, s * .5)
        path.quadTo(s * .42, s * .88, s * .06, s * .5)
        p.drawPath(path)
        p.setBrush(c)
        p.drawEllipse(QPointF(s * .42, s * .5), s * .11, s * .11)
        _pen(p, c, s, 0.08)
        p.drawLine(QPointF(s * .84, s * .1), QPointF(s * .84, s * .34))
        p.drawLine(QPointF(s * .72, s * .22), QPointF(s * .96, s * .22))
    return _icon(draw, color)


def cap(color=style.ICON) -> QIcon:
    """A head seen from above with electrodes: the montage."""
    def draw(p, c, s):
        _pen(p, c, s)
        p.drawEllipse(QPointF(s * .5, s * .54), s * .36, s * .38)
        p.drawPolyline(QPolygonF([QPointF(s * .42, s * .17), QPointF(s * .5, s * .06),
                                  QPointF(s * .58, s * .17)]))
        p.setBrush(c)
        for x, y in ((.5, .3), (.3, .45), (.5, .5), (.7, .45), (.36, .7), (.64, .7),
                     (.5, .78)):
            p.drawEllipse(QPointF(s * x, s * y), s * .045, s * .045)
    return _icon(draw, color)


def image(color=style.ICON) -> QIcon:
    """A stimulus picture."""
    def draw(p, c, s):
        _pen(p, c, s)
        p.drawRect(QRectF(s * .1, s * .18, s * .8, s * .64))
        p.drawPolyline(QPolygonF([QPointF(s * x, s * y) for x, y in (
            (.14, .76), (.38, .48), (.55, .66), (.66, .55), (.86, .76))]))
        p.setBrush(c)
        p.drawEllipse(QPointF(s * .7, s * .34), s * .06, s * .06)
    return _icon(draw, color)


def export(color=style.ICON) -> QIcon:
    """Export a table (arrow out of a tray)."""
    def draw(p, c, s):
        _pen(p, c, s)
        p.drawPolyline(QPolygonF([QPointF(s * .14, s * .55), QPointF(s * .14, s * .88),
                                  QPointF(s * .86, s * .88), QPointF(s * .86, s * .55)]))
        p.drawLine(QPointF(s * .5, s * .66), QPointF(s * .5, s * .1))
        p.drawPolyline(QPolygonF([QPointF(s * .3, s * .3), QPointF(s * .5, s * .1),
                                  QPointF(s * .7, s * .3)]))
    return _icon(draw, color)


def camera(color=style.ICON) -> QIcon:
    """Snapshot of the screen view."""
    def draw(p, c, s):
        _pen(p, c, s)
        path = QPainterPath()
        path.moveTo(s * .08, s * .3)
        path.lineTo(s * .32, s * .3)
        path.lineTo(s * .4, s * .18)
        path.lineTo(s * .6, s * .18)
        path.lineTo(s * .68, s * .3)
        path.lineTo(s * .92, s * .3)
        path.lineTo(s * .92, s * .82)
        path.lineTo(s * .08, s * .82)
        path.closeSubpath()
        p.drawPath(path)
        p.drawEllipse(QPointF(s * .5, s * .56), s * .16, s * .16)
    return _icon(draw, color)


def question(color=style.ICON) -> QIcon:
    def draw(p, c, s):
        _pen(p, c, s)
        p.drawEllipse(QPointF(s * .5, s * .5), s * .4, s * .4)
        path = QPainterPath()
        path.moveTo(s * .38, s * .38)
        path.cubicTo(s * .38, s * .22, s * .64, s * .22, s * .63, s * .38)
        path.cubicTo(s * .62, s * .5, s * .5, s * .5, s * .5, s * .6)
        p.drawPath(path)
        p.setBrush(c)
        p.drawEllipse(QPointF(s * .5, s * .74), s * .04, s * .04)
    return _icon(draw, color)
