"""Minimal XDF 1.0 writer (used for demo data and tests).

Spec: https://github.com/sccn/xdf/wiki/Specifications
"""

from __future__ import annotations

import struct
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

_FMT = {"float32": "<f", "double64": "<d", "int32": "<i", "int16": "<h"}


def _varlen(n: int) -> bytes:
    if n < 256:
        return b"\x01" + struct.pack("<B", n)
    if n < 2**32:
        return b"\x04" + struct.pack("<I", n)
    return b"\x08" + struct.pack("<Q", n)


def _chunk(tag: int, payload: bytes) -> bytes:
    body = struct.pack("<H", tag) + payload
    return _varlen(len(body)) + body


def write_xdf(path: str | Path, streams: list[dict]) -> None:
    """Write streams to XDF.

    Each stream dict: name, type, srate (0 = irregular), format ("float32",
    "double64", "string"), labels (list[str]), units (optional list),
    timestamps (n,), data (n, ch) array or list of strings for markers.
    """
    out = bytearray(b"XDF:")
    out += _chunk(1, b'<?xml version="1.0"?><info><version>1.0</version></info>')
    for sid, s in enumerate(streams, start=1):
        labels = s.get("labels") or ["ch1"]
        units = s.get("units") or [""] * len(labels)
        ctypes = s.get("ch_types") or [s["type"]] * len(labels)
        chans = "".join(
            f"<channel><label>{escape(l)}</label><unit>{escape(u)}</unit>"
            f"<type>{escape(t)}</type></channel>"
            for l, u, t in zip(labels, units, ctypes))
        hdr = (f'<?xml version="1.0"?><info><name>{escape(s["name"])}</name>'
               f'<type>{escape(s["type"])}</type><channel_count>{len(labels)}</channel_count>'
               f'<nominal_srate>{s["srate"]}</nominal_srate>'
               f'<channel_format>{s["format"]}</channel_format>'
               f'<source_id>brainstudio-{sid}</source_id><created_at>0</created_at>'
               f'<desc><channels>{chans}</channels></desc></info>')
        out += _chunk(2, struct.pack("<I", sid) + hdr.encode())
        ts = np.asarray(s["timestamps"], float)
        fmt = s["format"]
        payload = bytearray(struct.pack("<I", sid) + _varlen(len(ts)))
        for i, t in enumerate(ts):
            payload += b"\x08" + struct.pack("<d", t)
            if fmt == "string":  # one length-prefixed string per channel
                row = s["data"][i]
                for v in (row if isinstance(row, (list, tuple)) else [row]):
                    b = str(v).encode()
                    payload += _varlen(len(b)) + b
            else:
                payload += np.asarray(s["data"][i], dtype=_FMT[fmt][1:]).astype(
                    _FMT[fmt]).tobytes()
        out += _chunk(3, bytes(payload))
        # clock offsets: zero (single machine)
        for t in (ts[0], ts[-1]):
            out += _chunk(4, struct.pack("<I", sid) + struct.pack("<dd", t, 0.0))
        footer = (f'<?xml version="1.0"?><info><first_timestamp>{ts[0]}</first_timestamp>'
                  f'<last_timestamp>{ts[-1]}</last_timestamp>'
                  f'<sample_count>{len(ts)}</sample_count></info>')
        out += _chunk(6, struct.pack("<I", sid) + footer.encode())
    Path(path).write_bytes(bytes(out))
