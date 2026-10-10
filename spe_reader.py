"""
spe_reader.py
Lettore SPE (Princeton Instruments / LightField v3.0 e WinSpec legacy v2.x)
senza dipendenze esterne oltre numpy.

LightField v3.0 (.spe):
    [4100 byte header legacy, spesso vuoto/placeholder] +
    [dati binari: count frame, ognuno width*height*bpp byte] +
    [eventuali pochi byte di padding/metadata] +
    [footer XML che descrive TUTTO: width, height, count, pixelFormat]

Strategia: troviamo il footer XML (tag <SpeFormat ...>), lo parsiamo con
xml.etree (stdlib) per ottenere width/height/count/pixelFormat, poi
leggiamo i dati binari dai primi byte dopo l'header legacy (4100), per le
`count*width*height*bpp` byte attese. Funziona anche se il "padding" finale
tra dati e XML non è esattamente zero (lo ignoriamo, ci basiamo sulla
dimensione attesa dei dati, non sull'offset dell'XML meno l'header).

Se il footer XML non viene trovato, ricade sul parsing legacy a offset fissi
(WinSpec v2.5 puro).
"""

import os
import struct
import numpy as np
import xml.etree.ElementTree as ET

HEADER_SIZE = 4100

PIXELFORMAT_TO_DTYPE = {
    "MonochromeUnsigned16": "<u2",
    "MonochromeUnsigned32": "<u4",
    "MonochromeFloating32": "<f4",
    "MonochromeSigned16":   "<i2",
    "MonochromeSigned32":   "<i4",
}

LEGACY_DTYPE_MAP = {0: "<f4", 1: "<i4", 2: "<i2", 3: "<u2"}
LEGACY_OFFSETS = {
    "datatype":  (108, "<h"),
    "xdim":      (42,  "<H"),
    "ydim":      (656, "<H"),
    "NumFrames": (1446, "<i"),
}


def _read_legacy_field(buf, offset, fmt):
    return struct.unpack_from(fmt, buf, offset)[0]



def _parse_xml_metadata(xml_text):
    """Extract width, height, count, pixelFormat from the SPE 3.0 XML footer."""
    ns = {"s": "http://www.princetoninstruments.com/spe/2009"}
    root = ET.fromstring(xml_text)
    # Outer DataBlock(type=Frame) has count + pixelFormat; inner Region has width/height
    frame_block = root.find(".//s:DataFormat/s:DataBlock", ns)
    if frame_block is None:
        raise ValueError("XML footer found but <DataBlock> (Frame) not located.")
    count = int(frame_block.get("count"))
    pixel_format = frame_block.get("pixelFormat")

    region_block = frame_block.find("s:DataBlock", ns)
    if region_block is None:
        # fallback: width/height sometimes on the frame block itself
        width = int(frame_block.get("width"))
        height = int(frame_block.get("height"))
    else:
        width = int(region_block.get("width"))
        height = int(region_block.get("height"))

    return {"width": width, "height": height, "count": count,
            "pixel_format": pixel_format}


def _find_xml_footer_efficient(path, search_chunk=4_000_000):
    """
    Locate the SPE 3.0 XML footer without loading the whole (multi-GB) file
    into RAM: the footer is always near the END of the file, so we read only
    the last `search_chunk` bytes (grown if needed) and search backwards.
    Returns (xml_text, xml_start_absolute_offset, file_size) or
    (None, None, file_size) if no footer is found.
    """
    fsize = os.path.getsize(path)
    chunk = min(search_chunk, fsize)
    with open(path, "rb") as f:
        while True:
            f.seek(fsize - chunk)
            tail = f.read(chunk)
            start_in_tail = tail.find(b"<SpeFormat")
            if start_in_tail == -1:
                start_in_tail = tail.find(b"<?xml")
            if start_in_tail != -1:
                xml_start_abs = fsize - chunk + start_in_tail
                text = tail[start_in_tail:]
                end_null = text.find(b"\x00")
                if end_null != -1:
                    text = text[:end_null]
                try:
                    return text.decode("utf-8"), xml_start_abs, fsize
                except UnicodeDecodeError:
                    return text.decode("utf-8", errors="ignore"), xml_start_abs, fsize
            if chunk >= fsize:
                return None, None, fsize
            chunk = min(chunk * 4, fsize)


def read_spe_metadata(path):
    """
    Return dict: width, height, count, dtype (numpy dtype string), source
    ('xml' or 'legacy'), and frame_bytes / data_start_offset for reading.

    IMPORTANT: for the 'xml' source, `count` is NOT taken at face value from
    the XML footer's `count="..."` attribute -- that field has been observed
    to be stale/wrong on some LightField exports (e.g. reporting 500 when
    the file actually contains 1000 frames' worth of data). Instead, the
    real frame count is derived from the actual binary data size:
        real_count = (xml_start_offset - data_start_offset) // frame_bytes
    and a mismatch against the XML-declared count is reported via the
    'count_xml_declared' key (None if they agree) so callers can log it.
    """
    xml_text, xml_start, fsize = _find_xml_footer_efficient(path)
    if xml_text is not None:
        meta = _parse_xml_metadata(xml_text)
        dtype = PIXELFORMAT_TO_DTYPE.get(meta["pixel_format"])
        if dtype is None:
            raise ValueError(f"{path}: unrecognised pixelFormat "
                              f"'{meta['pixel_format']}' in XML footer.")
        meta["dtype"] = dtype
        meta["source"] = "xml"
        meta["data_start_offset"] = HEADER_SIZE

        frame_bytes = meta["width"] * meta["height"] * np.dtype(dtype).itemsize
        data_bytes_available = xml_start - HEADER_SIZE
        real_count = data_bytes_available // frame_bytes if frame_bytes else 0
        leftover = data_bytes_available - real_count * frame_bytes

        declared_count = meta["count"]
        if real_count != declared_count:
            meta["count_xml_declared"] = declared_count
            meta["count"] = real_count   # trust the actual bytes, not the XML field
        else:
            meta["count_xml_declared"] = None

        meta["frame_bytes"] = frame_bytes
        meta["leftover_bytes"] = leftover   # should be small (padding); large = suspicious
        return meta

    # ---- fallback: legacy WinSpec v2.x header ----
    with open(path, "rb") as f:
        header = f.read(HEADER_SIZE)
    datatype = _read_legacy_field(header, *LEGACY_OFFSETS["datatype"])
    xdim = _read_legacy_field(header, *LEGACY_OFFSETS["xdim"])
    ydim = _read_legacy_field(header, *LEGACY_OFFSETS["ydim"])
    numframes = _read_legacy_field(header, *LEGACY_OFFSETS["NumFrames"])
    dtype = LEGACY_DTYPE_MAP.get(datatype)
    if dtype is None or xdim == 0 or ydim == 0:
        raise ValueError(
            f"{path}: no XML footer found AND legacy header looks invalid "
            f"(datatype={datatype}, xdim={xdim}, ydim={ydim}). "
            f"Cannot determine frame geometry.")
    return {"width": xdim, "height": ydim, "count": numframes,
            "pixel_format": None, "dtype": dtype, "source": "legacy",
            "data_start_offset": HEADER_SIZE,
            "count_xml_declared": None, "frame_bytes": xdim * ydim * np.dtype(dtype).itemsize,
            "leftover_bytes": None}


def _first_real_frame_index(path, data_start_offset, frame_bytes, n_frames,
                            mean_threshold=20.0):
    """
    Some LightField acquisitions write a preallocated buffer where only the
    LAST portion actually contains real frames -- the leading frames are a
    never-overwritten all-zero (or near-zero) block. Detect the first frame
    whose MEAN value exceeds `mean_threshold` via binary search (assumes a
    single zero-to-signal transition with signal frames at the END of the
    stack, which is the pattern observed in practice -- not a scattered/
    intermittent dropout).

    Uses the frame MEAN rather than its max: the single frame straddling the
    empty/data boundary has been observed to contain an isolated saturated
    pixel (max=65535, i.e. 2**16-1, a sensor glitch/stuck pixel at the
    standby-to-acquisition transition) while the rest of the frame is still
    essentially empty (mean in the single digits) -- a max-based threshold
    would misclassify that glitch frame as "real" and let one corrupt frame
    skew any per-position statistic computed over the kept frames. Real
    OH-PLIF signal frames have mean values in the hundreds-to-low-thousands,
    not the single digits, so a mean threshold cleanly separates them.

    After binary search locates a candidate boundary, the boundary frame
    itself is re-checked against the FOLLOWING frame: if the candidate's
    mean is not within the same order of magnitude as the next frame's mean,
    the candidate is treated as a one-off transition glitch and skipped too.

    Returns 0 if signal is present from the start, or n_frames if NO frame
    in the whole stack has signal (caller should treat that as an empty/
    corrupt acquisition, not silently return zeros).
    """
    def frame_mean(i):
        with open(path, "rb") as f:
            f.seek(data_start_offset + i * frame_bytes)
            buf = f.read(frame_bytes)
        return float(np.frombuffer(buf, dtype="<u2").mean()) if len(buf) == frame_bytes else 0.0

    if n_frames == 0:
        return 0
    if frame_mean(0) > mean_threshold:
        return 0   # signal from the first frame -- nothing to skip
    if frame_mean(n_frames - 1) <= mean_threshold:
        return n_frames   # no signal anywhere in the stack

    lo, hi = 0, n_frames - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if frame_mean(mid) > mean_threshold:
            hi = mid
        else:
            lo = mid + 1

    # belt-and-suspenders: confirm the boundary frame isn't itself a
    # transition glitch (e.g. a stuck/saturated pixel skewing its mean just
    # over threshold while the surrounding frames tell a different story)
    if lo + 1 < n_frames:
        m_here = frame_mean(lo)
        m_next = frame_mean(lo + 1)
        if m_next > mean_threshold and m_here > 0 and (m_next / max(m_here, 1e-9) > 3
                                                        or m_here / max(m_next, 1e-9) > 3):
            lo += 1
    return lo


def count_real_frames(path, mean_threshold=20.0):
    """
    Lightweight check: return (n_real_frames, skipped_leading_empty) without
    reading the full data stack -- just the metadata + binary-search probes
    used by read_spe_stack's leading-empty-block detection. Useful for
    callers that need to size something (e.g. a list of per-frame markers)
    before actually reading pixel data.
    """
    meta = read_spe_metadata(path)
    width, height, count = meta["width"], meta["height"], meta["count"]
    frame_bytes = width * height * np.dtype(meta["dtype"]).itemsize
    if count == 0:
        return 0, 0
    start_frame = _first_real_frame_index(
        path, meta["data_start_offset"], frame_bytes, count, mean_threshold)
    if start_frame == count:
        raise ValueError(
            f"{path}: no frame in the entire {count}-frame stack has "
            f"signal above the noise floor -- acquisition looks empty "
            f"or corrupt.")
    return count - start_frame, start_frame


def read_spe_stack(path, max_frames=None, skip_leading_empty=True):
    """
    Read an .spe file and return a float32 ndarray of shape (N, height, width).
    `max_frames`: optionally cap how many frames are read (from the start
    of the REAL data, after any leading empty block has been skipped).
    `skip_leading_empty`: if True (default), automatically detect and skip
    a leading block of empty/zero frames (observed on some LightField
    acquisitions where a preallocated buffer is only partially filled) so
    callers never silently get an all-zero stack. Set False to disable
    this and always read from the file's nominal frame 0.
    """
    meta = read_spe_metadata(path)
    width, height, count = meta["width"], meta["height"], meta["count"]
    dtype = np.dtype(meta["dtype"])
    frame_bytes = width * height * dtype.itemsize

    start_frame = 0
    if skip_leading_empty and count > 0:
        start_frame = _first_real_frame_index(
            path, meta["data_start_offset"], frame_bytes, count)
        if start_frame == count:
            raise ValueError(
                f"{path}: no frame in the entire {count}-frame stack has "
                f"signal above the noise floor -- acquisition looks empty "
                f"or corrupt.")
        if start_frame > 0:
            meta["skipped_leading_empty_frames"] = start_frame

    n_available = count - start_frame
    n_to_read = n_available if max_frames is None else min(max_frames, n_available)
    read_offset = meta["data_start_offset"] + start_frame * frame_bytes

    with open(path, "rb") as f:
        f.seek(read_offset)
        buf = f.read(frame_bytes * n_to_read)

    expected = frame_bytes * n_to_read
    if len(buf) < expected:
        raise ValueError(
            f"{path}: expected {expected:,} bytes of frame data "
            f"({n_to_read} frames x {frame_bytes:,} bytes, starting at "
            f"frame {start_frame}) but only read {len(buf):,} bytes. "
            f"File may be truncated.")

    arr = np.frombuffer(buf, dtype=dtype, count=n_to_read * width * height)
    stack = arr.reshape(n_to_read, height, width).astype(np.float32)
    return stack, meta


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        meta = read_spe_metadata(p)
        print(f"{p}\n  source={meta['source']}  dtype={meta['dtype']}  "
              f"width={meta['width']}  height={meta['height']}  "
              f"count={meta['count']}  pixel_format={meta.get('pixel_format')}")
        if meta.get("count_xml_declared") is not None:
            print(f"  WARNING: XML footer declared count={meta['count_xml_declared']} "
                  f"but actual binary data contains {meta['count']} frames -- "
                  f"using the real count derived from file size.")
        if meta.get("leftover_bytes"):
            print(f"  leftover_bytes={meta['leftover_bytes']} "
                  f"(unaccounted bytes between last full frame and XML footer)")#!/usr/bin/env python3
