#!/usr/bin/env python3
"""
MRMS LLCC nowcast - a radar traffic light for the Cape.

CloudScope classifies FORECAST cloud from model condensate and turns it into a probability.
This does the observational counterpart: it reads the latest MRMS radar and lightning,
applies the NASA-STD-4010 Lightning Launch Commit Criteria that a radar can adjudicate, and
scores every grid cell as if the flight path ran through it:

    RED     a rule is violated
    YELLOW  no rule violated, but one would be if its standoff were 2 nmi longer, or
            there is cloud-to-ground lightning within 20 nmi
    GREEN   neither
    GREY    outside radar coverage - unknown, not clear

Alongside the traffic light it records WHAT each echo is - a cloud class read from how far up
the isotherm stack the echo reaches - so the reason for a colour is visible, not just the
colour.

THE 0 dBZ GATE
    Every rule is assessed against echo >= 0 dBZ. NASA-STD-4010 defines a non-transparent cloud
    by radar return, so the radar is the reference the rules are written against rather than a
    proxy for them.

WHAT THE PROBES ESTABLISHED (6 Oct 2026)
    - Every product sits on one 3500x7000, 0.01 deg CONUS grid. No regridding anywhere.
    - Sentinels differ by product: reflectivity marks "no echo" with -99; VII, echo tops,
      composite height and lightning use -1.
    - The GRIB2 carries no names or units, so units are set here: dBZ, kg/m2, echo top in
      KILOMETRES, composite height in METRES.
    - The 0 to -20 C isotherm slices have no vertical gaps and see ~99% of echo on a convective
      afternoon. VII only registers in the strongest cells, so it marks cores, not anvils.
    - Model_0degC_Height is 8 MB, dense and hourly, so it is fetched once an hour and cached.

CONSERVATIVE ANVIL BASE
    Radar cannot tell an anvil from the precipitation falling out of it. Any echo at the 0 C
    slice beneath an anvil therefore counts as the anvil reaching 0 C, so LLCCR 18's "entirely
    colder than 0 C" exception is rarely granted. That is the agreed choice.

FAIL CLOSED
    If composite reflectivity, the 0/-10/-20 C slices or lightning cannot be read, no frame is
    published. Treating a missing slice as "no echo" turned a 55 dBZ core green in testing.

NOT EVALUATED
    Surface electric fields (4.1.2); debris clouds (4.1.6); smoke plumes (4.1.9);
    triboelectrification (4.1.10); the 3-hour lightning clocks on the anvil rules, since only
    30 minutes of lightning is published. Green means no radar-visible violation, never GO.
"""

import datetime
import gzip
import json
import logging
import os
import re
import tempfile
import time

import numpy as np
import requests
from scipy.ndimage import distance_transform_edt, label, maximum_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
ROOT = "https://mrms.ncep.noaa.gov/2D"
ROOT3 = "https://mrms.ncep.noaa.gov/3DRefl"
UA = {"User-Agent": "CloudScope-MRMS/2.0 (launch weather nowcast)"}
OUT_DIR = os.environ.get("OUT_DIR", "site")

VIEWER_VERSION_EXPECTED = "mrms-v10"

DOMAIN = {"lat_min": 27.6, "lat_max": 29.6, "lon_min": -81.6, "lon_max": -79.6}

SITES = {
    "LC-39A":  (28.6084, -80.6043),
    "LC-39B":  (28.6272, -80.6208),
    "SLC-41":  (28.5833, -80.5834),
    "SLC-40":  (28.5619, -80.5772),
    "SLC-37B": (28.5317, -80.5657),
    "SLC-20":  (28.5085, -80.5546),
    "LZ-1":    (28.4857, -80.5444),
    "SLC-36":  (28.4707, -80.5379),
    "SLC-46":  (28.4584, -80.5271),
    "KTTS":    (28.6150, -80.6944),
    "KXMR":    (28.4675, -80.5664),
}

FRAMES_KEPT = 12             # the loop length

# Detached-anvil ORIGIN. A detached anvil is an anvil that came from a convective core and has
# separated from it - not merely floating echo with no core under it right now. The loop is the
# evidence: echo counts as detached anvil only if it lies where an ATTACHED anvil was in a
# recent frame, allowing for drift. Drift is set generously, so the error falls on the side of
# calling something anvil rather than missing one.
HISTORY_MIN = 60             # look back over the last hour of frames
DRIFT_KMH = 60.0             # ~32 kt: how far an anvil may have moved since an earlier frame
FRAME_SPACING_MIN = 5        # spacing of backfilled frames
FREEZING_MAX_AGE_MIN = 70

# Archive backfill. On a fresh deployment the loop would otherwise hold one frame for the
# first hour, filling one per run. MRMS keeps a few hours of timestamped files beside every
# .latest, so missing frames are rebuilt from those. Bounded per run because the workflow
# cancels an in-progress run when the next trigger arrives - a backfill that tries to do all
# eleven frames at once gets cancelled and publishes nothing.
BACKFILL_PER_RUN = 4
BACKFILL_BUDGET_S = 90      # setup (apt, pip) ~1 min, latest 2-D frame ~30 s, the 3-D volume
                            # another ~30 s - and the whole run has to finish inside the
                            # 5-minute trigger interval or the next trigger cancels it

# 3-D merged reflectivity. Measured 6 Oct 2026: 33 levels from 0.5 to 19 km, all from one scan
# time, 14.2 MB gzipped and ~30 s to fetch and decode, on the same 0.01 deg grid as the 2-D
# products so every column lines up with the traffic light. Fetched for the NEWEST frame each
# run only - archive frames stay 2-D, and the loop accumulates volumes one run at a time.
#
# Two sentinels, and they mean different things: -99 is no echo, -999 is NO COVERAGE - the beam
# cannot see that low there (it appeared only at 0.5-2.0 km). A column with -999 under its
# lowest echo has an unknown base, not a high one, which matters for "rained out".
VOL_NO_ECHO, VOL_NO_COVER, VOL_NONE = 255, 254, 253
VOL_MIN_LEVELS = 25        # fewer usable levels than this and the volume is dropped
MATCH_TOL_MIN = 3.0          # how far a product's file may sit from the frame's valid time

# (product, no-echo sentinel, units)
PRODUCTS = {
    "comp":  ("MergedReflectivityQCComposite",    -99.0, "dBZ"),
    "r0":    ("Reflectivity_0C",                  -99.0, "dBZ"),
    "r5":    ("Reflectivity_-5C",                 -99.0, "dBZ"),
    "r10":   ("Reflectivity_-10C",                -99.0, "dBZ"),
    "r15":   ("Reflectivity_-15C",                -99.0, "dBZ"),
    "r20":   ("Reflectivity_-20C",                -99.0, "dBZ"),
    "high":  ("LayerCompositeReflectivity_High",  -99.0, "dBZ"),
    "super": ("LayerCompositeReflectivity_Super", -99.0, "dBZ"),
    "hmax":  ("HeightCompositeReflectivity",       -1.0, "m"),
    "et18":  ("EchoTop_18",                        -1.0, "km"),
    "vii":   ("VII",                               -1.0, "kg/m2"),
    "cg":    ("NLDN_CG_030min_AvgDensity",         -1.0, "fl/km2/min"),
}
FREEZING = "Model_0degC_Height"

# Products whose absence would make a frame read greener than reality. Without any of these
# the frame is withheld.
ESSENTIAL = ("comp", "r0", "r10", "r20", "cg")

# --------------------------------------------------------------------------------------
# LLCC thresholds - NASA-STD-4010 (2017-06-27)
# --------------------------------------------------------------------------------------
LLCC = {
    "lightning_nm": 10.0,          # 4.1.1
    "lightning_watch_nm": 20.0,    # yellow only
    "cumulus_5nm": 5.0,            # LLCCR 16: top colder than -10 C within 5 nmi
    "cumulus_10nm": 10.0,          # LLCCR 17: top colder than -20 C within 10 nmi
    "plus5_c": 5.0,                # LLCCR 15: through cumulus topping at <= +5 C
    "attached_3nm": 3.0,           # LLCCR 18
    "attached_lightning_nm": 10.0, # LLCCR 19/20 while lightning is active
    "detached_3nm": 3.0,           # LLCCR 22
    "excep_nm": 5.0,               # exception: anvil within 5 nmi entirely colder than 0 C
    "mrr_dbz": 7.5,                # exception: MRR < +7.5 dBZ within 1 nmi
    "mrr_search_nm": 4.0,          # 4.2.2c
    "mrr_eval_nm": 1.0,
    "disturbed_nm": 5.0,           # 4.1.7
    "disturbed_dbz": 30.0,
    "core_dbz": 40.0,
    "yellow_margin_nm": 2.0,
}
LAPSE_C_PER_KM = 6.5

# Rule order is also the bit order in the per-cell data file - change both together.
RULE_KEYS = ["lightning", "cumulus_through", "cumulus_5nm", "cumulus_10nm",
             "attached_anvil", "detached_anvil", "disturbed", "thick_layer"]
RULE_NAMES = {
    "lightning":       "Lightning within 10 nmi (4.1.1)",
    "cumulus_through": "Flight through cumulus topping at or colder than +5 °C (4.1.3.1)",
    "cumulus_5nm":     "Cumulus topping colder than −10 °C within 5 nmi (4.1.3.2)",
    "cumulus_10nm":    "Cumulus topping colder than −20 °C within 10 nmi (4.1.3.3)",
    "attached_anvil":  "Attached anvil within 3 nmi (4.1.4)",
    "detached_anvil":  "Detached anvil within 3 nmi (4.1.5)",
    "disturbed":       "Disturbed weather (4.1.7)",
    "thick_layer":     "Thick cloud layer spanning 0 to −10 °C (4.1.8)",
}
# How each rule is tested on the radar grid, in the viewer's own words.
RULE_HOW = {
    "lightning":       "Any cloud-to-ground flash in the NLDN 30-minute density within 10 nmi.",
    "cumulus_through": "The point itself is in cumulus (not anvil) echo reaching an isotherm "
                       "level, or whose strongest return sits above the +5 °C height.",
    "cumulus_5nm":     "Cumulus or cumulonimbus echo of 0 dBZ or more at the −10 °C level "
                       "within 5 nmi. Anvil echo never counts, however cold it reaches.",
    "cumulus_10nm":    "Cumulus or cumulonimbus echo of 0 dBZ or more at the −20 °C level "
                       "within 10 nmi. Anvil echo never counts.",
    "attached_anvil":  "An attached anvil cell within 3 nmi, unless the exception holds: no echo "
                       "at the 0 °C level beneath any anvil within 5 nmi, and the largest "
                       "reflectivity within 1 nmi below 7.5 dBZ. Also violated by an attached "
                       "anvil within 10 nmi while lightning is within 10 nmi.",
    "detached_anvil":  "A detached anvil cell within 3 nmi, with the same exception.",
    "disturbed":       "The point is in echo whose top is colder than 0 °C, and there is 30 dBZ or "
                       "more within 5 nmi.",
    "thick_layer":     "The point has echo at both ends of a 10-degree span inside the 0 to "
                       "−20 °C band (0 and −10, −5 and −15, or −10 and −20 °C): a layer at least "
                       "~5,000 ft deep. Not applied to anvil.",
}
WATCH_HOW = ("Watch means no rule is broken, but one would be if its standoff were 2 nmi longer, "
             "or there is a cloud-to-ground flash within 20 nmi.")

# Rules deliberately not scored. Disturbed weather is applied to synoptic disturbances such as
# fronts, which radar alone cannot identify; on radar the test reduced to "30 dBZ within
# 5 nmi" and mislabelled convective scenes with a rule meant for something else. Measured
# over a convective loop it never once made a cell red on its own, so dropping it changes no
# colours - only removes a misleading reason.
RULES_OFF = ["disturbed"]

NOT_EVALUATED = [
    "Disturbed weather (4.1.7) - applied to synoptic disturbances such as fronts, which radar "
    "alone cannot identify",
    "Surface electric fields (4.1.2) and their field-mill exceptions",
    "Debris clouds (4.1.6), which need an observed detachment time",
    "Smoke plumes (4.1.9)",
    "Triboelectrification (4.1.10)",
    "The 3-hour lightning clocks on the anvil rules - only 30 minutes of lightning is published",
]

# Cloud class, read off how far up the isotherm stack an echo reaches. These are what the
# radar can actually say about an echo, and each maps onto the rules it drives.
CLASSES = [
    {"id": 0, "key": "clear", "name": "No echo", "color": "#00000000",
     "how": "Composite reflectivity below 0 dBZ. No rule applies here."},
    {"id": 1, "key": "warm", "name": "Shallow shower, below freezing", "color": "#94A3B8",
     "how": "Echo of 0 dBZ or more that shows up at none of the isotherm levels: the column's "
            "strongest echo sits below the measured 0 °C height. Can still trip flight-through "
            "cumulus if it reaches the +5 °C level, about 770 m below freezing."},
    {"id": 2, "key": "cu0", "name": "Cumulus topping 0 to −10 °C", "color": "#22D3EE",
     "how": "Echo of 0 dBZ or more at the 0 °C level, but none at −10 °C, so the top lies "
            "somewhere between 0 and −10 °C."},
    {"id": 3, "key": "cu10", "name": "Cumulus topping −10 to −20 °C", "color": "#3B82F6",
     "how": "Echo at the −10 °C level, but none at −20 °C, so the top lies between −10 and "
            "−20 °C. Drives the 5 nmi cumulus standoff."},
    {"id": 4, "key": "cu20", "name": "Cumulus topping below −20 °C", "color": "#A78BFA",
     "how": "Echo of 0 dBZ or more at the −20 °C level, so the top is colder than −20 °C. "
            "Drives the 10 nmi cumulus standoff."},
    {"id": 5, "key": "core", "name": "Convective core", "color": "#FF4FD8",
     "how": "Composite reflectivity of 40 dBZ or more, or any vertically integrated ice. VII only "
            "registers in strong cells, so it confirms a core rather than finding anvil. Takes "
            "precedence over every other class."},
    {"id": 6, "key": "att", "name": "Attached anvil", "color": "#FB923C",
     "how": "Echo weaker than 40 dBZ that reaches the −20 °C level, or sits only aloft (its "
            "strongest echo above the −20 °C height, or in the upper layer composite), and is "
            "joined to a convective core by unbroken echo."},
    {"id": 7, "key": "det", "name": "Detached anvil", "color": "#F5F5F4",
     "how": "An anvil that came from a convective core and has separated from it. Floating "
            "upper-level echo, no echo at 0 °C beneath it, no unbroken path back to a core now - "
            "and lying where an attached anvil was in the last hour, allowing ~32 kt of drift. "
            "Unconnected echo rooted down through 0 °C is a tower and stays cumulus."},
    {"id": 8, "key": "elev", "name": "Elevated layer, not convective", "color": "#C9A66B",
     "how": "Floating echo above freezing with no convective origin in the last hour of the "
            "loop: a mid- or upper-level layer cloud, not anvil. Scored by the thick-cloud-layer "
            "rule when it spans about 4,500 ft of the 0 to −20 °C band, never by the anvil or "
            "cumulus standoffs."},
]
# Radar cannot reliably tell convective from layered cloud, so a broad stratiform shield that
# reaches -20 C is classed as "cumulus topping below -20 C". The classes are named for the
# rule each one drives, which is what the vertical reach of the echo decides.
CLASS_NOTE = ("Classes come from how high the echo reaches through the isotherm levels, checked "
              "in this order: core, attached anvil, detached anvil, then cumulus by depth, then "
              "shallow shower. Once an anvil, always an anvil: anvil echo is scored only by the "
              "anvil rules and never trips a cumulus standoff. The convective core is the "
              "cumulonimbus tower and does count as cumulus, so a point near a core can still "
              "be within a cumulus standoff because of the core. Radar cannot reliably tell "
              "convective from layered cloud, so a broad rain shield reaching −20 °C is classed "
              "with cumulus topping below −20 °C.")
# Echo-top level, the vertical reach the class is built from.
# What the five isotherm levels say about the echo top, as a bracket. Radar samples only at
# 0, -5, -10, -15 and -20 C, so the honest statement is "echo here, none at the next level
# up" - the top lies between them. Index = echo-top level in the data file.
TOP_LEVELS = ["No echo",
              "Below the 0 °C level: no echo at any isotherm",
              "Between 0 and −5 °C: echo at 0 °C, none at −5 °C",
              "Between −5 and −10 °C: echo at −5 °C, none at −10 °C",
              "Between −10 and −15 °C: echo at −10 °C, none at −15 °C",
              "Between −15 and −20 °C: echo at −15 °C, none at −20 °C",
              "Colder than −20 °C: echo at the −20 °C level",
              "Only above the −20 °C level: elevated echo"]

STATUS = {-1: ("No coverage", "#6B7785"), 0: ("Clear", "#3FB97A"),
          1: ("Watch", "#F2C14E"), 2: ("Violating", "#E5484D")}

# Reflectivity colours with the bottom bin at 0 dBZ, as 45 WS displays it: the standard table
# shifted down 5 dB so cyan starts at 0. Every rule here is gated on 0 dBZ, so the weakest echo
# that matters is the first colour on the scale. (The earlier grey 0-5 band read as missing
# data.)
REFL_LEVELS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 95]
REFL_COLORS = ["#04E9E7", "#019FF4", "#0300F4", "#02FD02", "#01C501", "#008E00", "#FDF802",
               "#E5BC00", "#FD9500", "#FD0000", "#D40000", "#BC0000", "#F800FD", "#9854C6",
               "#FDFDFD"]

# Map chrome, tuned for the dark viewer. Layers are rendered on a transparent background so
# they can be stacked: the panel behind them supplies the colour.
COAST = "#8FA3B6"
PAD_INK = "#E8EEF3"
HALO = "#141D29"


# --------------------------------------------------------------------------------------
# Fetch and decode
# --------------------------------------------------------------------------------------
def _session():
    s = requests.Session()
    s.mount("https://", requests.adapters.HTTPAdapter(max_retries=3))
    s.headers.update(UA)
    return s


def fetch_url(sess, url):
    r = sess.get(url, timeout=90)
    r.raise_for_status()
    return gzip.decompress(r.content)


def latest_url(product):
    return f"{ROOT}/{product}/MRMS_{product}.latest.grib2.gz"


_STAMP_RE = re.compile(r'href="(MRMS_[^"]+?_(\d{8}-\d{6})\.grib2\.gz)"')


def listing(sess, product, cache):
    """[(datetime, url)] of the timestamped files MRMS keeps for a product, newest first."""
    if product in cache:
        return cache[product]
    try:
        r = sess.get(f"{ROOT}/{product}/", timeout=60)
        r.raise_for_status()
        out = []
        for name, stamp in _STAMP_RE.findall(r.text):
            t = datetime.datetime.strptime(stamp, "%Y%m%d-%H%M%S")
            out.append((t, f"{ROOT}/{product}/{name}"))
        out.sort(reverse=True)
    except Exception as e:
        logging.warning(f"listing {product}: {type(e).__name__}: {e}")
        out = []
    cache[product] = out
    return out


def decode_crop(raw, fill):
    """Decode one MRMS GRIB2 and cut out DOMAIN by index arithmetic.

    The grid definition says where every cell is, so the crop is computed from the first grid
    point and the increments rather than by building two 24.5-million-element latlons arrays.
    Returns (field north-up, lat_axis, lon_axis, valid_iso).
    """
    import pygrib
    fd, path = tempfile.mkstemp(suffix=".grib2")
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    try:
        grbs = pygrib.open(path)
        g = grbs.message(1)
        vals = g.values
        if np.ma.isMaskedArray(vals):
            vals = vals.filled(fill if fill is not None else np.nan)
        vals = np.asarray(vals, dtype=np.float32)
        try:
            valid = g.validDate.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            valid = None
        try:
            la1 = float(g["latitudeOfFirstGridPointInDegrees"])
            lo1 = float(g["longitudeOfFirstGridPointInDegrees"])
            dj = float(g["jDirectionIncrementInDegrees"])
            di = float(g["iDirectionIncrementInDegrees"])
            jpos = int(g["jScansPositively"])
            fast = True
        except Exception:
            fast = False
        if not fast:
            lats, lons = g.latlons()
        grbs.close()
    finally:
        os.remove(path)

    ny, nx = vals.shape
    if fast:
        lo1 = lo1 - 360.0 if lo1 > 180.0 else lo1
        lat_axis = (la1 + np.arange(ny) * dj) if jpos else (la1 - np.arange(ny) * dj)
        lon_axis = lo1 + np.arange(nx) * di
    else:
        lons = np.where(lons > 180, lons - 360.0, lons)
        lat_axis, lon_axis = lats[:, 0], lons[0, :]

    rows = np.where((lat_axis >= DOMAIN["lat_min"]) & (lat_axis <= DOMAIN["lat_max"]))[0]
    cols = np.where((lon_axis >= DOMAIN["lon_min"]) & (lon_axis <= DOMAIN["lon_max"]))[0]
    if rows.size == 0 or cols.size == 0:
        raise ValueError("domain not on this grid")
    # .copy(), not a slice. A slice is a VIEW, and a view keeps its whole parent alive: every
    # 200x200 crop was pinning the full 3500x7000 CONUS grid, 98 MB apiece. Twelve 2-D products
    # held ~1.2 GB, and the 33-level volume would have added ~3.2 GB more - enough to get the
    # run OOM-killed on a 7 GB runner. Found when the test harness itself was killed.
    sub = vals[rows.min():rows.max() + 1, cols.min():cols.max() + 1].copy()
    la = lat_axis[rows.min():rows.max() + 1].copy()
    lo = lon_axis[cols.min():cols.max() + 1].copy()
    if la[0] < la[-1]:
        sub, la = sub[::-1].copy(), la[::-1].copy()
    return sub, la, lo, valid


def freezing_level(sess, prev, state_dir):
    """The 0 C height, refetched only when the cached copy is over an hour old."""
    path = os.path.join(state_dir, "z0.npy")
    stamp = (prev.get("freezing") or {}).get("fetched")
    if stamp and os.path.exists(path):
        age = (datetime.datetime.now(datetime.timezone.utc)
               - datetime.datetime.fromisoformat(stamp)).total_seconds() / 60.0
        if age < FREEZING_MAX_AGE_MIN:
            return np.load(path), prev["freezing"], False
    z0, _, _, valid = decode_crop(fetch_url(sess, latest_url(FREEZING)), None)
    os.makedirs(state_dir, exist_ok=True)
    np.save(path, z0.astype(np.float32))
    return z0, {"fetched": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "valid": valid}, True


def list_levels_3d(sess):
    """[(directory name, height km)] from the /3DRefl/ index. Listed rather than hard-coded:
    the exact directory spelling (zero-padding of the height) is not worth guessing."""
    r = sess.get(ROOT3 + "/", timeout=60)
    r.raise_for_status()
    found = set(re.findall(r'href="(MergedReflectivityQC_(\d+\.\d+))/"', r.text))
    return sorted(((n, float(h)) for n, h in found), key=lambda t: t[1])


def fetch_volume(sess, levels, shape):
    """The newest 3-D volume cropped to DOMAIN: (vol[L,ny,nx] float32, heights, valid, note).
    Returns (None, ...) if too few levels could be read to trust it."""
    vol, heights, valids, failed = [], [], set(), []
    for name, h in levels:
        url = f"{ROOT3}/{name}/MRMS_{name}.latest.grib2.gz"
        try:
            sub, _, _, v = decode_crop(fetch_url(sess, url), -999.0)
        except Exception as e:
            failed.append(f"{h:g}")
            logging.warning(f"3-D {h:g} km: {type(e).__name__}: {e}")
            continue
        if sub.shape != shape:
            failed.append(f"{h:g}")
            continue
        vol.append(sub)
        heights.append(h)
        if v:
            valids.add(v)
    if len(vol) < VOL_MIN_LEVELS:
        return None, None, None, f"only {len(vol)} of {len(levels)} levels readable"
    note = (f"{len(vol)} levels" + (f", failed {failed}" if failed else "")
            + (f", {len(valids)} distinct valid times" if len(valids) > 1 else ""))
    return np.stack(vol), np.array(heights), (max(valids) if valids else None), note


def encode_volume(vol):
    """uint8 per cell: dBZ*2 (0-252), 255 no echo, 254 no coverage."""
    enc = np.where(vol >= 0, np.clip(np.round(vol * 2), 0, 252), VOL_NO_ECHO).astype(np.uint8)
    enc[vol <= -900] = VOL_NO_COVER
    return enc


def column_base_top(enc):
    """Echo base and top as level indices per column.

    base: index of the lowest level with echo, +64 if any level below it is NO COVERAGE - the
          true base may be lower than the radar can see. 255 = no echo in the column.
    top:  index of the highest level with echo. 255 = no echo.
    """
    echo = enc <= 252
    has = echo.any(axis=0)
    L = enc.shape[0]
    b = np.argmax(echo, axis=0)
    t = L - 1 - np.argmax(echo[::-1], axis=0)
    blind = np.zeros(has.shape, bool)
    for k in range(L):
        blind |= (enc[k] == VOL_NO_COVER) & (k < b)
    base = np.where(has, b + np.where(blind, 64, 0), 255).astype(np.uint8)
    top = np.where(has, t, 255).astype(np.uint8)
    return base, top


# --------------------------------------------------------------------------------------
# LLCC evaluation
# --------------------------------------------------------------------------------------
def _spacing(la, lo):
    lat_mid = float(np.mean(la))
    dlat_km = abs(float(la[0] - la[1])) * 111.32
    dlon_km = abs(float(lo[1] - lo[0])) * 111.32 * np.cos(np.radians(lat_mid))
    return dlat_km, dlon_km


def _disc(nm, dlat_km, dlon_km):
    km = nm * 1.852
    rj = max(1, int(np.ceil(km / dlat_km)))
    ri = max(1, int(np.ceil(km / dlon_km)))
    jj, ii = np.mgrid[-rj:rj + 1, -ri:ri + 1]
    return np.hypot(jj * dlat_km, ii * dlon_km) <= km


def attached_history(frames, valid_iso, la, lo):
    """Where attached anvil has been in the last HISTORY_MIN minutes, grown by drift.

    Read from the class plane of earlier frames' data files. Returns None when no earlier frame
    is available at all - evaluate() then falls back to the conservative behaviour.
    """
    t = datetime.datetime.strptime(valid_iso, "%Y-%m-%dT%H:%M:%SZ")
    dlat_km, dlon_km = _spacing(la, lo)
    n = la.size * lo.size
    seen, union = False, np.zeros((la.size, lo.size), bool)
    for f in frames:
        try:
            dt = (t - datetime.datetime.strptime(f["valid"], "%Y-%m-%dT%H:%M:%SZ")
                  ).total_seconds() / 60.0
        except Exception:
            continue
        if not (0 < dt <= HISTORY_MIN):
            continue
        path = os.path.join(OUT_DIR, f.get("data", ""))
        try:
            raw = np.fromfile(path, np.uint8)
        except Exception:
            continue
        if raw.size < 2 * n:
            continue
        seen = True
        att = raw[n:2 * n].reshape(la.size, lo.size) == 6
        if att.any():
            r_nm = DRIFT_KMH * dt / 60.0 / 1.852
            union |= maximum_filter(att.astype(np.uint8),
                                    footprint=_disc(r_nm, dlat_km, dlon_km)) > 0
    return union if seen else None


def evaluate(F, z0, la, lo, history=None):
    """Traffic light, per-rule grids, cloud class and echo-top level for every cell."""
    L = LLCC
    dlat_km, dlon_km = _spacing(la, lo)

    def near(mask, nm):
        if nm <= 0:
            return mask.copy()
        return maximum_filter(mask.astype(np.uint8),
                              footprint=_disc(nm, dlat_km, dlon_km)) > 0

    def peak(field, nm):
        return maximum_filter(field, footprint=_disc(nm, dlat_km, dlon_km))

    comp = F["comp"]
    echo = comp >= 0.0
    nodata = comp <= -900.0
    e = {k: F[k] >= 0.0 for k in ("r0", "r5", "r10", "r15", "r20")}
    any_iso = e["r0"] | e["r5"] | e["r10"] | e["r15"] | e["r20"]
    hmax = F["hmax"]

    z5 = z0 - L["plus5_c"] / LAPSE_C_PER_KM * 1000.0
    z20 = z0 + 20.0 / LAPSE_C_PER_KM * 1000.0

    aloft = echo & ~any_iso & ((hmax >= z20) | (F["super"] >= 0.0))
    top_below_0c = any_iso | aloft
    top_to_plus5 = any_iso | (echo & (hmax >= z5))

    core = (comp >= L["core_dbz"]) | (F["vii"] > 0.0)
    anvil = ((e["r20"] | aloft) & ~core) & echo
    lab, n = label(anvil | core, structure=np.ones((3, 3)))
    has_core = np.zeros(n + 1, bool)
    if n:
        has_core[np.unique(lab[core])] = True
        has_core[0] = False
    attached = anvil & has_core[lab]
    # Detached anvil is a FLOATING ice cloud. Unconnected echo that is rooted - with echo down
    # through the 0 C level - is a tower, not an anvil, and stays cumulus. The first version
    # called every unconnected -20 C echo a detached anvil, so "cumulus topping below -20 C"
    # could never occur and a 25 dBZ tower with no core nearby got a 3 nmi anvil standoff
    # instead of the 10 nmi cumulus one: the unsafe direction. A decayed shield that is still
    # raining out now reads as cumulus - the more conservative call - until loop history can
    # keep it anvil properly.
    floating = anvil & ~has_core[lab] & ~e["r0"]
    # Origin: floating echo is a detached anvil only where an attached anvil was recently
    # (the drift-grown history mask). With no history at all - the first runs after deploying
    # - fall back to calling it detached anvil, the conservative choice.
    if history is None:
        detached = floating
        elevated = np.zeros(comp.shape, bool)
    else:
        detached = floating & history
        elevated = floating & ~history
    anvil = attached | detached

    # LLCCR 18 exception, CONSERVATIVE: any echo at the 0 C slice under an anvil within 5 nmi
    # fails it, because radar cannot separate the anvil from the precipitation it drops.
    anvil_warm = near((attached | detached) & e["r0"], L["excep_nm"])
    mrr = peak(np.where(comp > -90, comp, -99.0).astype(np.float32), L["mrr_search_nm"])
    mrr_ok = peak(mrr, L["mrr_eval_nm"]) < L["mrr_dbz"]
    exception = ~anvil_warm & mrr_ok

    lightning = F["cg"] > 0.0

    # Once an anvil, always an anvil. Anvil echo is scored ONLY by the anvil rules - it is not
    # cumulus, so it must never trip a cumulus standoff, however cold it reaches. The first
    # version keyed every cumulus test on any echo at the isotherm, so an anvil point got
    # scored as cumulus by its own anvil echo. The parent cumulonimbus tower IS cumulus, so a
    # point near a core can still trip the cumulus rules - from the core, not the anvil.
    anvil_any = attached | detached
    # An elevated layer is not cumulus either: it must not drive a cumulus standoff.
    not_cu = anvil_any | elevated
    cu_echo = echo & ~not_cu
    cu10 = e["r10"] & ~not_cu
    cu20 = e["r20"] & ~not_cu
    # Thick cloud layer: echo spanning at least ~4,500 ft inside the 0 to -20 C band. Adjacent
    # isotherm slices 10 C apart are ~1.5 km (~5,000 ft) apart, so echo at both ends of any
    # 10-degree span counts. The first version only tested 0 to -10 C, so an elevated layer
    # spanning -5 to -15 C was never scored at all.
    thick = ((e["r0"] & e["r10"]) | (e["r5"] & e["r15"]) | (e["r10"] & e["r20"])) & ~anvil_any

    def rules(m):
        lit = near(lightning, L["lightning_nm"] + m)
        return {
            "lightning":       lit,
            "cumulus_through": near(cu_echo & top_to_plus5, m),
            "cumulus_5nm":     near(cu10, L["cumulus_5nm"] + m),
            "cumulus_10nm":    near(cu20, L["cumulus_10nm"] + m),
            "attached_anvil":  (near(attached, L["attached_3nm"] + m) & ~exception)
                               | (near(attached, L["attached_lightning_nm"] + m) & lit),
            "detached_anvil":  near(detached, L["detached_3nm"] + m) & ~exception,
            # Off: see RULES_OFF. Kept as an all-false grid so the bit layout of the per-cell
            # data file does not shift under the frames already in the loop.
            "disturbed":       np.zeros(comp.shape, bool),
            "thick_layer":     near(thick, m),
        }

    red_rules = rules(0.0)
    yel_rules = rules(L["yellow_margin_nm"])
    red = np.logical_or.reduce([red_rules[k] for k in RULE_KEYS])
    yellow = (np.logical_or.reduce([yel_rules[k] for k in RULE_KEYS])
              | near(lightning, L["lightning_watch_nm"])) & ~red

    status = np.zeros(comp.shape, np.int8)
    status[yellow] = 1
    status[red] = 2
    status[nodata] = -1

    # Echo-top level: the coldest isotherm slice still carrying echo.
    top = np.zeros(comp.shape, np.uint8)
    top[echo] = 1
    for lvl, k in ((2, "r0"), (3, "r5"), (4, "r10"), (5, "r15"), (6, "r20")):
        top[e[k]] = lvl
    top[aloft] = 7

    cls = np.zeros(comp.shape, np.uint8)
    cls[echo & (top == 1)] = 1
    cls[(top == 2) | (top == 3)] = 2
    cls[(top == 4) | (top == 5)] = 3
    cls[top == 6] = 4
    cls[top == 7] = 7 if history is None else 8   # aloft-only echo: detached or elevated layer
    cls[elevated] = 8
    cls[detached] = 7
    cls[attached] = 6
    cls[core] = 5
    cls[~echo] = 0

    dist_km = distance_transform_edt(~echo, sampling=(dlat_km, dlon_km))
    diag = {"attached": attached, "detached": detached, "elevated": elevated,
            "core": core, "echo": echo,
            "dist_echo_nm": dist_km / 1.852, "lightning": lightning}
    return status, red_rules, yel_rules, cls, top, diag


# Significance order for summarising what surrounds a pad. Deliberately NOT the class id:
# the first version took the highest id inside the window, and since detached anvil has the
# highest id, a few detached specks outranked a sky full of attached anvil and cores.
CLASS_RANK = {"core": 8, "att": 7, "det": 6, "cu20": 5, "cu10": 4, "elev": 3, "cu0": 2,
              "warm": 1}
SPECK_PCT = 1.0      # classes covering less of the 10 nmi disc than this are not listed


def pad_report(status, red_rules, yel_rules, cls, top, diag, F, la, lo):
    out = {}
    dlat_km, dlon_km = _spacing(la, lo)
    win = _disc(10.0, dlat_km, dlon_km)
    hw = win.shape[0] // 2, win.shape[1] // 2
    names = {c["id"]: c["name"] for c in CLASSES}
    keys = {c["id"]: c["key"] for c in CLASSES}
    for name, (plat, plon) in SITES.items():
        j = int(np.argmin(np.abs(la - plat)))
        i = int(np.argmin(np.abs(lo - plon)))
        j0, j1 = max(0, j - hw[0]), min(la.size, j + hw[0] + 1)
        i0, i1 = max(0, i - hw[1]), min(lo.size, i + hw[1] + 1)
        # A true 10 nmi disc, trimmed where the pad sits near the domain edge. The first
        # version used the bounding square, whose corners reach ~14 nmi.
        disc = win[(j0 - j + hw[0]):(j1 - j + hw[0]), (i0 - i + hw[1]):(i1 - i + hw[1])]
        box = F["comp"][j0:j1, i0:i1][disc]
        cbox = cls[j0:j1, i0:i1][disc]
        dbz10 = float(box.max()) if box.size and box.max() > -90 else None
        present = []
        for cid in np.unique(cbox):
            cid = int(cid)
            if cid == 0:
                continue
            pct = 100.0 * float((cbox == cid).sum()) / cbox.size
            if pct >= SPECK_PCT:
                present.append({"key": keys[cid], "name": names[cid], "pct": round(pct)})
        present.sort(key=lambda c: -CLASS_RANK[c["key"]])
        out[name] = {
            "status": int(status[j, i]),
            "red": [k for k in RULE_KEYS if red_rules[k][j, i]],
            "yellow": [k for k in RULE_KEYS if yel_rules[k][j, i] and not red_rules[k][j, i]],
            "class_here": names[int(cls[j, i])],
            "classes_10nm": present,
            "class_10nm": present[0]["name"] if present else None,
            "max_dbz_10nm": None if dbz10 is None or dbz10 < 0 else round(dbz10, 1),
            "nearest_echo_nm": round(float(diag["dist_echo_nm"][j, i]), 1),
            "lightning_10nm": bool(red_rules["lightning"][j, i]),
        }
    return out


# --------------------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------------------
def _figure(la, lo):
    pc = ccrs.PlateCarree()
    proj = ccrs.Mercator(central_longitude=0.5 * (DOMAIN["lon_min"] + DOMAIN["lon_max"]))
    x0, y0 = proj.transform_point(DOMAIN["lon_min"], DOMAIN["lat_min"], pc)
    x1, y1 = proj.transform_point(DOMAIN["lon_max"], DOMAIN["lat_max"], pc)
    h_in = 6.8
    fig = plt.figure(figsize=(h_in * (x1 - x0) / (y1 - y0), h_in), dpi=140)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1], projection=proj)
    ax.set_extent([DOMAIN["lon_min"], DOMAIN["lon_max"],
                   DOMAIN["lat_min"], DOMAIN["lat_max"]], crs=pc)
    ax.patch.set_alpha(0.0)
    try:
        ax.spines["geo"].set_visible(False)
    except Exception:
        pass
    LO, LA = np.meshgrid(lo, la)
    return fig, ax, pc, LO, LA


def _chrome(ax, pc, coast=True, pads=True):
    if coast:
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor=COAST,
                       linewidth=0.8, zorder=6)
    if pads:
        for _, (plat, plon) in SITES.items():
            ax.plot(plon, plat, marker="+", markersize=6, markeredgewidth=1.3, color=PAD_INK,
                    transform=pc, zorder=8,
                    path_effects=[pe.withStroke(linewidth=2.6, foreground=HALO)])


def _save(fig, path):
    fig.savefig(path, transparent=True)
    plt.close(fig)


def render_layers(status, cls, F, diag, la, lo, stem):
    """Four transparent layers, identical in extent so the viewer can stack or pair them.

    status      traffic light, filled
    statusline  traffic light as OUTLINES only - for laying over reflectivity. Standard radar
                colours include red, yellow and green, so filling status on top of them would
                be unreadable; edges are not.
    radar       composite reflectivity, standard colours plus a 0-5 dBZ band
    class       cloud class
    """
    paths = {}

    fig, ax, pc, LO, LA = _figure(la, lo)
    keys = [-1, 0, 1, 2]
    cmap = mcolors.ListedColormap([STATUS[k][1] for k in keys])
    norm = mcolors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], len(keys))
    ax.pcolormesh(LO, LA, status, cmap=cmap, norm=norm, shading="nearest",
                  transform=pc, zorder=2, alpha=0.78)
    _chrome(ax, pc)
    paths["status"] = f"{stem}_status.png"
    _save(fig, os.path.join(OUT_DIR, paths["status"]))

    fig, ax, pc, LO, LA = _figure(la, lo)
    for level, color, lw in ((1.5, STATUS[2][1], 1.6), (0.5, STATUS[1][1], 1.1)):
        try:
            ax.contour(LO, LA, status.astype(float), levels=[level], colors=[color],
                       linewidths=[lw], transform=pc, zorder=7)
        except Exception:
            pass
    paths["statusline"] = f"{stem}_statusline.png"
    _save(fig, os.path.join(OUT_DIR, paths["statusline"]))

    fig, ax, pc, LO, LA = _figure(la, lo)
    comp = np.where(F["comp"] >= 0, F["comp"], np.nan)
    rcmap = mcolors.ListedColormap(REFL_COLORS)
    rnorm = mcolors.BoundaryNorm(REFL_LEVELS, len(REFL_COLORS))
    ax.pcolormesh(LO, LA, comp, cmap=rcmap, norm=rnorm, shading="nearest",
                  transform=pc, zorder=2)
    if diag["lightning"].any():
        jj, ii = np.where(diag["lightning"])
        ax.scatter(lo[ii], la[jj], marker="x", s=9, linewidths=0.9, color="#FFFFFF",
                   transform=pc, zorder=9)
    _chrome(ax, pc)
    paths["radar"] = f"{stem}_radar.png"
    _save(fig, os.path.join(OUT_DIR, paths["radar"]))

    fig, ax, pc, LO, LA = _figure(la, lo)
    ccmap = mcolors.ListedColormap([c["color"] for c in CLASSES])
    cnorm = mcolors.BoundaryNorm(np.arange(-0.5, len(CLASSES) + 0.5), len(CLASSES))
    ax.pcolormesh(LO, LA, np.where(cls > 0, cls, np.nan), cmap=ccmap, norm=cnorm,
                  shading="nearest", transform=pc, zorder=2)
    _chrome(ax, pc)
    paths["class"] = f"{stem}_class.png"
    _save(fig, os.path.join(OUT_DIR, paths["class"]))
    return paths


def write_data(stem, status, cls, top, F, red_rules, yel_rules, vbase=None, vtop=None):
    """Per-cell readout for the viewer: eight uint8 planes, north-up, row-major.

        0 status + 1        (0 no coverage, 1 clear, 2 watch, 3 violating)
        1 cloud class id
        2 composite dBZ * 2, 255 = no echo
        3 echo-top level    (index into TOP_LEVELS)
        4 red rule bits     (bit n = RULE_KEYS[n])
        5 yellow rule bits
        6 3-D echo base     (level index, +64 if the radar is blind below; 255 none; 253 no volume)
        7 3-D echo top      (level index; 255 none; 253 no volume)

    Older frames have only the first six; the viewer checks the length.
    """
    dbz = np.where(F["comp"] >= 0, np.clip(np.round(F["comp"] * 2), 0, 254), 255)
    rbits = np.zeros(status.shape, np.uint8)
    ybits = np.zeros(status.shape, np.uint8)
    for b, k in enumerate(RULE_KEYS):
        rbits |= (red_rules[k].astype(np.uint8) << b)
        ybits |= ((yel_rules[k] & ~red_rules[k]).astype(np.uint8) << b)
    none = np.full(status.shape, VOL_NONE, np.uint8)
    planes = [(status + 1).astype(np.uint8), cls.astype(np.uint8), dbz.astype(np.uint8),
              top.astype(np.uint8), rbits, ybits,
              vbase if vbase is not None else none, vtop if vtop is not None else none]
    rel = f"{stem}.bin"
    with open(os.path.join(OUT_DIR, rel), "wb") as fp:
        fp.write(b"".join(p.tobytes() for p in planes))
    return rel


# --------------------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------------------
def build_frame(sess, sources, z0, levels=None, prior=None):
    """One complete frame from a {key: url} mapping.

    Always returns (frame, la, lo). On failure all three are None - one shape on every path,
    so no caller can unpack a None by mistake.
    """
    F, valid_times, la, lo = {}, {}, None, None
    for key, url in sources.items():
        product, sentinel, _ = PRODUCTS[key]
        if url is None:
            F[key] = None
            continue
        try:
            fld, fla, flo, v = decode_crop(fetch_url(sess, url), sentinel)
        except Exception as e:
            logging.warning(f"{product}: {type(e).__name__}: {e}")
            F[key] = None
            continue
        if la is None:
            la, lo = fla, flo
        elif fld.shape != (la.size, lo.size):
            logging.warning(f"{product}: grid {fld.shape} differs; skipped")
            F[key] = None
            continue
        F[key] = fld
        valid_times[key] = v
    if la is None:
        logging.error("no product could be read for this frame")
        return None, None, None
    gone = [PRODUCTS[k][0] for k in ESSENTIAL if F.get(k) is None]
    if gone:
        logging.error(f"essential product(s) missing: {gone}; frame withheld rather than "
                      f"published greener than reality")
        return None, None, None
    missing = [PRODUCTS[k][0] for k, v in F.items() if v is None]
    for k, v in F.items():
        if v is None:
            F[k] = np.full((la.size, lo.size), PRODUCTS[k][1], np.float32)
    if z0 is None or z0.shape != (la.size, lo.size):
        z0 = np.full((la.size, lo.size), 4800.0, np.float32)

    valid0 = valid_times.get("comp") or max((v for v in valid_times.values() if v), default=None)
    history = attached_history(prior or [], valid0, la, lo) if valid0 else None
    status, red_rules, yel_rules, cls, top, diag = evaluate(F, z0, la, lo, history)
    pads = pad_report(status, red_rules, yel_rules, cls, top, diag, F, la, lo)

    valid = valid_times.get("comp") or max((v for v in valid_times.values() if v), default=None)
    stamp = valid.replace(":", "").replace("-", "")[:13]
    stem = f"frames/{stamp}"
    images = render_layers(status, cls, F, diag, la, lo, stem)

    # The 3-D volume, for the newest frame only. Its failure never costs the 2-D frame.
    vbase = vtop = vol_rel = vol_valid = None
    vol_note = "not fetched"
    if levels:
        try:
            vol, heights, vol_valid, vol_note = fetch_volume(sess, levels, (la.size, lo.size))
            if vol is not None:
                enc = encode_volume(vol)
                vbase, vtop = column_base_top(enc)
                vol_rel = f"{stem}.vol"
                # gzip, under a plain extension so Pages serves it as bytes rather than
                # setting Content-Encoding and decompressing it behind the viewer's back
                with open(os.path.join(OUT_DIR, vol_rel), "wb") as fp:
                    fp.write(gzip.compress(enc.tobytes(), compresslevel=6))
                _VOL_LEVELS[:] = [float(h) for h in heights]
        except Exception as e:
            vol_note = f"failed: {type(e).__name__}: {e}"
        logging.info(f"3-D volume: {vol_note}")
    data = write_data(stem, status, cls, top, F, red_rules, yel_rules, vbase, vtop)

    vt = [datetime.datetime.strptime(v, "%Y-%m-%dT%H:%M:%SZ")
          for v in valid_times.values() if v]
    skew = round((max(vt) - min(vt)).total_seconds() / 60.0, 1) if len(vt) > 1 else 0.0
    counts = {str(k): int((status == k).sum()) for k in (-1, 0, 1, 2)}
    cls_counts = {c["key"]: int((cls == c["id"]).sum()) for c in CLASSES}
    frame = {"valid": valid, "stamp": stamp, "images": images, "data": data,
             "pads": pads, "counts": counts, "class_counts": cls_counts,
             "missing": missing, "skew_min": skew,
             "volume": vol_rel, "volume_valid": vol_valid, "volume_note": vol_note,
             "freezing_km": round(float(np.nanmean(z0)) / 1000.0, 2)}
    red_pads = [n for n, p in pads.items() if p["status"] == 2]
    yel_pads = [n for n, p in pads.items() if p["status"] == 1]
    logging.info(f"frame {valid}: pads violating {red_pads or '-'} watch {yel_pads or '-'}; "
                 f"skew {skew} min" + (f"; missing {missing}" if missing else ""))
    return frame, la, lo


_VOL_LEVELS = []     # heights of the levels in this run's volume, for the manifest


def _nearest(items, t, tol_min):
    best, gap = None, None
    for ti, url in items:
        g = abs((ti - t).total_seconds()) / 60.0
        if g <= tol_min and (gap is None or g < gap):
            best, gap = url, g
    return best


def backfill_sources(sess, have_valid, newest_valid, cache, budget):
    """Archive sources for the frames the loop is missing, newest gap first."""
    base = datetime.datetime.strptime(newest_valid, "%Y-%m-%dT%H:%M:%SZ")
    have = [datetime.datetime.strptime(v, "%Y-%m-%dT%H:%M:%SZ") for v in have_valid if v]
    comp_list = listing(sess, PRODUCTS["comp"][0], cache)
    out = []
    for k in range(1, FRAMES_KEPT):
        if len(out) >= budget:
            break
        target = base - datetime.timedelta(minutes=FRAME_SPACING_MIN * k)
        if any(abs((h - target).total_seconds()) < FRAME_SPACING_MIN * 30 for h in have):
            continue
        comp_url = _nearest(comp_list, target, FRAME_SPACING_MIN / 2.0)
        if comp_url is None:
            continue
        t_comp = next(t for t, u in comp_list if u == comp_url)
        src = {"comp": comp_url}
        for key, (product, _, _) in PRODUCTS.items():
            if key == "comp":
                continue
            src[key] = _nearest(listing(sess, product, cache), t_comp, MATCH_TOL_MIN)
        out.append(src)
    return out


# --------------------------------------------------------------------------------------
# Animated GIF export
# --------------------------------------------------------------------------------------
GIF_SCALE = 0.5            # each panel at half size keeps a 12-frame loop around 1-2 MB
GIF_FRAME_MS = 650
GIF_HOLD_MS = 1600         # linger on the newest frame so the loop reads as "now"
GIF_RINGS_NM = (3, 5, 10, 20)
RING_LABEL_DEG = {3: -20, 5: 22, 10: 40, 20: 48}   # matched in web/index.html
GIF_VARIANTS = {"status": "Status and radar", "class": "Cloud class and radar",
                "overlay": "Status on radar"}


def _font(size, bold=False):
    from PIL import ImageFont
    for name in (("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
                 "/usr/share/fonts/truetype/dejavu/" + ("DejaVuSans-Bold.ttf" if bold
                                                        else "DejaVuSans.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def _rings(img):
    """Range rings, drawn with the same Mercator geometry the viewer uses for its SVG."""
    from PIL import ImageDraw
    W, H = img.size
    d = DOMAIN
    psi = lambda x: np.log(np.tan(np.pi / 4 + np.radians(x) / 2))
    clat = float(np.mean([p[0] for p in SITES.values()]))
    clon = float(np.mean([p[1] for p in SITES.values()]))
    cx = (clon - d["lon_min"]) / (d["lon_max"] - d["lon_min"]) * W
    cy = (psi(d["lat_max"]) - psi(clat)) / (psi(d["lat_max"]) - psi(d["lat_min"])) * H
    span = d["lon_max"] - d["lon_min"]
    draw = ImageDraw.Draw(img)
    f = _font(10)
    for nm in GIF_RINGS_NM:
        r = nm * 1.852 / (111.32 * np.cos(np.radians(clat))) / span * W
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(159, 178, 196, 150), width=1)
        # Staggered angles: on one diagonal the 3 and 5 nmi labels land on top of each other.
        a = np.radians(RING_LABEL_DEG.get(nm, 45))
        draw.text((cx + r * np.cos(a) + 3, cy - r * np.sin(a) - 6), f"{nm} nmi", font=f,
                  fill=(159, 178, 196, 210))
    return img


def build_gifs(frames):
    """One looping GIF per view, oldest to newest, so a loop can be shared or briefed from.

    Built on the server rather than in the browser: it is the same picture for everyone, it
    needs no encoder shipped to the page, and it can be tested here.
    """
    from PIL import Image, ImageDraw
    ordered = sorted(frames, key=lambda f: f["valid"])
    if not ordered:
        return {}
    well = (17, 26, 37, 255)
    out = {}
    for variant, title in GIF_VARIANTS.items():
        pics = []
        for f in ordered:
            def layer(k):
                return Image.open(os.path.join(OUT_DIR, f["images"][k])).convert("RGBA")
            try:
                if variant == "overlay":
                    stack = [[layer("radar"), layer("statusline")]]
                else:
                    stack = [[layer(variant)], [layer("radar")]]
            except Exception as e:
                logging.warning(f"gif {variant}: frame {f['valid']} unreadable ({e}); skipped")
                continue
            panels = []
            for layers in stack:
                base = Image.new("RGBA", layers[0].size, well)
                for L_ in layers:
                    base = Image.alpha_composite(base, L_)
                w, h = base.size
                small = base.resize((int(w * GIF_SCALE), int(h * GIF_SCALE)), Image.LANCZOS)
                # Rings go on AFTER the downscale: drawn first, their labels shrank to ~5 px.
                panels.append(_rings(small))
            pw, ph = panels[0].size
            gap, head = 8, 46
            canvas = Image.new("RGBA", (pw * len(panels) + gap * (len(panels) - 1), ph + head),
                               (20, 29, 41, 255))
            for k, p in enumerate(panels):
                canvas.alpha_composite(p, (k * (pw + gap), head))
            dr = ImageDraw.Draw(canvas)
            dr.text((10, 8), "CloudScope Radar", font=_font(17, True), fill=(232, 238, 243))
            dr.text((10, 28), title, font=_font(11), fill=(159, 178, 196))
            stamp = f["valid"][11:16] + "Z  " + datetime.datetime.strptime(
                f["valid"][:10], "%Y-%m-%d").strftime("%a %d %b")
            tf = _font(17, True)
            tw = dr.textlength(stamp, font=tf) if hasattr(dr, "textlength") else 140
            dr.text((canvas.width - tw - 10, 8), stamp, font=tf, fill=(232, 238, 243))
            pics.append(canvas.convert("RGB").convert("P", palette=Image.ADAPTIVE, colors=160))
        if not pics:
            continue
        rel = f"loop_{variant}.gif"
        dur = [GIF_FRAME_MS] * (len(pics) - 1) + [GIF_HOLD_MS]
        pics[0].save(os.path.join(OUT_DIR, rel), save_all=True, append_images=pics[1:],
                     duration=dur, loop=0, disposal=2)
        out[variant] = rel
    return out


# --------------------------------------------------------------------------------------
def load_manifest():
    try:
        with open(os.path.join(OUT_DIR, "manifest.json")) as fp:
            return json.load(fp)
    except Exception:
        return {}


def main():
    frame_dir = os.path.join(OUT_DIR, "frames")
    state_dir = os.path.join(OUT_DIR, "state")
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(state_dir, exist_ok=True)
    sess = _session()
    prev = load_manifest()

    try:
        z0, fmeta, refreshed = freezing_level(sess, prev, state_dir)
    except Exception as e:
        logging.warning(f"freezing level unavailable ({e}); using 4,800 m")
        z0, fmeta, refreshed = None, {"fetched": None, "valid": None, "fallback": True}, False

    try:
        levels = list_levels_3d(sess)
    except Exception as e:
        logging.warning(f"3-D level list unavailable ({e}); this frame stays 2-D")
        levels = None
    newest, la, lo = build_frame(sess, {k: latest_url(p) for k, (p, _, _) in PRODUCTS.items()},
                                 z0, levels, prior=prev.get("frames", []))
    if newest is None:
        logging.error("latest frame withheld; previous frames left in place")
        return
    frames = [f for f in prev.get("frames", []) if f.get("valid") != newest["valid"]
              and f.get("images")]          # drop any frame from the v1 layout
    frames.insert(0, newest)

    # Backfill the loop from the archive, a few frames per run.
    if len(frames) < FRAMES_KEPT:
        t0 = time.monotonic()
        cache = {}
        srcs = backfill_sources(sess, [f["valid"] for f in frames], newest["valid"],
                                cache, BACKFILL_PER_RUN)
        done = 0
        for src in srcs:
            if time.monotonic() - t0 > BACKFILL_BUDGET_S:
                logging.info("backfill budget spent; the rest fill on later runs")
                break
            frame, _, _ = build_frame(sess, src, z0, prior=frames)
            if frame is not None:
                frames.append(frame)
                done += 1
        if srcs:
            logging.info(f"backfilled {done} of {len(srcs)} archive frame(s) "
                         f"in {time.monotonic() - t0:.0f} s")

    frames.sort(key=lambda f: f["valid"], reverse=True)
    frames = frames[:FRAMES_KEPT]

    keep = set()
    for f in frames:
        keep.update(os.path.basename(p) for p in f["images"].values())
        keep.add(os.path.basename(f["data"]))
        if f.get("volume"):
            keep.add(os.path.basename(f["volume"]))
    for fn in os.listdir(frame_dir):
        if fn not in keep:
            os.remove(os.path.join(frame_dir, fn))

    try:
        gifs = build_gifs(frames)
    except Exception as e:
        # A failed export must never cost the frame itself.
        logging.warning(f"GIF export failed: {type(e).__name__}: {e}")
        gifs = {}

    manifest = {
        "generated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gifs": gifs,
        "viewer_expected": VIEWER_VERSION_EXPECTED,
        "valid": newest["valid"], "freezing": fmeta, "domain": DOMAIN,
        "grid": {"ny": int(la.size), "nx": int(lo.size), "lat_n": float(la[0]),
                 "lon_w": float(lo[0]), "dlat": abs(float(la[0] - la[1])),
                 "dlon": abs(float(lo[1] - lo[0]))},
        "sites": {k: list(v) for k, v in SITES.items()},
        "frames": frames,
        "rule_keys": RULE_KEYS, "rules": RULE_NAMES, "rule_how": RULE_HOW,
        "watch_how": WATCH_HOW, "not_evaluated": NOT_EVALUATED, "rules_off": RULES_OFF,
        "classes": CLASSES, "class_note": CLASS_NOTE, "top_levels": TOP_LEVELS,
        "levels_km": _VOL_LEVELS or prev.get("levels_km") or [],
        "status": {str(k): {"name": v[0], "color": v[1]} for k, v in STATUS.items()},
        "refl": {"levels": REFL_LEVELS, "colors": REFL_COLORS},
        "thresholds": LLCC, "standard": "NASA-STD-4010 (2017-06-27)",
    }
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as fp:
        json.dump(manifest, fp, indent=1)
    logging.info(f"{len(frames)} frame(s) in the loop; freezing "
                 f"{'refreshed' if refreshed else 'cached'}")


if __name__ == "__main__":
    main()
