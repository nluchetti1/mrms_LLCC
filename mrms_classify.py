#!/usr/bin/env python3
"""
MRMS LLCC nowcast - a radar traffic light for the Cape.

CloudScope classifies FORECAST cloud from model condensate and turns it into a probability.
This does the observational counterpart: it reads the latest MRMS radar and lightning,
applies the NASA-STD-4010 Lightning Launch Commit Criteria that a radar can adjudicate, and
scores every grid cell as if the flight path ran through it:

    RED     a rule is violated
    YELLOW  no rule violated, but one would be if its standoff were 2 nmi longer, or
            there is lightning within 20 nmi
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
    Lightning means ANY of NLDN CG, GOES-19 GLM or the local MERLIN network.

LOCAL SENSORS (v20)
    MERLIN cloud-to-ground and cloud-to-cloud lightning and the LPLWS 1-minute field mills, from
    the range's public codiapi services. They feed 4.1.1 (lightning) and, for the first time on
    the live page, 4.1.2 (surface electric field). Best effort: they are fetched in a background
    thread with short timeouts, cached between runs, and a run that cannot reach them still
    publishes from GLM/NLDN with 4.1.2 marked not evaluated. See the LOCAL SENSORS section.

NOT EVALUATED
    The exceptions to 4.1.2, and 4.1.2 itself more than 5 nmi from every reporting mill; debris
    clouds (4.1.6); smoke plumes (4.1.9); triboelectrification (4.1.10); the 3-hour lightning
    clocks on the anvil rules, since only ~95 minutes of lightning is kept. Green means no
    violation the radar and sensors can see, never GO.
"""

import datetime
import gzip
import json
import logging
import os
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

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

VIEWER_VERSION_EXPECTED = "mrms-v21"

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
ESSENTIAL = ("comp", "r0", "r10", "r20")
# Lightning is essential too, but from ANY source: the frame is withheld only when none of the
# NLDN CG product, GLM or MERLIN (full 30-minute coverage) could be read (see build_frame).

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
    "thick_mrr_dbz": 7.5,          # thick-layer exception: MRR < +7.5 dBZ within 1 nmi (45 WS)
    "disturbed_nm": 5.0,           # 4.1.7
    "disturbed_dbz": 30.0,
    "core_dbz": 40.0,
    "yellow_margin_nm": 2.0,
    "field_mill_nm": 5.0,          # 4.1.2: |E| >= 1000 V/m within 5 nmi, prior 15 minutes
    "field_mill_vm": 1000.0,
    "field_mill_window_min": 15.0,
}
LAPSE_C_PER_KM = 6.5

# Rule order is also the bit order in the per-cell data file - change both together.
# Bit 6 was "disturbed", which has been off (always 0) since v12; v20 gives the slot to the
# field-mill rule, so frames already in the loop read it as "not violated" and the file format
# does not change.
RULE_KEYS = ["lightning", "cumulus_through", "cumulus_5nm", "cumulus_10nm",
             "attached_anvil", "detached_anvil", "field_mill", "thick_layer"]
RULE_NAMES = {
    "lightning":       "Lightning within 10 nmi (4.1.1) - MERLIN, GLM or NLDN CG",
    "cumulus_through": "Flight through cumulus topping at or colder than +5 °C (4.1.3.1)",
    "cumulus_5nm":     "Cumulus topping colder than −10 °C within 5 nmi (4.1.3.2)",
    "cumulus_10nm":    "Cumulus topping colder than −20 °C within 10 nmi (4.1.3.3)",
    "attached_anvil":  "Attached anvil within 3 nmi (4.1.4)",
    "detached_anvil":  "Detached anvil within 3 nmi (4.1.5)",
    "field_mill":      "Surface electric field of 1,000 V/m or more within 5 nmi (4.1.2)",
    "thick_layer":     "Thick cloud layer spanning 0 to −10 °C (4.1.8)",
}
# How each rule is tested on the radar grid, in the viewer's own words.
RULE_HOW = {
    "lightning":       "Any lightning within 10 nmi in the last 30 minutes: a MERLIN cloud-to-ground "
                       "or cloud-to-cloud event, a GOES-19 GLM flash (total lightning) or an NLDN "
                       "cloud-to-ground strike. Any one source counts.",
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
    "field_mill":      "A field mill within 5 nmi of the point measured |E| of 1,000 V/m or more in "
                       "any one-minute reading over the 15 minutes before the frame. Points more "
                       "than 5 nmi from every reporting mill are not evaluated, not clear. A mill "
                       "reading exactly 0 for the whole window counts as not reporting. The "
                       "rule's exceptions are not applied, the conservative side.",
    "thick_layer":     "The point has echo at both ends of a 10-degree span inside the 0 to "
                       "−20 °C band (0 and −10, −5 and −15, or −10 and −20 °C): a layer at least "
                       "~5,000 ft deep. Not applied to anvil or to cumulus - convective echo and "
                       "cores are excluded; echo whose type is unknown is still tested. Exception: "
                       "MRR below 7.5 dBZ within 1 nmi.",
}
WATCH_HOW = ("Watch means no rule is broken, but one would be if its standoff were 2 nmi longer, "
             "or there is lightning within 20 nmi.")

# Rules deliberately not scored. Disturbed weather is applied to synoptic disturbances such as
# fronts, which radar alone cannot identify; on radar the test reduced to "30 dBZ within
# 5 nmi" and mislabelled convective scenes with a rule meant for something else. Measured
# over a convective loop it never once made a cell red on its own, so dropping it changes no
# colours - only removes a misleading reason. Since v20 it has no rule key at all (its bit went
# to the field-mill rule); it stays listed under NOT_EVALUATED.
RULES_OFF = []

NOT_EVALUATED = [
    "Disturbed weather (4.1.7) - applied to synoptic disturbances such as fronts, which radar "
    "alone cannot identify",
    "The exceptions to the surface electric field rule (4.1.2), and the rule itself more than "
    "5 nmi from every reporting field mill",
    "Debris clouds (4.1.6), which need an observed detachment time",
    "Smoke plumes (4.1.9)",
    "Triboelectrification (4.1.10)",
    "The 3-hour lightning clocks on the anvil rules - only ~95 minutes of lightning is kept",
]

# Cloud class, read off how far up the isotherm stack an echo reaches. These are what the
# radar can actually say about an echo, and each maps onto the rules it drives.
CLASSES = [
    {"id": 0, "key": "clear", "name": "No echo", "color": "#00000000",
     "how": "Composite reflectivity below 0 dBZ. No rule applies here."},
    {"id": 1, "key": "warm", "name": "Shallow shower, below freezing", "color": "#94A3B8",
     "how": "Echo of 0 dBZ or more that shows up at none of the isotherm levels: the column's "
            "strongest echo sits below the measured 0 °C height. Can still trip flight-through "
            "cumulus if it reaches the +5 °C level, taken from the XMR model sounding."},
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
     "how": "An anvil that came from a convective core and has separated from it, lying where an "
            "attached anvil was in the last hour (allowing ~32 kt of drift) with no unbroken path "
            "back to a core now. Either floating - no echo at 0 °C beneath it - or still raining "
            "out as stratiform echo: anvil until it rains out. Unconnected convective echo rooted "
            "through 0 °C is a new tower and stays cumulus."},
    {"id": 8, "key": "elev", "name": "Layered cloud, not convective", "color": "#C9A66B",
     "how": "Echo the convective/stratiform separation (Steiner et al. 1995, run on the 3-D "
            "volume's ~3 km level) calls stratiform and that is not anvil, or floating echo with "
            "no convective origin in the last hour. Layered cloud, not cumulus: scored by the "
            "thick-cloud-layer rule when it spans about 4,500 ft of the 0 to −20 °C band, never by "
            "the cumulus or anvil standoffs."},
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


def listing(sess, product, cache, root=None):
    """[(datetime, url)] of the timestamped files MRMS keeps for a product, newest first."""
    ck = (root or ROOT, product)
    if ck in cache:
        return cache[ck]
    try:
        root = root or ROOT
        r = sess.get(f"{root}/{product}/", timeout=60)
        r.raise_for_status()
        out = []
        for name, stamp in _STAMP_RE.findall(r.text):
            t = datetime.datetime.strptime(stamp, "%Y%m%d-%H%M%S")
            out.append((t, f"{root}/{product}/{name}"))
        out.sort(reverse=True)
    except Exception as e:
        logging.warning(f"listing {product}: {type(e).__name__}: {e}")
        out = []
    cache[ck] = out
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


# --------------------------------------------------------------------------------------
# Isotherm heights from the XMR model sounding (Penn State BUFKIT)
# --------------------------------------------------------------------------------------
# Measured 6 Oct 2026: the +5 C and -20 C heights had been estimated from the MRMS freezing
# level and an assumed 6.5 C/km lapse rate. The HRRR and RAP soundings over XMR put the 0 to
# -20 C lapse at 5.9 and 5.2 C/km, which placed -20 C some 600 m (2,000 ft) higher than the
# estimate - and that height decides whether echo counts as "aloft only" for the anvil. These
# are MODEL soundings, not the balloon, but they exist every hour; the raob does not.
#
# PSU's robots policy discourages automated traffic, so both files are fetched at most once an
# hour and cached - the rate they update anyway.
BUFKIT = [("HRRR", "https://www.meteo.psu.edu/bufkit/data/HRRR/latest/hrrr_xmr.buf"),
          ("RAP",  "https://www.meteo.psu.edu/bufkit/data/RAP/latest/rap_xmr.buf")]
BUFKIT_REFETCH_MIN = 60
BUFKIT_MAX_RUN_AGE_H = 6       # newest run older than this (relative to the frame) -> estimate
BUFKIT_MAX_GAP_H = 2           # no profile within this of the frame -> estimate
ISO_C = [5.0, 0.0, -5.0, -10.0, -15.0, -20.0]


def parse_bufkit(text):
    """[(valid, [(height_m, tmpc), ...])] - the probe's parser, proven on PSU's real files.

    Each time block starts 'STID = ... TIME = YYMMDD/HHMM'. The profile header names the
    parameters across one or more lines (PRES TMPC ... / CFRL HGHT); the numbers that follow are
    those parameters per level, wrapped across lines, so they are read as one stream.
    """
    out = []
    for b in re.split(r"(?=STID\s*=)", text):
        m = re.search(r"TIME\s*=\s*(\d{6})/(\d{4})", b)
        if not m:
            continue
        try:
            valid = datetime.datetime.strptime(m.group(1) + m.group(2), "%y%m%d%H%M")
        except ValueError:
            continue
        hm = re.search(r"\n\s*(PRES(?:\s+[A-Z]{4})+)\s*\n((?:\s*[A-Z]{4}(?:\s+[A-Z]{4})*\s*\n)*)", b)
        if not hm:
            continue
        names = (hm.group(1) + " " + hm.group(2)).split()
        body = b[hm.end():]
        stop = re.search(r"\n\s*\n|\n[A-Z]{3,}\s*=|STN\s", body)
        if stop:
            body = body[:stop.start()]
        nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", body)]
        k = len(names)
        if "HGHT" not in names or "TMPC" not in names or len(nums) < k:
            continue
        ih, it = names.index("HGHT"), names.index("TMPC")
        rows = (nums[i:i + k] for i in range(0, len(nums) - k + 1, k))
        levels = sorted((r[ih], r[it]) for r in rows if r[it] > -900 and r[ih] > -900)
        if levels:
            out.append((valid, levels))
    return out


def isotherm_heights(levels):
    """Lowest warm-to-cold crossing of each ISO_C temperature, metres MSL."""
    out = {}
    for tc in ISO_C:
        for (z1, t1), (z2, t2) in zip(levels, levels[1:]):
            if z2 > z1 and t1 >= tc > t2:
                out[tc] = z1 + (t1 - tc) / (t1 - t2) * (z2 - z1)
                break
    return out


def load_soundings(sess, prev, state_dir):
    """The newest XMR model run, from cache unless it is over an hour old.
    Returns {model, init, fetched, profiles:[[valid_iso, levels]]} or None."""
    path = os.path.join(state_dir, "bufkit.json")
    now = datetime.datetime.now(datetime.timezone.utc)
    try:
        with open(path) as fp:
            cache = json.load(fp)
        age = (now - datetime.datetime.fromisoformat(cache["fetched"])).total_seconds() / 60
    except Exception:
        cache, age = None, None
    if cache is not None and age is not None and age < BUFKIT_REFETCH_MIN:
        return cache
    best = None
    for model, url in BUFKIT:
        try:
            r = sess.get(url, timeout=40)
            r.raise_for_status()
            profs = parse_bufkit(r.text)
        except Exception as e:
            logging.warning(f"BUFKIT {model}: {type(e).__name__}: {e}")
            continue
        if not profs:
            logging.warning(f"BUFKIT {model}: no profiles parsed")
            continue
        init = profs[0][0]
        if best is None or init > best["init_dt"]:       # newest run wins; HRRR keeps a tie
            best = {"model": model, "init_dt": init, "profiles": profs}
    if best is None:
        if cache is not None:
            logging.warning("BUFKIT: refresh failed; keeping the cached run")
        return cache
    cache = {"fetched": now.isoformat(), "model": best["model"],
             "init": best["init_dt"].strftime("%Y-%m-%dT%H:%MZ"),
             "profiles": [[v.strftime("%Y-%m-%dT%H:%MZ"), lv] for v, lv in best["profiles"]]}
    os.makedirs(state_dir, exist_ok=True)
    with open(path, "w") as fp:
        json.dump(cache, fp)
    logging.info(f"BUFKIT: {best['model']} {best['init_dt']:%H}Z run, "
                 f"{len(best['profiles'])} hourly profiles")
    return cache


def isotherms_for(snd, valid_iso):
    """(heights in m keyed by temperature, label) from the profile nearest the frame - or
    (None, reason) when the lapse-rate estimate should be used instead."""
    if not snd or not valid_iso:
        return None, "estimated: no BUFKIT sounding available"
    t = datetime.datetime.strptime(valid_iso[:16], "%Y-%m-%dT%H:%M")
    init = datetime.datetime.strptime(snd["init"], "%Y-%m-%dT%H:%MZ")
    vt = lambda p: datetime.datetime.strptime(p[0], "%Y-%m-%dT%H:%MZ")
    best = min(snd["profiles"], key=lambda p: abs((vt(p) - t).total_seconds()))
    gap_h = abs((vt(best) - t).total_seconds()) / 3600
    if gap_h > BUFKIT_MAX_GAP_H:
        return None, f"estimated: nearest BUFKIT profile is {gap_h:.0f} h from this frame"
    if (t - init).total_seconds() / 3600 > 18 + BUFKIT_MAX_RUN_AGE_H:
        return None, f"estimated: newest BUFKIT run ({snd['model']} {init:%H}Z) is too old"
    iso = isotherm_heights([tuple(x) for x in best[1]])
    if 5.0 not in iso or -20.0 not in iso:
        return None, "estimated: sounding does not cross +5 and -20 C"
    return iso, f"{snd['model']} {init:%H}Z run, profile valid {vt(best):%H}Z (PSU BUFKIT, XMR)"


# --------------------------------------------------------------------------------------
# Lightning from GOES-19 GLM, alongside NLDN CG
# --------------------------------------------------------------------------------------
# The NLDN product is cloud-to-ground ONLY, and LLCC 4.1.1 covers any lightning; most flashes
# are intracloud, and GLM sees total lightning. Measured 7 Oct 2026: one full-disk file per
# ~20 s, ~270 KB each, under a minute behind real time - 23 MB for a 30-minute window, but only
# ~15 files and ~4 MB per 5-minute run when just the new files are pulled. So flashes are kept in
# a rolling cache covering the loop plus the 30-minute window.
#
# GLM detection efficiency is high but not perfect, lowest for small, weak flashes, so it does not
# REPLACE NLDN: lightning is GLM or NLDN CG, and a lone CG that GLM misses still counts.
GLM_BUCKET = "https://noaa-goes19.s3.amazonaws.com"
GLM_WINDOW_MIN = 30        # 4.1.1
GLM_KEEP_MIN = 95          # backfilled frames reach back ~55 min, plus the 30-minute window
GLM_MAX_FILES = 320        # per run - a first run pulls ~285 files; later runs ~15
GLM_MARGIN_NM = 25.0       # keep flashes this far outside the map: 20 nmi watch plus slack


def _utcnow():
    """Current UTC as a naive datetime. CLOUDSCOPE_FAKE_NOW (ISO) overrides it for offline tests
    only - production never sets it, and without it the GLM path could not be tested at all."""
    v = os.environ.get("CLOUDSCOPE_FAKE_NOW")
    if v:
        return datetime.datetime.fromisoformat(v)
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


def _glm_keys(sess, now):
    out = []
    for h in range(int(np.ceil(GLM_KEEP_MIN / 60.0)) + 1, -1, -1):
        t = now - datetime.timedelta(hours=h)
        prefix = f"GLM-L2-LCFA/{t:%Y}/{t.timetuple().tm_yday:03d}/{t:%H}/"
        token = None
        while True:
            url = (f"{GLM_BUCKET}/?list-type=2&prefix={prefix}"
                   + (f"&continuation-token={requests.utils.quote(token)}" if token else ""))
            r = sess.get(url, timeout=60)
            r.raise_for_status()
            for key in re.findall(r"<Key>([^<]+)</Key>", r.text):
                m = re.search(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})", key)
                if m:
                    y, d, H, M, S = map(int, m.groups())
                    out.append((datetime.datetime(y, 1, 1, H, M, S)
                                + datetime.timedelta(days=d - 1), key))
            m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", r.text)
            if not m:
                break
            token = m.group(1)
    return sorted(set(out))


def _glm_read(raw):
    """[(lat, lon, time)] for good-quality flashes in one GLM L2 file."""
    import netCDF4
    try:
        ds = netCDF4.Dataset("glm", memory=raw)
    except Exception:
        fd, path = tempfile.mkstemp(suffix=".nc")
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        ds = netCDF4.Dataset(path)
        os.remove(path)
    try:
        lat = np.asarray(ds.variables["flash_lat"][:], float)
        lon = np.asarray(ds.variables["flash_lon"][:], float)
        off = np.asarray(ds.variables["flash_time_offset_of_first_event"][:], float)
        q = (np.asarray(ds.variables["flash_quality_flag"][:], int)
             if "flash_quality_flag" in ds.variables else np.zeros(lat.shape, int))
        base = datetime.datetime.strptime(ds.time_coverage_start[:19], "%Y-%m-%dT%H:%M:%S")
    finally:
        ds.close()
    mlat, mlon = GLM_MARGIN_NM * 1.852 / 111.32, GLM_MARGIN_NM * 1.852 / (111.32 * 0.87)
    keep = ((q == 0) & (lat >= DOMAIN["lat_min"] - mlat) & (lat <= DOMAIN["lat_max"] + mlat)
            & (lon >= DOMAIN["lon_min"] - mlon) & (lon <= DOMAIN["lon_max"] + mlon))
    return [(float(a), float(b), base + datetime.timedelta(seconds=float(s)))
            for a, b, s in zip(lat[keep], lon[keep], off[keep])]


def glm_update(sess, state_dir):
    """Bring the rolling flash cache up to date, pulling only files not seen before.
    Returns ({flashes:[(lat, lon, datetime)], ok, newest, note})."""
    path = os.path.join(state_dir, "glm.json")
    now = _utcnow()
    try:
        with open(path) as fp:
            cache = json.load(fp)
    except Exception:
        cache = {"done": [], "flashes": []}
    done = set(cache.get("done", []))
    flashes = [(a, b, datetime.datetime.fromisoformat(t)) for a, b, t in cache.get("flashes", [])]
    try:
        keys = _glm_keys(sess, now)
    except Exception as e:
        return {"flashes": flashes, "ok": False, "newest": None,
                "note": f"GLM listing failed ({type(e).__name__}); NLDN CG only"}
    cutoff = now - datetime.timedelta(minutes=GLM_KEEP_MIN)
    todo = [(t, k) for t, k in keys if t >= cutoff and k not in done]
    if len(todo) > GLM_MAX_FILES:
        todo = todo[-GLM_MAX_FILES:]          # newest first matters most for the newest frame
    got = failed = 0
    for t, k in todo:
        try:
            r = sess.get(f"{GLM_BUCKET}/{k}", timeout=60)
            r.raise_for_status()
            flashes += _glm_read(r.content)
            done.add(k)
            got += 1
        except Exception:
            failed += 1
    flashes = [f for f in flashes if f[2] >= cutoff]
    keep_keys = {k for t, k in keys if t >= cutoff}
    done &= keep_keys
    os.makedirs(state_dir, exist_ok=True)
    with open(path, "w") as fp:
        json.dump({"done": sorted(done),
                   "flashes": [[a, b, t.isoformat()] for a, b, t in flashes]}, fp)
    newest = max((t for t, k in keys), default=None)
    ok = bool(keys) and (newest is not None and (now - newest).total_seconds() < 15 * 60)
    note = (f"GLM: {got} new file(s)" + (f", {failed} failed" if failed else "")
            + f", {len(flashes)} flashes cached")
    logging.info(note)
    return {"flashes": flashes, "ok": ok, "newest": newest, "note": note}


# --------------------------------------------------------------------------------------
# GOES-19 ABI: cloud-top temperature over echo, and anvil-level winds for detached-anvil drift
# --------------------------------------------------------------------------------------
# Ground rule (45 WS): satellite never creates cloud where radar shows nothing. The LLCC
# footprint stays defined by echo >= 0 dBZ; satellite only refines how that echo is classed.
#
# Measured 8 Oct 2026: band 13 CONUS every 5 min, ~4 min behind, 3.7 MB; the 2 km cloud-top
# height (ACHA2KMC - the plain ACHAC product is a ~10 km grid, too coarse); band 14 derived motion
# winds every 15 min, ~17 min behind, with 14 anvil-level vectors near the Cape that morning
# (24 kt from 235). At the Cape GOES-East looks 33.9 deg off vertical, so a 12 km top APPEARS
# 8 km (4.4 nmi) out of place toward ~349 deg - more than the anvil standoff. Every pixel is moved
# back by its own height before it is matched to a radar column.
GOES_ABI = "https://noaa-goes19.s3.amazonaws.com"
SAT_TOL_MIN = {"CMIPC": 12, "ACHA2KMC": 20}
SAT_MATCH_KM = 2.5            # a radar cell takes the nearest corrected pixel within this
SAT_CONSISTENCY_M = 3000.0    # satellite top counts only within this above the radar echo top
WIND_RADIUS_KM = 150.0
WIND_TOP_HPA = 400.0          # anvil level: above this pressure
WIND_MIN_VECTORS = 5
WIND_MIN_UNC_KT = 10.0        # never trust the mean to better than this
WIND_MAX_AGE_MIN = 45
# Centre of the pads - the same point the range rings and the GIF rings use.
CAPE_LAT = float(np.mean([p[0] for p in SITES.values()]))
CAPE_LON = float(np.mean([p[1] for p in SITES.values()]))


def _goes_keys(sess, product, t, cache):
    """[(start, key)] for one ABI product in the hours around t, cached for the run."""
    out = []
    for h in (-1, 0, 1):
        tt = t + datetime.timedelta(hours=h)
        prefix = f"ABI-L2-{product}/{tt:%Y}/{tt.timetuple().tm_yday:03d}/{tt:%H}/"
        if prefix not in cache:
            keys, token = [], None
            try:
                while True:
                    url = (f"{GOES_ABI}/?list-type=2&prefix={prefix}"
                           + (f"&continuation-token={requests.utils.quote(token)}" if token else ""))
                    r = sess.get(url, timeout=60)
                    r.raise_for_status()
                    keys += re.findall(r"<Key>([^<]+)</Key>", r.text)
                    m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", r.text)
                    if not m:
                        break
                    token = m.group(1)
            except Exception:
                keys = []
            cache[prefix] = keys
        for k in cache[prefix]:
            m = re.search(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})", k)
            if m:
                y, d, H, M, S = map(int, m.groups())
                out.append((datetime.datetime(y, 1, 1, H, M, S) + datetime.timedelta(days=d - 1), k))
    return sorted(set(out))


def _nc_open(raw):
    import netCDF4
    try:
        return netCDF4.Dataset("abi", memory=raw), None
    except Exception:
        fd, path = tempfile.mkstemp(suffix=".nc")
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        return netCDF4.Dataset(path), path


def _fixed_grid(ds, margin_deg=0.35):
    """(lat, lon, values-slicer, geometry) for the window of a GOES-R fixed grid around DOMAIN,
    per the GOES-R Product User Guide."""
    p = ds.variables["goes_imager_projection"]
    req, rpol = float(p.semi_major_axis), float(p.semi_minor_axis)
    H = float(p.perspective_point_height) + req
    lam0 = np.radians(float(p.longitude_of_projection_origin))
    def to_xy(lat, lon):
        lat, lon = np.radians(lat), np.radians(lon)
        e2 = (req ** 2 - rpol ** 2) / req ** 2
        pc = np.arctan((rpol ** 2 / req ** 2) * np.tan(lat))
        rc = rpol / np.sqrt(1 - e2 * np.cos(pc) ** 2)
        sx = H - rc * np.cos(pc) * np.cos(lon - lam0)
        sy = -rc * np.cos(pc) * np.sin(lon - lam0)
        sz = rc * np.sin(pc)
        return np.arcsin(-sy / np.sqrt(sx ** 2 + sy ** 2 + sz ** 2)), np.arctan(sz / sx)
    xs, ys = [], []
    for la_ in (DOMAIN["lat_min"] - margin_deg, DOMAIN["lat_max"] + margin_deg):
        for lo_ in (DOMAIN["lon_min"] - margin_deg, DOMAIN["lon_max"] + margin_deg):
            a, b = to_xy(la_, lo_); xs.append(a); ys.append(b)
    x = np.asarray(ds.variables["x"][:], float); y = np.asarray(ds.variables["y"][:], float)
    ix = np.where((x >= min(xs)) & (x <= max(xs)))[0]
    iy = np.where((y >= min(ys)) & (y <= max(ys)))[0]
    if ix.size == 0 or iy.size == 0:
        raise ValueError("domain outside this file's grid")
    sl = (slice(iy.min(), iy.max() + 1), slice(ix.min(), ix.max() + 1))
    X, Y = np.meshgrid(x[sl[1]], y[sl[0]])
    a = np.sin(X) ** 2 + np.cos(X) ** 2 * (np.cos(Y) ** 2 + (req ** 2 / rpol ** 2) * np.sin(Y) ** 2)
    b = -2 * H * np.cos(X) * np.cos(Y)
    c = H ** 2 - req ** 2
    with np.errstate(invalid="ignore"):
        rs = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        sx = rs * np.cos(X) * np.cos(Y); sy = -rs * np.sin(X); sz = rs * np.cos(X) * np.sin(Y)
        lat = np.degrees(np.arctan((req ** 2 / rpol ** 2) * sz / np.sqrt((H - sx) ** 2 + sy ** 2)))
        lon = np.degrees(lam0 - np.arctan(sy / (H - sx)))
    return lat, lon, sl, (req, rpol, H, lam0)


def parallax_shift(lat, lon, h_m, geom):
    """(dlat, dlon) by which a cloud top h_m above (lat, lon) APPEARS displaced: the ray from the
    satellite through the top, continued to the first point it meets the ellipsoid. Vectorized.
    Checked against height x tan(zenith) at LC-39A: 8.1 km for a 12 km top."""
    req, rpol, H, lam0 = geom
    e2 = 1 - rpol ** 2 / req ** 2
    la, lo = np.radians(lat), np.radians(lon)
    N = req / np.sqrt(1 - e2 * np.sin(la) ** 2)
    P = np.stack([(N + h_m) * np.cos(la) * np.cos(lo), (N + h_m) * np.cos(la) * np.sin(lo),
                  (N * (1 - e2) + h_m) * np.sin(la)])
    S = np.array([H * np.cos(lam0), H * np.sin(lam0), 0.0]).reshape(3, *([1] * np.ndim(lat)))
    d = P - S
    A = (d[0] ** 2 + d[1] ** 2) / req ** 2 + d[2] ** 2 / rpol ** 2
    B = 2 * ((S[0] * d[0] + S[1] * d[1]) / req ** 2 + S[2] * d[2] / rpol ** 2)
    C = (S[0] ** 2 + S[1] ** 2) / req ** 2 + S[2] ** 2 / rpol ** 2 - 1
    t = (-B - np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)          # the NEAR root
    Q = S + t * d
    qlat = np.degrees(np.arctan2(Q[2], np.hypot(Q[0], Q[1]) * (1 - e2)))
    qlon = np.degrees(np.arctan2(Q[1], Q[0]))
    return qlat - lat, qlon - lon


def goes_tops(sess, valid_iso, la, lo, cache):
    """Parallax-corrected band-13 cloud-top temperature (C) and cloud-top height (m) on the MRMS
    grid, from the files nearest the frame. Returns ({bt_c, top_m}, label) or (None, reason)."""
    from scipy.spatial import cKDTree
    t = datetime.datetime.strptime(valid_iso[:19], "%Y-%m-%dT%H:%M:%S")
    picks = {}
    for prod, match in (("CMIPC", "C13_"), ("ACHA2KMC", "")):
        cands = [(abs((st - t).total_seconds()) / 60, st, k) for st, k in _goes_keys(sess, prod, t, cache)
                 if match in k]
        cands = [c for c in cands if c[0] <= SAT_TOL_MIN[prod]]
        if not cands:
            return None, f"no {prod} file within {SAT_TOL_MIN[prod]} min of this frame"
        picks[prod] = min(cands)
    out = {}
    for prod, var in (("CMIPC", "CMI"), ("ACHA2KMC", "HT")):
        r = sess.get(f"{GOES_ABI}/{picks[prod][2]}", timeout=120)
        r.raise_for_status()
        ds, tmp = _nc_open(r.content)
        try:
            lat, lon, sl, geom = _fixed_grid(ds)
            v = ds.variables[var][sl]
            v = np.asarray(v.filled(np.nan) if np.ma.isMaskedArray(v) else v, float)
        finally:
            ds.close()
            if tmp:
                os.remove(tmp)
        out[prod] = (lat, lon, v, geom)
    blat, blon, bt, geom = out["CMIPC"]
    hlat, hlon, ht, _ = out["ACHA2KMC"]
    lat0 = float(np.mean(la)); kx = 111.32 * np.cos(np.radians(lat0))
    km_xy = lambda a, b: np.column_stack([((b - lo[0]) * kx).ravel(), ((a - la[0]) * 111.32).ravel()])
    hv = np.isfinite(ht) & np.isfinite(hlat) & (ht > 0)
    if not hv.any():
        return None, "no cloud-top heights in the window (clear sky)"
    # 1. each band-13 pixel takes the cloud-top height of its OWN footprint (both products sit on
    #    the 2 km grid, at their apparent positions). 1.5 km, not wider: with 4 km, clear pixels
    #    beside a cloud borrowed its height and were moved as if they were cloud.
    d, i = cKDTree(km_xy(hlat[hv], hlon[hv])).query(km_xy(blat, blon), distance_upper_bound=1.5)
    h_pix = np.full(blat.size, np.nan)
    ok = np.isfinite(d)
    h_pix[ok] = ht[hv][i[ok]]
    use = ok & np.isfinite(bt.ravel()) & np.isfinite(blat.ravel())
    if not use.any():
        return None, "no band-13 pixels with a cloud-top height"
    # 2. move each pixel back to where its cloud top really is
    dla, dlo = parallax_shift(blat.ravel()[use], blon.ravel()[use], h_pix[use], geom)
    tla, tlo = blat.ravel()[use] - dla, blon.ravel()[use] - dlo
    # 3. nearest corrected pixel for each radar cell
    LO, LA = np.meshgrid(lo, la)
    d2, i2 = cKDTree(km_xy(tla, tlo)).query(km_xy(LA, LO), distance_upper_bound=SAT_MATCH_KM)
    bt_c = np.full(LA.size, np.nan, np.float32); top_m = np.full(LA.size, np.nan, np.float32)
    hit = np.isfinite(d2)
    bt_c[hit] = bt.ravel()[use][i2[hit]] - 273.15
    top_m[hit] = h_pix[use][i2[hit]]
    label = (f"GOES-19 band 13 {picks['CMIPC'][1]:%H:%M}Z, cloud-top height "
             f"{picks['ACHA2KMC'][1]:%H:%M}Z, parallax-corrected")
    return {"bt_c": bt_c.reshape(LA.shape), "top_m": top_m.reshape(LA.shape)}, label


def goes_anvil_wind(sess, cache):
    """Vector-mean anvil-level wind near the Cape from band-14 derived motion winds (band 8 as the
    fallback). Returns {u_kmh, v_kmh (motion TOWARD east/north), unc_kmh, kt, from_deg, n, label}
    or None, when there are too few vectors - drift then uses the all-directions allowance."""
    now = _utcnow()
    keys = _goes_keys(sess, "DMWC", now, cache)
    for band in ("C14", "C08"):
        cands = [(st, k) for st, k in keys if f"-M6{band}_" in k or f"{band}_G19" in k]
        cands = [c for c in cands if (now - c[0]).total_seconds() / 60 <= WIND_MAX_AGE_MIN]
        if not cands:
            continue
        st, key = max(cands)
        try:
            r = sess.get(f"{GOES_ABI}/{key}", timeout=60)
            r.raise_for_status()
            ds, tmp = _nc_open(r.content)
            try:
                g = lambda n: np.asarray(ds.variables[n][:], float)
                lat, lon, spd, dirn, p = (g("lat"), g("lon"), g("wind_speed"), g("wind_direction"),
                                          g("pressure"))
                dqf = g("DQF") if "DQF" in ds.variables else np.zeros(lat.shape)
            finally:
                ds.close()
                if tmp:
                    os.remove(tmp)
        except Exception as e:
            logging.warning(f"DMW {band}: {type(e).__name__}: {e}")
            continue
        k = np.cos(np.radians(CAPE_LAT))
        near = ((dqf == 0) & (p <= WIND_TOP_HPA)
                & (np.hypot((lat - CAPE_LAT) * 111.32, (lon - CAPE_LON) * 111.32 * k) <= WIND_RADIUS_KM))
        n = int(near.sum())
        if n < WIND_MIN_VECTORS:
            continue
        # direction is where the wind blows FROM; the anvil moves the opposite way
        u = -spd[near] * np.sin(np.radians(dirn[near])) * 3.6
        v = -spd[near] * np.cos(np.radians(dirn[near])) * 3.6
        U, V = float(u.mean()), float(v.mean())
        spread = float(np.sqrt(u.var() + v.var()))
        unc = max(WIND_MIN_UNC_KT * 1.852, spread)
        kt = np.hypot(U, V) / 1.852
        frm = float(np.degrees(np.arctan2(-U, -V)) % 360)
        return {"u_kmh": U, "v_kmh": V, "unc_kmh": unc, "kt": round(kt, 1), "from_deg": round(frm),
                "n": n, "band": band, "label": f"GOES-19 band {int(band[1:])} winds {st:%H:%M}Z, "
                f"{n} anvil-level vectors: {kt:.0f} kt from {frm:.0f} deg, +/-{unc / 1.852:.0f} kt"}
    return None


def glm_distance(flashes, valid_iso, la, lo):
    """(distance in nmi from every cell to the nearest flash in the 30 minutes up to the frame,
    lats, lons of those flashes). Measured from the flashes themselves, so a flash just off the
    map still reaches cells within 10 nmi of it."""
    from scipy.spatial import cKDTree
    t = datetime.datetime.strptime(valid_iso[:19], "%Y-%m-%dT%H:%M:%S")
    t0 = t - datetime.timedelta(minutes=GLM_WINDOW_MIN)
    pts = [(a, b) for a, b, tt in flashes if t0 < tt <= t]
    if not pts:
        return np.full((la.size, lo.size), np.inf, np.float32), np.array([]), np.array([])
    lat0 = float(np.mean(la))
    kx = 111.32 * np.cos(np.radians(lat0))
    P = np.array([((b - lo[0]) * kx, (a - la[0]) * 111.32) for a, b in pts])
    LO, LA = np.meshgrid(lo, la)
    G = np.column_stack([((LO - lo[0]) * kx).ravel(), ((LA - la[0]) * 111.32).ravel()])
    d, _ = cKDTree(P).query(G)
    arr = np.array(pts)
    return (d.reshape(la.size, lo.size) / 1.852).astype(np.float32), arr[:, 0], arr[:, 1]


# --------------------------------------------------------------------------------------
# Local sensors: MERLIN lightning and LPLWS field mills (v20)
# --------------------------------------------------------------------------------------
# Public codiapi services, one per feed. What is known comes from records a GitHub Actions
# runner fetched on 17 Oct 2026 (B. Cizek's ER_Weather_COP snapshot), not from our own probe:
#   - CG and cloud-to-cloud are SEPARATE hosts (lightningc2g / lightningc2c). Records carry id,
#     event_datetime (UTC, no zone, microseconds), lat_decimal, lon_decimal, signal_strength,
#     sensor_ids, an error ellipse (cloud-to-cloud semi-major axes reach ~10 km) and raw_line.
#   - /range takes 'date>' / 'date<' / 'lat>' ... keys, which requests sends as date%3E= - the
#     server decodes them. A window holding over 100,000 events is refused, so windows are
#     bisected.
#   - LPLWS /latest returns one row per mill for the newest minute: field_mill (int), value
#     (int V/m; fair weather read +80 to +226, so positive is fair weather), flag ('X' or null,
#     meaning unconfirmed - ignored), lat, lon, reading_date + reading_time (UTC), loaded_at
#     (local time with offset; ~2 minutes after the reading).
#   - Two mills (13, 33) carry API coordinates outside the Cape; their surveyed positions are
#     below. Five mills (1, 13, 27, 33, 34) read exactly 0 - dead or offline, not calm - so a mill
#     with no nonzero reading in the window is treated as NOT reporting. FM13 was retired in 2020
#     and FM35 replaced it at the same site; FM13 still appears, at 0.
#   - Reachability from Actions is intermittent: Cizek's snapshots show read timeouts on all four
#     hosts at once, and Colab (Google Cloud) never completed a TLS handshake. Hence a background
#     thread, short timeouts, no retries, and caches that let a later run backfill a gap.
# Not yet validated live: LPLWS /range. If it fails, /latest (validated) is used and the field-mill
# window is marked "sampled" - one minute per run, which can miss a short spike.
MERLIN = {"cg": "https://codiapi-lightningc2g.rad.spaceforce.mil/lightning_c2g",
          "cc": "https://codiapi-lightningc2c.rad.spaceforce.mil/lightning_c2c"}
LPLWS = "https://codiapi-lplws1.rad.spaceforce.mil/lplws_1min"
LOCAL_TIMEOUT = (6, 20)        # connect (includes the TLS handshake), read - seconds
LOCAL_JOIN_S = 75              # the main thread waits at most this long for the local fetch
MERLIN_KEEP_MIN = 95           # same reach as the GLM cache
MERLIN_CHUNK_MIN = 10
MERLIN_MAX_CALLS = 40          # per feed per run, bisection included
MERLIN_LAG_OK_MIN = 3          # coverage may end this far before a frame and still count as current
FM_KEEP_MIN = 80               # oldest backfilled frame (~55 min) + the 15-minute window + slack
FM_LAG_OK_MIN = 4              # LPLWS loads ~2 minutes after the reading
FM_CAPE_BOX = (28.0, 29.3, -81.5, -80.0)   # API coordinates outside this are not trusted
FM_SITES = {1: (28.70363, -80.66857), 2: (28.68755, -80.71951), 4: (28.66433, -80.63913),
            5: (28.65818, -80.69936), 6: (28.64397, -80.66639), 7: (28.64238, -80.74762),
            8: (28.63888, -80.62254), 9: (28.62142, -80.60797), 10: (28.62412, -80.70151),
            11: (28.60576, -80.67387), 12: (28.60226, -80.64094), 13: (28.60205, -80.58942),
            14: (28.57852, -80.60859), 15: (28.57676, -80.64310), 16: (28.57530, -80.57311),
            17: (28.56205, -80.67003), 18: (28.55518, -80.70212), 19: (28.54948, -80.62039),
            20: (28.54170, -80.64443), 21: (28.52575, -80.62210), 22: (28.50693, -80.69359),
            24: (28.47859, -80.67487), 25: (28.42474, -80.66380), 26: (28.54938, -80.56699),
            27: (28.50308, -80.55812), 28: (28.46024, -80.52722), 29: (28.46256, -80.58356),
            30: (28.47511, -80.55842), 31: (28.42033, -80.59116), 32: (28.43688, -80.58010),
            33: (28.41532, -80.60582), 34: (28.44982, -80.56350), 35: (28.60205, -80.58942)}
_ISO = "%Y-%m-%dT%H:%M:%S"


def _merlin_box():
    mlat, mlon = GLM_MARGIN_NM * 1.852 / 111.32, GLM_MARGIN_NM * 1.852 / (111.32 * 0.87)
    return {"lat>": round(DOMAIN["lat_min"] - mlat, 3), "lat<": round(DOMAIN["lat_max"] + mlat, 3),
            "lon>": round(DOMAIN["lon_min"] - mlon, 3), "lon<": round(DOMAIN["lon_max"] + mlon, 3)}


def _local_session():
    """No adapter retries: an unreachable host should cost one connect timeout, not four."""
    s = requests.Session()
    s.headers.update({"User-Agent": "CloudScope-MRMS/2.0 (launch weather nowcast)",
                      "Accept": "application/json"})
    return s


class _TooMany(Exception):
    pass


def _local_rows(sess, url, params=None):
    r = sess.get(url, params=params, timeout=LOCAL_TIMEOUT)
    if r.status_code >= 400:
        body = (r.text or "")[:2000].lower().replace(",", "")
        if "100000" in body or "too many" in body:
            raise _TooMany(body[:200])
        raise requests.HTTPError(f"HTTP {r.status_code}: {body[:160]}")
    data = r.json()
    if not isinstance(data, list):
        raise ValueError(f"expected a list, got {type(data).__name__}")
    return data


def _parse_utc(s):
    """'2026-10-17T12:55:24.677000' (or with 7 decimals, a space, or a Z) -> naive UTC."""
    if not s:
        return None
    s = str(s).strip().replace(" ", "T").rstrip("Z")
    if "+" in s[10:] or s[10:].count("-"):
        s = s[:19]                       # an offset would mean local time: not expected here
    head, _, frac = s.partition(".")
    try:
        t = datetime.datetime.strptime(head[:19], _ISO)
    except ValueError:
        return None
    if frac.isdigit():
        t += datetime.timedelta(microseconds=int(frac[:6].ljust(6, "0")))
    return t


def _load_json(path, default):
    try:
        with open(path) as fp:
            return json.load(fp)
    except Exception:
        return default


def _save_json(path, obj):
    """Atomic, so a run cancelled mid-write never leaves a torn cache for the next run."""
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "w") as fp:
        json.dump(obj, fp, separators=(",", ":"))
    os.replace(tmp, path)


def _coverage(cache, now, keep_min):
    """Where this run's fetch must start, and the coverage it extends (or a fresh one)."""
    keep0 = now - datetime.timedelta(minutes=keep_min)
    cto = _parse_utc(cache.get("covered_to"))
    cfrom = _parse_utc(cache.get("covered_from"))
    if cto is None or cfrom is None or cto < keep0:
        return keep0, keep0, True                      # nothing reusable: start over
    return max(keep0, cto - datetime.timedelta(minutes=2)), max(cfrom, keep0), False


def merlin_update(sess, state_dir, kind, now):
    """Bring one MERLIN feed's rolling cache up to `now`. Never raises. Returns
    {pts:[(lat, lon, t)], from, to, ok, calls, note}.

    Windows are fetched NEWEST FIRST, so if the call budget or the host gives out part-way, what
    is missing is the oldest data, not the 30 minutes the newest frame needs. Coverage is one
    interval: if the backward walk reaches the old coverage the two join; if not, coverage
    restarts at the oldest window reached (older cached events are still kept - they can only
    add lightning)."""
    path = os.path.join(state_dir, f"merlin_{kind}.json")
    cache = _load_json(path, {})
    start, cfrom, fresh = _coverage(cache, now, MERLIN_KEEP_MIN)
    old_to = None if fresh else _parse_utc(cache.get("covered_to"))
    rows = {} if fresh else {int(r[0]): r for r in cache.get("rows", [])}
    box, calls = _merlin_box(), [0]
    err, reached = None, now          # reached: oldest edge fetched so far, walking backwards

    def chunk(a, b):
        calls[0] += 1
        if calls[0] > MERLIN_MAX_CALLS:
            raise RuntimeError("too many requests for this window")
        p = {"date>": a.strftime(_ISO), "date<": b.strftime(_ISO), **box, "sort": "asc"}
        try:
            return _local_rows(sess, MERLIN[kind] + "/range", p)
        except _TooMany:
            if (b - a).total_seconds() <= 2:
                raise
            mid = a + datetime.timedelta(seconds=int((b - a).total_seconds()) // 2)
            return chunk(mid, b) + chunk(a, mid)

    try:
        b = now
        while b > start:
            a = max(b - datetime.timedelta(minutes=MERLIN_CHUNK_MIN), start)
            for r in chunk(a, b):
                t = _parse_utc(r.get("event_datetime"))
                la_, lo_ = r.get("lat_decimal"), r.get("lon_decimal")
                if r.get("id") is None or t is None or not isinstance(la_, (int, float)) \
                        or not isinstance(lo_, (int, float)):
                    continue
                rows[int(r["id"])] = [int(r["id"]), float(la_), float(lo_), t.strftime(_ISO + ".%f")]
            reached = a
            b = a
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:140]}"
    if reached >= now:                                   # nothing fetched this run
        cto = old_to
        cfrom = cfrom if old_to is not None else None
    else:
        cto = now
        if not (old_to is not None and reached <= old_to):
            cfrom = reached                              # did not join the old coverage
    keep0 = now - datetime.timedelta(minutes=MERLIN_KEEP_MIN)
    if cfrom is not None:
        cfrom = max(cfrom, keep0)
    kept = [r for r in rows.values() if _parse_utc(r[3]) and _parse_utc(r[3]) >= keep0]
    out = {"rows": kept, "covered_from": cfrom.strftime(_ISO) if cfrom else None,
           "covered_to": cto.strftime(_ISO) if cto else None}
    try:
        _save_json(path, out)
    except Exception as e:
        logging.warning(f"MERLIN {kind}: cache not saved ({e})")
    note = (f"MERLIN {kind.upper()}: " + (f"FAILED after {calls[0]} call(s) - {err}; " if err else
                                          f"{calls[0]} call(s); ")
            + f"{len(kept)} events cached, covered {out['covered_from'] or '-'} to "
              f"{out['covered_to'] or '-'}")
    logging.info(note)
    return {"pts": [(r[1], r[2], _parse_utc(r[3])) for r in kept], "from": cfrom, "to": cto,
            "ok": err is None, "calls": calls[0], "note": note}


def _mill_position(mill, lat, lon):
    """(lat, lon) from the API if it lies on the Cape, else the surveyed site; (None, None) if
    neither - one return shape on every path."""
    b = FM_CAPE_BOX
    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)) \
            and b[0] <= lat <= b[1] and b[2] <= lon <= b[3]:
        pos = (float(lat), float(lon))
    else:
        pos = FM_SITES.get(mill, (None, None))
    return pos


def lplws_update(sess, state_dir, now):
    """Bring the field-mill cache up to `now`. Never raises. Returns
    {rows:[(mill, lat, lon, t, value)], from, to, mode, ok, note}. mode is "range" (every minute)
    or "latest" (one minute per run - "sampled")."""
    path = os.path.join(state_dir, "lplws.json")
    cache = _load_json(path, {})
    start, cfrom, fresh = _coverage(cache, now, FM_KEEP_MIN)
    rows = {} if fresh else {(int(r[0]), r[3]): r for r in cache.get("rows", [])}
    mode, err, got, cto = "range", None, None, None

    def take(recs):
        n = 0
        for r in recs:
            m, v = r.get("field_mill"), r.get("value")
            t = _parse_utc(f"{r.get('reading_date')}T{r.get('reading_time')}")
            if not isinstance(m, int) or not isinstance(v, (int, float)) or t is None:
                continue
            pos = _mill_position(m, r.get("lat"), r.get("lon"))
            if pos[0] is None:
                continue
            rows[(m, t.strftime(_ISO))] = [m, pos[0], pos[1], t.strftime(_ISO), float(v)]
            n += 1
        return n

    try:
        got = take(_local_rows(sess, LPLWS + "/range",
                               {"date>": start.strftime(_ISO), "date<": now.strftime(_ISO)}))
        cto = now
    except requests.HTTPError as e:
        # The server answered but refused /range: fall back to the validated /latest.
        mode = "latest"
        try:
            got = take(_local_rows(sess, LPLWS + "/latest"))
            cto = now
            err = f"/range refused ({str(e)[:80]}); one minute from /latest"
        except Exception as e2:
            err = f"{type(e2).__name__}: {str(e2)[:140]}"
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:140]}"
    ok = cto is not None
    if mode == "latest" and ok:
        # A sample, not a record: coverage is just the span of readings held, and frames judge a
        # sampled window by whether any reading falls inside it (local_for_frame).
        ts = [_parse_utc(r[3]) for r in rows.values()]
        cfrom, cto = (min(ts), max(ts)) if ts else (None, None)
    if not ok:
        cto = None if fresh else _parse_utc(cache.get("covered_to"))
        cfrom = None if fresh else cfrom
    keep0 = now - datetime.timedelta(minutes=FM_KEEP_MIN)
    kept = [r for r in rows.values() if _parse_utc(r[3]) >= keep0]
    out = {"rows": kept, "covered_from": cfrom.strftime(_ISO) if cfrom else None,
           "covered_to": cto.strftime(_ISO) if cto else None,
           "mode": mode if ok else cache.get("mode", "range")}
    try:
        _save_json(path, out)
    except Exception as e:
        logging.warning(f"LPLWS: cache not saved ({e})")
    mills = sorted({r[0] for r in kept})
    note = (f"LPLWS ({out['mode']}): " + (f"{got} reading(s) fetched" if got is not None else "FAILED")
            + (f" - {err}" if err else "") + f"; {len(kept)} cached from {len(mills)} mills, "
              f"covered {out['covered_from'] or '-'} to {out['covered_to'] or '-'}")
    logging.info(note)
    return {"rows": [(r[0], r[1], r[2], _parse_utc(r[3]), r[4]) for r in kept], "from": cfrom,
            "to": cto, "mode": out["mode"], "ok": ok, "note": note}


def local_update(state_dir):
    """All three local feeds, in parallel, each on its own session. Never raises."""
    now = _utcnow().replace(microsecond=0)
    os.makedirs(state_dir, exist_ok=True)
    jobs = {"cg": lambda s: merlin_update(s, state_dir, "cg", now),
            "cc": lambda s: merlin_update(s, state_dir, "cc", now),
            "fm": lambda s: lplws_update(s, state_dir, now)}
    out = {"now": now}
    with ThreadPoolExecutor(max_workers=3) as pool:
        fut = {k: pool.submit(fn, _local_session()) for k, fn in jobs.items()}
        for k, f in fut.items():
            try:
                out[k] = f.result()
            except Exception as e:                   # the updaters never raise; belt and braces
                out[k] = None
                logging.warning(f"local feed {k}: {type(e).__name__}: {e}")
    return out


def local_from_cache(state_dir):
    """What the caches hold, without touching the network - used if the fetch thread overruns."""
    out = {"now": _utcnow().replace(microsecond=0)}
    for kind in ("cg", "cc"):
        c = _load_json(os.path.join(state_dir, f"merlin_{kind}.json"), {})
        out[kind] = {"pts": [(r[1], r[2], _parse_utc(r[3])) for r in c.get("rows", [])],
                     "from": _parse_utc(c.get("covered_from")), "to": _parse_utc(c.get("covered_to")),
                     "ok": False, "note": f"MERLIN {kind.upper()}: cache only (fetch overran)"}
    c = _load_json(os.path.join(state_dir, "lplws.json"), {})
    out["fm"] = {"rows": [(r[0], r[1], r[2], _parse_utc(r[3]), r[4]) for r in c.get("rows", [])],
                 "from": _parse_utc(c.get("covered_from")), "to": _parse_utc(c.get("covered_to")),
                 "mode": c.get("mode", "range"), "ok": False, "note": "LPLWS: cache only"}
    return out


class LocalFetch:
    """Runs local_update in a daemon thread so it overlaps the MRMS downloads. result() waits at
    most `timeout` seconds; past that the caches are read instead and the run carries on."""

    def __init__(self, state_dir):
        self.state_dir, self.value = state_dir, None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            self.value = local_update(self.state_dir)
        except Exception as e:
            logging.warning(f"local sensors: {type(e).__name__}: {e}")

    def result(self, timeout=None):
        # Read LOCAL_JOIN_S at call time: a default argument would freeze it at import.
        self.thread.join(LOCAL_JOIN_S if timeout is None else timeout)
        if self.value is not None:
            return self.value
        logging.warning("local sensors: fetch still running or failed; using cached data")
        return local_from_cache(self.state_dir)


def _points_nm(pts, la, lo):
    """Distance in nmi from every cell to the nearest of [(lat, lon)]; inf with no points."""
    from scipy.spatial import cKDTree
    if not len(pts):
        return np.full((la.size, lo.size), np.inf, np.float32)
    kx = 111.32 * np.cos(np.radians(float(np.mean(la))))
    P = np.array([((b - lo[0]) * kx, (a - la[0]) * 111.32) for a, b in pts])
    LO, LA = np.meshgrid(lo, la)
    G = np.column_stack([((LO - lo[0]) * kx).ravel(), ((LA - la[0]) * 111.32).ravel()])
    d, _ = cKDTree(P).query(G)
    return (d.reshape(la.size, lo.size) / 1.852).astype(np.float32)


def _cover_state(feed, t0, t1, lag_min):
    """'full' if the feed's coverage spans [t0, t1] (allowing lag at the end), 'partial' if it
    overlaps, 'none' otherwise."""
    if not feed or feed.get("from") is None or feed.get("to") is None:
        return "none"
    if feed["to"] < t0 or feed["from"] > t1:
        return "none"
    if feed["from"] <= t0 and feed["to"] >= t1 - datetime.timedelta(minutes=lag_min):
        return "full"
    return "partial"


def local_for_frame(local, valid_iso, la, lo):
    """Per-frame lightning and field-mill grids from the local caches.

    Returns {ltg_nm, cg_pts, cc_pts, cover_cg, cover_cc, merlin_ok, fm_hot_nm, fm_eval_nm,
    fm_max, mills, fm_cover, fm_mode}. MERLIN events only ever ADD lightning; merlin_ok (full
    30-minute coverage on either feed) is what lets MERLIN stand in for GLM and NLDN in the
    fail-closed check."""
    t = datetime.datetime.strptime(valid_iso[:19], _ISO)
    shape = (la.size, lo.size)
    out = {"ltg_nm": None, "cg_pts": [], "cc_pts": [], "cover_cg": "none", "cover_cc": "none",
           "merlin_ok": False, "fm_hot_nm": np.full(shape, np.inf, np.float32),
           "fm_eval_nm": np.full(shape, np.inf, np.float32),
           "fm_max": np.full(shape, np.nan, np.float32), "mills": [], "fm_cover": "none",
           "fm_mode": None}
    if not local:
        return out
    t0 = t - datetime.timedelta(minutes=GLM_WINDOW_MIN)
    allp = []
    for kind in ("cg", "cc"):
        feed = local.get(kind)
        cov = _cover_state(feed, t0, t, MERLIN_LAG_OK_MIN)
        out[f"cover_{kind}"] = cov
        if cov == "none":
            continue
        pts = [(a, b) for a, b, tt in feed["pts"] if tt is not None and t0 < tt <= t]
        out[f"{kind}_pts"] = pts
        allp += pts
        out["merlin_ok"] |= cov == "full"
    if allp or out["merlin_ok"]:
        out["ltg_nm"] = _points_nm(allp, la, lo)

    fm = local.get("fm")
    w0 = t - datetime.timedelta(minutes=LLCC["field_mill_window_min"])
    if fm and fm.get("mode") == "latest":
        cov = "sampled" if any(tt is not None and w0 < tt <= t for _, _, _, tt, _ in fm["rows"]) \
            else "none"
    else:
        cov = _cover_state(fm, w0, t, FM_LAG_OK_MIN)
    out["fm_cover"], out["fm_mode"] = cov, (fm or {}).get("mode")
    if cov == "none":
        return out
    per = {}
    for m, a, b, tt, v in fm["rows"]:
        if tt is None or not (w0 < tt <= t):
            continue
        d = per.setdefault(m, {"lat": a, "lon": b, "max": 0.0, "nonzero": False, "n": 0,
                               "last_t": None, "last": None})
        d["max"] = max(d["max"], abs(v))
        d["nonzero"] |= v != 0
        d["n"] += 1
        if d["last_t"] is None or tt > d["last_t"]:
            d["last_t"], d["last"] = tt, v
    rep = {m: d for m, d in per.items() if d["nonzero"]}
    out["mills"] = [[m, round(d["lat"], 5), round(d["lon"], 5), int(round(d["max"])),
                     int(round(d["last"])), d["n"], bool(m in rep)] for m, d in sorted(per.items())]
    if not rep:
        return out
    thr = LLCC["field_mill_vm"]
    hot = [(d["lat"], d["lon"]) for d in rep.values() if d["max"] >= thr]
    out["fm_hot_nm"] = _points_nm(hot, la, lo)
    out["fm_eval_nm"] = _points_nm([(d["lat"], d["lon"]) for d in rep.values()], la, lo)
    # Strongest reporting mill within the 5 nmi standoff of each cell, for the readout.
    LO, LA = np.meshgrid(lo, la)
    kx = 111.32 * np.cos(np.radians(float(np.mean(la))))
    best = np.full(shape, -1.0, np.float32)
    for d in rep.values():
        dist = np.hypot((LO - d["lon"]) * kx, (LA - d["lat"]) * 111.32) / 1.852
        best = np.where(dist <= LLCC["field_mill_nm"], np.maximum(best, d["max"]), best)
    out["fm_max"] = np.where(best >= 0, best, np.nan).astype(np.float32)
    return out


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


# --------------------------------------------------------------------------------------
# Convective / stratiform separation - Steiner, Houze & Yuter (1995)
# --------------------------------------------------------------------------------------
# Without it, any non-anvil echo reaching -10 C was "cumulus": a stratiform rain shield picked
# up a 5 nmi cumulus standoff, and the readout invented a cumulonimbus core to explain it.
# Steiner et al. (1995, J. Appl. Meteor. 34, 1978-2007) is the standard separation, and is what
# Py-ART's steiner_conv_strat implements. It wants reflectivity on a constant-height surface
# BELOW the melting layer, which the 3-D volume supplies: the level nearest 3 km and at least
# 1 km under the freezing level, so the bright band cannot masquerade as a convective peak.
#
#   background   linear-Z mean of echo within 11 km
#   centre       Z >= 40 dBZ, or Z - Zbg >= dZcc, where dZcc = 10 - Zbg^2/180 (0 <= Zbg < 42.43),
#                10 below 0 dBZ background, 0 above 42.43
#   radius       1 to 5 km around each centre, growing with Zbg (<25, 25-30, 30-35, 35-40, 40+)
STEINER = {"bg_radius_km": 11.0, "intense_dbz": 40.0, "target_km": 3.0,
           "below_freezing_km": 1.0, "min_km": 1.5}
# Isolated cells. Steiner judges peakedness against the mean of ECHO within 11 km, so a lone
# cell is measured against itself and is never peaked: a uniform 34 dBZ tower in clear air was
# classed stratiform in testing - layered cloud, no cumulus standoffs, the unsafe direction, and
# exactly Florida's isolated afternoon cumulus. Powell, Houze & Brodzik (2016) added explicit
# isolated-convective categories for this weakness. Here: an echo object small enough to fit in
# the background window has no meaningful background, so it is convective. Every error this
# makes is conservative - a small stratiform patch gets called convective, never the reverse.
ISOLATED_KM2 = float(np.pi * 11.0 ** 2)      # ~380 km2, the background window's area


def steiner_level(heights, z0_km):
    """Index of the volume level Steiner runs on: highest level at or below both 3 km and
    1 km under freezing, and not below 1.5 km."""
    top = min(STEINER["target_km"], z0_km - STEINER["below_freezing_km"])
    ok = [k for k, h in enumerate(heights) if STEINER["min_km"] <= h <= top + 1e-6]
    if ok:
        return ok[-1]
    near = [k for k, h in enumerate(heights) if h >= STEINER["min_km"]]
    return near[0] if near else 0


def steiner_conv_strat(z, dlat_km, dlon_km):
    """(convective, stratiform) masks from one constant-height reflectivity field in dBZ.
    Sentinels (-99 no echo, -999 no coverage) are simply not echo."""
    from scipy.ndimage import convolve
    echo = z >= 0.0
    zlin = np.where(echo, 10.0 ** (z / 10.0), 0.0)
    k = _disc(STEINER["bg_radius_km"] / 1.852, dlat_km, dlon_km).astype(float)
    num = convolve(zlin, k, mode="constant", cval=0.0)
    cnt = convolve(echo.astype(float), k, mode="constant", cval=0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        zbg = np.where(cnt > 0, 10.0 * np.log10(np.maximum(num / np.maximum(cnt, 1e-9), 1e-9)), -99.0)
    dzcc = np.where(zbg < 0, 10.0, np.where(zbg < 42.43, 10.0 - zbg ** 2 / 180.0, 0.0))
    centre = echo & ((z >= STEINER["intense_dbz"]) | ((z - zbg) >= dzcc))
    radius = np.select([zbg < 25, zbg < 30, zbg < 35, zbg < 40], [1, 2, 3, 4], 5)
    conv = np.zeros(z.shape, bool)
    for r in range(1, 6):
        m = centre & (radius == r)
        if m.any():
            conv |= maximum_filter(m.astype(np.uint8),
                                   footprint=_disc(r / 1.852, dlat_km, dlon_km)) > 0
    conv &= echo
    lab, n = label(echo, structure=np.ones((3, 3)))
    if n:
        area = np.bincount(lab.ravel(), minlength=n + 1) * dlat_km * dlon_km
        small = area <= ISOLATED_KM2
        small[0] = False
        conv |= small[lab]
    return conv, echo & ~conv


def separation(vol, heights, vbase, z0, iso, la, lo, vtop=None):
    """Everything the classifier reads from the volume.

    convective / stratiform   Steiner at the chosen level
    known                     columns where that level is not blind - elsewhere the split is
                              unknown and the conservative cumulus treatment stays
    below0                    echo base below the 0 C height, or a base the radar cannot see
                              under: what fails part (a) of the anvil exception
    """
    z0_m = iso[0.0] if iso and 0.0 in iso else float(np.nanmean(z0))
    k = steiner_level(list(heights), z0_m / 1000.0)
    lvl = vol[k]
    dlat_km, dlon_km = _spacing(la, lo)
    conv, strat = steiner_conv_strat(lvl, dlat_km, dlon_km)
    known = lvl > -900.0
    base_idx = (vbase & 63).astype(int)
    has = vbase != 255
    base_h = np.where(has, np.asarray(heights)[np.clip(base_idx, 0, len(heights) - 1)] * 1000.0,
                      np.inf)
    blind_below = has & ((vbase & 64) > 0)
    top_m = None
    if vtop is not None:
        top_m = np.where(vtop != 255, np.asarray(heights)[np.clip(vtop.astype(int), 0, len(heights) - 1)]
                         * 1000.0, np.nan)
    return {"conv": conv, "strat": strat & known, "known": known, "top_m": top_m,
            "clear_low": known & (lvl < 0.0),
            "below0": has & ((base_h < z0_m) | blind_below), "level_km": float(heights[k])}


def separation_lite(lvl, height_km, la, lo):
    """Steiner from ONE level, for frames without a full volume (archive frames, or a volume
    that failed). Measured 7 Oct 2026, MRMS PrecipFlag agreed with Steiner on 96.8% of cells -
    but every disagreement (611 cells) was Steiner convective / PrecipFlag stratiform, never the
    reverse, so PrecipFlag is the LESS conservative of the two exactly where it matters; and it
    said nothing at all about 52% of the box, where weak elevated echo sits under its "no
    precipitation" code. One 3-D level is ~600 KB and a second to decode, so every frame runs
    the same algorithm instead. Part (a) of the anvil exception still needs echo bases, so it
    falls back to the 0 C slice on these frames."""
    dlat_km, dlon_km = _spacing(la, lo)
    conv, strat = steiner_conv_strat(lvl, dlat_km, dlon_km)
    known = lvl > -900.0
    return {"conv": conv, "strat": strat & known, "known": known, "below0": None,
            "clear_low": known & (lvl < 0.0), "level_km": float(height_km)}


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


def attached_history(frames, valid_iso, la, lo, wind=None):
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
            if wind is not None:
                # Moved DOWNWIND by the measured anvil-level wind, then grown by its uncertainty -
                # instead of grown 32 kt in every direction, including upwind where an anvil
                # cannot go, and short of where a fast jet can carry one.
                from scipy.ndimage import shift as _shift
                hrs = dt / 60.0
                dj = -wind["v_kmh"] * hrs / dlat_km          # rows run north to south
                di = wind["u_kmh"] * hrs / dlon_km
                moved = _shift(att.astype(np.float32), (dj, di), order=0, cval=0.0) > 0.5
                r_nm = wind["unc_kmh"] * hrs / 1.852
            else:
                moved = att
                r_nm = DRIFT_KMH * dt / 60.0 / 1.852
            union |= maximum_filter(moved.astype(np.uint8),
                                    footprint=_disc(max(r_nm, 0.5), dlat_km, dlon_km)) > 0
    return union if seen else None


def evaluate(F, z0, la, lo, history=None, iso=None, sep=None, ltg_nm=None, sat=None, fm=None):
    """Traffic light, per-rule grids, cloud class and echo-top level for every cell.

    ltg_nm: distance (nmi) to the nearest GLM flash or MERLIN event in the 30-minute window,
    combined by the caller. fm: local_for_frame's field-mill grids, or None (4.1.2 not evaluated).
    """
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

    # Measured from the XMR sounding when available; otherwise the freezing level plus an
    # assumed lapse rate, which ran ~600 m low at -20 C on the day it was checked.
    if iso:
        z5 = np.full(comp.shape, iso[5.0], np.float32)
        z20 = np.full(comp.shape, iso[-20.0], np.float32)
    else:
        z5 = z0 - L["plus5_c"] / LAPSE_C_PER_KM * 1000.0
        z20 = z0 + 20.0 / LAPSE_C_PER_KM * 1000.0

    aloft = echo & ~any_iso & ((hmax >= z20) | (F["super"] >= 0.0))

    # Echo that missed every isotherm slice but whose strongest return sits ABOVE freezing:
    # it lies BETWEEN two slices. The fallback treated all echo with no slice hits as below
    # freezing, so a thin layer at 8.0 km between the -15 and -20 C slices was called a shallow
    # warm shower and could trip no cumulus rule. Its top temperature is read off its height
    # instead - from the sounding when there is one, otherwise the freezing level and assumed
    # lapse rate. Worked out here, before the rules, so they can use it.
    def z_of(tc):
        if iso and tc in iso:
            return np.full(comp.shape, iso[tc], np.float32)
        return z0 + (-tc) / LAPSE_C_PER_KM * 1000.0
    gap = echo & ~any_iso & ~aloft & (hmax >= z_of(0.0))
    gap_m10 = gap & (hmax >= z_of(-10.0))
    top_below_0c = any_iso | aloft | gap
    top_to_plus5 = any_iso | (echo & (hmax >= z5))

    core = (comp >= L["core_dbz"]) | (F["vii"] > 0.0)
    # ELEVATED echo: above freezing (it shows at an isotherm level, or sits only aloft) but with
    # nothing at the ~3 km Steiner level where the radar can see. Its base is well above where
    # cumulus bases sit here (~1 km), so it is never cumulus. Found at a shield's thinning edge:
    # the top there sags below -20 C, so the fringe dropped out of the anvil test, Steiner could
    # not judge it (no echo at 3 km), and it fell to the cumulus fallback. Now it joins the anvil
    # candidates - contiguous with an attached anvil it is that anvil's fringe; otherwise it
    # becomes detached anvil or an elevated layer by the same origin test as floating echo.
    elev = (echo & sep["clear_low"] & (any_iso | aloft)) if sep is not None \
        else np.zeros(comp.shape, bool)
    anvil = ((e["r20"] | aloft | elev) & ~core) & echo
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
    floating = anvil & ~has_core[lab] & (~e["r0"] | elev)
    # Origin: floating echo is a detached anvil only where an attached anvil was recently
    # (the drift-grown history mask). With no history at all - the first runs after deploying
    # - fall back to calling it detached anvil, the conservative choice.
    if history is None:
        detached = floating
        elevated = np.zeros(comp.shape, bool)
    else:
        detached = floating & history
        elevated = floating & ~history
        # Anvil until it rains out (45 WS): a shield whose cores have died but which is still
        # RAINING - echo down through 0 C - is a decayed anvil, not cumulus and not layered
        # cloud. The test: rooted, unconnected anvil-level echo, lying where an attached anvil
        # recently was, and STRATIFORM. Convective echo there is a new tower and stays cumulus,
        # so a cell building under an old anvil never gets downgraded to a 3 nmi standoff.
        # Without the separation this case stays cumulus, the conservative stand-in.
        if sep is not None:
            raining_out = anvil & ~has_core[lab] & e["r0"] & history & sep["strat"]
            detached = detached | raining_out
    anvil = attached | detached

    # LLCCR 18 exception, CONSERVATIVE: any echo at the 0 C slice under an anvil within 5 nmi
    # fails it, because radar cannot separate the anvil from the precipitation it drops.
    # Part (a), conservatively: the anvil within 5 nmi must lie entirely where it is colder than
    # 0 C. With a volume that is tested on the echo base itself against the 0 C height (from the
    # sounding when there is one) - any anvil column with echo below 0 C, or a base the radar is
    # blind under, fails it. Without one it falls back to echo at the 0 C isotherm slice.
    warm_cols = sep["below0"] if (sep is not None and sep.get("below0") is not None) else e["r0"]
    anvil_warm = near((attached | detached) & warm_cols, L["excep_nm"])
    mrr = peak(np.where(comp > -90, comp, -99.0).astype(np.float32), L["mrr_search_nm"])
    mrr_ok = peak(mrr, L["mrr_eval_nm"]) < L["mrr_dbz"]
    exception = ~anvil_warm & mrr_ok
    mrr1 = peak(mrr, L["mrr_eval_nm"])     # the MRR the exceptions test, per point
    # Thick cloud layer exception (45 WS, confirmed 8 Oct 2026): no violation when MRR is below
    # +7.5 dBZ within 1 nmi - the same MRR as anvil exception part (b). Until now the rule was
    # applied without it, the conservative side.
    thick_exc = mrr1 < L["thick_mrr_dbz"]

    lightning = F["cg"] > 0.0              # NLDN CG cells, 30-minute product

    def lit_within(r):
        """Lightning within r nmi: NLDN CG, or GLM / MERLIN via ltg_nm - any source counts."""
        m = near(lightning, r)
        if ltg_nm is not None:
            m |= ltg_nm <= r
        return m

    # Once an anvil, always an anvil. Anvil echo is scored ONLY by the anvil rules - it is not
    # cumulus, so it must never trip a cumulus standoff, however cold it reaches. The first
    # version keyed every cumulus test on any echo at the isotherm, so an anvil point got
    # scored as cumulus by its own anvil echo. The parent cumulonimbus tower IS cumulus, so a
    # point near a core can still trip the cumulus rules - from the core, not the anvil.
    anvil_any = attached | detached
    # Stratiform echo (Steiner) that is not anvil is layered cloud, not cumulus. Columns where
    # the separation is unknown - no volume, or blind at the Steiner level - keep the
    # conservative cumulus treatment.
    layered = (echo & sep["strat"] & ~anvil_any & ~core) if sep is not None \
        else np.zeros(comp.shape, bool)
    # Neither an elevated layer nor stratiform cloud is cumulus: neither drives a cumulus standoff.
    not_cu = anvil_any | elevated | layered
    cu_echo = echo & ~not_cu
    # Satellite cloud tops over echo. The 0 dBZ echo top is a LOWER bound on the cloud top, so a
    # satellite top colder than the radar shows can raise a cumulus to the -10 or -20 C class - but
    # only where it plausibly belongs to that cloud: within SAT_CONSISTENCY_M above the radar echo
    # top. The satellite sees the HIGHEST cloud, so a cirrus deck over shallow cumulus sits far
    # above the echo top and is ignored. Never on non-echo cells, never on anvil or layered cloud,
    # never lowering a class. The radar echo top is the 3-D volume's when there is one, otherwise
    # the upper edge of the isotherm bracket the echo reaches.
    sat_ok = np.zeros(comp.shape, bool)
    sat_m10 = np.zeros(comp.shape, bool)
    sat_m20 = np.zeros(comp.shape, bool)
    if sat is not None:
        rtop = np.select([e["r15"], e["r10"], e["r5"], e["r0"]],
                         [z_of(-20.0), z_of(-15.0), z_of(-10.0), z_of(-5.0)], z_of(0.0))
        rtop = np.where(gap, np.maximum(rtop, hmax), rtop)
        if sep is not None and sep.get("top_m") is not None:
            rtop = np.where(np.isfinite(sep["top_m"]), sep["top_m"], rtop)
        have = echo & np.isfinite(sat["bt_c"]) & np.isfinite(sat["top_m"])
        sat_ok = have & (sat["top_m"] <= rtop + SAT_CONSISTENCY_M)
        sat_m10 = sat_ok & (sat["bt_c"] <= -10.0) & ~not_cu
        sat_m20 = sat_ok & (sat["bt_c"] <= -20.0) & ~not_cu
    cu10 = (e["r10"] | gap_m10 | sat_m10) & ~not_cu
    cu20 = (e["r20"] | sat_m20) & ~not_cu
    sat_up = (sat_m10 & ~(e["r10"] | gap_m10)) | (sat_m20 & ~e["r20"])
    # Thick cloud layer: echo spanning at least ~4,500 ft inside the 0 to -20 C band. Adjacent
    # isotherm slices 10 C apart are ~1.5 km (~5,000 ft) apart, so echo at both ends of any
    # 10-degree span counts. The first version only tested 0 to -10 C, so an elevated layer
    # spanning -5 to -15 C was never scored at all.
    # Cumulus is not scored by the thick-layer rule - 45 WS practice, confirmed 7 Oct 2026. So
    # known-convective echo (Steiner) and convective cores are excluded along with anvil. Where
    # the separation is UNKNOWN - no volume level, or blind there - the rule still applies,
    # because the echo cannot be confirmed to be cumulus.
    known_cu = core | (sep["conv"] if sep is not None else np.zeros(comp.shape, bool))
    thick = (((e["r0"] & e["r10"]) | (e["r5"] & e["r15"]) | (e["r10"] & e["r20"]))
             & ~anvil_any & ~known_cu)

    # Surface electric field (4.1.2): a reporting mill within 5 nmi read |E| >= 1000 V/m in the
    # prior 15 minutes. Distances are measured from the mills themselves, so a mill just off a cell
    # still reaches it. Not evaluated (never violated) where no reporting mill is within 5 nmi.
    fm_hot = fm["fm_hot_nm"] if fm is not None else np.full(comp.shape, np.inf, np.float32)

    def rules(m):
        lit = lit_within(L["lightning_nm"] + m)
        return {
            "lightning":       lit,
            "cumulus_through": near(cu_echo & top_to_plus5, m),
            "cumulus_5nm":     near(cu10, L["cumulus_5nm"] + m),
            "cumulus_10nm":    near(cu20, L["cumulus_10nm"] + m),
            "attached_anvil":  (near(attached, L["attached_3nm"] + m) & ~exception)
                               | (near(attached, L["attached_lightning_nm"] + m) & lit),
            "detached_anvil":  near(detached, L["detached_3nm"] + m) & ~exception,
            "field_mill":      fm_hot <= L["field_mill_nm"] + m,
            "thick_layer":     near(thick, m) & ~thick_exc,
        }

    red_rules = rules(0.0)
    yel_rules = rules(L["yellow_margin_nm"])
    red = np.logical_or.reduce([red_rules[k] for k in RULE_KEYS])
    yellow = (np.logical_or.reduce([yel_rules[k] for k in RULE_KEYS])
              | lit_within(L["lightning_watch_nm"])) & ~red

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

    # Between-slice echo (see `gap` above): level from its height.
    top[gap] = 2
    top[gap & (hmax >= z_of(-5.0))] = 3
    top[gap & (hmax >= z_of(-10.0))] = 4
    top[gap & (hmax >= z_of(-15.0))] = 5

    cls = np.zeros(comp.shape, np.uint8)
    cls[echo & (top == 1)] = 1
    cls[(top == 2) | (top == 3)] = 2
    cls[(top == 4) | (top == 5)] = 3
    cls[top == 6] = 4
    # satellite-raised cumulus (only ever upward; anvil, layered and core overwrite below)
    cls[sat_m10 & np.isin(cls, (1, 2))] = 3
    cls[sat_m20 & np.isin(cls, (1, 2, 3))] = 4
    cls[top == 7] = 7 if history is None else 8   # aloft-only echo: detached or elevated layer
    cls[elevated] = 8
    cls[layered] = 8
    cls[detached] = 7
    cls[attached] = 6
    cls[core] = 5
    cls[~echo] = 0

    dist_km = distance_transform_edt(~echo, sampling=(dlat_km, dlon_km))
    cg_nm = (distance_transform_edt(~lightning, sampling=(dlat_km, dlon_km)) / 1.852
             if lightning.any() else np.full(comp.shape, np.inf))
    ltg_near = np.minimum(cg_nm, ltg_nm) if ltg_nm is not None else cg_nm
    # The anvil exception, per point, for the readout: does an anvil standoff even apply here,
    # and which part fails. MRR is the 4 nmi composite maximum, taken within 1 nmi.
    exc_near = near(anvil_any, L["attached_3nm"])
    mrr_here = peak(mrr, L["mrr_eval_nm"])
    # The masks each rule is built from, and the MRR - recorded by the climatology as distances so
    # the standoffs and thresholds can be re-applied later without re-scoring the radar.
    feat = {"cuthru": cu_echo & top_to_plus5, "cu10": cu10, "cu20": cu20, "att": attached,
            "det": detached, "warm": (attached | detached) & warm_cols, "thick": thick,
            "ltg": lightning, "mrr1": mrr1}
    diag = {"feat": feat, "sat_ok": sat_ok, "sat_up": sat_up, "sat": sat,
            "attached": attached, "detached": detached, "elevated": elevated,
            "layered": layered, "sep": sep is not None,
            "conv": sep["conv"] if sep is not None else None,
            "strat": sep["strat"] if sep is not None else None,
            "exc_near": exc_near, "exc_a_fail": anvil_warm, "exc_b_fail": ~mrr_ok,
            "mrr": mrr_here,
            "core": core, "echo": echo,
            "dist_echo_nm": dist_km / 1.852, "lightning": lightning, "ltg_near_nm": ltg_near,
            "fm_max": fm["fm_max"] if fm is not None else np.full(comp.shape, np.nan, np.float32)}
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
            # strongest 15-minute |E| among reporting mills within 5 nmi; None = not evaluated
            "field_mill_vm": (None if not np.isfinite(diag["fm_max"][j, i])
                              else int(round(float(diag["fm_max"][j, i])))),
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


FM_COLORS = ((500.0, "#3FB97A"), (1000.0, "#F2C14E"), (np.inf, "#E5484D"))   # < 500, < 1000, >=


def _local_marks(ax, pc, diag, lightning=True):
    """Field mills (filled by 15-minute max |E|; hollow = not reporting) and MERLIN events."""
    loc = diag.get("local") or {}
    if lightning:
        cc = loc.get("cc_pts") or []
        if cc:
            a = np.array(cc)
            ax.scatter(a[:, 1], a[:, 0], marker="o", s=5, linewidths=0.4, color="#FFFFFF",
                       edgecolors=HALO, transform=pc, zorder=10)
        cg = loc.get("cg_pts") or []
        if cg:
            a = np.array(cg)
            ax.scatter(a[:, 1], a[:, 0], marker="+", s=34, linewidths=1.5, color="#FB923C",
                       transform=pc, zorder=11,
                       path_effects=[pe.withStroke(linewidth=2.8, foreground=HALO)])
    for m, lat, lon, mx, last, n, ok in loc.get("mills") or []:
        if ok:
            col = next(c for lim, c in FM_COLORS if mx < lim)
            ax.scatter([lon], [lat], marker="o", s=30, color=col, edgecolors="#E8EEF3",
                       linewidths=1.0, transform=pc, zorder=7)      # under the pad crosses (8)
        else:
            ax.scatter([lon], [lat], marker="o", s=22, facecolors="none", edgecolors="#8FA3B6",
                       linewidths=0.9, transform=pc, zorder=7)


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
    _local_marks(ax, pc, diag, lightning=False)
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
    if diag["lightning"].any():                # NLDN CG cells
        jj, ii = np.where(diag["lightning"])
        ax.scatter(lo[ii], la[jj], marker="+", s=12, linewidths=1.0, color="#FDE047",
                   transform=pc, zorder=9)
    gl = diag.get("glm_pts")
    if gl is not None and len(gl[0]):          # GLM flashes, last 30 minutes
        ax.scatter(gl[1], gl[0], marker="x", s=9, linewidths=0.9, color="#FFFFFF",
                   transform=pc, zorder=10)
    _local_marks(ax, pc, diag)
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


def _sat_plane(diag, key, offset, scale):
    if diag is None or diag.get("sat") is None:
        return np.full(diag["echo"].shape if diag is not None else (1,), 255, np.uint8)
    v = diag["sat"][key]
    return np.where(np.isfinite(v), np.clip(np.round((v + offset) * scale), 0, 254), 255).astype(np.uint8)


def _fm_plane(diag, shape):
    """Plane 13 encoding (see write_data). Floors, so a reading below 1,000 never encodes as 100."""
    if diag is None:
        return np.full(shape, 255, np.uint8)
    v = np.nan_to_num(diag["fm_max"])
    code = np.where(v < 1000.0, np.floor(v / 10.0), 100.0 + np.floor((v - 1000.0) / 100.0))
    code = np.clip(code, 0, 254)
    return np.where(np.isfinite(diag["fm_max"]), code, 255).astype(np.uint8)


def write_data(stem, status, cls, top, F, red_rules, yel_rules, vbase=None, vtop=None, diag=None):
    """Per-cell readout for the viewer: eight uint8 planes, north-up, row-major.

        0 status + 1        (0 no coverage, 1 clear, 2 watch, 3 violating)
        1 cloud class id
        2 composite dBZ * 2, 255 = no echo
        3 echo-top level    (index into TOP_LEVELS)
        4 red rule bits     (bit n = RULE_KEYS[n])
        5 yellow rule bits
        6 3-D echo base     (level index, +64 if the radar is blind below; 255 none; 253 no volume)
        7 3-D echo top      (level index; 255 none; 253 no volume)
        8 MRR * 2           (largest composite within 4 nmi, taken within 1 nmi; 255 none)
        9 flags             bit0 an anvil standoff applies here (anvil within 3 nmi)
                            bit1 exception part (a) fails   bit2 part (b) fails
                            bit3 convective   bit4 stratiform   bit5 separation was available
       10 nearest lightning (GLM or NLDN CG, last 30 min), nmi * 5; 255 = none within 50 nmi
       11 satellite cloud-top temperature, C + 100 (parallax-corrected); 255 = none
       12 satellite cloud-top height, units of 100 m; 255 = none
          (plane 9 bit6: satellite top accepted for this echo; bit7: it raised the class)
       13 field mill: strongest 15-minute max |E| among reporting mills within 5 nmi.
          0-99: |E| / 10 (10 V/m steps below 1,000); 100-253: 1,000 + (code - 100) * 100 V/m, so
          code 100 is exactly the 1,000 V/m threshold and >= 100 violates; 254 = over 16,300 V/m;
          255 = not evaluated (no reporting mill within 5 nmi, or no mill data for this frame)
       (plane 10 now includes MERLIN cloud-to-ground and cloud-to-cloud events)

    Older frames have fewer planes; the viewer checks the length.
    """
    dbz = np.where(F["comp"] >= 0, np.clip(np.round(F["comp"] * 2), 0, 254), 255)
    rbits = np.zeros(status.shape, np.uint8)
    ybits = np.zeros(status.shape, np.uint8)
    for b, k in enumerate(RULE_KEYS):
        rbits |= (red_rules[k].astype(np.uint8) << b)
        ybits |= ((yel_rules[k] & ~red_rules[k]).astype(np.uint8) << b)
    none = np.full(status.shape, VOL_NONE, np.uint8)
    if diag is not None:
        m = diag["mrr"]
        mrr = np.where(m >= 0, np.clip(np.round(m * 2), 0, 254), 255).astype(np.uint8)
        fl = (diag["exc_near"].astype(np.uint8)
              | (diag["exc_a_fail"].astype(np.uint8) << 1)
              | (diag["exc_b_fail"].astype(np.uint8) << 2))
        if diag["sep"]:
            fl |= (diag["conv"].astype(np.uint8) << 3) | (diag["strat"].astype(np.uint8) << 4) | 32
        fl |= (diag["sat_ok"].astype(np.uint8) << 6) | (diag["sat_up"].astype(np.uint8) << 7)
    else:
        mrr = np.full(status.shape, 255, np.uint8)
        fl = np.zeros(status.shape, np.uint8)
    planes = [(status + 1).astype(np.uint8), cls.astype(np.uint8), dbz.astype(np.uint8),
              top.astype(np.uint8), rbits, ybits,
              vbase if vbase is not None else none, vtop if vtop is not None else none,
              mrr, fl.astype(np.uint8),
              (np.where(diag["ltg_near_nm"] <= 50.8, np.round(diag["ltg_near_nm"] * 5), 255)
               .astype(np.uint8) if diag is not None else np.full(status.shape, 255, np.uint8)),
              _sat_plane(diag, "bt_c", 100.0, 1.0), _sat_plane(diag, "top_m", 0.0, 0.01),
              _fm_plane(diag, status.shape)]
    rel = f"{stem}.bin"
    with open(os.path.join(OUT_DIR, rel), "wb") as fp:
        fp.write(b"".join(p.tobytes() for p in planes))
    return rel


# --------------------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------------------
def build_frame(sess, sources, z0, levels=None, prior=None, snd=None, steiner_src=None, glm=None,
                goes=None, local=None):
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
    valid_pre = valid_times.get("comp") or max((v for v in valid_times.values() if v), default=None)
    loc = (local_for_frame(local, valid_pre, la, lo) if (local and valid_pre)
           else local_for_frame(None, "2000-01-01T00:00:00", la, lo))
    if F.get("cg") is None and not (glm and glm.get("ok")) and not loc["merlin_ok"]:
        gone.append("lightning (none of NLDN CG, GLM or MERLIN)")
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
    wind = goes.get("wind") if goes else None
    history = attached_history(prior or [], valid0, la, lo, wind) if valid0 else None
    iso, iso_src = isotherms_for(snd, valid0)
    valid = valid0
    stamp = valid.replace(":", "").replace("-", "")[:13]
    stem = f"frames/{stamp}"

    # The 3-D volume, for the newest frame only - fetched BEFORE classifying, because the
    # convective/stratiform separation and the anvil exception's part (a) both read it. Its
    # failure never costs the 2-D frame.
    vbase = vtop = vol_rel = vol_valid = None
    sep, vol_note = None, "not fetched"
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
                sep = separation(vol, heights, vbase, z0, iso, la, lo, vtop)
                vol_note += f"; Steiner at {sep['level_km']:g} km"
        except Exception as e:
            vol_note = f"failed: {type(e).__name__}: {e}"
        logging.info(f"3-D volume: {vol_note}")

    # No full volume: fetch just the Steiner level, so the separation still runs.
    if sep is None and steiner_src:
        url, h_km = steiner_src
        try:
            lvl, _, _, _ = decode_crop(fetch_url(sess, url), -999.0)
            if lvl.shape == (la.size, lo.size):
                sep = separation_lite(lvl, h_km, la, lo)
                vol_note += f"; Steiner on the single {h_km:g} km level"
        except Exception as e:
            logging.warning(f"Steiner level {h_km:g} km: {type(e).__name__}: {e}")

    ltg_nm, glat, glon = (glm_distance(glm["flashes"], valid0, la, lo) if glm and glm.get("ok")
                          else (None, np.array([]), np.array([])))
    # MERLIN events only ever add lightning: the nearer of GLM and MERLIN.
    if loc["ltg_nm"] is not None:
        ltg_nm = loc["ltg_nm"] if ltg_nm is None else np.minimum(ltg_nm, loc["ltg_nm"])
    fm = loc if loc["fm_cover"] != "none" else None
    sat, sat_src = None, "not used"
    if goes is not None:
        try:
            sat, sat_src = goes_tops(sess, valid0, la, lo, goes["cache"])
        except Exception as e:
            sat, sat_src = None, f"failed: {type(e).__name__}: {e}"
    status, red_rules, yel_rules, cls, top, diag = evaluate(F, z0, la, lo, history, iso, sep, ltg_nm,
                                                            sat, fm)
    diag["local"] = loc
    inside = ((glat >= DOMAIN["lat_min"]) & (glat <= DOMAIN["lat_max"])
              & (glon >= DOMAIN["lon_min"]) & (glon <= DOMAIN["lon_max"])) if glat.size else glat
    diag["glm_pts"] = (glat[inside], glon[inside]) if glat.size else (glat, glon)
    nldn_ok = F.get("cg") is not None and "NLDN_CG_030min_AvgDensity" not in missing
    glm_ok = bool(glm and glm.get("ok"))
    have = ([f"MERLIN {k.upper()}" + ("" if loc[f"cover_{k}"] == "full" else " (partial)")
             for k in ("cg", "cc") if loc[f"cover_{k}"] != "none"]
            + (["GLM"] if glm_ok else []) + (["NLDN CG"] if nldn_ok else []))
    lack = ([f"MERLIN {k.upper()}" for k in ("cg", "cc") if loc[f"cover_{k}"] == "none"]
            + ([] if glm_ok else ["GLM"]) + ([] if nldn_ok else ["NLDN CG"]))
    ltg_src = " + ".join(have) + (f"; unavailable: {', '.join(lack)}" if lack else "")
    n_rep = sum(1 for m in loc["mills"] if m[6])
    sources = {"merlin_cg": loc["cover_cg"], "merlin_cc": loc["cover_cc"], "glm": glm_ok,
               "nldn": nldn_ok, "field_mills": {"coverage": loc["fm_cover"], "mode": loc["fm_mode"],
                                                "reporting": n_rep, "listed": len(loc["mills"])}}
    pads = pad_report(status, red_rules, yel_rules, cls, top, diag, F, la, lo)
    images = render_layers(status, cls, F, diag, la, lo, stem)
    data = write_data(stem, status, cls, top, F, red_rules, yel_rules, vbase, vtop, diag)

    vt = [datetime.datetime.strptime(v, "%Y-%m-%dT%H:%M:%SZ")
          for v in valid_times.values() if v]
    skew = round((max(vt) - min(vt)).total_seconds() / 60.0, 1) if len(vt) > 1 else 0.0
    counts = {str(k): int((status == k).sum()) for k in (-1, 0, 1, 2)}
    cls_counts = {c["key"]: int((cls == c["id"]).sum()) for c in CLASSES}
    frame = {"valid": valid, "stamp": stamp, "images": images, "data": data,
             "pads": pads, "counts": counts, "class_counts": cls_counts,
             "missing": missing, "skew_min": skew,
             "volume": vol_rel, "volume_valid": vol_valid, "volume_note": vol_note,
             "freezing_km": round((iso[0.0] if iso and 0.0 in iso
                                   else float(np.nanmean(z0))) / 1000.0, 2),
             "isotherms_km": ({f"{int(t):+d}": round(z / 1000.0, 2) for t, z in iso.items()}
                              if iso else None),
             "iso_source": iso_src,
             "lightning_source": ltg_src,
             "glm_flashes_30min": int(len(diag["glm_pts"][0])),
             "merlin_cg_30min": len(loc["cg_pts"]), "merlin_cc_30min": len(loc["cc_pts"]),
             "sources": sources,
             # [mill, lat, lon, max |E| 15 min, latest value, minutes, reporting]
             "field_mills": loc["mills"],
             "sat_source": sat_src if sat is not None else f"none: {sat_src}",
             "sat_upgraded_cells": int(diag["sat_up"].sum()),
             "anvil_wind": wind}
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


def backfill_sources(sess, have_valid, newest_valid, cache, budget, steiner=None):
    """Archive sources for the frames the loop is missing, newest gap first.
    Returns [(sources, steiner_src or None)]; steiner = (level directory, height km)."""
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
        st = None
        if steiner:
            name, h_km = steiner
            u = _nearest(listing(sess, name, cache, root=ROOT3), t_comp, MATCH_TOL_MIN)
            st = (u, h_km) if u else None
        out.append((src, st))
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
    for fn in os.listdir(state_dir):            # leftovers of a run cancelled mid-write
        if fn.endswith(".tmp"):
            os.remove(os.path.join(state_dir, fn))
    # Local sensors in the background: they overlap the MRMS downloads below, and an unreachable
    # host costs the run nothing more than its connect timeout.
    local_job = LocalFetch(state_dir)

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
    goes = {"cache": {}}
    try:
        goes["wind"] = goes_anvil_wind(sess, goes["cache"])
        logging.info(goes["wind"]["label"] if goes["wind"] else
                     "anvil wind: too few band-14/band-8 vectors; drift uses the 32 kt allowance")
    except Exception as e:
        goes["wind"] = None
        logging.warning(f"anvil wind unavailable ({type(e).__name__}: {e})")
    try:
        glm = glm_update(sess, state_dir)
    except Exception as e:
        logging.warning(f"GLM unavailable ({type(e).__name__}: {e}); NLDN CG only")
        glm = None
    try:
        snd = load_soundings(sess, prev, state_dir)
    except Exception as e:
        logging.warning(f"BUFKIT unavailable ({e}); isotherms estimated")
        snd = None
    # If the newest frame's full volume fails, it still gets Steiner from the single latest level.
    steiner_latest = None
    if levels:
        z0_km = float(np.nanmean(z0)) / 1000.0 if z0 is not None else 4.8
        name, h_km = levels[steiner_level([h for _, h in levels], z0_km)]
        steiner_latest = (f"{ROOT3}/{name}/MRMS_{name}.latest.grib2.gz", h_km)
    local = local_job.result()
    newest, la, lo = build_frame(sess, {k: latest_url(p) for k, (p, _, _) in PRODUCTS.items()},
                                 z0, levels, prior=prev.get("frames", []), snd=snd,
                                 steiner_src=steiner_latest, glm=glm, goes=goes, local=local)
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
        steiner = None
        if levels:
            z0_km = (newest.get("freezing_km") or 4.8)
            k = steiner_level([h for _, h in levels], z0_km)
            steiner = levels[k]
        srcs = backfill_sources(sess, [f["valid"] for f in frames], newest["valid"],
                                cache, BACKFILL_PER_RUN, steiner)
        done = 0
        for src, st in srcs:
            if time.monotonic() - t0 > BACKFILL_BUDGET_S:
                logging.info("backfill budget spent; the rest fill on later runs")
                break
            frame, _, _ = build_frame(sess, src, z0, prior=frames, snd=snd, steiner_src=st, glm=glm,
                                      goes=goes, local=local)
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
        "local_notes": [local[k]["note"] for k in ("cg", "cc", "fm") if local.get(k)],
        "field_mill_colors": [[None if not np.isfinite(lim) else lim, c] for lim, c in FM_COLORS],
    }
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as fp:
        json.dump(manifest, fp, indent=1)
    logging.info(f"{len(frames)} frame(s) in the loop; freezing "
                 f"{'refreshed' if refreshed else 'cached'}")


if __name__ == "__main__":
    main()
