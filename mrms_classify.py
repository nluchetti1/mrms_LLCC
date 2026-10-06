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
UA = {"User-Agent": "CloudScope-MRMS/2.0 (launch weather nowcast)"}
OUT_DIR = os.environ.get("OUT_DIR", "site")

VIEWER_VERSION_EXPECTED = "mrms-v2"

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
FRAME_SPACING_MIN = 5        # spacing of backfilled frames
FREEZING_MAX_AGE_MIN = 70

# Archive backfill. On a fresh deployment the loop would otherwise hold one frame for the
# first hour, filling one per run. MRMS keeps a few hours of timestamped files beside every
# .latest, so missing frames are rebuilt from those. Bounded per run because the workflow
# cancels an in-progress run when the next trigger arrives - a backfill that tries to do all
# eleven frames at once gets cancelled and publishes nothing.
BACKFILL_PER_RUN = 4
BACKFILL_BUDGET_S = 120     # setup (apt, pip) takes ~1 min, the latest frame ~30 s, and the
                            # whole run has to finish inside the 5-minute trigger interval
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
NOT_EVALUATED = [
    "Surface electric fields (4.1.2) and their field-mill exceptions",
    "Debris clouds (4.1.6), which need an observed detachment time",
    "Smoke plumes (4.1.9)",
    "Triboelectrification (4.1.10)",
    "The 3-hour lightning clocks on the anvil rules - only 30 minutes of lightning is published",
]

# Cloud class, read off how far up the isotherm stack an echo reaches. These are what the
# radar can actually say about an echo, and each maps onto the rules it drives.
CLASSES = [
    {"id": 0, "key": "clear",    "name": "No echo",                        "color": "#00000000"},
    {"id": 1, "key": "warm",     "name": "Shallow shower, below freezing", "color": "#5E7FA3"},
    {"id": 2, "key": "cu0",      "name": "Cumulus topping 0 to −10 °C",    "color": "#4FB3C9"},
    {"id": 3, "key": "cu10",     "name": "Cumulus topping −10 to −20 °C",  "color": "#3D7FD9"},
    {"id": 4, "key": "cu20",     "name": "Cumulus topping below −20 °C",   "color": "#5B4FD0"},
    {"id": 5, "key": "core",     "name": "Convective core",                "color": "#D946A8"},
    {"id": 6, "key": "att",      "name": "Attached anvil",                 "color": "#B892F2"},
    {"id": 7, "key": "det",      "name": "Detached anvil",                 "color": "#E4D3FA"},
]
# Echo-top level, the vertical reach the class is built from.
TOP_LEVELS = ["none", "below freezing", "0 °C", "−5 °C", "−10 °C", "−15 °C", "−20 °C",
              "above −20 °C (elevated)"]

STATUS = {-1: ("No coverage", "#6B7785"), 0: ("Clear", "#3FB97A"),
          1: ("Watch", "#F2C14E"), 2: ("Violating", "#E5484D")}

# Standard reflectivity colours, plus a 0-5 dBZ band. Most tables start at 5, but every rule
# here is gated at 0 dBZ, so the weakest echo the rules act on has to be visible.
REFL_LEVELS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 95]
REFL_COLORS = ["#5D7387", "#04E9E7", "#019FF4", "#0300F4", "#02FD02", "#01C501", "#008E00",
               "#FDF802", "#E5BC00", "#FD9500", "#FD0000", "#D40000", "#BC0000", "#F800FD",
               "#9854C6", "#FDFDFD"]

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
    sub = vals[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    la = lat_axis[rows.min():rows.max() + 1]
    lo = lon_axis[cols.min():cols.max() + 1]
    if la[0] < la[-1]:
        sub, la = sub[::-1], la[::-1]
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


def evaluate(F, z0, la, lo):
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
    detached = anvil & ~has_core[lab]

    # LLCCR 18 exception, CONSERVATIVE: any echo at the 0 C slice under an anvil within 5 nmi
    # fails it, because radar cannot separate the anvil from the precipitation it drops.
    anvil_warm = near((attached | detached) & e["r0"], L["excep_nm"])
    mrr = peak(np.where(comp > -90, comp, -99.0).astype(np.float32), L["mrr_search_nm"])
    mrr_ok = peak(mrr, L["mrr_eval_nm"]) < L["mrr_dbz"]
    exception = ~anvil_warm & mrr_ok

    lightning = F["cg"] > 0.0

    def rules(m):
        lit = near(lightning, L["lightning_nm"] + m)
        return {
            "lightning":       lit,
            "cumulus_through": near(echo & top_to_plus5, m),
            "cumulus_5nm":     near(e["r10"], L["cumulus_5nm"] + m),
            "cumulus_10nm":    near(e["r20"], L["cumulus_10nm"] + m),
            "attached_anvil":  (near(attached, L["attached_3nm"] + m) & ~exception)
                               | (near(attached, L["attached_lightning_nm"] + m) & lit),
            "detached_anvil":  near(detached, L["detached_3nm"] + m) & ~exception,
            "disturbed":       near(echo & top_below_0c, m)
                               & near(comp >= L["disturbed_dbz"], L["disturbed_nm"] + m),
            "thick_layer":     near(e["r0"] & e["r10"] & ~(attached | detached), m),
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
    cls[top == 7] = 7            # elevated echo with no core under it is detached until joined
    cls[detached] = 7
    cls[attached] = 6
    cls[core] = 5
    cls[~echo] = 0

    dist_km = distance_transform_edt(~echo, sampling=(dlat_km, dlon_km))
    diag = {"attached": attached, "detached": detached, "core": core, "echo": echo,
            "dist_echo_nm": dist_km / 1.852, "lightning": lightning}
    return status, red_rules, yel_rules, cls, top, diag


def pad_report(status, red_rules, yel_rules, cls, top, diag, F, la, lo):
    out = {}
    dlat_km, dlon_km = _spacing(la, lo)
    win = _disc(10.0, dlat_km, dlon_km)
    hw = win.shape[0] // 2, win.shape[1] // 2
    names = {c["id"]: c["name"] for c in CLASSES}
    for name, (plat, plon) in SITES.items():
        j = int(np.argmin(np.abs(la - plat)))
        i = int(np.argmin(np.abs(lo - plon)))
        j0, j1 = max(0, j - hw[0]), min(la.size, j + hw[0] + 1)
        i0, i1 = max(0, i - hw[1]), min(lo.size, i + hw[1] + 1)
        box = F["comp"][j0:j1, i0:i1]
        cbox = cls[j0:j1, i0:i1]
        dbz10 = float(box.max()) if box.size and box.max() > -90 else None
        # The most significant echo inside 10 nmi, by class id (higher = more significant).
        worst = int(cbox.max()) if cbox.size else 0
        out[name] = {
            "status": int(status[j, i]),
            "red": [k for k in RULE_KEYS if red_rules[k][j, i]],
            "yellow": [k for k in RULE_KEYS if yel_rules[k][j, i] and not red_rules[k][j, i]],
            "class_here": names[int(cls[j, i])],
            "class_10nm": names[worst] if worst else None,
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


def write_data(stem, status, cls, top, F, red_rules, yel_rules):
    """Per-cell readout for the viewer: six uint8 planes, north-up, row-major.

        0 status + 1        (0 no coverage, 1 clear, 2 watch, 3 violating)
        1 cloud class id
        2 composite dBZ * 2, 255 = no echo
        3 echo-top level    (index into TOP_LEVELS)
        4 red rule bits     (bit n = RULE_KEYS[n])
        5 yellow rule bits
    """
    dbz = np.where(F["comp"] >= 0, np.clip(np.round(F["comp"] * 2), 0, 254), 255)
    rbits = np.zeros(status.shape, np.uint8)
    ybits = np.zeros(status.shape, np.uint8)
    for b, k in enumerate(RULE_KEYS):
        rbits |= (red_rules[k].astype(np.uint8) << b)
        ybits |= ((yel_rules[k] & ~red_rules[k]).astype(np.uint8) << b)
    planes = [(status + 1).astype(np.uint8), cls.astype(np.uint8), dbz.astype(np.uint8),
              top.astype(np.uint8), rbits, ybits]
    rel = f"{stem}.bin"
    with open(os.path.join(OUT_DIR, rel), "wb") as fp:
        fp.write(b"".join(p.tobytes() for p in planes))
    return rel


# --------------------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------------------
def build_frame(sess, sources, z0):
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

    status, red_rules, yel_rules, cls, top, diag = evaluate(F, z0, la, lo)
    pads = pad_report(status, red_rules, yel_rules, cls, top, diag, F, la, lo)

    valid = valid_times.get("comp") or max((v for v in valid_times.values() if v), default=None)
    stamp = valid.replace(":", "").replace("-", "")[:13]
    stem = f"frames/{stamp}"
    images = render_layers(status, cls, F, diag, la, lo, stem)
    data = write_data(stem, status, cls, top, F, red_rules, yel_rules)

    vt = [datetime.datetime.strptime(v, "%Y-%m-%dT%H:%M:%SZ")
          for v in valid_times.values() if v]
    skew = round((max(vt) - min(vt)).total_seconds() / 60.0, 1) if len(vt) > 1 else 0.0
    counts = {str(k): int((status == k).sum()) for k in (-1, 0, 1, 2)}
    cls_counts = {c["key"]: int((cls == c["id"]).sum()) for c in CLASSES}
    frame = {"valid": valid, "stamp": stamp, "images": images, "data": data,
             "pads": pads, "counts": counts, "class_counts": cls_counts,
             "missing": missing, "skew_min": skew}
    red_pads = [n for n, p in pads.items() if p["status"] == 2]
    yel_pads = [n for n, p in pads.items() if p["status"] == 1]
    logging.info(f"frame {valid}: pads violating {red_pads or '-'} watch {yel_pads or '-'}; "
                 f"skew {skew} min" + (f"; missing {missing}" if missing else ""))
    return frame, la, lo


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

    newest, la, lo = build_frame(sess, {k: latest_url(p) for k, (p, _, _) in PRODUCTS.items()},
                                 z0)
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
            frame, _, _ = build_frame(sess, src, z0)
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
    for fn in os.listdir(frame_dir):
        if fn not in keep:
            os.remove(os.path.join(frame_dir, fn))

    manifest = {
        "generated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "viewer_expected": VIEWER_VERSION_EXPECTED,
        "valid": newest["valid"], "freezing": fmeta, "domain": DOMAIN,
        "grid": {"ny": int(la.size), "nx": int(lo.size), "lat_n": float(la[0]),
                 "lon_w": float(lo[0]), "dlat": abs(float(la[0] - la[1])),
                 "dlon": abs(float(lo[1] - lo[0]))},
        "sites": {k: list(v) for k, v in SITES.items()},
        "frames": frames,
        "rule_keys": RULE_KEYS, "rules": RULE_NAMES, "not_evaluated": NOT_EVALUATED,
        "classes": CLASSES, "top_levels": TOP_LEVELS,
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
