#!/usr/bin/env python3
"""
MRMS LLCC nowcast - a radar traffic light for the Cape.

CloudScope classifies FORECAST cloud from model condensate and turns it into a probability.
This does the observational counterpart: it reads the latest MRMS radar and lightning,
applies the NASA-STD-4010 Lightning Launch Commit Criteria that a radar can adjudicate, and
paints every grid cell as if the flight path ran through it:

    RED     a rule is violated
    YELLOW  no rule violated, but one would be if its standoff were 2 nmi longer, or
            there is cloud-to-ground lightning within 20 nmi
    GREEN   neither

THE 0 dBZ GATE
    Every rule is assessed against echo >= 0 dBZ. Where MRMS shows nothing at or above 0 dBZ,
    no rule applies. That is not a radar limitation being papered over: NASA-STD-4010 defines
    a non-transparent cloud by radar return, so the radar is the reference the rules are
    written against rather than a proxy for them.

WHAT THE PROBES ESTABLISHED (6 Oct 2026)
    - Every product sits on one 3500x7000, 0.01 deg CONUS grid. No regridding anywhere.
    - Sentinels differ by product: reflectivity marks "no echo" with -99, while VII, echo
      tops, composite height and lightning use -1. One blanket rule reads an echo top of -1
      as a height of -1 km, so each product names its own.
    - The GRIB2 carries no names or units (MRMS uses local tables pygrib does not have), so
      units are set here: dBZ, kg/m2, echo top in KILOMETRES, composite height in METRES.
    - The 0 to -20 C isotherm slices have no vertical gaps, and on a convective afternoon
      they see ~99% of echo. Elevated-only anvil is rare, and the layer composites behave as
      expected on it: Low sees warm echo, Super sees echo aloft.
    - Model_0degC_Height is 8 MB, dense, and hourly. It is fetched once an hour and cached.

CONSERVATIVE ANVIL BASE
    Radar cannot tell an anvil cloud from the precipitation falling out of it - both return
    echo at the 0 C isotherm. On the probe day 95% of echo reached the 0 C slice. So any echo
    at 0 C beneath an anvil counts as the anvil reaching 0 C, and LLCCR 18's "entirely
    colder than 0 C" exception is rarely granted. That is the agreed choice, not an accident.

NOT EVALUATED
    Surface electric fields (4.1.2) and their exceptions; debris clouds (4.1.6), which need an
    observed detachment time; smoke plumes (4.1.9); triboelectrification (4.1.10); the
    3-hour lightning clocks on the anvil rules, since only 30 minutes of lightning is
    available here. Green means no radar-visible violation, never GO.
"""

import datetime
import gzip
import json
import logging
import os
import tempfile

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
UA = {"User-Agent": "CloudScope-MRMS/1.0 (launch weather nowcast)"}
OUT_DIR = os.environ.get("OUT_DIR", "site")
FRAME_DIR = os.path.join(OUT_DIR, "frames")
STATE_DIR = os.path.join(OUT_DIR, "state")

VIEWER_VERSION_EXPECTED = "mrms-v1"

# Big enough to hold a 10 nmi standoff around every pad plus context to see weather coming.
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

FRAMES_KEPT = 12          # one hour at 5-minute cadence
FREEZING_MAX_AGE_MIN = 70  # the 0 C height is hourly; refetch once it is older than this

# (product, no-echo sentinel, units). The sentinel is per product - see the module notes.
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
FREEZING = ("Model_0degC_Height", None, "m")

# Products whose absence would make a frame read greener than reality. Composite reflectivity
# is the 0 dBZ gate itself; the 0, -10 and -20 C slices carry the cumulus and anvil rules;
# lightning carries 4.1.1. Without any of these the frame is withheld - see main().
ESSENTIAL = ("comp", "r0", "r10", "r20", "cg")

# --------------------------------------------------------------------------------------
# LLCC thresholds - NASA-STD-4010 (2017-06-27), requirement numbers in the comments
# --------------------------------------------------------------------------------------
LLCC = {
    "lightning_nm": 10.0,          # 4.1.1, within 10 nmi in the last 30 min
    "lightning_watch_nm": 20.0,    # yellow only - activity approaching
    "cumulus_5nm": 5.0,            # LLCCR 16: top colder than -10 C within 5 nmi
    "cumulus_10nm": 10.0,          # LLCCR 17: top colder than -20 C within 10 nmi
    "plus5_c": 5.0,                # LLCCR 15: flight through cumulus topping at <= +5 C
    "attached_3nm": 3.0,           # LLCCR 18
    "attached_lightning_nm": 10.0, # LLCCR 19/20, applied only while lightning is active
    "detached_3nm": 3.0,           # LLCCR 22
    "excep_nm": 5.0,               # exception: anvil within 5 nmi entirely colder than 0 C
    "mrr_dbz": 7.5,                # exception: MRR < +7.5 dBZ within 1 nmi
    "mrr_search_nm": 4.0,          # 4.2.2c: largest composite within 4 nmi
    "mrr_eval_nm": 1.0,
    "disturbed_nm": 5.0,           # 4.1.7
    "disturbed_dbz": 30.0,
    "core_dbz": 40.0,              # a convective core, for anvil connectivity
    "yellow_margin_nm": 2.0,       # how close to a standoff counts as "look closer"
}
LAPSE_C_PER_KM = 6.5               # to place +5 C and -20 C from the measured 0 C height

RULE_NAMES = {
    "lightning":       "Lightning within 10 nmi (4.1.1)",
    "cumulus_through": "Flight through cumulus, top <= +5 C (4.1.3.1)",
    "cumulus_5nm":     "Cumulus to -10 C within 5 nmi (4.1.3.2)",
    "cumulus_10nm":    "Cumulus to -20 C within 10 nmi (4.1.3.3)",
    "attached_anvil":  "Attached anvil within 3 nmi (4.1.4)",
    "detached_anvil":  "Detached anvil within 3 nmi (4.1.5)",
    "disturbed":       "Disturbed weather (4.1.7)",
    "thick_layer":     "Thick cloud layer 0 to -10 C (4.1.8)",
}
NOT_EVALUATED = [
    "Surface electric fields (4.1.2) and the field-mill exceptions",
    "Debris clouds (4.1.6) - need an observed detachment time",
    "Smoke plumes (4.1.9)",
    "Triboelectrification (4.1.10)",
    "The 3-hour lightning clocks on the anvil rules - only 30 min of lightning is published",
]

STATUS_COLORS = {-1: "#B9BEC4", 0: "#3E9B5F", 1: "#E3B23C", 2: "#C0392B"}
BG = "#FFFFFF"


# --------------------------------------------------------------------------------------
# Fetch and decode
# --------------------------------------------------------------------------------------
def _session():
    s = requests.Session()
    s.mount("https://", requests.adapters.HTTPAdapter(max_retries=3))
    s.headers.update(UA)
    return s


def fetch_raw(sess, product):
    """The .latest file for a product, gunzipped. MRMS publishes a stable .latest name for
    every product, so there is never a directory to list."""
    url = f"{ROOT}/{product}/MRMS_{product}.latest.grib2.gz"
    r = sess.get(url, timeout=90)
    r.raise_for_status()
    return gzip.decompress(r.content)


def decode_crop(raw, fill):
    """Decode one MRMS GRIB2 and cut out DOMAIN by index arithmetic.

    Calling latlons() on a 3500x7000 grid builds two 24.5-million-element arrays to read a
    40,000-cell box. The grid definition already says where every cell is, so the crop is
    computed from the first grid point and the increments instead, and latlons() is only a
    fallback if those keys are ever missing.
    Returns (field, lat_axis, lon_axis, valid_iso).
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
        raise ValueError(f"domain not on this grid: lat {lat_axis.min():.2f}..{lat_axis.max():.2f}, "
                         f"lon {lon_axis.min():.2f}..{lon_axis.max():.2f}")
    sub = vals[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    la = lat_axis[rows.min():rows.max() + 1]
    lo = lon_axis[cols.min():cols.max() + 1]
    # Store north-up regardless of how the file scans, so every field lines up.
    if la[0] < la[-1]:
        sub, la = sub[::-1], la[::-1]
    return sub, la, lo, valid


def fetch_product(sess, key):
    product, sentinel, _ = PRODUCTS[key]
    raw = fetch_raw(sess, product)
    return decode_crop(raw, sentinel)


def freezing_level(sess, prev):
    """The 0 C height, refetched only when the cached copy is over an hour old.

    It is 8 MB and hourly, against ~0.6 MB and two-minutely for everything else, so pulling
    it every five minutes would be most of this job's bandwidth for no new information.
    """
    path = os.path.join(STATE_DIR, "z0.npy")
    stamp = (prev.get("freezing") or {}).get("fetched")
    if stamp and os.path.exists(path):
        age = (datetime.datetime.now(datetime.timezone.utc)
               - datetime.datetime.fromisoformat(stamp)).total_seconds() / 60.0
        if age < FREEZING_MAX_AGE_MIN:
            return np.load(path), prev["freezing"], False
    raw = fetch_raw(sess, FREEZING[0])
    z0, _, _, valid = decode_crop(raw, None)
    os.makedirs(STATE_DIR, exist_ok=True)
    np.save(path, z0.astype(np.float32))
    meta = {"fetched": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "valid": valid}
    return z0, meta, True


# --------------------------------------------------------------------------------------
# LLCC evaluation
# --------------------------------------------------------------------------------------
def _disc(nm, dlat_km, dlon_km):
    km = nm * 1.852
    rj = max(1, int(np.ceil(km / dlat_km)))
    ri = max(1, int(np.ceil(km / dlon_km)))
    jj, ii = np.mgrid[-rj:rj + 1, -ri:ri + 1]
    return np.hypot(jj * dlat_km, ii * dlon_km) <= km


def evaluate(F, z0, la, lo):
    """Status grid (-1 no data, 0 green, 1 yellow, 2 red) and the per-rule grids behind it."""
    L = LLCC
    lat_mid = float(np.mean(la))
    dlat_km = abs(float(la[0] - la[1])) * 111.32
    dlon_km = abs(float(lo[1] - lo[0])) * 111.32 * np.cos(np.radians(lat_mid))

    def near(mask, nm):
        if nm <= 0:
            return mask.copy()
        return maximum_filter(mask.astype(np.uint8),
                              footprint=_disc(nm, dlat_km, dlon_km)) > 0

    def peak(field, nm):
        return maximum_filter(field, footprint=_disc(nm, dlat_km, dlon_km))

    ge0 = lambda k: F[k] >= 0.0                       # the 0 dBZ gate, per isotherm
    comp = F["comp"]
    echo = comp >= 0.0
    nodata = comp <= -900.0                           # outside radar coverage, not "clear"

    e0, e5, e10, e15, e20 = ge0("r0"), ge0("r5"), ge0("r10"), ge0("r15"), ge0("r20")
    any_iso = e0 | e5 | e10 | e15 | e20
    hmax = F["hmax"]                                  # METRES, -1 for no echo

    z5 = z0 - L["plus5_c"] / LAPSE_C_PER_KM * 1000.0
    z20 = z0 + 20.0 / LAPSE_C_PER_KM * 1000.0

    # Cloud top colder than 0 C: echo at any isotherm slice, or echo aloft that the slices
    # cannot see (an elevated shield whose column max sits above -20 C).
    aloft = echo & ~any_iso & (hmax >= z20)
    top_below_0c = any_iso | aloft | (F["super"] >= 0.0)
    # Reaching +5 C (LLCCR 15) needs echo up to ~770 m below the freezing level.
    top_to_plus5 = any_iso | (echo & (hmax >= z5))

    # Convective core, with VII as corroboration - the probe showed VII only registers in the
    # strongest cells, which makes it a core indicator rather than an anvil detector.
    core = (comp >= L["core_dbz"]) | (F["vii"] > 0.0)

    # Anvil: echo at or above -20 C, or in the Super layer, that is not itself a core.
    anvil = ((e20 | (F["super"] >= 0.0) | aloft) & ~core) & echo
    lab, n = label(anvil | core, structure=np.ones((3, 3)))
    has_core = np.zeros(n + 1, bool)
    if n:
        has_core[np.unique(lab[core])] = True
        has_core[0] = False
    attached = anvil & has_core[lab]
    detached = anvil & ~has_core[lab]

    # LLCCR 18 exception, CONSERVATIVE: the anvil within 5 nmi must be entirely colder than
    # 0 C, and radar cannot separate anvil from the precipitation it drops. So any echo at the
    # 0 C slice under the anvil fails the exception.
    anvil_warm = near((attached | detached) & e0, L["excep_nm"])
    mrr = peak(np.where(comp > -90, comp, -99.0).astype(np.float32), L["mrr_search_nm"])
    mrr_ok = peak(mrr, L["mrr_eval_nm"]) < L["mrr_dbz"]
    exception = ~anvil_warm & mrr_ok

    lightning = F["cg"] > 0.0

    def rules(m):
        """Every rule at its standoff plus margin m. m=0 is red; m>0 is the yellow test."""
        lit = near(lightning, L["lightning_nm"] + m)
        return {
            "lightning":       lit,
            "cumulus_through": near(echo & top_to_plus5, m),
            "cumulus_5nm":     near(e10, L["cumulus_5nm"] + m),
            "cumulus_10nm":    near(e20, L["cumulus_10nm"] + m),
            "attached_anvil":  (near(attached, L["attached_3nm"] + m) & ~exception)
                               | (near(attached, L["attached_lightning_nm"] + m) & lit),
            "detached_anvil":  near(detached, L["detached_3nm"] + m) & ~exception,
            "disturbed":       near(echo & top_below_0c, m)
                               & near(comp >= L["disturbed_dbz"], L["disturbed_nm"] + m),
            "thick_layer":     near(e0 & e10 & ~(attached | detached), m),
        }

    red_rules = rules(0.0)
    yel_rules = rules(L["yellow_margin_nm"])
    red = np.logical_or.reduce(list(red_rules.values()))
    yellow = (np.logical_or.reduce(list(yel_rules.values()))
              | near(lightning, L["lightning_watch_nm"])) & ~red

    status = np.zeros(comp.shape, np.int8)
    status[yellow] = 1
    status[red] = 2
    status[nodata] = -1

    dist_km = distance_transform_edt(~echo, sampling=(dlat_km, dlon_km))
    diag = {"attached": attached, "detached": detached, "core": core,
            "echo": echo, "dist_echo_nm": dist_km / 1.852, "lightning": lightning}
    return status, red_rules, yel_rules, diag


def pad_report(status, red_rules, yel_rules, diag, F, la, lo):
    out = {}
    lat_mid = float(np.mean(la))
    dlat_km = abs(float(la[0] - la[1])) * 111.32
    dlon_km = abs(float(lo[1] - lo[0])) * 111.32 * np.cos(np.radians(lat_mid))
    win = _disc(10.0, dlat_km, dlon_km)
    hw = win.shape[0] // 2, win.shape[1] // 2
    for name, (plat, plon) in SITES.items():
        j = int(np.argmin(np.abs(la - plat)))
        i = int(np.argmin(np.abs(lo - plon)))
        s = int(status[j, i])
        j0, j1 = max(0, j - hw[0]), min(la.size, j + hw[0] + 1)
        i0, i1 = max(0, i - hw[1]), min(lo.size, i + hw[1] + 1)
        box = F["comp"][j0:j1, i0:i1]
        dbz10 = float(box.max()) if box.size and box.max() > -90 else None
        out[name] = {
            "status": s,
            "red": [k for k, v in red_rules.items() if v[j, i]],
            "yellow": [k for k, v in yel_rules.items() if v[j, i] and not red_rules[k][j, i]],
            "max_dbz_10nm": None if dbz10 is None or dbz10 < 0 else round(dbz10, 1),
            "nearest_echo_nm": round(float(diag["dist_echo_nm"][j, i]), 1),
            "lightning_10nm": bool(red_rules["lightning"][j, i]),
        }
    return out


# --------------------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------------------
def render(status, F, diag, la, lo, valid, path):
    pc = ccrs.PlateCarree()
    proj = ccrs.Mercator(central_longitude=0.5 * (DOMAIN["lon_min"] + DOMAIN["lon_max"]))
    x0, y0 = proj.transform_point(DOMAIN["lon_min"], DOMAIN["lat_min"], pc)
    x1, y1 = proj.transform_point(DOMAIN["lon_max"], DOMAIN["lat_max"], pc)
    h_in = 6.8
    fig = plt.figure(figsize=(h_in * (x1 - x0) / (y1 - y0), h_in), dpi=140, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1], projection=proj)
    ax.set_extent([DOMAIN["lon_min"], DOMAIN["lon_max"],
                   DOMAIN["lat_min"], DOMAIN["lat_max"]], crs=pc)
    ax.set_facecolor(BG)

    keys = [-1, 0, 1, 2]
    cmap = mcolors.ListedColormap([STATUS_COLORS[k] for k in keys])
    norm = mcolors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], len(keys))
    LO, LA = np.meshgrid(lo, la)
    ax.pcolormesh(LO, LA, status, cmap=cmap, norm=norm, shading="nearest",
                  transform=pc, zorder=2, alpha=0.62)

    # Reflectivity contours on top, so the cores driving the colours stay visible.
    comp = np.where(F["comp"] > -90, F["comp"], np.nan)
    try:
        ax.contour(LO, LA, comp, levels=[30, 40, 50], colors=["#5A5A5A", "#2B2B2B", "#000000"],
                   linewidths=[0.5, 0.8, 1.1], transform=pc, zorder=3)
    except Exception:
        pass
    if diag["lightning"].any():
        jj, ii = np.where(diag["lightning"])
        ax.scatter(lo[ii], la[jj], marker="x", s=7, linewidths=0.8, color="#000000",
                   transform=pc, zorder=5)

    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="#2E3A44",
                   linewidth=0.9, zorder=4)
    for name, (plat, plon) in SITES.items():
        ax.plot(plon, plat, marker="+", markersize=6, markeredgewidth=1.3,
                color="#14181B", transform=pc, zorder=6)
    ax.text(0.012, 0.012, f"MRMS  valid {valid}", transform=ax.transAxes, fontsize=6.4,
            color="#5D6A6E", family="monospace", zorder=7,
            path_effects=[pe.withStroke(linewidth=2.0, foreground=BG)])
    try:
        ax.spines["geo"].set_edgecolor("#C3C8BC")
    except Exception:
        pass
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


# --------------------------------------------------------------------------------------
def load_manifest():
    try:
        with open(os.path.join(OUT_DIR, "manifest.json")) as fp:
            return json.load(fp)
    except Exception:
        return {}


def main():
    os.makedirs(FRAME_DIR, exist_ok=True)
    os.makedirs(STATE_DIR, exist_ok=True)
    sess = _session()
    prev = load_manifest()

    F, valid_times, la, lo = {}, {}, None, None
    for key in PRODUCTS:
        try:
            fld, fla, flo, v = fetch_product(sess, key)
        except Exception as e:
            # One missing product must not take the frame down. Treat it as "no echo" and
            # say so, rather than silently publishing a frame that is greener than reality.
            logging.warning(f"{PRODUCTS[key][0]}: {type(e).__name__}: {e} - treated as no echo")
            F[key] = None
            continue
        if la is None:
            la, lo = fla, flo
        elif fld.shape != (la.size, lo.size):
            logging.warning(f"{PRODUCTS[key][0]}: grid {fld.shape} differs from "
                            f"{(la.size, lo.size)}; skipped")
            F[key] = None
            continue
        F[key] = fld
        valid_times[key] = v
    if la is None:
        logging.error("No MRMS product could be read; leaving the previous frame in place.")
        return
    missing = [PRODUCTS[k][0] for k, v in F.items() if v is None]

    # Fail CLOSED on the products the rules actually stand on. Filling a missing isotherm
    # slice with "no echo" turns a 55 dBZ core green - found by testing, and exactly the
    # wrong direction for a safety tool to fail. So if any of these is absent, no frame is
    # published: the previous one stays up and the page's staleness banner fires, which is
    # honest. The others (layer composites, VII, echo top, composite height) only refine the
    # picture, so a frame without them is still conservative and still goes out, flagged.
    gone = [PRODUCTS[k][0] for k in ESSENTIAL if F.get(k) is None]
    if gone:
        logging.error(f"essential product(s) missing: {gone}. Withholding this frame rather "
                      f"than publish one that would read greener than reality.")
        return
    for k, v in F.items():
        if v is None:
            F[k] = np.full((la.size, lo.size), PRODUCTS[k][1], np.float32)

    try:
        z0, fmeta, refreshed = freezing_level(sess, prev)
        if z0.shape != (la.size, lo.size):
            raise ValueError(f"freezing grid {z0.shape} != {(la.size, lo.size)}")
    except Exception as e:
        logging.warning(f"freezing level unavailable ({e}); using 4,800 m")
        z0, fmeta, refreshed = np.full((la.size, lo.size), 4800.0, np.float32), \
            {"fetched": None, "valid": None, "fallback": True}, False

    status, red_rules, yel_rules, diag = evaluate(F, z0, la, lo)
    pads = pad_report(status, red_rules, yel_rules, diag, F, la, lo)

    valid = valid_times.get("comp") or max((v for v in valid_times.values() if v), default=None)
    stamp = (valid or datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")).replace(":", "").replace("-", "")[:13]
    png = f"frames/f_{stamp}.png"
    render(status, F, diag, la, lo, valid, os.path.join(OUT_DIR, png))

    # Ring buffer: keep the newest FRAMES_KEPT frames and drop the rest, so the published
    # branch stays a fixed size however long this runs.
    frames = [fr for fr in prev.get("frames", []) if fr.get("valid") != valid]
    counts = {str(k): int((status == k).sum()) for k in (-1, 0, 1, 2)}
    frames.insert(0, {"valid": valid, "image": png, "pads": pads, "counts": counts,
                      "missing": missing})
    frames = frames[:FRAMES_KEPT]
    keep = {os.path.basename(fr["image"]) for fr in frames}
    for fn in os.listdir(FRAME_DIR):
        if fn.endswith(".png") and fn not in keep:
            os.remove(os.path.join(FRAME_DIR, fn))

    skew = None
    vt = [datetime.datetime.strptime(v, "%Y-%m-%dT%H:%M:%SZ")
          for v in valid_times.values() if v]
    if len(vt) > 1:
        skew = round((max(vt) - min(vt)).total_seconds() / 60.0, 1)

    manifest = {
        "generated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "viewer_expected": VIEWER_VERSION_EXPECTED,
        "valid": valid, "product_valid": {PRODUCTS[k][0]: v for k, v in valid_times.items()},
        "product_skew_min": skew, "missing": missing,
        "freezing": fmeta, "domain": DOMAIN, "sites": list(SITES),
        "frames": frames,
        "rules": RULE_NAMES, "not_evaluated": NOT_EVALUATED, "thresholds": LLCC,
        "colors": {str(k): v for k, v in STATUS_COLORS.items()},
        "standard": "NASA-STD-4010 (2017-06-27)",
    }
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as fp:
        json.dump(manifest, fp, indent=1)

    red_pads = [n for n, p in pads.items() if p["status"] == 2]
    yel_pads = [n for n, p in pads.items() if p["status"] == 1]
    logging.info(f"valid {valid}: domain red {counts['2']} yellow {counts['1']} green "
                 f"{counts['0']} nodata {counts['-1']}; pads RED {red_pads or '-'} "
                 f"YELLOW {yel_pads or '-'}; skew {skew} min; freezing "
                 f"{'refreshed' if refreshed else 'cached'}"
                 + (f"; MISSING {missing}" if missing else ""))


if __name__ == "__main__":
    main()
