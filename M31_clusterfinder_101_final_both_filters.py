######################################################
import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import DBSCAN, HDBSCAN

######################################################

import sys

print('Read tile catalog:')
Name = input(
    "Enter the path to your tile FITS file: "
).strip().strip('"').strip("'")
Name = os.path.expanduser(Name)
if not os.path.isfile(Name):
    print(f"File not found: {Name}")
    sys.exit(1)
hdul = fits.open(Name)
fits.info(Name)
phast_table = hdul[1].data
print(np.shape(phast_table))
# cols = hdul[1].columns; cols.info()

# --- COLUMN NAMES: this tile file uses uppercase 'RA'/'DEC', unlike
# the earlier phast_subset.fits which used lowercase 'ra'/'dec'. ---
RA0 = phast_table['RA']
DEC0 = phast_table['DEC']

# --- SEARCH BOX CENTER: auto-derived from this file's own coverage.
# The old hardcoded RAo/DECo (from the original M31-south phast_subset
# work) do NOT overlap this tile at all -- this file covers
# RA ~10.054-10.070, Dec ~40.459-40.471, a completely different patch
# of sky. Since this tile is already a small, manageable cutout (not
# the full multi-million-star survey), we just center the box on the
# file's own middle and size it to comfortably include every star in
# the file (5% margin), rather than picking an arbitrary small window.
ra_span = RA0.max() - RA0.min()
dec_span = DEC0.max() - DEC0.min()
RAo = (RA0.max() + RA0.min()) / 2.0
DECo = (DEC0.max() + DEC0.min()) / 2.0
cos_DECo = np.cos(DECo / 180. * 3.141592)
dRA = (ra_span / 2.0) * 1.05
dDEC = (dec_span / 2.0) * 1.05

print(RAo, DECo)
print(dRA, dDEC)

######################################################

Cond = ((RAo-dRA)<=RA0) & (RA0<=(RAo+dRA)) & ((DECo-dDEC)<=DEC0) & (DEC0<=(DECo+dDEC))

# --- QUALITY CUT REMOVED for this file ---
# The earlier phast_subset.fits had a pre-computed 'opt_gst' quality
# flag column (good/bad photometry per star) that we folded into Cond.
# This tile file does NOT have that column at all (its only columns
# are RA, DEC, f475w_vega, f475w_err, f814w_vega, f814w_err), so that
# check has been removed here -- keeping it would crash with a
# KeyError on this file.
#
# Note: this file still has the same kind of placeholder bad-
# photometry values as before (both f475w_vega and f814w_vega go up
# to 99.999, which are not real magnitudes) -- there's just no
# pre-built flag to filter them out this time. If you want to guard
# against that manually, a simple sanity cut using the per-star error
# columns (f475w_err, f814w_err) or a plain magnitude ceiling would be
# the equivalent replacement, e.g.:
#     Cond = Cond & (phast_table['f814w_vega'] < 40) & (phast_table['f475w_vega'] < 40)

RA = RA0[Cond]
DEC = DEC0[Cond]
print(len(RA))
X = (RA - RAo) * cos_DECo * 60.0
Y = (DEC - DECo) * 60.0

######################################################
print('Read Zhang+ 2025:')
# NOTE: this reference catalog was built around the original M31-south
# phast_subset field (RA~10.07, Dec~40.65). This new tile is centered
# much further south (Dec~40.46), so it's unlikely any Zhang+2025
# clusters actually fall inside this tile's search box -- the overlay
# below may end up empty for this file. Left in place in case you
# still want the comparison; remove this block if it's not relevant
# for this particular tile.
# Optional: press Enter to skip. These southern tiles generally have no
# Zhang+2025 clusters in them, and a missing/incorrect path used to abort
# the entire run before any output was written.
Name_Zh25 = input(
    "Path to the Zhang+2025 catalog (press Enter to skip the overlay): "
).strip().strip('"').strip("'")
Name_Zh25 = os.path.expanduser(Name_Zh25) if Name_Zh25 else ''

if Name_Zh25 and os.path.isfile(Name_Zh25):
    hdul_zh = fits.open(Name_Zh25)
    fits.info(Name_Zh25)
    zh_table = hdul_zh[1].data
    print(np.shape(zh_table))
    RA_Zh25 = (zh_table['RAh']+zh_table['RAm']/60.+zh_table['RAs']/3600.)*15.
    sign = np.where(zh_table['DE-'] == '-', -1, 1)
    DEC_Zh25 = sign * (zh_table['DEd']+zh_table['DEm']/60.+zh_table['DEs']/3600.)
    X_Zh25 = (RA_Zh25 - RAo) * cos_DECo * 60.0
    Y_Zh25 = (DEC_Zh25 - DECo) * 60.0
else:
    if Name_Zh25:
        print(f"  Not found: {Name_Zh25} -- skipping the overlay.")
    RA_Zh25 = np.array([]); DEC_Zh25 = np.array([])
    X_Zh25 = np.array([]); Y_Zh25 = np.array([])

######################################################

fig = plt.figure(figsize=(16,8))
plt.subplot(121)
plt.xlabel('RA, deg'); plt.ylabel('Dec, deg')
plt.xlim(RAo+dRA, RAo-dRA); plt.ylim(DECo-dDEC, DECo+dDEC)
plt.plot(RA, DEC, '.', markersize=2, zorder=99)
plt.scatter(RA_Zh25, DEC_Zh25, s=400, edgecolor='red', facecolor='none')

plt.subplot(122)
plt.xlabel(r'$\Delta$RA, arcmin'); plt.ylabel(r'$\Delta$Dec, arcmin')
plt.xlim(np.max(X), np.min(X));
plt.ylim(np.min(Y), np.max(Y))
plt.plot(X, Y, '.', markersize=2, zorder=1)
plt.scatter(X_Zh25, Y_Zh25, s=400, edgecolor='red', facecolor='none', zorder=99)

######################################################

XY = np.vstack((X,Y)).transpose()
# print(np.shape(X)); print(np.shape(Y)); print(np.shape(XY))

# NOTE: this file has ~20,000 stars total (vs. the ~36,000-star window
# used to tune these EPS/MIN_SAMPLE values before) -- you will likely
# need to re-tune these for this file's star density, the same way the
# original commented-out trial-and-error was done.
# EPS, MIN_SAMPLE = 0.01,10   # No of clusters = 1
# EPS, MIN_SAMPLE = 0.005,10  # No of clusters = 744
EPS, MIN_SAMPLE = 0.004,10  # No of clusters = 45
# EPS, MIN_SAMPLE = 0.004,9  # No of clusters = 201
# EPS, MIN_SAMPLE = 0.0035,10  # No of clusters = 8
EPS, MIN_SAMPLE = 0.0037,10  # No of clusters = 24
EPS, MIN_SAMPLE = 0.0045,11  # No of clusters = 122
EPS, MIN_SAMPLE = 0.0045,13  # No of clusters = 13
EPS, MIN_SAMPLE = 0.0045,12  # No of clusters = 38
EPS, MIN_SAMPLE = 0.0055,15  # No of clusters = 61
EPS, MIN_SAMPLE = 0.0065,20  # No of clusters = 32
EPS, MIN_SAMPLE = 0.008,30  # No of clusters = 10  <====

# search
db = DBSCAN(eps=EPS, min_samples=MIN_SAMPLE).fit(XY)
labels = db.labels_

"""
export_for_topcat.py

Appends to your existing M31_clusterfinder_101.py, AFTER the DBSCAN
block (i.e. after `labels = db.labels_` exists). It builds one table
containing everything you need to explore your clusters in TOPCAT:

    ra, dec, X, Y, color, mag, cluster_label

...and writes it out as a FITS file you can load directly into TOPCAT.
"""

import os
import numpy as np
from astropy.table import Table

# ---------------------------------------------------------------
# Build color and magnitude arrays for the same stars in X, Y
# (i.e. using the same `Cond` spatial mask as your tile subset)
# ---------------------------------------------------------------
mag = phast_table['f814w_vega'][Cond]
color = phast_table['f475w_vega'][Cond] - phast_table['f814w_vega'][Cond]

# --- Individual filter magnitudes AND their errors, kept separately
# alongside the existing 'mag'/'color' columns. 'mag' above stays
# F814W-based for backward compatibility with any script that expects
# a single generic 'mag' column; these four give direct access to
# both filters plus their measurement uncertainties. ---
mag_F814W = phast_table['f814w_vega'][Cond]
mag_F814W_err = phast_table['f814w_err'][Cond]
mag_F475W = phast_table['f475w_vega'][Cond]
mag_F475W_err = phast_table['f475w_err'][Cond]

# ---------------------------------------------------------------
# Sanity check: everything must be the same length before export
# ---------------------------------------------------------------
lengths = {'RA': len(RA), 'DEC': len(DEC), 'X': len(X), 'Y': len(Y),
           'mag': len(mag), 'color': len(color), 'labels': len(labels),
           'mag_F814W': len(mag_F814W), 'mag_F814W_err': len(mag_F814W_err),
           'mag_F475W': len(mag_F475W), 'mag_F475W_err': len(mag_F475W_err)}
if len(set(lengths.values())) != 1:
    raise ValueError(f"Array length mismatch, check your Cond mask "
                      f"was applied consistently: {lengths}")

# ---------------------------------------------------------------
# Build and write the table
# ---------------------------------------------------------------
export_table = Table({
    'ra': RA,
    'dec': DEC,
    'X': X,
    'Y': Y,
    'color': color,
    'mag': mag,
    'mag_F814W': mag_F814W,
    'mag_F814W_err': mag_F814W_err,
    'mag_F475W': mag_F475W,
    'mag_F475W_err': mag_F475W_err,
    'cluster_label': labels,
})

out_path = input(
    "\nWhere should the DBSCAN export be written? "
    "[default: DBSCAN.fits next to the tile file]: "
).strip().strip('"').strip("'")
if out_path:
    out_path = os.path.expanduser(out_path)
else:
    out_path = os.path.join(os.path.dirname(os.path.abspath(Name)),
                             'DBSCAN.fits')

# --- Carry the tile reference point (RAo/DECo) in the FITS header. ---
# X and Y are arcmin offsets FROM this point:
#     X = (ra  - RAo) * cos(DECo) * 60
#     Y = (dec - DECo)            * 60
# Without RAo/DECo stored somewhere, every later script has to
# reverse-engineer them to get back to real sky coordinates. Saving
# them here (and passing them down the chain) removes that problem.
export_table.meta['RAO'] = float(RAo)
export_table.meta['DECO'] = float(DECo)
export_table.meta['COSDECO'] = float(cos_DECo)
export_table.meta['TILEFILE'] = os.path.basename(Name)

export_table.write(out_path, overwrite=True)
print(f"Wrote {len(export_table)} rows to {out_path}")
print("Columns:", export_table.colnames)
print("Cluster label counts:")
unique, counts = np.unique(labels, return_counts=True)
for u, c in zip(unique, counts):
    tag = "noise" if u == -1 else f"cluster {u}"
    print(f"  {tag}: {c} stars")

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
print(f"Silhouette Coefficient: {metrics.silhouette_score(XY, labels):.3f}")

unique_labels = set(labels)
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = XY[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col), markersize=8,)

    xy = XY[class_member_mask & ~core_samples_mask]

plt.title(f"Estimated number of clusters: {n_clusters_}")

######################################################

plt.tight_layout()
plt.savefig('M31_clusterfinder_101.png')
plt.show()
