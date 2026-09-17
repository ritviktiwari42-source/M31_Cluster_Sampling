######################################################
import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import DBSCAN, HDBSCAN

######################################################

RAo  = 10.07304
DECo = 40.6551
dRA  = 0.01
dDEC = dRA * np.cos(DECo/180.*3.141592)

RAo  = 10.07145
DECo = 40.65375
cos_DECo = np.cos(DECo/180.*3.141592)
dRA  = 0.005 / cos_DECo
dDEC = dRA

print(RAo,DECo)
print(dRA,dDEC)
# exit()

######################################################

print('Read PHAST subset:')
Name = os.path.expanduser('/home/ritvik-tiwari/Documents/ESO_Work/Bol_321_cluster/Tile_000111_10.061839_40.464915_00020443.fits')
hdul = fits.open(Name)
fits.info(Name)
# phast_table = hdul[1].data[:1000]
phast_table = hdul[1].data
print(np.shape(phast_table))
# cols = hdul[1].columns; cols.info()
RA0 = phast_table['ra']
DEC0 = phast_table['dec']
Cond = ((RAo-dRA)<=RA0) & (RA0<=(RAo+dRA)) & ((DECo-dDEC)<=DEC0) & (DEC0<=(DECo+dDEC))


# --- QUALITY CUT: remove stars with unreliable photometry ---
# 'opt_gst' is a pre-computed True/False flag from the PHAT/PHAST
# pipeline: True means this star's F475W and F814W measurements both
# passed their signal-to-noise, sharpness, and crowding checks. Stars
# that fail this (False) often carry placeholder/garbage values in one
# or both filters (e.g. color or mag around +/-75 or ~100), which
# silently wreck downstream steps -- CMD axis scaling, normalization,
# luminosity function bins, etc. Folding this into Cond here means
# every array built below (RA, DEC, X, Y, and anything derived from
# them later) is automatically clean, with no extra filtering needed
# in any later script.
Cond = Cond & (phast_table['opt_gst'] == True)
RA = RA0[Cond]
DEC = DEC0[Cond]
print(len(RA))
X = (RA - RAo) * cos_DECo * 60.0
Y = (DEC - DECo) * 60.0

######################################################
print('Read Zhang+ 2025:')
Name_Zh25 = os.path.expanduser(
    '~/Documents/ESO_Work/Phast_Subset_1/Zhang_etal_2025/J_ApJS_278_16_table7.dat.fits'
)
hdul_zh = fits.open(Name_Zh25)
fits.info(Name_Zh25)
zh_table = hdul_zh[1].data
print(np.shape(zh_table))

RA_Zh25 = (zh_table['RAh']+zh_table['RAm']/60.+zh_table['RAs']/3600.)*15.
sign = np.where(zh_table['DE-'] == '-', -1, 1)
DEC_Zh25 = sign * (zh_table['DEd']+zh_table['DEm']/60.+zh_table['DEs']/3600.)

X_Zh25 = (RA_Zh25 - RAo) * cos_DECo * 60.0
Y_Zh25 = (DEC_Zh25 - DECo) * 60.0

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

WHERE TO PUT THIS:
Paste this code into your main script, right after your DBSCAN block
(after `labels = db.labels_`), OR run it as a separate script provided
you still have X, Y, labels, and phast_table/Cond available (e.g. by
importing/re-running the earlier parts first).
"""

import os
import numpy as np
from astropy.table import Table

# ---------------------------------------------------------------
# Build color and magnitude arrays for the same stars in X, Y
# (i.e. using the same `Cond` spatial mask as your PHAST subset)
# ---------------------------------------------------------------
# Optional but recommended: apply the PHAT/PHAST quality flag first,
# so unreliable detections don't pollute your CMD.
# quality_cut = phast_table['opt_gst'][Cond] == True
#
# If you use the quality cut, apply it consistently to RA, DEC, X, Y,
# color, mag, and labels together -- easiest is to fold it into Cond
# itself back in the main script:
#     Cond = Cond & (phast_table['opt_gst'] == True)
# and then just rerun the RA/DEC/X/Y block. That keeps everything
# aligned by construction. The lines below assume you did that.

mag = phast_table['f814w_vega'][Cond]
color = phast_table['f475w_vega'][Cond] - phast_table['f814w_vega'][Cond]

# ---------------------------------------------------------------
# Sanity check: everything must be the same length before export
# ---------------------------------------------------------------
lengths = {'RA': len(RA), 'DEC': len(DEC), 'X': len(X), 'Y': len(Y),
           'mag': len(mag), 'color': len(color), 'labels': len(labels)}
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
    'cluster_label': labels,
})

out_path = os.path.expanduser('~/Documents/ESO_Work/dbscan_export_for_topcat.fits')
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
    # plt.plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col), markeredgecolor="k", markersize=8,)
    plt.plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col), markersize=8,)
    # plt.plot(xy[:, 0], xy[:, 1], "o", markerfacecolor='r', markeredgecolor='r', markersize=8,)

    xy = XY[class_member_mask & ~core_samples_mask]
    # plt.plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col), markeredgecolor="k", markersize=4,)
    # plt.plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col), markersize=2,)
    # plt.plot(xy[:, 0], xy[:, 1], "o", markerfacecolor='b', markeredgecolor='b', markersize=1,)

plt.title(f"Estimated number of clusters: {n_clusters_}")


######################################################



plt.tight_layout()
plt.savefig('M31_clusterfinder_101.png')
plt.show()


