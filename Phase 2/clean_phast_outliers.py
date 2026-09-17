"""
clean_phast_outliers.py

STANDALONE script -- removes photometric outliers (bad/unreliable
measurements) from your PHAST star catalog, using the pre-computed
'opt_gst' quality flag column (a True/False "good star" flag from the
PHAT/PHAST team, based on signal-to-noise, sharpness, and crowding in
both F475W and F814W).

Those bad-photometry stars are the ones responsible for the extreme
outlier points we kept seeing throughout this project (e.g. color~75,
mag~100) -- placeholder values from unreliable measurements, not real
stars. Keeping only opt_gst == True removes them at the source, before
DBSCAN, background-region selection, or any CMD/cleaning step ever
sees them.

WHAT IT DOES:
  1. Reads your phast_subset.fits (or any FITS catalog with an
     'opt_gst' column).
  2. Keeps only the rows where opt_gst == True.
  3. Saves the result as a new FITS file (all original columns kept,
     just fewer rows), so every later step in your pipeline
     (M31_clusterfinder_101, export_for_topcat.py, etc.) can simply
     use this cleaned file instead of the raw one.

USAGE:
    python clean_phast_outliers.py
    (prompts for the input path -- no command-line arguments needed)
"""

import sys
import os
import numpy as np
from astropy.io import fits


def clean_phast_file(input_path, output_path=None):
    """
    Reads input_path, keeps only rows with opt_gst == True, and writes
    the result to output_path (default: <input>_cleaned.fits, saved
    next to the input file).

    Returns (output_path, n_before, n_after).
    """
    with fits.open(input_path) as hdul:
        data = hdul[1].data
        names = data.columns.names

        if 'opt_gst' not in names:
            raise ValueError(
                f"'{input_path}' has no 'opt_gst' column. "
                f"Available columns: {names}"
            )

        n_before = len(data)
        good = np.array(data['opt_gst']).astype(bool)
        cleaned_data = data[good]
        n_after = len(cleaned_data)

        if output_path is None:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_cleaned{ext}"

        # Rebuild a fresh HDUList with the same structure, filtered data
        new_hdu = fits.BinTableHDU(data=cleaned_data, header=hdul[1].header)
        new_hdul = fits.HDUList([hdul[0].copy(), new_hdu])

        if os.path.exists(output_path):
            os.remove(output_path)
        new_hdul.writeto(output_path)

    return output_path, n_before, n_after


def run_interactive():
    print("clean_phast_outliers.py -- removes bad-photometry stars "
          "using the opt_gst flag\n")

    input_path = input(
        "Enter the path to your phast_subset.fits (or similar) file: "
    ).strip().strip('"').strip("'")
    input_path = os.path.expanduser(input_path)

    if not os.path.isfile(input_path):
        print(f"File not found: {input_path}")
        sys.exit(1)

    output_input = input(
        "Output path for the cleaned file [press Enter for "
        "'<input>_cleaned.fits' next to the input file]: "
    ).strip().strip('"').strip("'")
    output_path = os.path.expanduser(output_input) if output_input else None

    print("\nReading and filtering...")
    try:
        out_path, n_before, n_after = clean_phast_file(input_path, output_path)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    n_removed = n_before - n_after
    pct_removed = (n_removed / n_before * 100) if n_before else 0

    print(f"\nBefore: {n_before} stars")
    print(f"After:  {n_after} stars")
    print(f"Removed: {n_removed} stars ({pct_removed:.1f}%)")
    print(f"\nSaved cleaned catalog to: {out_path}")
    print("\nUse this cleaned file as the input to the rest of your "
          "pipeline (M31_clusterfinder_101, export_for_topcat.py, "
          "etc.) instead of the original phast_subset.fits.")


if __name__ == "__main__":
    run_interactive()
