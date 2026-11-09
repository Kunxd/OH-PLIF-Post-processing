#!/usr/bin/env python3
"""
wabel_watershed.py — Wabel's local thresholding as described in his PhD thesis.

Reference
---------
T.M. Wabel, "An Experimental Investigation of Premixed Combustion in Extreme
Turbulence", PhD thesis, University of Michigan, 2017,
  Sec. 2.4.1.4, p. 57  (algorithm, shown for the Overlap images, 50 %, 3 passes)
  Sec. 2.4.2,   p. 65  (OH-PLIF for S_T: "the same algorithm", 25 %)
Paper: Wabel et al., PCI 36 (2017) 1801-1808, p. 1803 (25 %, "iterated four times").

Algorithm (thesis wording, paraphrased)
  1. A watershed algorithm is applied to the image, treated as an elevation
     map (brightness = height). Its watershed lines are the ridge lines, along
     which the signal is a local maximum.
  2. Every pixel is compared with the signal at the NEAREST ridge point; if it
     is below `frac` times that value, it is set to zero.
  3. Steps 1-2 are repeated on the thresholded image (ridges are recomputed).

Difference to the moving-window version in run_wabel_batch.py
  wabel_local_threshold(img, ws, frac, n) compares each pixel with the maximum
  inside a fixed ws x ws square. Here the reference is the crest of the
  nearest bright structure, whatever its distance, and it is recomputed in
  every pass, so in principle the number of passes can change the result
  (on synthetic frames, 1 and 4 passes gave identical binaries).

Choices not specified by Wabel (state them in the thesis if this is used)
  * pre_median: size of a median filter applied ONLY to the elevation map
    used to find the ridges (suppresses ridges on single noise pixels).
    Wabel median- and level-set-filtered his grey images before thresholding
    (thesis p. 56); this is a lighter stand-in. 0 disables it.
  * connectivity of the watershed (1 = 4-neighbourhood, 2 = 8-neighbourhood).
  * markers: all regional minima of the elevation map (skimage default).
  * "nearest" ridge = smallest Euclidean distance.
  * ridges on pixels that are already zero are ignored; if no ridge is left,
    the frame maximum is used as the reference.

Used by compare_binarisation.py.  Drop-in for run_wabel_batch.binary_u8:
    bi = wabel_watershed_threshold(work.clip(0, 255).astype(np.uint8),
                                   frac, N_ITER)
instead of
    bi = wabel_local_threshold(work.clip(0, 255).astype(np.uint8),
                               WINDOW_SIZE, frac, N_ITER)
Needs scikit-image (pip3 install scikit-image).
"""
import numpy as np
from scipy.ndimage import distance_transform_edt, median_filter
from skimage.segmentation import watershed


def ridge_lines(img, pre_median=3, connectivity=1):
    """Boolean mask of watershed lines (ridges) of `img` seen as a landscape."""
    elev = img.astype(np.float64)
    if pre_median and pre_median > 1:
        elev = median_filter(elev, size=int(pre_median))
    labels = watershed(elev, markers=None, connectivity=connectivity,
                       watershed_line=True)
    return labels == 0


def wabel_watershed_threshold(img, frac, n_iter, pre_median=3, connectivity=1,
                              return_ridges=False):
    """Binarise `img` with Wabel's nearest-ridge local threshold.

    img      : 2-D array (e.g. uint8 after the 10 % noise floor)
    frac     : threshold fraction (0.25 for OH)
    n_iter   : number of passes (ridges recomputed every pass)
    returns  : boolean burnt mask (and the ridge masks of every pass if
               return_ridges=True)
    """
    work = img.astype(np.float64).copy()
    ridges_all = []
    for _ in range(int(n_iter)):
        ridges = ridge_lines(work, pre_median, connectivity)
        # ridges on zero plateaus carry no reference value -> ignore them
        ridges &= work > 0
        if not ridges.any():                       # degenerate: global maximum
            ref = np.full(work.shape, work.max())
        else:
            _, (iy, ix) = distance_transform_edt(~ridges, return_indices=True)
            ref = work[iy, ix]                     # signal at the nearest ridge
        work[work < frac * ref] = 0.0
        if return_ridges:
            ridges_all.append(ridges)
    burnt = work > 0
    return (burnt, ridges_all) if return_ridges else burnt
