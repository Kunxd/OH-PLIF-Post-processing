#!/usr/bin/env python3
"""
white_bg.py — OH-PLIF binary image post-processing: isolate the inner core
---------------------------------------------------------------------------
(original script by Prof. Pitsch's group, translated and adapted)

Goal: starting from a binarized image (0 = no flame, 1 = flame), set
everything OUTSIDE the flame to 1 (white), leaving only the unburnt inner
core at 0 (black).  The core is then the only black region left in the
image, topologically isolated.

Method: scipy.ndimage.binary_fill_holes fills every black region that is
completely enclosed by white — the core — while the background, which is
connected to the image border, is not filled.  A padding frame is added
first so that the decision does not depend on what touches the border.

ADAPTATIONS for our OH-PLIF frames (see README / chat notes)
  1. Padding.  The original padded with black (False) on all four sides.
     In our frames the core runs through the TOP and BOTTOM edge of every
     position (nozzle below, flame continues above), so it touches the black
     frame, counts as "outside" and is never filled → 0 core pixels.
     Now: top/bottom are padded WHITE (closes the core where the jet enters
     and leaves the field of view), left/right stay BLACK (that is where the
     ambient background is).  `pad_top_bottom_white=False` restores the
     original behaviour.
  2. Gaps.  A gap that really cuts through the flame band connects core and
     background; then the core is not enclosed and is NOT filled.  Optional
     `close_gaps_px` seals gaps up to roughly that size (morphological
     closing) for the fill decision only.
  3. Diagnostics are also returned (return_info=True), plots are saved to a
     file instead of plt.show() (works on a laptop terminal without display).
  4. Command-line validation on real frames from our results folders.

Usage
  python3 white_bg.py --synthetic                       # synthetic tests
  python3 white_bg.py                                   # real frames, OUTDIR from config_local.py
  python3 white_bg.py --results /path/to/Run1 --flame D12_CH4_Re15000 --close 5
"""

import os
import sys
import csv
import glob
import argparse
import numpy as np
from scipy.ndimage import binary_fill_holes, label, binary_closing
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def isolate_inner_core(binary_image, pad_width=5, seed_value=0, verbose=False,
                       pad_top_bottom_white=True, close_gaps_px=0,
                       return_info=False):
    """
    Set everything outside the flame to 1, isolating the inner core as the
    only region at 0.

    Parameters
    ----------
    binary_image : 2D array, bool or {0,1}
        Binarized OH-PLIF image.  1 = flame signal, 0 = no flame (both the
        inner core and the outer background).
    pad_width : int
        Thickness [px] of the artificial frame added around the image.
    seed_value : int
        Value meaning "no flame" in the input (default 0).  Only change if
        your convention is inverted.
    verbose : bool
        Print diagnostics (number of core components and their size).
    pad_top_bottom_white : bool
        True (default, for our frames): top/bottom padding white, left/right
        black.  False: original behaviour, black padding on all sides.
    close_gaps_px : int
        0 = off.  >1: seal gaps up to about this many pixels in the flame
        band before deciding what is enclosed.
    return_info : bool
        Also return a dict with diagnostics.

    Returns
    -------
    core_mask : 2D bool array (same shape as input)
        True on the isolated inner core (all other pixels False).
    processed_image : 2D uint8 array
        1 everywhere except the core, which is 0.
    info : dict (only if return_info=True)
        n_components, component sizes (largest first), core pixels.
    """
    img = np.asarray(binary_image)
    flame = (img != seed_value)

    work = flame
    if close_gaps_px and close_gaps_px > 1:
        k = int(close_gaps_px)
        yy, xx = np.mgrid[:k, :k]
        disk = ((yy - (k - 1) / 2) ** 2 + (xx - (k - 1) / 2) ** 2) <= ((k - 1) / 2 + 0.5) ** 2
        # pad before closing so the closing does not erode at the border
        work = binary_closing(np.pad(flame, k, mode="edge"), structure=disk)[k:-k, k:-k]
        work |= flame

    # Step 1: padding frame (see ADAPTATIONS above)
    padded = np.pad(work, pad_width=pad_width, mode="constant", constant_values=False)
    if pad_top_bottom_white:
        padded[:pad_width, pad_width:-pad_width] = True
        padded[-pad_width:, pad_width:-pad_width] = True

    # Step 2: fill every black region completely enclosed by white = the core
    # (and unburnt pockets).  The background stays black because it is
    # connected to the black left/right padding.
    filled = binary_fill_holes(padded)

    # core = pixels that were filled but were not flame before
    core_mask_padded = filled & (~padded)

    # Step 3: diagnostics — how many separate core regions?
    labeled_core, n_components = label(core_mask_padded)
    sizes = sorted((int(s) for s in np.bincount(labeled_core.ravel())[1:]), reverse=True)
    if verbose:
        print(f"Number of connected components in the isolated core: {n_components}")
        for i, s in enumerate(sizes, 1):
            print(f"  component {i}: {s} pixels")

    # Step 4: remove padding
    core_mask = core_mask_padded[pad_width:-pad_width, pad_width:-pad_width]

    # Step 5: final image: 1 everywhere except the core
    processed_image = np.ones_like(flame, dtype=np.uint8)
    processed_image[core_mask] = 0

    if return_info:
        return core_mask, processed_image, {"n_components": int(n_components),
                                            "component_sizes": sizes,
                                            "core_pixels": int(core_mask.sum())}
    return core_mask, processed_image


def validate_visually(binary_image, core_mask, processed_image, save_path=None, title=""):
    """Side-by-side check: original, isolated core, final image."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5 * binary_image.shape[0] / binary_image.shape[1] + 1))
    axes[0].imshow(np.asarray(binary_image, dtype=float), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title("Original (binarized)")
    lab, n = label(core_mask)
    axes[1].imshow(lab, cmap="nipy_spectral" if n > 1 else "gray", vmin=0, vmax=max(n, 1), interpolation="nearest")
    axes[1].set_title(f"Isolated inner core ({n} component{'s' if n != 1 else ''})")
    axes[2].imshow(processed_image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[2].set_title("Result (outside = 1)")
    for a in axes:
        a.axis("off")
    if title:
        fig.suptitle(title, fontsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────
# Synthetic tests
# ─────────────────────────────────────────────────────────────────────
def synthetic_tests(outdir):
    os.makedirs(outdir, exist_ok=True)
    print("SYNTHETIC TESTS\n")

    # A) the original test image: two branches + closed base, small notch
    a = np.zeros((300, 300), dtype=bool)
    a[50:250, 80:120] = True
    a[50:250, 180:220] = True
    a[230:250, 80:220] = True
    a[50:55, 95:105] = False

    # A2) same, but the branches run to the top edge (as in our frames, where
    #     the flame leaves the field of view) → white top padding closes it
    #     (the notch is filled in here; otherwise it becomes an enclosed
    #     hole inside the branch and counts as a second "core" component)
    a2 = a.copy(); a2[:55, 80:120] = True; a2[:50, 180:220] = True

    # B) our geometry: core enters at the bottom (nozzle) and closes at a tip,
    #    background left and right, product band touches top and bottom
    H, W = 400, 300
    y, x = np.mgrid[:H, :W]
    half = np.clip(60 * (y - 60) / (H - 60), 0, None)          # core half-width
    b = (np.abs(x - 150) >= half) & (np.abs(x - 150) < half + 50)
    b[:60] = np.abs(x[:60] - 150) < 50                          # closed tip region
    # C) as B, with a 3-px gap cut right through the left band
    c = b.copy(); c[250:253, :150] = False

    # (name, image, settings, expected number of core components, note)
    cases = [("A_original_test", a, dict(pad_top_bottom_white=False), 0,
              "core is open to the black area above the branches → cannot be isolated"),
             ("A2_branches_to_top", a2, dict(pad_top_bottom_white=False), 0,
              "original black padding: core touches top edge → not filled"),
             ("A2_branches_to_top", a2, dict(), 1, "white top/bottom padding"),
             ("B_bunsen_like", b, dict(pad_top_bottom_white=False), 0,
              "original black padding: core touches bottom edge → not filled"),
             ("B_bunsen_like", b, dict(), 1, "white top/bottom padding"),
             ("C_through_gap", c, dict(), 0, "gap through the band connects core and background"),
             ("C_through_gap", c, dict(close_gaps_px=5), 1, "gap sealed")]
    print(f"  {'':3}{'case':20s} {'settings':30s} {'found':>5} {'expected':>8}")
    for name, img, kw, exp, note in cases:
        core, proc, info = isolate_inner_core(img, return_info=True, **kw)
        desc = ", ".join(f"{k}={v}" for k, v in kw.items()) or "defaults"
        ok = "OK " if info["n_components"] == exp else "!! "
        print(f"  {ok}{name:20s} {desc:30s} {info['n_components']:5d} {exp:8d}   {note}")
        fn = f"{name}__{desc.replace('=', '').replace(', ', '_')}.png"
        validate_visually(img, core, proc, os.path.join(outdir, fn), f"{name}  ({desc})")
    print(f"\nPlots → {outdir}")


# ─────────────────────────────────────────────────────────────────────
# Real frames from a results folder
# ─────────────────────────────────────────────────────────────────────
def real_frames(results, outdir, flame_filter=None, close_px=0, max_plots=3, min_size=50):
    os.makedirs(outdir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(results, "*", "pos*", "instant_fronts*.npz")))
    if flame_filter:
        files = [f for f in files if flame_filter.lower() in f.lower()]
    if not files:
        print(f"No pos*/instant_fronts*.npz found under {results}")
        return
    print(f"Results folder : {results}\nFrames from    : {len(files)} position files"
          f"   close_gaps_px={close_px}   (components < {min_size} px counted as specks)\n")
    print(f"{'flame / position':<52} {'frames':>6} {'0 core':>7} {'1 core':>7} {'>1':>5}  "
          f"{'>1 if specks ignored':>20}")
    rows = []
    for f in files:
        pos = os.path.basename(os.path.dirname(f))
        flame = os.path.basename(os.path.dirname(os.path.dirname(f)))
        tag = os.path.basename(f)[len("instant_fronts"):-4].strip("_") or ""
        z = np.load(f)
        bins, idx = z["binaries"], z["frame_index"]
        counts = []
        worst = []
        for i in range(bins.shape[0]):
            core, proc, info = isolate_inner_core(bins[i], close_gaps_px=close_px, return_info=True)
            big = sum(1 for s in info["component_sizes"] if s >= min_size)
            counts.append((info["n_components"], big))
            rows.append([flame, pos, tag, int(idx[i]), info["n_components"], big,
                         info["core_pixels"], " ".join(map(str, info["component_sizes"][:5]))])
            worst.append((big != 1, big, i, core, proc))
        n0 = sum(1 for n, _ in counts if n == 0)
        n1 = sum(1 for n, _ in counts if n == 1)
        nm = sum(1 for n, _ in counts if n > 1)
        nb = sum(1 for _, b in counts if b > 1)
        label_ = f"{flame}/{pos}" + (f" [{tag}]" if tag else "")
        print(f"{label_:<52} {len(counts):6d} {n0:7d} {n1:7d} {nm:5d}  {nb:20d}")
        # plots: first frame + the frames with the most / no core components
        worst.sort(key=lambda w: (not w[0], -w[1]))
        pick = [0] + [w[2] for w in worst[:max_plots - 1] if w[0]]
        for i in dict.fromkeys(pick):
            core, proc = [(w[3], w[4]) for w in worst if w[2] == i][0]
            fn = f"{flame}_{pos}{'_' + tag if tag else ''}_frame{int(idx[i]):04d}.png"
            validate_visually(bins[i], core, proc, os.path.join(outdir, fn),
                              f"{flame} {pos} frame {int(idx[i])}  close_gaps_px={close_px}")
    out = os.path.join(outdir, "white_bg_check.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["flame", "position", "threshold_tag", "frame", "n_components",
                    f"n_components_>={min_size}px", "core_pixels", "largest_component_sizes"])
        w.writerows(rows)
    print(f"\nPer-frame table → {out}\nPlots            → {outdir}")
    print("Note: '0 core' is expected above the flame tip (no unburnt core there); in lower\n"
          "positions it means the core is connected to the background through a gap.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true", help="run only the synthetic tests")
    ap.add_argument("--results", help="results folder (default: OUTDIR from config_local.py)")
    ap.add_argument("--flame", help="only flames whose name contains this text")
    ap.add_argument("--close", type=int, default=0, help="close_gaps_px (default 0 = off)")
    ap.add_argument("--plots", type=int, default=3, help="check images per position (default 3)")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    if a.synthetic:
        synthetic_tests(os.path.join(here, "white_bg_synthetic"))
        return
    root = a.results
    if root is None:
        sys.path.insert(0, here)
        import run_wabel_batch as W
        root = W.OUTDIR
    real_frames(root, os.path.join(root, "white_bg_check" + (f"_close{a.close}" if a.close else "")),
                a.flame, a.close, a.plots)


if __name__ == "__main__":
    main()
