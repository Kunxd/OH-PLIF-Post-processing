"""
OH-PLIF binary image post-processing: isolamento del core interno
-------------------------------------------------------------------
Obiettivo: a partire da un'immagine binarizzata (0 = non-fiamma, 1 = fiamma),
impostare a 1 (bianco) tutto ciò che è ESTERNO alla fiamma, lasciando a 0
(nero) solo il core interno non bruciato. In questo modo il core diventa
l'unica regione nera residua nell'immagine, topologicamente isolata, anche
in presenza di piccoli gap nel contorno di fiamma dovuti a rumore.

Metodo: flood-fill a partire dai bordi dell'immagine (che sono per
costruzione sempre esterni alla fiamma), con padding di sicurezza per
evitare che il flood-fill "entri" nel core se la fiamma tocca il bordo
dell'immagine (es. in alto/basso dove il getto esce dal FOV).

Dipendenze: numpy, scipy, matplotlib (per la validazione visiva)
    pip install numpy scipy matplotlib
"""

import numpy as np
from scipy.ndimage import binary_fill_holes, label
import matplotlib.pyplot as plt


def isolate_inner_core(binary_image, pad_width=5, seed_value=0, verbose=False):
    """
    Imposta a 1 tutto ciò che è esterno alla fiamma, isolando il core
    interno come unica regione a 0.

    Parametri
    ----------
    binary_image : np.ndarray (2D), dtype bool o {0,1}
        Immagine binarizzata OH-PLIF. Convenzione: 1 = segnale di fiamma,
        0 = non-fiamma (sia core interno che sfondo esterno).
    pad_width : int
        Spessore (in pixel) della cornice bianca artificiale aggiunta
        attorno all'immagine prima del flood-fill. Serve a garantire che
        il seed del flood-fill sia sempre in una regione genuinamente
        esterna, anche se la fiamma tocca il bordo del frame originale.
        Valore tipico: 5-10 px, va bene anche di più se non sei sicuro.
    seed_value : int
        Valore che identifica "non-fiamma" nell'immagine originale
        (di default 0). Non modificare a meno che la tua convenzione
        sia invertita.
    verbose : bool
        Se True, stampa informazioni diagnostiche (numero di componenti
        connesse trovate, dimensione del core isolato).

    Ritorna
    -------
    core_mask : np.ndarray (2D), stessa shape di binary_image, dtype bool
        True dove si trova il core interno isolato (il resto è False,
        cioè "bianco"/esterno).
    processed_image : np.ndarray (2D), dtype uint8
        L'immagine finale con tutto l'esterno impostato a 1 e solo il
        core a 0, già ritagliata alla shape originale (padding rimosso).
    """

    binary_image = np.asarray(binary_image).astype(bool)

    # Step 1: aggiungo una cornice di pixel "non-fiamma" (False) attorno
    # all'immagine. Questo garantisce un seed esterno sicuro per il
    # flood-fill, anche se la fiamma tocca il bordo dell'immagine originale.
    padded = np.pad(binary_image, pad_width=pad_width,
                     mode='constant', constant_values=False)

    # Step 2: inverto temporaneamente la convenzione per usare
    # binary_fill_holes in modo utile.
    #
    # binary_fill_holes riempie tutti i "buchi" (regioni di False
    # completamente circondate da True) in un array binario. Qui la
    # applichiamo sull'immagine NON invertita: le regioni di False
    # (non-fiamma) che sono completamente "circondate" da fiamma (True)
    # vengono riempite a True. Questo corrisponde esattamente a
    # trasformare in "fiamma" tutto ciò che è topologicamente interno
    # (il core), lasciando inalterato lo sfondo esterno (che è connesso
    # al bordo dell'immagine per costruzione, quindi NON viene riempito).
    filled = binary_fill_holes(padded)

    # A questo punto:
    #   - filled == True  -> fiamma originale OPPURE core interno riempito
    #   - filled == False -> solo sfondo esterno
    #
    # Il core isolato è quindi: (filled == True) AND (padded == False)
    # cioè i pixel che sono stati "riempiti" ma che nell'immagine
    # originale erano a False (non facevano parte del segnale di fiamma).
    core_mask_padded = filled & (~padded)

    # Step 3: verifica diagnostica - quante componenti connesse di core
    # abbiamo trovato? Idealmente 1 (un solo core per fiamma), ma può
    # capitare di averne di più se ci sono gap nel contorno che spezzano
    # il core in più pezzi, o rumore residuo.
    labeled_core, n_components = label(core_mask_padded)

    if verbose:
        print(f"Numero di componenti connesse nel core isolato: {n_components}")
        for i in range(1, n_components + 1):
            size = np.sum(labeled_core == i)
            print(f"  Componente {i}: {size} pixel")

    # Step 4: rimuovo il padding per tornare alla shape originale.
    core_mask = core_mask_padded[pad_width:-pad_width, pad_width:-pad_width]

    # Step 5: costruisco l'immagine finale: 1 ovunque TRANNE nel core,
    # che resta 0.
    processed_image = np.ones_like(binary_image, dtype=np.uint8)
    processed_image[core_mask] = 0

    return core_mask, processed_image


def validate_visually(binary_image, core_mask, processed_image):
    """
    Plot di confronto per validazione visiva rapida: originale, core
    isolato, immagine finale processata.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(binary_image, cmap='gray')
    axes[0].set_title('Originale (post-binarizzazione)')
    axes[0].axis('off')

    axes[1].imshow(core_mask, cmap='gray')
    axes[1].set_title('Core interno isolato')
    axes[1].axis('off')

    axes[2].imshow(processed_image, cmap='gray')
    axes[2].set_title('Risultato finale\n(tutto esterno = 1)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


# =====================================================================
# ADDED (Axel's repo, not in the original): input/output folders only.
# Nothing above or below this block is changed.  This block only
#   - reads OUTDIR from config_local.py (same settings file as run_wabel_batch.py)
#   - saves every figure that validate_visually() shows into OUTDIR/white_bg/
#     (if that folder already exists: white_bg_2, white_bg_3, ... nothing is overwritten)
#   - writes run_info.txt into that folder: GitHub commit of this script, local
#     changes, date, command, settings, package versions, input files
#     (the commit is also stored in every PNG's metadata)
#   - lets you run the original functions on real binarized frames
#   - optional --pad-tb N: adds N WHITE rows on top and bottom of the image
#     BEFORE the original isolate_inner_core() runs, and removes them again
#     afterwards (her function itself still pads black on all sides)
# Usage:
#   python3 white_bg.py                  original synthetic test        → OUTDIR/white_bg/synthetic_test.png
#   python3 white_bg.py --real           all binary_*.png and instant_binary_composite_*.npy under OUTDIR
#   python3 white_bg.py FILE_OR_FOLDER   only these images (.png / .npy, or a folder with them)
#   add  --pad-tb 5  to any of these to use 5 white rows on top/bottom (default: off = original)
# =====================================================================
if __name__ == "__main__":
    import os, sys, glob, warnings, re, subprocess, datetime, platform, scipy, matplotlib
    _here = os.path.dirname(os.path.abspath(__file__))
    _argv = sys.argv[1:]; _pad_tb = 0
    if "--pad-tb" in _argv:
        _i = _argv.index("--pad-tb")
        try:
            _pad_tb = int(_argv[_i + 1])
        except (IndexError, ValueError):
            sys.exit("--pad-tb needs a number of pixels, e.g.  --pad-tb 5")
        del _argv[_i:_i + 2]
    if _pad_tb > 0:
        _iic_original = isolate_inner_core
        def isolate_inner_core(binary_image, pad_width=5, **kw):
            img = np.asarray(binary_image).astype(bool)
            img = np.pad(img, ((_pad_tb, _pad_tb), (0, 0)), constant_values=True)  # white rows
            core, proc = _iic_original(img, pad_width=pad_width, **kw)
            return core[_pad_tb:-_pad_tb], proc[_pad_tb:-_pad_tb]
    sys.path.insert(0, _here)
    try:
        from config_local import OUTDIR
        print(f"config_local.py : {os.path.join(_here, 'config_local.py')}")
    except ImportError:
        OUTDIR = os.path.join(_here, "Wabel_results")
        print("no config_local.py found → OUTDIR =", OUTDIR)
    # new output folder per run: white_bg, white_bg_2, white_bg_3, ...
    _out = os.path.join(OUTDIR, "white_bg"); _k = 1
    while os.path.exists(_out):
        _k += 1; _out = os.path.join(OUTDIR, f"white_bg_{_k}")
    os.makedirs(_out)

    # GitHub version of this script (git clone needed; 'unknown' otherwise)
    def _git(*a):
        try:
            return subprocess.run(["git", "-C", _here, *a], capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
    _commit = _git("rev-parse", "HEAD") or "unknown (not a git clone)"
    _cdate = _git("log", "-1", "--format=%ci")
    _remote = _git("config", "--get", "remote.origin.url")
    _dirty = _git("status", "--porcelain", "--", os.path.basename(__file__))
    _version = _commit[:7] + (" + LOCAL CHANGES (not on GitHub)" if _dirty else "")
    _info = os.path.join(_out, "run_info.txt")
    with open(_info, "w", encoding="utf-8") as _fh:
        _fh.write(f"script         : {os.path.abspath(__file__)}\n"
                  f"github         : {_remote}\n"
                  f"commit         : {_commit}  ({_cdate})\n"
                  f"local changes  : {'YES - white_bg.py differs from this commit' if _dirty else 'no'}\n"
                  f"run at         : {datetime.datetime.now().isoformat(timespec='seconds')}\n"
                  f"command        : {' '.join([os.path.basename(sys.executable)] + sys.argv)}\n"
                  f"OUTDIR         : {OUTDIR}\n"
                  f"versions       : python {platform.python_version()}, numpy {np.__version__}, "
                  f"scipy {scipy.__version__}, matplotlib {matplotlib.__version__}\n"
                  f"settings       : isolate_inner_core(pad_width=5, verbose=True)"
                  + (f", plus {_pad_tb} WHITE rows added top/bottom first (--pad-tb {_pad_tb})"
                     if _pad_tb else ", original black padding only") + "\n\n"
                  f"input files:\n")
    print(f"commit          : {_version}")
    print(f"white top/bottom: {str(_pad_tb) + ' px' if _pad_tb else 'off (original)'}")
    print(f"output folder   : {_out}\n")

    # save figures instead of opening windows (plt.show inside validate_visually)
    plt.switch_backend("Agg")
    warnings.filterwarnings("ignore", message=".*non-interactive.*")
    _save_as = [os.path.join(_out, "synthetic_test.png")]
    def _show_and_save(*a, **k):
        plt.savefig(_save_as[0], dpi=110, bbox_inches="tight",
                    metadata={"Software": f"white_bg.py {_remote} commit {_commit}"
                                          + (" + local changes" if _dirty else "")})
        plt.close("all")
        print(f"  figure → {_save_as[0]}")
    plt.show = _show_and_save

    _args = _argv
    if _args:
        _files = []
        for _p in ([OUTDIR] if _args == ["--real"] else _args):
            if os.path.isdir(_p):
                for _pat in ("binary_*.png", "instant_binary_composite_*.npy"):
                    _files += glob.glob(os.path.join(_p, "**", _pat), recursive=True)
            else:
                _files.append(_p)
        _files = sorted(f for f in _files
                        if not re.search(r"(^|[\\/])white_bg(_\d+)?[\\/]", os.path.relpath(f, OUTDIR)))
        if not _files:
            print("no binarized images found (binary_*.png / instant_binary_composite_*.npy)")
        for _f in _files:
            print(f"\n{os.path.relpath(_f, OUTDIR) if _f.startswith(OUTDIR) else _f}")
            if _f.endswith(".npy"):
                img = np.load(_f) > 0
            else:
                from PIL import Image
                img = np.array(Image.open(_f).convert("L")) > 0
            core_mask, processed = isolate_inner_core(img, pad_width=5, verbose=True)
            _rel = os.path.relpath(_f, OUTDIR) if _f.startswith(OUTDIR) else os.path.basename(_f)
            _save_as[0] = os.path.join(_out, os.path.splitext(_rel)[0].replace(os.sep, "__") + ".png")
            validate_visually(img, core_mask, processed)
            with open(_info, "a", encoding="utf-8") as _fh:
                _fh.write(f"  {os.path.abspath(_f)}  →  {os.path.basename(_save_as[0])}\n")
        sys.exit(0)
    with open(_info, "a", encoding="utf-8") as _fh:
        _fh.write("  (none: original synthetic test → synthetic_test.png)\n")
# ===================== end of ADDED block ============================


# ---------------------------------------------------------------------
# Esempio d'uso / test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Qui lo studente carica la sua immagine binarizzata reale, ad es.:
    #   from PIL import Image
    #   img = np.array(Image.open("frame_001_binary.png")) > 0
    #
    # Per ora, generiamo un'immagine sintetica di test che simula la
    # situazione: due "rami" di fiamma con un gap nel contorno, per
    # verificare che il metodo sia robusto anche in quel caso critico.

    test_img = np.zeros((300, 300), dtype=bool)
    # ramo sinistro
    test_img[50:250, 80:120] = True
    # ramo destro
    test_img[50:250, 180:220] = True
    # base che chiude il contorno in basso
    test_img[230:250, 80:220] = True
    # introduco volutamente un piccolo GAP nel contorno in alto, per
    # testare la robustezza rispetto a bordi di fiamma non perfettamente
    # chiusi
    test_img[50:55, 95:105] = False  # gap nel ramo sinistro

    core_mask, processed = isolate_inner_core(test_img, pad_width=5, verbose=True)
    validate_visually(test_img, core_mask, processed)
