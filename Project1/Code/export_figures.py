"""
export_figures.py -- the figures of the Project 1 report (FYS-STK3155/4155, fall 2026).

The script runs the code cells of the project notebook in order, without changing them, and
saves the figures that the report uses as PDF files at the size of a two-column page.

The data shown in every figure are those of the notebook figure. Only the presentation changes:
the physical size, the font sizes, the line widths and marker sizes, the figure title (the report
has captions instead) and a few label strings that refer to notebook cells or that need the
notation of the report (for example the normalised penalty, written with a bar in the report).

Usage
-----
    python export_figures.py Project1.ipynb Figures

The first argument is the notebook, the second the folder for the PDF files. An optional third
argument is a folder for PNG previews. A complete run takes a few minutes, because every cell of
the notebook is executed.

LLM-assisted
------------
Tool: Claude (Claude Opus 5.5, claude.ai, October 2026)
Role: Wrote this script (Level 4). It contains no numerical work: all numbers come from the
      cells of the notebook, which are executed unchanged.
Verification: every exported figure was compared with the figure shown in the notebook.
"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")                                  # write files only, open no window
import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

matplotlib.rcParams["font.family"] = "DejaVu Sans"     # Matplotlib's own default font, set explicitly
                                                       # so that every computer gives the same figures

COL, FULL = 3.40, 7.05      # width of one column and of the full text block of the report, in inches
THIRD = 2.32                # width of one of three panels that the report sets side by side

# index of the notebook cell that draws the figure -> (file name, width, height, options of restyle)
FIGS = {
    20: ("fig_ols_mse_r2", FULL, 2.3, {}),
    22: ("fig_ols_parameters", FULL, 2.35, dict(
        axes_opts={0: dict(ylim_top=2e5, legend_loc="upper center", legend_ncol=4)})),
    24: ("fig_ols_fits", COL, 2.35, dict(drop_titles=True)),
    26: ("fig_ols_n_sigma", FULL, 2.3, {}),
    37: ("fig_ridge_degree", FULL, 2.15, dict(legend_below=6)),
    41: ("fig_ridge_lambda_df", FULL, 2.3, dict(
        replace={"Ridge, degree 15: MSE against the penalty": "MSE against the penalty",
                 "Effective number of parameters": "effective number of parameters",
                 "p = 15 (OLS)": "15 (OLS)"})),
    45: ("fig_ridge_svd", FULL, 2.15, dict(
        replace={"Shrinkage factor, Eq. (3.48)": "shrinkage factor",
                 r"Spectrum of $X$ (degree 15, standardised)": "squared singular values",
                 "Coefficients in the singular basis": "coefficients in the singular basis",
                 r"mode $i$ (largest $\sigma_i$ first)": r"mode $i$"},
        legend_below=6, legend_below_panel=1)),
    53: ("fig_train_test_complexity", FULL, 1.95, dict(
        replace={"polynomial degree (model complexity)": "polynomial degree"}, drop_texts=True,
        move_legend=(0, 2, "upper right"))),
    57: ("fig_bias_variance", THIRD, 2.0, dict(
        drop_titles=True,
        replace={r"bias$^2$ (measured with $y$, absorbs $\sigma^2$)": r"bias$^2$ (measured)"})),
    59: ("fig_bias_variance_n", FULL, 1.95, dict(move_legend=(0, 2, "upper right"))),
    66: ("fig_cv_ols", THIRD, 2.0, dict(drop_titles=True)),
    70: ("fig_cv_bootstrap", THIRD, 2.0, dict(
        drop_titles=True, replace={"bootstrap error (c.4)": "bootstrap error"})),
    81: ("fig_gd_learning_rate", FULL, 2.3, dict(
        replace={r"Ridge, $\lambda=0.01$: $\kappa = 178$": r"Ridge, $\bar\lambda=0.01$: $\kappa = 178$",
                 r"degree 5: dotted $\eta^*$, dashed $2/\lambda_{\max}$":
                 r"dotted: $\eta^*$, dashed: $2/\lambda_{\max}$",
                 r"OLS, degree 5: $\eta_{\max} = 2/\lambda_{\max} = 0.360$":
                 r"OLS: $\eta_{\max} = 2/\lambda_{\max} = 0.360$"})),
    90: ("fig_optimisers_scan", FULL, 2.3, dict(
        replace={"OLS, degree 5": "OLS",
                 "Ridge, lambda = 0.01, degree 5": r"Ridge, $\bar\lambda = 0.01$"}, legend_below=7)),
    101: ("fig_lasso_gd", FULL, 2.3, dict(
        replace={r"Lasso, $\lambda = 0.01$, degree 5": "after 20000 iterations",
                 "best learning rate per method": "best learning rate of each method",
                 r"$\|\theta_k - \hat\theta_{\rm Lasso}\|_2$ after 20000 iterations":
                 r"$\|\theta_k - \hat\theta_{\rm Lasso}\|_2$",
                 "iteration $k$ (log scale)": "iteration $k$"})),
    110: ("fig_sgd_batch", FULL, 1.85, dict(
        replace={r"excess cost $C(\theta) - C(\hat\theta_{\rm OLS})$": "excess cost"},
        label_fs=7.5, tick_fs=6.0, title_fs=7.5)),
    112: ("fig_sgd_schedule", COL, 2.4, dict(
        drop_titles=True, replace={r"excess cost $C(\theta) - C(\hat\theta_{\rm OLS})$": "excess cost"})),
    122: ("fig_final_cv", FULL, 2.25, dict(
        replace={"the three methods against the degree": "best model per degree",
                 r"Ridge (best $\lambda$ per degree)": r"Ridge (best $\lambda$)",
                 r"Lasso (best converged $\lambda$ per degree)": r"Lasso (best converged $\bar\lambda$)",
                 r"$\log_{10}$ CV MSE, Ridge": r"Ridge: $\log_{10}$ CV MSE",
                 r"$\log_{10}$ CV MSE, Lasso": r"Lasso: $\log_{10}$ CV MSE"},
        ylabel_by_title={r"Lasso: $\log_{10}$ CV MSE": r"$\log_{10}\bar\lambda$"},
        axes_opts={0: dict(ylim_top=0.3, legend_loc="upper right")})),
}

NAMES = {"adagrad": "AdaGrad", "rmsprop": "RMSprop", "adam": "Adam"}   # spelling used in the report


def pretty(s):
    """Rewrite a label for print: 1e-04 -> 10^{-4}, 5e-02 -> 5x10^{-2}, 'n = 50' in italics,
    and the names of the optimisers as they are spelled in the report.

    LLM-assisted
    ------------
    Tool: Claude (Claude Opus 5.5, claude.ai, October 2026)
    Role: Wrote the function.
    """
    def sci(m):
        mant, exp = float(m.group(1)), int(m.group(2))
        if exp == 0:
            body = f"{mant:g}"
        elif mant == 1:
            body = f"10^{{{exp}}}"
        else:
            body = f"{mant:g}\\times10^{{{exp}}}"
        return "$" + body + "$"
    s = re.sub(r"(\d(?:\.\d+)?)e([+-]\d+)", sci, s)
    s = re.sub(r"^([nM]) = (\d+)$", r"$\1 = \2$", s)
    for old, new in NAMES.items():
        s = re.sub(rf"\b{old}\b", new, s)
    return s


def restyle(fig, width, height, drop_titles=False, drop_texts=False, replace=None, ylabel_by_title=None,
            axes_opts=None, move_legend=None, legend_below=None, legend_below_panel=None,
            label_fs=8.0, tick_fs=7.0, title_fs=8.0, leg_fs=6.3, text_fs=6.5,
            lw_scale=0.72, ms_scale=0.55):
    """Give a notebook figure the size and the fonts of the report. The plotted data are not touched.

    width, height   : size in inches (COL or FULL wide)
    drop_titles     : remove the titles of the panels (single-panel figures: the caption says it)
    drop_texts      : remove free text annotations inside the panels
    replace         : dict {label in the notebook: label in the report}
    ylabel_by_title : dict {panel title: new y label}
    axes_opts       : dict {panel index: dict(ylim_top=..., legend_loc=..., legend_ncol=...)}
    move_legend     : (from panel, to panel, location), when the legend would cover the curves
    legend_below    : number of columns of a legend placed under the panels instead of inside them
    legend_below_panel : index of the panel whose legend is moved there (default: the panels share
                      one legend, which is taken from the last panel that has one)

    LLM-assisted
    ------------
    Tool: Claude (Claude Opus 5.5, claude.ai, October 2026)
    Role: Wrote the function.
    Verification: every exported figure was compared with the figure shown in the notebook.
    """
    replace = replace or {}
    axes_opts = axes_opts or {}
    rep = lambda s: pretty(replace.get(s, s))
    leg_kw = dict(fontsize=leg_fs, handlelength=1.7, handletextpad=0.5, borderpad=0.35, labelspacing=0.25,
                  columnspacing=0.9, borderaxespad=0.4, framealpha=0.85)
    moved = None

    if fig._suptitle is not None:                 # the caption of the report replaces the figure title
        fig._suptitle.remove()
        fig._suptitle = None

    for ax in fig.axes:
        ax.set_title("" if drop_titles else rep(ax.get_title()), fontsize=title_fs, pad=3)
        ax.set_xlabel(rep(ax.get_xlabel()), fontsize=label_fs, labelpad=2)
        ax.set_ylabel(rep(ax.get_ylabel()), fontsize=label_fs, labelpad=2)
        ax.tick_params(axis="both", which="both", labelsize=tick_fs, pad=2)
        ax.tick_params(axis="both", which="major", length=2.5, width=0.6)
        ax.tick_params(axis="both", which="minor", length=1.5, width=0.4)
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
        for line in ax.lines:                     # thinner lines and smaller markers for the small format
            line.set_linewidth(max(0.6, line.get_linewidth() * lw_scale))
            line.set_markersize(line.get_markersize() * ms_scale)
        for coll in ax.collections:               # scatter plots
            try:
                coll.set_sizes(coll.get_sizes() * ms_scale ** 2)
            except Exception:
                pass
        for t in list(ax.texts):
            if drop_texts:
                t.remove()
            else:
                t.set_fontsize(text_fs)
        opt = axes_opts.get(fig.axes.index(ax), {})
        if "ylim_top" in opt:
            ax.set_ylim(top=opt["ylim_top"])
        if ylabel_by_title and ax.get_title() in ylabel_by_title:
            ax.set_ylabel(ylabel_by_title[ax.get_title()], fontsize=label_fs, labelpad=2)

        leg = ax.get_legend()
        if leg is not None:                       # rebuild the legend compactly, same entries and place
            handles = leg.legend_handles
            labels = [rep(t.get_text()) for t in leg.get_texts()]
            loc, ncols = opt.get("legend_loc", leg._loc), opt.get("legend_ncol", leg._ncols)
            leg.remove()
            for h in handles:                     # legend symbols as thin and as small as the plotted ones
                if hasattr(h, "set_markersize"):
                    h.set_markersize(h.get_markersize() * ms_scale)
                    h.set_linewidth(max(0.6, h.get_linewidth() * lw_scale))
            index = fig.axes.index(ax)
            if legend_below and legend_below_panel in (None, index):   # under the panels, not on the curves
                moved = (handles, labels)
                continue
            if move_legend and index == move_legend[0]:
                moved = (handles, labels)
                continue
            new = ax.legend(handles, labels, loc=loc, ncol=ncols, **leg_kw)
            new.get_frame().set_linewidth(0.5)
            # white markers (Lasso fits that did not converge) would be invisible on a white legend box
            if any(str(getattr(h, "get_color", lambda: "")()) in ("w", "white") for h in handles):
                new.get_frame().set_facecolor("0.72")

    fig.set_size_inches(width, height)
    fig.tight_layout(pad=0.35, w_pad=0.9, h_pad=0.6)
    if moved and move_legend:
        new = fig.axes[move_legend[1]].legend(*moved, loc=move_legend[2], **leg_kw)
        new.get_frame().set_linewidth(0.5)
    if moved and legend_below:
        new = fig.legend(*moved, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=legend_below, **leg_kw)
        new.get_frame().set_linewidth(0.5)
    return fig


def main(notebook, out_dir, preview_dir=None):
    """Execute the code cells of the notebook in order and save the figures listed in FIGS.

    LLM-assisted
    ------------
    Tool: Claude (Claude Opus 5.5, claude.ai, October 2026)
    Role: Wrote the function.
    Verification: the printed output of every cell was compared with the output stored in the notebook.
    """
    os.makedirs(out_dir, exist_ok=True)
    if preview_dir:
        os.makedirs(preview_dir, exist_ok=True)
    with open(notebook, encoding="utf-8") as fh:
        cells = json.load(fh)["cells"]
    shell = InteractiveShell.instance()           # the same interpreter that Jupyter uses for a notebook
    saved = []
    for i, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        with capture_output():                    # the printed output is not needed here
            result = shell.run_cell("".join(cell["source"]), store_history=False)
        error = result.error_before_exec or result.error_in_exec
        if error is not None:
            raise RuntimeError(f"cell {i} of {notebook} failed: {error!r}")
        numbers = plt.get_fignums()               # figures that this cell has opened
        if i in FIGS:
            if len(numbers) != 1:
                raise RuntimeError(f"cell {i} should draw exactly one figure; the notebook has changed, "
                                   "update the table FIGS")
            name, width, height, options = FIGS[i]
            fig = restyle(plt.figure(numbers[0]), width, height, **options)
            fig.savefig(os.path.join(out_dir, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
            if preview_dir:
                fig.savefig(os.path.join(preview_dir, name + ".png"), bbox_inches="tight", pad_inches=0.02,
                            dpi=170)
            saved.append(name)
            print(f"cell {i:3d} -> {name}.pdf", flush=True)
        plt.close("all")
    print(f"{len(saved)} of {len(FIGS)} figures written to {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: python export_figures.py NOTEBOOK.ipynb OUTPUT_FOLDER [PREVIEW_FOLDER]")
    main(*sys.argv[1:4])
