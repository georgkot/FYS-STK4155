# FYS-STK3155/4155 Project 1: Polynomial regression on Runge's function

Pietro Monacelli, Giorgio Spreafico, Elia Valenti, Georgios Kotrotsios — University of Oslo, fall 2026

OLS, Ridge and Lasso regression of Runge's function f(x) = 1/(1+25x²), with bootstrap and
cross-validation, and our own gradient descent, momentum, AdaGrad, RMSprop, Adam and SGD.

## Contents
- `Code/Project1.ipynb` — all code, organised by the parts a)–i) of the project, saved with all outputs
- `Code/export_figures.py` — runs the notebook unchanged and saves the report's figures as PDF
- `Code/requirements.txt` — Python packages needed
- `Results/Figures/` — the figures used in the report
- `Report/` — the report (PDF) and its LaTeX source

## How to run
    pip install -r Code/requirements.txt
    jupyter notebook Code/Project1.ipynb        # Run All; takes a few minutes
    python Code/export_figures.py Code/Project1.ipynb Results/Figures

All random choices use the seed 2026, so a full run reproduces the numbers in the report
(except computing times and differences at rounding level).

The use of Claude (Anthropic) for code and text is declared in Appendix A of the report and in the notebook.
