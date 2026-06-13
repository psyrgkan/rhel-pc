# Paper: Hamiltonian Predictive Coding

`main.tex` is a full draft of the paper, formatted to compile standalone with
a standard LaTeX install and to convert to the NeurIPS format with a one-line
change.

## Build (standalone)

```bash
cd paper
pdflatex main && pdflatex main      # twice for cross-references / citations
```

Figures are pulled from `../figures/` via `\graphicspath`. No `bibtex` run is
needed — the bibliography is a manual `thebibliography` with author-year
labels (natbib).

## Submitting to NeurIPS

The paper is written for NeurIPS (the venue of the RHEL paper it builds on).
To switch to the official format, download `neurips_2025.sty` from the NeurIPS
site and replace the marked "replaceable layout block" near the top of
`main.tex` with:

```latex
\usepackage[final]{neurips_2025}   % or [preprint], or no option for submission
```

That block also loads `natbib`, which `neurips_2025.sty` provides, so deleting
the block avoids a double load. Nothing else needs to change: the section
structure, theorem environments, figures, and author-year citations are all
compatible.

## Companion file

`reviewer_critique.md` is a deliberately tough, reviewer-style assessment of
this draft — the weaknesses that must be addressed before submission, with
severities and concrete remedies.
