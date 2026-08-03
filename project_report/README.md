# Capstone Project Final Report

This directory contains the modular LaTeX source for the capstone report.

## Build

From `project_report`:

```powershell
latexmk -pdf CapstoneProjectFinalReport.tex
```

The project uses XeLaTeX through `latexmkrc` so the report can use the installed
Times New Roman system font. When building outside `latexmk`, run:

```powershell
xelatex CapstoneProjectFinalReport.tex
```

Clean generated files:

```powershell
latexmk -C
```

The source is also compatible with Overleaf. Upload the complete
`project_report` directory and select `CapstoneProjectFinalReport.tex` as the
main document.

## Required assets

The report compiles without external images. Missing images render as labelled
placeholders. Before submission, add:

- `assets/fpt-university-logo.png`
- `assets/ui-main.png`
- `assets/ui-knowledge-bases.png`
- `assets/ui-ai-models.png`
- `assets/ui-evaluation.png`
- `assets/ui-trace.png`
- `assets/ui-analytics.png`
- `assets/ragxplain-insights.png`

Do not include API keys, credentials, private chat content, or unredacted
personal information in screenshots.

## Updating experiment results

The current `results_values.tex` records the completed four-class classifier
metrics and primary 12-cell WixQA configuration experiment. Update it only when
a new controlled experiment supersedes those values. The chapters refer to the
macros so reported numbers remain consistent.

## Current compilation status

The report compiles locally with MiKTeX/XeLaTeX through `latexmk`. Render and
inspect the generated PDF after every result-table or figure update.
