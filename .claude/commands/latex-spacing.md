# latex-spacing

Apply consistent `\vspace{0.3cm}` spacing to a LaTeX file.

## What it does

1. Adds `\vspace{0.3cm}` **before** every `\begin{...}` for: `equation`, `subequations`, `figure`, `figure*`, `table`, `table*`, `algorithm`
2. Adds `\vspace{0.3cm}` **after** every matching `\end{...}`
3. Skips insertion if `\vspace` is already present on the adjacent line (no double-spacing)

## How to invoke

Run on a specific file:
```
/latex-spacing IEEE Paper/use_this_file.tex
```
Or on the currently open file (no argument).

## Instructions for Claude

When this skill is invoked:

1. Determine the target `.tex` file:
   - If an argument is provided, use that path (resolve relative to the project root `c:\EEG\EPEC\EPEC_VS_code\SolarGeoRisk_EPEC_intertemporal`).
   - Otherwise use the file currently open in the IDE.

2. Run the script at `c:\EEG\EPEC\EPEC_VS_code\SolarGeoRisk_EPEC_intertemporal\add_vspace.py` on that file:
   ```bash
   python add_vspace.py "<target file>"
   ```
   (Run from the project root so relative paths resolve correctly.)

3. Also remove any `\setlength{\parindent}{0pt}` lines that appear directly under a `\section{...}` heading, as these cause paragraph breaks to render as bare linebreaks instead of indented paragraphs.

4. Report how many `\vspace` insertions were made and confirm the file was saved.
