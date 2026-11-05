# Paper

- [Latest English PDF](transformer-pc-geometry-and-feature-readout.pdf), with author contact.
- [Markdown](manuscript.md), with matching content and GitHub links to supporting materials.
- `main.tex`, `body.tex`, `article-style.tex`, and `figures/`: typesetting sources and the ten original figures.

The 40-page PDF uses an A4 layout with a left-aligned title, blue headings, TeX Gyre Termes body text, and TeX Gyre Heros headings. The scientific prose, numerical tables, equations, ten figures, and compact Appendices A–C are unchanged. Links inside the PDF point to the corresponding files on GitHub, so the PDF can be read separately from a local checkout. The [reference guide](../references/README.md) supplies the detailed materials.

Run `sh paper/build_pdf.sh` from the repository root to rebuild the PDF. The script uses Tectonic when available, or XeLaTeX with the packages declared in `main.tex` and `article-style.tex`. Set `TECTONIC` to select a Tectonic executable explicitly. Only document typesetting is performed.
