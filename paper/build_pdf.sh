#!/bin/sh
# Rebuild only the paper from its saved text and figures.
# Set TECTONIC to a Tectonic executable, or use XeLaTeX with the required packages.
set -eu
cd "$(dirname "$0")"
builddir=$(mktemp -d)
trap 'rm -rf "$builddir"' EXIT HUP INT TERM
if [ -n "${TECTONIC:-}" ]; then
  "$TECTONIC" -X compile --untrusted --outdir "$builddir" main.tex
elif command -v tectonic >/dev/null 2>&1; then
  tectonic -X compile --untrusted --outdir "$builddir" main.tex
else
  xelatex -interaction=nonstopmode -halt-on-error -output-directory="$builddir" main.tex
  xelatex -interaction=nonstopmode -halt-on-error -output-directory="$builddir" main.tex
fi
cp "$builddir/main.pdf" transformer-pc-geometry-and-feature-readout.pdf
