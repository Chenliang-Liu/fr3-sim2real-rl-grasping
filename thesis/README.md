# Thesis materials

- `robot-rl-thesis.pdf`: the supplied final submitted thesis, copied from the project-root PDF without re-typesetting.
- `robot-rl-thesis.tex` and `robot-rl-thesis.bib`: the supplied LaTeX source and bibliography. The source describes itself as a draft, and its metadata can differ from the submitted PDF; the PDF is the final reference.
- PNG figures and `nju-*.pdf` artwork: supporting assets supplied with the thesis source.
- `njuthesis.cls` and `njuthesis-*.def`: the supplied Nanjing University thesis template files, with their original notices retained.
- `LICENSE-LPPL.txt`: the template's supplied LaTeX Project Public License text.
- `njuthesis.dtx`: corresponding upstream template source from the official [NJUThesis v1.4.3 release](https://github.com/nju-lug/NJUThesis/blob/v1.4.3/source/njuthesis.dtx), added with original notices retained.

The source specifies the `njuthesis` class, XeLaTeX, and Biber. From this directory, with an appropriate TeX installation and template dependencies:

```bash
xelatex robot-rl-thesis.tex
biber robot-rl-thesis
xelatex robot-rl-thesis.tex
xelatex robot-rl-thesis.tex
```

Compilation has not been verified during repository preparation. Template-generated supporting files use LPPL-1.3c-or-later; this does not set the license of the thesis text, figures, videos, or research Python code. The separate reference theses in the original folder are not redistributed.
