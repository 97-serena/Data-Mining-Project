# Project report

Write each chapter in its own file in `chapters/`. The `%` comments are writing prompts. Set the authors and date in `thesis.tex`.

Install a TeX distribution with `latexmk` on your PATH.

**Terminal** — Run from the repository root.

```sh
latexmk -cd -pdf -outdir=build report/thesis.tex
```

**VS Code** — Open the repository, install **LaTeX Workshop**, and run **LaTeX Workshop: Build LaTeX project** from any chapter.

Both builds write `report/build/thesis.pdf`.
