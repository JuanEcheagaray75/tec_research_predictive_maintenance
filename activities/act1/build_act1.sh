#!/bin/sh
pdflatex base_review.tex
bibtex base_review
pdflatex base_review.tex