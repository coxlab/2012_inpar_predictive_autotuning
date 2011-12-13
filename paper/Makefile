all: abstract.pdf draft1.pdf

draft1.pdf: draft1.tex local.bib
	pdflatex draft1

abstract.pdf: abstract.tex
	pdflatex abstract

view: draft1.pdf
	open $^
