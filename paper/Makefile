all: final.pdf

final.pdf: final.tex local.bib
	pdflatex final
	pdflatex final
	bibtex final
	pdflatex final

draft1.pdf: draft1.tex local.bib
	pdflatex draft1
	pdflatex draft1
	bibtex draft1
	pdflatex draft1

abstract.pdf: abstract.tex
	pdflatex abstract

view: final.pdf
	open -a Preview $^
