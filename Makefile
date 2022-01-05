NAME=all-of-stats-solns
SRC_DIR=src
TEX_DIR=src/tex
CODEDIR=src/code

BUILDDIR = build
SRCFILES = $(shell ls src/code/*.py)

PDFLATEX=pdflatex --shell-escape %S

.PHONY: dir clean code

all: dir $(NAME).pdf

dir:
	mkdir -p src/output
	mkdir -p src/images

code: dir
	find src/code/*.py -print0 | xargs -0 -P 4 -I{} sh -c 'wp=$${1##*/}; s=$${wp%.*}; python $$1 > src/output/$$s.txt' -- {}
	find src/output/ -maxdepth 1 -type f -empty -delete
	sed -i -e 's/^/> /' src/output/*.txt
	mv *.png src/images

clean:
	rm -f $(TEX_DIR)/*.aux $(TEX_DIR)/*.log $(TEX_DIR)/*.fls $(TEX_DIR)/*.toc
	rm -f $(TEX_DIR)/*.fdb_latexmk $(TEX_DIR)/*.pdf
	rm -rf $(TEX_DIR)/_minted-main
	rm -rf $(CODEDIR)/__pycache__

$(NAME).pdf: $(SRC_DIR)/*
	cd $(SRC_DIR)/tex \
	 && latexmk -pdf -pdflatex="${PDFLATEX}" -use-make "main.tex" \
	 && cp main.pdf ../../$(NAME).pdf
