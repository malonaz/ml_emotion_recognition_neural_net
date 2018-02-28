### params
EXE = report.pdf
COMPILER = pdflatex
FLAGS = --interaction=batchmode -output-directory pdf
DIR = src

# get src objects
REPORT_OBJS = $(wildcard $(DIR)/*.tex)

# compile main
$(EXE): $(REPORT_OBJS)
	$(COMPILER) $(FLAGS) $(DIR)/$(@:.pdf=.tex)


.PHONY: clean open


open:
	google-chrome $(DIR)/$(EXE) &

clean:
	rm -rf pdf/*
