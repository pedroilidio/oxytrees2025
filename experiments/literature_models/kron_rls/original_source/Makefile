# This makefile runs all tests, then it builds the paper

OCTAVE = octave -qf
MAKE_TEX_TABLE = runhaskell MakeTexTable.hs

.PHONY: all all-tests weight-tests paper new-interactions
.DELETE_ON_ERROR:
all: all-tests weight-tests paper

results:
	mkdir -p $@

.PHONY: all-tests
all-tests: results/all-tests-loo.out results/all-tests-10cv.out
results/all-tests-loo.out:
	cd src && $(OCTAVE) -q --eval "run_all_tests(1);" > ../$@
results/all-tests-10cv.out:
	cd src && $(OCTAVE) -q --eval "run_all_tests(0);" > ../$@


.PHONY: paper/results
paper/results: paper/results-loo-small.tex paper/results-loo-full.tex paper/results-10cv-full.tex
paper/results-loo-small.tex: results/all-tests-loo.out results/results-by09.txt
	perl scripts/make-results-table.pl 0 $^ > $@
paper/results-loo-full.tex: results/all-tests-loo.out
	perl scripts/make-results-table.pl 1 $^ > $@
paper/results-10cv-full.tex: results/all-tests-10cv.out
	perl scripts/make-results-table.pl 1 $^ > $@


.PHONY: weight-tests
weight-tests: results/weight-tests-score-e.mat
results/weight-tests-score-e.mat:
	cd src && $(OCTAVE) -qf run_weight_tests.m


.PHONY: new-interactions new-interactions-html new-interactions-latex new-interactions-summary
new-interactions: $(foreach x,nr gpcr ic e,results/new-interactions-$(x).txt)
results/new-interactions-%.txt:
	cd src && octave -q --eval "find_new_interactions('$*',1000)" > ../$@

new-interactions-html: $(foreach x,nr gpcr ic e,results/new-interactions-$(x).html)
results/new-interactions-%.html: results/new-interactions-%.txt scripts/interactions-info.pl
	perl scripts/interactions-info.pl --mode=html < $< > $@
new-interactions-html: $(foreach x,nr gpcr ic e,results/new-interactions-$(x)-surprise.html)
results/new-interactions-%-surprise.html: results/new-interactions-%.txt scripts/interactions-info.pl
	perl scripts/interactions-info.pl --mode=html --only-surprising < $< > $@
new-interactions-html: $(foreach x,nr gpcr ic e,results/by09-interactions-$(x).html)
results/by09-interactions-%.html: results/by09_predpairs_def_max_%.txt scripts/interactions-info.pl
	perl scripts/interactions-info.pl --mode=html --max-rank=1000 < $< > $@
new-interactions-latex: $(foreach x,nr gpcr ic e,paper/new-interactions-$(x)-50.tex paper/new-interactions-$(x)-20.tex paper/new-interactions-$(x)-10.tex paper/new-interactions-$(x)-wide.tex)

paper/new-interactions-%-50.tex: results/new-interactions-%.txt scripts/interactions-info.pl
	perl scripts/interactions-info.pl --max-rank=50 --mode=latex < $< > $@
paper/new-interactions-%-20.tex: results/new-interactions-%.txt scripts/interactions-info.pl
	perl scripts/interactions-info.pl --max-rank=20 --mode=latex < $< > $@
paper/new-interactions-%-10.tex: results/new-interactions-%.txt scripts/interactions-info.pl
	perl scripts/interactions-info.pl --max-rank=10 --mode=latex < $< > $@
paper/new-interactions-%-wide.tex: results/new-interactions-%.txt scripts/interactions-info.pl
	perl scripts/interactions-info.pl --max-rank=20 --mode=latex-wide < $< > $@
new-interactions-raw: $(foreach x,nr gpcr ic e,results/new-interactions-$(x).raw)
results/new-interactions-%.raw: results/new-interactions-%.txt scripts/interactions-info.pl
	perl scripts/interactions-info.pl --mode=raw < $< > $@

new-interactions-summary: paper/new-interactions-summary.tex paper/by09-interactions-summary.tex
paper/new-interactions-summary.tex: $(foreach x,e ic gpcr nr,results/new-interactions-$(x).txt) scripts/interactions-info.pl Makefile
	perl scripts/interactions-info-summary.pl $^ > $@
paper/by09-interactions-summary.tex: $(foreach x,e ic gpcr nr,results/by09_predpairs_def_max_$(x).txt) scripts/interactions-info.pl Makefile
	perl scripts/interactions-info-summary.pl $^ > $@

.PHONY: new-interactions-diff
.INTERMEDIATE: results/*.raw-tsv
new-interactions-diff: $(foreach x,nr gpcr ic e,paper/interactions-diff-$(x).tex)
paper/interactions-diff-%.tex: results/new-interactions-%.raw-tsv results/by09-interactions-%.raw-tsv
	perl scripts/set-diff-latex.pl $^ > $@
results/new-interactions-%.raw-tsv: results/new-interactions-%.txt
	perl scripts/interactions-info.pl --mode=raw-tsv --max-rank=50 --only-confirmed $^ > $@
results/by09-interactions-%.raw-tsv: results/by09_predpairs_def_max_%.txt
	perl scripts/interactions-info.pl --mode=raw-tsv --max-rank=50 --only-confirmed $^ > $@


.PHONY: paper
paper: paper/results paper/paper.pdf
paper/paper.pdf: paper/paper.tex paper/drug_target.bib paper/results new-interactions-latex new-interactions-summary
	cd paper && \
	pdflatex paper -interaction=batchmode && \
	bibtex paper && \
	pdflatex paper -interaction=batchmode && \
	bibtex paper && \
	pdflatex paper -interaction=batchmode

paper/results.tex:
	# TODO
