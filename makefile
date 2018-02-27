
currentTest:  test_test


# TESTS
all: test_linear_layer test_relu_layer test_dropoutLayer test_classifiers test_fcnet test_overfit_fcnet test_train_fcnet



test_linear_layer:
	python -m test.test_layers TestLinearLayer


test_relu_layer:
	python -m test.test_layers TestReLULayer


test_dropoutLayer:
	python -m test.test_layers TestDropoutLayer


test_classifiers:
	python -m test.test_classifiers

test_fcnet:
	python -m test.test_fcnet

test_overfit_fcnet:
	python -m src.overfit_fcnet

test_train_fcnet:
	python -m src.train_fcnet

test_test:
	python -m src.test


### MANUAL 
manual:
	google-chrome manuals/assignment2_advanced.pdf&

pdf:
	pandoc manuals/assignment2_advanced.md --pdf-engine=xelatex -o manuals/assignment2_advanced.pdf -V geometry:margin=1in --variable url	     color=cyan --template eisvogel --listings


### REPORT
REPORT_DISCARDED_OUTPUT = report.aux report.log report.out
REPORT_SRC = report/report.tex report/introduction.tex report/layersimplementation.tex \
             report/dropout.tex report/softmax.tex report/neuralnetworkcreation.tex \
             report/hyperparameter.tex report/conclusion.tex

report: report/report.pdf


report/report.pdf: $(GRAPHS_OBJECTS:.dot=.pdf)  $(REPORT_SRC) 
#	pdflatex -interaction=batchmode report/report.tex
	pdflatex report/report.tex
	rm -rf $(REPORT_DISCARDED_OUTPUT)
	mv report.pdf report

