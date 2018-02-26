
currentTest: test_overfit_fcnet

all: test_linear_layer test_relu_layer test_dropoutLayer test_classifiers test_fcnet

pdf:
	pandoc manuals/assignment2_advanced.md --pdf-engine=xelatex -o manuals/assignment2_advanced.pdf -V geometry:margin=1in --variable urlcolor=cyan --template eisvogel --listings



.PHONY: manual 

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

manual:
	google-chrome manuals/assignment2_advanced.pdf&
