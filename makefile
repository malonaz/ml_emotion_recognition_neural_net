
default: testLinearLayer TestReLULayer

pdf:
	pandoc manuals/assignment2_advanced.md --pdf-engine=xelatex -o manuals/assignment2_advanced.pdf -V geometry:margin=1in --variable urlcolor=cyan --template eisvogel --listings

.PHONY: manual

testLinearLayer:
	python -m test.test_layers TestLinearLayer



TestReLULayer:
	python -m test.test_layers TestReLULayer


manual:
	google-chrome manuals/assignment2_advanced.pdf&
