
currentTest: test_random_search_optim


# TESTS
all: test_linear_layer test_relu_layer test_dropoutLayer test_classifiers test_fcnet test_overfit_fcnet test_train_fcnet test_optimize_hyperparameters test_bayesian_optim



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

test_grid_search_optim:
	python -m src.optimizers.grid_search

test_random_search_optim:
	python -m src.optimizers.random_search

test_bayesian_optim:
	python -m src.gaussian_process.gp

### MANUAL 
manual:
	google-chrome manuals/assignment2_advanced.pdf&

pdf:
	pandoc manuals/assignment2_advanced.md --pdf-engine=xelatex -o manuals/assignment2_advanced.pdf -V geometry:margin=1in --variable url	     color=cyan --template eisvogel --listings


