# ====================================================================================
# Makefile for the EuroSAT-ResNet101 Project
#
# Provides commands for environment setup, data processing, training, evaluation,
# and Docker management.
# ====================================================================================

# --- Variables ---
# Use the Python interpreter from the virtual environment
PYTHON = .venv/bin/python
PIP = .venv/bin/pip
IMAGE_NAME = eurosat-resnet101

# --- Setup and Installation ---
.PHONY: setup
setup: ## Set up the Python virtual environment and install dependencies
	@echo "--> Setting up Python virtual environment in .venv..."
	python3 -m venv .venv
	@echo "--> Installing dependencies from requirements.txt..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "--> Setup complete. Activate with 'source .venv/bin/activate'"

# --- Core ML Pipeline ---
.PHONY: preprocess
preprocess: ## Preprocess the raw EuroSAT dataset into tensors
	@echo "--> Running data preprocessing script..."
	$(PYTHON) src/preprocess.py

.PHONY: train
train: ## Train the ResNet-101 model on the processed data
	@echo "--> Starting model training..."
	$(PYTHON) src/train.py

.PHONY: evaluate
evaluate: ## Evaluate the trained model and print performance metrics
	@echo "--> Evaluating model performance..."
	$(PYTHON) src/eval.py

.PHONY: visualize
visualize: ## Generate prediction visualizations and confusion matrix
	@echo "--> Generating visualizations..."
	$(PYTHON) src/visualize.py

.PHONY: all
all: preprocess train evaluate visualize ## 🏃 Run the entire pipeline: preprocess, train, evaluate, and visualize
	@echo "--> Full pipeline executed successfully."

# --- Docker Management ---
.PHONY: docker-build
docker-build: ## Build the Docker image for the project
	@echo "--> Building Docker image: $(IMAGE_NAME)..."
	docker build -t $(IMAGE_NAME) .

.PHONY: docker-shell
docker-shell: ## Start an interactive shell inside the Docker container
	@echo "--> Starting interactive shell in Docker container..."
	docker run -it --gpus all \
		-v "$(CURDIR)/data:/app/data" \
		-v "$(CURDIR)/artifacts:/app/artifacts" \
		-v "$(CURDIR)/assets:/app/assets" \
		$(IMAGE_NAME) bash

# --- Housekeeping ---
.PHONY: clean
clean: ## 🧹 Remove generated files and directories
	@echo "--> Cleaning up project..."
	rm -rf .venv __pycache__ */__pycache__ .pytest_cache runs
	rm -rf artifacts assets
	@echo "--> Project cleaned."

.PHONY: help
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

