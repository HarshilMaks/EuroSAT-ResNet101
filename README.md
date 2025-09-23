<p align="center"><h1 align="center">EUROSAT-ResNet101</h1></p>
<p align="center">
	<em>Geo-AI Engine for ESG & Supply Chain Monitoring using ResNet-101 on EuroSAT</em>
</p>

<p align="center">
	<img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat&logo=opensourceinitiative&logoColor=white&color=0062ff" alt="license">
	<img src="https://img.shields.io/badge/Language-Python-blue.svg?style=flat&color=0062ff" alt="language">
	<img src="https://img.shields.io/badge/Status-Active-green.svg?style=flat&color=0062ff" alt="status">
	<img src="https://img.shields.io/badge/Framework-PyTorch-red.svg?style=flat&color=0062ff" alt="framework">
	<img src="https://img.shields.io/badge/Validation%20Accuracy-82.4%25-brightgreen.svg?style=flat" alt="accuracy">
</p>

<p align="center">Built with:</p>
<p align="center">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">
	<img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white" alt="PyTorch">
	<img src="https://img.shields.io/badge/NumPy-013243.svg?style=flat&logo=NumPy&logoColor=white" alt="NumPy">
	<img src="https://img.shields.io/badge/pandas-150458.svg?style=flat&logo=pandas&logoColor=white" alt="pandas">
	<img src="https://img.shields.io/badge/scikitlearn-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white" alt="scikitlearn">
	<img src="https://img.shields.io/badge/Docker-2496ED.svg?style=flat&logo=Docker&logoColor=white" alt="Docker">
</p>

---

## Quick Links

- [1. Business Problem: Opaque Supply Chains and ESG Pressure](#1-business-problem-opaque-supply-chains-and-esg-pressure)
- [2. Enterprise Workflow](#2-enterprise-workflow)
- [3. Technical Architecture](#3-technical-architecture)
- [4. Experiments \& Evaluation](#4-experiments--evaluation)
  - [Dataset Preview](#dataset-preview)
  - [Confusion Matrix](#confusion-matrix)
  - [Sample Predictions](#sample-predictions)
- [5. Business Value](#5-business-value)
- [6. Use Cases](#6-use-cases)
- [7. Project Structure](#7-project-structure)
- [8. Getting Started](#8-getting-started)
  - [Dataset Download](#dataset-download)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Testing](#testing)
- [9. Project Roadmap](#9-project-roadmap)
- [10. Contributing](#10-contributing)
- [11. License](#11-license)
- [12. Acknowledgments](#12-acknowledgments)

---

## 1. Business Problem: Opaque Supply Chains and ESG Pressure

Enterprises today operate in a landscape of growing environmental and regulatory risk:

* **Regulatory Compliance** — Regulations such as the EU Deforestation Regulation require proof that imports (soy, palm oil, coffee, etc.) are not linked to deforestation.
* **Investor Scrutiny** — Investors use ESG metrics to evaluate long-term risk. A lack of transparency can impact valuation and funding access.
* **Operational Blind Spots** — Companies lack scalable monitoring for upstream risks (e.g., floods, fires, illegal land use), leading to costly disruptions.

Manual monitoring is not scalable. This project demonstrates a **Geo-AI engine** that enables automated, large-scale environmental intelligence.

---

## 2. Enterprise Workflow

```mermaid
graph TD
    A["1. Define AOIs (GeoJSON/KML)"] --> B{"2. Automated Satellite Data Ingestion"}
    B -->|"Sentinel-2, Planet, etc."| C["3. Core Analysis: Geo-AI Engine"]
    C -->|"Classifies Land Use"| D["4. Automated Change Detection"]
    D -->|"Compare Current vs Historical"| E{"5. Significant Change?"}
    E -->|"Yes (>1% Deforestation)"| F["6a. Trigger Real-Time Alert"]
    E -->|"No"| G["6b. Log for Audit"]
    F --> H["Risk Dashboard"]
    G --> I["Compliance & ESG Reports"]
```

---

## 3. Technical Architecture

<p align="center">
  <img src="assets/Images/resnet_architecture.png" width="600">
  <br>
  <em>ResNet-101 Architecture - Deep Residual Learning for Image Recognition</em>
</p>

* **Model**: ResNet-101 (custom implementation from scratch, inspired by [Microsoft's ResNet-101](https://huggingface.co/microsoft/resnet-101)).
* **Architecture**: 101-layer deep residual network with bottleneck blocks and skip connections.
* **Dataset**: EuroSAT (27k labeled images across 10 classes).
* **Pipeline**:

  * Data ingestion (Sentinel-2)
  * Preprocessing (resizing, normalization)
  * Augmentation (random crops, flips, rotations)
  * Training (PyTorch from scratch implementation)
  * Evaluation (confusion matrix, accuracy, F1-score)

---

## 4. Experiments & Evaluation

* **Training Setup**:

  * Optimizer: Adam (lr=1e-3, weight\_decay=1e-4)
  * Batch Size: 32
  * Epochs: 20
  * Architecture: ResNet-101 with custom classifier (dropout=0.2, hidden\_size=512)

* **Results**:

  * Validation Accuracy: **\~82.4%**
  * F1 Score: **\~0.82**
  * Model Size: \~44.5M parameters (only \~5K trainable with frozen backbone)

* **Analysis**:
  The ResNet-101 model demonstrates solid performance through deep residual learning. The confusion matrix shows good separation across all 10 EuroSAT categories, with strongest accuracy on agricultural and urban land use classes. Misclassifications mainly occur in visually similar categories such as pasture vs. herbaceous vegetation, which is expected given the complexity of satellite imagery.

---

### Dataset Preview

![EuroSAT Dataset](assets/eurosat_rgb_preview.png)
*Sample images from the EuroSAT RGB dataset, covering diverse land use categories.*

### Confusion Matrix

![Confusion Matrix](assets/eurosat_rgb_confusion_matrix.png)
*The confusion matrix illustrates correct and misclassified predictions across all 10 EuroSAT classes. Most misclassifications occur between visually similar land types.*

### Sample Predictions

![Predictions](assets/eurosat_rgb_predictions.png)
*Examples of model predictions on unseen EuroSAT images. The model reliably identifies the majority of classes, with occasional confusion among similar vegetation types.*

---

## 5. Business Value

This system provides:

* **Automated ESG Insights** — Enables supply chain monitoring with verifiable land-use classification.
* **Early Detection of Environmental Risk** — Flags changes such as deforestation or urban encroachment.
* **Scalable Monitoring** — Covers large geographic areas with minimal manual effort.
* **Reliable Analysis** — While the model achieves \~82% accuracy, it provides actionable intelligence suitable for decision-making and further human review.
* **Deep Learning Framework** — Leverages ResNet-101 for robust feature extraction from satellite imagery.

---

## 6. Use Cases

* Deforestation detection in palm oil or coffee supply chains.
* Flood monitoring for insurers and governments.
* ESG risk analysis for investment funds.
* Urban sprawl detection for smart city planning.
* Agricultural land classification for yield prediction.
* Multi-temporal change detection using deep residual features.

---

## 7. Project Structure

```
└── EuroSAT-ResNet101/
    ├── Dockerfile
    ├── LICENSE
    ├── Makefile
    ├── README.md
    ├── backend/
    │   └── app/
    │       ├── __init__.py
    │       ├── core/
    │       ├── main.py
    │       ├── routes/
    │       └── utils/
    ├── notebooks/
    │   └── exploration.ipynb
    ├── requirements.txt
    └── src/
        ├── dataset.py
        ├── eval.py
        ├── main.py
        ├── model.py
        ├── preprocess.py
        ├── train.py
        └── utils.py
```

---

## 8. Getting Started

### Dataset Download

The RGB version of EuroSAT dataset is available on Kaggle:
[Kaggle: EuroSAT Dataset](https://www.kaggle.com/datasets/raoofnaushad/eurosat-sentinel2-dataset)

### Prerequisites

* Python 3.8+
* PyTorch 1.9+
* CUDA-compatible GPU (recommended)
* 8GB+ RAM

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/HarshilMaks/EuroSAT-ResNet101
cd EuroSAT-ResNet101
pip install -r requirements.txt
```

Using Docker:

```bash
docker build -t eurosat-resnet101 .
```

### Usage

Run the main application:

```bash
python src/main.py
```

Train the ResNet-101 model:

```bash
python src/train.py
```

Using Docker:

```bash
docker run -it eurosat-resnet101
```

### Testing

Run tests to verify the setup:

```bash
pytest
```

---

## 9. Project Roadmap

* [x] Data ingestion pipeline
* [x] ResNet-101 from scratch implementation
* [x] Training & evaluation pipeline
* [ ] Multi-temporal change detection with ResNet-101
* [ ] Real-time inference API (FastAPI backend)
* [ ] Dashboard integration for ESG reporting
* [ ] Model optimization and quantization
* [ ] Transfer learning experiments

---

## 10. Contributing

* [Join Discussions](https://github.com/HarshilMaks/EuroSAT-ResNet101/discussions)
* [Report Issues](https://github.com/HarshilMaks/EuroSAT-ResNet101/issues)
* [Submit Pull Requests](https://github.com/HarshilMaks/EuroSAT-ResNet101/pulls)

Steps:

1. Fork repo
2. Clone locally
3. Create feature branch
4. Commit & push
5. Open PR

---

## 11. License

This project is licensed under the [Apache 2.0 License](LICENSE).

---

## 12. Acknowledgments

* [EuroSAT Dataset](https://github.com/phelber/EuroSAT)
* [Microsoft ResNet-101](https://huggingface.co/microsoft/resnet-101) for architectural inspiration
* PyTorch team for the deep learning framework
* Sentinel-2 open satellite data

This work builds upon:

> P. Helber, B. Bischke, A. Dengel, D. Borth, "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification," IEEE JSTARS, vol. 12, no. 7, pp. 2217-2226, July 2019.

> K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," CVPR, 2016.

> O. Adedeji, P. Owoade, O. Ajayi, O. Arowolo, "Image Augmentation for Satellite Images," arXiv:2207.14580, 2022.
