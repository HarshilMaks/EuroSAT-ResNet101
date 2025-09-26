<h1 align="center">EUROSAT-ResNet101</h1>
<p align="center">
<em>Geo-AI Engine for ESG & Supply Chain Monitoring using ResNet-101 on EuroSAT</em>
</p>
<p align="center">
<img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat&logo=opensourceinitiative&logoColor=white&color=0062ff" alt="license">
<img src="https://img.shields.io/badge/Language-Python-blue.svg?style=flat&color=0062ff" alt="language">
<img src="https://img.shields.io/badge/Status-Active-green.svg?style=flat&color=0062ff" alt="status">
<img src="https://img.shields.io/badge/Framework-PyTorch-red.svg?style=flat&color=0062ff" alt="framework">
<img src="https://img.shields.io/badge/Overall%20Accuracy-82.35%25-brightgreen.svg?style=flat" alt="accuracy">
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

Quick Links
-----------
- [1\. Business Problem](#1-business-problem-opaque-supply-chains-and-esg-pressure)
- [2\. Enterprise Workflow](#2-enterprise-workflow)
- [3\. Technical Architecture](#3-technical-architecture)
- [4\. Experiments & Evaluation](#4-experiments--evaluation)
- [5\. Business Value](#5-business-value)
- [6\. Use Cases](#6-use-cases)
- [7\. Project Structure](#7-project-structure)
- [8\. Getting Started](#8-getting-started)
- [9\. Project Roadmap](#9-project-roadmap)
- [10\. Contributing](#10-contributing)
- [11\. License](#11-license)
- [12\. Acknowledgments](#12-acknowledgments)

1. Business Problem: Opaque Supply Chains and ESG Pressure
-----------------------------------------------------------
Enterprises today operate in a landscape of growing environmental and regulatory risk:

- **Regulatory Compliance** — Regulations such as the EU Deforestation Regulation require proof that imports (soy, palm oil, coffee, etc.) are not linked to deforestation.
- **Investor Scrutiny** — Investors use ESG metrics to evaluate long-term risk. A lack of transparency can impact valuation and funding access.
- **Operational Blind Spots** — Companies lack scalable monitoring for upstream risks (e.g., floods, fires, illegal land use), leading to costly disruptions.

Manual monitoring is not scalable. This project demonstrates a **Geo-AI engine** that enables automated, large-scale environmental intelligence.

2. Enterprise Workflow
-----------------------
<details><summary>View Enterprise Workflow (Mermaid Diagram)</summary>

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
````

</details>

3. Technical Architecture

---

<p align="center">
<img src="assets/Images/resnet_architecture.png" width="600">
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

4. Experiments & Evaluation

---

* **Training Setup**:

  * Optimizer: AdamW
  * Batch Size: 32
  * Total Epochs: 50 (15 + 20 + 15 in 3 phases)
  * Architecture: ResNet-101 with custom classifier (dropout=0.3, hidden_size=1024)
* **Results**:

  * Overall Accuracy: **82.35%**
  * Weighted Avg F1-Score: **0.820**
  * Model Size: ~44.6M parameters (fully trainable in the final phase)
* **Analysis**:

  The model achieves a solid 82.35% overall accuracy and a weighted F1-score of 0.820. Performance is exceptionally strong for classes with distinct visual features, such as SeaLake (F1: 0.956), Industrial (F1: 0.932), and Residential (F1: 0.924), all of which benefit from high precision and recall.

  The primary challenge lies in distinguishing between spectrally similar classes. For instance, **HerbaceousVegetation** has a low F1-score (0.689) driven by poor recall (0.579), indicating the model often misclassifies it as other vegetation types like Pasture or Forest. Similarly, **Highway** (F1: 0.674) suffers from low recall (0.614), as it is frequently confused with River or surrounding AnnualCrop land. These results highlight opportunities for improvement, such as targeted data augmentation or employing attention mechanisms to help the model focus on more subtle distinguishing features.

### Dataset Preview

*Sample images from the EuroSAT RGB dataset, covering diverse land use categories.*

### Confusion Matrix

*The confusion matrix illustrates correct and misclassified predictions across all 10 EuroSAT classes. Most misclassifications occur between visually similar land types.*

### Sample Predictions

*Examples of model predictions on unseen EuroSAT images. The model reliably identifies the majority of classes, with occasional confusion among similar vegetation types.*

5. Business Value

---

* **Automated ESG Insights** — Verifiable land-use classification for supply chain monitoring.
* **Early Risk Detection** — Timely alerts for deforestation, flooding, or encroachment.
* **Scalable Monitoring** — Wide geographic coverage with minimal manual effort.
* **Reliable Analysis** — ~82% accuracy provides actionable intelligence for decision support.

6. Use Cases

---

* Deforestation detection in palm oil or coffee supply chains.
* Flood monitoring for insurers and governments.
* ESG risk analysis for investment funds.
* Urban sprawl detection for smart city planning.
* Agricultural land classification for yield prediction.
* Multi-temporal change detection using deep residual features.

7. Project Structure

---

```text
└── EuroSAT-ResNet101/
    ├── Dockerfile
    ├── LICENSE
    ├── README.md
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

8. Getting Started

---

### Dataset Download

The RGB version of EuroSAT dataset is available on Kaggle:
[EuroSAT RGB Dataset on Kaggle](https://www.kaggle.com/datasets/phelber/eurosat)

### Prerequisites

* Python 3.8+
* PyTorch 1.9+
* CUDA-compatible GPU (recommended)
* 8GB+ RAM

### Installation

```bash
git clone https://github.com/HarshilMaks/EuroSAT-ResNet101.git
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
docker run -it --gpus all \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$(pwd)/assets:/app/assets" \
  eurosat-resnet101
```

### Testing

Run tests to verify the setup:

```bash
pytest
```

9. Project Roadmap

---

* [x] Data ingestion pipeline
* [x] ResNet-101 from scratch implementation
* [x] Training & evaluation pipeline
* [ ] Multi-temporal change detection with ResNet-101
* [ ] Real-time inference API (FastAPI backend)
* [ ] Dashboard integration for ESG reporting
* [ ] Model optimization and quantization
* [ ] Transfer learning experiments

10. Contributing

---

* [Join Discussions](https://github.com/HarshilMaks/EuroSAT-ResNet101/discussions)
* [Report Issues](https://github.com/HarshilMaks/EuroSAT-ResNet101/issues)
* [Submit Pull Requests](https://github.com/HarshilMaks/EuroSAT-ResNet101/pulls)

Steps:

1. Fork repo

2. Clone locally

3. Create feature branch

4. Commit & push

5. Open PR

6. License

---

This project is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

12. Acknowledgments

---

* [EuroSAT Dataset](https://github.com/phelber/EuroSAT)
* [Microsoft ResNet-101](https://huggingface.co/microsoft/resnet-101) for architectural inspiration
* PyTorch team for the deep learning framework
* Sentinel-2 open satellite data

This work builds upon:

> P. Helber, B. Bischke, A. Dengel, D. Borth, "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification," IEEE JSTARS, vol. 12, no. 7, pp. 2217-2226, July 2019.
>
> K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," CVPR, 2016.
>
> O. Adedeji, P. Owoade, O. Ajayi, O. Arowolo, "Image Augmentation for Satellite Images," arXiv:2207.14580, 2022.