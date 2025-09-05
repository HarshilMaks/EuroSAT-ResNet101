<div align="center">
  <a href="https://github.com/phelber/EuroSAT">
    <img src="https://github.com/HarshilMaks/EuroSAT-ResNet50/blob/main/EuroSAT_sample.jpg?raw=true" alt="EuroSAT Sample Images" width="800"/>
  </a>

  # Geo-AI Engine for Automated ESG & Supply Chain Monitoring

  **A high-performance geospatial intelligence engine designed to help enterprises de-risk global supply chains, automate environmental compliance reporting, and monitor critical assets. This system utilizes a fine-tuned ResNet-50 model on the EuroSAT benchmark, providing a powerful foundation for real-world environmental and operational intelligence.**

  [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
  [![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

</div>

---

## 1. The Business Problem: Opaque Supply Chains & Mounting ESG Pressure

Modern enterprises face significant operational, financial, and reputational risk from opaque global supply chains. Key challenges include:

*   **Regulatory Compliance:** New regulations (e.g., the EU Deforestation Regulation) require companies to prove their imported commodities (palm oil, soy, coffee) are not linked to deforestation. Non-compliance results in heavy fines and market access restrictions.
*   **Investor Demands:** Investors increasingly use Environmental, Social, and Governance (ESG) metrics to evaluate risk. A lack of transparency in a company's environmental footprint can negatively impact its valuation and access to capital.
*   **Operational Blindness:** Without timely intelligence, companies cannot effectively monitor upstream assets (farms, forests, mines) for risks like floods, fires, or unauthorized land use changes that can disrupt operations.

Manual monitoring is unscalable and cost-prohibitive. This project provides the core analytical engine for an automated, scalable solution.

## 2. Enterprise Application Workflow

This engine is designed to be integrated into a larger enterprise monitoring system. The typical operational workflow transforms raw data into actionable alerts and reports.

```mermaid
graph TD
    A[1. Define Areas of Interest (AOIs) via GeoJSON/KML] --> B{2. Automated Satellite Data Ingestion};
    B -->|Sentinel-2, Planet, etc.| C[3. Core Analysis: Geo-AI Engine];
    C -->|Classifies Land Use (Forest, Crop, etc.)| D[4. Automated Change Detection];
    D -->|Compares Current vs. Previous State| E{5. Change Verified?};
    E -->|Yes (>1% Deforestation)| F[6a. Trigger Real-Time Alert];
    E -->|No| G[6b. Log 'No Change' for Audit];
    F --> H[Risk Management Dashboard];
    G --> I[Compliance & ESG Reports];
```

## 3. Technical Architecture

### Core Model: Fine-Tuned ResNet-50
The model is a **ResNet-50** architecture, pre-trained on ImageNet, and subsequently fine-tuned for the LULC classification task. This transfer learning approach leverages powerful, pre-existing feature hierarchies, allowing for high accuracy with efficient training.

### Benchmark Dataset: EuroSAT
The model was trained on **EuroSAT**, the established benchmark for satellite-based land use classification. It consists of 27,000 labeled and geo-referenced images from the **Sentinel-2 satellite**, providing a robust foundation for the model's capabilities.

The 10 classes represent critical land cover types for ESG and supply chain analysis:
*   `Forest` vs. `AnnualCrop` / `Pasture`: Directly detects deforestation.
*   `River` / `SeaLake`: Monitors water levels for flood risk assessment.
*   `Industrial` / `Residential`: Tracks encroachment on protected lands.

### Performance Enhancement: Data Augmentation
To improve model generalization and robustness against variations in satellite imagery, data augmentation is employed. As demonstrated by recent research on this exact dataset ([Adedeji et al., 2022](https://arxiv.org/abs/2207.14580)), augmentation is a critical step. Our approach uses geometric augmentations, a proven technique for enhancing performance in satellite image analysis.

<div align="center">
  <img src="https://raw.githubusercontent.com/satellite-image-deep-learning/techniques/main/augmentation/figures/geometric.png" alt="Data Augmentation" width="500"/>
  <br>
  <em>Example of geometric augmentations (flipping, rotation) to create robust models.</em>
</div>

## 4. Model Performance
The engine achieves a high degree of accuracy, providing a reliable basis for automated monitoring systems.

*   **Validation Accuracy:** **~94%**

For a detailed analysis, a confusion matrix should be generated to evaluate class-specific performance and identify potential areas for further tuning, such as differentiating between `PermanentCrop` and `Forest`.

<div align="center">
  <em>(Placeholder for Confusion Matrix Image)</em>
  <br>
  <strong>A confusion matrix is essential to visualize model performance on a per-class basis.</strong>
</div>

## 5. Getting Started

### Prerequisites
This project requires Python 3.8+ and several core libraries. Ensure your environment is set up with the following:

*   **PyTorch & Torchvision:** For model creation and training.
*   **NumPy:** For numerical operations.
*   **Matplotlib:** For data visualization (e.g., plotting metrics).
*   **Pillow (PIL):** For image processing.

### Installation & Data
1.  **Clone the Repository**
    ```bash
    git clone https://github.com/HarshilMaks/EuroSAT-ResNet50.git
    cd EuroSAT-ResNet50
    ```
2.  **Download the EuroSAT Dataset**
    The RGB version of the dataset is available from multiple sources. We recommend the version hosted on Kaggle for ease of use:
    [Kaggle: EuroSAT Dataset](https://www.kaggle.com/datasets/raoofnaushad/eurosat-sentinel2-dataset)

3.  **Install Dependencies**
    It is highly recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

## 6. Usage

*   **Model Training:**
    ```bash
    python train.py --data_dir /path/to/EuroSAT_RGB --epochs 20
    ```
*   **Inference:**
    ```bash
    python predict.py --model_path /path/to/model.pth --image_path /path/to/asset_image.jpg
    ```

## 7. Next Steps & Commercialization Path

This engine is a powerful proof-of-concept. To develop it into a full enterprise solution, the following steps are recommended:

*   **Develop a Change Detection Module:** Implement algorithms to compare classified images over time and quantify changes.
*   **Build an Alerting API:** Create an endpoint to integrate with enterprise systems (e.g., Slack, email, BI tools).
*   **Create a BI Dashboard:** Develop a user-facing dashboard to visualize asset locations, time-series changes, and generate compliance reports.

## Contributing

Contributions focusing on enhancing model accuracy, improving inference speed, or building out the enterprise workflow components are highly welcome. Please open an issue to discuss your ideas.

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.

## Acknowledgments & Citations

*   This work is built upon the foundational **EuroSAT dataset** provided by:
    > P. Helber, B. Bischke, A. Dengel, D. Borth, "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 12, no. 7, pp. 2217-2226, July 2019.

*   Our methodology is informed by recent advancements in data augmentation for satellite imagery, as explored in:
    > O. Adedeji, P. Owoade, O. Ajayi, O. Arowolo, "Image Augmentation for Satellite Images," arXiv preprint arXiv:2207.14580, 2022.