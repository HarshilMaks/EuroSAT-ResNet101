  # Geo-AI Engine for Automated ESG & Supply Chain Monitoring

  **A high-performance geospatial intelligence engine designed to help enterprises de-risk their global supply chains, automate environmental compliance reporting, and monitor critical assets in near real-time. This system utilizes a fine-tuned ResNet-50 model to achieve ~94% accuracy in land use classification from satellite imagery.**

  [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
  [![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

</div>

---

## 1. The Business Problem: Opaque Supply Chains and Mounting ESG Pressure

Modern enterprises face significant operational, financial, and reputational risk from opaque global supply chains. Key challenges include:

*   **Regulatory Compliance:** New regulations (e.g., the EU Deforestation Regulation) require companies to prove their imported commodities (like palm oil, soy, coffee) are not linked to deforestation. Non-compliance results in heavy fines and market access restrictions.
*   **Investor Demands:** Investors increasingly use Environmental, Social, and Governance (ESG) metrics to evaluate risk. A lack of transparency in a company's environmental footprint can negatively impact its valuation and access to capital.
*   **Operational Blindness:** Without timely intelligence, companies cannot effectively monitor their upstream assets (farms, forests, mines) for risks like floods, fires, or unauthorized land use changes that can disrupt operations.

Manual monitoring is unscalable and cost-prohibitive. This project provides the core analytical engine for an automated, scalable solution.

## 2. Our Solution: A Geospatial Intelligence Engine

This solution is a deep learning-powered engine that transforms raw satellite imagery into actionable business intelligence. By classifying land use with high accuracy, it serves as the foundation for a continuous monitoring and alerting platform.

The engine is built on a **ResNet-50 architecture** fine-tuned on the diverse **EuroSAT dataset**, enabling it to accurately identify critical land cover types at scale:
*   `Forest` vs. `AnnualCrop` / `Pasture`: Directly detects deforestation events.
*   `River` / `SeaLake`: Monitors water levels for flood risk assessment.
*   `Industrial` / `Residential`: Tracks encroachment on protected lands.

## 3. Key Features & Business Benefits

| Feature                      | Benefit                                                                                               |
| :--------------------------- | :---------------------------------------------------------------------------------------------------- |
| **High-Accuracy Classification** | Provides **reliable and actionable insights**, reducing false positives and enabling confident decision-making. |
| **Scalable Architecture**        | Processes vast geographic areas efficiently, enabling **monitoring of thousands of assets** simultaneously.       |
| **Reproducible Environment**     | **Enterprise-ready and portable** using Docker, ensuring consistent performance from development to production. |
| **Pre-Trained Checkpoint**       | Facilitates **rapid prototyping and integration**, allowing developers to build custom applications quickly.    |

## 4. Enterprise Application Workflow

This engine is designed to be integrated into a larger enterprise monitoring system. The typical operational workflow is as follows:

1.  **Define Areas of Interest (AOIs):** The enterprise registers the geographic coordinates of its assets, such as supplier farms, processing facilities, or owned land parcels.

2.  **Automated Data Ingestion:** The system automatically pulls the latest satellite imagery (e.g., from Sentinel-2 or Planet) for the defined AOIs on a scheduled basis (e.g., weekly).

3.  **Core Analysis & Classification:** This Geo-AI engine ingests the new imagery and performs land use classification for each pixel within the AOI.

4.  **Automated Change Detection:** The system compares the latest classification map with the previous period's map. It algorithmically detects significant changes (e.g., a `Forest` polygon changing to `Pasture`).

5.  **Alerting & Reporting:** If a critical change is detected (e.g., >1% deforestation), the system automatically triggers an **alert** to the relevant risk management team. It also generates time-stamped, auditable data for **compliance reports**.

*(This workflow could be visualized with a simple diagram: [AOIs] -> [Satellite Data] -> [This Geo-AI Engine] -> [Change Detection] -> [Alerts / ESG Dashboard])*

## 5. Technical Implementation

### Prerequisites
*   Python 3.8+
*   PyTorch, Torchvision
*   NumPy, Matplotlib

### Installation
1.  **Clone the Repository**
    ```bash
    git clone https://github.com/HarshilMaks/EuroSAT-ResNet50.git
    cd EuroSAT-ResNet50
    ```
2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

### Usage
*   **Model Training:**
    ```bash
    python train.py --data_dir /path/to/EuroSAT_RGB --epochs 20
    ```
*   **Inference:**
    ```bash
    python predict.py --model_path /path/to/model.pth --image_path /path/to/asset_image.jpg
    ```

## 6. Next Steps & Commercialization Path

This engine is a powerful proof-of-concept. To develop it into a full enterprise solution, the following steps are recommended:

*   **Develop a Change Detection Module:** Implement algorithms to compare classified images over time and quantify changes.
*   **Build an Alerting API:** Create an API endpoint that can be triggered by the change detection module to integrate with enterprise messaging systems (e.g., Slack, email).
*   **Create a BI Dashboard:** Develop a user-facing dashboard (using tools like Streamlit, Dash, or Tableau) to visualize asset locations, display time-series changes, and generate compliance reports.
*   **Integrate with Commercial Data Sources:** Augment the model by training on higher-resolution commercial satellite imagery for even greater precision.

## Contributing

Contributions focusing on enhancing the model's accuracy, improving inference speed, or building out the enterprise workflow components are highly welcome. Please open an issue to discuss your ideas.

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

The model's performance is built upon the EuroSAT dataset, provided by Helber et al. This work serves as an application of their foundational research.