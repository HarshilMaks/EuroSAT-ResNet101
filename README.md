# EuroSAT Land Use & Cover Classification with ResNet-50

[![License: ](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

This project presents a high-accuracy land use and land cover classification model built by fine-tuning a ResNet-50 architecture on the EuroSAT dataset. It achieves **~94% validation accuracy** in classifying satellite imagery into 10 distinct categories, offering a robust solution for environmental monitoring, urban planning, and agricultural applications.

![Sample EuroSAT Images](https://github.com/HarshilMaks/EuroSAT-ResNet50/blob/main/EuroSAT_sample.jpg?raw=true)

## Table of Contents

- [EuroSAT Land Use \& Cover Classification with ResNet-50](#eurosat-land-use--cover-classification-with-resnet-50)
  - [Table of Contents](#table-of-contents)
  - [The Problem: Why Land Cover Classification Matters](#the-problem-why-land-cover-classification-matters)
  - [Our Solution: Fine-Tuned ResNet-50](#our-solution-fine-tuned-resnet-50)
  - [Key Features](#key-features)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Usage](#usage)
    - [Training the Model](#training-the-model)
    - [Running Inference](#running-inference)
  - [Model Performance](#model-performance)
  - [Real-World Use Cases](#real-world-use-cases)
  - [Project Structure](#project-structure)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)
    - [Suggestion for a Real-World Use Case](#suggestion-for-a-real-world-use-case)

## The Problem: Why Land Cover Classification Matters

Understanding how land is used and what covers its surface is crucial for addressing some of the world's most pressing challenges. From monitoring the effects of climate change to ensuring food security, Land Use and Land Cover (LULC) classification from satellite imagery provides critical data for:

*   **Environmental Monitoring:** Tracking deforestation, monitoring water bodies, and detecting the impact of natural disasters.
*   **Urban Planning:** Analyzing urban sprawl, managing infrastructure development, and ensuring sustainable city growth.
*   **Agricultural Management:** Classifying crop types, monitoring crop health, and optimizing land use for food production.

## Our Solution: Fine-Tuned ResNet-50

This project leverages a pre-trained ResNet-50 model, a powerful deep convolutional neural network, and fine-tunes it on the **EuroSAT dataset**. The EuroSAT dataset contains 27,000 labeled 64x64 pixel satellite images from the Sentinel-2 satellite, categorized into 10 classes:

*   Annual Crop
*   Forest
*   Herbaceous Vegetation
*   Highway
*   Industrial
*   Pasture
*   Permanent Crop
*   Residential
*   River
*   Sea & Lake

By fine-tuning only the final classification layer, we adapt the powerful feature extraction capabilities of ResNet-50 to this specific task, achieving high accuracy with minimal training time.

## Key Features

*   **High Accuracy:** Achieves approximately 94% accuracy on the validation set.
*   **Efficient Training:** Utilizes transfer learning for rapid model development.
*   **Reproducible:** Comes with a `Makefile` and `Docker` configuration for easy setup and consistent results.
*   **Extensible:** The codebase is modular, allowing for easy adaptation to new datasets or models.
*   **Ready for Deployment:** Includes a saved model checkpoint for immediate use in inference pipelines.

## Getting Started

### Prerequisites

*   Python 3.8+
*   PyTorch
*   Torchvision
*   NumPy
*   Matplotlib
*   Docker (optional)

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/HarshilMaks/EuroSAT-ResNet50.git
    cd EuroSAT-ResNet50
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3.  (Optional) Build the Docker image:
    ```bash
    make build
    ```

## Usage

### Training the Model

To train the model from scratch, run the following command:

```bash
python train.py --data_dir path/to/EuroSAT_RGB --epochs 20
```

### Running Inference

To run inference on a single image or a directory of images, use the provided script:

```bash
python predict.py --model_path path/to/your/model.pth --image_path path/to/your/image.jpg
```

## Model Performance

The model's performance is summarized below:

*   **Validation Accuracy:** ~94%

*(Here, you could embed a confusion matrix image or a training/validation accuracy graph)*

## Real-World Use Cases

This model can be a foundational component for various impactful applications. See the [USE_CASES.md](USE_CASES.md) file for a detailed list of potential real-world applications and how this project can be extended to address them.

## Project Structure

```
├── data
├── models
├── notebooks
├── scripts
│   ├── train.py
│   └── predict.py
├── .gitignore
├── Dockerfile
├── Makefile
├── README.md
└── requirements.txt
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss potential changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

*   The EuroSAT dataset was provided by Helber et al. in the paper "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification".

---

### Suggestion for a Real-World Use Case

While all the use cases listed by ChatGPT are valid, here is a highly impactful and less saturated idea that could bring significant positive change:

**Hyper-Local Climate Change Impact Monitoring for Small Island Nations**

*   **The Problem:** Small Island Developing States (SIDS) are on the front lines of climate change. They face existential threats from sea-level rise, coastal erosion, and increased frequency of extreme weather events. However, they often lack the resources for granular, real-time monitoring of these changes.
*   **Your Solution as a Tool:**
    *   **Coastal Erosion Monitoring:** By regularly processing new Sentinel-2 imagery (which is freely available), your model can automatically detect changes in coastlines (`SeaLake` vs. `Residential` or `Pasture`). This can provide an early warning system for vulnerable communities.
    *   **Mangrove Forest Health:** Mangroves are crucial for coastal protection. Your model could be extended to identify the health and density of these forests (`Forest` class), flagging areas of degradation that require intervention.
    *   **Post-Cyclone Damage Assessment:** After a tropical cyclone, your model could rapidly assess the extent of damage by classifying areas that have changed from `Residential` or `Forest` to bare land or water, helping to direct aid and recovery efforts more efficiently.
*   **Why it's Impactful:**
    *   **Direct Humanitarian Impact:** This directly helps some of the world's most vulnerable populations adapt to climate change.
    *   **High Visibility:** Projects focused on climate adaptation for SIDS often attract attention from international organizations like the UN, World Bank, and environmental NGOs.
    *   **Technically Feasible:** The core technology you've built is directly applicable. The main challenge would be building the data pipeline to ingest and process new satellite imagery, which is a great next step for the project.
*   **How to Frame it:** You could position your project as a "proof-of-concept for a low-cost, open-source tool for climate resilience in Small Island Nations."

This use case is compelling because it's specific, addresses a critical and underserved need, and has a clear path for expansion from your current project. It's a story that is much more powerful than a generic "deforestation monitor."