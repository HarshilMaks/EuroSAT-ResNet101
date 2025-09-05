# EuroSAT • Land Use & Cover Classification with ResNet-50

**This project fine-tunes ResNet-50 on satellite imagery (EuroSAT-RGB) to classify land cover types with high accuracy (~94%).**

### Dataset — EuroSAT-RGB

The dataset contains **27,000** labeled satellite images (64×64) of **10 land cover classes**, captured by Sentinel-2 satellite. Classes include: *annual crop, forest, residential, industrial, river, sea/lake*, etc.  
Balanced and georeferenced — ideal for land-use classification benchmark.  
Source: *EuroSAT, Helber et al., 2017* :contentReference[oaicite:1]{index=1}

Sample images per class:  
*(embed one or a grid of class examples — see above image UI)*  

### Why This Matters

**Why it matters**: Land use and land cover (LULC) classification from satellite imagery enables crucial applications:

- **Urban planning**: Identify changes in urban growth (buildings, roads).  
- **Environmental monitoring**: Detect deforestation, water bodies, soil erosion.  
- **Agricultural oversight**: Monitor crop types and health at scale.  
- **Disaster response**: Quickly assess flood or wildfire impact areas. :contentReference[oaicite:2]{index=2}

Your trained model can be integrated into GIS pipelines or mobile apps to provide near real-time land categorization — valuable in resource-constrained environments.

### What Does This Project Offer?

| Feature                          | Description |
|----------------------------------|-------------|
| **Fast inference**               | Saved ResNet-50 checkpoint (~25M params) for quick predictions. |
| **Solid accuracy**               | ~94% val accuracy with only classifier fine-tuned. |
| **High transfer potential**      | Can be extended for multi-spectral data, other regions, or temporal monitoring. |
| **Educational clarity**          | Clean code + Makefile + Docker examples = practical, resume-worthy. |

### Original Dataset vs. My Approach

The EuroSAT-RGB dataset provides static land cover samples across Europe.  
Your model builds on it by:

- Fine-tuning with **ResNet-50** plus ReLU head for modern, robust feature extraction.
- Leaning on **modular architecture**: preprocessing, training, inference, deployment separated cleanly.
- Delivering both **reproducibility** (Makefile, clear scripts) and **portability** (Docker-ready).

This creates a springboard for customized LULC systems — beyond static classification.

### What Human Problem Can This Solve?

For instance, **in regions prone to seasonal flooding**, this model could:

- Detect **expanding rivers or lakes** in near-real-time from satellite snapshots.
- Trigger alerts for flood risk areas (especially in developing regions).
- Save lives and reduce damage by informing stakeholders quickly.

Compared to agriculture-only tools, this model emphasizes **environmental monitoring and safety**, an underexplored yet impactful direction.
