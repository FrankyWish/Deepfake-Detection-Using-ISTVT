# DeepFake Detection: CNN vs. Video Transformer (ISTVT)

A course project (EE656) implementing and comparing deepfake video detectors, based on the
**ISTVT** paper:

> C. Zhao, C. Wang, G. Hu, H. Chen, C. Liu, J. Tang,
> *"ISTVT: Interpretable Spatial-Temporal Video Transformer for Deepfake Detection,"*
> IEEE TIFS, vol. 18, pp. 1335–1348, 2023.
> [doi:10.1109/TIFS.2023.3239223](https://doi.org/10.1109/TIFS.2023.3239223)

Two approaches are implemented and compared:

1. **Xception CNN** — classifies individual frames using spatial artifacts (blending
   edges, texture inconsistencies).
2. **Spatial-Temporal Video Transformer** — a ViT spatial backbone combined with a
   temporal transformer that models inconsistency *across* frames, plus attention-map
   visualization to show which regions drive the prediction.

## Results

**Xception CNN** (frame-level, test set)

| Metric | Value |
|---|---|
| Accuracy | 98.4% |
| Precision / Recall / F1 (weighted) | 0.98 |

**Video Transformer** (video-level, test set)

| Metric | Value |
|---|---|
| Accuracy | 75.0% |
| Precision / Recall / F1 (weighted) | 0.76 / 0.75 / 0.75 |
| Best validation accuracy | 80.0% |

## Repository structure

```
.
├── README.md
├── requirements.txt
├── notebooks/
│   ├── istvt_video_transformer.ipynb   # ViT + temporal transformer
│   └── xception_cnn.ipynb              # Xception frame classifier
├── checkpoints/                         # trained weights are saved here
└── report/
    ├── project_report.pdf
    └── presentation.pptx
```

## Setup

```bash
git clone https://github.com/<your-username>/deepfake-detection-istvt.git
cd deepfake-detection-istvt
pip install -r requirements.txt
```

## Usage

Place the preprocessed face-frame dataset at the repo root as `final_FFpp/`:

```
final_FFpp/{train,validation,test}/{real,fake}/*.png
```

Then open and run a notebook top to bottom:

```bash
jupyter notebook notebooks/istvt_video_transformer.ipynb
jupyter notebook notebooks/xception_cnn.ipynb
```

Training saves the best model to `checkpoints/best_model.pth`.

## Authors

EE656 — Group 2: Nithin T M, Kumari Ritika, Akash, Daksh Shettar.

## License

Educational / research use only. The ISTVT paper is © IEEE — access it via the
[official DOI](https://doi.org/10.1109/TIFS.2023.3239223).
