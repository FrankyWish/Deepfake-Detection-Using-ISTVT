# Model Checkpoints

Trained weights are **not** stored in this repository because `.pth` files are large
and are produced by training (see the notebooks), not committed to git.

Expected files after a training run:

| File | Produced by | Description |
|------|-------------|-------------|
| `best_model.pth`  | `notebooks/istvt_video_transformer.ipynb` | Best ISTVT checkpoint (highest validation accuracy) |
| `final_model.pth` | `notebooks/istvt_video_transformer.ipynb` / `notebooks/xception_cnn.ipynb` | Final-epoch checkpoint |

## How to get a checkpoint

1. **Train it yourself** — run the relevant notebook end-to-end with the dataset in
   place (`final_FFpp/`). The notebook calls `torch.save(model.state_dict(), ...)`
   and the `.pth` lands here.
2. **Use a shared checkpoint** — if a teammate already trained one, drop their
   `best_model.pth` into this folder. Loading is:

   ```python
   model = ISTVT(spatial_backbone='vit_small_patch16_224',
                 temporal_depth=2, temporal_num_heads=4)
   model.load_state_dict(torch.load('checkpoints/best_model.pth',
                                    map_location='cpu'))
   model.eval()
   ```

> The checkpoint must match the architecture hyper-parameters it was trained with
> (backbone, `temporal_depth`, `temporal_num_heads`), or `load_state_dict` will fail.

## Sharing large weights

Plain git is a poor fit for large binaries. Prefer one of:
- **Git LFS**: `git lfs track "*.pth"`
- A release asset or a Google Drive / cloud link referenced in the main README.
