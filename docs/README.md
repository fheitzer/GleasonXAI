# GleasonXAI Documentation

Comprehensive documentation for the GleasonXAI codebase - a pathologist-like explainable AI system for interpretable Gleason grading in prostate cancer.

**Published in:** Nature Communications (2025) | [DOI: 10.1038/s41467-025-64712-4](https://doi.org/10.1038/s41467-025-64712-4)

---

## Documentation Structure

### 🚀 Getting Started

**[00_OVERVIEW.md](00_OVERVIEW.md)** - Start here!
- Project overview and scientific context
- Quick start guide
- Repository structure
- System requirements
- Citation information

### 🏗️ Technical Deep Dives

**[01_ARCHITECTURE.md](01_ARCHITECTURE.md)** - Model Architecture
- U-Net with EfficientNet-B4 encoder
- Multi-level prediction system
- Sliding window inference
- Ensemble strategy
- Design decisions and rationale

**[02_DATA_PIPELINE.md](02_DATA_PIPELINE.md)** - Data Processing
- GleasonX dataset class
- Multi-annotator soft labels
- Label hierarchy (3 levels)
- Data augmentation pipelines
- Tissue masking and preprocessing

**[03_TRAINING.md](03_TRAINING.md)** - Training Guide
- Loss functions (SoftDiceLoss, TreeLoss)
- Optimization and hyperparameters
- PyTorch Lightning module
- Metrics and logging
- Troubleshooting

### 📚 Reference Materials

**[07_CODE_REFERENCE.md](07_CODE_REFERENCE.md)** - Complete API Reference
- Detailed documentation of all modules
- Class and function signatures
- Implementation details
- Usage examples

**[AI_ASSISTANT_GUIDE.md](AI_ASSISTANT_GUIDE.md)** - For AI Assistants
- Comprehensive context for LLM assistants
- Common tasks and solutions
- Important implementation details
- Code modification guidelines
- Quick reference tables

---

## Quick Navigation

### By User Type

**Researchers / New Users**
1. Start with [00_OVERVIEW.md](00_OVERVIEW.md) for context
2. Read [01_ARCHITECTURE.md](01_ARCHITECTURE.md) to understand the model
3. Follow [03_TRAINING.md](03_TRAINING.md) to train your first model

**Developers**
1. Review [AI_ASSISTANT_GUIDE.md](AI_ASSISTANT_GUIDE.md) for codebase overview
2. Use [07_CODE_REFERENCE.md](07_CODE_REFERENCE.md) as API reference
3. Check [02_DATA_PIPELINE.md](02_DATA_PIPELINE.md) for data handling

**Data Scientists**
1. Read [02_DATA_PIPELINE.md](02_DATA_PIPELINE.md) for data processing
2. Study [03_TRAINING.md](03_TRAINING.md) for training procedures
3. Reference [07_CODE_REFERENCE.md](07_CODE_REFERENCE.md) for metrics

### By Task

**Training a Model**
→ [03_TRAINING.md](03_TRAINING.md)

**Understanding the Architecture**
→ [01_ARCHITECTURE.md](01_ARCHITECTURE.md)

**Working with Data**
→ [02_DATA_PIPELINE.md](02_DATA_PIPELINE.md)

**Implementing New Features**
→ [AI_ASSISTANT_GUIDE.md](AI_ASSISTANT_GUIDE.md) + [07_CODE_REFERENCE.md](07_CODE_REFERENCE.md)

**Reproducing Paper Results**
→ [00_OVERVIEW.md](00_OVERVIEW.md) + [03_TRAINING.md](03_TRAINING.md)

---

## Key Concepts

### Hierarchical Classification
GleasonXAI uses a 3-level label hierarchy:
- **Level 0:** Gleason patterns (3, 4, 5)
- **Level 1:** Fine-grained explanations (10 classes) ← **Default**
- **Level 2:** Sub-explanations (detailed)

→ Details in [00_OVERVIEW.md](00_OVERVIEW.md#key-features)

### Soft Label Training
Multiple pathologist annotations averaged to create probability distributions.
- Captures diagnostic uncertainty
- Improves generalization
- Achieves superior performance (Dice: 0.713 vs. 0.691)

→ Details in [02_DATA_PIPELINE.md](02_DATA_PIPELINE.md#soft-label-generation)

### Ensemble Prediction
3 independently trained models with identical architecture.
- Reduces variance
- Improves robustness
- Better calibration

→ Details in [01_ARCHITECTURE.md](01_ARCHITECTURE.md#ensemble-strategy)

---

## Code Examples

### Load Dataset
```python
from gleasonxai.gleason_data import GleasonX

dataset = GleasonX(
    path="/path/to/data/GleasonXAI",
    label_level=1,  # Fine-grained explanations
    scaling="MicronsCalibrated",
    seed=42
)

image, label, metadata = dataset[0]
```
→ Full details in [07_CODE_REFERENCE.md](07_CODE_REFERENCE.md#gleasonx-dataset-class)

### Train Model
```bash
python scripts/run_training.py \
    dataset.label_level=1 \
    loss_functions=soft_dice_balanced \
    experiment=MyExperiment
```
→ Full guide in [03_TRAINING.md](03_TRAINING.md#quick-start)

### Run Inference
```bash
python scripts/run_gleasonXAI.py \
    --images /path/to/images \
    --save_path /path/to/output
```
→ Details in [AI_ASSISTANT_GUIDE.md](AI_ASSISTANT_GUIDE.md#task-6-inference-on-new-data)

---

## Documentation Statistics

| Document | Size | Lines | Purpose |
|----------|------|-------|---------|
| 00_OVERVIEW.md | ~25 KB | ~550 | Project overview |
| 01_ARCHITECTURE.md | ~35 KB | ~750 | Architecture details |
| 02_DATA_PIPELINE.md | ~40 KB | ~900 | Data pipeline |
| 03_TRAINING.md | ~45 KB | ~950 | Training guide |
| 07_CODE_REFERENCE.md | ~55 KB | ~1200 | API reference |
| AI_ASSISTANT_GUIDE.md | ~35 KB | ~800 | AI assistant guide |
| **Total** | **~235 KB** | **~5150 lines** | Complete documentation |

---

## Technology Stack Reference

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.10.13 |
| Deep Learning | PyTorch | 2.1.1 |
| Training Framework | PyTorch Lightning | 2.2.0 |
| Model Architecture | segmentation-models-pytorch | 0.3.3 |
| Encoder | EfficientNet-B4 | ImageNet pretrained |
| Configuration | Hydra | 1.3.2 |
| Augmentation | Albumentations | 1.3.1 |
| Medical Imaging | MONAI | 1.3.0 |
| Experiment Tracking | Weights & Biases | 0.17.1 |
| Dependency Management | uv | Latest |

---

## Repository Links

- **Main Repository:** [src/ProQuant-AI/GleasonXAI/](../)
- **Source Code:** [src/gleasonxai/](../src/gleasonxai/)
- **Scripts:** [scripts/](../scripts/)
- **Configs:** [configs/](../configs/)
- **Tests:** [tests/](../tests/)
- **Notebooks:** [notebooks/](../notebooks/)

---

## External Resources

### Publications
- **Nature Paper:** https://www.nature.com/articles/s41467-025-64712-4
- **arXiv Preprint:** https://arxiv.org/abs/2410.15012

### Data
- **Figshare Dataset:** https://springernature.figshare.com/articles/dataset/Pathologist-like_explainable_AI_for_interpretable_Gleason_grading_in_prostate_cancer/27301845
- **Gleason2019 Challenge:** https://gleason2019.grand-challenge.org/

### Institution
- **DKFZ Heidelberg:** https://www.dkfz.de/en/
- **Press Release:** https://www.dkfz.de/en/news/press-releases/detail/transparent-artificial-intelligence-improves-assessment-of-prostate-cancer-aggressiveness

---

## Citation

If you use GleasonXAI in your research, please cite:

```bibtex
@article{mehrtens2025gleasonxai,
  title={Pathologist-like explainable AI for interpretable Gleason grading in prostate cancer},
  author={Mehrtens, Hendrik A. and Mittmann, Gesa and others},
  journal={Nature Communications},
  volume={16},
  number={8959},
  year={2025},
  doi={10.1038/s41467-025-64712-4}
}
```

---

## Version Information

- **Documentation Version:** 1.0.0
- **Codebase Version:** 1.2.0
- **Last Updated:** 2026-01-29
- **Python Version:** 3.10.13
- **PyTorch Version:** 2.1.1

---

## Contact

- **Authors:** Hendrik Mehrtens, Gesa Mittmann
- **Institution:** German Cancer Research Center (DKFZ), Heidelberg
- **Email:**
  - hendrikalexander.mehrtens@dkfz-heidelberg.de
  - gesa.mittmann@dkfz-heidelberg.de

---

## Contributing

When contributing to the documentation:

1. **Maintain consistency** with existing style
2. **Update all affected documents** when making changes
3. **Test code examples** before documenting
4. **Use markdown features** (tables, code blocks, links)
5. **Keep documentation in sync** with codebase version

---

## License

See [LICENSE](../LICENSE) file in repository root.

---

**Happy coding! 🔬🤖**

For questions or issues, please contact the authors or open an issue in the repository.
