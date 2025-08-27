# 🛰️ RPFusionNet: Region-Patch Fusion Network for Large-Scale Remote Sensing Segmentation

A novel semantic segmentation framework designed for **large-scale high-resolution remote sensing images**, addressing **scale variation**, **weak/no-texture regions**, and **patch-level limitations**.
 RPFusionNet introduces a **dual-branch architecture (REGION + PATCH)** with hierarchical pooling and efficient global-local feature fusion.

------

## 🚀 Features

```markdown
- 🌍 **REGION branch** for global context modeling with multi-scale pooling (SPU)
- 🧩 **PATCH branch** for local semantic feature extraction (TransUNet-based SFE)
- 🔗 **Information Aggregation Module (IAM)** for global-local fusion
- 🌀 **Auxiliary & Global decoders (ADM, GDM)** with parameter sharing
- 📉 **Joint loss function** (BCE for both REGION and PATCH)
- ⚡ Efficient FLOPs & reduced parameters via fixed SFE and shared decoders
```

## 🧪 Experimental Results

| Dataset   | IoU (PTRSegNet) | IoU (SegRPNet) | Gain   |
| --------- | --------------- | -------------- | ------ |
| WBDS      | 91.46%          | **92.08%**     | +0.62% |
| AIDS      | 88.69%          | **89.99%**     | +1.30% |
| Vaihingen | 86.56%          | **88.44%**     | +1.88% |

- 📊 RPFusionNet outperforms **DeepLabv3+, PSPNet, UANet, TransUNet, RSMamba, UNetMamba** etc.
- 🎯 Gains are especially strong in **large buildings, weak-texture regions, and dense small-object areas**.
- ⚡ FLOPs: **3.85 G** | Trainable Params: **46.83 M**

------

## 📦 Installation

> Requirements: GPU (e.g., 3090), CUDA 11+, Python 3.10

```bash
git clone https://github.com/yourname/RPFusionNet.git
cd RPFusionNet
conda create -n RPFusionNet python=3.10
conda activate RPFusionNet

pip install -r requirements.txt
```

------

## ⚙️ Usage

### 🔧 Step 1: Train on dataset

```bash
python scripts/train.py --dataset WBDS
```

### 🧪 Step 2: Evaluate model

```bash
python scripts/evaluate.py --dataset WBDS
```

### 💡 Step 3: Inference

```bash
python scripts/inference.py --input path/to/image.png
```

------

## 📊 Ablation Studies

- **Different SFE (U-Net, ViT, TransUNet)** → TransUNet achieves best IoU (+4.1% over ViT).
- **Pooling strategy**: Multi-scale SPU outperforms single-scale by +2.05% IoU.
- **Module combinations**: GDM + GFE + ADM contribute progressively (+4.19% IoU overall).

------

## 📚 Citation

If you use SegRPNet in your research, please cite:

```bibtex
@article{SegRPNet2025,
  title={SegRPNet:A Large-Format High-Resolution Remote Sensing Image Feature Extraction Method for Farmland Protection},
  author={Your Name},
  year={2025}
}
```

------

## 👨‍💻 Authors

```
Li Sha
Undergraduate @ Huazhong Agricultural University
Rank 2/58 | Contributor
Research Area: Remote Sensing & Deep Learning
```

