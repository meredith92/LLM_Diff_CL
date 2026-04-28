# PCB Conductor Continual Learning - Paper Code Guide

This document explains the code structure corresponding to each section of the paper.

## 📋 Table of Contents

### Section 4: Methodology

#### 4.2 Language-Guided Diffusion Model
- **Code**: `tools/train_mask_ddpm_min.py`
- **Status**: ✅ Complete
- **Description**: Trains a diffusion model on binary segmentation masks with language guidance
- **Usage**:
  ```bash
  python tools/train_mask_ddpm_min.py \
    --mask_root data/masks/train \
    --out_dir work_dirs/diffusion_pretraining \
    --epochs 50 \
    --batch_size 16
  ```

#### 4.3 Diffusion Uncertainty Analysis
- **Code**: `tools/uncertainty_analysis.py`
- **Status**: ✅ Complete
- **Description**: Analyzes uncertainty in diffusion predictions
  - Timestep-wise uncertainty: variance across denoising steps
  - Spatial uncertainty: pixel-level confidence maps
  - Uncertainty-error correlation
  - Domain adaptation uncertainty

- **Usage**:
  ```bash
  python tools/uncertainty_analysis.py \
    --dataset-dir data/predictions \
    --output-dir work_dirs/uncertainty_analysis \
    --plot \
    --save-maps
  ```

- **Key Features**:
  - `DiffusionUncertaintyAnalyzer` class for analysis
  - Visualization of uncertainty maps
  - Uncertainty distribution plotting
  - Timestep uncertainty profile computation

#### 4.3.2 Rule-based Topology Constraints
- **Code**: `projects/pcb_conductor/tools/pseudo_judge.py`
- **Status**: ✅ Complete (existing)
- **Description**: Applies rule-based topology constraints
  - Pin connectivity rules
  - Structural regularity enforcement
  - Topology violation detection

#### 4.3.3 LLM Reasoning
- **Code**: `projects/pcb_conductor/tools/mt_struct_continual.py` (llm_judge part)
- **Status**: ✅ Complete (existing)
- **Description**: Uses LLM for semantic segmentation guidance

#### 4.4 Continual Adaptation Mechanism
- **Code**: `projects/pcb_conductor/tools/mt_struct_continual.py` (loss + EMA)
- **Status**: ✅ Complete (existing)
- **Description**: Implements continual learning with EMA regularization
  - Exponential moving average for stability
  - Loss functions for domain adaptation
  - Knowledge retention mechanisms

---

### Section 5: Experiments

#### 5.1 Setup - Orchestration
- **Code**: `tools/train_continual.py`
- **Status**: ✅ Complete
- **Description**: Full end-to-end continual learning pipeline
  - Orchestrates diffusion pretraining
  - Domain A training
  - Domain B continual finetuning
  - Multi-stage evaluation

- **Usage**:
  ```bash
  python tools/train_continual.py \
    --base-work-dir work_dirs/continual_experiment \
    --mask-root data/masks/domain_a \
    --domain-a-config projects/pcb_conductor/configs/segformer_mt_vb.py \
    --domain-b-config projects/pcb_conductor/configs/segformer_mt_vb.py
  ```

- **Pipeline Stages**:
  1. **Stage 1**: Diffusion pretraining on Domain A masks
  2. **Stage 2**: Segmentation model training on Domain A
  3. **Stage 3a**: Evaluation on both domains (after A)
  4. **Stage 4**: Continual finetuning on Domain B
  5. **Stage 3b**: Final evaluation on both domains

#### 5.2 Main Results - Continual Learning Evaluation
- **Code**: `projects/pcb_conductor/tools/continual_eval.py`
- **Status**: ✅ Complete (existing)
- **Description**: Evaluates continual learning metrics
  - Target performance (mIoU on new domain)
  - Forgetting (performance drop on old domain)
  - Plasticity (improvement on new domain)
  - Backward transfer (impact on source domain)
  - Forward transfer (benefit to target domain)

#### 5.3 Forgetting Analysis
- **Code**: `projects/pcb_conductor/tools/visualize_results.py` (ForgettingCurveVisualizer)
- **Status**: ✅ Complete (updated)
- **Description**: Visualizes and analyzes forgetting curves
  - Source domain stability across training stages
  - Plasticity-stability tradeoff analysis
  - Comparison of different methods

- **Key Visualizations**:
  - Source domain performance over time
  - Stability vs Plasticity tradeoff
  - Forgetting rate comparison

#### 5.4 Ablation Study
- **Code**: `projects/pcb_conductor/tools/run_ablations.py`
- **Status**: ✅ Complete (existing)
- **Description**: Systematic evaluation of component contributions
  - Diffusion prior impact
  - EMA regularization effect
  - LLM guidance contribution
  - Rule-based topology rules
  - Combined full model performance

- **Usage**:
  ```bash
  python projects/pcb_conductor/tools/run_ablations.py \
    --base-config projects/pcb_conductor/configs/segformer_mt_vb.py \
    --work-dir work_dirs/ablation_study \
    --launcher none
  ```

- **Ablation Variants**:
  - `full`: Full method (Diffusion + Judge + LLM)
  - `no_diffusion`: w/o Diffusion Prior
  - `no_judge`: w/o Structural Judge
  - `diffusion_frozen`: Diffusion Frozen
  - `diffusion_finetune`: Diffusion Finetuned
  - `no_selective`: w/o Selective Weight Map
  - `no_skeleton`: w/o Skeleton Consistency Loss

#### 5.5 Structural Analysis
- **Code**: `tools/structure_analysis.py`
- **Status**: ✅ Complete
- **Description**: Analyzes topological properties of predictions
  - Pin detection from segmentation masks
  - Pin measurements (count, spacing, alignment)
  - Topology quality metrics:
    - Pin count agreement with GT
    - Alignment score (how well pins are aligned)
    - Uniformity score (pin size consistency)
    - Topology violation rate
  - Domain-specific structural insights

- **Usage**:
  ```bash
  python tools/structure_analysis.py \
    --pred-dir work_dirs/predictions \
    --gt-dir data/ground_truth \
    --output-dir work_dirs/structure_analysis \
    --visualize \
    --plot
  ```

- **Key Features**:
  - `StructureAnalyzer` class for pin detection
  - Spatial constraint enforcement
  - Topology rule compliance checking
  - Pin visualization with confidence scores

#### 5.6 Diffusion Prior Visualization
- **Code**: `projects/pcb_conductor/tools/visualize_results.py` (UncertaintyVisualizer extensions)
- **Status**: ✅ Complete (updated)
- **Description**: Visualizes diffusion-based predictions and uncertainty
  - Multiple diffusion samples from same condition
  - Uncertainty maps (variance across samples)
  - Denoising trajectory visualization
  - Diffusion prior diversity

- **Key Methods**:
  - `plot_diffusion_prior_samples()`: Shows diverse samples
  - `plot_denoising_trajectory()`: Evolution from noise to clean
  - `plot_uncertainty_samples()`: Uncertainty from multiple samples

---

## 📊 Summary of Created Files

### New Files Created

| File | Lines | Purpose | Section |
|------|-------|---------|---------|
| `tools/uncertainty_analysis.py` | 500+ | Diffusion uncertainty analysis | 4.3 |
| `tools/ablation_study.py` | 400+ | Ablation study framework | 5.4 | ❌ **已删除（重复）** |
| `tools/structure_analysis.py` | 600+ | Topological structure analysis | 5.5 |
| `tools/train_continual.py` | Complete | End-to-end pipeline orchestration | 5.1 |
| `projects/pcb_conductor/tools/visualize_results.py` | Updated | Enhanced visualization with diffusion viz | 5.3, 5.6 |

### Updated Files

| File | Changes |
|------|---------|
| `projects/pcb_conductor/tools/visualize_results.py` | Added `plot_diffusion_prior_samples()` and `plot_denoising_trajectory()` methods |

---

## 🚀 Running the Full Experiment

### 1. Preparation
```bash
# Ensure data directories exist
mkdir -p data/masks/domain_a
mkdir -p data/images/domain_a
mkdir -p data/masks/domain_b
mkdir -p data/images/domain_b
```

### 2. Run Full Pipeline
```bash
python tools/train_continual.py \
  --base-work-dir work_dirs/continual_experiment \
  --mask-root data/masks/domain_a \
  --domain-a-config projects/pcb_conductor/configs/segformer_mt_vb.py \
  --domain-b-config projects/pcb_conductor/configs/segformer_mt_vb.py
```

### 3. Run Ablation Study
```bash
python projects/pcb_conductor/tools/run_ablations.py \
  --base-config projects/pcb_conductor/configs/segformer_mt_vb.py \
  --work-dir work_dirs/ablation_study
```

### 4. Analyze Results

#### Uncertainty Analysis
```bash
python tools/uncertainty_analysis.py \
  --dataset-dir work_dirs/continual_experiment \
  --output-dir work_dirs/uncertainty_analysis \
  --plot \
  --save-maps
```

#### Structure Analysis
```bash
python tools/structure_analysis.py \
  --pred-dir work_dirs/continual_experiment/predictions \
  --gt-dir data/ground_truth \
  --output-dir work_dirs/structure_analysis \
  --visualize \
  --plot
```

#### Generate Visualizations
```bash
python projects/pcb_conductor/tools/visualize_results.py
# Outputs figures to: figures/
```

---

## 📈 Output Structure

```
work_dirs/
├── continual_experiment/          # Main experiment results
│   ├── stage1_diffusion_pretraining/
│   ├── stage2_train_domain_a/
│   ├── stage3_evaluate_both/
│   ├── stage4_continual_finetune_domain_b/
│   └── continual_results.json
├── ablations/                      # Ablation study results
│   ├── baseline/
│   ├── with_diffusion/
│   ├── with_llm/
│   ├── with_topology/
│   ├── ours_full/
│   └── ablation_results.json
├── uncertainty_analysis/           # Uncertainty analysis
│   ├── uncertainty_summary.json
│   ├── uncertainty_map.png
│   ├── uncertainty_distribution.png
│   └── timestep_uncertainty.png
└── structure_analysis/             # Structure analysis
    ├── structure_summary.json
    ├── pin_detection.png
    └── structure_metrics.png

figures/
├── fig2_forgetting_curves.png      # Forgetting analysis
├── fig3_uncertainty_maps.png       # Uncertainty visualization
├── fig4_pin_count_analysis.png     # Structure analysis
├── fig5_ablation_results.png       # Ablation comparison
├── fig6_error_types.png            # Error type distribution
├── fig7_diffusion_prior_samples.png # Diffusion prior diversity
└── fig8_denoising_trajectory.png   # Denoising evolution
```

---

## 🔧 Dependencies

- PyTorch >= 1.9.0
- MMSegmentation
- NumPy, SciPy
- Matplotlib, Seaborn
- OpenCV (cv2)
- PIL

Install with:
```bash
pip install -r requirements.txt
```

---

## 📚 References to Paper Sections

| Code File | Paper Section | Key Contribution |
|-----------|---------------|------------------|
| `train_mask_ddpm_min.py` | 4.2 | Language-guided diffusion model |
| `uncertainty_analysis.py` | 4.3 | Diffusion uncertainty quantification |
| `pseudo_judge.py` | 4.3.2 | Rule-based topology constraints |
| `mt_struct_continual.py` | 4.3.3, 4.4 | LLM reasoning + continual learning |
| `train_continual.py` | 5.1 | Experimental setup |
| `continual_eval.py` | 5.2 | Main results evaluation |
| `visualize_results.py` | 5.3, 5.6 | Forgetting & diffusion visualization |
| `ablation_study.py` | 5.4 | Component ablation analysis |
| `structure_analysis.py` | 5.5 | Structural topology analysis |

---

## 💡 Notes

1. **Data Preparation**: Ensure proper data directory structure before running experiments
2. **Config Files**: Modify config paths to match your setup
3. **GPU Memory**: Some stages may require high GPU memory; adjust batch sizes if needed
4. **Results Tracking**: All results are logged in work directories with JSON summaries
5. **Visualization**: High-quality figures (300 DPI) saved automatically

---

## ✅ Implementation Status

- ✅ **Section 4.2**: Language-Guided Diffusion (train_mask_ddpm_min.py)
- ✅ **Section 4.3**: Diffusion Uncertainty (uncertainty_analysis.py)
- ✅ **Section 4.3.2**: Rule-based Topology (pseudo_judge.py)
- ✅ **Section 4.3.3**: LLM Reasoning (mt_struct_continual.py)
- ✅ **Section 4.4**: Continual Adaptation (mt_struct_continual.py)
- ✅ **Section 5.1**: Setup (train_continual.py)
- ✅ **Section 5.2**: Main Results (continual_eval.py)
- ✅ **Section 5.3**: Forgetting Analysis (visualize_results.py)
- ✅ **Section 5.4**: Ablation Study (ablation_study.py)
- ✅ **Section 5.5**: Structural Analysis (structure_analysis.py)
- ✅ **Section 5.6**: Diffusion Prior Visualization (visualize_results.py)

---

Generated: 2026-04-28
