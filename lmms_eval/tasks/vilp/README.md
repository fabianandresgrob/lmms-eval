# ViLP: Probing Visual Language Priors in VLMs

## Description

ViLP (Visual Language Prior) probes the strength of visual language priors in Vision-Language Models by constructing Question-Image-Answer triplets that deliberately deviate from standard training distributions. Each question has 3 images: the first represents the prior (common pattern), while the second and third test visual grounding capabilities.

**Paper**: [Probing Visual Language Priors in VLMs](https://arxiv.org/abs/2501.00569)
**Project Page**: [vilp-team.github.io](https://vilp-team.github.io)
**Dataset**: [ViLP/ViLP](https://huggingface.co/datasets/ViLP/ViLP)

## Task Variants

### 1. vilp (ViLP-F)
Evaluates models with the fact included in the question.

### 2. vilp_without_fact (ViLP-P)
Evaluates models without the fact in the question (removes first sentence).

## Metrics

- **vilp_score**: Mean accuracy on images 2 and 3 (visual grounding)
- **vilp_prior**: Accuracy on image 1 (prior strength)

## Dataset Structure

- 300 questions with 3 images each (900 total evaluations)
- Single-word outputs with normalization for comparison
- Images deliberately deviate from training distribution

## Usage

**ViLP-F (with fact):**
```bash
python -m lmms_eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
  --tasks vilp \
  --batch_size 1 \
  --device cuda:0
```

**ViLP-P (without fact):**
```bash
python -m lmms_eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
  --tasks vilp_without_fact \
  --batch_size 1 \
  --device cuda:0
```

## Evaluation Protocol

1. Each question is presented with 3 images sequentially
2. Models generate single-word answers
3. Outputs are normalized (e.g., "four" → "4", "sphere" → "round")
4. Results are aggregated:
   - **ViLP Prior**: Accuracy on first image (tests prior knowledge)
   - **ViLP Score**: Mean accuracy on images 2-3 (tests visual grounding)

## Citation

```bibtex
@article{luo2024probing,
  title={Probing Visual Language Priors in VLMs},
  author={Luo, Tiange and Cao, Ang and Lee, Gunhee and Johnson, Justin and Lee, Honglak},
  journal={arXiv preprint arXiv:2501.00569},
  year={2024},
  url={https://arxiv.org/abs/2501.00569}
}
```
