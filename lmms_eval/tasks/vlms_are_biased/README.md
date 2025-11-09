# VLMs Are Biased

## Description

This benchmark tests how prior knowledge in Vision-Language Models (VLMs) affects their accuracy on standard, objective visual tasks of counting and identification. The dataset contains counterfactual images (e.g., an Adidas logo with 4 stripes instead of 3) across 7 diverse domains.

**Paper**: [Vision Language Models are Biased](https://arxiv.org/abs/2505.23941)
**Project Page**: [vlmsarebiased.github.io](https://vlmsarebiased.github.io)
**Dataset**: [anvo25/vlms-are-biased](https://huggingface.co/datasets/anvo25/vlms-are-biased)

## Domains

The benchmark covers 7 domains:
- **Chess/Games**: Chess pieces, Go boards, Xiangqi, Sudoku
- **Logos**: Nike, Adidas, Maserati, Mercedes-Benz, Audi
- **Animals**: Mammals and birds with modified limb counts
- **Optical Illusions**: Ebbinghaus, Müller-Lyer, Ponzo, and others
- **Patterned Grids**: Dice and tally mark patterns
- **Flags**: Star and stripe count variations
- **Game Boards**: Various board games with dimension variations

## Metrics

- **accuracy**: Overall counting accuracy across all examples
- **accuracy_by_domain**: Per-domain accuracy breakdown

## Usage

```bash
python -m lmms_eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
  --tasks vlms_are_biased \
  --batch_size 1 \
  --device cuda:0
```

## Expected Results

According to the paper, state-of-the-art VLMs achieve approximately **17.05% accuracy** on average across domains, indicating strong bias toward memorized knowledge over visual recognition.

## Citation

```bibtex
@misc{vlmsarebiased,
  title={Vision Language Models are Biased},
  author={An Vo and Khai-Nguyen Nguyen and Mohammad Reza Taesiri and Vy Tuong Dang and Anh Totti Nguyen and Daeyoung Kim},
  year={2025},
  eprint={2505.23941},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2505.23941},
}
```
