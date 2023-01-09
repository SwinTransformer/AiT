# All in Tokens: Unifying Output Space of Visual Tasks via Soft Token
By [Jia Ning](https://scholar.google.com/citations?user=hW0AexsAAAAJ&hl=en)\*, [Chen Li](https://github.com/LC-Edward)\*, [Zheng Zhang](https://stupidzz.github.io/), [Zigang Geng](https://scholar.google.com/citations?user=MdFYVoAAAAAJ&hl=zh-CN), [Qi Dai](https://scholar.google.com/citations?user=NSJY12IAAAAJ), [Kun He](https://scholar.google.com/citations?user=YTQnGJsAAAAJ&hl=en), [Han Hu](https://ancientmooner.github.io/)

Code and model are coming soon!
## Introduction
**AiT** is initially described in [arxiv](https://arxiv.org/pdf/2301.02229.pdf), which is a framework to unify the output space of visual tasks. We demonstrate a single unified model that simultaneously handles two typical visual tasks of instance segmentation and depth estimation, which have discrete/fixed-length and continuous/varied-length outputs, respectively. We propose several new techniques that take into account the particularity of visual tasks: 1) Soft tokens. We employ soft tokens to represent the task output. Unlike hard tokens in the common VQ-VAE which are assigned one-hot to discrete codebooks/vocabularies, the soft tokens are assigned softly to the codebook embeddings. Soft tokens can improve the accuracy of both the next token inference and decoding the task output; 2) Mask augmentation. Many visual tasks have corruption, undefined or invalid values in label annotations, i.e., occluded area of depth maps. We show that a mask augmentation technique can greatly benefit these tasks. With these new techniques and other designs, we show that the proposed general-purpose task solver can perform both instance segmentation and depth estimation well. Particularly, we achieve 0.279 RMSE on the specific task of NYUv2 depth estimation, setting a new record on this benchmark.

![teaser](figures/teaser.png)

## Citation
If you find our work useful in your research, please cite:
```
@article{ning2023all,
  title={All in Tokens: Unifying Output Space of Visual Tasks via Soft Token},
  author={Jia Ning and Chen Li and Zheng Zhang and Zigang Geng and Qi Dai and Kun He and Han Hu},
  journal={arXiv preprint arXiv:2301.02229},
  year={2023}
}
```
