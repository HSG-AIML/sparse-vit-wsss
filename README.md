# Sparse Vision Transformer for WSSS

This is the repository for the following paper:
*  Sparse Multimodal Vision Transformer for Weakly Supervised Semantic Segmentation

Table of Contents
=================

   * [Introduction](#introduction)
      * [Pruning Attention Heads](#pruning-attention-heads)
   * [Experiments](#experiments)
      * [Dependencies](#dependencies)
      * [Data](#data)
      * [Model training](#model-training)
      * [Pseudomask generation](#pseudomask-generation)
   * [Comments](#comments)
   * [Code](#code)


# Introduction

In the paper, we show that:

* in remote sensing, the vast majority of heads in vision Transformers can be removed without seriously affecting performance.

* the remaining heads are meaningful, and can be utilized to infer pseudomasks for land cover mapping.

* that weak supervision combined with attention sparsity can effectively reduce the need of fine-grained labeled data, even on small-scale datasets.

## Pruning Attention Heads

We modify the vision Transformer architecture by incorporating gating units into each head. These scalar gates are multiplied by the output of each head. They aim to reduce the impact of less significant heads by disabling them completely through a stochastic relaxation method using [Hard Concrete distribution](https://openreview.net/pdf?id=H1Y8hhg0b).

---
# Experiments

## Dependencies

__Python:__ This project uses Python 3.8, and the dependencies listed in `requirements.txt` can be installed with `pip`, `conda`, or any other package manager, in a virtual environment or other. For example, using `pip`:
```bash
pip install -r requirements.txt
```

__Hardware:__ The model can be trained on one or several GPUs, with the argument `num_devices` (default value = 1).

## Data

In this project the DFC dataset is used, with Sentinel-1 and Sentinel-2 data. The model can handle both unimodal (Sentinel-2) or multimodal (early fusion of Sentinel-1 and Sentinel-2) input, with the `--multimodal` flag.


## Model training

Training the vision Transformer can be done using with `run.py`. Attention pruning can be performed during training with the `--prune` flag.
Vision Transformer’s performance depends heavily on the batch size: this can be done with the `--train_batch_size` argument or by accumulating the gradients for several batches and then making an update with the `--accumulate_grad` argument.

An example command for training the model would be:
```bash
python run.py --prune --train_batch_size 2 --accumulate_grad 4 --optimizer adamw --lr_scheduler --learning_rate 0.001 --depth 12 --patch_size 14 --imgsize 224 224
```

## Pseudomask generation

Once the model is trained, pseudomasks can be generated using `generate_pseudomasks.py`, and by specifying the path to the trained model checkpoint using `--model_checkpoint` argument.

# Comments

To obtain better results, a refinement step can be performed by training a standard UNet model for a few epochs on the generated pseudomasks.

# Acknowledging this work

If you would like to cite our work, please use the following reference:

* Hanna, J., Mommert, M., & Borth, D. (2023). Sparse Multimodal Vision Transformer for Weakly Supervised Semantic Segmentation, Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2023, pp. 2144-2153


# Code
This repository incorporates code from the following sources:
* [Data handling](https://github.com/lukasliebel/dfc2020_baseline)
* [Vision transformers](https://github.com/lucidrains/vit-pytorch)
* [Attention pruning](https://github.com/lena-voita/the-story-of-heads)
