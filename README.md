<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180"></a>
</p>

## ICLR'22 Improving Non-Autoregressive Translation Models Without Distillation
[[paper](http://www.cs.toronto.edu/~mvolkovs/ICLR2022_CMLMC.pdf)]

Authors: Xiao Shi Huang, Felipe Perez, [Maksims Volkovs](http://www.cs.toronto.edu/~mvolkovs)

<a name="intro"/>

## Introduction
This repository contains a full implementation of the CMLMC implemented with the fairseq library, and includes both training and evaluation routines on the IWSLT'14 De-En dataset.

<a name="env"/>

## Environment
The python code is developed and tested on the following environment:
* Python 3.7.9
* Pytorch 1.10.0

Experiments on IWSLT'14 De-En and En-De datasets (included in this repo) were run on NVIDIA V100 GPU with 32GB GPU memory; all other experiments were run on an IBM server with 160 POWER9 CPUs, 600GB RAM and 4 Tesla V100 GPUs

<a name="dataset"/>

## Dataset

The IWSLT'14 De-En and En-De dataset were included in this repo; for the WMT'14 En-De and WMT'16 En-Ro datasets refer to the fairseq's instructions [here](https://github.com/pytorch/fairseq/tree/master/examples/translation) 

## Running The Code

1. `./trainNAT.sh` will train and evaluate both the CMLM benchmark and the CMLMC model on IWSLT'14 De-En raw dataset.
2. (Optionally) launch tensorboard to monitor progress by `tensorboard --logdir=<log_path>`

This script runs the 512-1024-4 Transformer NAR model (see paper for details). By default all avialable GPUs are used, but parameters such as batchsize are set for for 1 GPU. If multiple GPUs are avaialbe, either point the script to only one GPU or adjust model parameters accordingly.
