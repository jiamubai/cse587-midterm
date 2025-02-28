# CSE587 Spring 2025 Midterm Project

Sentiment Analysis with RNNs and CNNs

## Members

* Jiamu Bai
* Ryo Kamoi

## Report

We provide a report for our project. Please refer to [report.pdf](report.pdf)

## Setup

We use miniconda3 for managing our environment.

```bash
bash setup.sh

# download punkt tokenizer
ipython
> import nltk
> nltk.download('punkt_tab')
> exit()
```

We run our code in the following environment. You may need to change the version of the packages if you are working in a different environment.

* NVIDIA RTX A6000
* CUDA 11.8

## Run Experiments

The following code runs all experiments. Please refer to [run.sh](run.sh) for details.

```bash
bash run.sh
```
