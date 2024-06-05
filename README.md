# DetectGPT
An open-source Pytorch implementation of [DetectGPT](https://arxiv.org/pdf/2301.11305.pdf)

[![Follow on Twitter](https://img.shields.io/twitter/follow/BurhanUlT?style=social)](https://x.com/BurhanUlT)


## Introduction
DetectGPT is an amazing method to determine whether a piece of text is written by large language models (like ChatGPT, GPT3, GPT2, BLOOM etc). However, we couldn't find any open-source implementation of it. Therefore this is the implementation of the paper.

## installation
pip install -r requirements.txt


## Usage
***Here v1.1 refers to DetectGPT, v1.0 is GPTZero***

### Using Python function
```python3 infer.py```
#### example
```
from model import GPT2PPL
model = GPT2PPL()
sentence = "your text here"
model(sentence, "number of words per chunk", "v1.1")
```  
### Using Python input
```python3 local_infer.py```
#### example
```
Please enter your sentence: (Press Enter twice to start processing)
Hello World.
My name is mike.
(empty line)
```

## Acknowledgements
1. Mitchell, Eric, et al. "DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature." arXiv preprint arXiv:2301.11305 (2023).



