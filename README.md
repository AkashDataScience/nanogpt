[![LinkedIn][linkedin-shield]][linkedin-url]

## :jigsaw: Objective

- Pick language dataset from Shakespeare plays
- Use character level encoding
- Implement nanoGPT from scratch
- Train nanoDPT model for language generation task

## Prerequisites
* [![Python][Python.py]][python-url]
* [![Pytorch][PyTorch.tensor]][torch-url]
* [![Hugging face][HuggingFace.transformers]][huggingface-url]

## :open_file_folder: Files
- [**nano_gpt.ipynb**](nano_gpt.ipynb)
    - Jupyter notebook to implement basic Bigram language model
- [**nanogpt.py**](nanogpt.py)
    - This is the main file of this project
    - This file contains encoder-decoder logic
    - It has all the building blocks require to make nanoGPT transformer model
    - Training loop is also in this file

## :building_construction: Model Architecture
The model is implemented based on Language Models are Few-Shot Learners. The transformer
architecture comprises N Decoder blocks, each equipped with multi-head attention mechanisms. Our
particular model includes 6 of these blocks. Tokens are initially embedded into 384-dimensional
vectors (d_model), and each block utilizes multi-head attention with 8 heads (h), followed by
feed-forward networks sized at 2048 (d_ff). Furthermore, positional encodings are integrated to
capture sequence context effectively.

**Model Dimensions:**

- Embedding Dimension (n_embd): Size of the embedding vectors (Default=384).
- Attention Heads (n_head): Number of Attention Heads for Multi-Head Attention (Default=6).
- Decoder Blocks (n_layer): Nmber of Decoder Blocks for GPT (Default=6)

## :golfing: Approach

- Model: Architecture is simillar to GPT-2 with less number of decoder block. 
- Training Data: `input.txt` file containes plays written by Shakespeare
- Evaluation: Model is evaluated after every 500 steps. It can be changed by using `eval_interval`
argument. 


## Installation

1. Clone the repo
```
git clone https://github.com/AkashDataScience/nanogpt.git
```
2. Go inside folder
```
 cd nanogpt
```
3. Install dependencies
```
pip install -r requirements.txt
```

## Training

```
# Start training with:
python nanogpt.py

# To train with 128 batch size and 10000 iterations
python nanogpt.py --batch_size=128 --max_iters=10000

# To train from jupyter notebook with 8 heads and 0.1 dropout
%run nanogpt.py --n_head=8 --dropout=0.1
```

## Usage 
Please refer to [ERA V2 Session 19](https://github.com/AkashDataScience/ERA-V2/tree/master/Week-19)

## Contact

Akash Shah - akashmshah19@gmail.com  
Project Link - [ERA V2](https://github.com/AkashDataScience/ERA-V2/tree/master)

## Acknowledgments
This repo is developed using references listed below:
* [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
* [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165)
* [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2s)


[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/akash-m-shah/
[Python.py]:https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[python-url]: https://www.python.org/
[PyTorch.tensor]: https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white
[torch-url]: https://pytorch.org/
[HuggingFace.transformers]: https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange
[huggingface-url]: https://huggingface.co/