# pytorch_xworld
This is a pytorch implementation of the xworld experiment in [*A Deep Compositional Framework for Human-like Language Acquisition in Virtual Environment*](https://arxiv.org/abs/1703.09831), Haonan Yu, Haichao Zhang, Wei Xu, arXiv 1703.09831, 2017.

To run the code
* build the XWorld environment from https://github.com/PaddlePaddle/XWorld 
* add the directory holding the ".so" file of the environment to the PYTHON_PATH
* for training, run `python trainer.py --cuda`
* for evaluation, run `python evaluate.py --cuda` 
