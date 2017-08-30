# pytorch_xworld
This is a pytorch implementation of the xworld experiment in [*A Deep Compositional Framework for Human-like Language Acquisition in Virtual Environment*](https://arxiv.org/abs/1703.09831), Haonan Yu, Haichao Zhang, Wei Xu, arXiv 1703.09831, 2017.

To run the code:
* build the XWorld environment from https://github.com/PaddlePaddle/XWorld 
* add the directory holding the ".so" file of the environment to the PYTHON_PATH
* for training, run `python trainer.py --cuda`
* for evaluation, run `python evaluate.py --cuda` 

Note:
* It seems the XWorld environment only supports python 2.7 now. Due to that, one can only use python 2.7 to run the code. 
* The code is written and tested under pytorch 0.1.12. Due to a lot of changes in the new version (2.0.0), I'm not sure whether the code runs perfectly in the newest version. So, a safe choice is to first try on the older pytorch version. The installation goes by `conda install pytorch=0.1.12 torchvision cuda80 -c soumith`. 
