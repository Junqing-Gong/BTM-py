# BTM-py
Biterm Topic Model in Python3

## Biterm Topic Model
[Source code of BTM](https://github.com/xiaohuiyan/BTM)

> Xiaohui Yan, Jiafeng Guo, Yanyan Lan, Xueqi Cheng. A Biterm Topic Model For Short Text. WWW2013.

## Usage
```
cd src
python main.py / python main_torch.py   # CPU / GPU
```
Either use numpy or use torch, it is tens of times slower than with c++, because biterms must compute the conditional distribution one by one in the Gibbs sampling, so it is recommended to use the [source code](https://github.com/xiaohuiyan/BTM) directly.

## 20News
See README.md in dataset dir to install 20News, then `python run_on_20news.py`.
