# DEMO: Horse Colic Data Set

>[Horse Colic Data Set](http://archive.ics.uci.edu/ml/datasets/Horse+Colic)
>[从疝气病症状预测病马的死亡率](https://cuijiahua.com/blog/2017/11/ml_7_logistic_2.html)

## 1. 数据处理
Abstract: Well documented attributes; 368 instances with 28 attributes (continuous, discrete, and nominal); 30% missing values

样本个数: 368
features特征个数: 28个
30%数据缺失

### 1.1 数据缺失处理方法
- 均值填补
- 补充 -1
- 忽略该样本
- 使用相似样本均值填补
- 用机器学习算法预测缺失值

本案例中方法:
- 在training set, 我们用0来替换所有缺失值。
- 在test set 数据有缺失, 我们将其遗弃。

### 1.2 `确保数据集是[干净并可用]`

## 2. Logistic 回归分类器

- `def loadData`: 读数据
- `def sigmoid`: sigmoid function
- `def train`: MBGD(mini-batch gradient descent)
- `def test`: predict by test data
- `def classifier`: classify data