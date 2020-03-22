>[NN](https://www.ctolib.com/topics-138867.html)

# 识别手写数字

# 1. 加载数据
数据说明: 
- 训练集合共100个数据, 每一行长度为785, 第一个数字为真实的值, 后面为图像的像素值
- 流行的图像处理软件通常用8位表示一个像素, 这样总共有256个灰度等级 (像素值在0~255)
- 需要对每个像素值进行处理每一项除以 255 在乘以 9, 将每个数字转为 0 ~ 9 的个位数

```python
# 例如第3行的数据, str - arr
item_arr = data_array[3].split(',')
# image_array = (np.asfarray(item_arr[1:]) / 255 * 9).astype(int).reshape(28, 28)
"""
(1) 像素处理: 第一个数字为真实的值, 后面为图像的像素值: item_arr[1:]
(2) 像素转化: item_arr[i] / 255 * 9, 8位一个像素点, *9(我们的数字有0-9)
(3) 像素矩阵: reshape(28, 28)
"""
image_array = np.array([int(float(item_arr[i]) / 255 * 9)  for i in range(len(item_arr[1:]))])
image_array = image_array.reshape(28, 28)

"""
image_array: 
[[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 1 4 7 5 4 4 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 3 8 8 8 8 8 8 2 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 8 8 8 8 8 8 8 2 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 3 8 8 7 8 8 8 4 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 7 8 7 8 8 8 1 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 2 7 8 8 8 8 1 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 7 8 8 8 6 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 3 4 8 8 8 8 3 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 1 7 8 8 8 8 8 8 1 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 2 8 8 8 8 8 8 8 5 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 3 8 8 8 8 8 8 6 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 2 3 3 3 8 8 6 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 8 7 1 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 8 8 7 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 8 8 4 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 6 3 0 0 0 0 0 0 8 8 8 1 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 5 8 7 2 1 0 1 3 7 8 8 4 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 2 6 8 8 7 6 8 8 8 8 7 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 4 3 8 8 8 8 8 8 8 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 4 4 5 8 8 8 1 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
"""
```
![image]()



# 2. 构建神经网络

(1) 初始化函数
- 每次layer的neuron的数量
- 每个neuron的 weights 和 bias (初始化随机生成)

(2) 训练模型

(3) 正向传播

## 2.1 初始化
```python
class NerualNetwork():
    def __init__ (self):
        pass
    
    def train (self):
        pass
    
    def forward_propagation (self):
        pass
```
 
这里特殊说明: 每个neuron的weights (初始化随机生成)
![image]()

例如输入层(4个结点), 隐藏层(3个结点), 输出层(2个结点)
- 隐藏层有 3 * 4的矩阵 (因为隐藏层要连接上一层的4个结点)
- 输出层有 2 * 3的矩阵 (因为隐藏层要连接上一层的3个结点)

**numpy.random.rand(x, y) - 0.5**

**减去0.5是为了保证生成的权重所有权重都能维持在 -0.5 ~ 0.5 之间的一个随机值**

```python
def __init__ (self, input_num, hidden_num, output_num, alpha):
    # 设置 input layer, hidden layer, output layer neuron的数量
    self.input_num = input_num
    self.hidden_num = hidden_num
    self.output_num = output_num
    
    # weights: 从输入层到隐藏层, 从隐藏层到输出层
    self.input_hidden_weights = np.random.rand(hidden_num, input_num) - 0.5
    self.hidden_output_weights = np.random.rand(output_num, hidden_num) - 0.5

    # learning rate
    self.alpha = alpha
    
    # sigmoid function
    self.sigmoid_fn = lambda z: 1 / (1 + np.exp(-z))

    # sigmoid derivatives
    self.sigmoid_derivatives = lambda activiation: activiation * (1 - activiation)

    pass
```


## 2.2 正向传播
```python
def forward_propagation (self, inputs_list):
    # 1. 将输入的数组转化为一个二维数组
    inputs = np.array(inputs_list, ndmin=2).T
    # [1,2,3] => [[1,2,3]].T
    
    # 2. hidden layers (z = hidden_inputs; a = sigmoid(z) = hidden_outputs)
    hidden_inputs = np.dot(self.input_hidden_weights, inputs)
    hidden_outputs = self.sigmoid_fn(hidden_inputs)
    
    # 3. output layers
    final_inputs = np.dot(self.hidden_output_weights, hidden_outputs)
    final_outputs = self.sigmoid_fn(final_inputs)

    return final_outputs
```

## 2.3 训练

1. 输入值 和 目标值 处理
```python
inputs = np.array(inputs_list, ndmin = 2).T
targets = np.array(target_list, ndmin = 2).T

"""
eg: inputs_list = [1, 2, 3]
np.array(inputs_list, ndmin = 2): [[ 1, 2, 3 ]]

inputs = [
    [1],
    [2],
    [3]
]
"""
```

1. forward propagation
```python
# 2. 向前传播
# 隐藏层输入输出
hidden_inputs = np.dot(self.input_hidden_weights, inputs)
hidden_outputs = self.sigmoid_fn(hidden_inputs)
# 最后层输入输出
final_inputs = np.dot(self.hidden_output_weights, hidden_outputs)
final_outputs = self.sigmoid_fn(final_inputs)
```

3. 每层的差值

```python
# 3. 预测层 和 目标值 差值
# 差值 = 目标值 - 输出的预测值
output_errors = targets - final_outputs
# 差值 = 隐藏层的差值
hidden_errors = np.dot(self.hidden_output_weights.T, output_errors)
```

4. back propagation
$dy / dw = (dy / da) * (da / dz) * (dz / dw)$
- $dy / da$ = 1/2 * 2 * (y - a) = 差值 errors
- $da / dz$ = sigmoid_derivatives(a)
- $dz / dw$ = x

X * sigmoid_derivatives(a) * error