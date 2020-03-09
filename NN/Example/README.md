# Exercise

## 1. Data
|  Name   | Weight  | Height | Gender
|  ----  | ----  | ----  | ----  |
| A | 133 | 65 | F
| B | 160 | 72 | M
| C | 152 | 70 | M
| D | 120 | 60 | F

Aim: redict gender by weight and height

## 2. Create a NN

1. Features: [Weight, Height] -> x1, x2
```python
Features = np.array([x1, x2])
```

2. DataSet 标准化数据
> I arbitrarily chose the shift amounts (135135 and 6666) to make the numbers look nice. Normally, you’d shift by the mean.
> 选取了135和66来标准化数据，通常会使用平均值
> Weight: -135, Height: -66
> F: 1, M: 0

(-2, -1) 1
(25, 6) 0
(17, 4) 0
(-15, -6) 1

3. Layers
- input: Features x1, x2
- output: F(=1), M(=0)
- hidden: h1, h2
