# EDA 代码逐行解释（notebook 第 1–2 节）

> 对照 `SID_Assignment1_implementation.ipynb` 阅读。每段先讲**思路（为什么做）**，再逐行讲**代码（怎么做）**。
> 标注 【W5】【W7】等 = 该写法出自第几周 tutorial；【非 tutorial】= tutorial 中没有，已在 notebook 里用 `# NOTE` 标出。

---

## 1. Setup and data

### Cell：导入库

**思路**：把后面反复用到的工具一次性导入。tutorial W5/W7 的第一个 cell 也是这样做的。

```python
import pandas as pd                 # 表格数据：读 CSV、算统计量、分组        【W1 起】
import numpy as np                  # 数值计算：linspace、log 等             【W1 起】
import matplotlib.pyplot as plt     # 画图                                   【W1 起】
import seaborn as sns               # 更方便的统计图，这里只用于热力图         【W2 pairplot】
import statsmodels.api as sm        # OLS 回归，输出 summary、AIC、p 值       【W2、W7】
import warnings                     # Python 自带的警告管理                  【W7】
warnings.filterwarnings('ignore')   # 不显示警告（本 notebook 实测不产生警告） 【W7】
%matplotlib inline                  # 让图显示在 notebook 里（新版 Jupyter 默认如此） 【W5、W7】
sns.set_context('notebook')         # 图中字体大小适合 notebook 阅读           【W5、W7】
sns.set_style('ticks')              # 白底、坐标轴带刻度的图风格               【W7】
```
- 图只用 `plt.show()` 显示在 notebook 里，不另存为图片文件（写报告时直接截图），所以不需要 `os` 和 `plt.savefig`。

### Cell：读数据、定义变量名

**思路**：读入两份数据；把「响应变量名」和「预测变量名列表」存进变量，后面所有代码都引用它们，不用反复手打列名（W7 c8 的做法）。

```python
train = pd.read_csv('WeeklyRent_training.csv')        # 训练集 5000 行，含 WeeklyRent
test = pd.read_csv('WeeklyRent_test_noLabel.csv')     # 测试集 1000 行，无 WeeklyRent
```
- 相对路径：作业规定 CSV 与 notebook 在同一文件夹。

```python
response = 'WeeklyRent'                                                  # 要预测的列
predictors = [x for x in list(train.columns) if x != response]           # 其余 7 列都是预测变量
```
- 列表推导式 (list comprehension)：遍历所有列名，留下「不是 WeeklyRent」的。与 W7 `predictors=[x for x in list(data.columns) if x not in exclude]` 同一写法。

```python
continuous = ['DistanceCBD', 'FloorArea', 'PropertyAge']   # 连续变量：画散点、检查非线性
binary = ['NearTrain', 'Furnished', 'HighDemandArea']      # 0/1 变量：画箱线图、分组比较
train.head()                                               # 看前 5 行，确认读对了
```

---

## 2. Exploratory data analysis

### 2.1 数据结构

**思路**：先知道数据有多大、每列是什么类型，决定后续怎么处理（比如文字列要做 dummies）。

```python
print('Training data: {} rows, {} columns'.format(train.shape[0], train.shape[1]))
```
- `train.shape` 返回 `(行数, 列数)`，`[0]` 取行数、`[1]` 取列数。
- `'{}'.format(...)` 把数字填进 `{}` 的位置（W5 的打印风格）。

```python
train.dtypes      # 每列的数据类型：float64 = 小数，int64 = 整数
```
- 结论：全是数值，没有文字列，所以**不需要** `pd.get_dummies`。

### 2.2 缺失值和重复行

```python
train.isna().sum().sum()      # isna() 把每个格子变成 True/False（是否缺失）；
                              # 第一个 sum() 按列数缺失个数，第二个 sum() 把所有列加总
train.duplicated().sum()      # duplicated() 标记「和前面某行完全相同」的行；sum() 数个数
```
- 结果都是 0 → 不需要填补缺失或删重复。

### 2.3 描述统计

```python
train.describe().T.round(4)
```
- `describe()`：每列的 count、mean、std、min、四分位数、max。
- `.T`：转置 (transpose)，让每个变量占一行，更好读。
- `.round(4)`：保留 4 位小数（作业要求）。
- 从这里发现 `DistanceCBD` 最大值正好 40、`FloorArea` 最小值正好 35 → 2.5 节去查。

### 2.4 训练集 vs 测试集

**思路**：模型在训练集上学、在测试集上被打分。如果测试集的变量范围和训练集差很多，模型就要「外推」，预测会不准。所以先比较两者的分布。

```python
comparison = pd.DataFrame({
    'Train mean': train[predictors].mean(),
    'Test mean': test[predictors].mean(),
    ...
})
comparison.round(4)
```
- 用字典 `{列名: 数据}` 建一个新表，每列是一种统计量，每行是一个变量，并排比较。

```python
fig, ax = plt.subplots(2, 4, figsize=(16, 7))    # 一张大图，2 行 4 列共 8 个小图；figsize 是宽×高（英寸）
ax = ax.flatten()                                # 把 2×4 的小图数组拉平成 1 维，方便用 ax[0]…ax[7]

for i, var in enumerate(predictors):             # enumerate 同时给出序号 i 和变量名 var
    ax[i].hist(train[var], bins=30, density=True, alpha = 0.5, label = "Train")
    ax[i].hist(test[var], bins=30, density=True, alpha = 0.5, label = "Test")
```
- 两个直方图叠在一起比较（W1 task 2 用 `alpha = 0.5` 叠加男女消费直方图，同一写法）。
- `bins=30`：分成 30 个柱子。
- `density=True`：纵轴画「比例」而不是「个数」。训练集 5000 行、测试集 1000 行，画个数的话高度没法比。
- `alpha = 0.5`：半透明，两层都看得见。

```python
    ax[i].set_xlabel(var); ax[i].set_ylabel("Density"); ax[i].legend()   # 坐标轴标签、图例
ax[7].axis('off')                         # 只有 7 个变量，第 8 个格子关掉
fig.suptitle("...")                       # 整张大图的总标题
plt.tight_layout()                        # 自动调整间距，防止标签重叠
plt.show()                                # 显示图
```

### 2.5 截断值和房龄长尾

**思路**：如果很多行都「刚好」等于最大值 40，而第二大的值是 39.65，说明 40 以上的都被记成了 40（截断，capping）。这会影响模型在边界的表现。

```python
n_dist_40 = (train['DistanceCBD'] == 40).sum()
```
- `== 40` 得到 True/False 序列；`.sum()` 时 True 算 1 → 数出等于 40 的行数（43）。

```python
next_largest = train.loc[train['DistanceCBD'] < 40, 'DistanceCBD'].max()
```
- `.loc[行条件, 列名]`：取出所有「小于 40」的行的 DistanceCBD，再取最大值（39.65）。

- FloorArea = 35 同理（106 行，下一个值 35.2）。

```python
train['PropertyAge'].quantile([0.5, 0.9, 0.95, 0.99])   # 中位数和 90/95/99% 分位数
(train['PropertyAge'] > 60).sum()                       # 房龄超过 60 年的有几套
train['PropertyAge'].nlargest(10).tolist()              # 最大的 10 个值
```
- 用来判断长尾是否「离谱」。结论：最大 100 年，现实中合理，保留。

### 2.6 响应变量分布

```python
plt.hist(train[response], bins=50)     # 租金直方图
train[response].skew()                 # 偏度：0 = 对称；> 1 属于明显右偏，才考虑取 log
```
- 偏度 0.27 → 接近对称 → 不需要对租金取对数。

### 2.7 租金 vs 连续变量（散点 + 直线 + 二次曲线）

**思路**：散点图看关系形状。同时画一条直线和一条二次曲线：如果二次曲线明显弯离直线，说明关系是非线性的，模型要加平方项。拟合方法用 W6 的 `PolynomialFeatures` + `LinearRegression`。

```python
from sklearn.preprocessing import PolynomialFeatures      # 【W6】
from sklearn.linear_model import LinearRegression         # 【W6】

for i, var in enumerate(continuous):                      # 对 3 个连续变量各画一张图
    x = train[[var]].values                               # 双中括号 → 取成「表」(二维)，.values 变成数组；sklearn 要求 X 是二维
    y = train[response]
    x_points = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
```
- `np.linspace(最小, 最大, 100)`：在数据范围内均匀取 100 个点，用来画平滑曲线（W2 画回归线的做法）。
- `.reshape(-1, 1)`：变成 100 行 1 列，符合 sklearn 输入格式（W6 `x.reshape(-1,1)`）。

```python
    lin_reg = LinearRegression().fit(x, y)                # 直线：rent = b0 + b1·x
    poly_transformer = PolynomialFeatures(2)              # 把 x 变成 [1, x, x²]
    poly_reg = LinearRegression().fit(poly_transformer.fit_transform(x), y)   # 二次曲线：rent = b0 + b1·x + b2·x²
```

```python
    plt.scatter(x, y, alpha = 0.1, s = 5)                 # 5000 个点很密：alpha 调透明、s 调小点
    plt.plot(x_points, lin_reg.predict(x_points), color = "red", label = "Linear fit")
    plt.plot(x_points, poly_reg.predict(poly_transformer.transform(x_points)), color = "orange", label = "Quadratic fit")
```
- 注意第二行用 `transform`（不是 `fit_transform`）：用同一个转换器把画图用的点也变成 `[1, x, x²]`。

### 2.8 箱线图和分组均值 【箱线图为非 tutorial，CLAUDE.md 要求】

**思路**：比较不同组（例如高需求区 0/1）的租金分布和平均值。

```python
group_vars = ['Bedrooms'] + binary                 # 列表相加：['Bedrooms', 'NearTrain', 'Furnished', 'HighDemandArea']

for i, var in enumerate(group_vars):
    levels = sorted(train[var].unique())           # 这个变量有哪些取值，如 [0, 1] 或 [1,2,3,4,5]
    ax[i].boxplot([train.loc[train[var] == level, response] for level in levels])
```
- 列表推导式为每个取值取出一组租金，`boxplot` 为每组画一个箱子。
- 箱子中线 = 中位数；箱体 = 25%–75%；须线外的点 = 可能的异常值。

```python
    ax[i].set_xticklabels(levels)                  # 横轴标上组名
```

```python
train.groupby(var)[response].agg(['count', 'mean']).round(4)
```
- `groupby`：按组分开；`.agg(['count','mean'])`：每组算个数和平均租金。
- 得出：高需求区 +222、近车站 +137、带家具 +83 澳元。

### 2.9 相关性

**思路**：相关系数衡量两个变量的**线性**关系强弱（−1 到 1）。两个用途：(1) 看哪些变量和租金关系强；(2) 看预测变量之间是否高度相关（多重共线性）。

```python
correlations = train.corr()        # 【W2、W5】所有变量两两的相关系数
correlations.round(4)
```

```python
sns.heatmap(correlations, annot = True, fmt = ".2f", cmap = "RdBu_r", vmin = -1, vmax = 1)   # 【非 tutorial，CLAUDE.md 要求】
```
- `annot=True`：格子里写上数字；`fmt=".2f"`：两位小数。
- `cmap="RdBu_r"`：红 = 正相关，蓝 = 负相关；`vmin/vmax` 固定色阶在 −1 到 1。
- 发现：Bedrooms 与 FloorArea 相关 0.88 → 多重共线性。

### 2.10 交互项探索

**思路（第一步：画图）**：对每组分别拟合一条直线；若两条线斜率差很多，说明「这个变量的影响在两组不一样」= 交互效应。

```python
pairs = [('DistanceCBD', 'HighDemandArea'), ...]            # 要检查的 (变量, 分组变量) 组合

for i, (var, group) in enumerate(pairs):
    for level, color in zip([0, 1], ["#1F77B4", "#FF7F0E"]):   # 组 0 蓝色、组 1 橙色（W5 的两种颜色）
        subset = train[train[group] == level]                   # 只取这一组的数据
        x_with_intercept = sm.add_constant(subset[[var]], prepend=True)   # 加截距列 【W2、W7】
        est = sm.OLS(subset[response], x_with_intercept).fit()            # 组内简单回归 【W2、W7】
        x_points = np.linspace(train[var].min(), train[var].max(), 100)
        ax[i].plot(x_points, est.params.iloc[0] + est.params.iloc[1] * x_points, ...)
```
- `est.params.iloc[0]` = 截距，`iloc[1]` = 斜率；用 `b0 + b1·x` 画线（W2 c30 同一写法）。
- 图例里写上斜率，便于比较。

**思路（第二步：控制其他变量后的筛查）**：分组画图会被其他变量混淆（例如高需求区本来就更靠近 CBD）。所以在「含全部 7 个变量的 OLS」基础上，**每次只加一个候选项**，看 AIC 降了多少（W7 用 AIC 判断变量是否值得保留）。

```python
import itertools                                             # 【非 tutorial】Python 自带，用来生成所有两两组合

x_with_intercept = sm.add_constant(train[predictors], prepend=True)
base = sm.OLS(train[response], x_with_intercept).fit()       # 基准模型：7 个线性项
```

```python
for var1, var2 in itertools.combinations(predictors, 2):     # 7 个变量两两组合，共 21 对
    x = train[predictors].copy()                             # 复制一份，避免改动原数据
    x['new_term'] = x[var1] * x[var2]                        # 交互项 = 两列相乘
    est = sm.OLS(train[response], sm.add_constant(x, prepend=True)).fit()
    screening.append({'Term': var1 + ' x ' + var2,
                      'AIC drop': base.aic - est.aic,        # AIC 降幅：越大 = 这一项越有用
                      'p-value': est.pvalues['new_term']})   # 这一项系数的 p 值
```
- 平方项同理：`x['new_term'] = x[var] ** 2`（只对非 0/1 变量，因为 0² = 0、1² = 1，没有新信息）。

```python
screening = pd.DataFrame(screening).sort_values('AIC drop', ascending = False).set_index('Term')
```
- 把结果做成表（W6 c27 用 list of dicts → DataFrame → `set_index("Model")` 的同一写法），按 AIC 降幅从大到小排。

### 2.11 基线 OLS 残差

**思路**：残差 = 真实值 − 模型预测值。如果模型抓住了所有规律，残差应该是围绕 0 的随机云；若残差随某变量呈弯曲形状，说明模型漏掉了该变量的非线性。

```python
x_with_intercept = sm.add_constant(train[predictors], prepend=True)   # 【W7 c24】
ols = sm.OLS(train[response], x_with_intercept)
est = ols.fit()
print(est.summary())         # 系数、p 值、R²、AIC、BIC
```

```python
residuals = est.resid             # 每套房的残差
fitted = est.fittedvalues         # 每套房的预测值
ax[0].scatter(fitted, residuals, ...)      # 残差 vs 预测值：检查整体模式
ax[0].axhline(0, color = "red")            # 画 y = 0 参考线
```

```python
poly_reg = LinearRegression().fit(poly_transformer.fit_transform(x), residuals)
```
- 对「残差 vs 变量」再拟合一条二次曲线（橙线），让弯曲趋势一目了然：距离呈 U 形、面积呈倒 U 形。

### 2.12 结论总结

- 纯 markdown，没有代码：把以上发现整理成 9 条，并说明它们对建模的意义。
