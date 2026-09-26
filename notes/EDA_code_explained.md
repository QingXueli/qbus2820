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
- 从这里发现 `DistanceCBD` 最大值正好 40、`FloorArea` 最小值正好 35 → 可能是截断值，在 2.11 总结里说明。

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

### 2.5 响应变量分布

**思路（为什么做）**：看要预测的变量 `WeeklyRent` 本身长什么样——集中在哪、对不对称、有没有长尾。
如果严重右偏（多数房子便宜、少数豪宅特别贵），通常要先对租金取对数 (log) 再建模，否则豪宅会过度影响模型。
这一步决定**要不要对响应变量做变换**。

**代码（写法同 W1、W2：`plt.figure()` → `plt.hist()` → 标签/标题 → `plt.show()`）**

```python
fig = plt.figure()                              # 新建一张空白图

plt.hist(train[response], bins=50)              # 租金直方图；response = 'WeeklyRent'
                                                # bins=50：把租金范围切成 50 段，柱高 = 这一段里的房子数
plt.xlabel("Weekly Rent (AUD)")                 # 横轴：租金（澳元）
plt.ylabel("Number of Properties")              # 纵轴：房子数量
plt.title("Distribution of Weekly Rent")        # 标题

plt.show()                                      # 显示图

print('Mean: {0:.4f}, median: {1:.4f}, std: {2:.4f}, skewness: {3:.4f}'.format(
    train[response].mean(), train[response].median(), train[response].std(), train[response].skew()))
```
- `{0:.4f}`：把 `.format()` 里第 0 个值填进来，保留 4 位小数；`{1:.4f}` 是第 1 个值，以此类推（W1 `"{0:.2f}".format(...)` 的写法）。
- `.mean()` 均值、`.median()` 中位数、`.std()` 标准差、`.skew()` 偏度。
- `.median()`、`.skew()` 不在 tutorial 里（W1 只用 `.mean()`、`.var()`），但属于同类的基础 pandas 函数，按决定不加 NOTE。

**图怎么读**
- 钟形 (bell shape)：中间高、两边低、左右大致对称。
- 大部分房子在 **600–1100 澳元**，最高的柱子在 850 左右。
- 两边尾巴差不多长（最低约 200，最高约 1740），**没有一条特别长的右尾**。

**数字怎么读**

| 统计量 | 值 | 含义 |
|---|---|---|
| 均值 (mean) | 860.6811 | 平均每周租金约 861 澳元 |
| 中位数 (median) | 849.0900 | 一半房子低于 849，一半高于 |
| 标准差 (std) | 215.4138 | 租金大多在均值上下约 215 澳元范围内 |
| 偏度 (skewness) | 0.2660 | 衡量是否对称（见下） |

**偏度 (skewness)**
- = 0：完全对称；> 0：右偏（右尾更长，少数贵房子把分布往右拉）；< 0：左偏。
- 经验标准：|偏度| < 0.5 基本对称；> 1 明显偏斜，可以考虑取 log。
- 本数据 0.27 → 基本对称。均值 861 与中位数 849 只差 12 澳元，也说明没有被少数豪宅拉高。

**结论与对建模的影响**
- 租金分布大致对称、接近正态，没有严重长尾 → **不需要对租金取 log**，直接在原始澳元尺度上建模。
- 好处：作业按**原始尺度的 MSE** 打分；不取 log 就不用把预测值转换回去，也避免了转换带来的偏差。

**可写在 notebook 这一节后面的结论（英文 Markdown）**
```markdown
The distribution of weekly rent is roughly symmetric and bell-shaped (mean 860.68, median 849.09, skewness 0.27),
with no long tail. Therefore the response is modelled on its original scale, which also matches the scale
on which the test MSE is calculated.
```

### 2.6 租金 vs 连续变量（散点 + 直线 + 二次曲线）

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
    # 直线：rent = b0 + b1·x
    lin_reg = LinearRegression()                          # 建一个普通最小二乘 (OLS) 模型
    lin_reg.fit(x, y)                                     # 用 x 拟合租金

    # 二次曲线：rent = b0 + b1·x + b2·x²（W6 的拆开写法）
    poly_transformer = PolynomialFeatures(2)              # 准备把 x 变成 [1, x, x²]
    poly_x = poly_transformer.fit_transform(x)            # 新的 3 列数据
    poly_reg = LinearRegression()                         # 另一个 OLS 模型
    poly_reg.fit(poly_x, y)                               # 用 [1, x, x²] 拟合租金 → 得到 b0、b1、b2
```

```python
    plt.scatter(x, y, alpha = 0.1, s = 5)                 # 5000 个点很密：alpha 调透明、s 调小点
    plt.plot(x_points, lin_reg.predict(x_points), color = "red", label = "Linear fit")
    plt.plot(x_points, poly_reg.predict(poly_transformer.transform(x_points)), color = "green", label = "Quadratic fit")
```
- 注意第二行用 `transform`（不是 `fit_transform`）：用同一个转换器把画图用的点也变成 `[1, x, x²]`。

### 2.7 箱线图和分组均值 【箱线图为非 tutorial，CLAUDE.md 要求】

**思路**：比较不同组（例如高需求区 0/1）的租金分布和平均值。

```python
group_vars = ['Bedrooms'] + binary                 # 列表相加：['Bedrooms', 'NearTrain', 'Furnished', 'HighDemandArea']

for i, var in enumerate(group_vars):
    levels = sorted(train[var].unique())           # 这个变量有哪些取值，如 [0, 1] 或 [1,2,3,4,5]

    groups = []                                    # 空列表
    for level in levels:                           # 对每个组
        rent_in_group = train.loc[train[var] == level, response]   # 取出这一组的租金
        groups.append(rent_in_group)                               # 放进列表

    ax[i].boxplot(groups)                          # 列表里有几组，就画几个箱子
```
- 以 NearTrain 为例：`groups` = [2690 个不近车站的租金, 2310 个近车站的租金] → 两个箱子。
- 「先建空列表、循环里 `append`」是 tutorial 调参时的同款写法（W3/W5 的 `losses.append(loss)`）。
- 箱子中线 = 中位数；箱体 = 25%–75%；须线外的点 = 可能的异常值。

```python
    ax[i].set_xticklabels(levels)                  # 横轴标上组名
```

```python
train.groupby(var)[response].agg(['count', 'mean']).round(4)
```
- `groupby`：按组分开；`.agg(['count','mean'])`：每组算个数和平均租金。
- 得出：高需求区 +222、近车站 +137、带家具 +83 澳元。

### 2.8 相关性

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

### 2.9 交互项探索

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

**第二步（AIC 筛查）已移到建模部分**：它需要拟合、比较模型，属于建模步骤。代码暂存在 `notes/phase2_candidate_screening.md`。

### 2.10 基线 OLS 和异常值检查

**思路**：用 7 个线性项拟合一个基准 OLS（W7 写法），看整体效果（R² = 0.929，系数、p 值），并用它的残差检查异常值。
残差 = 实际租金 − 预测租金，每套房子一个值（`est.resid`），不在 summary 表里。残差图已按决定删除。

```python
x_with_intercept = sm.add_constant(train[predictors], prepend=True)   # 加截距列
ols = sm.OLS(train[response], x_with_intercept)                       # 租金 ~ 7 个变量
est = ols.fit()
print(est.summary())                                                  # 系数、p 值、R²、AIC、BIC
residuals = est.resid                                                 # 每套房子的残差
```

**异常值检查（标准化残差）**

**思路**：回归里的异常值 = 模型预测得特别离谱的点。「多离谱才算离谱」要有一把尺子：把残差除以误差的标准差，
得到标准化残差（多少个标准差）。如果误差近似正态，99.73% 的点应落在 ±3 之内，超出 ±3 的就算异常。

```python
from scipy import stats                        # 【W7 c24 导入过】正态分布的概率
sigma_hat = np.sqrt(est.mse_resid)             # 误差标准差的估计 = √(残差均方)；W6 的 sigma2_hat 同一概念
std_resid = residuals / sigma_hat              # 标准化残差：这套房子的误差是几个标准差
outliers = np.abs(std_resid) > 3               # 绝对值超过 3 → True
expected = len(train) * 2 * (1 - stats.norm.cdf(3))   # 纯属偶然、预计会有几个：5000 × P(|Z|>3) ≈ 13.5
```
- `stats.norm.cdf(3)`：标准正态小于 3 的概率（0.99865）；`1 - ...` 是大于 3 的概率；`× 2` 加上小于 −3 的一侧。

```python
outlier_rows = train[outliers].copy()                  # 取出被标记的行
outlier_rows['std_resid'] = std_resid[outliers]        # 加一列：它们的标准化残差
outlier_rows.sort_values('std_resid').round(4)         # 按残差从负到正排序
```
- 结果：17 个（预期 13.5，差不多），最大 4.24 → 没有极端异常值。
- 规律：正残差（低估）多是离 CBD 30–40 km 的远郊；负残差（高估）多是 5 间卧室的大房子 → 是线性模型没拟合好弯曲，不是数据错误 → 全部保留。

### 2.11 结论总结

- 纯 markdown，没有代码：把以上发现整理成 9 条，并说明它们对建模的意义。
