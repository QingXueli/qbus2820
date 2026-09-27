# 第 3 节（建模与模型选择）代码逐行解释

> 对照 `SID_Assignment1_implementation.ipynb` 第 3 节阅读。每一小节：**思路 → 逐行解释 → 可直接复制的英文注释版**。
> 【W5】等表示写法出自哪一周 tutorial，【L4–5】表示出自哪一讲 lecture。

---

## 整体思路（先看这个）

```
3.1 准备「考场」：定义同一套 10 折（kf）和一个打分函数（cv_mse）
3.2–3.9 每个模型：建好 → 立刻用 cv_mse 打分 → 分数存进 results 列表
3.10 把 results 变成一张表，按 CV MSE 从小到大排，选冠军
```

- **CV MSE**：把训练集分成 10 份，每次用 9 份训练、1 份当「新题」测试，10 次测试误差的平均。越小越好。
- **SE（标准误）**：10 次误差的波动程度。两个模型的 CV MSE 相差小于约 1 个 SE，就可以认为「差不多」。

---

## 3.1 Cross-validation set-up

### 思路
所有模型必须在**同一套分折**上比较才公平（W6 原话：*"To ensure that each model uses the same K-Folds … store the K-Fold split into a Python object"*）。
再写一个函数 `cv_mse`，以后每个模型只要一行就能算出 CV MSE。

### 逐行
```python
from sklearn.model_selection import KFold, cross_val_score
```
- 导入两个 CV 工具【W5】：`KFold` 负责「怎么分折」，`cross_val_score` 负责「每一折训练 + 打分」。

```python
y = train[response]
```
- 把租金这一列存成 `y`，后面反复用，省得每次写 `train['WeeklyRent']`。

```python
kf = KFold(10, shuffle=True, random_state=1)
```
- `10`：分 10 折。
- `shuffle=True`：分之前先**打乱顺序**。W5 说默认不打乱；我们的数据不是随机排列的，所以要打乱【W5 注释里的写法】。
- `random_state=1`：固定随机种子 → 每次运行分得一模一样，结果可复现；也保证**所有模型用同一套分折**。

```python
def cv_mse(model, X):
```
- 定义一个函数：给它「一个模型」和「用哪些列」，它返回 CV MSE 和 SE。W5 的 `knn_test` 也是自己写的函数。

```python
    scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
```
- 对 10 折分别「用 9 折训练、在第 10 折上算 MSE」，返回 10 个分数【W5】。
- `scoring='neg_mean_squared_error'`：sklearn 规定「分数越高越好」，所以给的是**负的** MSE【W5 讲过】。

```python
    fold_mse = -scores
```
- 加负号，变回正常的 MSE（10 个数）。

```python
    mean_mse = np.mean(fold_mse)
```
- 10 个 MSE 求平均 = **CV MSE**。

```python
    se = np.std(fold_mse, ddof=1) / np.sqrt(len(fold_mse))
```
- 标准误 = 10 个 MSE 的标准差 ÷ √10。
- `ddof=1`：样本标准差（分母 n−1）。

```python
    return mean_mse, se
```
- 把两个数一起返回；调用时写 `mse, se = cv_mse(...)` 一次接住两个。

```python
results = []
```
- 空列表，每个模型算完 CV 就 `append` 一条记录进去，最后变成比较表【W6 c27 的写法】。

### 英文注释版
```python
from sklearn.model_selection import KFold, cross_val_score

y = train[response]                              # weekly rent, used by every model

# 10 folds; shuffle because the file is not in random order; random_state fixes the folds
# so every model is evaluated on exactly the same folds (Tutorial 6)
kf = KFold(10, shuffle=True, random_state=1)

def cv_mse(model, X):
    # Train on 9 folds and compute the MSE on the 10th, for each of the 10 folds
    # scikit-learn returns the NEGATIVE MSE (its convention: higher score = better)
    scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    fold_mse = -scores                                      # turn back into positive MSEs
    mean_mse = np.mean(fold_mse)                            # CV MSE = average of the 10 fold MSEs
    se = np.std(fold_mse, ddof=1) / np.sqrt(len(fold_mse))  # standard error of the CV MSE
    return mean_mse, se

results = []   # each model adds one row here; turned into a comparison table at the end
```

---

## 3.2 Baseline models: M0 and M1

### 思路
- **M0**：什么都不学，永远猜平均租金。是「最差也该达到」的底线。
- **M1**：7 个变量的普通线性回归，最简单的真正模型。

### M0 逐行
```python
fold_mse = []
for train_index, valid_index in kf.split(train):
```
- `kf.split(train)` 依次给出每一折的**训练行号**和**验证行号**。循环 10 次。
- M0 没有现成的 sklearn 模型，所以手写循环（和 `cross_val_score` 做的事一样）。

```python
    prediction = y.iloc[train_index].mean()
```
- `y.iloc[train_index]`：取出训练那 9 折的租金；`.mean()`：平均值 → 就是 M0 的预测（约 861）。
- 注意只用**训练折**算平均，验证折不参与，否则就「偷看答案」了。

```python
    fold_mse.append(np.mean((y.iloc[valid_index] - prediction) ** 2))
```
- 验证折的真实租金 − 预测 → 平方 → 平均 = 这一折的 MSE，存进列表。

```python
m0_mse = np.mean(fold_mse)
m0_se = np.std(fold_mse, ddof=1) / np.sqrt(len(fold_mse))
```
- 和 `cv_mse` 函数里一样：10 折平均 = CV MSE，再算标准误。

```python
results.append({'Model': 'M0: Null model', 'CV MSE': m0_mse, 'SE': m0_se})
```
- 把结果作为一个字典（名字、CV MSE、SE）存进 `results`。

```python
print('M0 prediction on all training data: {0:.4f} AUD'.format(y.mean()))
print('M0 CV MSE: {0:.4f} (SE {1:.4f})'.format(m0_mse, m0_se))
```
- 打印结果，4 位小数。结果：预测 860.6811，CV MSE 46409.4061。

### M1 逐行
```python
from sklearn.linear_model import LinearRegression
```
- sklearn 的 OLS【W6】。CV 要用 sklearn 的模型（`cross_val_score` 需要），所以这里不用 statsmodels。两者算出的系数完全一样。

```python
mse, se = cv_mse(LinearRegression(), train[predictors])
```
- 一行搞定：用 7 个原始变量做 OLS，算 10 折 CV。结果 CV MSE 3302.3821。

```python
results.append({'Model': 'M1: OLS, 7 linear terms', 'CV MSE': mse, 'SE': se})
print('M1 CV MSE: {0:.4f} (SE {1:.4f})'.format(mse, se))
```

### 英文注释版
```python
# M0: predict the average rent of the training folds; manual loop over the same folds
fold_mse = []
for train_index, valid_index in kf.split(train):              # row numbers of the 9 training folds and the validation fold
    prediction = y.iloc[train_index].mean()                    # mean rent of the training folds only (no peeking)
    fold_mse.append(np.mean((y.iloc[valid_index] - prediction) ** 2))   # MSE on the validation fold

m0_mse = np.mean(fold_mse)                                     # CV MSE
m0_se = np.std(fold_mse, ddof=1) / np.sqrt(len(fold_mse))      # standard error
results.append({'Model': 'M0: Null model', 'CV MSE': m0_mse, 'SE': m0_se})
```
```python
from sklearn.linear_model import LinearRegression

# M1: OLS with the seven original predictors, evaluated by 10-fold CV
mse, se = cv_mse(LinearRegression(), train[predictors])
results.append({'Model': 'M1: OLS, 7 linear terms', 'CV MSE': mse, 'SE': se})
```

---

## 3.3 M2: best subset selection

### 思路（L4–5 的三步）
1. 每个大小 k（1–7）：把所有 k 个变量的组合都拟合一遍。
2. 每个大小里，留下 **RSS 最小**的那个（同样大小时 RSS 小 = 拟合好）。
3. 7 个「每个大小的冠军」再用 **CV** 比，选出最终大小。

### 逐行
```python
import itertools
```
- Python 自带工具，`itertools.combinations(列表, k)` 列出「从列表里挑 k 个」的所有组合。tutorial 没有 → 标了 NOTE。

```python
best_subsets = []
for k in range(1, len(predictors) + 1):
```
- `range(1, 8)` → k = 1, 2, …, 7。

```python
    best_rss = np.inf
```
- 先把「目前最好的 RSS」设成无穷大，这样第一个组合一定能更新它。

```python
    for subset in itertools.combinations(predictors, k):
```
- 例如 k=2 时，依次得到 (DistanceCBD, Bedrooms)、(DistanceCBD, FloorArea)……共 21 个组合。

```python
        x_with_intercept = sm.add_constant(train[list(subset)], prepend=True)
        est = sm.OLS(y, x_with_intercept).fit()
```
- 用这个组合拟合 OLS【W7 写法】。

```python
        if est.ssr < best_rss:
            best_rss = est.ssr
            best_subset = list(subset)
```
- `est.ssr` = 残差平方和 RSS。比目前最好的还小，就更新「最好的 RSS」和「最好的组合」。

```python
    mse, se = cv_mse(LinearRegression(), train[best_subset])
```
- 这个大小的冠军算 CV MSE（第 3 步）。注意这一行**缩进在外层循环**里：每个 k 算一次。

```python
    best_subsets.append({'Size': k, 'Predictors': best_subset, 'RSS': best_rss, 'CV MSE': mse, 'SE': se})
best_subsets = pd.DataFrame(best_subsets).set_index('Size')
best_subsets.round(4)
```
- 7 行结果做成表，以 Size 为行名。

### 画图 + 选大小
```python
plt.plot(best_subsets.index, best_subsets['CV MSE'], marker = 'o', color = '#FF7F0E')
```
- 横轴大小 1–7，纵轴 CV MSE；`marker='o'` 每个点画个圆点。

```python
best_size = best_subsets['CV MSE'].idxmin()
```
- `idxmin()`：返回 CV MSE 最小的那一行的**行名**（这里是 7）。

```python
results.append({'Model': 'M2: Best subset ({} predictors)'.format(best_size),
                'CV MSE': best_subsets.loc[best_size, 'CV MSE'], 'SE': best_subsets.loc[best_size, 'SE']})
```
- `.loc[行, 列]` 取出那一格的数值，存进 results。

**结果**：CV MSE 从 34147（1 个变量）一路降到 3302（7 个）→ 7 个都有用，等于 M1。

---

## 3.4 Expanded features: `make_features`

### 思路
EDA 发现弯曲和交互 → 需要平方项和交互项。写成**函数**，因为最后测试集也要生成完全一样的列。

### 逐行
```python
def make_features(df):
    features = df[predictors].copy()
```
- 取出 7 个原始列，`.copy()` 复制一份，不改动原数据。

```python
    for var in ['DistanceCBD', 'FloorArea', 'Bedrooms', 'PropertyAge']:
        features[var + '_SQ'] = features[var] ** 2
```
- 4 个非 0/1 变量，各加一列平方，如 `DistanceCBD_SQ`（命名学 W7 的 `Acq_Expense_SQ`）。`** 2` = 平方。

```python
    for j in range(len(predictors)):
        for k in range(j + 1, len(predictors)):
            var1 = predictors[j]
            var2 = predictors[k]
            features[var1 + '_x_' + var2] = features[var1] * features[var2]
```
- 两层循环列出所有「两两组合」：k 从 j+1 开始 → 每对只出现一次、不和自己配对 → 7×6÷2 = 21 对。
- 两列相乘 = 交互项，如 `Bedrooms_x_HighDemandArea`。

```python
    return features
```

```python
X_all = make_features(train)
extra_terms = [c for c in X_all.columns if c not in predictors]
```
- `X_all`：训练集的 32 列。
- `extra_terms`：列表推导式，挑出「不是原始 7 列」的那 25 个新列（4 平方 + 21 交互）。

---

## 3.5 M3: EDA-driven OLS

### 思路
把 EDA 的每个发现变成一个项，检验「猜想」是否成立。

### 逐行
```python
eda_terms = ['DistanceCBD_SQ', 'FloorArea_SQ', 'Bedrooms_SQ',
             'Bedrooms_x_HighDemandArea', 'FloorArea_x_HighDemandArea', 'DistanceCBD_x_NearTrain']
```
- 手动列出 EDA 建议的 6 个项。

```python
x_with_intercept = sm.add_constant(X_all[predictors + eda_terms], prepend=True)
m3 = sm.OLS(y, x_with_intercept).fit()
```
- `predictors + eda_terms`：两个列表相加 = 7 个主效应 + 6 个新项（层级原则：主效应都在）。
- 用 statsmodels 拟合，为了看 p 值【W7】。

```python
print(pd.DataFrame({'coef': m3.params[eda_terms], 'p-value': m3.pvalues[eda_terms]}).round(4))
```
- 只把 6 个新项的系数和 p 值做成小表打印（类似 W7 的 `pd.DataFrame(lasso.coef_...)`）。

```python
mse, se = cv_mse(LinearRegression(), X_all[predictors + eda_terms])
results.append({'Model': 'M3: EDA-driven OLS, 6 extra terms', 'CV MSE': mse, 'SE': se})
```
- 同样的列，算 CV MSE（2075.4951）。

---

## 3.6 M4: forward stepwise selection

### 思路（L4–5 FSS）
从 7 个主效应出发；每一步**试加**每个剩余的项，加入让 RSS 最小的那个；一直加到 25 个都进来，得到一条「路径」（26 个模型）。最后在路径上挑 AIC 最小、BIC 最小的两个，各算 CV。

### 逐行
```python
def fit_ols(terms):
    x_with_intercept = sm.add_constant(X_all[predictors + terms], prepend=True)
    return sm.OLS(y, x_with_intercept).fit()
```
- 小工具：给一组额外项，返回「7 个主效应 + 这些项」的 OLS 结果。避免重复写三行。

```python
forward_path = []
current = []
remaining = list(extra_terms)
```
- `forward_path`：记录每一步的模型；`current`：已经加进去的项；`remaining`：还没加的项（一开始是全部 25 个）。

```python
est = fit_ols(current)
forward_path.append({'Extra terms': 0, 'Terms': list(current), 'AIC': est.aic, 'BIC': est.bic})
```
- 第 0 步：只有主效应的模型，记下它的 AIC、BIC。
- `list(current)`：存一份**副本**；不写 `list(...)` 的话，后面 current 变了，记录也会跟着变。

```python
while len(remaining) > 0:
```
- 只要还有没加的项，就继续。

```python
    best_rss = np.inf
    for term in remaining:
        est = fit_ols(current + [term])
        if est.ssr < best_rss:
            best_rss = est.ssr
            best_term = term
```
- 逐个试加每个剩余项，找出让 RSS 最小的那个 `best_term`。

```python
    current.append(best_term)
    remaining.remove(best_term)
```
- 正式加入：放进 current，从 remaining 里删掉。

```python
    est = fit_ols(current)
    forward_path.append({'Extra terms': len(current), 'Terms': list(current), 'AIC': est.aic, 'BIC': est.bic})
```
- 记下这一步的模型和 AIC、BIC。

```python
forward_path = pd.DataFrame(forward_path).set_index('Extra terms')
```
- 26 行的路径表。

### 按 AIC / BIC 选模型并算 CV
```python
for criterion in ['AIC', 'BIC']:
    terms = forward_path.loc[forward_path[criterion].idxmin(), 'Terms']
```
- `forward_path['AIC'].idxmin()`：AIC 最小的那一步（例如 7）；`.loc[那一步, 'Terms']` 取出那一步的项列表。

```python
    mse, se = cv_mse(LinearRegression(), X_all[predictors + terms])
    results.append({'Model': 'M4: Forward stepwise ({}), {} extra terms'.format(criterion, len(terms)),
                    'CV MSE': mse, 'SE': se})
```
- 对选中的模型算 CV，存进 results。

---

## 3.7 M5: backward stepwise selection

### 思路（L4–5 BSS、W7 后向剔除）
反过来：从 25 个额外项全部在模型里开始；每一步**试删**每个项，删掉「删了之后 RSS 最小」（即最没用）的那个；一直删到 0 个。再按 AIC / BIC 选。

### 和 forward 的区别（其余写法一样）
```python
current = list(extra_terms)                 # 从全部 25 个开始
while len(current) > 0:
    ...
    for term in current:
        reduced = [t for t in current if t != term]   # 试删：除了 term 以外的所有项
        est = fit_ols(reduced)
        if est.ssr < best_rss: ...                    # 删了之后 RSS 最小 → 它最没用
            worst_term = term
    current.remove(worst_term)                        # 正式删掉
```
- `[t for t in current if t != term]`：列表推导式，得到「去掉 term 之后」的列表。

### 路径图
```python
for i, (path, name) in enumerate([(forward_path, 'Forward'), (backward_path, 'Backward')]):
    ax[i].plot(path.index, path['AIC'], ...)
    ax[i].plot(path.index, path['BIC'], ...)
```
- 左图 forward、右图 backward，各画 AIC、BIC 随「额外项个数」的变化。曲线最低点就是选中的模型。

---

## 3.8 M6: ridge, lasso and elastic net

### 思路
32 列全用，但加惩罚防止过拟合。惩罚强度 λ（sklearn 叫 `alpha`）要选 → 像 W5 选 k 一样**手写循环**，每个 λ 都算 CV MSE。
标准化放进 `Pipeline` → **每一折只用训练部分算 mean/std**（你要求的，防止泄露）。

### 逐行
```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
```
- `make_pipeline(A, B)`：把「先 A 再 B」绑成一个模型；`cross_val_score` 在每一折里会先对训练部分 fit A（算 mean/std），再用它转换验证部分。

```python
alphas = np.logspace(-3, 2, 30)
```
- 30 个 λ，从 0.001 到 100，在**对数刻度**上均匀分布（0.001, 0.0015, …, 100），因为 λ 的合适值可能差好几个数量级。

```python
ridge_cv = []
lasso_cv = []
for alpha in alphas:
    mse, se = cv_mse(make_pipeline(StandardScaler(), Ridge(alpha = alpha)), X_all)
    ridge_cv.append(mse)
    mse, se = cv_mse(make_pipeline(StandardScaler(), Lasso(alpha = alpha, max_iter = 100000)), X_all)
    lasso_cv.append(mse)
```
- 每个 λ：Ridge 算一次 CV、Lasso 算一次 CV，结果存进各自的列表【W5 `cv_rmse.append` 的写法】。
- `max_iter=100000`：Lasso 用迭代算法求解，给足迭代次数保证收敛。

```python
ax[i].set_xscale('log')
```
- 横轴用对数刻度，λ 才会均匀排开。

```python
best_ridge_alpha = alphas[np.argmin(ridge_cv)]
```
- `np.argmin`：CV MSE 最小的**位置**；用它从 `alphas` 取出对应的 λ【W5 `np.argmin`】。

### Elastic net
```python
best_enet_mse = np.inf
for l1_ratio in [0.2, 0.5, 0.8]:
    for alpha in alphas:
        model = make_pipeline(StandardScaler(), ElasticNet(alpha = alpha, l1_ratio = l1_ratio, max_iter = 100000))
        mse, se = cv_mse(model, X_all)
        if mse < best_enet_mse:
            best_enet_mse, best_enet_se = mse, se
            best_enet = (alpha, l1_ratio)
```
- Elastic net 有两个参数：λ 和 `l1_ratio`（Lasso 成分占多少；1 = 纯 Lasso，0 = 纯 Ridge）。
- 两层循环试 3 × 30 = 90 种组合，保留 CV MSE 最小的一组（类似 W5 `knn_test` 里的 `if cv_score >= best_score`）。

---

## 3.9 M7: KNN

### 思路
找 k 个最相似的房子，平均它们的租金【L3、W5】。先标准化（否则面积 35–191 会压过 0/1 变量）。k 用 CV 选。

### 逐行
```python
neighbours = np.arange(1, 51)
knn_cv = []
for k in neighbours:
    knn = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors = k))
    mse, se = cv_mse(knn, train[predictors])
    knn_cv.append(mse)
```
- 和 W5 几乎一样：k 从 1 到 50，每个 k 算 CV，存进列表。

```python
best_k = neighbours[np.argmin(knn_cv)]
```
- CV 最小的 k（结果 7）。

---

## 3.10 Model comparison

```python
comparison_table = pd.DataFrame(results).set_index('Model').sort_values('CV MSE')
comparison_table.round(4)
```
- `pd.DataFrame(results)`：字典列表 → 表格【W6 c27】。
- `.set_index('Model')`：模型名作为行名。
- `.sort_values('CV MSE')`：按 CV MSE 从小到大排，第一行就是最好的。

```python
close = comparison_table[comparison_table['CV MSE'] < 2500]
```
- 只保留 CV MSE < 2500 的模型画图（M0、M1、KNN 太大，放一起会把差异压扁）。

```python
plt.barh(close.index, close['CV MSE'], xerr = close['SE'], color = '#1F77B4')
```
- 横向柱状图；`xerr=close['SE']`：每根柱子画一条 ±1 SE 的误差线 → 误差线重叠 = 差别不显著。

```python
plt.gca().invert_yaxis()
plt.xlim(1900, 2100)
```
- 把最好的放最上面；横轴只显示 1900–2100，差异才看得清。
