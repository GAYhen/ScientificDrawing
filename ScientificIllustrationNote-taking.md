# 科研论文配图笔记

首先感谢Datacharm作者出的这本书《科研论文配图》，本次是由我通过阅读《科研论文配图》这本书针对**科研配图**和**数学建模**绘图需求做出的笔记，本次笔记不涉及基础部分绘制，只说明不同风格下绘制，绘制较为惊艳的图，由于采用Latex字体，中文可能会不支持，所以尽量不要使用中文，**如果需要使用中文请事先声明字体还有坐标轴**，内容难免出错，希望各位在复现的时候进行反馈，笔者订正。**本文档只有文字，需要可视化请查看ipynb或者html文件。**针对后续多变量绘图和其他图像绘制以后更新，本Note对上书做出总结和加入自己的绘图代码，如若侵权，麻烦联系。-------AHNU_GAYhen


**为了解决 matplotlib 图表中文显示的问题，我们需要修改`pyplot`模块的`rcParams`配置参数，具体的操作如下所示。**

```python
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'].insert(0, 'SimHei')
plt.rcParams['axes.unicode_minus'] = False
```

本笔记主要使用的绘图编程语言为Python，对应绘图库的版本和名称如下，也可以直接pip install requirements.txt

```python
Package                       Version
----------------------------- --------
alabaster                     0.7.16
appdirs                       1.4.4
asttokens                     2.4.1
attrs                         23.2.0
Babel                         2.15.0
backcall                      0.2.0
beautifulsoup4                4.12.3
biokit                        0.5.0
biopython                     1.83
bioservices                   1.11.2
cattrs                        23.2.3
certifi                       2024.2.2
cffi                          1.16.0
charset-normalizer            3.3.2
click                         8.1.7
colorama                      0.4.6
colorlog                      6.8.2
colormap                      1.1.0
comm                          0.2.2
contourpy                     1.2.1
cycler                        0.12.1
debugpy                       1.8.1
decorator                     5.1.1
docutils                      0.21.2
easydev                       0.13.2
et-xmlfile                    1.1.0
exceptiongroup                1.2.1
executing                     2.0.1
fonttools                     4.51.0
gevent                        24.2.1
greenlet                      3.0.3
grequests                     0.7.0
idna                          3.7
imagesize                     1.4.1
importlib_metadata            7.1.0
importlib_resources           6.4.0
ipykernel                     6.28.0
ipython                       8.18.1
jedi                          0.19.1
Jinja2                        3.1.4
joblib                        1.4.2
jupyter_client                8.6.1
jupyter_core                  5.7.2
KDEpy                         1.1.9
kiwisolver                    1.4.5
line_profiler                 4.1.3
lxml                          5.2.2
MarkupSafe                    2.1.5
matplotlib                    3.9.0
matplotlib-inline             0.1.7
nest-asyncio                  1.6.0
numpy                         1.26.4
numpydoc                      1.7.0
openpyxl                      3.1.2
packaging                     24.0
pandas                        1.5.3
parso                         0.8.4
patsy                         0.5.6
pexpect                       4.9.0
pickleshare                   0.7.5
pillow                        10.3.0
pip                           24.0
platformdirs                  4.2.2
prompt-toolkit                3.0.43
proplot                       0.9.7
psutil                        5.9.8
ptitprince                    0.2.7
ptyprocess                    0.7.0
pure-eval                     0.2.2
pycparser                     2.22
Pygments                      2.18.0
pyparsing                     3.1.2
python-dateutil               2.8.2
pytz                          2024.1
pywin32                       306
pyzmq                         26.0.3
requests                      2.31.0
requests-cache                1.2.0
SciencePlots                  2.1.1
scikit-learn                  1.4.2
scikit-posthocs               0.9.0
scipy                         1.13.0
seaborn                       0.11.0
setuptools                    69.5.1
six                           1.16.0
snowballstemmer               2.2.0
soupsieve                     2.5
Sphinx                        7.3.7
sphinxcontrib-applehelp       1.0.8
sphinxcontrib-devhelp         1.0.6
sphinxcontrib-htmlhelp        2.0.5
sphinxcontrib-jsmath          1.0.1
sphinxcontrib-qthelp          1.0.7
sphinxcontrib-serializinghtml 1.1.10
stack-data                    0.6.3
statannotations               0.6.0
statsmodels                   0.14.2
suds-community                1.1.2
superviolin                   1.0.6
tabulate                      0.9.0
threadpoolctl                 3.5.0
tomli                         2.0.1
tornado                       6.4
tqdm                          4.66.4
traitlets                     5.14.3
typing_extensions             4.11.0
tzdata                        2024.1
url-normalize                 1.4.3
urllib3                       2.2.1
wcwidth                       0.2.13
wheel                         0.43.0
wrapt                         1.16.0
xlrd                          2.0.1
xmltodict                     0.13.0
zipp                          3.18.1
zope.event                    5.0
zope.interface                6.4
```

笔者自己的普通美化函数，最基本的Matplotlib即可实现：

```python
# 高分辨率图像
%config InlineBackend.figure_format = 'retina' ##生成高分辨率图像

# 设置字体和坐标轴
plt.rcParams['font.sans-serif'].insert(0, 'SimHei')
plt.rcParams['axes.unicode_minus'] = False

# 字体为Latex 新罗马
rc('text', usetex=True)	# 设置使用LaTeX作为默认字体
rc('font', family='serif', weight='bold')	# 设置字体为新罗马，并且坐标轴加粗

# 网格线绘制
axs[0].grid(True, linestyle=(0,(5,2)), **linewidth=1.5**) 	# 设置网格线为较长较宽的虚线
axs[1].grid(True, linestyle=(0,(5,2)), linewidth=0.5, ***alpha=0.5***)  # 设置网格线为更细更虚的虚线
```

## 科研配图色彩网站

**笔者自用的有如下：**

```http
https://colorbrewer2.org
```

```http
https://www.zhongguose.com/#anyuzi
```

```http
https://flatuicolors.com/
```

```http
https://color.adobe.com/zh/explore
```

针对每个网站对应的色彩网站使用方法请读者自查。

## 绘图工具

本笔记教程主要用到了绘图库如下(按住Ctrl点击进入对应的示例)：

[Matplotlib](https://matplotlib.org/stable/gallery/index.html)

[Seaborn](https://seaborn.pydata.org/examples/index.html)

[Proplot](https://github.com/proplot-dev/proplot)

[Scienceplot](https://github.com/garrettj403/SciencePlots?tab=readme-ov-file)

对应的用法和示例自行学习（笔者插：现在不是有GPT吗，学习成本Down Down！！！直接问GPT）

## 单变量绘制

### 直方图

```python
import numpy as np
import matplotlib.pyplot as plt

# 生成随机数据
data = np.random.randn(1000)

# 设置绘图风格
plt.style.use('science')

# 绘制直方图
plt.figure(figsize=(6, 4))
plt.hist(data, bins=30, alpha=0.75, color='blue', edgecolor='black')
plt.title("Histogram of Random Data", fontsize=16, fontfamily='serif')
plt.xlabel("Value", fontsize=14, fontfamily='serif')
plt.ylabel("Frequency", fontsize=14, fontfamily='serif')
plt.grid(True)
plt.show()
```

### 密度图

```Python
import seaborn as sns

# 设置绘图风格
plt.style.use('science')

# 绘制密度图
plt.figure(figsize=(6, 4))
sns.kdeplot(data, color='blue', fill=True)
plt.title("Density Plot of Random Data", fontsize=16, fontfamily='serif')
plt.xlabel("Value", fontsize=14, fontfamily='serif')
plt.ylabel("Density", fontsize=14, fontfamily='serif')
plt.grid(True)
plt.show()
```

### 直方密度组合图

```Python
# 设置绘图风格
plt.style.use('science')

# 绘制组合图
plt.figure(figsize=(6, 4))
sns.histplot(data, bins=30, kde=True, color='blue', edgecolor='black')
plt.title("Combined Histogram and Density Plot", fontsize=16, fontfamily='serif')
plt.xlabel("Value", fontsize=14, fontfamily='serif')
plt.ylabel("Frequency / Density", fontsize=14, fontfamily='serif')
plt.grid(True)
plt.show()
```

### Q-Q图和P-P图

```pyhon
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 生成正态分布的随机数据
data = np.random.normal(loc=0, scale=1, size=1000)

# 设置绘图风格
plt.style.use('science')

# 绘制Q-Q图
plt.figure(figsize=(6, 4))
stats.probplot(data, dist="norm", plot=plt)
plt.title('Q-Q plot', fontsize=16, fontfamily='serif')
plt.xlabel('Theoretical quantiles', fontsize=14, fontfamily='serif')
plt.ylabel('Sample quantiles', fontsize=14, fontfamily='serif')
plt.grid(True)
plt.show()
```

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 生成正态分布的随机数据
data = np.random.normal(loc=0, scale=1, size=1000)

# 设置绘图风格
plt.style.use('science')

# 绘制Q-Q图
plt.figure(figsize=(6, 4))
stats.probplot(data, dist="norm", plot=plt)
plt.title('Q-Q plot', fontsize=16, fontfamily='serif')
plt.xlabel('Theoretical quantiles', fontsize=14, fontfamily='serif')
plt.ylabel('Sample quantiles', fontsize=14, fontfamily='serif')
plt.grid(True)
plt.show()
```

### 经验分布函数

```python
# a）Matplotlib中经验分布函数图的属性添加示例
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True


#生成数据
data1 = np.random.normal(loc=20, scale=5, size=400)
data2 = np.random.normal(loc=40, scale=5, size=800)
ecdf_data = np.hstack((data1, data2))

# fit a ecdf
ecdf = ECDF(ecdf_data)
fig,ax = plt.subplots(figsize=(4.5,3.5),dpi=100,facecolor="w")
ax.plot(ecdf.x, ecdf.y,color="#2FBE8F",lw=1.5,label="ECDF")
#第二条ecdf线
ecdf_full = ECDF(np.random.normal(loc = ecdf_data.mean(), 
                                        scale = ecdf_data.std(), 
                                        size = 100000))
ax.plot(ecdf_full.x,ecdf_full.y,"k",lw=1)

xs = ecdf.x
ys = ecdf.y
percent_values = [.25,.50,.75]
# 循环绘制
for p in percent_values:
    value = xs[np.where(ys > p)[0][0] - 1]
    pvalue = ys[np.where(ys > p)[0][0] - 1]
    ax.scatter(value,pvalue,s=30,color="#2FBE8F",ec="k",zorder=3)
    ax.hlines(y=p, xmin=0, xmax = value,color="r",ls="--",lw=1)
    ax.text(x=value/3,y=pvalue+.05,s=f'{int(100*p)}%',color="r",va="center")
    ax.vlines(x=value, ymin=0, ymax = pvalue,color="r",ls="--",lw=1)
    ax.text(x = value+.5, y = 0.02,s = f'{value:.1f}',color="r",ha="left")

ax.scatter(value,pvalue,color="#2FBE8F",ec="k",label="Test Point")
ax.set_xlim(0,max(ecdf_data)+2)
ax.set_ylim(0,1.05)
ax.set_ylabel('Percentile')
ax.set_xlabel('Normal Distribution Values')
ax.set_title('ECDF of Normal Distribution',fontsize=16)
ax.legend(fontsize=9)
plt.show() 
```

## 双变量绘制

### 误差线柱状图

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

# 设定matplotlib参数
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True

# 假设有一个iris数据集
iris = sns.load_dataset("iris")  # 使用seaborn内置的鸢尾花数据集

# 自定义颜色
palette = ["#BC3C29FF", "#0072B5FF", "#E18727FF"]

# 绘制柱状图
fig, ax = plt.subplots(figsize=(4, 3.5), dpi=100, facecolor="w")
sns.barplot(x="species", y="sepal_length", data=iris, palette=palette,
            estimator=np.mean, ci="sd", capsize=.1, errwidth=1, errcolor="k",
            saturation=1, edgecolor="k", linewidth=1, ax=ax)

x1, x2 = 0, 1  # 柱子的位置
y, h = max(mean_setosa, mean_versicolor) + 0.2, 0.2  # 线的高度位置和额外高度

plt.show()
```

### 热力图

本次热力图绘制未使用seaborn和matplot使用的是biokit

```python
# 安装指令
pip install biokit

# 本次笔记只是用了一个图像风格，biokit还有更多其他图像风格
```

```python
import pandas as pd
import numpy as np
import proplot as pplt
import seaborn as sns
from biokit.viz import corrplot #需要安装biokit库
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from proplot import rc
rc["axes.labelsize"] = 15
rc['tick.labelsize'] = 13
rc["suptitle.size"] = 15
rc["font.family"] = "Times New Roman"
rc["xtick.minor.visible"] = False
rc["ytick.minor.visible"] = False
rc["xtick.bottom"] = False
rc["ytick.left"] = False

heatmap_data = pd.read_excel(r"相关性热力图_P值.xlsx")

# a）BioKit相关性矩阵热力图（circle）

#method="circle"
c = corrplot.Corrplot(heatmap_data.corr())
fig,ax = plt.subplots(figsize=(4,3.5),dpi=100,facecolor="w")
ax = c.plot(colorbar=True, method='circle', shrink=.9,fontsize=12,rotation=0,ax=ax)
plt.show()
```

### 边际组合图

```python
import pandas as pd
import numpy as np
import proplot as pplt
import seaborn as sns
from biokit.viz import corrplot  # 需要安装biokit库
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from proplot import rc
rc["axes.labelsize"] = 15
rc['tick.labelsize'] = 13
rc["suptitle.size"] = 15
rc["figure.facecolor"] = "w"
rc["font.family"] = "Times New Roman"
rc["xtick.minor.visible"] = False
rc["ytick.minor.visible"] = False
rc["xtick.bottom"] = False
rc["ytick.left"] = False

# 使用 matplotlib 的内置颜色映射 'viridis' 替代 'parula'
penguins = sns.load_dataset("penguins")
sns.jointplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", 
              hue="species", palette="viridis")
plt.show()
```

### 相关性散点图（无误差线）

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = .8
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True


scatter_data = pd.read_excel(r"散点图样例数据2.xlsx")

# a）Matplotlib 相关性散点图完善示例

from scipy import stats

x = scatter_data["values"]
y = scatter_data["pred values"]
z = scatter_data["3_value"].values
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
slope, intercept, r_value, p_value, std_err

#绘制最佳拟合线
best_line_x = np.linspace(-10,10)
best_line_y=best_line_x
#绘制拟合线
y3 = slope*x + intercept

fig,ax = plt.subplots(figsize=(4,3.5),dpi=100,facecolor="w")
scatter = ax.scatter(x=x,y=y,edgecolor=None, c='k', s=13,marker='s',label="Data")
bestline = ax.plot(best_line_x,best_line_y,color='k',linewidth=1.5,linestyle='--',label="1:1 Line")
linreg = ax.plot(x,y3,color='r',linewidth=1.5,linestyle='-',label="Fitted Line")
ax.set_xlim((-.1, 1.8))
ax.set_ylim((-.1, 1.8))
ax.set_xticks(np.arange(0, 2, step=0.2))
ax.set_yticks(np.arange(0, 2, step=0.2))

# 添加文本信息
fontdict = {"size":13,"fontstyle":"italic"}
ax.text(0.,1.6,r'$R=$'+str(round(r_value,2)),fontdict=fontdict)
ax.text(0.,1.4,"P $<$ "+str(0.001),fontdict=fontdict)
ax.text(0.,1.2,r'$y=$'+str(round(slope,3))+'$x$'+" + "+str(round(intercept,3)),fontdict=fontdict)
ax.text(0.,1.0,r'$N=$'+ str(len(x)),fontdict=fontdict)

ax.set_xlabel("Variable 01")
ax.set_ylabel("Variable 02")
ax.legend(loc="lower right")

plt.show()
```

### 相关性散点图（有误差线）

```python
# d）SciencePlots 相关性（误差）散点图绘制示例
from scipy import stats
import numpy as np
from sklearn.metrics import mean_squared_error

plt.style.use('science') # 设置全局绘图样式

data_err = pd.read_excel(r"散点图样例数据2.xlsx",sheet_name="data02")

x = data_err["values"]
y = data_err["pred values"]
x_err = data_err["x_error"]
y_err = data_err["y_error"]
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
rmse = np.sqrt(mean_squared_error(x,y))
#绘制1:1拟合线
best_line_x = np.linspace(-10,10)
best_line_y=best_line_x
#绘制拟合线
y3 = slope*x + intercept
#开始绘图
fig,ax = plt.subplots(figsize=(4,3.5),dpi=100,facecolor="w")
scatter = ax.scatter(x=x,y=y,edgecolor=None, c='k', s=18,label="Data")
bestline = ax.plot(best_line_x,best_line_y,color='k',linewidth=1.5,linestyle='--',label="1:1 Line")
linreg = ax.plot(x,y3,color='r',linewidth=1.5,linestyle='-',label="Fitted Line")
# 添加误差
errorbar = ax.errorbar(x,y,xerr=x_err,yerr=y_err,ecolor="k", elinewidth=.4,capsize=0,alpha=.7,
            linestyle="",mfc="none",mec="none",zorder=-1)
ax.set_xlim((-.1, 1.8))
ax.set_ylim((-.1, 1.8))
ax.set_xticks(np.arange(0, 2, step=0.2))
ax.set_yticks(np.arange(0, 2, step=0.2))
# 添加文本信息
fontdict = {"size":13,"fontstyle":"italic"}
ax.text(0.,1.6,r'$R=$'+str(round(r_value,2)),fontdict=fontdict)
ax.text(0.,1.4,"$P <$ "+str(0.001),fontdict=fontdict)
ax.text(0.,1.2,r'$y=$'+str(round(slope,3))+'$x$'+" + "+str(round(intercept,3)),fontdict=fontdict)
ax.text(0.,1.0,r'$N=$'+ str(len(x)),fontdict=fontdict)

ax.set_xlabel("Variable 01")
ax.set_ylabel("Variable 02")
ax.legend(loc="lower right",frameon=True)

plt.show()
```

### 雨云图

需要安装ptitprince

```python
# 安装指令
pip install ptitprince
```



```python
import pandas as pd
import numpy as np
import ptitprince as pt # 需要单独安装
import seaborn as sns
from scipy import stats 
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = .8
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["xtick.minor.visible"] = False
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["xtick.top"] = False
plt.rcParams["ytick.right"] = False

# a）PtitPrince云雨图基本样式一

rain_data = sns.load_dataset("iris")
colors = ["#2FBE8F","#459DFF","#FF5B9B","#FFCC37","#751DFE"]

fig,ax = plt.subplots(figsize=(4,3.5),dpi=100,facecolor="w")

ax=pt.half_violinplot(x = "species", y ="sepal_width", data = rain_data, palette = colors[:3],
      bw=0.2, cut=2,scale = "area", width = 0.8, linewidth=1,inner="box",saturation=1)
ax.set_xlabel("Sepal")
ax.set_ylabel("Sepal\_width")

plt.show()
```

```python
# b）PtitPrince云雨图基本样式二

fig,ax = plt.subplots(figsize=(4,3.5),dpi=100,facecolor="w")
ax=pt.half_violinplot(x = "species", y ="sepal_width", data = rain_data, palette = colors[:3],
      bw=0.2, cut=2,scale = "area", width = 0.8, linewidth=1,inner="quartile",saturation=1)
ax.set_xlabel("Sepal")
ax.set_ylabel("Sepal\_width")

plt.show()
```

```python
# c）PtitPrince云雨图基本样式三

fig,ax = plt.subplots(figsize=(4,3.5),dpi=100,facecolor="w")
ax=pt.half_violinplot(x = "species", y ="sepal_width", data = rain_data, palette = colors[:3],
      bw=0.2, cut=2,scale = "area", width = 0.8, linewidth=1,inner="box",saturation=1)

ax=sns.stripplot(x = "species", y ="sepal_width", data = rain_data, palette = colors[:3], 
                 edgecolor="k",linewidth=.4,size = 4, jitter = .08, zorder = 0,)
ax.set_xlabel("Sepal")
ax.set_ylabel("Sepal\_width")

plt.show()
```

```python
# d）PtitPrince云雨图基本样式四
fig,ax = plt.subplots(figsize=(4,3.5),dpi=100,facecolor="w")
ax=pt.half_violinplot(x = "species", y ="sepal_width", data = rain_data, palette = colors[:3],
      bw=0.2, cut=2,scale = "area", width = 0.8, linewidth=1,inner=None,saturation=1)
ax=sns.stripplot(x = "species", y ="sepal_width", data = rain_data, palette = colors[:3], 
                 edgecolor="k",linewidth=.4,size = 4, jitter = .08, zorder = 5,)
ax=sns.boxplot(x = "species", y ="sepal_width", data = rain_data,width = .2,saturation = 1,
               boxprops = {'facecolor':'none', "zorder":2},
               medianprops={"color":"k","linewidth":1.5},showcaps=True,showfliers=False,zorder=0)
ax.set_xlabel("Sepal")
ax.set_ylabel("Sepal\_width")

plt.show()
```

```python
# e）RainCloud()函数绘制的云雨图并排组合样式一
fig,ax = plt.subplots(figsize=(4,3.5),dpi=100,facecolor="w")
ax=pt.RainCloud(x = "species", y ="sepal_width", data = rain_data, palette = colors[:3],
                width_viol =.72,width_box=.2,move=.16,saturation = 1,linewidth=.5,box_showfliers=False,
                box_linewidth =1,point_size=4,ax=ax)
ax.set_xlabel("Sepal")
ax.set_ylabel("Sepal\_width")

plt.show()
```

```python
#  f）RainCloud()函数绘制的云雨图并排组合样式二

fig,ax = plt.subplots(figsize=(4,3.5),dpi=100,facecolor="w")
ax=pt.RainCloud(x = "species", y ="sepal_width", data = rain_data, palette = colors[:3],
                width_viol =.72,width_box=.2,move=.15,saturation = 1,linewidth=.5,box_showfliers=False,
                box_linewidth =1,point_size=4,pointplot = True,
                ax=ax)
ax.set_xlabel("Sepal")
ax.set_ylabel("Sepal\_width")

plt.show()
```

### 高级小提琴图（加入统计信息）

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from superviolin import test_plot,plot
from superviolin.plot import Superviolin

plt.rcParams["figure.facecolor"] = "w"
plt.rcParams["axes.spines.right"] = True
plt.rcParams["axes.spines.top"] = True
plt.rcParams['text.usetex'] = True

file_name = r"demo_data.csv"
violin = Superviolin(filename=file_name,condition="drug",value="variable",dpi=100,cmap="Dark2",
                    linewidth=0.7,return_stats=True,stats_on_plot="yes")
violin.generate_plot()

plt.show()
```

### 山脊图

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from KDEpy import NaiveKDE

# 设置图形风格参数
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True

# 数据准备（确保 color 和 depth 数组长度相同）
colors = ['E', 'E', 'I', 'J', 'J', 'I', 'H', 'E', 'H', 'J', 'J', 'F', 'J', 'E', 'E', 'I', 'J', 'J', 'J',
          'I', 'E', 'H', 'J', 'J', 'G', 'I', 'J', 'D', 'F', 'F', 'F', 'E', 'E', 'D', 'F', 'E', 'H', 'D', 'I',
          'I', 'J', 'D', 'D', 'H', 'F', 'H', 'H', 'E', 'H', 'F', 'G', 'I', 'E', 'D', 'I', 'J', 'I', 'I', 'I',
          'I', 'D', 'D', 'D', 'I', 'G', 'I', 'G', 'G', 'E', 'D', 'H', 'H', 'H', 'H', 'F', 'E', 'D', 'D', 'E',
          'E', 'D', 'E', 'I', 'E', 'G', 'H', 'H', 'H', 'I', 'E', 'E', 'G', 'E', 'G', 'E', 'F', 'F', 'E', 'H']
depths = np.random.normal(60, 5, len(colors))  # 生成与 colors 数组长度相匹配的 depth 数据
group_data = pd.DataFrame({'color': colors, 'depth': depths})
sorted_index = sorted(set(group_data['color']), key=str.lower)

# 绘图
fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=100, facecolor="w")
for i, index in enumerate(sorted_index):
    data = group_data[group_data["color"] == index]["depth"].values
    x, y = NaiveKDE(kernel="Gaussian", bw=.8).fit(data).evaluate()
    ax.plot(x, 6 * y + i, lw=.6, color="k", zorder=100 - i)
    ax.fill(x, 6 * y + i, color="gray", alpha=.6, zorder=100 - i)

ax.grid(which="major", axis="y", ls="--", lw=.7, color="gray", zorder=-1)
ax.set_xlim(50, 72)
ax.yaxis.set_tick_params(labelleft=True)
ax.set_yticks(np.arange(len(sorted_index)))
ax.set_yticklabels(sorted_index)
ax.set_xlabel("Depth")
ax.set_ylabel("Color")
ax.tick_params(which="both", top=False, right=False)
ax.tick_params(which="minor", axis="both", left=False, bottom=False)
for spin in ["top", "right", "bottom", "left"]:
    ax.spines[spin].set_visible(False)  # 正确使用 set_visible 方法

plt.show()
```

### 山脊渐变图

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from KDEpy import NaiveKDE
from matplotlib.patches import PathPatch
from matplotlib.path import Path

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True

# 数据准备
colors = ['E', 'E', 'I', 'J', 'J', 'I', 'H', 'E', 'H', 'J', 'J', 'F', 'J', 'E', 'E', 'I', 'J', 'J', 'J',
          'I', 'E', 'H', 'J', 'J', 'G', 'I', 'J', 'D', 'F', 'F', 'F', 'E', 'E', 'D', 'F', 'E', 'H', 'D', 'I',
          'I', 'J', 'D', 'D', 'H', 'F', 'H', 'H', 'E', 'H', 'F', 'G', 'I', 'E', 'D', 'I', 'J', 'I', 'I', 'I',
          'I', 'D', 'D', 'D', 'I', 'G', 'I', 'G', 'G', 'E', 'D', 'H', 'H', 'H', 'H', 'F', 'E', 'D', 'D', 'E',
          'E', 'D', 'E', 'I', 'E', 'G', 'H', 'H', 'H', 'I', 'E', 'E', 'G', 'E', 'G', 'E', 'F', 'F', 'E', 'H']
depths = np.random.normal(60, 5, len(colors))
group_data = pd.DataFrame({'color': colors, 'depth': depths})
sorted_index = sorted(set(group_data['color']), key=str.lower)

fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=100, facecolor="w")
for i, index in enumerate(sorted_index):
    data = group_data[group_data["color"] == index]["depth"].values
    x, y = NaiveKDE(kernel="Gaussian", bw=.8).fit(data).evaluate()
    y_scaled = 6 * y + i
    ax.plot(x, y_scaled, lw=.6, color="k", zorder=100 - i)
    
    # Create a clip path that conforms to the line plot
    vertices = np.array([x, y_scaled]).T
    vertices = np.vstack([[x[0], i], vertices, [x[-1], i]])
    codes = [Path.MOVETO] + [Path.LINETO] * (len(x) - 1) + [Path.LINETO] + [Path.CLOSEPOLY]
    clip_path = PathPatch(Path(vertices, codes), transform=ax.transData)

    # Add gradient fill using imshow
    img_data = np.linspace(0, 1, len(x)).reshape(1, -1)
    im = ax.imshow(img_data, aspect='auto', cmap="plasma", extent=[x.min(), x.max(), i, i + 1], zorder=99 - i)
    im.set_clip_path(clip_path)
    ax.add_patch(clip_path)

ax.grid(which="major", axis="y", ls="--", lw=.7, color="gray", zorder=-1)
ax.set_xlim(50, 72)
ax.set_ylim(0, len(sorted_index))
ax.set_yticks(np.arange(len(sorted_index)) + 0.5)
ax.set_yticklabels(sorted_index)
ax.set_xlabel("Depth")
ax.set_ylabel("Color")
ax.tick_params(which="both", top=False, right=False)
for spin in ["top", "right", "bottom", "left"]:
    ax.spines[spin].set_visible(False)

plt.show()
```

