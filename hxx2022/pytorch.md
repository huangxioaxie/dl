#### 日常启动

```
cd \Users\hys\Desktop\bishe\code\pythonwork\d2l-zh\pytorch
conda activate d2l
Jupyter Notebook


cd \pythonwork\d2l-zh\pytorch
conda activate pytorch


Jupyter Notebook
```



# 1. Pytorch环境配置和安装

### 关于Anaconda

做的项目A和项目B分别基于python2和python3，而第电脑只能安装一个环境，这个时候Anaconda就派上了用场，它可以创建多个互不干扰的环境，分别运行不同版本的软件包，以达到兼容的目的

### 安装Anaconda

官网 https://www.anaconda.com 最新但是不稳定，推荐去历史版本https://repo.anaconda.com下3.6

Anaconda3-5.2.0-Windows-x86_64.exe

 https://repo.anaconda.com/archive/Anaconda2-5.2.0-Windows-x86_64.exe

记住安装路径: C:\Users\huangxiaoxie\Anaconda2



建立并切换到工作目录

```
md \pythonwork
cd \pythonwork
```

Anaconda Prompt 输入环境： conda create -n [环境名称] python=3.6

```
conda create -n pytorch python=3.6  

```

conda环境安装 jupyter的包 

```
conda install nb_conda
```

Anaconda Prompt 激活环境：  conda activate [环境名称]

```
conda activate pytorch
```

Anaconda Prompt 卸载环境：  

```
conda activate base
conda uninstall -n  [环境名称] --all
```

### 安装Pytorch

查看显卡型号是否支持cuda：

https://www.geforce.cn/hardware/technology/cuda/supported-gpus

打开 命令行输入： nvidia-smi  查看驱动版本 和 cuda版本是否支持

pytorch安装界面： https://pytorch.org/get-started/locally/

安装指令：

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

conda 在实践中出现了问题，直接用pip也可以

```
pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio===0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

检查是否安装好，命令行输入

```
python
import torch
import torchvision
print(torch.__version__)
print(torch.cuda.is_available())
```

关闭Pytorch Anaconda虚拟环境

```
conda deactivate
```

此就安装完成，同学们现在可以愉快地进行机器学习项目学习与开发了。（github上有无数开源代码，可以对感兴趣的项目直接进行搜索，然后对项目clone（需安装git）或直接download，也可以fork到自己的仓库（然后使用git pull到本地），当自己脑子短路或者什么的上去找找灵感吧）

在跑别人的项目时如果遇到相应module缺失的情况，打开Pytorch Anaconda虚拟环境用conda或pip安装即可解决。（建议优先使用conda，conda会分析依赖包，会将依赖包一同安装）

如果需要使用本虚拟环境在Notebook中跑项目，进入工作目录激活虚拟环境，输入Jupyter Notebook运行即可。

```
Jupyter Notebook
```

如果需要使用本虚拟环境在Pycharm进行项目开发，将设置里的Project Interpreter改为相应Anaconda文件目录下的Pytorch虚拟环境中的python.exe文件即可。(如：D:\Anaconda3\envs\pytorch\python.exe）



# 2  Jupyter 编辑文本的基本用法

# 1. jupyter 常用命令

## 1.1 什么是 jupyter notebook

jupyter notebook是一款开源的Web应用程序，该应用程序可以用来创建并共享实施代码，方程式，可视化以及文本说明。jupyter notebook基于IPython解释器，是一个基于Web的交互式计算环境。从不正规的角度讲，可以将jupyter notebook看成是一个Web版的IPython，实际上，jupyter notebook之前的名称就叫做IPython notebook。

## 1.2 启动jupyter

启动jupyter可以使用命令：
jupyter notebook（jupyter-notebook）
即可。当执行命令后，就会启动jupyter服务，同时打开浏览器页面，显示jupyter的home页面。默认情况下，会使用当前所在的目录作为根目录。

## 1.3 修改默认的主目录

home页面会显示在当前主目录下的文件以及路径（文件夹），我们可以直接打开查看文件或进入目录。如果我们需要打开的文件不在根目录下，但又不想上传，我们可以修改启动jupyter后默认的主目录：

- 切换到指定目录后，启动jupyter服务。
- 在启动jupyter服务时，同时使用--notebook-dir=主目录。
  eg: jupyter notebook --notebook-dir=c:\anaconda
- 建议修改jupyter 的配置文件，直接点击jupyter界面，跳转到浏览器，打开指定的路径。方法：[点击我访问博客地址](https://links.jianshu.com/go?to=https%3A%2F%2Fblog.csdn.net%2Fcaterfreelyf%2Farticle%2Fdetails%2F79774311)

## 1.4 单元格

jupyter notebook文档由一些列单元格组成，我们可以在单元格中输入相关的代码或者说明文字。单元格有以下几种类型：

- code 代码单元格，用来编写程序。
- Markdown 支持Markdown语法的单元格，用来编写描述程序的文字。
- Raw NBConvert 原生类型单元格，内容会原样显示。在使用NBConvert转换后才会显示成特殊的格式。
- Heading 标题单元格，已经不在支持使用。

## 1.5 命令模式与编辑模式

此外，jupyter notebook的单元格分为两种模式：

- 命令模式 单元格处于选中状态，此时单元格左侧为粗蓝色线条，其余为细灰色线条。
- 编辑模式 单元格处于编辑状态，此时单元格左侧为粗绿色线条，其余为细绿色线条。

## 1.6 常用快捷键

jupyter notebook常用的快捷键如下：

### 1.6.1 命令模式

- Y :单元格转换成code类型。
- M :单元格转换成Markdown类型。
- R :单元格转换成Raw NBConvert类型。
- Enter :进入编辑模式。
- A :在当前单元格上方插入新单元格。
- B :在当前单元格下方插入新单元格。
- C :复制当前单元格。
- D(两次） :删除当前单元格。
- V :粘贴到当前单元格的下方。
- Shift + V :粘贴到当前单元格的上方。
- Z :撤销删除。
- Ctrl+Shift+"-":快速将一个代码块分割成两块

### 1.6.2 编辑模式

- Tab 代码补全
- Shift + Tab 显示doc文档信息。
- Esc 进入命令模式。

### 1.6.3 通用模式

- Ctrl + Enter 运行单元格，然后该单元格处于命令模式。
- Shift + Enter 运行单元格，并切换到下一个单元格，如果下方没有单元格，则会新建一个单元格。
- Alt + Enter 运行单元格，并在下方新增一个单元格。

# 2. Markdown

## 2.1 什么是Markdown



```undefined
Markdown是一种使用纯文本格式语法的轻量级标记语言，它允许人们使用易读易写的纯文本格式编写文档，然后转换成格式丰富的HTML页面。Markdown同时也支持HTML标签。在Markdown类型的单元格中，支持使用Markdown语法与LaTex数学公式。
```

### 2.2 标题

标题可以使用1 ~ 6个`#`跟随一个空格来表示1 ~ 6级标题。

- # 一级标题

- ## 二级标题

- ### 三级标题

- #### 四级标题

- ##### 五级标题

- ###### 六级标题

- \####### 七级标题
  注：Markdown 只支持1-6级标题，不支持更低级别的标题。如上所示，当输入7个`#`号加空格时，它会当成文本处理，不再是标题。

### 2.3 无序列表

无序列表可以使用`*`，`-`或`+`后跟随一个空格来表示。也可以通过不同的符号混合表示多级列表。例子见2.2 显示。

### 2.4 有序列表

有序列表使用数字跟随一个点（.）表示。

1. 这是一个有序列表
2. 这也是一个有序列表

### 2.5 换行

使用两个或以上的空白符。空白符：空格符，制表符，换行符等的统称。

- 效果所示，我要换行。
  效果所示，我要换行。
  我想让一段话，首行缩进2个字符，或者使用空格符，空几个字符咋办呢？【注意：不要漏掉分号。】
- 插入一个空格 (non-breaking space)：使用'&nbsp'加上';'
- 插入两个空格 (en space):使用'&ensp'加上';'
- 插入四个空格 (em space):使用'&emsp'加上';';
- 插入细空格 (thin space):使用'&thinsp'加上';'

### 2.6 粗体 / 斜体

使用`**`或`__`包围的字体为粗体。使用`*`或`_`包围的字体为斜体。

- 展示粗体效果，**这是粗体**，**这也是粗体**
- 展示斜体效果，*这是斜体*，*这也是斜体*
- 思考：怎么表示粗斜体？一共有多少中实现方法？
- 展示粗斜体效果，***这是粗斜体\***，***这也是粗斜体\***，***这也是粗斜体\***，***这也是粗斜体\***，***这也是粗斜体***，***这也是粗斜体***

### 2.7 删除线

使用`~~`包围的字体会带有删除线效果。

- 展示删除线效果，~~这是删除线的效果~~

### 2.8 代码

可以使用`代码`来标记代码部分。
使用```（或Tab缩进）来标记代码块。在```后面加上相应的语言，可以使代码的关键字高亮显示。

- 标记代码： `print(" Hello World")`
- 标记代码块两种实现方式：



```bash
    print(" Hello World")
    print(" Hello World")
    print(" Hello World") 
```



```bash
print(" Hello World")
print(" Hello World")
print(" Hello World")
```

### 2.9 引用

使用`>`前缀来引用一段内容。

> **[这是一段引用内容]** Python是一种计算机程序设计语言。是一种面向对象的动态类型语言，最初被设计用于编写自动化脚本(shell)，随着版本的不断更新和语言新功能的添加，越来越多被用于独立的、大型项目的开发。

### 2.10 分割线

使用`***`或者`---`来加入分割线。

- 展示分割线
- Python是一种计算机程序设计语言。`是一种面向对象的动态类型语言，最初被设计用于编写自动化脚本(shell)，`随着版本的不断更新和语言新功能的添加，越来越多被用于独立的、大型项目的开发。

------

------

- 注：上面有两个分割线。

  注：与代码标记的区别。

### 2.11 链接与图片

图片：`[图片上传失败...(image-afdddf-1555948309118)]`
链接：`[文字](链接地址)`

- 插入图片

  

  [图片上传失败...(image-6d21de-1555948309118)]

  ![img](https://upload-images.jianshu.io/upload_images/16223186-f4a2f39887a0f5a2.jpg?imageMogr2/auto-orient/strip|imageView2/2/w/500/format/webp)

  1.jpg

- 插入链接
  [点击我访问百度](https://links.jianshu.com/go?to=http%3A%2F%2Fwww.baidu.com)

### 2.12 LaTex

LaTex是一个文件准备系统（document preparation system），用来进行排版，支持复杂的数学公式表示。LaTex公式使用$公式$或$$公式$$进行界定。 在Markdown类型的单元格中，支持LaTex数学公式。
LaTex在线编辑：[http://latex.codecogs.com/eqneditor/editor.php](https://links.jianshu.com/go?to=http%3A%2F%2Flatex.codecogs.com%2Feqneditor%2Feditor.php)
![y=x^2](https://math.jianshu.com/math?formula=y%3Dx%5E2)





## 李沐机器学习笔记

```
pip install d2l==0.17.3
mkdir d2l-zh && cd d2l-zh
curl https://zh-v2.d2l.ai/d2l-zh-2.0.0.zip -o d2l-zh.zip
unzip d2l-zh.zip && rm d2l-zh.zip
cd pytorch
```





### git配置

```
git config --global user.name "huangxioaxie"
git config --global user.email 1247324842@qq.com
```





# 最近用`thulac`分词发现很多小问题：

1. 加载模型后会直接print一个模型加载成功，一来看它不爽，二来如果多进程的话，这样直接往stdout上print东西可能会有麻烦。
2. 在`pythong>=3.8`版本以上会raise一个`AttributeError: module 'time' has no attribute 'clock'`，因为`time.clock`在3.8之后已经彻底被depracated了。
3. 在`cut_f`函数打开非系统默认编码的文件会raise一个解码错误。例如在默认GBK编码的系统中给函数传入一个utf8的文件，会raise`UnicodeDecodeError: 'gbk' codec can't decode byte …… illegal multibyte sequence `。

修改源码的话很简单，分别是：

1. 注释掉print行
2. 把time.clock改成time.pref_counter
3. 给open函数加一个`encoding='xxx'`

但是依赖修改源码对代码的可移植性是个破坏，所以记录一下不改源码的方案。

以下思路都是把方法定义成context manager，以便把我们对系统做的小动作恢复原样，所以首先：

```python3
from contextlib import contextmanager
```

## 屏蔽print

```python3
from io import StringIO

@contextmanager
def redirect_stdout_to_null():
    sys.stdout = NullIO()
    try:
        yield
    finally:
        sys.stdout = sys.__stdout__
```

之后可以使用下面方法来屏蔽掉输出

```python3
with redirect_stdout_to_null():
    thu = thulac.thulac(seg_only=True, T2S=True)
```

## 修正time.clock问题

```python3
import time

@contextmanager
def add_clock_method_to_time():
    py_gt_3_8 = not hasattr(time, "clock")
    if py_gt_3_8:
        setattr(time, "clock", time.perf_counter)
    try:
        yield
    finally:
        if py_gt_3_8:
            delattr(time, "clock")
```

同理可以在出现`AttributeError`的地方使用`with add_clock_method_to_time()`语句。

## 修正编码问题

```python3
@contextmanager
def use_utf8_open():
    from functools import partial
    import builtins

    builtin_open = open
    utf8_open = partial(open, encoding="utf-8")
    builtins.open = utf8_open

    try:
        yield
    finally:
        builtins.open = builtin_open
```

我这里直接改成了utf8，改成其他编码同理，之后就可以：

```python3
with use_utf8_open():
    thu.cut_f(input,output)
```