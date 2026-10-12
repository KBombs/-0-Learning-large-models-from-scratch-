# 中文机器学习入门课程

适合已掌握 Python 基础、准备从零学习机器学习的读者。学习方式：阅读写好的代码、解释和运行结果，无需手敲示例。

本项目从数据表、绘图和基础机器学习开始，逐步学习神经网络、PyTorch、Transformer 和大语言模型应用。目前已提供第 01 至 08 课；PyTorch 与 Transformer 课程将按学习进度继续补充。

教程尽量使用中文：讲解、代码注释、运行输出、表格字段和图表文字均以中文为主。Python 语法和库接口保留原名，必要的英文术语提供中文解释。

## 课程索引

按编号学习。每课都有 `README.md`（讲解）、`lesson.py`（完整示例）。生成的图片放在该课的 `outputs/` 中。

| 课次 | 内容 | 讲解 | 代码 |
| --- | --- | --- | --- |
| 01 | 数据表入门：创建表格、取列、求平均值 | [阅读](courses/01_数据表入门/README.md) | [代码](courses/01_数据表入门/lesson.py) |
| 02 | 数据筛选：条件判断、筛选行、统计结果 | [阅读](courses/02_数据筛选/README.md) | [代码](courses/02_数据筛选/lesson.py) |
| 03 | 散点图：观察面积与价格的关系 | [阅读](courses/03_散点图/README.md) | [代码](courses/03_散点图/lesson.py) |
| 04 | 线性回归：训练第一个模型并预测 | [阅读](courses/04_线性回归/README.md) | [代码](courses/04_线性回归/lesson.py) |
| 05 | 模型评估：训练集、测试集、平均绝对误差 | [阅读](courses/05_模型评估/README.md) | [代码](courses/05_模型评估/lesson.py) |
| 06 | 过拟合：训练表现与新样本表现的差异 | [阅读](courses/06_过拟合/README.md) | [代码](courses/06_过拟合/lesson.py) |
| 07 | 分类入门：预测类别、决策树与准确率 | [阅读](courses/07_分类入门/README.md) | [代码](courses/07_分类入门/lesson.py) |
| 08 | 神经网络入门：神经元、权重、偏置与前向传播 | [阅读](courses/08_神经网络入门/README.md) | [代码](courses/08_神经网络入门/lesson.py) |

当前已提供第 01 至 08 课。下一课计划学习损失函数：怎样把预测误差变成训练目标，完成后会继续放在 `courses/` 下。

## 安装与开始学习

需要 Python 3.10 或更高版本，建议使用 Python 3.12。项目通过 [uv](https://docs.astral.sh/uv/getting-started/installation/) 管理依赖，`uv.lock` 固定依赖版本，便于复现课程结果。

首次使用时，先根据 uv 官方安装说明安装 uv。用 PyCharm 打开本项目，在项目根目录的终端执行：

```powershell
uv sync --locked
uv run --locked python "courses/01_数据表入门/lesson.py"
```

`uv sync` 会在项目内创建 `.venv` 并安装所需依赖。换课时替换最后一行的课程路径即可；所有课程均可独立运行，不需要先运行其他课。

先从[第一课讲解](courses/01_数据表入门/README.md)开始。各课的 `outputs/` 已附上示例图片，即使暂时不运行代码，也可以直接阅读讲解和图像。

## 在 PyCharm 中查看

1. 展开 `courses`，进入想看的课次。
2. 先阅读该课的 `README.md`，再打开 `lesson.py`。
3. 需要重新运行时，右键该课的 `lesson.py`，选择 Run。
4. 有绘图的课程，其图片可以直接从该课的 `outputs/` 中打开。

所有课程共用项目根目录的 `.venv`，Python 解释器选择 `.venv\Scripts\python.exe`。
每课都能独立运行，无需先运行其他课。

## 项目结构

```text
pytorch_demo01/
├── README.md                 课程索引
├── LICENSE                   MIT 开源许可证
├── courses/                  按课次存放学习资料
│   ├── 01_数据表入门/
│   ├── 02_数据筛选/
│   ├── 03_散点图/
│   ├── 04_线性回归/
│   ├── 05_模型评估/
│   ├── 06_过拟合/
│   ├── 07_分类入门/
│   └── 08_神经网络入门/
├── pyproject.toml            项目依赖
├── uv.lock                   依赖版本
└── .venv/                    安装时创建的本地运行环境（不上传）
```

## 终端运行（可选）

在项目根目录的 PowerShell 中运行，例如第五课：

```powershell
.\.venv\Scripts\python.exe ".\courses\05_模型评估\lesson.py"
```

如需重新安装依赖：

```powershell
uv sync --locked --cache-dir .uv-cache
```

本课程的房屋数据均为虚构教学数据，第七课使用 scikit-learn 自带的鸢尾花测量数据。学习路线是数据处理、基础机器学习、神经网络与 PyTorch、Transformer、大语言模型应用。

## 中文图表显示

示例优先使用 Windows 的微软雅黑或黑体。其他系统若出现汉字方框，请安装可用的中文字体（例如 Noto Sans CJK SC），并在对应 `lesson.py` 的 `font.sans-serif` 列表中加入该字体名称。仓库内已有的示例图片可以直接查看。

## 开源许可证

本项目采用 [MIT 许可证](LICENSE)。你可以学习、使用、修改和分发项目内容，也可以用于商业用途；分发时需要保留原版权声明和许可证。具体条款以 `LICENSE` 为准。
