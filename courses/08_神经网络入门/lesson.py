from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.linear_model import LinearRegression

# 所有图表和运行结果优先使用中文。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
pd.set_option("display.unicode.east_asian_width", True)


def linear_neuron(x, weight, bias):
    """一个输入的线性神经元：输入乘以权重，再加上偏置。"""
    return weight * x + bias


def relu(z):
    """ReLU 激活函数：负数变成零，零和正数保持原值。"""
    return max(0, z)


def small_network(area):
    # 为看清计算过程，下面的参数由我们设定，没有经过训练。
    # 隐藏神经元 1 保留面积；神经元 2 提取超过 80 平方米的部分。
    hidden_1 = relu(1 * area + 0)
    hidden_2 = relu(1 * area - 80)

    # 输出层将两个中间结果组合成一个价格，单位为万元。
    # 回归任务的输出层可以直接输出数值，这里不再使用 ReLU。
    price = 2 * hidden_1 + 1 * hidden_2 + 0
    return hidden_1, hidden_2, price


def draw_network(output_dir):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("90 平方米房屋：小网络的前向计算（参数人为设定）", pad=18)

    # 三列依次是输入、隐藏层和输出；框内展示本次计算的数值。
    nodes = [
        (0.02, 0.40, "输入\n面积 = 90 平方米", "#e0f2fe"),
        (0.39, 0.65, "隐藏神经元 1\nReLU(1 × 90 + 0)\n= 90", "#dcfce7"),
        (0.39, 0.15, "隐藏神经元 2\nReLU(1 × 90 − 80)\n= 10", "#dcfce7"),
        (0.76, 0.40, "输出神经元\n2 × 90 + 1 × 10 + 0\n= 190 万元", "#ffedd5"),
    ]
    for x, y, label, color in nodes:
        box = FancyBboxPatch(
            (x, y), 0.22, 0.20, boxstyle="round,pad=0.012",
            facecolor=color, edgecolor="#475569", linewidth=1.2,
        )
        ax.add_patch(box)
        ax.text(x + 0.11, y + 0.10, label, ha="center", va="center", fontsize=11)

    connections = [
        ((0.25, 0.54), (0.37, 0.74), (0.30, 0.70), "权重 1"),
        ((0.25, 0.46), (0.37, 0.26), (0.30, 0.29), "权重 1"),
        ((0.63, 0.74), (0.74, 0.54), (0.69, 0.70), "权重 2"),
        ((0.63, 0.26), (0.74, 0.46), (0.69, 0.29), "权重 1"),
    ]
    for start, end, text_position, label in connections:
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=15,
            linewidth=1.5, color="#475569",
        ))
        ax.text(*text_position, label, ha="center", va="center", fontsize=10)

    ax.text(0.13, 0.93, "输入层", ha="center", fontsize=12)
    ax.text(0.50, 0.93, "隐藏层：两个神经元", ha="center", fontsize=12)
    ax.text(0.87, 0.93, "输出层", ha="center", fontsize=12)
    ax.text(
        0.50, 0.035, "先从左往右计算；隐藏层使用 ReLU，输出层直接求加权和。",
        ha="center", fontsize=11,
    )
    fig.tight_layout()
    chart_path = output_dir / "network_structure.png"
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)
    return chart_path


def draw_curves(output_dir, weight, bias):
    # 将一张图分成两部分：先看激活函数，再看它带来的形状变化。
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    inputs = list(range(-100, 101))
    axes[0].plot(inputs, [relu(z) for z in inputs], color="teal", linewidth=2.5)
    axes[0].axhline(0, color="gray", linewidth=0.8)
    axes[0].axvline(0, color="gray", linewidth=0.8)
    axes[0].set_title("ReLU：负数归零，正数保留")
    axes[0].set_xlabel("激活前的数值")
    axes[0].set_ylabel("激活后的数值")

    areas = list(range(40, 121))
    linear_prices = [linear_neuron(area, weight, bias) for area in areas]
    network_prices = [small_network(area)[2] for area in areas]
    axes[1].plot(areas, linear_prices, label="第 04 课学到的直线", linewidth=2)
    axes[1].plot(areas, network_prices, label="人为设定的小网络", linewidth=2)
    axes[1].axvline(80, color="gray", linestyle="--", alpha=0.7)
    axes[1].annotate(
        "超过 80 平方米：第二个神经元输出正值", xy=(80, 160),
        xytext=(43, 250), fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "gray"},
    )
    axes[1].set_title("观察关系的形状（不比较预测好坏）")
    axes[1].set_xlabel("面积（平方米）")
    axes[1].set_ylabel("输出价格（万元）")
    axes[1].legend(loc="upper left", fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    chart_path = output_dir / "activation_and_curves.png"
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)
    return chart_path


def main():
    # 复用第 04 课的虚构数据，演示训练得到的参数如何参与计算。
    df = pd.DataFrame({
        "面积": [50, 60, 80, 100, 110],
        "价格": [100, 120, 160, 200, 220],
    })
    model = LinearRegression()
    model.fit(df[["面积"]], df["价格"])
    weight = float(model.coef_[0])
    bias = float(model.intercept_)

    print("一、将第 04 课的直线理解成一个线性神经元")
    print(f"训练学到的权重：{weight:.2f}｜偏置：{bias:.2f}")
    prediction = linear_neuron(90, weight, bias)
    print(f"90 平方米：{weight:.2f} × 90 + {bias:.2f} = {prediction:.2f} 万元")
    library_prediction = model.predict(pd.DataFrame({"面积": [90]}))[0]
    print(f"库的预测结果：{library_prediction:.2f} 万元")

    print("\n二、激活函数 ReLU：负数变零，正数保留")
    for z in [-10, 0, 10]:
        print(f"ReLU({z}) = {relu(z)}")

    print("\n三、小网络：下面的权重和偏置由我们设定，没有训练")
    print("隐藏神经元 1：ReLU(面积)")
    print("隐藏神经元 2：ReLU(面积 − 80)")
    print("输出价格：2 × 神经元 1 的输出 + 1 × 神经元 2 的输出 + 0")
    results = []
    for area in [70, 90, 100]:
        hidden_1, hidden_2, price = small_network(area)
        results.append({
            "面积（平方米）": area,
            "隐藏神经元1输出": hidden_1,
            "隐藏神经元2输出": hidden_2,
            "网络输出（万元）": price,
        })
    print(pd.DataFrame(results).to_string(index=False))
    print("\n从输入向输出计算一次，叫前向传播；训练还需要调整参数。")
    print("本课演示计算结构，未使用测试集评价小网络的预测质量。")

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    for path in [draw_network(output_dir), draw_curves(output_dir, weight, bias)]:
        print("图像已保存到：", path)


if __name__ == "__main__":
    main()
