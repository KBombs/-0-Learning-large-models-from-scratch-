from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# 使用中文字体，表格显示时按汉字宽度对齐。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
pd.set_option("display.unicode.east_asian_width", True)


def main():
    # 沿用第五课的虚构数据和划分，让两个模型在相同条件下比较。
    df = pd.DataFrame({
        "面积": [40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150],
        "价格": [95, 108, 125, 142, 150, 184, 198, 215, 232, 248, 270, 278],
    })
    X = df[["面积"]]
    y = df["价格"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7
    )

    # 本例中的直线有两个可调整参数：系数和截距。
    simple_model = LinearRegression()

    # 曲线有更多自由度，数值缩放帮助高次方计算保持稳定。
    # 整个流程只从训练集学习缩放参数和模型参数。
    complex_model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=8, include_bias=False),
        LinearRegression(),
    )
    models = {
        "直线模型": simple_model,
        "八次曲线": complex_model,
    }

    # NumPy 生成密集的面积取值，方便画出平滑曲线。
    grid = pd.DataFrame({"面积": np.linspace(40, 150, 400)})
    records = []
    test_results = X_test.copy()
    test_results["实际价格"] = y_test
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)

    for ax, (name, model) in zip(axes, models.items()):
        model.fit(X_train, y_train)
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        train_mae = mean_absolute_error(y_train, train_predictions)
        test_mae = mean_absolute_error(y_test, test_predictions)
        records.append({
            "模型": name,
            "训练集MAE": train_mae,
            "测试集MAE": test_mae,
        })
        test_results[name] = test_predictions

        ax.scatter(X_train["面积"], y_train, color="teal", s=45,
                   label="训练数据", zorder=3)
        ax.scatter(X_test["面积"], y_test, color="darkorange", marker="s",
                   s=65, label="测试样本的实际价格", zorder=3)
        ax.plot(grid["面积"], model.predict(grid), color="crimson",
                label="学到的预测关系")
        ax.scatter(X_test["面积"], test_predictions, color="crimson",
                   marker="x", s=65, zorder=4)
        ax.vlines(X_test["面积"], y_test, test_predictions,
                  color="gray", linestyles="dashed")
        ax.set_title(f"{name}\n训练误差：{train_mae:.2f}｜测试误差：{test_mae:.2f}")
        ax.set_xlabel("面积（平方米）")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("价格（万元）")
    fig.suptitle("过拟合：训练表现与新样本预测的差异（误差指标为 MAE）")
    fig.tight_layout()

    print(f"训练样本数：{len(X_train)}｜测试样本数：{len(X_test)}")
    print("\n平均绝对误差（MAE）比较，单位：万元")
    print(pd.DataFrame(records).round(2).to_string(index=False))
    print("\n测试样本的价格预测，单位：万元")
    print(test_results.sort_values("面积").round(2).to_string(index=False))
    print("\n训练误差显示为 0.00 是保留两位小数的结果，内部仍可能有极小误差。")
    print("本例用于教学对比；实际选模型时，应保留独立的最终测试数据。")

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    chart_path = output_dir / "overfitting.png"
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)
    print("\n图像已保存到：", chart_path)


if __name__ == "__main__":
    main()
