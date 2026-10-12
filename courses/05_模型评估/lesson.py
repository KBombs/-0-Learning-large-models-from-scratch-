from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# 使用中文字体，表格显示时按汉字宽度对齐。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
pd.set_option("display.unicode.east_asian_width", True)


def main():
    # 虚构教学数据带有变化：面积单位为平方米，价格单位为万元。
    df = pd.DataFrame({
        "面积": [40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150],
        "价格": [95, 108, 125, 142, 150, 184, 198, 215, 232, 248, 270, 278],
    })
    X = df[["面积"]]
    y = df["价格"]

    # 训练前，将输入和对应答案一起划分，保持配对关系。
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # 先预测，再与保留的测试答案比较。
    mae = mean_absolute_error(y_test, predictions)

    # 简单基准：每次都猜训练集的平均价格。
    mean_price = y_train.mean()
    baseline_predictions = [mean_price] * len(y_test)
    baseline_mae = mean_absolute_error(y_test, baseline_predictions)

    results = X_test.copy()
    results["实际价格"] = y_test
    results["预测价格"] = predictions
    results["绝对误差"] = (
        results["预测价格"] - results["实际价格"]
    ).abs()
    results = results.sort_values("面积")

    print(f"训练样本数：{len(X_train)}")
    print(f"测试样本数：{len(X_test)}")
    print("\n训练数据（面积：平方米，价格：万元）：")
    print(df.loc[X_train.index].sort_values("面积").to_string(index=False))
    print("\n测试结果：")
    print(results.round(2).to_string(index=False))
    print("\n价格和误差的单位均为万元。")
    print(f"模型平均绝对误差（MAE）：{mae:.2f}")
    print(f"基准猜测值（训练集平均价格）：{mean_price:.2f}")
    print(f"基准平均绝对误差（MAE）：{baseline_mae:.2f}")
    print("平均绝对误差：预测与答案的绝对差，再求平均值。")

    plt.figure(figsize=(7.5, 4.8))
    plt.scatter(X_train["面积"], y_train, color="teal", s=55,
                label="训练数据")
    plt.scatter(results["面积"], results["实际价格"], color="darkorange",
                marker="s", s=75, label="测试样本的实际价格", zorder=3)
    plt.scatter(results["面积"], results["预测价格"], color="crimson",
                marker="x", s=85, label="测试样本的预测价格", zorder=4)
    plt.vlines(results["面积"], results["实际价格"],
               results["预测价格"], color="gray", linestyles="dashed")
    plt.xlabel("面积（平方米）")
    plt.ylabel("价格（万元）")
    plt.title("在测试数据上检查预测效果（教学示例）")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    chart_path = output_dir / "model_evaluation.png"
    plt.savefig(chart_path, dpi=160)
    plt.close()
    print("\n图像已保存到：", chart_path)


if __name__ == "__main__":
    main()
