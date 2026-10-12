from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# 使用中文字体，表格显示时按汉字宽度对齐。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
pd.set_option("display.unicode.east_asian_width", True)


def main():
    # 虚构教学数据：面积单位为平方米，价格单位为万元。
    df = pd.DataFrame({
        "面积": [50, 60, 80, 100, 110],
        "价格": [100, 120, 160, 200, 220],
    })

    # X 保持表格形式：每行是一套房屋，每列是一个输入特征。
    X = df[["面积"]]
    y = df["价格"]

    model = LinearRegression()
    model.fit(X, y)

    # 这套房屋是新输入，没有出现在训练数据中。
    new_house = pd.DataFrame({"面积": [90]})
    predicted_price = model.predict(new_house)[0]

    print("训练数据（面积：平方米，价格：万元）：")
    print(df)
    print("\n输入大小（样本数，特征数）：", X.shape)
    print(f"学到的系数：{model.coef_[0]:.4f}")
    print(f"学到的截距：{model.intercept_:.4f}")
    print("模型公式：价格 = 系数 × 面积 + 截距")
    print(f"\n90 平方米房屋的预测价格：{predicted_price:.2f} 万元")

    # 两个端点就足以画出直线模型的预测关系。
    line_areas = pd.DataFrame({"面积": [50, 110]})
    line_prices = model.predict(line_areas)

    plt.figure(figsize=(7, 4.5))
    plt.scatter(df["面积"], df["价格"], color="teal", s=70,
                label="训练数据", zorder=3)
    plt.plot(line_areas["面积"], line_prices, color="coral",
             label="学到的直线")
    plt.scatter(new_house["面积"], [predicted_price], color="crimson",
                marker="*", s=180, label="90 平方米房屋的预测", zorder=4)
    plt.xlabel("面积（平方米）")
    plt.ylabel("价格（万元）")
    plt.title("第一个模型：线性回归（教学数据）")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    chart_path = output_dir / "linear_regression.png"
    plt.savefig(chart_path, dpi=160)
    plt.close()
    print("\n图像已保存到：", chart_path)


if __name__ == "__main__":
    main()
