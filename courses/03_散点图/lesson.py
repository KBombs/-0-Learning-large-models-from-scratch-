from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# 使用中文字体，让标题和坐标轴正常显示汉字。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    # 虚构教学数据：面积单位为平方米，价格单位为万元。
    df = pd.DataFrame({
        "面积": [50, 60, 80, 100, 110],
        "价格": [100, 120, 160, 200, 220],
    })

    # 每个点代表一套房屋：横坐标是面积，纵坐标是价格。
    plt.figure(figsize=(7, 4.5))
    plt.scatter(df["面积"], df["价格"], color="teal", s=70)
    plt.xlabel("面积（平方米）")
    plt.ylabel("价格（万元）")
    plt.title("房屋面积与价格的关系（教学数据）")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    chart_path = output_dir / "area_price.png"
    plt.savefig(chart_path, dpi=160)
    plt.close()
    print("图像已保存到：", chart_path)


if __name__ == "__main__":
    main()
