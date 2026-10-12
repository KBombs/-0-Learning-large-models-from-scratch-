import pandas as pd

pd.set_option("display.unicode.east_asian_width", True)


def main():
    # 虚构教学数据：面积单位为平方米，价格单位为万元。
    data = {
        "面积": [50, 60, 80, 100, 110],
        "价格": [100, 120, 160, 200, 220],
    }
    df = pd.DataFrame(data)

    print("房屋数据（面积：平方米，价格：万元）：")
    print(df)
    print("\n面积这一列：")
    print(df["面积"])
    print("\n平均面积（平方米）：", df["面积"].mean())
    print("表格大小（行数，列数）：", df.shape)
    print("\n价格这一列：")
    print(df["价格"])
    print("平均价格（万元）：", df["价格"].mean())


if __name__ == "__main__":
    main()
