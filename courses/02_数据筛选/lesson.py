import pandas as pd

pd.set_option("display.unicode.east_asian_width", True)


def main():
    # 虚构教学数据：面积单位为平方米，价格单位为万元。
    df = pd.DataFrame({
        "面积": [50, 60, 80, 100, 110],
        "价格": [100, 120, 160, 200, 220],
    })
    print("房屋数据（面积：平方米，价格：万元）：")
    print(df)

    # 先逐个比较面积，再保留判断结果为 True 的行。
    condition = df["面积"] >= 80
    print("\n判断条件：面积大于或等于 80 平方米")
    print(condition)
    large_houses = df[condition]
    print("\n面积大于或等于 80 平方米的房屋：")
    print(large_houses)

    # 也可以把比较和筛选合成一个表达式。
    expensive_houses = df[df["价格"] >= 200]
    print("\n价格大于或等于 200 万元的房屋：")
    print(expensive_houses)
    average_area = expensive_houses["面积"].mean()
    print("这些房屋的平均面积（平方米）：", average_area)


if __name__ == "__main__":
    main()
