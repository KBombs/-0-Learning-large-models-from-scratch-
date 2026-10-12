from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

# 使用中文字体，表格显示时按汉字宽度对齐。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
pd.set_option("display.unicode.east_asian_width", True)


def main():
    # 鸢尾花是真实的小型数据集，随 scikit-learn 提供，测量单位为厘米。
    iris = load_iris(as_frame=True)
    X = iris.data.copy()
    X.columns = ["花萼长度（厘米）", "花萼宽度（厘米）", "花瓣长度（厘米）", "花瓣宽度（厘米）"]
    y = iris.target
    class_names = {0: "山鸢尾", 1: "变色鸢尾", 2: "维吉尼亚鸢尾"}

    # 划分时保持三个品种的比例。
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7, stratify=y
    )

    # 在检查测试结果前，预先固定树的深度限制。
    model = DecisionTreeClassifier(max_depth=3, random_state=7)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    correct_count = int((predictions == y_test.to_numpy()).sum())

    # 简单基准：每次都猜训练集中数量最多的一个品种。
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_accuracy = accuracy_score(y_test, baseline.predict(X_test))

    results = X_test.copy()
    results["实际品种"] = y_test.map(class_names)
    results["预测品种"] = [class_names[label] for label in predictions]
    results["预测正确"] = results["实际品种"] == results["预测品种"]
    display_results = results.copy()
    display_results["预测正确"] = results["预测正确"].map({True: "是", False: "否"})

    print("输入特征（测量单位：厘米）：")
    print(X.head().to_string(index=False))
    print("\n类别编码与品种名称：", class_names)
    print(f"\n训练样本数：{len(X_train)}｜测试样本数：{len(X_test)}")
    print(f"测试中判断正确的数量：{correct_count}/{len(X_test)}")
    print(f"测试集准确率：{accuracy:.2%}")
    print(f"每次猜同一品种的基准准确率：{baseline_accuracy:.2%}")
    print("\n前六个测试样本的结果：")
    print(display_results.head(6).to_string())
    print("\n判断错误的测试样本：")
    mistakes = display_results[~results["预测正确"]]
    if mistakes.empty:
        print("这次划分没有判断错误的样本。")
    else:
        print(mistakes.to_string())
    print("\n从训练数据中学到的判断规则：")
    rules = export_text(model, feature_names=list(X.columns))
    for label, name in class_names.items():
        rules = rules.replace(f"class: {label}", f"品种：{name}")
    print(rules)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    plot_tree(
        model, feature_names=list(X.columns),
        class_names=list(class_names.values()), filled=True, rounded=True,
        impurity=False, precision=2, fontsize=9, ax=ax,
    )
    # 将绘图库自动生成的节点和分支标签翻译成中文。
    translations = {
        "samples =": "样本数 =", "value =": "各类数量 =",
        "class =": "预测品种 =", "True": "是", "False": "否",
    }
    for text in ax.texts:
        content = text.get_text()
        for english, chinese in translations.items():
            content = content.replace(english, chinese)
        text.set_text(content)
    ax.set_title("鸢尾花分类：从训练数据中学到的判断规则")
    fig.tight_layout()

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    chart_path = output_dir / "decision_tree.png"
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)
    print("\n图像已保存到：", chart_path)


if __name__ == "__main__":
    main()
