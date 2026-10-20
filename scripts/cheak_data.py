from collections import Counter
from pathlib import Path

import pandas as pd

root = Path("~/competition/dataset/glaucoma_first").expanduser()

for split in ("train", "valid", "test"):
    df = pd.read_csv(root / f"{split}.csv")
    names = df["Image_name"].astype(str)

    files = {
        p.name for p in (root / split).iterdir()
        if p.is_file()
    }
    folded = Counter(name.casefold() for name in files)

    exact = []
    case_only = []
    missing = []
    ambiguous = []

    for name in names:
        if name in files:
            exact.append(name)
        elif folded[name.casefold()] == 1:
            case_only.append(name)
        elif folded[name.casefold()] > 1:
            ambiguous.append(name)
        else:
            missing.append(name)

    print(f"\n===== {split} =====")
    print("CSV 记录数:", len(df))
    print("重复文件名记录数:", names.duplicated().sum())
    print("年龄缺失数:", df["Age"].isna().sum())
    print("文件名完全匹配:", len(exact))
    print("仅大小写不同:", len(case_only))
    print("找不到图片:", len(missing))
    print("大小写匹配存在歧义:", len(ambiguous))

    if case_only:
        print("大小写差异示例:", case_only[:3])
    if missing:
        print("缺失示例:", missing[:5])
    if ambiguous:
        print("歧义示例:", ambiguous[:5])

    if "True_Label" in df.columns:
        labels = df["True_Label"]
        print("标签数量:", labels.value_counts(dropna=False).to_dict())
        print("非法标签数:", (~labels.isin([0, 1])).sum())
        print(f"阳性比例: {labels.eq(1).mean():.4%}")