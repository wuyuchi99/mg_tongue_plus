from pathlib import Path
import pandas as pd

# ============================================================
# 1. 路徑
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

# ============================================================
# 2. 讀取四位醫師整理後的 CSV
# ============================================================

doctor_files = {
    "D1": OUTPUT_DIR / "D1_labels.csv",
    "D2": OUTPUT_DIR / "D2_labels.csv",
    "D3": OUTPUT_DIR / "D3_labels.csv",
    "D4": OUTPUT_DIR / "D4_labels.csv",
}

doctor_data = {}

for doctor, file in doctor_files.items():

    df = pd.read_csv(
        file,
        dtype={"image_id": str}
    )

    # 去除前後空格
    df["image_id"] = df["image_id"].str.strip()

    doctor_data[doctor] = df

    print(
        doctor,
        "資料筆數：",
        len(df),
        "不同照片數：",
        df["image_id"].nunique()
    )


# ============================================================
# 3. 建立照片編號集合
# ============================================================

id_sets = {
    doctor: set(df["image_id"].dropna())
    for doctor, df in doctor_data.items()
}

all_ids = set().union(*id_sets.values())


# ============================================================
# 4. 找每位醫師缺少的照片
# ============================================================

print("\n" + "=" * 60)
print("每位醫師缺少的照片")
print("=" * 60)

for doctor in doctor_data:

    missing = sorted(
        all_ids - id_sets[doctor]
    )

    print(f"\n{doctor} 缺少 {len(missing)} 張：")

    if missing:

        for image_id in missing:
            print("  -", image_id)

    else:
        print("  無")


# ============================================================
# 5. 檢查是否有重複照片編號
# ============================================================

print("\n" + "=" * 60)
print("重複照片編號檢查")
print("=" * 60)

for doctor, df in doctor_data.items():

    duplicated = df[
        df["image_id"].duplicated(
            keep=False
        )
    ]

    print(f"\n{doctor}：")

    if duplicated.empty:
        print("  無重複")
    else:
        print(
            duplicated[
                ["image_id"]
            ].sort_values("image_id")
        )


# ============================================================
# 6. 四位醫師共同都有的病例
# ============================================================

common_ids = set.intersection(
    *id_sets.values()
)

print("\n" + "=" * 60)
print("共同病例")
print("=" * 60)

print(
    "四位醫師共同都有標記的照片：",
    len(common_ids)
)

print(
    "全部曾出現過的照片：",
    len(all_ids)
)