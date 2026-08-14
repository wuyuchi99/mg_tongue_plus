from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 1. 路徑設定
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

doctor_files = {
    "D1": OUTPUT_DIR / "D1_labels.csv",
    "D2": OUTPUT_DIR / "D2_labels.csv",
    "D3": OUTPUT_DIR / "D3_labels.csv",
    "D4": OUTPUT_DIR / "D4_labels.csv",
}


# ============================================================
# 2. 讀取四位醫師資料
# ============================================================

doctor_data = {}

for doctor, file in doctor_files.items():
    df = pd.read_csv(file, dtype={"image_id": str})
    df["image_id"] = df["image_id"].astype(str).str.strip()
    doctor_data[doctor] = df

    print(f"{doctor}：資料筆數 {len(df)}，不同照片 {df['image_id'].nunique()}")


# ============================================================
# 3. 找出四位醫師共同都有的病例
# ============================================================

common_ids = set(doctor_data["D1"]["image_id"])

for doctor in ["D2", "D3", "D4"]:
    common_ids = common_ids.intersection(set(doctor_data[doctor]["image_id"]))

common_ids = sorted(common_ids)

print("\n" + "=" * 60)
print("共同病例數")
print("=" * 60)
print("四位醫師共同都有的照片：", len(common_ids))


# ============================================================
# 4. 只保留共同病例，並依 image_id 排序
# ============================================================

for doctor in doctor_data:
    df = doctor_data[doctor]
    df = df[df["image_id"].isin(common_ids)].copy()
    df = df.sort_values("image_id").reset_index(drop=True)
    doctor_data[doctor] = df


# ============================================================
# 5. 標籤欄位
# ============================================================

label_columns = [
    col for col in doctor_data["D1"].columns
    if col not in ["image_id", "doctor"]
]

print("標籤欄位數量：", len(label_columns))


# ============================================================
# 6. 建立 long-format 共識表
# ============================================================

rows = []

for image_id in common_ids:

    d1_row = doctor_data["D1"][doctor_data["D1"]["image_id"] == image_id].iloc[0]
    d2_row = doctor_data["D2"][doctor_data["D2"]["image_id"] == image_id].iloc[0]
    d3_row = doctor_data["D3"][doctor_data["D3"]["image_id"] == image_id].iloc[0]
    d4_row = doctor_data["D4"][doctor_data["D4"]["image_id"] == image_id].iloc[0]

    for label in label_columns:

        v1 = int(d1_row[label])
        v2 = int(d2_row[label])
        v3 = int(d3_row[label])
        v4 = int(d4_row[label])

        vote_count = v1 + v2 + v3 + v4

        # 共識規則：
        # 4:0 / 3:1 -> 1
        # 0:4 / 1:3 -> 0
        # 2:2 -> uncertain
        if vote_count >= 3:
            consensus = 1
            consensus_type = "positive_consensus"
        elif vote_count <= 1:
            consensus = 0
            consensus_type = "negative_consensus"
        else:
            consensus = -1   # 用 -1 表示 uncertain
            consensus_type = "uncertain"

        agreement = max(vote_count, 4 - vote_count) / 4.0

        rows.append({
            "image_id": image_id,
            "label": label,
            "D1": v1,
            "D2": v2,
            "D3": v3,
            "D4": v4,
            "vote_count": vote_count,
            "agreement": agreement,
            "consensus": consensus,
            "consensus_type": consensus_type
        })

consensus_long = pd.DataFrame(rows)


# ============================================================
# 7. 輸出 long-format 共識表
# ============================================================

consensus_long_output = OUTPUT_DIR / "consensus_long.csv"
consensus_long.to_csv(consensus_long_output, index=False, encoding="utf-8-sig")

print("\n" + "=" * 60)
print("已輸出 consensus_long.csv")
print("=" * 60)
print("資料列數：", len(consensus_long))


# ============================================================
# 8. 建立 wide-format 共識表
#    一列一張照片，30 個共識欄位
#    uncertain 會標成 -1
# ============================================================

consensus_wide = (
    consensus_long
    .pivot(index="image_id", columns="label", values="consensus")
    .reset_index()
)

consensus_wide_output = OUTPUT_DIR / "consensus_wide.csv"
consensus_wide.to_csv(consensus_wide_output, index=False, encoding="utf-8-sig")

print("已輸出 consensus_wide.csv")
print("照片數：", len(consensus_wide))


# ============================================================
# 9. 統計共識情況
# ============================================================

summary = (
    consensus_long["consensus_type"]
    .value_counts()
    .rename_axis("consensus_type")
    .reset_index(name="count")
)

summary_output = OUTPUT_DIR / "consensus_summary.csv"
summary.to_csv(summary_output, index=False, encoding="utf-8-sig")

print("\n" + "=" * 60)
print("共識統計")
print("=" * 60)
print(summary)

print("\n已輸出 consensus_summary.csv")


# ============================================================
# 10. 再細看每個標籤的共識狀況
# ============================================================

label_summary = (
    consensus_long
    .groupby(["label", "consensus_type"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

label_summary_output = OUTPUT_DIR / "consensus_by_label.csv"
label_summary.to_csv(label_summary_output, index=False, encoding="utf-8-sig")

print("已輸出 consensus_by_label.csv")


# ============================================================
# 11. 顯示 uncertain 數量
# ============================================================

uncertain_count = (consensus_long["consensus"] == -1).sum()

print("\n" + "=" * 60)
print("Uncertain 統計")
print("=" * 60)
print("uncertain 數量：", uncertain_count)

print("\n完成。")
