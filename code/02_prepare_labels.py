from pathlib import Path
import pandas as pd
import re


# ============================================================
# 1. 專案路徑
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DOCTOR_DIR = BASE_DIR / "doctors"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. 設定舌區與舌色
# ============================================================

REGIONS = {
    "舌尖": "A1舌質色 [舌尖]",
    "舌中": "A1舌質色 [舌中]",
    "舌左邊": "A1舌質色 [舌左邊]",
    "舌右邊": "A1舌質色 [舌右邊]",
    "舌根": "A1舌質色 [舌根]",
}

COLORS = [
    "淡紅",
    "淡白",
    "鮮紅",
    "暗紅",
    "青紫",
    "灰黑",
]


# ============================================================
# 3. 標準化照片編號
# ============================================================

def normalize_image_id(value):

    if pd.isna(value):
        return None

    value = str(value).strip()

    # 移除 jpg / jpeg / png 副檔名
    value = re.sub(
        r"\.(jpg|jpeg|png)$",
        "",
        value,
        flags=re.IGNORECASE
    )

    return value


# ============================================================
# 4. 判斷某顏色是否存在於儲存格
# ============================================================

def has_color(cell_value, color):

    if pd.isna(cell_value):
        return 0

    text = str(cell_value).strip()

    if text == "":
        return 0

    return int(color in text)


# ============================================================
# 5. 找四份 Excel
# ============================================================

excel_files = sorted(DOCTOR_DIR.glob("*.xlsx"))

print("=" * 60)
print("四位醫師標籤整理")
print("=" * 60)

print(f"找到 Excel：{len(excel_files)} 份")

for file in excel_files:
    print(" -", file.name)


# ============================================================
# 6. 逐位醫師轉換
# ============================================================

all_doctors = []

for doctor_index, excel_file in enumerate(excel_files, start=1):

    doctor_name = f"D{doctor_index}"

    print("\n" + "=" * 60)
    print(f"處理 {doctor_name}: {excel_file.name}")
    print("=" * 60)

    df = pd.read_excel(excel_file)

    # 檢查必要欄位
    required_columns = [
        "照片編號",
        *REGIONS.values()
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        print("⚠ 缺少欄位：")

        for col in missing_columns:
            print(" -", col)

        continue

    # 標準化照片編號
    df["image_id"] = df["照片編號"].apply(normalize_image_id)

    # 建立輸出資料
    result = pd.DataFrame()

    result["image_id"] = df["image_id"]
    result["doctor"] = doctor_name

    # 五區 × 六色 = 30 個 binary labels
    for region_name, excel_column in REGIONS.items():

        for color in COLORS:

            output_column = f"{region_name}_{color}"

            result[output_column] = (
                df[excel_column]
                .apply(lambda x: has_color(x, color))
            )

    # 移除沒有照片編號的空白列
    result = result[result["image_id"].notna()].copy()

    # 單一醫師輸出
    doctor_output = OUTPUT_DIR / f"{doctor_name}_labels.csv"

    result.to_csv(
        doctor_output,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"✓ 輸出：{doctor_output.name}")
    print(f"資料筆數：{len(result)}")

    all_doctors.append(result)


# ============================================================
# 7. 合併四位醫師
# ============================================================

if all_doctors:

    combined = pd.concat(
        all_doctors,
        ignore_index=True
    )

    combined_output = OUTPUT_DIR / "all_doctors_labels.csv"

    combined.to_csv(
        combined_output,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 60)
    print("合併結果")
    print("=" * 60)

    print("總資料列數：", len(combined))
    print("不同照片數：", combined["image_id"].nunique())
    print("醫師數：", combined["doctor"].nunique())

    print("✓ 輸出：all_doctors_labels.csv")


# ============================================================
# 8. 統計各標籤出現次數
# ============================================================

if all_doctors:

    label_columns = [
        f"{region}_{color}"
        for region in REGIONS.keys()
        for color in COLORS
    ]

    counts = combined[label_columns].sum().sort_values(
        ascending=False
    )

    counts_output = OUTPUT_DIR / "label_counts.csv"

    counts.to_csv(
        counts_output,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 60)
    print("各標籤出現次數")
    print("=" * 60)

    print(counts)

    print("\n✓ 輸出：label_counts.csv")