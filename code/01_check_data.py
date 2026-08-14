from pathlib import Path
import pandas as pd
import json


# ============================================================
# 1. 設定專案路徑
# ============================================================

# 01_check_data.py 位於 tongue_project/code/
# 所以 parent.parent 就是 tongue_project/
BASE_DIR = Path(__file__).resolve().parent.parent

DOCTOR_DIR = BASE_DIR / "doctors"
DATA_DIR = BASE_DIR / "images_json"
OUTPUT_DIR = BASE_DIR / "output"


# ============================================================
# 2. 檢查四份醫師 Excel
# ============================================================

excel_files = sorted(DOCTOR_DIR.glob("*.xlsx"))

print("=" * 60)
print("一、醫師 Excel 檢查")
print("=" * 60)

print(f"找到 Excel：{len(excel_files)} 份")

for file in excel_files:
    print("✓", file.name)

if len(excel_files) == 4:
    print("\n✓ 醫師 Excel 數量正確：4 份")
else:
    print(f"\n⚠ 預期 4 份，目前有 {len(excel_files)} 份")


# ============================================================
# 3. 檢查 Excel 必要欄位
# ============================================================

required_columns = [
    "照片編號",
    "A1舌質色 [舌尖]",
    "A1舌質色 [舌中]",
    "A1舌質色 [舌左邊]",
    "A1舌質色 [舌右邊]",
    "A1舌質色 [舌根]",
]

print("\n")
print("=" * 60)
print("二、Excel 欄位檢查")
print("=" * 60)

for file in excel_files:

    print(f"\n檢查：{file.name}")

    try:
        df = pd.read_excel(file)

        print(f"資料列數：{len(df)}")

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if not missing_columns:
            print("✓ 五個舌區欄位全部存在")

        else:
            print("⚠ 缺少以下欄位：")

            for column in missing_columns:
                print("  -", column)

    except Exception as e:
        print("⚠ Excel 無法讀取：", e)


# ============================================================
# 4. 搜尋圖片
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG",
}

image_files = [
    file
    for file in DATA_DIR.iterdir()
    if file.is_file()
    and file.suffix in IMAGE_EXTENSIONS
]

print("\n")
print("=" * 60)
print("三、照片檢查")
print("=" * 60)

print(f"找到照片：{len(image_files)} 張")

if len(image_files) == 100:
    print("✓ 照片數量正確：100 張")
else:
    print(f"⚠ 預期 100 張，目前有 {len(image_files)} 張")


# ============================================================
# 5. 搜尋 JSON
# ============================================================

json_files = list(DATA_DIR.glob("*.json"))

print("\n")
print("=" * 60)
print("四、JSON 檢查")
print("=" * 60)

print(f"找到 JSON：{len(json_files)} 份")

if len(json_files) == 100:
    print("✓ JSON 數量正確：100 份")
else:
    print(f"⚠ 預期 100 份，目前有 {len(json_files)} 份")


# ============================================================
# 6. 比較圖片與 JSON 檔名
# ============================================================

image_names = {
    file.stem
    for file in image_files
}

json_names = {
    file.stem
    for file in json_files
}

images_without_json = sorted(
    image_names - json_names
)

json_without_images = sorted(
    json_names - image_names
)


print("\n")
print("=" * 60)
print("五、照片 ↔ JSON 配對檢查")
print("=" * 60)


if len(images_without_json) == 0:
    print("✓ 所有照片都有同名 JSON")
else:
    print(
        f"⚠ 有 {len(images_without_json)} 張照片沒有 JSON："
    )

    for name in images_without_json:
        print("  -", name)


if len(json_without_images) == 0:
    print("✓ 所有 JSON 都有同名照片")
else:
    print(
        f"⚠ 有 {len(json_without_images)} 個 JSON 沒有照片："
    )

    for name in json_without_images:
        print("  -", name)


# ============================================================
# 7. 檢查 Labelme JSON
# ============================================================

print("\n")
print("=" * 60)
print("六、Labelme JSON 內容檢查")
print("=" * 60)

valid_json = 0
problem_json = []

labels_found = set()

for json_file in json_files:

    try:

        with open(
            json_file,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        shapes = data.get("shapes", [])

        if len(shapes) == 0:

            problem_json.append(
                (
                    json_file.name,
                    "沒有 shapes"
                )
            )

            continue

        has_tongue = False

        for shape in shapes:

            label = shape.get("label")

            if label is not None:
                labels_found.add(label)

            if label == "tongue":
                has_tongue = True

        if not has_tongue:

            problem_json.append(
                (
                    json_file.name,
                    "找不到 tongue label"
                )
            )

            continue

        valid_json += 1

    except Exception as e:

        problem_json.append(
            (
                json_file.name,
                str(e)
            )
        )


print(f"有效 JSON：{valid_json}")

print("JSON 中找到的 label：")
print(labels_found)


if len(problem_json) == 0:
    print("✓ 所有 JSON 都有 tongue 標記")

else:

    print(
        f"⚠ 有 {len(problem_json)} 份 JSON 有問題："
    )

    for filename, problem in problem_json:
        print(
            f"  - {filename}: {problem}"
        )


# ============================================================
# 8. 最後總結
# ============================================================

print("\n")
print("=" * 60)
print("資料檢查總結")
print("=" * 60)

print(f"醫師 Excel：{len(excel_files)}")
print(f"照片：{len(image_files)}")
print(f"JSON：{len(json_files)}")
print(f"有效 JSON：{valid_json}")

print(
    "照片缺 JSON：",
    len(images_without_json)
)

print(
    "JSON 缺照片：",
    len(json_without_images)
)

print(
    "JSON 問題數：",
    len(problem_json)
)

print("=" * 60)