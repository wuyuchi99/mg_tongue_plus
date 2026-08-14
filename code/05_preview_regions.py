from pathlib import Path
import json
import cv2
import numpy as np


# ============================================================
# 1. 路徑設定
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "images_json"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. 指定一張測試圖片
#    先改成你資料夾裡真的存在的檔名
# ============================================================

IMAGE_NAME = "9.jpg" 

IMAGE_PATH = DATA_DIR / IMAGE_NAME
JSON_PATH = DATA_DIR / f"{Path(IMAGE_NAME).stem}.json"


# ============================================================
# 3. 讀取圖片
# ============================================================

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(
        f"找不到圖片：{IMAGE_PATH}"
    )

height, width = image.shape[:2]


# ============================================================
# 4. 讀取 Labelme JSON
# ============================================================

with open(
    JSON_PATH,
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)


# ============================================================
# 5. 建立舌體 mask
# ============================================================

tongue_mask = np.zeros(
    (height, width),
    dtype=np.uint8
)

tongue_found = False

for shape in data.get("shapes", []):

    if shape.get("label") != "tongue":
        continue

    points = np.array(
        shape["points"],
        dtype=np.int32
    )

    cv2.fillPoly(
        tongue_mask,
        [points],
        255
    )

    tongue_found = True


if not tongue_found:
    raise ValueError(
        "JSON 裡找不到 label = tongue"
    )


# ============================================================
# 6. 找舌體 bounding box
# ============================================================

ys, xs = np.where(tongue_mask > 0)

x_min = xs.min()
x_max = xs.max()

y_min = ys.min()
y_max = ys.max()

tongue_width = x_max - x_min + 1
tongue_height = y_max - y_min + 1


# ============================================================
# 7. 暫定五區比例
# ============================================================

TIP_RATIO = 0.25
ROOT_RATIO = 0.25

LEFT_RATIO = 0.30
CENTER_RATIO = 0.40
RIGHT_RATIO = 0.30


# ============================================================
# 8. 計算上下分界
# ============================================================

root_end = int(
    y_min + tongue_height * ROOT_RATIO
)

tip_start = int(
    y_max - tongue_height * TIP_RATIO
)


# ============================================================
# 9. 計算左右分界
# ============================================================

left_end = int(
    x_min + tongue_width * LEFT_RATIO
)

center_end = int(
    x_min + tongue_width *
    (LEFT_RATIO + CENTER_RATIO)
)


# ============================================================
# 10. 建立五個區域 mask
# ============================================================

region_masks = {}

# 舌根
root_mask = np.zeros_like(tongue_mask)

root_mask[
    y_min:root_end,
    :
] = 255

root_mask = cv2.bitwise_and(
    root_mask,
    tongue_mask
)

region_masks["舌根"] = root_mask


# 舌尖
tip_mask = np.zeros_like(tongue_mask)

tip_mask[
    tip_start:y_max + 1,
    :
] = 255

tip_mask = cv2.bitwise_and(
    tip_mask,
    tongue_mask
)

region_masks["舌尖"] = tip_mask


# 中段高度
middle_y1 = root_end
middle_y2 = tip_start


# 舌左邊
left_mask = np.zeros_like(tongue_mask)

left_mask[
    middle_y1:middle_y2,
    x_min:left_end
] = 255

left_mask = cv2.bitwise_and(
    left_mask,
    tongue_mask
)

region_masks["舌左邊"] = left_mask


# 舌中
center_mask = np.zeros_like(tongue_mask)

center_mask[
    middle_y1:middle_y2,
    left_end:center_end
] = 255

center_mask = cv2.bitwise_and(
    center_mask,
    tongue_mask
)

region_masks["舌中"] = center_mask


# 舌右邊
right_mask = np.zeros_like(tongue_mask)

right_mask[
    middle_y1:middle_y2,
    center_end:x_max + 1
] = 255

right_mask = cv2.bitwise_and(
    right_mask,
    tongue_mask
)

region_masks["舌右邊"] = right_mask


# ============================================================
# 11. 建立彩色預覽圖
# ============================================================

preview = image.copy()

overlay = np.zeros_like(image)

region_colors = {
    "舌尖": (255, 0, 0),
    "舌中": (0, 255, 0),
    "舌左邊": (0, 255, 255),
    "舌右邊": (255, 0, 255),
    "舌根": (0, 0, 255),
}


for region_name, mask in region_masks.items():

    color = region_colors[region_name]

    overlay[
        mask > 0
    ] = color


# 半透明疊加
preview = cv2.addWeighted(
    image,
    0.65,
    overlay,
    0.35,
    0
)


# ============================================================
# 12. 畫出舌體輪廓
# ============================================================

contours, _ = cv2.findContours(
    tongue_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

cv2.drawContours(
    preview,
    contours,
    -1,
    (255, 255, 255),
    2
)


# ============================================================
# 13. 儲存預覽
# ============================================================

OUTPUT_PATH = (
    OUTPUT_DIR /
    f"{Path(IMAGE_NAME).stem}_regions_preview.jpg"
)

cv2.imwrite(
    str(OUTPUT_PATH),
    preview
)

print("=" * 60)
print("五區預覽完成")
print("=" * 60)

print("圖片：", IMAGE_NAME)
print("輸出：", OUTPUT_PATH)

print("\n區域像素數：")

for region_name, mask in region_masks.items():

    pixel_count = np.sum(
        mask > 0
    )

    print(
        region_name,
        ":",
        pixel_count
    )