from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd


# ============================================================
# 1. 專案路徑
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "images_json"
OUTPUT_DIR = BASE_DIR / "output"


# ============================================================
# 2. 要分析的兩種五區分法
# ============================================================

SEGMENTATION_METHODS = [
    "tip_middle_root_1_2_2",
    "tip_middle_root_1_3_1",
]


# ============================================================
# 3. 基本設定
# ============================================================

REGIONS = [
    "舌尖",
    "舌中",
    "舌左邊",
    "舌右邊",
    "舌根",
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}

# Histogram 分成 8 個區間
HIST_BINS = 8


# ============================================================
# 4. 找出所有圖片
# ============================================================

image_files = sorted([
    file
    for file in DATA_DIR.iterdir()
    if (
        file.is_file()
        and
        file.suffix.lower() in IMAGE_EXTENSIONS
    )
])


print("=" * 70)
print("舌質色彩特徵萃取")
print("=" * 70)

print("原始圖片數：", len(image_files))

print("分析方法：")

for method in SEGMENTATION_METHODS:
    print(" -", method)


# ============================================================
# 5. 讀取某一種分區方法的比例設定
# ============================================================

def load_segmentation_settings(method_name):

    settings_path = (
        OUTPUT_DIR
        / method_name
        / "segmentation_settings.csv"
    )

    if not settings_path.exists():

        raise FileNotFoundError(
            f"\n找不到：{settings_path}\n"
            f"請先用 06_split_all_regions.py 跑完 "
            f"{method_name}"
        )

    settings = pd.read_csv(settings_path)

    if len(settings) == 0:
        raise ValueError(
            f"{settings_path} 是空的"
        )

    row = settings.iloc[0]

    config = {
        "tip_ratio": float(row["tip_ratio"]),
        "middle_ratio": float(row["middle_ratio"]),
        "root_ratio": float(row["root_ratio"]),
        "left_ratio": float(row["left_ratio"]),
        "center_ratio": float(row["center_ratio"]),
        "right_ratio": float(row["right_ratio"]),
    }

    return config


# ============================================================
# 6. Labelme JSON → 完整舌體 Mask
# ============================================================

def create_tongue_mask(
    json_path,
    height,
    width
):

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    tongue_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    tongue_found = False

    for shape in data.get(
        "shapes",
        []
    ):

        if shape.get("label") != "tongue":
            continue

        points = shape.get(
            "points",
            []
        )

        if len(points) < 3:
            continue

        points = np.array(
            points,
            dtype=np.int32
        )

        cv2.fillPoly(
            tongue_mask,
            [points],
            255
        )

        tongue_found = True

    if not tongue_found:
        return None

    return tongue_mask


# ============================================================
# 7. 完整舌體 Mask → 五區 Mask
# ============================================================

def create_region_masks(
    tongue_mask,
    config
):

    tip_ratio = config["tip_ratio"]
    root_ratio = config["root_ratio"]

    left_ratio = config["left_ratio"]
    center_ratio = config["center_ratio"]

    ys, xs = np.where(
        tongue_mask > 0
    )

    if (
        len(xs) == 0
        or
        len(ys) == 0
    ):
        return None

    # --------------------------------------------------------
    # Bounding Box
    # --------------------------------------------------------

    x_min = int(xs.min())
    x_max = int(xs.max())

    y_min = int(ys.min())
    y_max = int(ys.max())

    tongue_width = (
        x_max
        - x_min
        + 1
    )

    tongue_height = (
        y_max
        - y_min
        + 1
    )

    # --------------------------------------------------------
    # 縱向分界
    #
    # 圖片上方 = 舌根
    # 圖片下方 = 舌尖
    # --------------------------------------------------------

    root_end = int(
        y_min
        + tongue_height
        * root_ratio
    )

    tip_start = int(
        y_max
        - tongue_height
        * tip_ratio
    )

    middle_y1 = root_end
    middle_y2 = tip_start

    # --------------------------------------------------------
    # 橫向分界
    # --------------------------------------------------------

    left_end = int(
        x_min
        + tongue_width
        * left_ratio
    )

    center_end = int(
        x_min
        + tongue_width
        * (
            left_ratio
            + center_ratio
        )
    )

    region_masks = {}

    # ========================================================
    # 舌根
    # ========================================================

    mask = np.zeros_like(
        tongue_mask
    )

    mask[
        y_min:root_end,
        :
    ] = 255

    region_masks["舌根"] = (
        cv2.bitwise_and(
            mask,
            tongue_mask
        )
    )

    # ========================================================
    # 舌尖
    # ========================================================

    mask = np.zeros_like(
        tongue_mask
    )

    mask[
        tip_start:y_max + 1,
        :
    ] = 255

    region_masks["舌尖"] = (
        cv2.bitwise_and(
            mask,
            tongue_mask
        )
    )

    # ========================================================
    # 舌左邊
    # ========================================================

    mask = np.zeros_like(
        tongue_mask
    )

    mask[
        middle_y1:middle_y2,
        x_min:left_end
    ] = 255

    region_masks["舌左邊"] = (
        cv2.bitwise_and(
            mask,
            tongue_mask
        )
    )

    # ========================================================
    # 舌中
    # ========================================================

    mask = np.zeros_like(
        tongue_mask
    )

    mask[
        middle_y1:middle_y2,
        left_end:center_end
    ] = 255

    region_masks["舌中"] = (
        cv2.bitwise_and(
            mask,
            tongue_mask
        )
    )

    # ========================================================
    # 舌右邊
    # ========================================================

    mask = np.zeros_like(
        tongue_mask
    )

    mask[
        middle_y1:middle_y2,
        center_end:x_max + 1
    ] = 255

    region_masks["舌右邊"] = (
        cv2.bitwise_and(
            mask,
            tongue_mask
        )
    )

    return region_masks


# ============================================================
# 8. Histogram 計算
# ============================================================

def calculate_histogram(
    values,
    value_min,
    value_max,
    bins
):

    hist, _ = np.histogram(
        values,
        bins=bins,
        range=(
            value_min,
            value_max
        )
    )

    hist = hist.astype(
        np.float64
    )

    total = hist.sum()

    if total > 0:
        hist = hist / total

    return hist


# ============================================================
# 9. 單一區域色彩特徵
# ============================================================

def extract_color_features(
    image_bgr,
    region_mask
):

    valid_pixels = (
        region_mask > 0
    )

    pixel_count = int(
        np.sum(
            valid_pixels
        )
    )

    if pixel_count == 0:
        return None

    # ========================================================
    # RGB
    # ========================================================

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    rgb_pixels = (
        image_rgb[
            valid_pixels
        ]
        .astype(
            np.float32
        )
    )

    # ========================================================
    # HSV
    # ========================================================

    image_hsv = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2HSV
    )

    hsv_pixels = (
        image_hsv[
            valid_pixels
        ]
        .astype(
            np.float32
        )
    )

    # ========================================================
    # CIELAB
    # ========================================================

    image_lab = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2LAB
    )

    lab_pixels = (
        image_lab[
            valid_pixels
        ]
        .astype(
            np.float32
        )
    )

    features = {
        "pixel_count":
            pixel_count
    }


    # ========================================================
    # 10. RGB Mean / Standard Deviation
    # ========================================================

    rgb_names = [
        "R",
        "G",
        "B",
    ]

    for index, name in enumerate(
        rgb_names
    ):

        values = (
            rgb_pixels[
                :,
                index
            ]
        )

        features[
            f"{name}_mean"
        ] = float(
            np.mean(values)
        )

        features[
            f"{name}_std"
        ] = float(
            np.std(values)
        )


    # ========================================================
    # 11. HSV Mean / Standard Deviation
    # ========================================================

    hsv_names = [
        "H",
        "S",
        "V",
    ]

    for index, name in enumerate(
        hsv_names
    ):

        values = (
            hsv_pixels[
                :,
                index
            ]
        )

        features[
            f"{name}_mean"
        ] = float(
            np.mean(values)
        )

        features[
            f"{name}_std"
        ] = float(
            np.std(values)
        )


    # ========================================================
    # 12. Lab Mean / Standard Deviation
    # ========================================================

    lab_names = [
        "L",
        "a",
        "b",
    ]

    for index, name in enumerate(
        lab_names
    ):

        values = (
            lab_pixels[
                :,
                index
            ]
        )

        features[
            f"Lab_{name}_mean"
        ] = float(
            np.mean(values)
        )

        features[
            f"Lab_{name}_std"
        ] = float(
            np.std(values)
        )


    # ========================================================
    # 13. RGB Histograms
    # ========================================================

    for index, name in enumerate(
        rgb_names
    ):

        histogram = (
            calculate_histogram(
                rgb_pixels[
                    :,
                    index
                ],
                0,
                256,
                HIST_BINS
            )
        )

        for bin_index, value in enumerate(
            histogram
        ):

            features[
                f"{name}_hist_{bin_index}"
            ] = float(value)


    # ========================================================
    # 14. HSV Histograms
    #
    # OpenCV：
    # H = 0~179
    # S = 0~255
    # V = 0~255
    # ========================================================

    hsv_ranges = [
        (0, 180),
        (0, 256),
        (0, 256),
    ]

    for index, name in enumerate(
        hsv_names
    ):

        histogram = (
            calculate_histogram(
                hsv_pixels[
                    :,
                    index
                ],
                hsv_ranges[index][0],
                hsv_ranges[index][1],
                HIST_BINS
            )
        )

        for bin_index, value in enumerate(
            histogram
        ):

            features[
                f"{name}_hist_{bin_index}"
            ] = float(value)


    # ========================================================
    # 15. Lab Histograms
    # ========================================================

    for index, name in enumerate(
        lab_names
    ):

        histogram = (
            calculate_histogram(
                lab_pixels[
                    :,
                    index
                ],
                0,
                256,
                HIST_BINS
            )
        )

        for bin_index, value in enumerate(
            histogram
        ):

            features[
                f"Lab_{name}_hist_{bin_index}"
            ] = float(value)

    return features


# ============================================================
# 16. 單一分區方法的完整處理
# ============================================================

def process_method(
    method_name
):

    print("\n")
    print("=" * 70)
    print(
        "開始處理：",
        method_name
    )
    print("=" * 70)

    # --------------------------------------------------------
    # 載入 06 的比例
    # --------------------------------------------------------

    config = (
        load_segmentation_settings(
            method_name
        )
    )

    print(
        "舌尖 : 舌中 : 舌根 = "
        f"{config['tip_ratio']:.2f} : "
        f"{config['middle_ratio']:.2f} : "
        f"{config['root_ratio']:.2f}"
    )

    print(
        "左 : 中 : 右 = "
        f"{config['left_ratio']:.2f} : "
        f"{config['center_ratio']:.2f} : "
        f"{config['right_ratio']:.2f}"
    )


    method_output_dir = (
        OUTPUT_DIR
        / method_name
    )


    records = []

    success_images = 0
    failed_images = 0


    # ========================================================
    # 17. 處理所有圖片
    # ========================================================

    for number, image_path in enumerate(
        image_files,
        start=1
    ):

        image_id = (
            image_path.stem
        )

        json_path = (
            DATA_DIR
            /
            f"{image_id}.json"
        )

        print(
            f"[{number}/{len(image_files)}] "
            f"{image_path.name}"
        )

        # ----------------------------------------------------
        # 沒有 JSON
        # ----------------------------------------------------

        if not json_path.exists():

            print(
                "   ⚠ 找不到 JSON"
            )

            failed_images += 1

            continue


        # ----------------------------------------------------
        # 讀取圖片
        # ----------------------------------------------------

        image = cv2.imread(
            str(
                image_path
            )
        )

        if image is None:

            print(
                "   ⚠ 圖片讀取失敗"
            )

            failed_images += 1

            continue


        height, width = (
            image.shape[:2]
        )


        # ----------------------------------------------------
        # 舌體 Mask
        # ----------------------------------------------------

        try:

            tongue_mask = (
                create_tongue_mask(
                    json_path,
                    height,
                    width
                )
            )

        except Exception as error:

            print(
                "   ⚠ JSON 錯誤：",
                error
            )

            failed_images += 1

            continue


        if tongue_mask is None:

            print(
                "   ⚠ 找不到 tongue"
            )

            failed_images += 1

            continue


        # ----------------------------------------------------
        # 五區 Masks
        # ----------------------------------------------------

        region_masks = (
            create_region_masks(
                tongue_mask,
                config
            )
        )

        if region_masks is None:

            print(
                "   ⚠ 五區建立失敗"
            )

            failed_images += 1

            continue


        region_success = 0


        # ====================================================
        # 18. 五區各自萃取特徵
        # ====================================================

        for region_name in REGIONS:

            region_mask = (
                region_masks[
                    region_name
                ]
            )

            features = (
                extract_color_features(
                    image,
                    region_mask
                )
            )

            if features is None:

                print(
                    f"   ⚠ {region_name} "
                    "沒有有效像素"
                )

                continue


            row = {

                "segmentation_method":
                    method_name,

                "image_id":
                    image_id,

                "region":
                    region_name,
            }


            row.update(
                features
            )


            records.append(
                row
            )


            region_success += 1


        if region_success == 5:

            success_images += 1

            print(
                "   ✓ 五區特徵完成"
            )

        else:

            print(
                f"   ⚠ 只有 "
                f"{region_success}/5 區完成"
            )


    # ========================================================
    # 19. 建立 DataFrame
    # ========================================================

    feature_df = pd.DataFrame(
        records
    )


    # ========================================================
    # 20. 儲存該方法的 color_features.csv
    # ========================================================

    output_path = (
        method_output_dir
        /
        "color_features.csv"
    )


    feature_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )


    # ========================================================
    # 21. NaN 檢查
    # ========================================================

    if not feature_df.empty:

        feature_columns = [
            column
            for column
            in feature_df.columns

            if column not in [
                "segmentation_method",
                "image_id",
                "region"
            ]
        ]

        nan_count = int(
            feature_df[
                feature_columns
            ]
            .isna()
            .sum()
            .sum()
        )

    else:

        feature_columns = []
        nan_count = 0


    # ========================================================
    # 22. 顯示該方法結果
    # ========================================================

    print("\n")
    print("-" * 70)

    print(
        method_name,
        "完成"
    )

    print(
        "完整完成照片：",
        success_images
    )

    print(
        "失敗／跳過照片：",
        failed_images
    )

    print(
        "區域資料列：",
        len(
            feature_df
        )
    )

    print(
        "影像特徵數：",
        len(
            feature_columns
        )
    )

    print(
        "NaN：",
        nan_count
    )

    print(
        "輸出：",
        output_path
    )

    print("-" * 70)


    return feature_df


# ============================================================
# 23. 自動跑兩種分區方法
# ============================================================

all_method_data = []


for method_name in SEGMENTATION_METHODS:

    method_df = (
        process_method(
            method_name
        )
    )

    all_method_data.append(
        method_df
    )


# ============================================================
# 24. 額外建立兩種方法的總表
#
# 後面比較用，很方便
# ============================================================

if all_method_data:

    combined = pd.concat(
        all_method_data,
        ignore_index=True
    )

    combined_output = (
        OUTPUT_DIR
        /
        "color_features_all_methods.csv"
    )

    combined.to_csv(
        combined_output,
        index=False,
        encoding="utf-8-sig"
    )


    print("\n")
    print("=" * 70)
    print("兩種分區方法全部完成")
    print("=" * 70)

    print(
        "總資料列：",
        len(
            combined
        )
    )

    print(
        "分區方法數：",
        combined[
            "segmentation_method"
        ].nunique()
    )

    print(
        "照片數：",
        combined[
            "image_id"
        ].nunique()
    )

    print(
        "總表輸出：",
        combined_output
    )

    print("=" * 70)