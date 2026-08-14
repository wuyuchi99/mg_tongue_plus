from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd


# ============================================================
# 1. 分區方法選擇
#
# 第一次跑 1:2:2
# 跑完之後改成 1:3:1，再跑一次
# ============================================================

# METHOD_NAME = "tip_middle_root_1_2_2"

# 第二個版本要跑時改成：
METHOD_NAME = "tip_middle_root_1_3_1"


# ============================================================
# 2. 兩種分區方法設定
#
# 注意：
#
# 名稱中的比例順序都是：
# 舌尖 : 舌中 : 舌根
#
# 但是照片座標是：
# 上方 = 舌根
# 下方 = 舌尖
#
# 橫向兩種方法統一：
# 舌左邊 : 舌中 : 舌右邊
# = 1 : 2 : 1
# = 25% : 50% : 25%
# ============================================================

METHOD_CONFIGS = {

    # --------------------------------------------------------
    # 方法一
    # 舌尖 : 舌中 : 舌根 = 1 : 2 : 2
    #
    # 舌尖 = 20%
    # 舌中 = 40%
    # 舌根 = 40%
    # --------------------------------------------------------

    "tip_middle_root_1_2_2": {

        "folder":
            "tip_middle_root_1_2_2",

        "description":
            "Tongue tip : middle : root = 1 : 2 : 2; "
            "left : center : right = 1 : 2 : 1",

        "tip_ratio":
            0.20,

        "root_ratio":
            0.40,

        "left_ratio":
            0.25,

        "center_ratio":
            0.50,

        "right_ratio":
            0.25,
    },


    # --------------------------------------------------------
    # 方法二
    # 舌尖 : 舌中 : 舌根 = 1 : 3 : 1
    #
    # 舌尖 = 20%
    # 舌中 = 60%
    # 舌根 = 20%
    # --------------------------------------------------------

    "tip_middle_root_1_3_1": {

        "folder":
            "tip_middle_root_1_3_1",

        "description":
            "Tongue tip : middle : root = 1 : 3 : 1; "
            "left : center : right = 1 : 2 : 1",

        "tip_ratio":
            0.20,

        "root_ratio":
            0.20,

        "left_ratio":
            0.25,

        "center_ratio":
            0.50,

        "right_ratio":
            0.25,
    }
}


# ============================================================
# 3. 檢查 METHOD_NAME
# ============================================================

if METHOD_NAME not in METHOD_CONFIGS:

    raise ValueError(
        "METHOD_NAME 設定錯誤。\n"
        "只能使用：\n"
        "tip_middle_root_1_2_2\n"
        "或\n"
        "tip_middle_root_1_3_1"
    )


CONFIG = METHOD_CONFIGS[
    METHOD_NAME
]


TIP_RATIO = (
    CONFIG[
        "tip_ratio"
    ]
)

ROOT_RATIO = (
    CONFIG[
        "root_ratio"
    ]
)

MIDDLE_RATIO = (
    1.0
    - TIP_RATIO
    - ROOT_RATIO
)


LEFT_RATIO = (
    CONFIG[
        "left_ratio"
    ]
)

CENTER_RATIO = (
    CONFIG[
        "center_ratio"
    ]
)

RIGHT_RATIO = (
    CONFIG[
        "right_ratio"
    ]
)


# ============================================================
# 4. 安全檢查比例
# ============================================================

vertical_sum = (
    TIP_RATIO
    + MIDDLE_RATIO
    + ROOT_RATIO
)

horizontal_sum = (
    LEFT_RATIO
    + CENTER_RATIO
    + RIGHT_RATIO
)


if not np.isclose(
    vertical_sum,
    1.0
):

    raise ValueError(
        "舌尖、舌中、舌根比例總和不是 1"
    )


if not np.isclose(
    horizontal_sum,
    1.0
):

    raise ValueError(
        "左、中、右比例總和不是 1"
    )


# ============================================================
# 5. 專案路徑
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


DATA_DIR = (
    BASE_DIR
    / "images_json"
)


OUTPUT_DIR = (
    BASE_DIR
    / "output"
)


METHOD_DIR = (
    OUTPUT_DIR
    / CONFIG["folder"]
)


REGION_DIR = (
    METHOD_DIR
    / "regions"
)


PREVIEW_DIR = (
    METHOD_DIR
    / "region_previews"
)


# ============================================================
# 6. 建立資料夾
# ============================================================

OUTPUT_DIR.mkdir(
    exist_ok=True
)

METHOD_DIR.mkdir(
    exist_ok=True
)

REGION_DIR.mkdir(
    exist_ok=True
)

PREVIEW_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# 7. 五個舌區
# ============================================================

REGIONS = [
    "舌尖",
    "舌中",
    "舌左邊",
    "舌右邊",
    "舌根",
]


for region in REGIONS:

    (
        REGION_DIR
        / region
    ).mkdir(
        exist_ok=True
    )


# ============================================================
# 8. 支援的圖片格式
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


# ============================================================
# 9. 找出所有圖片
# ============================================================

image_files = sorted([
    file
    for file in DATA_DIR.iterdir()

    if (
        file.is_file()
        and
        file.suffix.lower()
        in IMAGE_EXTENSIONS
    )
])


# ============================================================
# 10. 顯示目前設定
# ============================================================

print("=" * 70)
print("舌象五區批次分割")
print("=" * 70)

print(
    "分區方法：",
    METHOD_NAME
)

print(
    "方法說明：",
    CONFIG[
        "description"
    ]
)

print()

print(
    "舌尖：",
    f"{TIP_RATIO * 100:.0f}%"
)

print(
    "舌中：",
    f"{MIDDLE_RATIO * 100:.0f}%"
)

print(
    "舌根：",
    f"{ROOT_RATIO * 100:.0f}%"
)

print()

print(
    "舌左邊：",
    f"{LEFT_RATIO * 100:.0f}%"
)

print(
    "中央：",
    f"{CENTER_RATIO * 100:.0f}%"
)

print(
    "舌右邊：",
    f"{RIGHT_RATIO * 100:.0f}%"
)

print()

print(
    "找到圖片：",
    len(
        image_files
    )
)


# ============================================================
# 11. Labelme JSON → 舌體 Mask
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

        data = json.load(
            f
        )


    tongue_mask = np.zeros(
        (
            height,
            width
        ),
        dtype=np.uint8
    )


    tongue_found = False


    for shape in data.get(
        "shapes",
        []
    ):

        # 只讀 tongue
        if (
            shape.get(
                "label"
            )
            !=
            "tongue"
        ):

            continue


        points = (
            shape.get(
                "points",
                []
            )
        )


        if len(
            points
        ) < 3:

            continue


        points = np.array(
            points,
            dtype=np.int32
        )


        cv2.fillPoly(
            tongue_mask,
            [
                points
            ],
            255
        )


        tongue_found = True


    if not tongue_found:

        return None


    return tongue_mask


# ============================================================
# 12. 舌體 Mask → 五區 Mask
# ============================================================

def create_region_masks(
    tongue_mask
):

    # --------------------------------------------------------
    # 找到舌體所有像素
    # --------------------------------------------------------

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
    # 舌體 Bounding Box
    # --------------------------------------------------------

    x_min = int(
        xs.min()
    )

    x_max = int(
        xs.max()
    )

    y_min = int(
        ys.min()
    )

    y_max = int(
        ys.max()
    )


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


    # ========================================================
    # 13. 計算縱向分界
    #
    # y 越小 = 圖片越上面 = 舌根
    #
    # y 越大 = 圖片越下面 = 舌尖
    # ========================================================

    root_end = int(
        y_min
        +
        tongue_height
        *
        ROOT_RATIO
    )


    tip_start = int(
        y_max
        -
        tongue_height
        *
        TIP_RATIO
    )


    # 中段就是：
    #
    # root_end
    # ↓
    # 舌中
    # ↓
    # tip_start

    middle_y1 = (
        root_end
    )

    middle_y2 = (
        tip_start
    )


    # ========================================================
    # 14. 計算橫向分界
    # ========================================================

    left_end = int(
        x_min
        +
        tongue_width
        *
        LEFT_RATIO
    )


    center_end = int(
        x_min
        +
        tongue_width
        *
        (
            LEFT_RATIO
            +
            CENTER_RATIO
        )
    )


    # ========================================================
    # 15. 建立區域 Masks
    # ========================================================

    region_masks = {}


    # ========================================================
    # 舌根
    # ========================================================

    root_mask = np.zeros_like(
        tongue_mask
    )


    root_mask[
        y_min:root_end,
        :
    ] = 255


    root_mask = (
        cv2.bitwise_and(
            root_mask,
            tongue_mask
        )
    )


    region_masks[
        "舌根"
    ] = root_mask


    # ========================================================
    # 舌尖
    # ========================================================

    tip_mask = np.zeros_like(
        tongue_mask
    )


    tip_mask[
        tip_start:y_max + 1,
        :
    ] = 255


    tip_mask = (
        cv2.bitwise_and(
            tip_mask,
            tongue_mask
        )
    )


    region_masks[
        "舌尖"
    ] = tip_mask


    # ========================================================
    # 舌左邊
    # ========================================================

    left_mask = np.zeros_like(
        tongue_mask
    )


    left_mask[
        middle_y1:middle_y2,
        x_min:left_end
    ] = 255


    left_mask = (
        cv2.bitwise_and(
            left_mask,
            tongue_mask
        )
    )


    region_masks[
        "舌左邊"
    ] = left_mask


    # ========================================================
    # 舌中
    # ========================================================

    center_mask = np.zeros_like(
        tongue_mask
    )


    center_mask[
        middle_y1:middle_y2,
        left_end:center_end
    ] = 255


    center_mask = (
        cv2.bitwise_and(
            center_mask,
            tongue_mask
        )
    )


    region_masks[
        "舌中"
    ] = center_mask


    # ========================================================
    # 舌右邊
    # ========================================================

    right_mask = np.zeros_like(
        tongue_mask
    )


    right_mask[
        middle_y1:middle_y2,
        center_end:x_max + 1
    ] = 255


    right_mask = (
        cv2.bitwise_and(
            right_mask,
            tongue_mask
        )
    )


    region_masks[
        "舌右邊"
    ] = right_mask


    return region_masks


# ============================================================
# 16. 裁出單一區域
# ============================================================

def crop_region(
    image,
    region_mask
):

    ys, xs = np.where(
        region_mask > 0
    )


    if (
        len(xs) == 0
        or
        len(ys) == 0
    ):

        return None


    x1 = int(
        xs.min()
    )

    x2 = int(
        xs.max()
    )

    y1 = int(
        ys.min()
    )

    y2 = int(
        ys.max()
    )


    # --------------------------------------------------------
    # 舌頭以外的地方設為黑色
    # --------------------------------------------------------

    masked_image = (
        cv2.bitwise_and(
            image,
            image,
            mask=region_mask
        )
    )


    # --------------------------------------------------------
    # 裁成該區域的最小 Bounding Box
    # --------------------------------------------------------

    cropped = (
        masked_image[
            y1:y2 + 1,
            x1:x2 + 1
        ]
    )


    return cropped


# ============================================================
# 17. 預覽圖顏色
#
# 注意：
# OpenCV 使用 BGR
#
# 這些顏色只用來看預覽，
# 不會影響原始舌象資料
# ============================================================

PREVIEW_COLORS = {

    # 藍色
    "舌尖": (
        255,
        0,
        0
    ),

    # 綠色
    "舌中": (
        0,
        255,
        0
    ),

    # 黃色
    "舌左邊": (
        0,
        255,
        255
    ),

    # 紫色
    "舌右邊": (
        255,
        0,
        255
    ),

    # 紅色
    "舌根": (
        0,
        0,
        255
    ),
}


# ============================================================
# 18. 開始批次處理
# ============================================================

records = []

success_count = 0
skip_count = 0


for index, image_path in enumerate(
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
        f"[{index}/{len(image_files)}] "
        f"{image_path.name}"
    )


    # --------------------------------------------------------
    # 沒有 JSON
    # --------------------------------------------------------

    if not json_path.exists():

        print(
            "   ⚠ 沒有對應 JSON，跳過"
        )

        skip_count += 1

        continue


    # --------------------------------------------------------
    # 讀取圖片
    # --------------------------------------------------------

    image = cv2.imread(
        str(
            image_path
        )
    )


    if image is None:

        print(
            "   ⚠ 圖片無法讀取"
        )

        skip_count += 1

        continue


    height, width = (
        image.shape[:2]
    )


    # --------------------------------------------------------
    # 建立完整舌體 Mask
    # --------------------------------------------------------

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
            "   ⚠ JSON 讀取失敗：",
            error
        )

        skip_count += 1

        continue


    if tongue_mask is None:

        print(
            "   ⚠ 找不到 tongue label"
        )

        skip_count += 1

        continue


    # --------------------------------------------------------
    # 建立五區
    # --------------------------------------------------------

    region_masks = (
        create_region_masks(
            tongue_mask
        )
    )


    if region_masks is None:

        print(
            "   ⚠ 無法建立五區"
        )

        skip_count += 1

        continue


    # ========================================================
    # 19. 建立彩色預覽
    # ========================================================

    overlay = np.zeros_like(
        image
    )


    for (
        region_name,
        region_mask
    ) in region_masks.items():

        overlay[
            region_mask > 0
        ] = (
            PREVIEW_COLORS[
                region_name
            ]
        )


    preview = cv2.addWeighted(
        image,
        0.65,
        overlay,
        0.35,
        0
    )


    # --------------------------------------------------------
    # 畫白色舌體外輪廓
    # --------------------------------------------------------

    contours, _ = (
        cv2.findContours(
            tongue_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
    )


    cv2.drawContours(
        preview,
        contours,
        -1,
        (
            255,
            255,
            255
        ),
        2
    )


    # --------------------------------------------------------
    # 儲存 Preview
    # --------------------------------------------------------

    preview_path = (
        PREVIEW_DIR
        /
        f"{image_id}_preview.jpg"
    )


    cv2.imwrite(
        str(
            preview_path
        ),
        preview
    )


    # ========================================================
    # 20. 儲存五個舌區
    # ========================================================

    valid_regions = 0


    for (
        region_name,
        region_mask
    ) in region_masks.items():


        cropped = (
            crop_region(
                image,
                region_mask
            )
        )


        if cropped is None:

            print(
                f"   ⚠ {region_name} "
                "沒有有效像素"
            )

            continue


        region_output = (
            REGION_DIR
            /
            region_name
            /
            f"{image_id}.png"
        )


        cv2.imwrite(
            str(
                region_output
            ),
            cropped
        )


        pixel_count = int(
            np.sum(
                region_mask > 0
            )
        )


        # ----------------------------------------------------
        # Manifest 紀錄
        # ----------------------------------------------------

        records.append({

            "segmentation_method":
                METHOD_NAME,

            "image_id":
                image_id,

            "region":
                region_name,

            "original_image":
                str(
                    image_path
                ),

            "region_file":
                str(
                    region_output
                ),

            "pixel_count":
                pixel_count,

            "tip_ratio":
                TIP_RATIO,

            "middle_ratio":
                MIDDLE_RATIO,

            "root_ratio":
                ROOT_RATIO,

            "left_ratio":
                LEFT_RATIO,

            "center_ratio":
                CENTER_RATIO,

            "right_ratio":
                RIGHT_RATIO,
        })


        valid_regions += 1


    # --------------------------------------------------------
    # 確認五區都有
    # --------------------------------------------------------

    if valid_regions == 5:

        success_count += 1

        print(
            "   ✓ 五區完成"
        )


    else:

        print(
            f"   ⚠ 只有 "
            f"{valid_regions}/5 區完成"
        )


# ============================================================
# 21. 建立 Manifest
# ============================================================

manifest = pd.DataFrame(
    records
)


manifest_path = (
    METHOD_DIR
    /
    "regions_manifest.csv"
)


manifest.to_csv(
    manifest_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 22. 儲存分區設定
#
# 之後寫論文或檢查版本時可以直接查看
# ============================================================

settings = pd.DataFrame([
    {

        "segmentation_method":
            METHOD_NAME,

        "description":
            CONFIG[
                "description"
            ],

        "tip_ratio":
            TIP_RATIO,

        "middle_ratio":
            MIDDLE_RATIO,

        "root_ratio":
            ROOT_RATIO,

        "left_ratio":
            LEFT_RATIO,

        "center_ratio":
            CENTER_RATIO,

        "right_ratio":
            RIGHT_RATIO,
    }
])


settings_path = (
    METHOD_DIR
    /
    "segmentation_settings.csv"
)


settings.to_csv(
    settings_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 23. 最終總結
# ============================================================

print("\n")
print("=" * 70)
print("舌象五區分割完成")
print("=" * 70)


print(
    "分區方法：",
    METHOD_NAME
)


print(
    "舌尖 : 舌中 : 舌根 =",
    f"{TIP_RATIO:.2f} : "
    f"{MIDDLE_RATIO:.2f} : "
    f"{ROOT_RATIO:.2f}"
)


print(
    "左 : 中 : 右 =",
    f"{LEFT_RATIO:.2f} : "
    f"{CENTER_RATIO:.2f} : "
    f"{RIGHT_RATIO:.2f}"
)


print()


print(
    "原始圖片：",
    len(
        image_files
    )
)


print(
    "成功完成五區：",
    success_count
)


print(
    "跳過／異常：",
    skip_count
)


print(
    "產生區域圖片：",
    len(
        records
    )
)


print("\n輸出資料夾：")

print(
    METHOD_DIR
)


print("\n五區圖片：")

print(
    REGION_DIR
)


print("\n預覽圖片：")

print(
    PREVIEW_DIR
)


print("\nManifest：")

print(
    manifest_path
)


print("\n分區設定：")

print(
    settings_path
)


print("=" * 70)