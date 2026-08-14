from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. 路徑
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

VOTE_FILE = (
    OUTPUT_DIR
    / "interrater_vote_details.csv"
)


# ============================================================
# 2. 基本設定
# ============================================================

REGIONS = [
    "舌尖",
    "舌中",
    "舌左邊",
    "舌右邊",
    "舌根",
]

COLORS = [
    "淡紅",
    "淡白",
    "鮮紅",
    "暗紅",
    "青紫",
    "灰黑",
]

DOCTORS = [
    "D1",
    "D2",
    "D3",
    "D4",
]


# ============================================================
# 3. 讀取資料
# ============================================================

if not VOTE_FILE.exists():

    raise FileNotFoundError(
        f"找不到：{VOTE_FILE}"
    )


df = pd.read_csv(
    VOTE_FILE,
    dtype={
        "image_id": str
    }
)


df["image_id"] = (
    df["image_id"]
    .astype(str)
    .str.strip()
    .str.replace(
        r"\.0$",
        "",
        regex=True
    )
)


# ============================================================
# 4. 判斷每一個 region-color item 的醫師一致型態
# ============================================================

df["is_unanimous"] = (

    (df["D1"] == df["D2"])
    &
    (df["D1"] == df["D3"])
    &
    (df["D1"] == df["D4"])
)


df["positive_votes"] = (
    df[DOCTORS]
    .sum(
        axis=1
    )
)


df["agreement_type"] = np.select(

    [
        df["positive_votes"].isin(
            [
                0,
                4
            ]
        ),

        df["positive_votes"].isin(
            [
                1,
                3
            ]
        ),

        df["positive_votes"] == 2,
    ],

    [
        "4:0",
        "3:1",
        "2:2",
    ],

    default="unknown"
)


# ============================================================
# 5. Majority Ground Truth
#
# 0/1 votes -> negative
# 3/4 votes -> positive
# 2 votes   -> uncertain (-1)
# ============================================================

df["majority_label"] = np.select(

    [
        df["positive_votes"] <= 1,
        df["positive_votes"] >= 3,
        df["positive_votes"] == 2,
    ],

    [
        0,
        1,
        -1,
    ],

    default=-1
)


print("=" * 90)
print("依舌質色定義 Unanimous Image")
print("=" * 90)

print(
    "總影像數：",
    df["image_id"].nunique()
)


# ============================================================
# 6. 對每一種顏色：
#
# 一張照片的五個區域全部四醫師一致
# → unanimous training image
#
# 只要五區中至少一區不是四醫師完全一致
# → disagreement image
# ============================================================

image_color_records = []


for color in COLORS:

    color_df = (
        df[
            df["color"]
            ==
            color
        ]
        .copy()
    )


    for image_id, group in color_df.groupby(
        "image_id"
    ):

        n_regions = len(
            group
        )

        unanimous_regions = int(
            group[
                "is_unanimous"
            ].sum()
        )

        three_one_regions = int(
            np.sum(
                group[
                    "agreement_type"
                ]
                ==
                "3:1"
            )
        )

        two_two_regions = int(
            np.sum(
                group[
                    "agreement_type"
                ]
                ==
                "2:2"
            )
        )


        # ----------------------------------------------------
        # 該顏色五區全部四醫師相同
        # ----------------------------------------------------

        unanimous_image = (

            n_regions
            ==
            5

            and

            unanimous_regions
            ==
            5
        )


        image_color_records.append({

            "image_id":
                image_id,

            "color":
                color,

            "n_regions":
                n_regions,

            "unanimous_regions":
                unanimous_regions,

            "three_one_regions":
                three_one_regions,

            "two_two_regions":
                two_two_regions,

            "unanimous_image":
                unanimous_image,

            "disagreement_image":
                not unanimous_image,
        })


image_color_df = pd.DataFrame(
    image_color_records
)


# ============================================================
# 7. 輸出 image × color 分組
# ============================================================

image_color_output = (
    OUTPUT_DIR
    /
    "unanimous_image_by_color_details.csv"
)


image_color_df.to_csv(
    image_color_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 8. 每種顏色統計：
#
# train images
# disagreement test images
# ============================================================

summary_records = []


for color in COLORS:

    color_image_df = (
        image_color_df[
            image_color_df[
                "color"
            ]
            ==
            color
        ]
        .copy()
    )


    unanimous_ids = set(

        color_image_df.loc[
            color_image_df[
                "unanimous_image"
            ],
            "image_id"
        ]

    )


    disagreement_ids = set(

        color_image_df.loc[
            color_image_df[
                "disagreement_image"
            ],
            "image_id"
        ]

    )


    # ========================================================
    # 9. 訓練資料：
    # unanimous images 的五區
    # ========================================================

    train_items = (
        df[
            (
                df["color"]
                ==
                color
            )
            &
            (
                df["image_id"]
                .isin(
                    unanimous_ids
                )
            )
        ]
        .copy()
    )


    train_positive = int(
        np.sum(
            train_items[
                "majority_label"
            ]
            ==
            1
        )
    )


    train_negative = int(
        np.sum(
            train_items[
                "majority_label"
            ]
            ==
            0
        )
    )


    # ========================================================
    # 10. Disagreement test images
    #
    # 所有可以建立 Ground Truth 的 item：
    # 4:0 + 3:1
    #
    # 2:2 不計準確率
    # ========================================================

    test_items = (
        df[
            (
                df["color"]
                ==
                color
            )
            &
            (
                df["image_id"]
                .isin(
                    disagreement_ids
                )
            )
        ]
        .copy()
    )


    test_evaluable = (
        test_items[
            test_items[
                "majority_label"
            ]
            !=
            -1
        ]
        .copy()
    )


    test_three_one = (
        test_items[
            test_items[
                "agreement_type"
            ]
            ==
            "3:1"
        ]
        .copy()
    )


    test_two_two = (
        test_items[
            test_items[
                "agreement_type"
            ]
            ==
            "2:2"
        ]
        .copy()
    )


    test_positive = int(
        np.sum(
            test_evaluable[
                "majority_label"
            ]
            ==
            1
        )
    )


    test_negative = int(
        np.sum(
            test_evaluable[
                "majority_label"
            ]
            ==
            0
        )
    )


    three_one_positive = int(
        np.sum(
            test_three_one[
                "majority_label"
            ]
            ==
            1
        )
    )


    three_one_negative = int(
        np.sum(
            test_three_one[
                "majority_label"
            ]
            ==
            0
        )
    )


    summary_records.append({

        "color":
            color,

        # ----------------------------------------------------
        # Image-level
        # ----------------------------------------------------

        "unanimous_train_images":
            len(
                unanimous_ids
            ),

        "disagreement_test_images":
            len(
                disagreement_ids
            ),

        # ----------------------------------------------------
        # Training region items
        # ----------------------------------------------------

        "train_region_items":
            len(
                train_items
            ),

        "train_positive":
            train_positive,

        "train_negative":
            train_negative,

        "train_can_binary_classify":
            (
                train_positive > 0
                and
                train_negative > 0
            ),

        # ----------------------------------------------------
        # Test region items
        # ----------------------------------------------------

        "test_total_region_items":
            len(
                test_items
            ),

        "test_evaluable_items":
            len(
                test_evaluable
            ),

        "test_positive":
            test_positive,

        "test_negative":
            test_negative,

        # ----------------------------------------------------
        # 特別看 3:1 hard cases
        # ----------------------------------------------------

        "test_three_one_items":
            len(
                test_three_one
            ),

        "test_three_one_positive":
            three_one_positive,

        "test_three_one_negative":
            three_one_negative,

        # ----------------------------------------------------
        # 2:2
        # ----------------------------------------------------

        "test_two_two_items":
            len(
                test_two_two
            ),
    })


summary_df = pd.DataFrame(
    summary_records
)


summary_output = (
    OUTPUT_DIR
    /
    "unanimous_train_test_feasibility_by_color.csv"
)


summary_df.to_csv(
    summary_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 11. 另外輸出 Train / Test image IDs
# ============================================================

split_records = []


for _, row in image_color_df.iterrows():

    if row[
        "unanimous_image"
    ]:

        dataset_group = (
            "unanimous_train"
        )

    else:

        dataset_group = (
            "disagreement_test"
        )


    split_records.append({

        "image_id":
            row[
                "image_id"
            ],

        "color":
            row[
                "color"
            ],

        "dataset_group":
            dataset_group,

        "unanimous_regions":
            row[
                "unanimous_regions"
            ],

        "three_one_regions":
            row[
                "three_one_regions"
            ],

        "two_two_regions":
            row[
                "two_two_regions"
            ],
    })


split_df = pd.DataFrame(
    split_records
)


split_output = (
    OUTPUT_DIR
    /
    "unanimous_train_test_image_split_by_color.csv"
)


split_df.to_csv(
    split_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 12. Terminal 顯示
# ============================================================

print("\n")
print("=" * 90)
print("各舌質色 Unanimous Train / Disagreement Test 可行性")
print("=" * 90)


display_columns = [

    "color",

    "unanimous_train_images",

    "disagreement_test_images",

    "train_region_items",

    "train_positive",

    "train_negative",

    "train_can_binary_classify",

    "test_evaluable_items",

    "test_positive",

    "test_negative",

    "test_three_one_items",

    "test_two_two_items",
]


print(
    summary_df[
        display_columns
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 13. 顯示各色訓練是否可行
# ============================================================

print("\n")
print("=" * 90)
print("初步判斷")
print("=" * 90)


for _, row in summary_df.iterrows():

    color = (
        row[
            "color"
        ]
    )


    train_images = int(
        row[
            "unanimous_train_images"
        ]
    )


    positive = int(
        row[
            "train_positive"
        ]
    )


    negative = int(
        row[
            "train_negative"
        ]
    )


    if not row[
        "train_can_binary_classify"
    ]:

        status = (
            "❌ 無法訓練二元分類器"
        )

    elif positive < 10:

        status = (
            "⚠ 陽性極少"
        )

    elif positive < 20:

        status = (
            "⚠ 陽性偏少"
        )

    elif train_images < 20:

        status = (
            "⚠ 訓練影像偏少"
        )

    else:

        status = (
            "✓ 可進一步建立模型"
        )


    print(
        f"{color}: "
        f"train images={train_images}, "
        f"positive={positive}, "
        f"negative={negative} "
        f"→ {status}"
    )


# ============================================================
# 14. 輸出
# ============================================================

print("\n")
print("=" * 90)
print("輸出檔案")
print("=" * 90)


print(
    image_color_output
)

print(
    summary_output
)

print(
    split_output
)

print("=" * 90)