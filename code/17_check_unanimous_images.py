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

EXPECTED_ITEMS_PER_IMAGE = (
    len(REGIONS)
    *
    len(COLORS)
)


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
# 4. 確認醫師欄位
# ============================================================

doctor_columns = [
    "D1",
    "D2",
    "D3",
    "D4",
]


for column in doctor_columns:

    if column not in df.columns:

        raise KeyError(
            f"找不到欄位：{column}"
        )


print("=" * 85)
print("四位醫師完全一致影像檢查")
print("=" * 85)

print(
    "影像數：",
    df["image_id"].nunique()
)

print(
    "每張理論判斷項目：",
    EXPECTED_ITEMS_PER_IMAGE
)

print(
    "總 items：",
    len(df)
)


# ============================================================
# 5. 判斷每一個 region × color 是否四醫師完全相同
# ============================================================

df["is_unanimous"] = (

    (df["D1"] == df["D2"])
    &
    (df["D1"] == df["D3"])
    &
    (df["D1"] == df["D4"])
)


df["is_three_one"] = (
    df["positive_votes"]
    .isin(
        [
            1,
            3
        ]
    )
)


df["is_two_two"] = (
    df["positive_votes"]
    ==
    2
)


# ============================================================
# 6. 每張照片統計 30 個判斷
# ============================================================

image_records = []


for image_id, group in df.groupby(
    "image_id"
):

    n_items = len(group)

    unanimous_items = int(
        group["is_unanimous"].sum()
    )

    three_one_items = int(
        group["is_three_one"].sum()
    )

    two_two_items = int(
        group["is_two_two"].sum()
    )

    unanimous_rate = (
        unanimous_items
        /
        n_items
    )


    image_records.append({

        "image_id":
            image_id,

        "n_items":
            n_items,

        "unanimous_items":
            unanimous_items,

        "unanimous_rate":
            unanimous_rate,

        "three_one_items":
            three_one_items,

        "two_two_items":
            two_two_items,

        # 四位醫師對整張照片
        # 30 個 binary labels 全部完全一致
        "strict_unanimous":
            (
                n_items
                ==
                EXPECTED_ITEMS_PER_IMAGE
                and
                unanimous_items
                ==
                EXPECTED_ITEMS_PER_IMAGE
            ),

        # 額外提供較寬鬆門檻
        "unanimous_90_percent":
            (
                unanimous_rate
                >=
                0.90
            ),

        "unanimous_80_percent":
            (
                unanimous_rate
                >=
                0.80
            ),

        "unanimous_70_percent":
            (
                unanimous_rate
                >=
                0.70
            ),
    })


image_summary = pd.DataFrame(
    image_records
)


image_summary = (
    image_summary
    .sort_values(
        [
            "unanimous_rate",
            "image_id"
        ],
        ascending=[
            False,
            True
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 7. 輸出每張照片一致性
# ============================================================

summary_output = (
    OUTPUT_DIR
    /
    "unanimous_image_summary.csv"
)


image_summary.to_csv(
    summary_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 8. 找 Strict Unanimous Image IDs
# ============================================================

strict_ids = set(

    image_summary.loc[
        image_summary[
            "strict_unanimous"
        ],
        "image_id"
    ]

)


non_strict_ids = set(

    image_summary.loc[
        ~image_summary[
            "strict_unanimous"
        ],
        "image_id"
    ]

)


strict_id_df = pd.DataFrame({

    "image_id":
        sorted(
            strict_ids
        )
})


non_strict_id_df = pd.DataFrame({

    "image_id":
        sorted(
            non_strict_ids
        )
})


strict_output = (
    OUTPUT_DIR
    /
    "strict_unanimous_image_ids.csv"
)


non_strict_output = (
    OUTPUT_DIR
    /
    "non_unanimous_image_ids.csv"
)


strict_id_df.to_csv(
    strict_output,
    index=False,
    encoding="utf-8-sig"
)


non_strict_id_df.to_csv(
    non_strict_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 9. 不同一致率門檻有多少照片
# ============================================================

threshold_records = []


thresholds = [
    1.00,
    0.90,
    0.80,
    0.70,
    0.60,
]


for threshold in thresholds:

    n_images = int(

        np.sum(
            image_summary[
                "unanimous_rate"
            ]
            >=
            threshold
        )
    )


    threshold_records.append({

        "minimum_unanimous_rate":
            threshold,

        "n_images":
            n_images,

        "percentage_of_100_images":
            n_images
            /
            len(
                image_summary
            )
            *
            100,
    })


threshold_df = pd.DataFrame(
    threshold_records
)


threshold_output = (
    OUTPUT_DIR
    /
    "unanimous_image_threshold_counts.csv"
)


threshold_df.to_csv(
    threshold_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 10. Strict Unanimous 子集的類別分布
#
# 這一步非常重要：
# 即使有不少完全一致照片，
# 也要看每一種舌色是否有 positive。
# ============================================================

strict_vote_df = (
    df[
        df[
            "image_id"
        ].isin(
            strict_ids
        )
    ]
    .copy()
)


class_records = []


for color in COLORS:

    subset = (
        strict_vote_df[
            strict_vote_df[
                "color"
            ]
            ==
            color
        ]
        .copy()
    )


    if len(subset) == 0:

        positive = 0
        negative = 0

    else:

        # strict unanimous 情況下，
        # D1 = D2 = D3 = D4，
        # 所以直接使用 D1 即可。
        positive = int(
            np.sum(
                subset[
                    "D1"
                ]
                ==
                1
            )
        )

        negative = int(
            np.sum(
                subset[
                    "D1"
                ]
                ==
                0
            )
        )


    class_records.append({

        "color":
            color,

        "strict_unanimous_images":
            len(
                strict_ids
            ),

        "positive":
            positive,

        "negative":
            negative,

        "total_region_items":
            positive
            +
            negative,

        "can_train_binary_model":
            (
                positive > 0
                and
                negative > 0
            ),
    })


class_df = pd.DataFrame(
    class_records
)


class_output = (
    OUTPUT_DIR
    /
    "strict_unanimous_class_distribution.csv"
)


class_df.to_csv(
    class_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 11. 顯示結果
# ============================================================

print("\n")
print("=" * 85)
print("影像層級一致性門檻")
print("=" * 85)

print(
    threshold_df.to_string(
        index=False
    )
)


print("\n")
print("=" * 85)
print("100% Strict Unanimous Images")
print("=" * 85)

print(
    "四位醫師對 30 個標籤全部相同的照片數：",
    len(
        strict_ids
    )
)


if len(
    strict_ids
) > 0:

    print(
        "影像編號："
    )

    print(
        sorted(
            strict_ids
        )
    )

else:

    print(
        "沒有任何一張照片達到 30/30 完全一致。"
    )


print("\n")
print("=" * 85)
print("Strict Unanimous 子集舌質色分布")
print("=" * 85)

print(
    class_df.to_string(
        index=False
    )
)


# ============================================================
# 12. 顯示一致率最高的前 20 張
# ============================================================

print("\n")
print("=" * 85)
print("一致率最高的前 20 張照片")
print("=" * 85)


display_columns = [
    "image_id",
    "unanimous_items",
    "unanimous_rate",
    "three_one_items",
    "two_two_items",
]


display_top = (
    image_summary[
        display_columns
    ]
    .head(
        20
    )
    .copy()
)


display_top[
    "unanimous_rate"
] = (
    display_top[
        "unanimous_rate"
    ]
    .round(
        4
    )
)


print(
    display_top.to_string(
        index=False
    )
)


# ============================================================
# 13. 最後提醒
# ============================================================

print("\n")
print("=" * 85)
print("下一步判斷")
print("=" * 85)


if len(
    strict_ids
) < 20:

    print(
        "⚠ Strict unanimous 影像少於 20 張。"
    )

    print(
        "不建議直接用這些影像訓練 "
        "SVM / RF / CNN。"
    )

    print(
        "下一步應考慮較寬鬆的一致率門檻，"
        "或改為只使用四醫師完全一致的"
        " region-color items。"
    )

else:

    print(
        "Strict unanimous 影像數量可進一步檢查"
        "各舌質色 positive / negative 分布。"
    )


print("\n輸出檔案：")

print(
    summary_output
)

print(
    strict_output
)

print(
    non_strict_output
)

print(
    threshold_output
)

print(
    class_output
)

print("=" * 85)