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
# 2. 顏色
# ============================================================

COLORS = [
    "淡紅",
    "淡白",
    "鮮紅",
    "暗紅",
    "青紫",
    "灰黑",
]


# ============================================================
# 3. 讀取資料
# ============================================================

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
# 4. 確認 positive_votes
# ============================================================

if "positive_votes" not in df.columns:

    df["positive_votes"] = (
        df[
            [
                "D1",
                "D2",
                "D3",
                "D4"
            ]
        ]
        .sum(
            axis=1
        )
    )


# ============================================================
# 5. Agreement 類型
# ============================================================

def get_agreement_type(
    votes
):

    if votes == 4:

        return (
            "unanimous_positive"
        )

    elif votes == 0:

        return (
            "unanimous_negative"
        )

    elif votes == 3:

        return (
            "three_one_positive"
        )

    elif votes == 1:

        return (
            "three_one_negative"
        )

    elif votes == 2:

        return (
            "two_two"
        )

    else:

        return (
            "unknown"
        )


df["agreement_detail"] = (
    df["positive_votes"]
    .apply(
        get_agreement_type
    )
)


# ============================================================
# 6. Majority label
# ============================================================

df["majority_label"] = np.select(

    [
        df["positive_votes"] >= 3,
        df["positive_votes"] <= 1,
        df["positive_votes"] == 2,
    ],

    [
        1,
        0,
        -1,
    ],

    default=-1
)


# ============================================================
# 7. 每種舌色統計
# ============================================================

records = []


for color in COLORS:

    subset = (
        df[
            df[
                "color"
            ]
            ==
            color
        ]
        .copy()
    )


    unanimous_positive = int(
        np.sum(
            subset[
                "agreement_detail"
            ]
            ==
            "unanimous_positive"
        )
    )


    unanimous_negative = int(
        np.sum(
            subset[
                "agreement_detail"
            ]
            ==
            "unanimous_negative"
        )
    )


    three_one_positive = int(
        np.sum(
            subset[
                "agreement_detail"
            ]
            ==
            "three_one_positive"
        )
    )


    three_one_negative = int(
        np.sum(
            subset[
                "agreement_detail"
            ]
            ==
            "three_one_negative"
        )
    )


    two_two = int(
        np.sum(
            subset[
                "agreement_detail"
            ]
            ==
            "two_two"
        )
    )


    unanimous_total = (
        unanimous_positive
        +
        unanimous_negative
    )


    three_one_total = (
        three_one_positive
        +
        three_one_negative
    )


    records.append({

        "color":
            color,

        # ----------------------------------------------
        # 高共識訓練候選
        # ----------------------------------------------

        "unanimous_positive":
            unanimous_positive,

        "unanimous_negative":
            unanimous_negative,

        "unanimous_total":
            unanimous_total,

        "unanimous_positive_rate":
            (
                unanimous_positive
                /
                unanimous_total
                if unanimous_total > 0
                else np.nan
            ),

        "can_train_binary":
            (
                unanimous_positive > 0
                and
                unanimous_negative > 0
            ),

        # ----------------------------------------------
        # 3:1 hard test
        # ----------------------------------------------

        "three_one_positive":
            three_one_positive,

        "three_one_negative":
            three_one_negative,

        "three_one_total":
            three_one_total,

        "three_one_positive_rate":
            (
                three_one_positive
                /
                three_one_total
                if three_one_total > 0
                else np.nan
            ),

        "can_evaluate_three_one":
            (
                three_one_positive > 0
                and
                three_one_negative > 0
            ),

        # ----------------------------------------------
        # 2:2 exploratory
        # ----------------------------------------------

        "two_two":
            two_two,
    })


summary_df = pd.DataFrame(
    records
)


# ============================================================
# 8. 輸出
# ============================================================

OUTPUT_FILE = (
    OUTPUT_DIR
    / "unanimous_item_training_feasibility.csv"
)


summary_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print("=" * 100)
print("Unanimous Region-Color Training Feasibility")
print("=" * 100)


display_df = (
    summary_df.copy()
)


for column in [
    "unanimous_positive_rate",
    "three_one_positive_rate",
]:

    display_df[
        column
    ] = (
        display_df[
            column
        ]
        .round(
            4
        )
    )


print(
    display_df.to_string(
        index=False
    )
)


print("\n")
print("=" * 100)
print("初步判斷")
print("=" * 100)


for _, row in summary_df.iterrows():

    color = (
        row[
            "color"
        ]
    )

    up = int(
        row[
            "unanimous_positive"
        ]
    )

    un = int(
        row[
            "unanimous_negative"
        ]
    )

    hp = int(
        row[
            "three_one_positive"
        ]
    )

    hn = int(
        row[
            "three_one_negative"
        ]
    )


    if up == 0:

        status = (
            "❌ 無 unanimous positive，無法訓練"
        )

    elif up < 10:

        status = (
            "⚠ unanimous positive 極少"
        )

    elif up < 20:

        status = (
            "⚠ unanimous positive 偏少"
        )

    elif (
        hp == 0
        or
        hn == 0
    ):

        status = (
            "⚠ 可訓練，但 3:1 測試集缺少一類"
        )

    else:

        status = (
            "✓ 適合進行高共識 → disagreement 實驗"
        )


    print(
        f"{color}: "
        f"train positive={up}, "
        f"train negative={un}, "
        f"3:1 positive={hp}, "
        f"3:1 negative={hn} "
        f"→ {status}"
    )


print("\n輸出：")
print(
    OUTPUT_FILE
)

print("=" * 100)
