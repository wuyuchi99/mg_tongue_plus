from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold


# ============================================================
# 1. 基本設定
# ============================================================

SEED = 42
MAX_FOLDS = 5


# 正式主要分析
PRIMARY_COLORS = [
    "淡紅",
    "鮮紅",
]


# 探索性分析
EXPLORATORY_COLORS = [
    "暗紅",
    "青紫",
]


ANALYSIS_COLORS = (
    PRIMARY_COLORS
    +
    EXPLORATORY_COLORS
)


# ============================================================
# 2. 路徑
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

VOTE_FILE = (
    OUTPUT_DIR
    / "interrater_vote_details.csv"
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
# 4. positive_votes
# ============================================================

if "positive_votes" not in df.columns:

    doctor_columns = [
        "D1",
        "D2",
        "D3",
        "D4",
    ]

    df["positive_votes"] = (
        df[
            doctor_columns
        ]
        .sum(
            axis=1
        )
    )


# ============================================================
# 5. Agreement 類型
# ============================================================

def get_agreement_type(
    positive_votes
):

    if positive_votes == 4:

        return "unanimous_positive"

    elif positive_votes == 0:

        return "unanimous_negative"

    elif positive_votes == 3:

        return "three_one_positive"

    elif positive_votes == 1:

        return "three_one_negative"

    elif positive_votes == 2:

        return "two_two"

    else:

        return "unknown"


df["agreement_detail"] = (
    df["positive_votes"]
    .apply(
        get_agreement_type
    )
)


# ============================================================
# 6. Majority Ground Truth
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
# 7. 建立 Fold Assignment
#
# IMPORTANT：
#
# fold 是 image_id / patient level。
#
# 同一 image_id 的所有 region
# 一定在同一 fold。
#
# ============================================================

assignment_records = []

fold_check_records = []


print("=" * 100)
print("Unanimous Training → 3:1 Disagreement Testing")
print("建立共同 Patient-level Folds")
print("=" * 100)


for color in ANALYSIS_COLORS:

    print("\n")
    print("-" * 100)
    print(f"處理舌質色：{color}")
    print("-" * 100)


    color_df = (
        df[
            df["color"]
            ==
            color
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # 排除 2:2
    #
    # fold stratification 使用所有具有 majority label
    # 的 items：
    #
    # unanimous + 3:1
    #
    # 但真正訓練時只使用 unanimous
    # 真正測試時只使用 3:1
    # ========================================================

    fold_pool = (
        color_df[
            color_df[
                "majority_label"
            ]
            !=
            -1
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # 8. 確認病人數
    # ========================================================

    n_patients = (
        fold_pool[
            "image_id"
        ]
        .nunique()
    )


    print(
        "可用病人數：",
        n_patients
    )


    # ========================================================
    # 9. StratifiedGroupKFold
    # ========================================================

    sgkf = StratifiedGroupKFold(
        n_splits=MAX_FOLDS,
        shuffle=True,
        random_state=SEED
    )


    X_dummy = np.zeros(
        (
            len(
                fold_pool
            ),
            1
        )
    )


    y = (
        fold_pool[
            "majority_label"
        ]
        .to_numpy()
    )


    groups = (
        fold_pool[
            "image_id"
        ]
        .to_numpy()
    )


    # image_id -> fold
    image_fold_map = {}


    for fold_id, (
        train_index,
        test_index
    ) in enumerate(
        sgkf.split(
            X_dummy,
            y,
            groups
        ),
        start=1
    ):

        test_image_ids = set(
            fold_pool.iloc[
                test_index
            ][
                "image_id"
            ]
            .unique()
        )


        for image_id in test_image_ids:

            if image_id in image_fold_map:

                raise ValueError(
                    f"{color}: "
                    f"image_id={image_id} "
                    "被分配到多個 fold"
                )


            image_fold_map[
                image_id
            ] = fold_id


    # ========================================================
    # 10. 儲存 image-level fold assignment
    # ========================================================

    for image_id, fold_id in (
        image_fold_map.items()
    ):

        assignment_records.append({

            "color":
                color,

            "image_id":
                image_id,

            "fold":
                fold_id,

            "analysis_type":
                (
                    "primary"
                    if color
                    in PRIMARY_COLORS
                    else
                    "exploratory"
                ),
        })


    # ========================================================
    # 11. Fold-by-fold 檢查
    #
    # TRAIN：
    #   只取 training patients 的
    #   unanimous 4:0 / 0:4
    #
    # TEST：
    #   只取 held-out patients 的
    #   3:1 / 1:3
    # ========================================================

    for fold_id in range(
        1,
        MAX_FOLDS + 1
    ):

        test_ids = {

            image_id

            for image_id, fold
            in image_fold_map.items()

            if fold == fold_id
        }


        train_ids = (

            set(
                image_fold_map.keys()
            )
            -
            test_ids
        )


        # ----------------------------------------------------
        # TRAIN unanimous only
        # ----------------------------------------------------

        train_df = (
            color_df[
                (
                    color_df[
                        "image_id"
                    ]
                    .isin(
                        train_ids
                    )
                )
                &
                (
                    color_df[
                        "agreement_detail"
                    ]
                    .isin(
                        [
                            "unanimous_positive",
                            "unanimous_negative",
                        ]
                    )
                )
            ]
            .copy()
        )


        # ----------------------------------------------------
        # TEST 3:1 only
        # ----------------------------------------------------

        test_df = (
            color_df[
                (
                    color_df[
                        "image_id"
                    ]
                    .isin(
                        test_ids
                    )
                )
                &
                (
                    color_df[
                        "agreement_detail"
                    ]
                    .isin(
                        [
                            "three_one_positive",
                            "three_one_negative",
                        ]
                    )
                )
            ]
            .copy()
        )


        train_positive = int(
            np.sum(
                train_df[
                    "majority_label"
                ]
                ==
                1
            )
        )


        train_negative = int(
            np.sum(
                train_df[
                    "majority_label"
                ]
                ==
                0
            )
        )


        test_positive = int(
            np.sum(
                test_df[
                    "majority_label"
                ]
                ==
                1
            )
        )


        test_negative = int(
            np.sum(
                test_df[
                    "majority_label"
                ]
                ==
                0
            )
        )


        train_valid = (
            train_positive > 0
            and
            train_negative > 0
        )


        test_valid = (
            test_positive > 0
            and
            test_negative > 0
        )


        fold_valid = (
            train_valid
            and
            test_valid
        )


        fold_check_records.append({

            "color":
                color,

            "analysis_type":
                (
                    "primary"
                    if color
                    in PRIMARY_COLORS
                    else
                    "exploratory"
                ),

            "fold":
                fold_id,

            "train_patients":
                len(
                    train_ids
                ),

            "test_patients":
                len(
                    test_ids
                ),

            "train_unanimous_items":
                len(
                    train_df
                ),

            "train_positive":
                train_positive,

            "train_negative":
                train_negative,

            "test_three_one_items":
                len(
                    test_df
                ),

            "test_positive":
                test_positive,

            "test_negative":
                test_negative,

            "train_valid":
                train_valid,

            "test_valid":
                test_valid,

            "fold_valid":
                fold_valid,
        })


# ============================================================
# 12. 建立 DataFrame
# ============================================================

assignment_df = pd.DataFrame(
    assignment_records
)


fold_check_df = pd.DataFrame(
    fold_check_records
)


# ============================================================
# 13. Leakage Check
#
# 同 color、同 image_id
# 只能出現一個 fold
# ============================================================

leakage_check = (
    assignment_df
    .groupby(
        [
            "color",
            "image_id",
        ]
    )[
        "fold"
    ]
    .nunique()
)


if (
    leakage_check
    >
    1
).any():

    raise ValueError(
        "發現 patient-level fold leakage"
    )


# ============================================================
# 14. 輸出 Fold Assignment
# ============================================================

ASSIGNMENT_OUTPUT = (
    OUTPUT_DIR
    /
    "unanimous_disagreement_fold_assignments.csv"
)


assignment_df.to_csv(
    ASSIGNMENT_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 15. 輸出 Fold Check
# ============================================================

CHECK_OUTPUT = (
    OUTPUT_DIR
    /
    "unanimous_disagreement_fold_check.csv"
)


fold_check_df.to_csv(
    CHECK_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 16. Terminal 顯示
# ============================================================

print("\n")
print("=" * 100)
print("Fold-by-Fold Feasibility")
print("=" * 100)


display_columns = [

    "color",

    "analysis_type",

    "fold",

    "train_patients",

    "test_patients",

    "train_unanimous_items",

    "train_positive",

    "train_negative",

    "test_three_one_items",

    "test_positive",

    "test_negative",

    "fold_valid",
]


print(
    fold_check_df[
        display_columns
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 17. Color-level Summary
# ============================================================

print("\n")
print("=" * 100)
print("各舌質色 Fold 可行性")
print("=" * 100)


summary_records = []


for color in ANALYSIS_COLORS:

    subset = (
        fold_check_df[
            fold_check_df[
                "color"
            ]
            ==
            color
        ]
    )


    valid_folds = int(
        subset[
            "fold_valid"
        ].sum()
    )


    all_valid = (
        valid_folds
        ==
        MAX_FOLDS
    )


    summary_records.append({

        "color":
            color,

        "analysis_type":
            (
                "primary"
                if color
                in PRIMARY_COLORS
                else
                "exploratory"
            ),

        "valid_folds":
            valid_folds,

        "total_folds":
            MAX_FOLDS,

        "all_folds_valid":
            all_valid,

        "minimum_train_positive":
            int(
                subset[
                    "train_positive"
                ].min()
            ),

        "minimum_test_positive":
            int(
                subset[
                    "test_positive"
                ].min()
            ),

        "minimum_test_negative":
            int(
                subset[
                    "test_negative"
                ].min()
            ),
    })


summary_df = pd.DataFrame(
    summary_records
)


print(
    summary_df.to_string(
        index=False
    )
)


SUMMARY_OUTPUT = (
    OUTPUT_DIR
    /
    "unanimous_disagreement_fold_summary.csv"
)


summary_df.to_csv(
    SUMMARY_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 18. 最終判斷
# ============================================================

print("\n")
print("=" * 100)
print("最終判斷")
print("=" * 100)


for _, row in summary_df.iterrows():

    color = row[
        "color"
    ]


    if row[
        "all_folds_valid"
    ]:

        if row[
            "analysis_type"
        ] == "primary":

            print(
                f"✓ {color}: "
                "5 folds 全部可用，"
                "可進入正式三模型比較。"
            )

        else:

            print(
                f"△ {color}: "
                "5 folds 可執行，"
                "但 unanimous positive 很少，"
                "僅作探索性分析。"
            )

    else:

        print(
            f"❌ {color}: "
            f"只有 "
            f"{row['valid_folds']}/"
            f"{MAX_FOLDS} folds 可用，"
            "暫不進入三模型正式比較。"
        )


print("\n輸出檔案：")

print(
    ASSIGNMENT_OUTPUT
)

print(
    CHECK_OUTPUT
)

print(
    SUMMARY_OUTPUT
)

print("=" * 100)
