from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. 路徑
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"


METHODS = [
    "tip_middle_root_1_2_2",
    "tip_middle_root_1_3_1",
]


COLORS = [
    "淡紅",
    "淡白",
    "鮮紅",
    "暗紅",
    "青紫",
    "灰黑",
]


# ============================================================
# 2. 檔案路徑
# ============================================================

VOTE_FILE = (
    OUTPUT_DIR
    / "interrater_vote_details.csv"
)

FULL_AGREEMENT_FILE = (
    OUTPUT_DIR
    / "interrater_agreement_by_color.csv"
)

SVM_RF_FILE = (
    OUTPUT_DIR
    / "svm_rf_all_methods_color_metrics.csv"
)

CNN_FILE = (
    OUTPUT_DIR
    / "cnn_all_methods_color_metrics.csv"
)


TARGET_FILES = {
    method:
        OUTPUT_DIR
        / method
        / "ml_target_summary.csv"

    for method in METHODS
}


ML_DATASET_FILES = {
    method:
        OUTPUT_DIR
        / method
        / "ml_dataset.csv"

    for method in METHODS
}


# ============================================================
# 3. 檢查檔案
# ============================================================

required_files = [
    VOTE_FILE,
    FULL_AGREEMENT_FILE,
    SVM_RF_FILE,
    CNN_FILE,
    *TARGET_FILES.values(),
    *ML_DATASET_FILES.values(),
]


for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"找不到：{file_path}"
        )


# ============================================================
# 4. image_id 標準化
# ============================================================

def normalize_image_id(series):

    return (
        series
        .astype(str)
        .str.strip()
        .str.replace(
            r"\.0$",
            "",
            regex=True
        )
    )


# ============================================================
# 5. Fleiss' Kappa
# ============================================================

def fleiss_kappa_binary(ratings):

    ratings = np.asarray(
        ratings,
        dtype=int
    )


    if (
        ratings.ndim != 2
        or
        ratings.shape[0] == 0
    ):

        return np.nan


    n_items = ratings.shape[0]
    n_raters = ratings.shape[1]


    n_positive = (
        ratings.sum(
            axis=1
        )
    )


    n_negative = (
        n_raters
        -
        n_positive
    )


    # --------------------------------------------------------
    # 每一個 item 的 observed agreement
    # --------------------------------------------------------

    p_item = (

        (
            n_positive
            *
            (
                n_positive
                -
                1
            )
        )

        +

        (
            n_negative
            *
            (
                n_negative
                -
                1
            )
        )

    ) / (

        n_raters
        *
        (
            n_raters
            -
            1
        )

    )


    observed = (
        p_item.mean()
    )


    # --------------------------------------------------------
    # 全體類別比例
    # --------------------------------------------------------

    total_ratings = (
        n_items
        *
        n_raters
    )


    p_positive = (
        n_positive.sum()
        /
        total_ratings
    )


    p_negative = (
        n_negative.sum()
        /
        total_ratings
    )


    expected = (
        p_positive ** 2
        +
        p_negative ** 2
    )


    denominator = (
        1
        -
        expected
    )


    if np.isclose(
        denominator,
        0
    ):

        return np.nan


    kappa = (

        observed
        -
        expected

    ) / denominator


    return float(kappa)


# ============================================================
# 6. Agreement 統計
# ============================================================

def calculate_agreement(ratings):

    ratings = np.asarray(
        ratings,
        dtype=int
    )


    votes = (
        ratings.sum(
            axis=1
        )
    )


    n_items = len(votes)


    unanimous = int(
        np.sum(
            (votes == 0)
            |
            (votes == 4)
        )
    )


    three_one = int(
        np.sum(
            (votes == 1)
            |
            (votes == 3)
        )
    )


    two_two = int(
        np.sum(
            votes == 2
        )
    )


    return {

        "n_items":
            n_items,

        "unanimous_n":
            unanimous,

        "unanimous_rate":
            unanimous / n_items
            if n_items
            else np.nan,

        "three_one_n":
            three_one,

        "three_one_rate":
            three_one / n_items
            if n_items
            else np.nan,

        "two_two_n":
            two_two,

        "two_two_rate":
            two_two / n_items
            if n_items
            else np.nan,

        "fleiss_kappa":
            fleiss_kappa_binary(
                ratings
            ),
    }


# ============================================================
# 7. 找出真正進入 AI 的病例
# ============================================================

model_ids_by_method = {}


for method in METHODS:

    ml_df = pd.read_csv(
        ML_DATASET_FILES[
            method
        ],
        dtype={
            "image_id": str
        }
    )


    ml_df["image_id"] = (
        normalize_image_id(
            ml_df[
                "image_id"
            ]
        )
    )


    model_ids_by_method[
        method
    ] = set(
        ml_df[
            "image_id"
        ].unique()
    )


# ============================================================
# 8. 確認兩種分區使用完全相同病例
# ============================================================

reference_ids = (
    model_ids_by_method[
        METHODS[0]
    ]
)


for method in METHODS[1:]:

    if (
        model_ids_by_method[
            method
        ]
        !=
        reference_ids
    ):

        raise ValueError(
            "兩種分區方法使用的 image_id 不一致，"
            "不可直接進行公平比較。"
        )


MODEL_IMAGE_IDS = (
    reference_ids
)


print("=" * 80)
print("醫師一致性 × AI 效能整合分析")
print("=" * 80)

print(
    "目前 AI 使用病例數：",
    len(
        MODEL_IMAGE_IDS
    )
)

print(
    "✓ 1:2:2 與 1:3:1 使用相同病例"
)


# ============================================================
# 9. 讀取完整 100 張醫師投票
# ============================================================

vote_df = pd.read_csv(
    VOTE_FILE,
    dtype={
        "image_id": str
    }
)


vote_df["image_id"] = (
    normalize_image_id(
        vote_df[
            "image_id"
        ]
    )
)


print(
    "完整醫師評分病例數：",
    vote_df[
        "image_id"
    ].nunique()
)


# ============================================================
# 10. 只取「AI 實際使用病例」
#
# correlation 必須使用同一批病例，
# 不直接拿完整 100 張 agreement
# 與目前部分病例 AI result 比。
# ============================================================

model_vote_df = (
    vote_df[
        vote_df[
            "image_id"
        ].isin(
            MODEL_IMAGE_IDS
        )
    ]
    .copy()
)


print(
    "AI 子集醫師評分 items：",
    len(
        model_vote_df
    )
)


expected_items = (
    len(
        MODEL_IMAGE_IDS
    )
    *
    5
    *
    6
)


print(
    "理論應有 items：",
    expected_items
)


if (
    len(
        model_vote_df
    )
    !=
    expected_items
):

    print(
        "⚠ 警告：AI 子集的醫師評分 item "
        "數與理論值不一致"
    )

else:

    print(
        "✓ AI 子集醫師評分完整"
    )


# ============================================================
# 11. 重新計算 AI 子集的「各顏色醫師一致性」
# ============================================================

subset_agreement_records = []


for color in COLORS:

    subset = (
        model_vote_df[
            model_vote_df[
                "color"
            ]
            ==
            color
        ]
        .copy()
    )


    ratings = (
        subset[
            [
                "D1",
                "D2",
                "D3",
                "D4"
            ]
        ]
        .to_numpy()
    )


    result = (
        calculate_agreement(
            ratings
        )
    )


    subset_agreement_records.append({

        "color":
            color,

        "subset_n_items":
            result[
                "n_items"
            ],

        "subset_unanimous_n":
            result[
                "unanimous_n"
            ],

        "subset_unanimous_rate":
            result[
                "unanimous_rate"
            ],

        "subset_three_one_n":
            result[
                "three_one_n"
            ],

        "subset_three_one_rate":
            result[
                "three_one_rate"
            ],

        "subset_two_two_n":
            result[
                "two_two_n"
            ],

        "subset_two_two_rate":
            result[
                "two_two_rate"
            ],

        "subset_fleiss_kappa":
            result[
                "fleiss_kappa"
            ],
    })


subset_agreement_df = pd.DataFrame(
    subset_agreement_records
)


subset_agreement_output = (
    OUTPUT_DIR
    /
    "interrater_agreement_model_subset_by_color.csv"
)


subset_agreement_df.to_csv(
    subset_agreement_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 12. 讀取完整 100 張 Agreement
#
# 這份只作描述用，
# 不直接拿來與目前 AI 做 correlation。
# ============================================================

full_agreement = pd.read_csv(
    FULL_AGREEMENT_FILE
)


full_agreement = (
    full_agreement[
        [
            "color",
            "n_items",
            "unanimous_rate",
            "three_one_rate",
            "two_two_rate",
            "fleiss_kappa",
        ]
    ]
    .rename(
        columns={
            "n_items":
                "full_n_items",

            "unanimous_rate":
                "full_unanimous_rate",

            "three_one_rate":
                "full_three_one_rate",

            "two_two_rate":
                "full_two_two_rate",

            "fleiss_kappa":
                "full_fleiss_kappa",
        }
    )
)


# ============================================================
# 13. 讀取兩種 segmentation 的 Ground Truth 統計
# ============================================================

target_tables = {}


for method in METHODS:

    target_df = pd.read_csv(
        TARGET_FILES[
            method
        ]
    )


    target_df = (
        target_df[
            [
                "color",
                "positive",
                "negative",
                "uncertain",
                "usable",
            ]
        ]
        .sort_values(
            "color"
        )
        .reset_index(
            drop=True
        )
    )


    target_tables[
        method
    ] = target_df


# ============================================================
# 14. 確認兩個 segmentation 的 Ground Truth 數量相同
# ============================================================

target_a = (
    target_tables[
        METHODS[0]
    ]
)


target_b = (
    target_tables[
        METHODS[1]
    ]
)


if not target_a.equals(
    target_b
):

    raise ValueError(
        "1:2:2 與 1:3:1 的 Ground Truth "
        "數量不一致。"
    )


print(
    "✓ 兩種分區使用相同 Ground Truth"
)


target_summary = (
    target_a.copy()
)


target_summary[
    "positive_rate"
] = (

    target_summary[
        "positive"
    ]

    /

    target_summary[
        "usable"
    ]
)


target_summary[
    "uncertain_rate"
] = (

    target_summary[
        "uncertain"
    ]

    /

    (
        target_summary[
            "usable"
        ]

        +

        target_summary[
            "uncertain"
        ]
    )
)


# ============================================================
# 15. 加入資料可靠性標記
# ============================================================

def sample_status(
    positive
):

    if positive == 0:

        return (
            "not_evaluable_no_positive"
        )

    elif positive < 10:

        return (
            "extremely_sparse"
        )

    elif positive < 20:

        return (
            "sparse"
        )

    else:

        return (
            "evaluable"
        )


target_summary[
    "data_status"
] = (

    target_summary[
        "positive"
    ]
    .apply(
        sample_status
    )
)


# ============================================================
# 16. 讀取 SVM / Random Forest
# ============================================================

svm_rf = pd.read_csv(
    SVM_RF_FILE
)


svm_rf[
    "cnn_architecture"
] = pd.NA


# ============================================================
# 17. 讀取 CNN
# ============================================================

cnn = pd.read_csv(
    CNN_FILE
)


# ============================================================
# 18. 統一模型欄位
# ============================================================

performance_columns = [
    "segmentation_method",
    "model",
    "cnn_architecture",
    "color",
    "n_samples",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "balanced_accuracy",
    "TN",
    "FP",
    "FN",
    "TP",
]


svm_rf_standard = (
    svm_rf[
        performance_columns
    ]
    .copy()
)


cnn_standard = (
    cnn[
        performance_columns
    ]
    .copy()
)


all_performance = pd.concat(
    [
        svm_rf_standard,
        cnn_standard
    ],
    ignore_index=True
)


# ============================================================
# 19. 儲存所有「顏色 × 模型 × 分區」結果
# ============================================================

all_performance_output = (
    OUTPUT_DIR
    /
    "all_models_color_performance.csv"
)


all_performance.to_csv(
    all_performance_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 20. 每種顏色找 Best Model
#
# 第一排序：F1
# 第二排序：Balanced Accuracy
# ============================================================

best_records = []


for color in COLORS:

    subset = (
        all_performance[
            all_performance[
                "color"
            ]
            ==
            color
        ]
        .copy()
    )


    if len(
        subset
    ) == 0:

        best_records.append({

            "color":
                color,

            "best_segmentation_method":
                pd.NA,

            "best_model":
                pd.NA,

            "best_accuracy":
                np.nan,

            "best_precision":
                np.nan,

            "best_recall":
                np.nan,

            "best_f1":
                np.nan,

            "best_balanced_accuracy":
                np.nan,
        })

        continue


    subset = (
        subset
        .sort_values(
            [
                "f1",
                "balanced_accuracy"
            ],
            ascending=[
                False,
                False
            ]
        )
        .reset_index(
            drop=True
        )
    )


    best = (
        subset.iloc[0]
    )


    best_records.append({

        "color":
            color,

        "best_segmentation_method":
            best[
                "segmentation_method"
            ],

        "best_model":
            best[
                "model"
            ],

        "best_accuracy":
            best[
                "accuracy"
            ],

        "best_precision":
            best[
                "precision"
            ],

        "best_recall":
            best[
                "recall"
            ],

        "best_f1":
            best[
                "f1"
            ],

        "best_balanced_accuracy":
            best[
                "balanced_accuracy"
            ],
    })


best_df = pd.DataFrame(
    best_records
)


# ============================================================
# 21. 建立整合研究總表
# ============================================================

integrated = (
    target_summary

    .merge(
        full_agreement,
        on="color",
        how="left"
    )

    .merge(
        subset_agreement_df,
        on="color",
        how="left"
    )

    .merge(
        best_df,
        on="color",
        how="left"
    )
)


# ============================================================
# 22. 固定顏色順序
# ============================================================

integrated[
    "color"
] = pd.Categorical(
    integrated[
        "color"
    ],
    categories=COLORS,
    ordered=True
)


integrated = (
    integrated
    .sort_values(
        "color"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 23. 儲存整合表
# ============================================================

integrated_output = (
    OUTPUT_DIR
    /
    "integrated_color_analysis.csv"
)


integrated.to_csv(
    integrated_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 24. Pearson / Spearman correlation
#
# 注意：
# 只有可評估顏色才計算。
# 目前顏色數很少，因此只作 exploratory analysis。
# ============================================================

def calculate_correlations(
    dataframe,
    x_column,
    y_column
):

    usable = (
        dataframe[
            [
                x_column,
                y_column
            ]
        ]
        .dropna()
        .copy()
    )


    n = len(
        usable
    )


    if n < 3:

        return {
            "n_colors":
                n,

            "pearson_r":
                np.nan,

            "spearman_rho":
                np.nan,
        }


    pearson = (
        usable[
            x_column
        ]
        .corr(
            usable[
                y_column
            ]
        )
    )


    x_rank = (
        usable[
            x_column
        ]
        .rank(
            method="average"
        )
    )


    y_rank = (
        usable[
            y_column
        ]
        .rank(
            method="average"
        )
    )


    spearman = (
        x_rank.corr(
            y_rank
        )
    )


    return {

        "n_colors":
            n,

        "pearson_r":
            pearson,

        "spearman_rho":
            spearman,
    }


# ============================================================
# 25. 要探索的變項
# ============================================================

predictors = [

    "positive",

    "positive_rate",

    "subset_unanimous_rate",

    "subset_three_one_rate",

    "subset_two_two_rate",

    "subset_fleiss_kappa",
]


outcomes = [

    "best_f1",

    "best_balanced_accuracy",
]


correlation_records = []


for predictor in predictors:

    for outcome in outcomes:

        result = (
            calculate_correlations(
                integrated,
                predictor,
                outcome
            )
        )


        correlation_records.append({

            "predictor":
                predictor,

            "outcome":
                outcome,

            "n_colors":
                result[
                    "n_colors"
                ],

            "pearson_r":
                result[
                    "pearson_r"
                ],

            "spearman_rho":
                result[
                    "spearman_rho"
                ],

            "interpretation":
                "exploratory_only_small_n",
        })


correlation_df = pd.DataFrame(
    correlation_records
)


correlation_output = (
    OUTPUT_DIR
    /
    "exploratory_color_correlations.csv"
)


correlation_df.to_csv(
    correlation_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 26. Terminal 顯示整合結果
# ============================================================

print("\n")
print("=" * 80)
print("各舌質色整合結果")
print("=" * 80)


display_columns = [

    "color",

    "positive",

    "negative",

    "uncertain",

    "positive_rate",

    "subset_unanimous_rate",

    "subset_two_two_rate",

    "subset_fleiss_kappa",

    "best_segmentation_method",

    "best_model",

    "best_f1",

    "best_balanced_accuracy",

    "data_status",
]


display_df = (
    integrated[
        display_columns
    ]
    .copy()
)


round_columns = [

    "positive_rate",

    "subset_unanimous_rate",

    "subset_two_two_rate",

    "subset_fleiss_kappa",

    "best_f1",

    "best_balanced_accuracy",
]


for column in round_columns:

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


# ============================================================
# 27. 顯示 Exploratory Correlations
# ============================================================

print("\n")
print("=" * 80)
print("探索性相關分析")
print("=" * 80)

print(
    "注意：顏色數很少，"
    "此分析只能作 exploratory observation，"
    "不可視為正式統計推論。"
)


correlation_display = (
    correlation_df.copy()
)


correlation_display[
    "pearson_r"
] = (
    correlation_display[
        "pearson_r"
    ]
    .round(
        4
    )
)


correlation_display[
    "spearman_rho"
] = (
    correlation_display[
        "spearman_rho"
    ]
    .round(
        4
    )
)


print(
    correlation_display[
        [
            "predictor",
            "outcome",
            "n_colors",
            "pearson_r",
            "spearman_rho",
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 28. 最終輸出
# ============================================================

print("\n")
print("=" * 80)
print("分析完成")
print("=" * 80)


print(
    "AI 子集醫師一致性："
)

print(
    subset_agreement_output
)


print(
    "\n全部模型舌色結果："
)

print(
    all_performance_output
)


print(
    "\n整合研究總表："
)

print(
    integrated_output
)


print(
    "\n探索性相關："
)

print(
    correlation_output
)


print("=" * 80)
