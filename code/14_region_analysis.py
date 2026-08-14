from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)


# ============================================================
# 1. 路徑設定
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"


SVM_RF_FILE = (
    OUTPUT_DIR
    / "svm_rf_all_methods_predictions.csv"
)

CNN_FILE = (
    OUTPUT_DIR
    / "cnn_all_methods_predictions.csv"
)

VOTE_FILE = (
    OUTPUT_DIR
    / "interrater_vote_details.csv"
)

ML_DATASET_FILE = (
    OUTPUT_DIR
    / "tip_middle_root_1_2_2"
    / "ml_dataset.csv"
)


# ============================================================
# 2. 分區方法
# ============================================================

SEGMENTATION_METHODS = [
    "tip_middle_root_1_2_2",
    "tip_middle_root_1_3_1",
]


# ============================================================
# 3. 三個模型
# ============================================================

MODELS = [
    "SVM",
    "Random Forest",
    "CNN",
]


# ============================================================
# 4. 五個舌區
# ============================================================

REGIONS = [
    "舌尖",
    "舌中",
    "舌左邊",
    "舌右邊",
    "舌根",
]


# ============================================================
# 5. 六種舌質色
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
# 6. 檢查檔案
# ============================================================

required_files = [
    SVM_RF_FILE,
    CNN_FILE,
    VOTE_FILE,
    ML_DATASET_FILE,
]


for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"找不到：{file_path}"
        )


# ============================================================
# 7. image_id 標準化
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
# 8. 讀取 Prediction
# ============================================================

svm_rf = pd.read_csv(
    SVM_RF_FILE,
    dtype={
        "image_id": str
    }
)


cnn = pd.read_csv(
    CNN_FILE,
    dtype={
        "image_id": str
    }
)


required_prediction_columns = [
    "segmentation_method",
    "model",
    "color",
    "fold",
    "image_id",
    "region",
    "true",
    "pred",
]


for name, df in [
    ("SVM/RF", svm_rf),
    ("CNN", cnn),
]:

    missing_columns = [
        column
        for column in required_prediction_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"{name} 缺少欄位："
            f"{missing_columns}"
        )


svm_rf = (
    svm_rf[
        required_prediction_columns
    ]
    .copy()
)


cnn = (
    cnn[
        required_prediction_columns
    ]
    .copy()
)


predictions = pd.concat(
    [
        svm_rf,
        cnn
    ],
    ignore_index=True
)


# ============================================================
# 9. 清理欄位
# ============================================================

predictions["image_id"] = (
    normalize_image_id(
        predictions[
            "image_id"
        ]
    )
)


for column in [
    "region",
    "color",
    "model",
    "segmentation_method",
]:

    predictions[column] = (
        predictions[column]
        .astype(str)
        .str.strip()
    )


predictions["true"] = (
    predictions["true"]
    .astype(int)
)


predictions["pred"] = (
    predictions["pred"]
    .astype(int)
)


# ============================================================
# 10. 檢查 Prediction 是否重複
#
# 每一種：
# 分區 × 模型 × 顏色 × 病人 × 區域
#
# 應該只有一筆 out-of-fold prediction
# ============================================================

duplicate_columns = [
    "segmentation_method",
    "model",
    "color",
    "image_id",
    "region",
]


duplicates = (
    predictions
    .duplicated(
        subset=duplicate_columns,
        keep=False
    )
)


if duplicates.any():

    duplicate_rows = (
        predictions[
            duplicates
        ][
            duplicate_columns
        ]
    )

    print(
        duplicate_rows.head(
            20
        )
    )

    raise ValueError(
        "Prediction 中發現重複樣本，"
        "請先檢查前面的模型輸出。"
    )


print("=" * 85)
print("五舌區 AI 效能 × 醫師一致性分析")
print("=" * 85)

print(
    "Prediction rows：",
    len(
        predictions
    )
)

print(
    "模型：",
    sorted(
        predictions[
            "model"
        ]
        .unique()
        .tolist()
    )
)

print(
    "區域：",
    sorted(
        predictions[
            "region"
        ]
        .unique()
        .tolist()
    )
)


# ============================================================
# 11. Metrics
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    y_true = np.asarray(
        y_true,
        dtype=int
    )

    y_pred = np.asarray(
        y_pred,
        dtype=int
    )


    positive = int(
        np.sum(
            y_true == 1
        )
    )


    negative = int(
        np.sum(
            y_true == 0
        )
    )


    # --------------------------------------------------------
    # 必須同時有 positive + negative
    # 才視為真正可評估 binary classifier
    # --------------------------------------------------------

    evaluable_binary = (
        positive > 0
        and
        negative > 0
    )


    accuracy = (
        accuracy_score(
            y_true,
            y_pred
        )
    )


    precision = (
        precision_score(
            y_true,
            y_pred,
            zero_division=0
        )
    )


    recall = (
        recall_score(
            y_true,
            y_pred,
            zero_division=0
        )
    )


    f1 = (
        f1_score(
            y_true,
            y_pred,
            zero_division=0
        )
    )


    if evaluable_binary:

        balanced_accuracy = (
            balanced_accuracy_score(
                y_true,
                y_pred
            )
        )

    else:

        balanced_accuracy = np.nan


    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            y_pred,
            labels=[
                0,
                1
            ]
        )
        .ravel()
    )


    return {

        "n_samples":
            len(
                y_true
            ),

        "positive":
            positive,

        "negative":
            negative,

        "positive_rate":
            positive
            /
            len(
                y_true
            )
            if len(
                y_true
            )
            else np.nan,

        "evaluable_binary":
            evaluable_binary,

        "accuracy":
            float(
                accuracy
            ),

        "precision":
            float(
                precision
            ),

        "recall":
            float(
                recall
            ),

        "f1":
            float(
                f1
            ),

        "balanced_accuracy":
            (
                float(
                    balanced_accuracy
                )
                if not np.isnan(
                    balanced_accuracy
                )
                else np.nan
            ),

        "TN":
            int(
                tn
            ),

        "FP":
            int(
                fp
            ),

        "FN":
            int(
                fn
            ),

        "TP":
            int(
                tp
            ),
    }


# ============================================================
# 12. 每個：
#
# segmentation × model × color × region
#
# 計算 AI 效能
# ============================================================

performance_records = []


for segmentation_method in SEGMENTATION_METHODS:

    for model in MODELS:

        for color in COLORS:

            for region in REGIONS:

                subset = (
                    predictions[
                        (
                            predictions[
                                "segmentation_method"
                            ]
                            ==
                            segmentation_method
                        )
                        &
                        (
                            predictions[
                                "model"
                            ]
                            ==
                            model
                        )
                        &
                        (
                            predictions[
                                "color"
                            ]
                            ==
                            color
                        )
                        &
                        (
                            predictions[
                                "region"
                            ]
                            ==
                            region
                        )
                    ]
                    .copy()
                )


                # 灰黑通常沒有 prediction
                if len(
                    subset
                ) == 0:

                    continue


                metrics = (
                    calculate_metrics(
                        subset[
                            "true"
                        ],
                        subset[
                            "pred"
                        ]
                    )
                )


                performance_records.append({

                    "segmentation_method":
                        segmentation_method,

                    "model":
                        model,

                    "color":
                        color,

                    "region":
                        region,

                    "n_patients":
                        subset[
                            "image_id"
                        ].nunique(),

                    **metrics
                })


performance_df = pd.DataFrame(
    performance_records
)


# ============================================================
# 13. 輸出 Region × Color × Model
# ============================================================

performance_output = (
    OUTPUT_DIR
    /
    "model_performance_by_region_color.csv"
)


performance_df.to_csv(
    performance_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 14. 建立模型 × 分區 × Region 的 Macro Result
#
# 只平均：
# 同時具有 positive + negative 的顏色
#
# 避免「某區完全沒有陽性」
# 卻被當成正常 F1 評估。
# ============================================================

region_summary_records = []


for segmentation_method in SEGMENTATION_METHODS:

    for model in MODELS:

        for region in REGIONS:

            subset_all = (
                performance_df[
                    (
                        performance_df[
                            "segmentation_method"
                        ]
                        ==
                        segmentation_method
                    )
                    &
                    (
                        performance_df[
                            "model"
                        ]
                        ==
                        model
                    )
                    &
                    (
                        performance_df[
                            "region"
                        ]
                        ==
                        region
                    )
                ]
                .copy()
            )


            subset = (
                subset_all[
                    subset_all[
                        "evaluable_binary"
                    ]
                    ==
                    True
                ]
                .copy()
            )


            if len(
                subset
            ) == 0:

                continue


            region_summary_records.append({

                "segmentation_method":
                    segmentation_method,

                "model":
                    model,

                "region":
                    region,

                "n_evaluable_colors":
                    len(
                        subset
                    ),

                "macro_accuracy":
                    subset[
                        "accuracy"
                    ].mean(),

                "macro_precision":
                    subset[
                        "precision"
                    ].mean(),

                "macro_recall":
                    subset[
                        "recall"
                    ].mean(),

                "macro_f1":
                    subset[
                        "f1"
                    ].mean(),

                "macro_balanced_accuracy":
                    subset[
                        "balanced_accuracy"
                    ].mean(),
            })


region_summary_df = pd.DataFrame(
    region_summary_records
)


region_summary_output = (
    OUTPUT_DIR
    /
    "model_performance_by_region.csv"
)


region_summary_df.to_csv(
    region_summary_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 15. 找出目前 AI 實際使用的 54 位病人
# ============================================================

ml_dataset = pd.read_csv(
    ML_DATASET_FILE,
    dtype={
        "image_id": str
    }
)


ml_dataset["image_id"] = (
    normalize_image_id(
        ml_dataset[
            "image_id"
        ]
    )
)


AI_IMAGE_IDS = set(
    ml_dataset[
        "image_id"
    ].unique()
)


print(
    "\nAI 使用病人數：",
    len(
        AI_IMAGE_IDS
    )
)


# ============================================================
# 16. 讀取四醫師原始 Vote
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


# 只保留 AI 相同 54 張
vote_subset = (
    vote_df[
        vote_df[
            "image_id"
        ].isin(
            AI_IMAGE_IDS
        )
    ]
    .copy()
)


expected_items = (
    len(
        AI_IMAGE_IDS
    )
    *
    5
    *
    6
)


print(
    "AI 子集醫師評分 items：",
    len(
        vote_subset
    )
)

print(
    "理論 items：",
    expected_items
)


if len(
    vote_subset
) != expected_items:

    raise ValueError(
        "醫師評分子集數量不正確"
    )


print(
    "✓ AI 與醫師資料使用相同病例"
)


# ============================================================
# 17. Fleiss' Kappa
# ============================================================

def fleiss_kappa_binary(
    ratings
):

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


    n_items = (
        ratings.shape[
            0
        ]
    )


    n_raters = (
        ratings.shape[
            1
        ]
    )


    positive_votes = (
        ratings.sum(
            axis=1
        )
    )


    negative_votes = (
        n_raters
        -
        positive_votes
    )


    item_agreement = (

        (
            positive_votes
            *
            (
                positive_votes
                -
                1
            )
        )

        +

        (
            negative_votes
            *
            (
                negative_votes
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


    observed_agreement = (
        item_agreement.mean()
    )


    total_ratings = (
        n_items
        *
        n_raters
    )


    p_positive = (
        positive_votes.sum()
        /
        total_ratings
    )


    p_negative = (
        negative_votes.sum()
        /
        total_ratings
    )


    expected_agreement = (

        p_positive ** 2

        +

        p_negative ** 2

    )


    denominator = (
        1
        -
        expected_agreement
    )


    if np.isclose(
        denominator,
        0
    ):

        return np.nan


    kappa = (

        observed_agreement
        -
        expected_agreement

    ) / denominator


    return float(
        kappa
    )


# ============================================================
# 18. 醫師一致性函數
# ============================================================

def calculate_physician_agreement(
    dataframe
):

    ratings = (
        dataframe[
            [
                "D1",
                "D2",
                "D3",
                "D4"
            ]
        ]
        .to_numpy(
            dtype=int
        )
    )


    positive_votes = (
        ratings.sum(
            axis=1
        )
    )


    n_items = len(
        ratings
    )


    unanimous_n = int(
        np.sum(
            (positive_votes == 0)
            |
            (positive_votes == 4)
        )
    )


    three_one_n = int(
        np.sum(
            (positive_votes == 1)
            |
            (positive_votes == 3)
        )
    )


    two_two_n = int(
        np.sum(
            positive_votes == 2
        )
    )


    return {

        "n_items":
            n_items,

        "unanimous_n":
            unanimous_n,

        "unanimous_rate":
            unanimous_n
            /
            n_items
            if n_items
            else np.nan,

        "three_one_n":
            three_one_n,

        "three_one_rate":
            three_one_n
            /
            n_items
            if n_items
            else np.nan,

        "two_two_n":
            two_two_n,

        "two_two_rate":
            two_two_n
            /
            n_items
            if n_items
            else np.nan,

        "fleiss_kappa":
            fleiss_kappa_binary(
                ratings
            ),
    }


# ============================================================
# 19. 每個 Region × Color 的醫師一致性
# ============================================================

agreement_records = []


for region in REGIONS:

    for color in COLORS:

        subset = (
            vote_subset[
                (
                    vote_subset[
                        "region"
                    ]
                    ==
                    region
                )
                &
                (
                    vote_subset[
                        "color"
                    ]
                    ==
                    color
                )
            ]
            .copy()
        )


        result = (
            calculate_physician_agreement(
                subset
            )
        )


        agreement_records.append({

            "region":
                region,

            "color":
                color,

            **result
        })


agreement_df = pd.DataFrame(
    agreement_records
)


agreement_output = (
    OUTPUT_DIR
    /
    "interrater_agreement_model_subset_by_region_color.csv"
)


agreement_df.to_csv(
    agreement_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 20. 五區醫師整體一致性
#
# 每區將六種顏色一起計算
# ============================================================

region_agreement_records = []


for region in REGIONS:

    subset = (
        vote_subset[
            vote_subset[
                "region"
            ]
            ==
            region
        ]
        .copy()
    )


    result = (
        calculate_physician_agreement(
            subset
        )
    )


    region_agreement_records.append({

        "region":
            region,

        **result
    })


region_agreement_df = pd.DataFrame(
    region_agreement_records
)


region_agreement_output = (
    OUTPUT_DIR
    /
    "interrater_agreement_model_subset_by_region.csv"
)


region_agreement_df.to_csv(
    region_agreement_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 21. 每個 Region × Color 找最佳 AI
# ============================================================

best_region_color_records = []


for region in REGIONS:

    for color in COLORS:

        subset = (
            performance_df[
                (
                    performance_df[
                        "region"
                    ]
                    ==
                    region
                )
                &
                (
                    performance_df[
                        "color"
                    ]
                    ==
                    color
                )
                &
                (
                    performance_df[
                        "evaluable_binary"
                    ]
                    ==
                    True
                )
            ]
            .copy()
        )


        if len(
            subset
        ) == 0:

            best_region_color_records.append({

                "region":
                    region,

                "color":
                    color,

                "best_segmentation_method":
                    pd.NA,

                "best_model":
                    pd.NA,

                "best_positive":
                    np.nan,

                "best_negative":
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
            subset.iloc[
                0
            ]
        )


        best_region_color_records.append({

            "region":
                region,

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

            "best_positive":
                best[
                    "positive"
                ],

            "best_negative":
                best[
                    "negative"
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


best_region_color_df = pd.DataFrame(
    best_region_color_records
)


# ============================================================
# 22. 合併：
#
# Region × Color
# 醫師一致性
# +
# 最佳 AI 表現
# ============================================================

integrated_df = (
    agreement_df
    .merge(
        best_region_color_df,
        on=[
            "region",
            "color"
        ],
        how="left",
        validate="one_to_one"
    )
)


integrated_output = (
    OUTPUT_DIR
    /
    "region_color_integrated_analysis.csv"
)


integrated_df.to_csv(
    integrated_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 23. 找每個 Region 最佳模型組合
#
# 依 macro F1 排名
# ============================================================

best_region_records = []


for region in REGIONS:

    subset = (
        region_summary_df[
            region_summary_df[
                "region"
            ]
            ==
            region
        ]
        .copy()
    )


    if len(
        subset
    ) == 0:

        continue


    subset = (
        subset
        .sort_values(
            [
                "macro_f1",
                "macro_balanced_accuracy"
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


    best = subset.iloc[
        0
    ]


    best_region_records.append({

        "region":
            region,

        "best_segmentation_method":
            best[
                "segmentation_method"
            ],

        "best_model":
            best[
                "model"
            ],

        "n_evaluable_colors":
            best[
                "n_evaluable_colors"
            ],

        "best_macro_f1":
            best[
                "macro_f1"
            ],

        "best_macro_balanced_accuracy":
            best[
                "macro_balanced_accuracy"
            ],
    })


best_region_df = pd.DataFrame(
    best_region_records
)


best_region_output = (
    OUTPUT_DIR
    /
    "best_model_by_region.csv"
)


best_region_df.to_csv(
    best_region_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 24. 將醫師 Region Agreement 加入最佳 AI Region
# ============================================================

region_overview = (
    region_agreement_df
    .merge(
        best_region_df,
        on="region",
        how="left",
        validate="one_to_one"
    )
)


region_overview_output = (
    OUTPUT_DIR
    /
    "region_overview_physician_ai.csv"
)


region_overview.to_csv(
    region_overview_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 25. Terminal 顯示：
# 五區最佳 AI
# ============================================================

print("\n")
print("=" * 85)
print("五舌區最佳 AI 組合")
print("=" * 85)


display_best_region = (
    best_region_df[
        [
            "region",
            "best_segmentation_method",
            "best_model",
            "n_evaluable_colors",
            "best_macro_f1",
            "best_macro_balanced_accuracy",
        ]
    ]
    .copy()
)


display_best_region[
    "best_macro_f1"
] = (
    display_best_region[
        "best_macro_f1"
    ]
    .round(
        4
    )
)


display_best_region[
    "best_macro_balanced_accuracy"
] = (
    display_best_region[
        "best_macro_balanced_accuracy"
    ]
    .round(
        4
    )
)


print(
    display_best_region.to_string(
        index=False
    )
)


# ============================================================
# 26. Terminal 顯示：
# 五區醫師一致性
# ============================================================

print("\n")
print("=" * 85)
print("五舌區醫師一致性")
print("=" * 85)


display_agreement = (
    region_agreement_df[
        [
            "region",
            "n_items",
            "unanimous_rate",
            "three_one_rate",
            "two_two_rate",
            "fleiss_kappa",
        ]
    ]
    .copy()
)


for column in [
    "unanimous_rate",
    "three_one_rate",
    "two_two_rate",
    "fleiss_kappa",
]:

    display_agreement[
        column
    ] = (
        display_agreement[
            column
        ]
        .round(
            4
        )
    )


print(
    display_agreement.to_string(
        index=False
    )
)


# ============================================================
# 27. 每區所有模型表現
# ============================================================

print("\n")
print("=" * 85)
print("五舌區 × 模型 Macro F1")
print("=" * 85)


macro_display = (
    region_summary_df[
        [
            "segmentation_method",
            "model",
            "region",
            "n_evaluable_colors",
            "macro_f1",
            "macro_balanced_accuracy",
        ]
    ]
    .copy()
)


macro_display[
    "macro_f1"
] = (
    macro_display[
        "macro_f1"
    ]
    .round(
        4
    )
)


macro_display[
    "macro_balanced_accuracy"
] = (
    macro_display[
        "macro_balanced_accuracy"
    ]
    .round(
        4
    )
)


print(
    macro_display.to_string(
        index=False
    )
)


# ============================================================
# 28. 找 AI 最容易 / 最難的 Region
#
# 使用各區的最佳 macro F1
# ============================================================

if len(
    best_region_df
) > 0:

    easiest = (
        best_region_df
        .sort_values(
            "best_macro_f1",
            ascending=False
        )
        .iloc[
            0
        ]
    )


    hardest = (
        best_region_df
        .sort_values(
            "best_macro_f1",
            ascending=True
        )
        .iloc[
            0
        ]
    )


    print("\n")
    print("=" * 85)
    print("區域辨識難度")
    print("=" * 85)


    print(
        "目前 AI 最容易辨識區域：",
        easiest[
            "region"
        ],
        " Macro F1 =",
        round(
            easiest[
                "best_macro_f1"
            ],
            4
        )
    )


    print(
        "目前 AI 最難辨識區域：",
        hardest[
            "region"
        ],
        " Macro F1 =",
        round(
            hardest[
                "best_macro_f1"
            ],
            4
        )
    )


# ============================================================
# 29. 輸出檔案
# ============================================================

print("\n")
print("=" * 85)
print("分析完成")
print("=" * 85)


print(
    "\nRegion × Color × Model："
)

print(
    performance_output
)


print(
    "\n模型五區 Macro Performance："
)

print(
    region_summary_output
)


print(
    "\nAI 子集 Region × Color 醫師一致性："
)

print(
    agreement_output
)


print(
    "\nAI 子集五區醫師一致性："
)

print(
    region_agreement_output
)


print(
    "\nRegion × Color 整合分析："
)

print(
    integrated_output
)


print(
    "\n各區最佳模型："
)

print(
    best_region_output
)


print(
    "\n五區 AI + 醫師總覽："
)

print(
    region_overview_output
)


print("=" * 85)
