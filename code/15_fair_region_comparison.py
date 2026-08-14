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
# 1. 專案路徑
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

ML_DATASET_FILE = (
    OUTPUT_DIR
    / "tip_middle_root_1_2_2"
    / "ml_dataset.csv"
)

VOTE_FILE = (
    OUTPUT_DIR
    / "interrater_vote_details.csv"
)


# ============================================================
# 2. 兩種分區
# ============================================================

SEGMENTATION_METHODS = [
    "tip_middle_root_1_2_2",
    "tip_middle_root_1_3_1",
]


# ============================================================
# 3. 三種模型
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
# 5. 六種原始舌質色
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
    ML_DATASET_FILE,
    VOTE_FILE,
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
# 8. 讀取 Ground Truth Dataset
#
# Ground Truth 與分區方法無關，
# 所以只需要使用其中一種 ml_dataset 即可。
# ============================================================

ml_df = pd.read_csv(
    ML_DATASET_FILE,
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


ml_df["region"] = (
    ml_df["region"]
    .astype(str)
    .str.strip()
)


print("=" * 90)
print("五舌區公平比較：Common-color Analysis")
print("=" * 90)

print(
    "AI 病例數：",
    ml_df[
        "image_id"
    ].nunique()
)


# ============================================================
# 9. 檢查每一區 × 每種顏色的 Ground Truth 分布
#
# 只有同時存在：
#
# positive > 0
# negative > 0
#
# 才算該區可評估此顏色。
# ============================================================

availability_records = []


for region in REGIONS:

    region_df = (
        ml_df[
            ml_df[
                "region"
            ]
            ==
            region
        ]
        .copy()
    )


    for color in COLORS:

        target_column = (
            f"y_{color}"
        )


        if (
            target_column
            not in
            region_df.columns
        ):

            raise KeyError(
                f"找不到欄位："
                f"{target_column}"
            )


        target = (
            region_df[
                target_column
            ]
        )


        positive = int(
            np.sum(
                target == 1
            )
        )


        negative = int(
            np.sum(
                target == 0
            )
        )


        uncertain = int(
            np.sum(
                target == -1
            )
        )


        evaluable = (
            positive > 0
            and
            negative > 0
        )


        availability_records.append({

            "region":
                region,

            "color":
                color,

            "positive":
                positive,

            "negative":
                negative,

            "uncertain":
                uncertain,

            "evaluable":
                evaluable,
        })


availability_df = pd.DataFrame(
    availability_records
)


availability_output = (
    OUTPUT_DIR
    / "region_color_ground_truth_availability.csv"
)


availability_df.to_csv(
    availability_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 10. 找出五個區域「共同可評估」的顏色
#
# 只有在：
#
# 舌尖
# 舌中
# 舌左
# 舌右
# 舌根
#
# 五區全部都 evaluable
# 才納入公平 Region Comparison。
# ============================================================

COMMON_COLORS = []


for color in COLORS:

    subset = (
        availability_df[
            availability_df[
                "color"
            ]
            ==
            color
        ]
    )


    if (
        len(
            subset
        )
        ==
        len(
            REGIONS
        )
        and
        subset[
            "evaluable"
        ].all()
    ):

        COMMON_COLORS.append(
            color
        )


print("\n")
print("=" * 90)
print("各區 Ground Truth 可評估狀況")
print("=" * 90)


availability_display = (
    availability_df
    .pivot(
        index="color",
        columns="region",
        values="evaluable"
    )
    .reindex(
        COLORS
    )
)


print(
    availability_display.to_string()
)


print("\n共同可公平比較的舌色：")

print(
    COMMON_COLORS
)


print(
    "共同顏色數：",
    len(
        COMMON_COLORS
    )
)


if len(
    COMMON_COLORS
) == 0:

    raise ValueError(
        "沒有任何顏色在五個舌區都同時具有正、負樣本。"
    )


# ============================================================
# 11. 讀取 SVM / RF Predictions
# ============================================================

svm_rf = pd.read_csv(
    SVM_RF_FILE,
    dtype={
        "image_id": str
    }
)


# ============================================================
# 12. 讀取 CNN Predictions
# ============================================================

cnn = pd.read_csv(
    CNN_FILE,
    dtype={
        "image_id": str
    }
)


required_columns = [
    "segmentation_method",
    "model",
    "color",
    "fold",
    "image_id",
    "region",
    "true",
    "pred",
]


for name, dataframe in [
    ("SVM/RF", svm_rf),
    ("CNN", cnn),
]:

    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]


    if missing:

        raise ValueError(
            f"{name} Prediction "
            f"缺少欄位：{missing}"
        )


svm_rf = (
    svm_rf[
        required_columns
    ]
    .copy()
)


cnn = (
    cnn[
        required_columns
    ]
    .copy()
)


# ============================================================
# 13. 合併 Prediction
# ============================================================

predictions = pd.concat(
    [
        svm_rf,
        cnn
    ],
    ignore_index=True
)


predictions[
    "image_id"
] = (
    normalize_image_id(
        predictions[
            "image_id"
        ]
    )
)


for column in [
    "segmentation_method",
    "model",
    "color",
    "region",
]:

    predictions[column] = (
        predictions[column]
        .astype(str)
        .str.strip()
    )


predictions[
    "true"
] = (
    predictions[
        "true"
    ]
    .astype(int)
)


predictions[
    "pred"
] = (
    predictions[
        "pred"
    ]
    .astype(int)
)


# ============================================================
# 14. 只留下共同可評估顏色
# ============================================================

fair_predictions = (
    predictions[
        predictions[
            "color"
        ].isin(
            COMMON_COLORS
        )
    ]
    .copy()
)


print("\n")
print(
    "Common-color prediction rows：",
    len(
        fair_predictions
    )
)


# ============================================================
# 15. Metrics Function
# ============================================================

def calculate_binary_metrics(
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


    if len(
        np.unique(
            y_true
        )
    ) >= 2:

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
# 16. 每個：
#
# segmentation × model × region × common color
#
# 計算 individual color metrics
# ============================================================

color_metric_records = []


for segmentation_method in (
    SEGMENTATION_METHODS
):

    for model in MODELS:

        for region in REGIONS:

            for color in COMMON_COLORS:

                subset = (
                    fair_predictions[
                        (
                            fair_predictions[
                                "segmentation_method"
                            ]
                            ==
                            segmentation_method
                        )
                        &
                        (
                            fair_predictions[
                                "model"
                            ]
                            ==
                            model
                        )
                        &
                        (
                            fair_predictions[
                                "region"
                            ]
                            ==
                            region
                        )
                        &
                        (
                            fair_predictions[
                                "color"
                            ]
                            ==
                            color
                        )
                    ]
                    .copy()
                )


                if len(
                    subset
                ) == 0:

                    continue


                metrics = (
                    calculate_binary_metrics(
                        subset[
                            "true"
                        ],
                        subset[
                            "pred"
                        ]
                    )
                )


                color_metric_records.append({

                    "segmentation_method":
                        segmentation_method,

                    "model":
                        model,

                    "region":
                        region,

                    "color":
                        color,

                    "n_samples":
                        len(
                            subset
                        ),

                    "n_patients":
                        subset[
                            "image_id"
                        ].nunique(),

                    **metrics
                })


color_metrics_df = pd.DataFrame(
    color_metric_records
)


color_metrics_output = (
    OUTPUT_DIR
    / "fair_region_common_color_metrics.csv"
)


color_metrics_df.to_csv(
    color_metrics_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 17. 公平五區比較
#
# 每個：
#
# segmentation × model × region
#
# 計算：
#
# A. Macro metrics
#    每種共同顏色先各自算，再平均
#
# B. Micro metrics
#    將所有共同顏色 prediction 合併計算
# ============================================================

fair_region_records = []


for segmentation_method in (
    SEGMENTATION_METHODS
):

    for model in MODELS:

        for region in REGIONS:

            # ------------------------------------------------
            # 這個區域的 per-color metrics
            # ------------------------------------------------

            color_subset = (
                color_metrics_df[
                    (
                        color_metrics_df[
                            "segmentation_method"
                        ]
                        ==
                        segmentation_method
                    )
                    &
                    (
                        color_metrics_df[
                            "model"
                        ]
                        ==
                        model
                    )
                    &
                    (
                        color_metrics_df[
                            "region"
                        ]
                        ==
                        region
                    )
                ]
                .copy()
            )


            if (
                len(
                    color_subset
                )
                !=
                len(
                    COMMON_COLORS
                )
            ):

                print(
                    "⚠ 警告：",
                    segmentation_method,
                    model,
                    region,
                    "沒有完整共同顏色結果"
                )

                continue


            # =================================================
            # Macro
            # =================================================

            macro_accuracy = (
                color_subset[
                    "accuracy"
                ].mean()
            )


            macro_precision = (
                color_subset[
                    "precision"
                ].mean()
            )


            macro_recall = (
                color_subset[
                    "recall"
                ].mean()
            )


            macro_f1 = (
                color_subset[
                    "f1"
                ].mean()
            )


            macro_ba = (
                color_subset[
                    "balanced_accuracy"
                ].mean()
            )


            # =================================================
            # Micro
            #
            # 把所有 common color 的 binary decisions
            # 合併後一次計算 TP / FP / FN
            # =================================================

            prediction_subset = (
                fair_predictions[
                    (
                        fair_predictions[
                            "segmentation_method"
                        ]
                        ==
                        segmentation_method
                    )
                    &
                    (
                        fair_predictions[
                            "model"
                        ]
                        ==
                        model
                    )
                    &
                    (
                        fair_predictions[
                            "region"
                        ]
                        ==
                        region
                    )
                    &
                    (
                        fair_predictions[
                            "color"
                        ].isin(
                            COMMON_COLORS
                        )
                    )
                ]
                .copy()
            )


            micro_metrics = (
                calculate_binary_metrics(
                    prediction_subset[
                        "true"
                    ],
                    prediction_subset[
                        "pred"
                    ]
                )
            )


            fair_region_records.append({

                "segmentation_method":
                    segmentation_method,

                "model":
                    model,

                "region":
                    region,

                "n_common_colors":
                    len(
                        COMMON_COLORS
                    ),

                "common_colors":
                    ",".join(
                        COMMON_COLORS
                    ),

                "n_binary_decisions":
                    len(
                        prediction_subset
                    ),

                "n_patients":
                    prediction_subset[
                        "image_id"
                    ].nunique(),

                # --------------------------------------------
                # Macro
                # --------------------------------------------

                "macro_accuracy":
                    macro_accuracy,

                "macro_precision":
                    macro_precision,

                "macro_recall":
                    macro_recall,

                "macro_f1":
                    macro_f1,

                "macro_balanced_accuracy":
                    macro_ba,

                # --------------------------------------------
                # Micro
                # --------------------------------------------

                "micro_accuracy":
                    micro_metrics[
                        "accuracy"
                    ],

                "micro_precision":
                    micro_metrics[
                        "precision"
                    ],

                "micro_recall":
                    micro_metrics[
                        "recall"
                    ],

                "micro_f1":
                    micro_metrics[
                        "f1"
                    ],

                "micro_balanced_accuracy":
                    micro_metrics[
                        "balanced_accuracy"
                    ],

                "micro_TN":
                    micro_metrics[
                        "TN"
                    ],

                "micro_FP":
                    micro_metrics[
                        "FP"
                    ],

                "micro_FN":
                    micro_metrics[
                        "FN"
                    ],

                "micro_TP":
                    micro_metrics[
                        "TP"
                    ],
            })


fair_region_df = pd.DataFrame(
    fair_region_records
)


fair_region_output = (
    OUTPUT_DIR
    / "fair_region_comparison.csv"
)


fair_region_df.to_csv(
    fair_region_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 18. 每一種模型組合內部：
# 將五個舌區依 Macro F1 排名
# ============================================================

ranking_records = []


for segmentation_method in (
    SEGMENTATION_METHODS
):

    for model in MODELS:

        subset = (
            fair_region_df[
                (
                    fair_region_df[
                        "segmentation_method"
                    ]
                    ==
                    segmentation_method
                )
                &
                (
                    fair_region_df[
                        "model"
                    ]
                    ==
                    model
                )
            ]
            .copy()
        )


        subset[
            "macro_f1_rank"
        ] = (
            subset[
                "macro_f1"
            ]
            .rank(
                ascending=False,
                method="min"
            )
            .astype(int)
        )


        subset[
            "micro_f1_rank"
        ] = (
            subset[
                "micro_f1"
            ]
            .rank(
                ascending=False,
                method="min"
            )
            .astype(int)
        )


        ranking_records.append(
            subset
        )


ranking_df = pd.concat(
    ranking_records,
    ignore_index=True
)


ranking_output = (
    OUTPUT_DIR
    / "fair_region_ranking.csv"
)


ranking_df.to_csv(
    ranking_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 19. 五區跨所有 6 組模型的描述性總覽
#
# 注意：
# 這只是描述，
# 不是正式統計推論。
# ============================================================

overall_region_records = []


for region in REGIONS:

    subset = (
        ranking_df[
            ranking_df[
                "region"
            ]
            ==
            region
        ]
        .copy()
    )


    overall_region_records.append({

        "region":
            region,

        "n_model_combinations":
            len(
                subset
            ),

        "mean_macro_f1":
            subset[
                "macro_f1"
            ].mean(),

        "std_macro_f1":
            subset[
                "macro_f1"
            ].std(),

        "mean_micro_f1":
            subset[
                "micro_f1"
            ].mean(),

        "std_micro_f1":
            subset[
                "micro_f1"
            ].std(),

        "macro_rank_1_count":
            int(
                np.sum(
                    subset[
                        "macro_f1_rank"
                    ]
                    ==
                    1
                )
            ),

        "micro_rank_1_count":
            int(
                np.sum(
                    subset[
                        "micro_f1_rank"
                    ]
                    ==
                    1
                )
            ),

        "mean_macro_rank":
            subset[
                "macro_f1_rank"
            ].mean(),

        "mean_micro_rank":
            subset[
                "micro_f1_rank"
            ].mean(),
    })


overall_region_df = pd.DataFrame(
    overall_region_records
)


overall_region_df = (
    overall_region_df
    .sort_values(
        [
            "mean_macro_f1",
            "mean_micro_f1"
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


overall_region_output = (
    OUTPUT_DIR
    / "fair_region_overall_summary.csv"
)


overall_region_df.to_csv(
    overall_region_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 20. 重新計算「共同顏色」下的醫師一致性
#
# 這樣醫師五區比較與 AI 一樣，
# 都只使用 COMMON_COLORS。
# ============================================================

vote_df = pd.read_csv(
    VOTE_FILE,
    dtype={
        "image_id": str
    }
)


vote_df[
    "image_id"
] = (
    normalize_image_id(
        vote_df[
            "image_id"
        ]
    )
)


AI_IMAGE_IDS = set(
    ml_df[
        "image_id"
    ].unique()
)


vote_subset = (
    vote_df[
        (
            vote_df[
                "image_id"
            ].isin(
                AI_IMAGE_IDS
            )
        )
        &
        (
            vote_df[
                "color"
            ].isin(
                COMMON_COLORS
            )
        )
    ]
    .copy()
)


# ============================================================
# 21. Fleiss' Kappa
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


    return float(

        (
            observed_agreement
            -
            expected_agreement
        )

        /

        denominator
    )


# ============================================================
# 22. 醫師五區公平 Agreement
# ============================================================

physician_records = []


for region in REGIONS:

    region_vote = (
        vote_subset[
            vote_subset[
                "region"
            ]
            ==
            region
        ]
        .copy()
    )


    ratings = (
        region_vote[
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


    physician_records.append({

        "region":
            region,

        "common_colors":
            ",".join(
                COMMON_COLORS
            ),

        "n_items":
            n_items,

        "unanimous_rate":
            unanimous_n
            /
            n_items,

        "three_one_rate":
            three_one_n
            /
            n_items,

        "two_two_rate":
            two_two_n
            /
            n_items,

        "fleiss_kappa":
            fleiss_kappa_binary(
                ratings
            ),
    })


physician_df = pd.DataFrame(
    physician_records
)


physician_output = (
    OUTPUT_DIR
    / "physician_agreement_common_colors_by_region.csv"
)


physician_df.to_csv(
    physician_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 23. AI + 醫師公平五區總表
# ============================================================

region_final = (
    overall_region_df
    .merge(
        physician_df,
        on="region",
        how="left",
        validate="one_to_one"
    )
)


region_final_output = (
    OUTPUT_DIR
    / "fair_region_ai_physician_summary.csv"
)


region_final.to_csv(
    region_final_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 24. 顯示：
# Overall Best Model = 1:3:1 + SVM
#
# 用固定模型比較五區，
# 避免每區自己挑最好模型造成偏差。
# ============================================================

REFERENCE_METHOD = (
    "tip_middle_root_1_3_1"
)

REFERENCE_MODEL = (
    "SVM"
)


reference_result = (
    fair_region_df[
        (
            fair_region_df[
                "segmentation_method"
            ]
            ==
            REFERENCE_METHOD
        )
        &
        (
            fair_region_df[
                "model"
            ]
            ==
            REFERENCE_MODEL
        )
    ]
    .copy()
)


reference_result = (
    reference_result
    .sort_values(
        "macro_f1",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 25. Terminal 顯示 Ground Truth
# ============================================================

print("\n")
print("=" * 90)
print("共同顏色的 Ground Truth 分布")
print("=" * 90)


common_availability = (
    availability_df[
        availability_df[
            "color"
        ].isin(
            COMMON_COLORS
        )
    ]
    .copy()
)


print(
    common_availability[
        [
            "region",
            "color",
            "positive",
            "negative",
            "uncertain"
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 26. Terminal 顯示：
# 固定 1:3:1 + SVM 的五區公平比較
# ============================================================

print("\n")
print("=" * 90)
print("固定模型公平比較：1:3:1 + SVM")
print("=" * 90)


reference_display = (
    reference_result[
        [
            "region",
            "n_common_colors",
            "macro_f1",
            "macro_balanced_accuracy",
            "micro_f1",
            "micro_balanced_accuracy",
        ]
    ]
    .copy()
)


for column in [
    "macro_f1",
    "macro_balanced_accuracy",
    "micro_f1",
    "micro_balanced_accuracy",
]:

    reference_display[
        column
    ] = (
        reference_display[
            column
        ]
        .round(
            4
        )
    )


print(
    reference_display.to_string(
        index=False
    )
)


# ============================================================
# 27. Terminal 顯示：
# 六種模型組合跨區域總覽
# ============================================================

print("\n")
print("=" * 90)
print("五區跨全部模型組合的描述性總覽")
print("=" * 90)


overall_display = (
    overall_region_df.copy()
)


for column in [
    "mean_macro_f1",
    "std_macro_f1",
    "mean_micro_f1",
    "std_micro_f1",
    "mean_macro_rank",
    "mean_micro_rank",
]:

    overall_display[
        column
    ] = (
        overall_display[
            column
        ]
        .round(
            4
        )
    )


print(
    overall_display.to_string(
        index=False
    )
)


# ============================================================
# 28. Terminal 顯示：
# 公平醫師五區 Agreement
# ============================================================

print("\n")
print("=" * 90)
print("共同顏色下的五區醫師一致性")
print("=" * 90)


physician_display = (
    physician_df[
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

    physician_display[
        column
    ] = (
        physician_display[
            column
        ]
        .round(
            4
        )
    )


print(
    physician_display.to_string(
        index=False
    )
)


# ============================================================
# 29. 最終判斷：
# 固定 1:3:1 + SVM
# ============================================================

if len(
    reference_result
) > 0:

    easiest = (
        reference_result.iloc[
            0
        ]
    )


    hardest = (
        reference_result.iloc[
            -1
        ]
    )


    print("\n")
    print("=" * 90)
    print("公平 Region Difficulty 結果")
    print("=" * 90)


    print(
        "共同顏色：",
        COMMON_COLORS
    )


    print(
        "固定模型：",
        REFERENCE_METHOD,
        "+",
        REFERENCE_MODEL
    )


    print(
        "\n目前最容易辨識：",
        easiest[
            "region"
        ],
        "| Macro F1 =",
        round(
            easiest[
                "macro_f1"
            ],
            4
        ),
        "| Micro F1 =",
        round(
            easiest[
                "micro_f1"
            ],
            4
        )
    )


    print(
        "目前最難辨識：",
        hardest[
            "region"
        ],
        "| Macro F1 =",
        round(
            hardest[
                "macro_f1"
            ],
            4
        ),
        "| Micro F1 =",
        round(
            hardest[
                "micro_f1"
            ],
            4
        )
    )


# ============================================================
# 30. 輸出
# ============================================================

print("\n")
print("=" * 90)
print("分析完成")
print("=" * 90)


print(
    "\nGround Truth 可評估狀況："
)

print(
    availability_output
)


print(
    "\nCommon-color individual metrics："
)

print(
    color_metrics_output
)


print(
    "\n公平 Region Comparison："
)

print(
    fair_region_output
)


print(
    "\n五區 Ranking："
)

print(
    ranking_output
)


print(
    "\n五區 Overall Summary："
)

print(
    overall_region_output
)


print(
    "\n共同顏色醫師 Agreement："
)

print(
    physician_output
)


print(
    "\nAI + 醫師公平五區總表："
)

print(
    region_final_output
)


print("=" * 90)

