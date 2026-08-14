from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
)


# ============================================================
# 1. 基本設定
# ============================================================

SEED = 42
BOOTSTRAP_ITERATIONS = 2000


PRIMARY_COLORS = [
    "淡紅",
    "鮮紅",
]


SEGMENTATION_METHODS = [
    "tip_middle_root_1_2_2",
    "tip_middle_root_1_3_1",
]


MODELS = [
    "SVM",
    "Random Forest",
    "CNN",
]


# ============================================================
# 2. 路徑
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


OUTPUT_DIR = (
    BASE_DIR
    /
    "output"
)


SVM_RF_FILE = (
    OUTPUT_DIR
    /
    "unanimous_to_disagreement_svm_rf_predictions.csv"
)


CNN_FILE = (
    OUTPUT_DIR
    /
    "unanimous_to_disagreement_cnn_predictions.csv"
)


# ============================================================
# 3. 檢查檔案
# ============================================================

for file_path in [
    SVM_RF_FILE,
    CNN_FILE,
]:

    if not file_path.exists():

        raise FileNotFoundError(
            f"找不到：{file_path}"
        )


# ============================================================
# 4. 讀取 Predictions
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


# ============================================================
# 5. 統一欄位
# ============================================================

required_columns = [

    "color",
    "segmentation_method",
    "model",
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

        if column
        not in dataframe.columns
    ]


    if missing:

        raise ValueError(
            f"{name} 缺少欄位：{missing}"
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


predictions = pd.concat(
    [
        svm_rf,
        cnn,
    ],
    ignore_index=True
)


# ============================================================
# 6. 清理格式
# ============================================================

predictions["image_id"] = (
    predictions[
        "image_id"
    ]
    .astype(str)
    .str.strip()
    .str.replace(
        r"\.0$",
        "",
        regex=True
    )
)


for column in [
    "color",
    "segmentation_method",
    "model",
    "region",
]:

    predictions[column] = (
        predictions[
            column
        ]
        .astype(str)
        .str.strip()
    )


predictions["true"] = (
    predictions[
        "true"
    ]
    .astype(int)
)


predictions["pred"] = (
    predictions[
        "pred"
    ]
    .astype(int)
)


# ============================================================
# 7. 只取正式 primary colors
# ============================================================

predictions = (
    predictions[
        predictions[
            "color"
        ]
        .isin(
            PRIMARY_COLORS
        )
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


print("=" * 100)
print(
    "Unanimous Training → 3:1 Disagreement"
)
print(
    "Patient-level Paired Bootstrap"
)
print("=" * 100)


print(
    "總 prediction rows：",
    len(
        predictions
    )
)


# ============================================================
# 8. 檢查每個 model × segmentation
#
# 每組理論上：
#
# 淡紅 173
# 鮮紅 170
#
# 共 343 predictions
# ============================================================

combination_counts = (

    predictions
    .groupby(
        [
            "segmentation_method",
            "model",
        ]
    )
    .size()
    .reset_index(
        name="n_predictions"
    )
)


print("\n")
print(
    combination_counts.to_string(
        index=False
    )
)


# ============================================================
# 9. 檢查六種 combination 是否使用完全相同的測試案例
# ============================================================

KEY_COLUMNS = [
    "color",
    "image_id",
    "region",
    "true",
]


reference_keys = None


for segmentation_method in (
    SEGMENTATION_METHODS
):

    for model in MODELS:

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
            ]
            .copy()
        )


        key_set = set(

            map(
                tuple,

                subset[
                    KEY_COLUMNS
                ]
                .to_numpy()
            )
        )


        if reference_keys is None:

            reference_keys = (
                key_set
            )

        else:

            if (
                key_set
                !=
                reference_keys
            ):

                raise ValueError(

                    f"{segmentation_method} / {model} "
                    "與其他模型使用的測試案例不同。"
                    "請先停止分析。"
                )


print(
    "\n✓ 六種模型組合使用完全相同的 3:1 測試案例"
)


# ============================================================
# 10. 檢查 Prediction 是否重複
# ============================================================

duplicate_columns = [

    "color",
    "segmentation_method",
    "model",
    "image_id",
    "region",
]


duplicates = (
    predictions
    .duplicated(
        subset=
            duplicate_columns,
        keep=False
    )
)


if duplicates.any():

    raise ValueError(
        "發現重複 prediction。"
    )


# ============================================================
# 11. Weighted binary metrics
# ============================================================

def calculate_binary_metrics(
    y_true,
    y_pred,
    weights=None
):

    y_true = np.asarray(
        y_true,
        dtype=int
    )


    y_pred = np.asarray(
        y_pred,
        dtype=int
    )


    if weights is None:

        weights = np.ones(
            len(
                y_true
            ),
            dtype=float
        )

    else:

        weights = np.asarray(
            weights,
            dtype=float
        )


    positive_weight = (
        weights[
            y_true == 1
        ]
        .sum()
    )


    negative_weight = (
        weights[
            y_true == 0
        ]
        .sum()
    )


    if (
        positive_weight <= 0
        or
        negative_weight <= 0
    ):

        return None


    return {

        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred,
                    sample_weight=
                        weights
                )
            ),

        "precision":
            float(
                precision_score(
                    y_true,
                    y_pred,
                    sample_weight=
                        weights,
                    zero_division=0
                )
            ),

        "recall":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    sample_weight=
                        weights,
                    zero_division=0
                )
            ),

        "f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    sample_weight=
                        weights,
                    zero_division=0
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y_true,
                    y_pred,
                    sample_weight=
                        weights
                )
            ),
    }


# ============================================================
# 12. Macro over 淡紅 + 鮮紅
# ============================================================

def calculate_macro_metrics(
    dataframe,
    weight_column=None
):

    color_results = []


    for color in PRIMARY_COLORS:

        subset = (
            dataframe[
                dataframe[
                    "color"
                ]
                ==
                color
            ]
        )


        if len(
            subset
        ) == 0:

            return None


        if weight_column is None:

            weights = None

        else:

            weights = (
                subset[
                    weight_column
                ]
                .to_numpy(
                    dtype=float
                )
            )


        result = (
            calculate_binary_metrics(

                subset[
                    "true"
                ],

                subset[
                    "pred"
                ],

                weights
            )
        )


        if result is None:

            return None


        color_results.append(
            result
        )


    result_df = pd.DataFrame(
        color_results
    )


    return {

        "accuracy":
            float(
                result_df[
                    "accuracy"
                ]
                .mean()
            ),

        "precision":
            float(
                result_df[
                    "precision"
                ]
                .mean()
            ),

        "recall":
            float(
                result_df[
                    "recall"
                ]
                .mean()
            ),

        "f1":
            float(
                result_df[
                    "f1"
                ]
                .mean()
            ),

        "balanced_accuracy":
            float(
                result_df[
                    "balanced_accuracy"
                ]
                .mean()
            ),
    }


# ============================================================
# 13. CI function
# ============================================================

def percentile_ci(
    values
):

    values = np.asarray(
        values,
        dtype=float
    )


    values = (
        values[
            ~np.isnan(
                values
            )
        ]
    )


    return (

        float(
            np.percentile(
                values,
                2.5
            )
        ),

        float(
            np.percentile(
                values,
                97.5
            )
        )
    )


# ============================================================
# 14. 取得所有具有 primary 3:1 cases 的病人
#
# bootstrap 單位 = image_id
#
# 若同一病人同時出現在淡紅與鮮紅，
# 必須一起重新抽樣。
# ============================================================

MASTER_PATIENTS = sorted(

    predictions[
        "image_id"
    ]
    .unique()
    .tolist()
)


N_PATIENTS = len(
    MASTER_PATIENTS
)


print(
    "Bootstrap patient 數：",
    N_PATIENTS
)


# ============================================================
# 15. Point Estimates
# ============================================================

point_records = []


for segmentation_method in (
    SEGMENTATION_METHODS
):

    for model in MODELS:

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
            ]
            .copy()
        )


        metrics = (
            calculate_macro_metrics(
                subset
            )
        )


        point_records.append({

            "segmentation_method":
                segmentation_method,

            "model":
                model,

            **metrics,
        })


point_df = pd.DataFrame(
    point_records
)


# ============================================================
# 16. Patient-level bootstrap
#
# 同一 bootstrap 抽樣，同時套用六種模型，
# 才能進行真正 paired comparison。
# ============================================================

rng = np.random.default_rng(
    SEED
)


bootstrap_records = []


for bootstrap_id in range(
    BOOTSTRAP_ITERATIONS
):

    sampled_patients = (
        rng.choice(

            MASTER_PATIENTS,

            size=
                N_PATIENTS,

            replace=
                True
        )
    )


    patient_counts = (

        pd.Series(
            sampled_patients
        )
        .value_counts()
        .to_dict()
    )


    bootstrap_valid = (
        True
    )


    iteration_results = []


    for segmentation_method in (
        SEGMENTATION_METHODS
    ):

        for model in MODELS:

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
                ]
                .copy()
            )


            subset[
                "_weight"
            ] = (

                subset[
                    "image_id"
                ]
                .map(
                    patient_counts
                )
                .fillna(
                    0
                )
                .astype(float)
            )


            metrics = (
                calculate_macro_metrics(

                    subset,

                    weight_column=
                        "_weight"
                )
            )


            if metrics is None:

                bootstrap_valid = (
                    False
                )

                break


            iteration_results.append({

                "bootstrap_id":
                    bootstrap_id,

                "segmentation_method":
                    segmentation_method,

                "model":
                    model,

                **metrics,
            })


        if not bootstrap_valid:

            break


    if bootstrap_valid:

        bootstrap_records.extend(
            iteration_results
        )


bootstrap_df = pd.DataFrame(
    bootstrap_records
)


valid_iterations = (
    bootstrap_df[
        "bootstrap_id"
    ]
    .nunique()
)


print(
    "有效 bootstrap iterations：",
    valid_iterations
)


# ============================================================
# 17. 六模型組合 CI
# ============================================================

summary_records = []


for _, point_row in (
    point_df.iterrows()
):

    segmentation_method = (
        point_row[
            "segmentation_method"
        ]
    )


    model = (
        point_row[
            "model"
        ]
    )


    boot = (
        bootstrap_df[
            (
                bootstrap_df[
                    "segmentation_method"
                ]
                ==
                segmentation_method
            )
            &
            (
                bootstrap_df[
                    "model"
                ]
                ==
                model
            )
        ]
    )


    f1_ci = (
        percentile_ci(
            boot[
                "f1"
            ]
        )
    )


    ba_ci = (
        percentile_ci(
            boot[
                "balanced_accuracy"
            ]
        )
    )


    accuracy_ci = (
        percentile_ci(
            boot[
                "accuracy"
            ]
        )
    )


    summary_records.append({

        "segmentation_method":
            segmentation_method,

        "model":
            model,

        "accuracy":
            point_row[
                "accuracy"
            ],

        "accuracy_ci_lower":
            accuracy_ci[
                0
            ],

        "accuracy_ci_upper":
            accuracy_ci[
                1
            ],

        "f1":
            point_row[
                "f1"
            ],

        "f1_ci_lower":
            f1_ci[
                0
            ],

        "f1_ci_upper":
            f1_ci[
                1
            ],

        "balanced_accuracy":
            point_row[
                "balanced_accuracy"
            ],

        "balanced_accuracy_ci_lower":
            ba_ci[
                0
            ],

        "balanced_accuracy_ci_upper":
            ba_ci[
                1
            ],
    })


summary_df = pd.DataFrame(
    summary_records
)


summary_df[
    "rank_by_f1"
] = (

    summary_df[
        "f1"
    ]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)


summary_df = (

    summary_df
    .sort_values(
        "rank_by_f1"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 18. Paired comparison helper
# ============================================================

def paired_bootstrap_comparison(
    segmentation_a,
    model_a,
    segmentation_b,
    model_b,
    comparison_type
):

    a = (

        bootstrap_df[
            (
                bootstrap_df[
                    "segmentation_method"
                ]
                ==
                segmentation_a
            )
            &
            (
                bootstrap_df[
                    "model"
                ]
                ==
                model_a
            )
        ][
            [
                "bootstrap_id",
                "accuracy",
                "f1",
                "balanced_accuracy",
            ]
        ]
        .rename(
            columns={

                "accuracy":
                    "accuracy_a",

                "f1":
                    "f1_a",

                "balanced_accuracy":
                    "ba_a",
            }
        )
    )


    b = (

        bootstrap_df[
            (
                bootstrap_df[
                    "segmentation_method"
                ]
                ==
                segmentation_b
            )
            &
            (
                bootstrap_df[
                    "model"
                ]
                ==
                model_b
            )
        ][
            [
                "bootstrap_id",
                "accuracy",
                "f1",
                "balanced_accuracy",
            ]
        ]
        .rename(
            columns={

                "accuracy":
                    "accuracy_b",

                "f1":
                    "f1_b",

                "balanced_accuracy":
                    "ba_b",
            }
        )
    )


    paired = (
        a.merge(
            b,
            on="bootstrap_id",
            how="inner",
            validate="one_to_one"
        )
    )


    paired[
        "delta_accuracy"
    ] = (
        paired[
            "accuracy_a"
        ]
        -
        paired[
            "accuracy_b"
        ]
    )


    paired[
        "delta_f1"
    ] = (
        paired[
            "f1_a"
        ]
        -
        paired[
            "f1_b"
        ]
    )


    paired[
        "delta_ba"
    ] = (
        paired[
            "ba_a"
        ]
        -
        paired[
            "ba_b"
        ]
    )


    # --------------------------------------------------------
    # point estimate
    # --------------------------------------------------------

    point_a = (
        point_df[
            (
                point_df[
                    "segmentation_method"
                ]
                ==
                segmentation_a
            )
            &
            (
                point_df[
                    "model"
                ]
                ==
                model_a
            )
        ]
        .iloc[
            0
        ]
    )


    point_b = (
        point_df[
            (
                point_df[
                    "segmentation_method"
                ]
                ==
                segmentation_b
            )
            &
            (
                point_df[
                    "model"
                ]
                ==
                model_b
            )
        ]
        .iloc[
            0
        ]
    )


    delta_f1 = (
        point_a[
            "f1"
        ]
        -
        point_b[
            "f1"
        ]
    )


    delta_ba = (
        point_a[
            "balanced_accuracy"
        ]
        -
        point_b[
            "balanced_accuracy"
        ]
    )


    delta_accuracy = (
        point_a[
            "accuracy"
        ]
        -
        point_b[
            "accuracy"
        ]
    )


    f1_ci = (
        percentile_ci(
            paired[
                "delta_f1"
            ]
        )
    )


    ba_ci = (
        percentile_ci(
            paired[
                "delta_ba"
            ]
        )
    )


    accuracy_ci = (
        percentile_ci(
            paired[
                "delta_accuracy"
            ]
        )
    )


    # --------------------------------------------------------
    # 解讀
    # --------------------------------------------------------

    if (
        f1_ci[0] > 0
    ):

        f1_interpretation = (
            "A_higher"
        )

    elif (
        f1_ci[1] < 0
    ):

        f1_interpretation = (
            "B_higher"
        )

    else:

        f1_interpretation = (
            "CI_crosses_zero"
        )


    return {

        "comparison_type":
            comparison_type,

        "segmentation_a":
            segmentation_a,

        "model_a":
            model_a,

        "segmentation_b":
            segmentation_b,

        "model_b":
            model_b,

        "delta_definition":
            "A_minus_B",

        "delta_accuracy":
            delta_accuracy,

        "delta_accuracy_ci_lower":
            accuracy_ci[
                0
            ],

        "delta_accuracy_ci_upper":
            accuracy_ci[
                1
            ],

        "delta_f1":
            delta_f1,

        "delta_f1_ci_lower":
            f1_ci[
                0
            ],

        "delta_f1_ci_upper":
            f1_ci[
                1
            ],

        "delta_balanced_accuracy":
            delta_ba,

        "delta_balanced_accuracy_ci_lower":
            ba_ci[
                0
            ],

        "delta_balanced_accuracy_ci_upper":
            ba_ci[
                1
            ],

        "f1_interpretation":
            f1_interpretation,
    }


# ============================================================
# 19. 固定 segmentation，比較三模型
# ============================================================

comparison_records = []


MODEL_PAIRS = [

    (
        "SVM",
        "Random Forest"
    ),

    (
        "SVM",
        "CNN"
    ),

    (
        "Random Forest",
        "CNN"
    ),
]


for segmentation_method in (
    SEGMENTATION_METHODS
):

    for model_a, model_b in (
        MODEL_PAIRS
    ):

        comparison_records.append(

            paired_bootstrap_comparison(

                segmentation_method,
                model_a,

                segmentation_method,
                model_b,

                comparison_type=
                    "model_comparison_fixed_segmentation"
            )
        )


# ============================================================
# 20. 固定模型，比較 1:3:1 vs 1:2:2
#
# A = 1:3:1
# B = 1:2:2
# ============================================================

for model in MODELS:

    comparison_records.append(

        paired_bootstrap_comparison(

            "tip_middle_root_1_3_1",
            model,

            "tip_middle_root_1_2_2",
            model,

            comparison_type=
                "segmentation_comparison_fixed_model"
        )
    )


comparison_df = pd.DataFrame(
    comparison_records
)


# ============================================================
# 21. 模型跨兩種 segmentation 的平均表現
#
# 不挑「最好分區」，
# 而是對兩種 segmentation 取平均。
#
# 用來回答：
# 整體而言哪一個模型較穩定？
# ============================================================

model_average_point = (

    point_df
    .groupby(
        "model",
        as_index=False
    )
    .agg(

        mean_accuracy=(
            "accuracy",
            "mean"
        ),

        mean_f1=(
            "f1",
            "mean"
        ),

        mean_balanced_accuracy=(
            "balanced_accuracy",
            "mean"
        ),
    )
)


model_average_bootstrap = (

    bootstrap_df
    .groupby(
        [
            "bootstrap_id",
            "model",
        ],
        as_index=False
    )
    .agg(

        mean_accuracy=(
            "accuracy",
            "mean"
        ),

        mean_f1=(
            "f1",
            "mean"
        ),

        mean_balanced_accuracy=(
            "balanced_accuracy",
            "mean"
        ),
    )
)


model_average_records = []


for model in MODELS:

    point = (
        model_average_point[
            model_average_point[
                "model"
            ]
            ==
            model
        ]
        .iloc[
            0
        ]
    )


    boot = (
        model_average_bootstrap[
            model_average_bootstrap[
                "model"
            ]
            ==
            model
        ]
    )


    f1_ci = (
        percentile_ci(
            boot[
                "mean_f1"
            ]
        )
    )


    ba_ci = (
        percentile_ci(
            boot[
                "mean_balanced_accuracy"
            ]
        )
    )


    model_average_records.append({

        "model":
            model,

        "mean_f1":
            point[
                "mean_f1"
            ],

        "mean_f1_ci_lower":
            f1_ci[
                0
            ],

        "mean_f1_ci_upper":
            f1_ci[
                1
            ],

        "mean_balanced_accuracy":
            point[
                "mean_balanced_accuracy"
            ],

        "mean_balanced_accuracy_ci_lower":
            ba_ci[
                0
            ],

        "mean_balanced_accuracy_ci_upper":
            ba_ci[
                1
            ],
    })


model_average_df = pd.DataFrame(
    model_average_records
)


model_average_df[
    "rank_by_mean_f1"
] = (

    model_average_df[
        "mean_f1"
    ]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)


model_average_df = (

    model_average_df
    .sort_values(
        "rank_by_mean_f1"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 22. 模型平均 Pairwise Comparison
# ============================================================

model_average_comparison_records = []


for model_a, model_b in (
    MODEL_PAIRS
):

    a = (

        model_average_bootstrap[
            model_average_bootstrap[
                "model"
            ]
            ==
            model_a
        ][
            [
                "bootstrap_id",
                "mean_f1",
                "mean_balanced_accuracy",
            ]
        ]
        .rename(
            columns={

                "mean_f1":
                    "f1_a",

                "mean_balanced_accuracy":
                    "ba_a",
            }
        )
    )


    b = (

        model_average_bootstrap[
            model_average_bootstrap[
                "model"
            ]
            ==
            model_b
        ][
            [
                "bootstrap_id",
                "mean_f1",
                "mean_balanced_accuracy",
            ]
        ]
        .rename(
            columns={

                "mean_f1":
                    "f1_b",

                "mean_balanced_accuracy":
                    "ba_b",
            }
        )
    )


    paired = (
        a.merge(
            b,
            on="bootstrap_id",
            how="inner"
        )
    )


    paired[
        "delta_f1"
    ] = (
        paired[
            "f1_a"
        ]
        -
        paired[
            "f1_b"
        ]
    )


    paired[
        "delta_ba"
    ] = (
        paired[
            "ba_a"
        ]
        -
        paired[
            "ba_b"
        ]
    )


    point_a = (
        model_average_point[
            model_average_point[
                "model"
            ]
            ==
            model_a
        ]
        .iloc[
            0
        ]
    )


    point_b = (
        model_average_point[
            model_average_point[
                "model"
            ]
            ==
            model_b
        ]
        .iloc[
            0
        ]
    )


    delta_f1 = (
        point_a[
            "mean_f1"
        ]
        -
        point_b[
            "mean_f1"
        ]
    )


    delta_ba = (
        point_a[
            "mean_balanced_accuracy"
        ]
        -
        point_b[
            "mean_balanced_accuracy"
        ]
    )


    f1_ci = (
        percentile_ci(
            paired[
                "delta_f1"
            ]
        )
    )


    ba_ci = (
        percentile_ci(
            paired[
                "delta_ba"
            ]
        )
    )


    if f1_ci[0] > 0:

        interpretation = (
            f"{model_a}_higher"
        )

    elif f1_ci[1] < 0:

        interpretation = (
            f"{model_b}_higher"
        )

    else:

        interpretation = (
            "CI_crosses_zero"
        )


    model_average_comparison_records.append({

        "model_a":
            model_a,

        "model_b":
            model_b,

        "delta_mean_f1":
            delta_f1,

        "delta_mean_f1_ci_lower":
            f1_ci[
                0
            ],

        "delta_mean_f1_ci_upper":
            f1_ci[
                1
            ],

        "delta_mean_balanced_accuracy":
            delta_ba,

        "delta_mean_balanced_accuracy_ci_lower":
            ba_ci[
                0
            ],

        "delta_mean_balanced_accuracy_ci_upper":
            ba_ci[
                1
            ],

        "f1_interpretation":
            interpretation,
    })


model_average_comparison_df = pd.DataFrame(
    model_average_comparison_records
)


# ============================================================
# 23. 輸出
# ============================================================

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    /
    "unanimous_disagreement_bootstrap_ci.csv"
)


COMPARISON_OUTPUT = (
    OUTPUT_DIR
    /
    "unanimous_disagreement_paired_bootstrap_comparison.csv"
)


MODEL_AVERAGE_OUTPUT = (
    OUTPUT_DIR
    /
    "unanimous_disagreement_model_average_bootstrap.csv"
)


MODEL_AVERAGE_COMPARISON_OUTPUT = (
    OUTPUT_DIR
    /
    "unanimous_disagreement_model_average_comparison.csv"
)


BOOTSTRAP_OUTPUT = (
    OUTPUT_DIR
    /
    "unanimous_disagreement_bootstrap_replicates.csv"
)


summary_df.to_csv(
    SUMMARY_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


comparison_df.to_csv(
    COMPARISON_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


model_average_df.to_csv(
    MODEL_AVERAGE_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


model_average_comparison_df.to_csv(
    MODEL_AVERAGE_COMPARISON_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


bootstrap_df.to_csv(
    BOOTSTRAP_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 24. Terminal：六種組合 CI
# ============================================================

print("\n")
print("=" * 100)
print(
    "六種組合 Patient-level Bootstrap 95% CI"
)
print("=" * 100)


display_summary = (

    summary_df[
        [
            "segmentation_method",
            "model",
            "f1",
            "f1_ci_lower",
            "f1_ci_upper",
            "balanced_accuracy",
            "balanced_accuracy_ci_lower",
            "balanced_accuracy_ci_upper",
            "rank_by_f1",
        ]
    ]
    .copy()
)


for column in [
    "f1",
    "f1_ci_lower",
    "f1_ci_upper",
    "balanced_accuracy",
    "balanced_accuracy_ci_lower",
    "balanced_accuracy_ci_upper",
]:

    display_summary[
        column
    ] = (

        display_summary[
            column
        ]
        .round(
            4
        )
    )


print(
    display_summary.to_string(
        index=False
    )
)


# ============================================================
# 25. Terminal：Paired comparisons
# ============================================================

print("\n")
print("=" * 100)
print(
    "Paired Bootstrap Comparisons"
)
print("=" * 100)


display_comparison = (

    comparison_df[
        [
            "comparison_type",
            "segmentation_a",
            "model_a",
            "segmentation_b",
            "model_b",
            "delta_f1",
            "delta_f1_ci_lower",
            "delta_f1_ci_upper",
            "delta_balanced_accuracy",
            "delta_balanced_accuracy_ci_lower",
            "delta_balanced_accuracy_ci_upper",
            "f1_interpretation",
        ]
    ]
    .copy()
)


for column in [
    "delta_f1",
    "delta_f1_ci_lower",
    "delta_f1_ci_upper",
    "delta_balanced_accuracy",
    "delta_balanced_accuracy_ci_lower",
    "delta_balanced_accuracy_ci_upper",
]:

    display_comparison[
        column
    ] = (

        display_comparison[
            column
        ]
        .round(
            4
        )
    )


print(
    display_comparison.to_string(
        index=False
    )
)


# ============================================================
# 26. Terminal：模型跨兩種 segmentation 平均
# ============================================================

print("\n")
print("=" * 100)
print(
    "三模型跨兩種分區之平均表現"
)
print("=" * 100)


display_model_average = (
    model_average_df.copy()
)


for column in [
    "mean_f1",
    "mean_f1_ci_lower",
    "mean_f1_ci_upper",
    "mean_balanced_accuracy",
    "mean_balanced_accuracy_ci_lower",
    "mean_balanced_accuracy_ci_upper",
]:

    display_model_average[
        column
    ] = (

        display_model_average[
            column
        ]
        .round(
            4
        )
    )


print(
    display_model_average.to_string(
        index=False
    )
)


print("\n")
print("=" * 100)
print(
    "三模型平均表現 Pairwise Comparison"
)
print("=" * 100)


display_average_comparison = (
    model_average_comparison_df.copy()
)


for column in [
    "delta_mean_f1",
    "delta_mean_f1_ci_lower",
    "delta_mean_f1_ci_upper",
    "delta_mean_balanced_accuracy",
    "delta_mean_balanced_accuracy_ci_lower",
    "delta_mean_balanced_accuracy_ci_upper",
]:

    display_average_comparison[
        column
    ] = (

        display_average_comparison[
            column
        ]
        .round(
            4
        )
    )


print(
    display_average_comparison.to_string(
        index=False
    )
)


# ============================================================
# 27. 輸出位置
# ============================================================

print("\n")
print("=" * 100)
print(
    "輸出檔案"
)
print("=" * 100)


print(
    SUMMARY_OUTPUT
)

print(
    COMPARISON_OUTPUT
)

print(
    MODEL_AVERAGE_OUTPUT
)

print(
    MODEL_AVERAGE_COMPARISON_OUTPUT
)

print(
    BOOTSTRAP_OUTPUT
)

print("=" * 100)
