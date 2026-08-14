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

# Patient-level bootstrap 次數
BOOTSTRAP_ITERATIONS = 2000


# ============================================================
# 2. 路徑
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


# ============================================================
# 3. 分區方法
# ============================================================

SEGMENTATION_METHODS = [
    "tip_middle_root_1_2_2",
    "tip_middle_root_1_3_1",
]


# ============================================================
# 4. 模型
# ============================================================

MODELS = [
    "SVM",
    "Random Forest",
    "CNN",
]


# ============================================================
# 5. 可評估顏色
#
# 灰黑沒有 consensus positive，
# 所以不納入正式 classifier evaluation。
# ============================================================

EVALUABLE_COLORS = [
    "淡紅",
    "淡白",
    "鮮紅",
    "暗紅",
    "青紫",
]


# ============================================================
# 6. 檢查檔案
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
# 7. 讀取 Predictions
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
            f"{name} prediction 缺少欄位："
            f"{missing}"
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
        cnn
    ],
    ignore_index=True
)


# ============================================================
# 8. 清理資料格式
# ============================================================

predictions["image_id"] = (
    predictions["image_id"]
    .astype(str)
    .str.strip()
    .str.replace(
        r"\.0$",
        "",
        regex=True
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


predictions["true"] = (
    predictions["true"]
    .astype(int)
)


predictions["pred"] = (
    predictions["pred"]
    .astype(int)
)


# ============================================================
# 9. 只留下五個真正可評估舌色
# ============================================================

predictions = (
    predictions[
        predictions[
            "color"
        ].isin(
            EVALUABLE_COLORS
        )
    ]
    .copy()
)


print("=" * 90)
print("Final Patient-level Bootstrap Analysis")
print("=" * 90)

print(
    "總 prediction rows：",
    len(
        predictions
    )
)

print(
    "病人數：",
    predictions[
        "image_id"
    ].nunique()
)

print(
    "舌質色：",
    EVALUABLE_COLORS
)


# ============================================================
# 10. 檢查重複 Prediction
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

    print(
        predictions[
            duplicates
        ][
            duplicate_columns
        ].head(
            20
        )
    )

    raise ValueError(
        "發現重複 out-of-fold predictions。"
    )


# ============================================================
# 11. Weighted Binary Metrics
#
# bootstrap 時不需要真的把病人資料複製很多次，
# 而是使用 patient sampling weight。
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    sample_weight=None,
):

    y_true = np.asarray(
        y_true,
        dtype=int
    )

    y_pred = np.asarray(
        y_pred,
        dtype=int
    )


    if sample_weight is None:

        sample_weight = np.ones(
            len(
                y_true
            ),
            dtype=float
        )

    else:

        sample_weight = np.asarray(
            sample_weight,
            dtype=float
        )


    positive_weight = (
        sample_weight[
            y_true == 1
        ].sum()
    )


    negative_weight = (
        sample_weight[
            y_true == 0
        ].sum()
    )


    # --------------------------------------------------------
    # bootstrap replicate 中必須仍有 positive + negative
    # --------------------------------------------------------

    if (
        positive_weight <= 0
        or
        negative_weight <= 0
    ):

        return None


    accuracy = accuracy_score(
        y_true,
        y_pred,
        sample_weight=sample_weight
    )


    precision = precision_score(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        zero_division=0
    )


    recall = recall_score(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        zero_division=0
    )


    f1 = f1_score(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        zero_division=0
    )


    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred,
            sample_weight=sample_weight
        )
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
            float(
                balanced_accuracy
            ),
    }


# ============================================================
# 12. Macro Metrics
#
# 每個 color 先獨立算 metrics，
# 再對五色取平均。
# ============================================================

def calculate_macro_metrics(
    dataframe,
    weight_column=None,
):

    color_results = []


    for color in EVALUABLE_COLORS:

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
            calculate_metrics(
                subset[
                    "true"
                ].to_numpy(),

                subset[
                    "pred"
                ].to_numpy(),

                sample_weight=
                    weights
            )
        )


        # ----------------------------------------------------
        # 若某 bootstrap replicate 中某個顏色
        # 完全沒有陽性或陰性，
        # 整次 replicate 不使用。
        # ----------------------------------------------------

        if result is None:

            return None


        color_results.append(
            result
        )


    color_df = pd.DataFrame(
        color_results
    )


    return {

        "accuracy":
            color_df[
                "accuracy"
            ].mean(),

        "precision":
            color_df[
                "precision"
            ].mean(),

        "recall":
            color_df[
                "recall"
            ].mean(),

        "f1":
            color_df[
                "f1"
            ].mean(),

        "balanced_accuracy":
            color_df[
                "balanced_accuracy"
            ].mean(),
    }


# ============================================================
# 13. Bootstrap CI
# ============================================================

def percentile_ci(
    values
):

    values = np.asarray(
        values,
        dtype=float
    )


    values = values[
        ~np.isnan(
            values
        )
    ]


    if len(
        values
    ) == 0:

        return (
            np.nan,
            np.nan
        )


    lower = np.percentile(
        values,
        2.5
    )


    upper = np.percentile(
        values,
        97.5
    )


    return (
        float(
            lower
        ),
        float(
            upper
        )
    )


# ============================================================
# 14. 建立病人 Bootstrap Weight
# ============================================================

def generate_patient_weights(
    dataframe,
    rng
):

    patients = (
        dataframe[
            "image_id"
        ]
        .drop_duplicates()
        .tolist()
    )


    n_patients = len(
        patients
    )


    sampled = rng.choice(
        patients,
        size=n_patients,
        replace=True
    )


    patient_counts = (
        pd.Series(
            sampled
        )
        .value_counts()
        .to_dict()
    )


    weights = (
        dataframe[
            "image_id"
        ]
        .map(
            patient_counts
        )
        .fillna(
            0
        )
        .astype(float)
        .to_numpy()
    )


    return weights


# ============================================================
# 15. 六種 Model × Segmentation 的 point estimate + CI
# ============================================================

summary_records = []

bootstrap_detail_records = []


for segmentation_method in SEGMENTATION_METHODS:

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
            .reset_index(
                drop=True
            )
        )


        if len(
            subset
        ) == 0:

            continue


        point_metrics = (
            calculate_macro_metrics(
                subset
            )
        )


        if point_metrics is None:

            raise ValueError(
                f"{segmentation_method} / {model} "
                "無法計算完整五色 Macro Metrics"
            )


        rng = np.random.default_rng(
            SEED
        )


        bootstrap_results = []


        for bootstrap_id in range(
            BOOTSTRAP_ITERATIONS
        ):

            weights = (
                generate_patient_weights(
                    subset,
                    rng
                )
            )


            subset[
                "_bootstrap_weight"
            ] = weights


            result = (
                calculate_macro_metrics(
                    subset,
                    weight_column=
                        "_bootstrap_weight"
                )
            )


            if result is None:

                continue


            result[
                "bootstrap_id"
            ] = bootstrap_id


            bootstrap_results.append(
                result
            )


        bootstrap_df = pd.DataFrame(
            bootstrap_results
        )


        if bootstrap_df.empty:

            raise ValueError(
                f"{segmentation_method} / {model} "
                "沒有有效 bootstrap replicate"
            )


        accuracy_ci = (
            percentile_ci(
                bootstrap_df[
                    "accuracy"
                ]
            )
        )


        precision_ci = (
            percentile_ci(
                bootstrap_df[
                    "precision"
                ]
            )
        )


        recall_ci = (
            percentile_ci(
                bootstrap_df[
                    "recall"
                ]
            )
        )


        f1_ci = (
            percentile_ci(
                bootstrap_df[
                    "f1"
                ]
            )
        )


        ba_ci = (
            percentile_ci(
                bootstrap_df[
                    "balanced_accuracy"
                ]
            )
        )


        summary_records.append({

            "segmentation_method":
                segmentation_method,

            "model":
                model,

            "n_patients":
                subset[
                    "image_id"
                ].nunique(),

            "n_colors":
                len(
                    EVALUABLE_COLORS
                ),

            "n_predictions":
                len(
                    subset
                ),

            "bootstrap_iterations_requested":
                BOOTSTRAP_ITERATIONS,

            "bootstrap_iterations_valid":
                len(
                    bootstrap_df
                ),

            "accuracy":
                point_metrics[
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

            "precision":
                point_metrics[
                    "precision"
                ],

            "precision_ci_lower":
                precision_ci[
                    0
                ],

            "precision_ci_upper":
                precision_ci[
                    1
                ],

            "recall":
                point_metrics[
                    "recall"
                ],

            "recall_ci_lower":
                recall_ci[
                    0
                ],

            "recall_ci_upper":
                recall_ci[
                    1
                ],

            "f1":
                point_metrics[
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
                point_metrics[
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


        # ----------------------------------------------------
        # 儲存 bootstrap replicate
        # ----------------------------------------------------

        bootstrap_df[
            "segmentation_method"
        ] = segmentation_method


        bootstrap_df[
            "model"
        ] = model


        bootstrap_detail_records.append(
            bootstrap_df
        )


        print(
            f"{segmentation_method} | "
            f"{model} | "
            f"F1={point_metrics['f1']:.4f} "
            f"[{f1_ci[0]:.4f}, {f1_ci[1]:.4f}] | "
            f"BA={point_metrics['balanced_accuracy']:.4f} "
            f"[{ba_ci[0]:.4f}, {ba_ci[1]:.4f}]"
        )


# ============================================================
# 16. Bootstrap CI Summary
# ============================================================

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


summary_output = (
    OUTPUT_DIR
    /
    "final_model_bootstrap_ci.csv"
)


summary_df.to_csv(
    summary_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 17. 儲存 Bootstrap 詳細 Replicates
# ============================================================

bootstrap_detail_df = pd.concat(
    bootstrap_detail_records,
    ignore_index=True
)


bootstrap_detail_output = (
    OUTPUT_DIR
    /
    "final_model_bootstrap_replicates.csv"
)


bootstrap_detail_df.to_csv(
    bootstrap_detail_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 18. Paired Model Comparison
#
# 固定同一 segmentation，
# 比較：
#
# SVM - Random Forest
# SVM - CNN
# Random Forest - CNN
#
# Delta = Model A - Model B
# ============================================================

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


paired_records = []

paired_detail_records = []


for segmentation_method in SEGMENTATION_METHODS:

    for model_a, model_b in MODEL_PAIRS:

        a = (
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
                    model_a
                )
            ][
                [
                    "image_id",
                    "region",
                    "color",
                    "true",
                    "pred"
                ]
            ]
            .rename(
                columns={
                    "pred":
                        "pred_a"
                }
            )
        )


        b = (
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
                    model_b
                )
            ][
                [
                    "image_id",
                    "region",
                    "color",
                    "true",
                    "pred"
                ]
            ]
            .rename(
                columns={
                    "pred":
                        "pred_b"
                }
            )
        )


        paired = (
            a.merge(
                b,
                on=[
                    "image_id",
                    "region",
                    "color",
                    "true",
                ],
                how="inner",
                validate="one_to_one"
            )
            .reset_index(
                drop=True
            )
        )


        if (
            len(
                paired
            )
            !=
            len(
                a
            )
            or
            len(
                paired
            )
            !=
            len(
                b
            )
        ):

            raise ValueError(
                f"{segmentation_method}: "
                f"{model_a} vs {model_b} "
                "使用的測試樣本並不完全相同"
            )


        # ----------------------------------------------------
        # 建立兩份相同 sample 的 DataFrame
        # ----------------------------------------------------

        paired_a = (
            paired[
                [
                    "image_id",
                    "region",
                    "color",
                    "true",
                    "pred_a"
                ]
            ]
            .rename(
                columns={
                    "pred_a":
                        "pred"
                }
            )
            .copy()
        )


        paired_b = (
            paired[
                [
                    "image_id",
                    "region",
                    "color",
                    "true",
                    "pred_b"
                ]
            ]
            .rename(
                columns={
                    "pred_b":
                        "pred"
                }
            )
            .copy()
        )


        point_a = (
            calculate_macro_metrics(
                paired_a
            )
        )


        point_b = (
            calculate_macro_metrics(
                paired_b
            )
        )


        point_delta_f1 = (
            point_a[
                "f1"
            ]
            -
            point_b[
                "f1"
            ]
        )


        point_delta_ba = (
            point_a[
                "balanced_accuracy"
            ]
            -
            point_b[
                "balanced_accuracy"
            ]
        )


        point_delta_accuracy = (
            point_a[
                "accuracy"
            ]
            -
            point_b[
                "accuracy"
            ]
        )


        # ====================================================
        # Paired patient bootstrap
        #
        # 同一次 bootstrap：
        # A、B 使用完全相同的病人抽樣。
        # ====================================================

        patients = (
            paired[
                "image_id"
            ]
            .drop_duplicates()
            .tolist()
        )


        n_patients = len(
            patients
        )


        rng = np.random.default_rng(
            SEED
        )


        bootstrap_deltas = []


        for bootstrap_id in range(
            BOOTSTRAP_ITERATIONS
        ):

            sampled = rng.choice(
                patients,
                size=n_patients,
                replace=True
            )


            patient_counts = (
                pd.Series(
                    sampled
                )
                .value_counts()
                .to_dict()
            )


            weights = (
                paired[
                    "image_id"
                ]
                .map(
                    patient_counts
                )
                .fillna(
                    0
                )
                .astype(float)
                .to_numpy()
            )


            paired_a[
                "_bootstrap_weight"
            ] = weights


            paired_b[
                "_bootstrap_weight"
            ] = weights


            result_a = (
                calculate_macro_metrics(
                    paired_a,
                    weight_column=
                        "_bootstrap_weight"
                )
            )


            result_b = (
                calculate_macro_metrics(
                    paired_b,
                    weight_column=
                        "_bootstrap_weight"
                )
            )


            if (
                result_a is None
                or
                result_b is None
            ):

                continue


            bootstrap_deltas.append({

                "bootstrap_id":
                    bootstrap_id,

                "delta_accuracy":
                    (
                        result_a[
                            "accuracy"
                        ]
                        -
                        result_b[
                            "accuracy"
                        ]
                    ),

                "delta_f1":
                    (
                        result_a[
                            "f1"
                        ]
                        -
                        result_b[
                            "f1"
                        ]
                    ),

                "delta_balanced_accuracy":
                    (
                        result_a[
                            "balanced_accuracy"
                        ]
                        -
                        result_b[
                            "balanced_accuracy"
                        ]
                    ),
            })


        delta_df = pd.DataFrame(
            bootstrap_deltas
        )


        f1_ci = (
            percentile_ci(
                delta_df[
                    "delta_f1"
                ]
            )
        )


        ba_ci = (
            percentile_ci(
                delta_df[
                    "delta_balanced_accuracy"
                ]
            )
        )


        accuracy_ci = (
            percentile_ci(
                delta_df[
                    "delta_accuracy"
                ]
            )
        )


        # ====================================================
        # CI 是否跨 0
        # ====================================================

        if (
            f1_ci[0] > 0
            and
            f1_ci[1] > 0
        ):

            f1_direction = (
                f"{model_a}_higher"
            )

        elif (
            f1_ci[0] < 0
            and
            f1_ci[1] < 0
        ):

            f1_direction = (
                f"{model_b}_higher"
            )

        else:

            f1_direction = (
                "CI_crosses_zero"
            )


        paired_records.append({

            "segmentation_method":
                segmentation_method,

            "model_a":
                model_a,

            "model_b":
                model_b,

            "delta_definition":
                "model_a_minus_model_b",

            "n_patients":
                n_patients,

            "n_predictions":
                len(
                    paired
                ),

            "bootstrap_iterations_requested":
                BOOTSTRAP_ITERATIONS,

            "bootstrap_iterations_valid":
                len(
                    delta_df
                ),

            "delta_accuracy":
                point_delta_accuracy,

            "delta_accuracy_ci_lower":
                accuracy_ci[
                    0
                ],

            "delta_accuracy_ci_upper":
                accuracy_ci[
                    1
                ],

            "delta_f1":
                point_delta_f1,

            "delta_f1_ci_lower":
                f1_ci[
                    0
                ],

            "delta_f1_ci_upper":
                f1_ci[
                    1
                ],

            "delta_balanced_accuracy":
                point_delta_ba,

            "delta_balanced_accuracy_ci_lower":
                ba_ci[
                    0
                ],

            "delta_balanced_accuracy_ci_upper":
                ba_ci[
                    1
                ],

            "f1_bootstrap_interpretation":
                f1_direction,
        })


        delta_df[
            "segmentation_method"
        ] = segmentation_method


        delta_df[
            "model_a"
        ] = model_a


        delta_df[
            "model_b"
        ] = model_b


        paired_detail_records.append(
            delta_df
        )


# ============================================================
# 19. 輸出 Paired Comparison
# ============================================================

paired_df = pd.DataFrame(
    paired_records
)


paired_output = (
    OUTPUT_DIR
    /
    "final_model_paired_bootstrap_comparison.csv"
)


paired_df.to_csv(
    paired_output,
    index=False,
    encoding="utf-8-sig"
)


paired_detail_df = pd.concat(
    paired_detail_records,
    ignore_index=True
)


paired_detail_output = (
    OUTPUT_DIR
    /
    "final_model_paired_bootstrap_replicates.csv"
)


paired_detail_df.to_csv(
    paired_detail_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 20. Terminal：
# 六模型正式 CI
# ============================================================

print("\n")
print("=" * 90)
print("六種模型組合：Patient-level Bootstrap 95% CI")
print("=" * 90)


display_summary = (
    summary_df[
        [
            "segmentation_method",
            "model",
            "n_patients",
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
# 21. Terminal：
# 模型 Pairwise Comparison
# ============================================================

print("\n")
print("=" * 90)
print("固定分區下的模型 Paired Bootstrap Comparison")
print("=" * 90)

print(
    "Delta = Model A - Model B"
)


display_paired = (
    paired_df[
        [
            "segmentation_method",
            "model_a",
            "model_b",
            "delta_f1",
            "delta_f1_ci_lower",
            "delta_f1_ci_upper",
            "delta_balanced_accuracy",
            "delta_balanced_accuracy_ci_lower",
            "delta_balanced_accuracy_ci_upper",
            "f1_bootstrap_interpretation",
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

    display_paired[
        column
    ] = (
        display_paired[
            column
        ]
        .round(
            4
        )
    )


print(
    display_paired.to_string(
        index=False
    )
)


# ============================================================
# 22. 解讀說明
# ============================================================

print("\n")
print("=" * 90)
print("解讀")
print("=" * 90)


print(
    "Bootstrap 單位：image_id（病人），"
    "同一病人的所有舌區一起重新抽樣。"
)


print(
    "95% CI 使用 percentile bootstrap。"
)


print(
    "模型比較 Delta = Model A - Model B。"
)


print(
    "若 Delta F1 的 95% CI 全部 > 0，"
    "表示目前 bootstrap 下 Model A 表現穩定較高。"
)


print(
    "若 95% CI 跨過 0，"
    "則目前資料不足以顯示穩定模型差異。"
)


print(
    "此分析仍應搭配點估計、class imbalance "
    "與各舌色結果一起解讀。"
)


# ============================================================
# 23. 輸出檔案
# ============================================================

print("\n")
print("=" * 90)
print("輸出檔案")
print("=" * 90)


print(
    summary_output
)

print(
    paired_output
)

print(
    bootstrap_detail_output
)

print(
    paired_detail_output
)

print("=" * 90)
