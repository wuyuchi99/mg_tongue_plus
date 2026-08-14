from pathlib import Path
from math import comb

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
BOOTSTRAP_ITERATIONS = 1000


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
# 3. 要比較的兩種分區方法
#
# 所有差值都定義成：
#
# 1:3:1 - 1:2:2
#
# 所以：
# delta > 0 → 1:3:1 較好
# delta < 0 → 1:2:2 較好
# ============================================================

METHOD_A = "tip_middle_root_1_2_2"
METHOD_B = "tip_middle_root_1_3_1"


METHOD_A_SHORT = "1:2:2"
METHOD_B_SHORT = "1:3:1"


# ============================================================
# 4. 模型與顏色
# ============================================================

MODELS = [
    "SVM",
    "Random Forest",
    "CNN",
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
# 5. 檢查檔案
# ============================================================

for file_path in [
    SVM_RF_FILE,
    CNN_FILE
]:

    if not file_path.exists():

        raise FileNotFoundError(
            f"找不到：{file_path}"
        )


# ============================================================
# 6. 讀取 SVM / Random Forest Predictions
# ============================================================

svm_rf = pd.read_csv(
    SVM_RF_FILE,
    dtype={
        "image_id": str
    }
)


# ============================================================
# 7. 讀取 CNN Predictions
# ============================================================

cnn = pd.read_csv(
    CNN_FILE,
    dtype={
        "image_id": str
    }
)


# ============================================================
# 8. 統一必要欄位
# ============================================================

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
    ("CNN", cnn)
]:

    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:

        raise ValueError(
            f"{name} prediction 檔缺少欄位："
            f"{missing}"
        )


svm_rf = svm_rf[
    required_columns
].copy()


cnn = cnn[
    required_columns
].copy()


# ============================================================
# 9. 合併三種模型 Predictions
# ============================================================

predictions = pd.concat(
    [
        svm_rf,
        cnn
    ],
    ignore_index=True
)


# ============================================================
# 10. 標準化資料型別
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


predictions["region"] = (
    predictions["region"]
    .astype(str)
    .str.strip()
)


predictions["color"] = (
    predictions["color"]
    .astype(str)
    .str.strip()
)


predictions["model"] = (
    predictions["model"]
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
# 11. 建立 sample key
# ============================================================

predictions["sample_key"] = (
    predictions["image_id"]
    +
    "__"
    +
    predictions["region"]
)


print("=" * 85)
print("1:2:2 vs 1:3:1 Paired Comparison")
print("=" * 85)

print(
    "Prediction rows：",
    len(predictions)
)

print(
    "Models：",
    sorted(
        predictions["model"]
        .unique()
        .tolist()
    )
)


# ============================================================
# 12. Metrics 函數
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


    accuracy = accuracy_score(
        y_true,
        y_pred
    )


    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )


    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )


    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )


    # Balanced Accuracy 需要兩類都有
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
            )
            if not np.isnan(
                balanced_accuracy
            )
            else np.nan,
    }


# ============================================================
# 13. McNemar Exact Test
#
# 比較兩種分區對「同一筆測試資料」
# 是否具有不同正確率。
#
# b：
# 1:2:2 對、1:3:1 錯
#
# c：
# 1:2:2 錯、1:3:1 對
#
# H0：
# b 與 c 的機率相同。
#
# 使用 exact binomial test。
# ============================================================

def mcnemar_exact_pvalue(
    b,
    c
):

    n = (
        b
        +
        c
    )


    if n == 0:

        return 1.0


    smaller = min(
        b,
        c
    )


    cumulative_probability = 0.0


    for k in range(
        smaller + 1
    ):

        cumulative_probability += (

            comb(
                n,
                k
            )

            *

            (
                0.5 ** n
            )
        )


    p_value = min(
        1.0,
        2.0
        *
        cumulative_probability
    )


    return float(
        p_value
    )


# ============================================================
# 14. Patient-level Bootstrap
#
# 重要：
# 不是抽 region，
# 而是抽 image_id。
#
# 同一病人的所有區域一起被抽進來。
# ============================================================

def patient_bootstrap(
    paired_df,
    iterations=1000,
    seed=42
):

    rng = np.random.default_rng(
        seed
    )


    patients = (
        paired_df[
            "image_id"
        ]
        .unique()
        .tolist()
    )


    n_patients = len(
        patients
    )


    bootstrap_records = []


    for bootstrap_id in range(
        iterations
    ):

        sampled_patients = rng.choice(
            patients,
            size=n_patients,
            replace=True
        )


        sampled_parts = []


        # ----------------------------------------------------
        # 每一次抽到某病人，就把該病人的所有可用區域加入
        #
        # 如果同一病人被抽到兩次，
        # 就加入兩次。
        # ----------------------------------------------------

        for patient_index, patient_id in enumerate(
            sampled_patients
        ):

            patient_rows = (
                paired_df[
                    paired_df[
                        "image_id"
                    ]
                    ==
                    patient_id
                ]
                .copy()
            )


            # 建立 bootstrap patient id，
            # 避免同一病人重複抽樣時被視為同一列
            patient_rows[
                "bootstrap_patient"
            ] = (
                f"{patient_id}"
                f"__{patient_index}"
            )


            sampled_parts.append(
                patient_rows
            )


        bootstrap_df = pd.concat(
            sampled_parts,
            ignore_index=True
        )


        y_true = (
            bootstrap_df[
                "true"
            ]
            .to_numpy()
        )


        # ----------------------------------------------------
        # 如果這次 bootstrap 剛好沒有兩種類別，
        # 不計算這次。
        #
        # 對淡白這種陽性只有 5 個的類別特別重要。
        # ----------------------------------------------------

        if len(
            np.unique(
                y_true
            )
        ) < 2:

            continue


        pred_a = (
            bootstrap_df[
                "pred_a"
            ]
            .to_numpy()
        )


        pred_b = (
            bootstrap_df[
                "pred_b"
            ]
            .to_numpy()
        )


        metrics_a = calculate_metrics(
            y_true,
            pred_a
        )


        metrics_b = calculate_metrics(
            y_true,
            pred_b
        )


        bootstrap_records.append({

            "delta_accuracy":
                metrics_b[
                    "accuracy"
                ]
                -
                metrics_a[
                    "accuracy"
                ],

            "delta_f1":
                metrics_b[
                    "f1"
                ]
                -
                metrics_a[
                    "f1"
                ],

            "delta_balanced_accuracy":
                metrics_b[
                    "balanced_accuracy"
                ]
                -
                metrics_a[
                    "balanced_accuracy"
                ],
        })


    bootstrap_df = pd.DataFrame(
        bootstrap_records
    )


    return bootstrap_df


# ============================================================
# 15. Bootstrap CI
# ============================================================

def bootstrap_ci(
    bootstrap_df,
    column
):

    if (
        bootstrap_df.empty
        or
        column not in bootstrap_df.columns
    ):

        return (
            np.nan,
            np.nan
        )


    values = (
        bootstrap_df[
            column
        ]
        .dropna()
        .to_numpy()
    )


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
        float(lower),
        float(upper)
    )


# ============================================================
# 16. 結果容器
# ============================================================

comparison_records = []

bootstrap_all_records = []


# ============================================================
# 17. 三個模型 × 六種舌色逐一比較
# ============================================================

for model in MODELS:

    print("\n")
    print("=" * 85)
    print(
        "Model：",
        model
    )
    print("=" * 85)


    for color in COLORS:

        model_color = (
            predictions[
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
            ]
            .copy()
        )


        # ----------------------------------------------------
        # 如果這個顏色完全沒有模型結果
        # 例如灰黑
        # ----------------------------------------------------

        if len(
            model_color
        ) == 0:

            print(
                color,
                "：無可比較結果"
            )

            continue


        # ====================================================
        # 18. 分開 1:2:2 / 1:3:1
        # ====================================================

        method_a_df = (
            model_color[
                model_color[
                    "segmentation_method"
                ]
                ==
                METHOD_A
            ]
            .copy()
        )


        method_b_df = (
            model_color[
                model_color[
                    "segmentation_method"
                ]
                ==
                METHOD_B
            ]
            .copy()
        )


        if (
            len(
                method_a_df
            ) == 0
            or
            len(
                method_b_df
            ) == 0
        ):

            print(
                color,
                "：其中一種分區沒有結果"
            )

            continue


        # ====================================================
        # 19. 準備 merge
        # ====================================================

        merge_columns = [
            "sample_key",
            "image_id",
            "region",
            "fold",
            "true",
        ]


        a = (
            method_a_df[
                merge_columns
                +
                [
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
            method_b_df[
                merge_columns
                +
                [
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


        # ====================================================
        # 20. Paired Merge
        # ====================================================

        paired = a.merge(
            b,
            on=[
                "sample_key",
                "image_id",
                "region",
                "fold",
                "true",
            ],
            how="inner",
            validate="one_to_one"
        )


        # ----------------------------------------------------
        # 確認沒有樣本遺失
        # ----------------------------------------------------

        if (
            len(
                paired
            )
            !=
            len(
                method_a_df
            )
            or
            len(
                paired
            )
            !=
            len(
                method_b_df
            )
        ):

            raise ValueError(
                f"{model} / {color}："
                "1:2:2 與 1:3:1 "
                "測試樣本不完全一致"
            )


        # ====================================================
        # 21. 基本效能
        # ====================================================

        y_true = (
            paired[
                "true"
            ]
            .to_numpy()
        )


        pred_a = (
            paired[
                "pred_a"
            ]
            .to_numpy()
        )


        pred_b = (
            paired[
                "pred_b"
            ]
            .to_numpy()
        )


        metrics_a = (
            calculate_metrics(
                y_true,
                pred_a
            )
        )


        metrics_b = (
            calculate_metrics(
                y_true,
                pred_b
            )
        )


        # ====================================================
        # 22. 每筆是否判對
        # ====================================================

        correct_a = (
            pred_a
            ==
            y_true
        )


        correct_b = (
            pred_b
            ==
            y_true
        )


        both_correct = int(
            np.sum(
                correct_a
                &
                correct_b
            )
        )


        only_a_correct = int(
            np.sum(
                correct_a
                &
                ~correct_b
            )
        )


        only_b_correct = int(
            np.sum(
                ~correct_a
                &
                correct_b
            )
        )


        both_wrong = int(
            np.sum(
                ~correct_a
                &
                ~correct_b
            )
        )


        # ====================================================
        # 23. McNemar Exact Test
        # ====================================================

        mcnemar_p = (
            mcnemar_exact_pvalue(
                only_a_correct,
                only_b_correct
            )
        )


        # ====================================================
        # 24. Patient-level Bootstrap
        # ====================================================

        bootstrap_df = (
            patient_bootstrap(
                paired,
                iterations=
                    BOOTSTRAP_ITERATIONS,
                seed=
                    SEED
            )
        )


        delta_accuracy_ci = (
            bootstrap_ci(
                bootstrap_df,
                "delta_accuracy"
            )
        )


        delta_f1_ci = (
            bootstrap_ci(
                bootstrap_df,
                "delta_f1"
            )
        )


        delta_ba_ci = (
            bootstrap_ci(
                bootstrap_df,
                "delta_balanced_accuracy"
            )
        )


        # ====================================================
        # 25. Bootstrap 詳細資料
        # ====================================================

        if not bootstrap_df.empty:

            bootstrap_df[
                "model"
            ] = model


            bootstrap_df[
                "color"
            ] = color


            bootstrap_df[
                "method_a"
            ] = METHOD_A


            bootstrap_df[
                "method_b"
            ] = METHOD_B


            bootstrap_all_records.append(
                bootstrap_df
            )


        # ====================================================
        # 26. 判斷 F1 哪種分區較高
        # ====================================================

        delta_f1 = (
            metrics_b[
                "f1"
            ]
            -
            metrics_a[
                "f1"
            ]
        )


        if np.isclose(
            delta_f1,
            0
        ):

            f1_winner = (
                "tie"
            )


        elif delta_f1 > 0:

            f1_winner = (
                METHOD_B_SHORT
            )


        else:

            f1_winner = (
                METHOD_A_SHORT
            )


        # ====================================================
        # 27. 紀錄結果
        # ====================================================

        comparison_records.append({

            "model":
                model,

            "color":
                color,

            "n_samples":
                len(
                    paired
                ),

            "n_patients":
                paired[
                    "image_id"
                ].nunique(),

            # ------------------------------------------------
            # 1:2:2
            # ------------------------------------------------

            "accuracy_1_2_2":
                metrics_a[
                    "accuracy"
                ],

            "precision_1_2_2":
                metrics_a[
                    "precision"
                ],

            "recall_1_2_2":
                metrics_a[
                    "recall"
                ],

            "f1_1_2_2":
                metrics_a[
                    "f1"
                ],

            "balanced_accuracy_1_2_2":
                metrics_a[
                    "balanced_accuracy"
                ],

            # ------------------------------------------------
            # 1:3:1
            # ------------------------------------------------

            "accuracy_1_3_1":
                metrics_b[
                    "accuracy"
                ],

            "precision_1_3_1":
                metrics_b[
                    "precision"
                ],

            "recall_1_3_1":
                metrics_b[
                    "recall"
                ],

            "f1_1_3_1":
                metrics_b[
                    "f1"
                ],

            "balanced_accuracy_1_3_1":
                metrics_b[
                    "balanced_accuracy"
                ],

            # ------------------------------------------------
            # Delta = 1:3:1 - 1:2:2
            # ------------------------------------------------

            "delta_accuracy":
                metrics_b[
                    "accuracy"
                ]
                -
                metrics_a[
                    "accuracy"
                ],

            "delta_f1":
                delta_f1,

            "delta_balanced_accuracy":
                metrics_b[
                    "balanced_accuracy"
                ]
                -
                metrics_a[
                    "balanced_accuracy"
                ],

            # ------------------------------------------------
            # Paired correctness
            # ------------------------------------------------

            "both_correct":
                both_correct,

            "only_1_2_2_correct":
                only_a_correct,

            "only_1_3_1_correct":
                only_b_correct,

            "both_wrong":
                both_wrong,

            # ------------------------------------------------
            # McNemar
            # ------------------------------------------------

            "mcnemar_exact_p":
                mcnemar_p,

            # ------------------------------------------------
            # Patient bootstrap
            # ------------------------------------------------

            "bootstrap_valid_iterations":
                len(
                    bootstrap_df
                ),

            "delta_accuracy_ci_lower":
                delta_accuracy_ci[
                    0
                ],

            "delta_accuracy_ci_upper":
                delta_accuracy_ci[
                    1
                ],

            "delta_f1_ci_lower":
                delta_f1_ci[
                    0
                ],

            "delta_f1_ci_upper":
                delta_f1_ci[
                    1
                ],

            "delta_balanced_accuracy_ci_lower":
                delta_ba_ci[
                    0
                ],

            "delta_balanced_accuracy_ci_upper":
                delta_ba_ci[
                    1
                ],

            "f1_winner":
                f1_winner,
        })


        print(
            f"{color}: "
            f"F1 "
            f"1:2:2={metrics_a['f1']:.4f}, "
            f"1:3:1={metrics_b['f1']:.4f}, "
            f"Δ={delta_f1:+.4f}, "
            f"McNemar p={mcnemar_p:.4f}"
        )


# ============================================================
# 28. 建立比較 DataFrame
# ============================================================

comparison_df = pd.DataFrame(
    comparison_records
)


# ============================================================
# 29. 輸出完整比較結果
# ============================================================

comparison_output = (
    OUTPUT_DIR
    /
    "segmentation_paired_comparison_by_model_color.csv"
)


comparison_df.to_csv(
    comparison_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 30. 儲存 Bootstrap 詳細資料
# ============================================================

if bootstrap_all_records:

    bootstrap_all_df = pd.concat(
        bootstrap_all_records,
        ignore_index=True
    )


else:

    bootstrap_all_df = pd.DataFrame()


bootstrap_output = (
    OUTPUT_DIR
    /
    "segmentation_patient_bootstrap_details.csv"
)


bootstrap_all_df.to_csv(
    bootstrap_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 31. 每個模型 Macro Average
#
# 灰黑因沒有模型結果，
# 自動不會進入平均。
# ============================================================

model_summary_records = []


for model in MODELS:

    subset = (
        comparison_df[
            comparison_df[
                "model"
            ]
            ==
            model
        ]
        .copy()
    )


    if len(
        subset
    ) == 0:

        continue


    model_summary_records.append({

        "model":
            model,

        "n_colors":
            len(
                subset
            ),

        "macro_accuracy_1_2_2":
            subset[
                "accuracy_1_2_2"
            ].mean(),

        "macro_accuracy_1_3_1":
            subset[
                "accuracy_1_3_1"
            ].mean(),

        "macro_delta_accuracy":
            subset[
                "delta_accuracy"
            ].mean(),

        "macro_f1_1_2_2":
            subset[
                "f1_1_2_2"
            ].mean(),

        "macro_f1_1_3_1":
            subset[
                "f1_1_3_1"
            ].mean(),

        "macro_delta_f1":
            subset[
                "delta_f1"
            ].mean(),

        "macro_balanced_accuracy_1_2_2":
            subset[
                "balanced_accuracy_1_2_2"
            ].mean(),

        "macro_balanced_accuracy_1_3_1":
            subset[
                "balanced_accuracy_1_3_1"
            ].mean(),

        "macro_delta_balanced_accuracy":
            subset[
                "delta_balanced_accuracy"
            ].mean(),

        "f1_wins_1_2_2":
            int(
                np.sum(
                    subset[
                        "f1_winner"
                    ]
                    ==
                    METHOD_A_SHORT
                )
            ),

        "f1_wins_1_3_1":
            int(
                np.sum(
                    subset[
                        "f1_winner"
                    ]
                    ==
                    METHOD_B_SHORT
                )
            ),

        "f1_ties":
            int(
                np.sum(
                    subset[
                        "f1_winner"
                    ]
                    ==
                    "tie"
                )
            ),
    })


model_summary_df = pd.DataFrame(
    model_summary_records
)


model_summary_output = (
    OUTPUT_DIR
    /
    "segmentation_paired_comparison_model_summary.csv"
)


model_summary_df.to_csv(
    model_summary_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 32. 顯示每個 Model × Color 的重點結果
# ============================================================

print("\n")
print("=" * 85)
print("分區方法 Paired Comparison 結果")
print("=" * 85)


display_columns = [
    "model",
    "color",
    "n_samples",
    "f1_1_2_2",
    "f1_1_3_1",
    "delta_f1",
    "delta_f1_ci_lower",
    "delta_f1_ci_upper",
    "mcnemar_exact_p",
    "f1_winner",
]


display_df = (
    comparison_df[
        display_columns
    ]
    .copy()
)


numeric_columns = [
    "f1_1_2_2",
    "f1_1_3_1",
    "delta_f1",
    "delta_f1_ci_lower",
    "delta_f1_ci_upper",
    "mcnemar_exact_p",
]


for column in numeric_columns:

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
# 33. 顯示模型層級 Macro Summary
# ============================================================

print("\n")
print("=" * 85)
print("模型層級 Macro Comparison")
print("=" * 85)


summary_display = (
    model_summary_df.copy()
)


summary_numeric = [
    "macro_accuracy_1_2_2",
    "macro_accuracy_1_3_1",
    "macro_delta_accuracy",
    "macro_f1_1_2_2",
    "macro_f1_1_3_1",
    "macro_delta_f1",
    "macro_balanced_accuracy_1_2_2",
    "macro_balanced_accuracy_1_3_1",
    "macro_delta_balanced_accuracy",
]


for column in summary_numeric:

    summary_display[
        column
    ] = (
        summary_display[
            column
        ]
        .round(
            4
        )
    )


print(
    summary_display.to_string(
        index=False
    )
)


# ============================================================
# 34. 提醒如何解讀 CI
# ============================================================

print("\n")
print("=" * 85)
print("解讀方式")
print("=" * 85)

print(
    "所有 Delta = 1:3:1 - 1:2:2"
)

print(
    "Delta > 0：1:3:1 較高"
)

print(
    "Delta < 0：1:2:2 較高"
)

print(
    "若 95% bootstrap CI 跨過 0，"
    "代表目前資料不足以顯示穩定方向。"
)

print(
    "McNemar p 值比較的是兩種分區在"
    "同一批樣本上的正確/錯誤差異。"
)

print(
    "目前樣本數仍小，且同時比較多個"
    "model/color，因此 p 值應視為探索性結果。"
)


# ============================================================
# 35. 輸出檔案
# ============================================================

print("\n")
print("=" * 85)
print("輸出檔案")
print("=" * 85)


print(
    comparison_output
)

print(
    model_summary_output
)

print(
    bootstrap_output
)


print("=" * 85)
