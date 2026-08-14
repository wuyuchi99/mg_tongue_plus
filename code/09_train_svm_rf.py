from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix
)


# ============================================================
# 1. 專案路徑
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"


# ============================================================
# 2. 兩種舌區分法
# ============================================================

SEGMENTATION_METHODS = [
    "tip_middle_root_1_2_2",
    "tip_middle_root_1_3_1",
]


# ============================================================
# 3. 六種舌質色
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
# 4. 讀取兩種方法的 Dataset
# ============================================================

datasets = {}

for method_name in SEGMENTATION_METHODS:

    dataset_path = (
        OUTPUT_DIR
        / method_name
        / "ml_dataset.csv"
    )

    if not dataset_path.exists():

        raise FileNotFoundError(
            f"找不到：{dataset_path}\n"
            "請先執行 08_build_ml_dataset.py"
        )

    df = pd.read_csv(
        dataset_path,
        dtype={"image_id": str}
    )

    df["image_id"] = (
        df["image_id"]
        .astype(str)
        .str.strip()
    )

    df["region"] = (
        df["region"]
        .astype(str)
        .str.strip()
    )

    # 建立唯一 Key
    df["sample_key"] = (
        df["image_id"]
        + "__"
        + df["region"]
    )

    datasets[method_name] = df


# ============================================================
# 5. 確認兩種分區資料完全對應
# ============================================================

reference_method = SEGMENTATION_METHODS[0]

reference_keys = set(
    datasets[
        reference_method
    ]["sample_key"]
)


for method_name in SEGMENTATION_METHODS[1:]:

    keys = set(
        datasets[
            method_name
        ]["sample_key"]
    )

    if keys != reference_keys:

        raise ValueError(
            f"{method_name} 與 "
            f"{reference_method} 的病例/區域不一致"
        )


print("=" * 75)
print("兩種分區 × SVM / Random Forest")
print("=" * 75)

for method_name, df in datasets.items():

    print(
        method_name,
        "資料列：",
        len(df),
        "照片：",
        df["image_id"].nunique()
    )

print("✓ 兩種分區的病例與舌區完全一致")


# ============================================================
# 6. 找特徵欄位
# ============================================================

target_columns = [
    f"y_{color}"
    for color in COLORS
]

exclude_columns = [
    "segmentation_method",
    "image_id",
    "region",
    "sample_key",
    *target_columns
]


feature_columns = [
    column
    for column in datasets[
        reference_method
    ].columns
    if column not in exclude_columns
]


print(
    "影像特徵數：",
    len(feature_columns)
)


# ============================================================
# 7. 建立模型
# ============================================================

def build_models():

    # SVM
    svm_model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced"
            )
        )
    ])


    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )


    return {
        "SVM": svm_model,
        "Random Forest": rf_model
    }


# ============================================================
# 8. 結果容器
# ============================================================

fold_results = []
color_results = []
prediction_records = []


# ============================================================
# 9. 六種舌色逐一處理
# ============================================================

for color in COLORS:

    print("\n")
    print("=" * 75)
    print("舌質色：", color)
    print("=" * 75)

    target_column = f"y_{color}"


    # ========================================================
    # 10. 先使用 reference 方法建立共同 CV folds
    # ========================================================

    reference_df = (
        datasets[
            reference_method
        ]
        .copy()
    )


    # 只使用 Ground Truth = 0 或 1
    reference_df = reference_df[
        reference_df[
            target_column
        ].isin(
            [0, 1]
        )
    ].copy()


    # 固定排序
    reference_df = (
        reference_df
        .sort_values(
            "sample_key"
        )
        .reset_index(
            drop=True
        )
    )


    y_reference = (
        reference_df[
            target_column
        ]
        .astype(int)
    )


    groups_reference = (
        reference_df[
            "image_id"
        ]
    )


    positive_count = int(
        (
            y_reference == 1
        ).sum()
    )

    negative_count = int(
        (
            y_reference == 0
        ).sum()
    )

    patient_count = (
        groups_reference.nunique()
    )


    print(
        "可使用區域：",
        len(reference_df)
    )

    print(
        "照片數：",
        patient_count
    )

    print(
        "陽性：",
        positive_count
    )

    print(
        "陰性：",
        negative_count
    )


    # ========================================================
    # 11. 決定 Fold 數
    # ========================================================

    n_splits = min(
        5,
        positive_count,
        negative_count,
        patient_count
    )


    if n_splits < 2:

        print(
            "⚠ 樣本不足，無法做交叉驗證"
        )

        continue


    print(
        "Cross-validation：",
        n_splits,
        "fold"
    )


    # ========================================================
    # 12. 建立共同 folds
    #
    # 同一 image_id 永遠不會同時出現在 Train / Test
    # ========================================================

    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )


    dummy_X = np.zeros(
        (
            len(reference_df),
            1
        )
    )


    splits = list(
        cv.split(
            dummy_X,
            y_reference,
            groups_reference
        )
    )


    # ========================================================
    # 13. 兩種分區方法
    # ========================================================

    for method_name in SEGMENTATION_METHODS:

        print("\n")
        print(
            "分區方法：",
            method_name
        )


        method_df = (
            datasets[
                method_name
            ]
            .copy()
        )


        # 只保留相同 usable samples
        usable_keys = (
            reference_df[
                "sample_key"
            ]
            .tolist()
        )


        method_df = (
            method_df[
                method_df[
                    "sample_key"
                ].isin(
                    usable_keys
                )
            ]
            .copy()
        )


        # 依 Reference 的順序重新排列
        method_df = (
            method_df
            .set_index(
                "sample_key"
            )
            .loc[
                usable_keys
            ]
            .reset_index()
        )


        X = (
            method_df[
                feature_columns
            ]
            .astype(float)
        )


        y = (
            method_df[
                target_column
            ]
            .astype(int)
        )


        groups = (
            method_df[
                "image_id"
            ]
        )


        # ====================================================
        # 14. SVM 與 Random Forest
        # ====================================================

        models = build_models()


        for model_name, model in models.items():

            print(
                "  模型：",
                model_name
            )


            all_true = []
            all_pred = []


            # ================================================
            # 15. 跑共同的 folds
            # ================================================

            for fold_number, (
                train_index,
                test_index
            ) in enumerate(
                splits,
                start=1
            ):


                X_train = X.iloc[
                    train_index
                ]

                X_test = X.iloc[
                    test_index
                ]


                y_train = y.iloc[
                    train_index
                ]

                y_test = y.iloc[
                    test_index
                ]


                train_groups = set(
                    groups.iloc[
                        train_index
                    ]
                )


                test_groups = set(
                    groups.iloc[
                        test_index
                    ]
                )


                # ============================================
                # Data Leakage 安全檢查
                # ============================================

                overlap = (
                    train_groups
                    &
                    test_groups
                )


                if overlap:

                    raise RuntimeError(
                        "發現 Data Leakage！"
                    )


                # ============================================
                # Training Fold 必須有兩類
                # ============================================

                if (
                    y_train.nunique()
                    <
                    2
                ):

                    print(
                        f"    Fold {fold_number} "
                        "只有單一類別，跳過"
                    )

                    continue


                # ============================================
                # 訓練
                # ============================================

                model.fit(
                    X_train,
                    y_train
                )


                # ============================================
                # 預測
                # ============================================

                y_pred = (
                    model.predict(
                        X_test
                    )
                )


                # ============================================
                # Metrics
                # ============================================

                accuracy = (
                    accuracy_score(
                        y_test,
                        y_pred
                    )
                )


                precision = (
                    precision_score(
                        y_test,
                        y_pred,
                        zero_division=0
                    )
                )


                recall = (
                    recall_score(
                        y_test,
                        y_pred,
                        zero_division=0
                    )
                )


                f1 = (
                    f1_score(
                        y_test,
                        y_pred,
                        zero_division=0
                    )
                )


                balanced_accuracy = (
                    balanced_accuracy_score(
                        y_test,
                        y_pred
                    )
                )


                fold_results.append({

                    "segmentation_method":
                        method_name,

                    "model":
                        model_name,

                    "color":
                        color,

                    "fold":
                        fold_number,

                    "train_samples":
                        len(
                            train_index
                        ),

                    "test_samples":
                        len(
                            test_index
                        ),

                    "train_patients":
                        len(
                            train_groups
                        ),

                    "test_patients":
                        len(
                            test_groups
                        ),

                    "accuracy":
                        accuracy,

                    "precision":
                        precision,

                    "recall":
                        recall,

                    "f1":
                        f1,

                    "balanced_accuracy":
                        balanced_accuracy,
                })


                # ============================================
                # 儲存每筆 Prediction
                # ============================================

                test_rows = (
                    method_df.iloc[
                        test_index
                    ]
                    .reset_index(
                        drop=True
                    )
                )


                for i in range(
                    len(
                        test_rows
                    )
                ):

                    prediction_records.append({

                        "segmentation_method":
                            method_name,

                        "model":
                            model_name,

                        "color":
                            color,

                        "fold":
                            fold_number,

                        "image_id":
                            test_rows.loc[
                                i,
                                "image_id"
                            ],

                        "region":
                            test_rows.loc[
                                i,
                                "region"
                            ],

                        "true":
                            int(
                                y_test.iloc[
                                    i
                                ]
                            ),

                        "pred":
                            int(
                                y_pred[
                                    i
                                ]
                            ),
                    })


                all_true.extend(
                    y_test.tolist()
                )

                all_pred.extend(
                    y_pred.tolist()
                )


                print(
                    f"    Fold {fold_number}: "
                    f"Accuracy={accuracy:.3f}, "
                    f"F1={f1:.3f}"
                )


            # =================================================
            # 16. 整體 Out-of-fold 結果
            # =================================================

            if len(
                all_true
            ) == 0:

                continue


            accuracy = (
                accuracy_score(
                    all_true,
                    all_pred
                )
            )


            precision = (
                precision_score(
                    all_true,
                    all_pred,
                    zero_division=0
                )
            )


            recall = (
                recall_score(
                    all_true,
                    all_pred,
                    zero_division=0
                )
            )


            f1 = (
                f1_score(
                    all_true,
                    all_pred,
                    zero_division=0
                )
            )


            balanced_accuracy = (
                balanced_accuracy_score(
                    all_true,
                    all_pred
                )
            )


            tn, fp, fn, tp = (
                confusion_matrix(
                    all_true,
                    all_pred,
                    labels=[
                        0,
                        1
                    ]
                )
                .ravel()
            )


            color_results.append({

                "segmentation_method":
                    method_name,

                "model":
                    model_name,

                "color":
                    color,

                "n_samples":
                    len(
                        all_true
                    ),

                "positive":
                    positive_count,

                "negative":
                    negative_count,

                "folds":
                    n_splits,

                "accuracy":
                    accuracy,

                "precision":
                    precision,

                "recall":
                    recall,

                "f1":
                    f1,

                "balanced_accuracy":
                    balanced_accuracy,

                "TN":
                    int(tn),

                "FP":
                    int(fp),

                "FN":
                    int(fn),

                "TP":
                    int(tp),
            })


            print(
                "    Overall:"
                f" Accuracy={accuracy:.4f},"
                f" F1={f1:.4f},"
                f" BalancedAcc="
                f"{balanced_accuracy:.4f}"
            )


# ============================================================
# 17. 儲存 Fold 結果
# ============================================================

fold_df = pd.DataFrame(
    fold_results
)


fold_output = (
    OUTPUT_DIR
    /
    "svm_rf_all_methods_fold_metrics.csv"
)


fold_df.to_csv(
    fold_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 18. 儲存每個舌色結果
# ============================================================

color_df = pd.DataFrame(
    color_results
)


color_output = (
    OUTPUT_DIR
    /
    "svm_rf_all_methods_color_metrics.csv"
)


color_df.to_csv(
    color_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 19. 儲存 Predictions
# ============================================================

prediction_df = pd.DataFrame(
    prediction_records
)


prediction_output = (
    OUTPUT_DIR
    /
    "svm_rf_all_methods_predictions.csv"
)


prediction_df.to_csv(
    prediction_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 20. 產生四組模型總比較
#
# 六種顏色結果取 Macro Average
# ============================================================

summary = (
    color_df
    .groupby(
        [
            "segmentation_method",
            "model"
        ]
    )[
        [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "balanced_accuracy"
        ]
    ]
    .mean()
    .reset_index()
)


summary_output = (
    OUTPUT_DIR
    /
    "svm_rf_segmentation_comparison.csv"
)


summary.to_csv(
    summary_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 21. 找每組的 F1 排名
# ============================================================

summary = (
    summary
    .sort_values(
        "f1",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


summary[
    "rank_by_f1"
] = (
    np.arange(
        1,
        len(summary) + 1
    )
)


rank_output = (
    OUTPUT_DIR
    /
    "svm_rf_segmentation_ranking.csv"
)


summary.to_csv(
    rank_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 22. 最終顯示
# ============================================================

print("\n")
print("=" * 75)
print("分區方法 × SVM / Random Forest 最終比較")
print("=" * 75)


display_columns = [
    "rank_by_f1",
    "segmentation_method",
    "model",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "balanced_accuracy"
]


print(
    summary[
        display_columns
    ].to_string(
        index=False
    )
)


print("\n輸出：")

print(
    fold_output
)

print(
    color_output
)

print(
    prediction_output
)

print(
    summary_output
)

print(
    rank_output
)

print("=" * 75)