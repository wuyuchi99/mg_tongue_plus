from pathlib import Path

import numpy as np
import pandas as pd

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
    confusion_matrix,
)


# ============================================================
# 1. 基本設定
# ============================================================

SEED = 42

SEGMENTATION_METHODS = [
    "tip_middle_root_1_2_2",
    "tip_middle_root_1_3_1",
]

PRIMARY_COLORS = [
    "淡紅",
    "鮮紅",
]

EXPLORATORY_COLORS = [
    "暗紅",
    "青紫",
]

ANALYSIS_COLORS = (
    PRIMARY_COLORS
    +
    EXPLORATORY_COLORS
)

MODELS = [
    "SVM",
    "Random Forest",
]


# ============================================================
# 2. 路徑
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

VOTE_FILE = (
    OUTPUT_DIR
    / "interrater_vote_details.csv"
)

FOLD_FILE = (
    OUTPUT_DIR
    / "unanimous_disagreement_fold_assignments.csv"
)


# ============================================================
# 3. 自動尋找 07 產生的 color feature CSV
#
# 我們不使用 label 欄位，只需要：
# segmentation_method
# image_id
# region
# RGB / HSV / LAB / histogram features
# ============================================================

def find_feature_file():

    preferred_candidates = [

        OUTPUT_DIR
        / "color_features_all_methods.csv",

        OUTPUT_DIR
        / "all_methods_color_features.csv",

        OUTPUT_DIR
        / "color_features_combined.csv",

        OUTPUT_DIR
        / "all_color_features.csv",
    ]

    for file_path in preferred_candidates:

        if file_path.exists():

            test_df = pd.read_csv(
                file_path,
                nrows=5
            )

            required = {
                "segmentation_method",
                "image_id",
                "region",
            }

            if required.issubset(
                test_df.columns
            ):

                return file_path


    # --------------------------------------------------------
    # 若檔名不同，自動掃描 output/
    # --------------------------------------------------------

    exclude_keywords = [
        "prediction",
        "metrics",
        "agreement",
        "bootstrap",
        "consensus",
        "fold",
        "ranking",
        "summary",
        "comparison",
        "integrated",
        "unanimous",
    ]


    candidates = []


    for file_path in OUTPUT_DIR.glob(
        "*.csv"
    ):

        lower_name = (
            file_path.name.lower()
        )


        if any(
            keyword in lower_name
            for keyword in exclude_keywords
        ):

            continue


        try:

            test_df = pd.read_csv(
                file_path,
                nrows=10
            )

        except Exception:

            continue


        required = {
            "segmentation_method",
            "image_id",
            "region",
        }


        if not required.issubset(
            test_df.columns
        ):

            continue


        numeric_columns = (
            test_df
            .select_dtypes(
                include=[
                    np.number
                ]
            )
            .columns
            .tolist()
        )


        # 色彩特徵理論上會有很多 numeric columns
        if len(
            numeric_columns
        ) >= 20:

            candidates.append(
                file_path
            )


    if len(
        candidates
    ) == 0:

        raise FileNotFoundError(
            "找不到 07_extract_color_features.py "
            "產生的 combined feature CSV。"
        )


    # 優先檔案大小較大的
    candidates = sorted(
        candidates,
        key=lambda x: x.stat().st_size,
        reverse=True
    )


    return candidates[0]


FEATURE_FILE = find_feature_file()


print("=" * 100)
print("Unanimous Training → 3:1 Disagreement Testing")
print("SVM + Random Forest")
print("=" * 100)

print(
    "使用 feature file：",
    FEATURE_FILE
)


# ============================================================
# 4. 讀取資料
# ============================================================

features = pd.read_csv(
    FEATURE_FILE,
    dtype={
        "image_id": str
    }
)


votes = pd.read_csv(
    VOTE_FILE,
    dtype={
        "image_id": str
    }
)


folds = pd.read_csv(
    FOLD_FILE,
    dtype={
        "image_id": str
    }
)


# ============================================================
# 5. image_id 格式統一
# ============================================================

for dataframe in [
    features,
    votes,
    folds,
]:

    dataframe["image_id"] = (
        dataframe["image_id"]
        .astype(str)
        .str.strip()
        .str.replace(
            r"\.0$",
            "",
            regex=True
        )
    )


# ============================================================
# 6. 確認 votes
# ============================================================

if "positive_votes" not in votes.columns:

    votes["positive_votes"] = (
        votes[
            [
                "D1",
                "D2",
                "D3",
                "D4",
            ]
        ]
        .sum(
            axis=1
        )
    )


def agreement_type(
    votes_count
):

    if votes_count == 4:

        return "unanimous_positive"

    elif votes_count == 0:

        return "unanimous_negative"

    elif votes_count == 3:

        return "three_one_positive"

    elif votes_count == 1:

        return "three_one_negative"

    elif votes_count == 2:

        return "two_two"

    return "unknown"


votes["agreement_detail"] = (
    votes["positive_votes"]
    .apply(
        agreement_type
    )
)


votes["majority_label"] = np.select(

    [
        votes["positive_votes"] >= 3,
        votes["positive_votes"] <= 1,
        votes["positive_votes"] == 2,
    ],

    [
        1,
        0,
        -1,
    ],

    default=-1
)


# ============================================================
# 7. 找出真正 feature columns
# ============================================================

metadata_columns = {

    "image_id",
    "region",
    "segmentation_method",

    "fold",

    "label",
    "color",

    "淡紅",
    "淡白",
    "鮮紅",
    "暗紅",
    "青紫",
    "灰黑",

    "target",
    "true",
    "pred",
}


numeric_columns = (
    features
    .select_dtypes(
        include=[
            np.number
        ]
    )
    .columns
    .tolist()
)


feature_columns = [

    column

    for column in numeric_columns

    if column
    not in metadata_columns
]


if len(
    feature_columns
) < 20:

    raise ValueError(
        f"偵測到的 feature columns 太少："
        f"{len(feature_columns)}"
    )


print(
    "偵測到 feature 數：",
    len(
        feature_columns
    )
)


# ============================================================
# 8. 檢查 segmentation methods
# ============================================================

print(
    "Feature segmentation methods：",
    sorted(
        features[
            "segmentation_method"
        ]
        .unique()
        .tolist()
    )
)


# ============================================================
# 9. 模型
# ============================================================

def build_model(
    model_name
):

    if model_name == "SVM":

        return Pipeline(
            [
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
                ),
            ]
        )


    elif model_name == "Random Forest":

        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1
        )


    else:

        raise ValueError(
            f"Unknown model：{model_name}"
        )


# ============================================================
# 10. Metrics
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

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
            accuracy_score(
                y_true,
                y_pred
            ),

        "precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "f1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred
            ),

        "tn":
            int(
                tn
            ),

        "fp":
            int(
                fp
            ),

        "fn":
            int(
                fn
            ),

        "tp":
            int(
                tp
            ),
    }


# ============================================================
# 11. 主迴圈
# ============================================================

prediction_records = []

fold_metric_records = []


for color in ANALYSIS_COLORS:

    print("\n")
    print("=" * 100)
    print(
        f"舌質色：{color}"
    )
    print("=" * 100)


    color_votes = (
        votes[
            votes[
                "color"
            ]
            ==
            color
        ][
            [
                "image_id",
                "region",
                "positive_votes",
                "agreement_detail",
                "majority_label",
            ]
        ]
        .copy()
    )


    color_folds = (
        folds[
            folds[
                "color"
            ]
            ==
            color
        ]
        .copy()
    )


    # ========================================================
    # 每個 segmentation
    # ========================================================

    for segmentation_method in (
        SEGMENTATION_METHODS
    ):

        segmentation_features = (
            features[
                features[
                    "segmentation_method"
                ]
                ==
                segmentation_method
            ]
            .copy()
        )


        # ----------------------------------------------------
        # Features + physician votes
        # ----------------------------------------------------

        data = (
            segmentation_features
            .merge(
                color_votes,
                on=[
                    "image_id",
                    "region"
                ],
                how="inner",
                validate="one_to_one"
            )
        )


        print(
            f"\n{segmentation_method}"
        )


        # ====================================================
        # 每個 fold
        # ====================================================

        for fold_id in sorted(
            color_folds[
                "fold"
            ].unique()
        ):

            fold_id = int(
                fold_id
            )


            test_ids = set(

                color_folds.loc[
                    color_folds[
                        "fold"
                    ]
                    ==
                    fold_id,
                    "image_id"
                ]
            )


            train_ids = set(

                color_folds.loc[
                    color_folds[
                        "fold"
                    ]
                    !=
                    fold_id,
                    "image_id"
                ]
            )


            # ------------------------------------------------
            # Leakage check
            # ------------------------------------------------

            overlap = (
                train_ids
                &
                test_ids
            )


            if len(
                overlap
            ) > 0:

                raise ValueError(
                    f"Patient leakage："
                    f"{color}, fold={fold_id}, "
                    f"{overlap}"
                )


            # ------------------------------------------------
            # TRAIN
            #
            # 四醫師完全一致：
            # 4 positive / 0 positive
            # ------------------------------------------------

            train_df = (
                data[
                    (
                        data[
                            "image_id"
                        ]
                        .isin(
                            train_ids
                        )
                    )
                    &
                    (
                        data[
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


            # ------------------------------------------------
            # TEST
            #
            # 只測 3:1 / 1:3
            # ------------------------------------------------

            test_df = (
                data[
                    (
                        data[
                            "image_id"
                        ]
                        .isin(
                            test_ids
                        )
                    )
                    &
                    (
                        data[
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


            X_train = (
                train_df[
                    feature_columns
                ]
                .to_numpy(
                    dtype=float
                )
            )


            y_train = (
                train_df[
                    "majority_label"
                ]
                .to_numpy(
                    dtype=int
                )
            )


            X_test = (
                test_df[
                    feature_columns
                ]
                .to_numpy(
                    dtype=float
                )
            )


            y_test = (
                test_df[
                    "majority_label"
                ]
                .to_numpy(
                    dtype=int
                )
            )


            # ------------------------------------------------
            # Fold sanity check
            # ------------------------------------------------

            train_classes = set(
                np.unique(
                    y_train
                )
            )


            test_classes = set(
                np.unique(
                    y_test
                )
            )


            if train_classes != {
                0,
                1
            }:

                raise ValueError(
                    f"{color} / "
                    f"{segmentation_method} / "
                    f"fold {fold_id}: "
                    f"train classes={train_classes}"
                )


            if test_classes != {
                0,
                1
            }:

                raise ValueError(
                    f"{color} / "
                    f"{segmentation_method} / "
                    f"fold {fold_id}: "
                    f"test classes={test_classes}"
                )


            # =================================================
            # 兩個模型
            # =================================================

            for model_name in MODELS:

                model = build_model(
                    model_name
                )


                model.fit(
                    X_train,
                    y_train
                )


                y_pred = (
                    model.predict(
                        X_test
                    )
                )


                metrics = (
                    calculate_metrics(
                        y_test,
                        y_pred
                    )
                )


                # --------------------------------------------
                # Fold metrics
                # --------------------------------------------

                fold_metric_records.append({

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

                    "segmentation_method":
                        segmentation_method,

                    "model":
                        model_name,

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

                    "train_items":
                        len(
                            train_df
                        ),

                    "train_positive":
                        int(
                            np.sum(
                                y_train
                                ==
                                1
                            )
                        ),

                    "train_negative":
                        int(
                            np.sum(
                                y_train
                                ==
                                0
                            )
                        ),

                    "test_items":
                        len(
                            test_df
                        ),

                    "test_positive":
                        int(
                            np.sum(
                                y_test
                                ==
                                1
                            )
                        ),

                    "test_negative":
                        int(
                            np.sum(
                                y_test
                                ==
                                0
                            )
                        ),

                    **metrics,
                })


                # --------------------------------------------
                # Prediction records
                # --------------------------------------------

                for row_index, (
                    true_value,
                    pred_value
                ) in enumerate(
                    zip(
                        y_test,
                        y_pred
                    )
                ):

                    original_row = (
                        test_df.iloc[
                            row_index
                        ]
                    )


                    prediction_records.append({

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

                        "segmentation_method":
                            segmentation_method,

                        "model":
                            model_name,

                        "fold":
                            fold_id,

                        "image_id":
                            original_row[
                                "image_id"
                            ],

                        "region":
                            original_row[
                                "region"
                            ],

                        "positive_votes":
                            int(
                                original_row[
                                    "positive_votes"
                                ]
                            ),

                        "agreement_detail":
                            original_row[
                                "agreement_detail"
                            ],

                        "true":
                            int(
                                true_value
                            ),

                        "pred":
                            int(
                                pred_value
                            ),
                    })


                print(
                    f"Fold {fold_id} | "
                    f"{model_name} | "
                    f"train={len(train_df)} | "
                    f"test={len(test_df)} | "
                    f"F1={metrics['f1']:.4f} | "
                    f"BA={metrics['balanced_accuracy']:.4f}"
                )


# ============================================================
# 12. 建立 DataFrame
# ============================================================

predictions_df = pd.DataFrame(
    prediction_records
)


fold_metrics_df = pd.DataFrame(
    fold_metric_records
)


# ============================================================
# 13. 檢查 OOF prediction 重複
# ============================================================

duplicate_columns = [
    "color",
    "segmentation_method",
    "model",
    "image_id",
    "region",
]


duplicates = (
    predictions_df
    .duplicated(
        subset=duplicate_columns,
        keep=False
    )
)


if duplicates.any():

    print(
        predictions_df[
            duplicates
        ][
            duplicate_columns
        ]
        .head(
            20
        )
    )

    raise ValueError(
        "發現重複 OOF prediction。"
    )


# ============================================================
# 14. 對五 folds OOF predictions 計算 Color-level Metrics
# ============================================================

color_metric_records = []


for (
    color,
    analysis_type,
    segmentation_method,
    model_name
), group in predictions_df.groupby(

    [
        "color",
        "analysis_type",
        "segmentation_method",
        "model",
    ]
):

    metrics = (
        calculate_metrics(
            group[
                "true"
            ],
            group[
                "pred"
            ]
        )
    )


    color_metric_records.append({

        "color":
            color,

        "analysis_type":
            analysis_type,

        "segmentation_method":
            segmentation_method,

        "model":
            model_name,

        "n_test_items":
            len(
                group
            ),

        "n_test_patients":
            group[
                "image_id"
            ].nunique(),

        "positive":
            int(
                np.sum(
                    group[
                        "true"
                    ]
                    ==
                    1
                )
            ),

        "negative":
            int(
                np.sum(
                    group[
                        "true"
                    ]
                    ==
                    0
                )
            ),

        **metrics,
    })


color_metrics_df = pd.DataFrame(
    color_metric_records
)


# ============================================================
# 15. Primary Colors Macro Summary
#
# 只用淡紅 + 鮮紅
# ============================================================

primary_df = (
    color_metrics_df[
        color_metrics_df[
            "analysis_type"
        ]
        ==
        "primary"
    ]
    .copy()
)


primary_summary = (
    primary_df
    .groupby(
        [
            "segmentation_method",
            "model",
        ],
        as_index=False
    )
    .agg(

        n_colors=(
            "color",
            "nunique"
        ),

        macro_accuracy=(
            "accuracy",
            "mean"
        ),

        macro_precision=(
            "precision",
            "mean"
        ),

        macro_recall=(
            "recall",
            "mean"
        ),

        macro_f1=(
            "f1",
            "mean"
        ),

        macro_balanced_accuracy=(
            "balanced_accuracy",
            "mean"
        ),
    )
)


primary_summary[
    "rank_by_macro_f1"
] = (
    primary_summary[
        "macro_f1"
    ]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)


primary_summary = (
    primary_summary
    .sort_values(
        "rank_by_macro_f1"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 16. 四色探索性 Summary
# ============================================================

all_summary = (
    color_metrics_df
    .groupby(
        [
            "segmentation_method",
            "model",
        ],
        as_index=False
    )
    .agg(

        n_colors=(
            "color",
            "nunique"
        ),

        macro_accuracy=(
            "accuracy",
            "mean"
        ),

        macro_precision=(
            "precision",
            "mean"
        ),

        macro_recall=(
            "recall",
            "mean"
        ),

        macro_f1=(
            "f1",
            "mean"
        ),

        macro_balanced_accuracy=(
            "balanced_accuracy",
            "mean"
        ),
    )
)


all_summary[
    "rank_by_macro_f1"
] = (
    all_summary[
        "macro_f1"
    ]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)


all_summary = (
    all_summary
    .sort_values(
        "rank_by_macro_f1"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 17. 輸出
# ============================================================

PREDICTION_OUTPUT = (
    OUTPUT_DIR
    /
    "unanimous_to_disagreement_svm_rf_predictions.csv"
)


FOLD_METRIC_OUTPUT = (
    OUTPUT_DIR
    /
    "unanimous_to_disagreement_svm_rf_fold_metrics.csv"
)


COLOR_METRIC_OUTPUT = (
    OUTPUT_DIR
    /
    "unanimous_to_disagreement_svm_rf_color_metrics.csv"
)


PRIMARY_SUMMARY_OUTPUT = (
    OUTPUT_DIR
    /
    "unanimous_to_disagreement_svm_rf_primary_summary.csv"
)


ALL_SUMMARY_OUTPUT = (
    OUTPUT_DIR
    /
    "unanimous_to_disagreement_svm_rf_exploratory_summary.csv"
)


predictions_df.to_csv(
    PREDICTION_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


fold_metrics_df.to_csv(
    FOLD_METRIC_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


color_metrics_df.to_csv(
    COLOR_METRIC_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


primary_summary.to_csv(
    PRIMARY_SUMMARY_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


all_summary.to_csv(
    ALL_SUMMARY_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 18. 顯示 Color-level Results
# ============================================================

print("\n")
print("=" * 100)
print("Color-level OOF Results：3:1 disagreement test")
print("=" * 100)


display_color = (
    color_metrics_df[
        [
            "color",
            "analysis_type",
            "segmentation_method",
            "model",
            "n_test_items",
            "positive",
            "negative",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "balanced_accuracy",
        ]
    ]
    .copy()
)


for column in [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "balanced_accuracy",
]:

    display_color[
        column
    ] = (
        display_color[
            column
        ]
        .round(
            4
        )
    )


print(
    display_color.to_string(
        index=False
    )
)


# ============================================================
# 19. Primary Summary
# ============================================================

print("\n")
print("=" * 100)
print("PRIMARY RESULTS：淡紅 + 鮮紅")
print("=" * 100)


display_primary = (
    primary_summary.copy()
)


for column in [
    "macro_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "macro_balanced_accuracy",
]:

    display_primary[
        column
    ] = (
        display_primary[
            column
        ]
        .round(
            4
        )
    )


print(
    display_primary.to_string(
        index=False
    )
)


# ============================================================
# 20. Exploratory Summary
# ============================================================

print("\n")
print("=" * 100)
print("EXPLORATORY RESULTS：淡紅 + 鮮紅 + 暗紅 + 青紫")
print("=" * 100)


display_all = (
    all_summary.copy()
)


for column in [
    "macro_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "macro_balanced_accuracy",
]:

    display_all[
        column
    ] = (
        display_all[
            column
        ]
        .round(
            4
        )
    )


print(
    display_all.to_string(
        index=False
    )
)


# ============================================================
# 21. 輸出位置
# ============================================================

print("\n")
print("=" * 100)
print("輸出檔案")
print("=" * 100)

print(
    PREDICTION_OUTPUT
)

print(
    FOLD_METRIC_OUTPUT
)

print(
    COLOR_METRIC_OUTPUT
)

print(
    PRIMARY_SUMMARY_OUTPUT
)

print(
    ALL_SUMMARY_OUTPUT
)

print("=" * 100)
