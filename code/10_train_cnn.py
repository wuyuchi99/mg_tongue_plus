from pathlib import Path
import random
import copy

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix
)


# ============================================================
# 1. 基本設定
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# 2. 專案路徑
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"


# ============================================================
# 3. 兩種舌區分法
# ============================================================

SEGMENTATION_METHODS = [
    "tip_middle_root_1_2_2",
    "tip_middle_root_1_3_1",
]


# ============================================================
# 4. 六種舌質色
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
# 5. CNN 訓練參數
# ============================================================

IMAGE_SIZE = 224

BATCH_SIZE = 16

# 第一階段：只訓練分類頭
HEAD_EPOCHS = 5

# 第二階段：解凍最後幾層微調
FINETUNE_EPOCHS = 3

HEAD_LR = 1e-3
FINETUNE_LR = 1e-4

WEIGHT_DECAY = 1e-4


# ============================================================
# 6. 選擇運算裝置
#
# Apple Silicon / 支援 MPS 的 Mac → MPS
# 否則 → CPU
# ============================================================

if torch.backends.mps.is_available():

    DEVICE = torch.device("mps")

else:

    DEVICE = torch.device("cpu")


print("=" * 80)
print("CNN Transfer Learning")
print("=" * 80)

print(
    "運算裝置：",
    DEVICE
)

print(
    "模型：MobileNetV3-Small"
)


# ============================================================
# 7. 讀取 SVM/RF 已使用的 Fold
#
# 這是關鍵：
# CNN 必須使用與 SVM / RF 相同的 Test Fold
# ============================================================

fold_file = (
    OUTPUT_DIR
    / "svm_rf_all_methods_predictions.csv"
)


if not fold_file.exists():

    raise FileNotFoundError(
        "\n找不到："
        f"{fold_file}\n"
        "請先執行 09_train_svm_rf.py"
    )


fold_data = pd.read_csv(
    fold_file,
    dtype={
        "image_id": str
    }
)


fold_data["image_id"] = (
    fold_data["image_id"]
    .astype(str)
    .str.strip()
)


fold_data["region"] = (
    fold_data["region"]
    .astype(str)
    .str.strip()
)


# ============================================================
# 8. 只取一組作為 Fold Reference
#
# 因為：
# 1:2:2 / 1:3:1
# SVM / RF
#
# 本來就用了相同 Fold，
# 所以只取 1:2:2 + SVM 即可。
# ============================================================

fold_reference = (
    fold_data[
        (
            fold_data["segmentation_method"]
            ==
            "tip_middle_root_1_2_2"
        )
        &
        (
            fold_data["model"]
            ==
            "SVM"
        )
    ]
    .copy()
)


if len(fold_reference) == 0:

    raise ValueError(
        "在 svm_rf_all_methods_predictions.csv "
        "中找不到 1:2:2 + SVM 的 Fold 資料"
    )


# ============================================================
# 9. 讀取兩種 ML Dataset
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
        dtype={
            "image_id": str
        }
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


    df["sample_key"] = (
        df["image_id"]
        + "__"
        + df["region"]
    )


    datasets[
        method_name
    ] = df


# ============================================================
# 10. Pad 成正方形
#
# 不直接把長方形拉成 224 × 224，
# 避免影像被嚴重變形。
# ============================================================

class PadToSquare:

    def __call__(
        self,
        image
    ):

        width, height = (
            image.size
        )


        size = max(
            width,
            height
        )


        new_image = Image.new(
            "RGB",
            (
                size,
                size
            ),
            (
                0,
                0,
                0
            )
        )


        left = (
            size - width
        ) // 2


        top = (
            size - height
        ) // 2


        new_image.paste(
            image,
            (
                left,
                top
            )
        )


        return new_image


# ============================================================
# 11. ImageNet Normalize
# ============================================================

weights = (
    MobileNet_V3_Small_Weights.DEFAULT
)


normalize = transforms.Normalize(
    mean=[
        0.485,
        0.456,
        0.406
    ],
    std=[
        0.229,
        0.224,
        0.225
    ]
)


# ============================================================
# 12. Training Transform
#
# 注意：
# 不使用 ColorJitter
# 因為我們的研究目標就是顏色。
# ============================================================

train_transform = transforms.Compose([

    PadToSquare(),

    transforms.Resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    ),

    # 很小幅度的幾何增強
    transforms.RandomRotation(
        degrees=5
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.ToTensor(),

    normalize
])


# ============================================================
# 13. Test Transform
#
# Test 不做任何隨機增強
# ============================================================

test_transform = transforms.Compose([

    PadToSquare(),

    transforms.Resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    ),

    transforms.ToTensor(),

    normalize
])


# ============================================================
# 14. Dataset
# ============================================================

class TongueDataset(
    Dataset
):

    def __init__(
        self,
        dataframe,
        method_name,
        target_column,
        transform
    ):

        self.df = (
            dataframe
            .reset_index(
                drop=True
            )
        )

        self.method_name = (
            method_name
        )

        self.target_column = (
            target_column
        )

        self.transform = (
            transform
        )


    def __len__(
        self
    ):

        return len(
            self.df
        )


    def __getitem__(
        self,
        index
    ):

        row = (
            self.df.iloc[
                index
            ]
        )


        image_id = str(
            row[
                "image_id"
            ]
        )


        region = str(
            row[
                "region"
            ]
        )


        image_path = (
            OUTPUT_DIR
            / self.method_name
            / "regions"
            / region
            / f"{image_id}.png"
        )


        if not image_path.exists():

            raise FileNotFoundError(
                f"找不到 CNN 區域圖片："
                f"{image_path}"
            )


        image = (
            Image.open(
                image_path
            )
            .convert(
                "RGB"
            )
        )


        if self.transform:

            image = (
                self.transform(
                    image
                )
            )


        label = float(
            row[
                self.target_column
            ]
        )


        return (
            image,
            torch.tensor(
                label,
                dtype=torch.float32
            )
        )


# ============================================================
# 15. 建立 MobileNetV3-Small
# ============================================================

def build_model():

    model = mobilenet_v3_small(
        weights=weights
    )


    # --------------------------------------------------------
    # 先凍結所有 CNN feature extractor
    # --------------------------------------------------------

    for parameter in (
        model.features.parameters()
    ):

        parameter.requires_grad = False


    # --------------------------------------------------------
    # MobileNet 最後 classifier：
    #
    # 原本：
    # Linear(... → 1000 classes)
    #
    # 改成：
    # Linear(... → 1)
    #
    # 因為每次訓練一個舌質色：
    # 0 / 1 二元分類
    # --------------------------------------------------------

    input_features = (
        model.classifier[
            -1
        ].in_features
    )


    model.classifier[
        -1
    ] = nn.Linear(
        input_features,
        1
    )


    return model


# ============================================================
# 16. 解凍最後幾層做 Fine-tuning
# ============================================================

def unfreeze_last_layers(
    model
):

    # 只解凍最後兩個 feature blocks
    for block in (
        list(
            model.features.children()
        )[-2:]
    ):

        for parameter in (
            block.parameters()
        ):

            parameter.requires_grad = True


# ============================================================
# 17. 單一 Epoch 訓練
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion
):

    model.train()

    total_loss = 0.0


    for images, labels in loader:

        images = (
            images.to(
                DEVICE
            )
        )


        labels = (
            labels
            .to(
                DEVICE
            )
            .unsqueeze(
                1
            )
        )


        optimizer.zero_grad()


        logits = model(
            images
        )


        loss = criterion(
            logits,
            labels
        )


        loss.backward()


        optimizer.step()


        total_loss += (
            loss.item()
            *
            images.size(0)
        )


    return (
        total_loss
        /
        len(
            loader.dataset
        )
    )


# ============================================================
# 18. 預測
# ============================================================

def predict(
    model,
    loader
):

    model.eval()


    true_labels = []
    predictions = []
    probabilities = []


    with torch.no_grad():

        for images, labels in loader:

            images = (
                images.to(
                    DEVICE
                )
            )


            logits = (
                model(
                    images
                )
                .squeeze(
                    1
                )
            )


            probs = (
                torch.sigmoid(
                    logits
                )
                .cpu()
                .numpy()
            )


            preds = (
                probs
                >=
                0.5
            ).astype(
                int
            )


            true_labels.extend(
                labels
                .numpy()
                .astype(int)
                .tolist()
            )


            predictions.extend(
                preds.tolist()
            )


            probabilities.extend(
                probs.tolist()
            )


    return (
        np.array(
            true_labels
        ),
        np.array(
            predictions
        ),
        np.array(
            probabilities
        )
    )


# ============================================================
# 19. 儲存所有結果
# ============================================================

fold_results = []
color_results = []
prediction_records = []


# ============================================================
# 20. 六種舌色逐一訓練
# ============================================================

for color in COLORS:

    target_column = (
        f"y_{color}"
    )


    print("\n")
    print("=" * 80)
    print(
        "舌質色：",
        color
    )
    print("=" * 80)


    # --------------------------------------------------------
    # 取得這個 Color 的共同 Fold Mapping
    # --------------------------------------------------------

    color_folds = (
        fold_reference[
            fold_reference[
                "color"
            ]
            ==
            color
        ][
            [
                "image_id",
                "region",
                "fold",
                "true"
            ]
        ]
        .drop_duplicates(
            subset=[
                "image_id",
                "region"
            ]
        )
        .copy()
    )


    if len(
        color_folds
    ) == 0:

        print(
            "⚠ 找不到 Fold 資料，跳過"
        )

        continue


    color_folds[
        "sample_key"
    ] = (
        color_folds[
            "image_id"
        ]
        +
        "__"
        +
        color_folds[
            "region"
        ]
    )


    fold_numbers = sorted(
        color_folds[
            "fold"
        ]
        .unique()
        .tolist()
    )


    print(
        "Fold：",
        fold_numbers
    )


    # ========================================================
    # 21. 兩種分區方法
    # ========================================================

    for method_name in (
        SEGMENTATION_METHODS
    ):


        print("\n")
        print("-" * 80)

        print(
            "分區方法：",
            method_name
        )

        print("-" * 80)


        method_df = (
            datasets[
                method_name
            ]
            .copy()
        )


        # ----------------------------------------------------
        # 只保留此色可使用的 0 / 1 樣本
        # ----------------------------------------------------

        method_df = (
            method_df[
                method_df[
                    target_column
                ].isin(
                    [
                        0,
                        1
                    ]
                )
            ]
            .copy()
        )


        # ----------------------------------------------------
        # 加入 Fold
        # ----------------------------------------------------

        method_df = (
            method_df.merge(
                color_folds[
                    [
                        "sample_key",
                        "fold"
                    ]
                ],
                on="sample_key",
                how="inner",
                validate="one_to_one"
            )
        )


        all_true = []
        all_pred = []


        # ====================================================
        # 22. 每個 Fold
        # ====================================================

        for fold_number in (
            fold_numbers
        ):


            print(
                f"\nFold {fold_number}"
            )


            train_df = (
                method_df[
                    method_df[
                        "fold"
                    ]
                    !=
                    fold_number
                ]
                .copy()
            )


            test_df = (
                method_df[
                    method_df[
                        "fold"
                    ]
                    ==
                    fold_number
                ]
                .copy()
            )


            # ================================================
            # Data Leakage 檢查
            # ================================================

            train_patients = set(
                train_df[
                    "image_id"
                ]
            )


            test_patients = set(
                test_df[
                    "image_id"
                ]
            )


            overlap = (
                train_patients
                &
                test_patients
            )


            if overlap:

                raise RuntimeError(
                    "發現病人 Data Leakage！"
                )


            print(
                " Train samples：",
                len(train_df)
            )

            print(
                " Test samples：",
                len(test_df)
            )

            print(
                " Train patients：",
                len(train_patients)
            )

            print(
                " Test patients：",
                len(test_patients)
            )


            # ================================================
            # Training 必須同時有 0 和 1
            # ================================================

            if (
                train_df[
                    target_column
                ].nunique()
                <
                2
            ):

                print(
                    " ⚠ Training 只有一類，跳過"
                )

                continue


            # ================================================
            # Dataset / DataLoader
            # ================================================

            train_dataset = TongueDataset(
                dataframe=train_df,
                method_name=method_name,
                target_column=target_column,
                transform=train_transform
            )


            test_dataset = TongueDataset(
                dataframe=test_df,
                method_name=method_name,
                target_column=target_column,
                transform=test_transform
            )


            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0
            )


            test_loader = DataLoader(
                test_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0
            )


            # ================================================
            # 處理 Class Imbalance
            #
            # positive 很少時，增加陽性 Loss 權重
            # ================================================

            positive_count = int(
                (
                    train_df[
                        target_column
                    ]
                    ==
                    1
                ).sum()
            )


            negative_count = int(
                (
                    train_df[
                        target_column
                    ]
                    ==
                    0
                ).sum()
            )


            pos_weight_value = (
                negative_count
                /
                max(
                    positive_count,
                    1
                )
            )


            pos_weight = torch.tensor(
                [
                    pos_weight_value
                ],
                dtype=torch.float32,
                device=DEVICE
            )


            criterion = (
                nn.BCEWithLogitsLoss(
                    pos_weight=pos_weight
                )
            )


            # ================================================
            # 建立 Pretrained CNN
            # ================================================

            model = (
                build_model()
                .to(
                    DEVICE
                )
            )


            # ================================================
            # 第一階段：
            # Freeze Backbone
            # 只訓練 classifier
            # ================================================

            optimizer = torch.optim.AdamW(

                filter(
                    lambda p:
                    p.requires_grad,
                    model.parameters()
                ),

                lr=HEAD_LR,

                weight_decay=
                    WEIGHT_DECAY
            )


            print(
                " Stage 1：訓練分類頭"
            )


            for epoch in range(
                1,
                HEAD_EPOCHS + 1
            ):

                loss = (
                    train_one_epoch(
                        model,
                        train_loader,
                        optimizer,
                        criterion
                    )
                )


                print(
                    f"   Epoch "
                    f"{epoch}/{HEAD_EPOCHS}"
                    f" loss={loss:.4f}"
                )


            # ================================================
            # 第二階段：
            # Fine-tune 最後兩個 CNN blocks
            # ================================================

            unfreeze_last_layers(
                model
            )


            optimizer = torch.optim.AdamW(

                filter(
                    lambda p:
                    p.requires_grad,
                    model.parameters()
                ),

                lr=FINETUNE_LR,

                weight_decay=
                    WEIGHT_DECAY
            )


            print(
                " Stage 2：Fine-tuning"
            )


            for epoch in range(
                1,
                FINETUNE_EPOCHS + 1
            ):

                loss = (
                    train_one_epoch(
                        model,
                        train_loader,
                        optimizer,
                        criterion
                    )
                )


                print(
                    f"   Epoch "
                    f"{epoch}/{FINETUNE_EPOCHS}"
                    f" loss={loss:.4f}"
                )


            # ================================================
            # Test
            # ================================================

            (
                y_true,
                y_pred,
                y_prob
            ) = predict(
                model,
                test_loader
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


            balanced_accuracy = (
                balanced_accuracy_score(
                    y_true,
                    y_pred
                )
            )


            fold_results.append({

                "segmentation_method":
                    method_name,

                "model":
                    "CNN",

                "cnn_architecture":
                    "MobileNetV3-Small",

                "color":
                    color,

                "fold":
                    fold_number,

                "train_samples":
                    len(train_df),

                "test_samples":
                    len(test_df),

                "train_patients":
                    len(train_patients),

                "test_patients":
                    len(test_patients),

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


            # ================================================
            # 儲存 Prediction
            # ================================================

            test_df = (
                test_df
                .reset_index(
                    drop=True
                )
            )


            for i in range(
                len(
                    test_df
                )
            ):

                prediction_records.append({

                    "segmentation_method":
                        method_name,

                    "model":
                        "CNN",

                    "color":
                        color,

                    "fold":
                        fold_number,

                    "image_id":
                        test_df.loc[
                            i,
                            "image_id"
                        ],

                    "region":
                        test_df.loc[
                            i,
                            "region"
                        ],

                    "true":
                        int(
                            y_true[
                                i
                            ]
                        ),

                    "pred":
                        int(
                            y_pred[
                                i
                            ]
                        ),

                    "probability":
                        float(
                            y_prob[
                                i
                            ]
                        ),
                })


            all_true.extend(
                y_true.tolist()
            )


            all_pred.extend(
                y_pred.tolist()
            )


            print(
                f" Accuracy="
                f"{accuracy:.4f}"
            )

            print(
                f" F1="
                f"{f1:.4f}"
            )

            print(
                f" Balanced Accuracy="
                f"{balanced_accuracy:.4f}"
            )


            # 釋放模型
            del model

            if DEVICE.type == "mps":

                torch.mps.empty_cache()


        # ====================================================
        # 23. 該 Color 的 Out-of-fold 整體結果
        # ====================================================

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
                "CNN",

            "cnn_architecture":
                "MobileNetV3-Small",

            "color":
                color,

            "n_samples":
                len(
                    all_true
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
            "\n Overall "
            f"{color}:"
        )

        print(
            f" Accuracy="
            f"{accuracy:.4f}"
        )

        print(
            f" F1="
            f"{f1:.4f}"
        )

        print(
            f" Balanced Accuracy="
            f"{balanced_accuracy:.4f}"
        )


# ============================================================
# 24. 儲存 Fold Metrics
# ============================================================

fold_df = pd.DataFrame(
    fold_results
)


fold_output = (
    OUTPUT_DIR
    /
    "cnn_all_methods_fold_metrics.csv"
)


fold_df.to_csv(
    fold_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 25. 儲存 Color Metrics
# ============================================================

color_df = pd.DataFrame(
    color_results
)


color_output = (
    OUTPUT_DIR
    /
    "cnn_all_methods_color_metrics.csv"
)


color_df.to_csv(
    color_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 26. 儲存 Predictions
# ============================================================

prediction_df = pd.DataFrame(
    prediction_records
)


prediction_output = (
    OUTPUT_DIR
    /
    "cnn_all_methods_predictions.csv"
)


prediction_df.to_csv(
    prediction_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 27. CNN 兩種分區總比較
#
# 六種顏色 Macro Average
# ============================================================

cnn_summary = (
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


cnn_summary_output = (
    OUTPUT_DIR
    /
    "cnn_segmentation_comparison.csv"
)


cnn_summary.to_csv(
    cnn_summary_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 28. 合併 SVM / RF / CNN
# ============================================================

svm_rf_file = (
    OUTPUT_DIR
    /
    "svm_rf_segmentation_comparison.csv"
)


if svm_rf_file.exists():

    svm_rf_summary = pd.read_csv(
        svm_rf_file
    )


    all_models = pd.concat(
        [
            svm_rf_summary,
            cnn_summary
        ],
        ignore_index=True
    )


    all_models = (
        all_models
        .sort_values(
            "f1",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    all_models[
        "rank_by_f1"
    ] = np.arange(
        1,
        len(
            all_models
        ) + 1
    )


    all_models_output = (
        OUTPUT_DIR
        /
        "all_models_segmentation_comparison.csv"
    )


    all_models.to_csv(
        all_models_output,
        index=False,
        encoding="utf-8-sig"
    )


    print("\n")
    print("=" * 80)
    print("SVM / Random Forest / CNN 最終比較")
    print("=" * 80)


    print(
        all_models[
            [
                "rank_by_f1",
                "segmentation_method",
                "model",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "balanced_accuracy"
            ]
        ]
        .to_string(
            index=False
        )
    )


    print(
        "\n總比較檔：",
        all_models_output
    )


print("\n")
print("=" * 80)
print("CNN 完成")
print("=" * 80)

print(
    "Fold metrics：",
    fold_output
)

print(
    "Color metrics：",
    color_output
)

print(
    "Predictions：",
    prediction_output
)

print(
    "CNN 分區比較：",
    cnn_summary_output
)

print("=" * 80)