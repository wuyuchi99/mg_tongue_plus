from pathlib import Path
import re
import random

import numpy as np
import pandas as pd

from PIL import Image, ImageOps

import torch
import torch.nn as nn

from torch.utils.data import (
    Dataset,
    DataLoader,
)

from torchvision import (
    transforms,
    models,
)

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

BATCH_SIZE = 16

STAGE1_EPOCHS = 5
STAGE2_EPOCHS = 3

STAGE1_LR = 1e-3
STAGE2_LR = 1e-4

IMAGE_SIZE = 224


# ============================================================
# 2. 兩種舌區分割方法
# ============================================================

SEGMENTATION_METHODS = [
    "tip_middle_root_1_2_2",
    "tip_middle_root_1_3_1",
]


# ============================================================
# 3. 正式分析舌質色
#
# 只使用：
# 淡紅
# 鮮紅
#
# 因為這兩色具有較足夠的 unanimous positive
# ============================================================

PRIMARY_COLORS = [
    "淡紅",
    "鮮紅",
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
# 5. 路徑
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


VOTE_FILE = (
    OUTPUT_DIR
    /
    "interrater_vote_details.csv"
)


FOLD_FILE = (
    OUTPUT_DIR
    /
    "unanimous_disagreement_fold_assignments.csv"
)


SVM_RF_COLOR_FILE = (
    OUTPUT_DIR
    /
    "unanimous_to_disagreement_svm_rf_color_metrics.csv"
)


# ============================================================
# 6. 檢查必要檔案
# ============================================================

for file_path in [
    VOTE_FILE,
    FOLD_FILE,
]:

    if not file_path.exists():

        raise FileNotFoundError(
            f"找不到：{file_path}"
        )


# ============================================================
# 7. Random Seed
# ============================================================

def set_seed(
    seed
):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


set_seed(
    SEED
)


# ============================================================
# 8. Device
#
# Mac Apple Silicon 優先使用 MPS
# ============================================================

if (
    torch.backends.mps.is_available()
    and
    torch.backends.mps.is_built()
):

    DEVICE = torch.device(
        "mps"
    )

elif torch.cuda.is_available():

    DEVICE = torch.device(
        "cuda"
    )

else:

    DEVICE = torch.device(
        "cpu"
    )


print("=" * 100)
print(
    "Unanimous Training → 3:1 Disagreement Testing"
)
print(
    "CNN：MobileNetV3-Small"
)
print("=" * 100)

print(
    "Device：",
    DEVICE
)


# ============================================================
# 9. 讀取醫師標註
# ============================================================

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
# 10. 統一 image_id 格式
# ============================================================

for dataframe in [
    votes,
    folds,
]:

    dataframe["image_id"] = (
        dataframe[
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


# ============================================================
# 11. positive_votes
# ============================================================

if "positive_votes" not in votes.columns:

    doctor_columns = [
        "D1",
        "D2",
        "D3",
        "D4",
    ]

    votes["positive_votes"] = (
        votes[
            doctor_columns
        ]
        .sum(
            axis=1
        )
    )


# ============================================================
# 12. 醫師一致型態
# ============================================================

def get_agreement_type(
    positive_votes
):

    if positive_votes == 4:

        return (
            "unanimous_positive"
        )

    elif positive_votes == 0:

        return (
            "unanimous_negative"
        )

    elif positive_votes == 3:

        return (
            "three_one_positive"
        )

    elif positive_votes == 1:

        return (
            "three_one_negative"
        )

    elif positive_votes == 2:

        return (
            "two_two"
        )

    else:

        return (
            "unknown"
        )


votes["agreement_detail"] = (
    votes[
        "positive_votes"
    ]
    .apply(
        get_agreement_type
    )
)


# ============================================================
# 13. Majority Ground Truth
#
# 3 或 4 位 positive → 1
# 0 或 1 位 positive → 0
# 2:2               → -1
# ============================================================

votes["majority_label"] = np.select(

    [
        votes[
            "positive_votes"
        ]
        >=
        3,

        votes[
            "positive_votes"
        ]
        <=
        1,

        votes[
            "positive_votes"
        ]
        ==
        2,
    ],

    [
        1,
        0,
        -1,
    ],

    default=-1
)


# ============================================================
# 14. 建立所有合法 image_id
# ============================================================

ALL_IMAGE_IDS = (
    votes[
        "image_id"
    ]
    .drop_duplicates()
    .astype(str)
    .tolist()
)


# 先從較長 ID 開始找
# 避免 1 誤判成 11、101 等
ALL_IMAGE_IDS = sorted(

    ALL_IMAGE_IDS,

    key=lambda x: (
        len(
            str(
                x
            )
        ),
        str(
            x
        )
    ),

    reverse=True
)


# ============================================================
# 15. 從 filename 判斷 image_id
# ============================================================

def detect_image_id(
    path
):

    filename = (
        path.name
    )


    for image_id in ALL_IMAGE_IDS:

        pattern = (
            r"(?<!\d)"
            +
            re.escape(
                str(
                    image_id
                )
            )
            +
            r"(?!\d)"
        )


        if re.search(
            pattern,
            filename
        ):

            return str(
                image_id
            )


    return None


# ============================================================
# 16. 建立五區影像索引
#
# 你的實際資料夾：
#
# output/
#   tip_middle_root_1_2_2/
#       regions/
#           舌尖/
#           舌中/
#           舌左邊/
#           舌右邊/
#           舌根/
#
#   tip_middle_root_1_3_1/
#       regions/
#           ...
#
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


image_records = []


for segmentation_method in (
    SEGMENTATION_METHODS
):

    method_dir = (
        OUTPUT_DIR
        /
        segmentation_method
        /
        "regions"
    )


    if not method_dir.exists():

        raise FileNotFoundError(
            f"找不到資料夾：{method_dir}"
        )


    for region in REGIONS:

        region_dir = (
            method_dir
            /
            region
        )


        if not region_dir.exists():

            raise FileNotFoundError(
                f"找不到舌區資料夾：{region_dir}"
            )


        region_files = [

            path

            for path in (
                region_dir.iterdir()
            )

            if (
                path.is_file()
                and
                path.suffix.lower()
                in
                IMAGE_EXTENSIONS
            )
        ]


        print(
            f"{segmentation_method} | "
            f"{region} | "
            f"{len(region_files)} 張"
        )


        for path in region_files:

            image_id = (
                detect_image_id(
                    path
                )
            )


            if image_id is None:

                print(
                    "⚠ 無法辨認 image_id：",
                    path
                )

                continue


            image_records.append({

                "segmentation_method":
                    segmentation_method,

                "image_id":
                    image_id,

                "region":
                    region,

                "image_path":
                    str(
                        path
                    ),
            })


# ============================================================
# 17. 建立 DataFrame
# ============================================================

image_index = pd.DataFrame(
    image_records
)


if image_index.empty:

    raise FileNotFoundError(
        "沒有成功建立任何五區影像索引。"
    )


# ============================================================
# 18. 檢查重複
# ============================================================

duplicate_keys = [
    "segmentation_method",
    "image_id",
    "region",
]


duplicates = (
    image_index
    .duplicated(
        subset=duplicate_keys,
        keep=False
    )
)


if duplicates.any():

    duplicate_df = (
        image_index[
            duplicates
        ]
        .sort_values(
            duplicate_keys
        )
    )


    print("\n")
    print(
        "發現重複影像："
    )


    print(
        duplicate_df
        .head(
            30
        )
        .to_string(
            index=False
        )
    )


    raise ValueError(
        "同一 segmentation / image_id / region "
        "出現超過一張影像。"
    )


# ============================================================
# 19. 檢查每種分區是否恰好 500 張
# ============================================================

print("\n")
print("=" * 100)
print(
    "五區影像索引檢查"
)
print("=" * 100)


print(
    "總 region images：",
    len(
        image_index
    )
)


method_counts = (
    image_index
    .groupby(
        "segmentation_method"
    )
    .size()
)


print(
    method_counts
)


for segmentation_method in (
    SEGMENTATION_METHODS
):

    n_images = int(

        np.sum(
            image_index[
                "segmentation_method"
            ]
            ==
            segmentation_method
        )
    )


    if n_images != 500:

        raise ValueError(
            f"{segmentation_method} "
            f"找到 {n_images} 張五區影像，"
            "理論上應為 500 張。"
            "請先停止，不要繼續訓練 CNN。"
        )


# ============================================================
# 20. 檢查每種分區是否有 100 位病人
# ============================================================

for segmentation_method in (
    SEGMENTATION_METHODS
):

    subset = (
        image_index[
            image_index[
                "segmentation_method"
            ]
            ==
            segmentation_method
        ]
    )


    n_patients = (
        subset[
            "image_id"
        ]
        .nunique()
    )


    print(
        f"{segmentation_method} "
        f"病人數：{n_patients}"
    )


    if n_patients != 100:

        raise ValueError(
            f"{segmentation_method} "
            f"只有 {n_patients} 位病人，"
            "理論上應為 100。"
        )


print(
    "✓ 五區影像檢查完成"
)


# ============================================================
# 21. Pad To Square
#
# 保留原始比例，
# 先補成正方形再 resize
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


        left = (
            size
            -
            width
        ) // 2


        top = (
            size
            -
            height
        ) // 2


        right = (
            size
            -
            width
            -
            left
        )


        bottom = (
            size
            -
            height
            -
            top
        )


        return ImageOps.expand(

            image,

            border=(
                left,
                top,
                right,
                bottom
            ),

            fill=(
                0,
                0,
                0
            )
        )


# ============================================================
# 22. ImageNet normalization
# ============================================================

normalization = (
    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406,
        ],

        std=[
            0.229,
            0.224,
            0.225,
        ]
    )
)


# ============================================================
# 23. Training Transform
#
# IMPORTANT：
# 不使用 ColorJitter
#
# 因為研究目標就是舌質顏色，
# 不能人為改變影像色彩。
# ============================================================

train_transform = (
    transforms.Compose(
        [
            PadToSquare(),

            transforms.Resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE
                )
            ),

            transforms.RandomRotation(
                degrees=5,
                fill=0
            ),

            transforms.RandomHorizontalFlip(
                p=0.5
            ),

            transforms.ToTensor(),

            normalization,
        ]
    )
)


# ============================================================
# 24. Test Transform
# ============================================================

test_transform = (
    transforms.Compose(
        [
            PadToSquare(),

            transforms.Resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE
                )
            ),

            transforms.ToTensor(),

            normalization,
        ]
    )
)


# ============================================================
# 25. Dataset
# ============================================================

class TongueDataset(
    Dataset
):

    def __init__(
        self,
        dataframe,
        transform
    ):

        self.dataframe = (
            dataframe
            .reset_index(
                drop=True
            )
            .copy()
        )

        self.transform = (
            transform
        )


    def __len__(
        self
    ):

        return len(
            self.dataframe
        )


    def __getitem__(
        self,
        index
    ):

        row = (
            self.dataframe.iloc[
                index
            ]
        )


        image_path = (
            row[
                "image_path"
            ]
        )


        image = (
            Image.open(
                image_path
            )
            .convert(
                "RGB"
            )
        )


        image = (
            self.transform(
                image
            )
        )


        label = torch.tensor(

            float(
                row[
                    "majority_label"
                ]
            ),

            dtype=torch.float32
        )


        return (
            image,
            label,
            index
        )


# ============================================================
# 26. MobileNetV3-Small
# ============================================================

def build_model():

    weights = (
        models
        .MobileNet_V3_Small_Weights
        .DEFAULT
    )


    model = (
        models.mobilenet_v3_small(
            weights=weights
        )
    )


    # --------------------------------------------------------
    # Stage 1：
    # 先 freeze feature extractor
    # --------------------------------------------------------

    for parameter in (
        model.features.parameters()
    ):

        parameter.requires_grad = (
            False
        )


    # --------------------------------------------------------
    # 最後一層改成 binary classifier
    # --------------------------------------------------------

    in_features = (
        model.classifier[
            3
        ]
        .in_features
    )


    model.classifier[
        3
    ] = nn.Linear(
        in_features,
        1
    )


    return model.to(
        DEVICE
    )


# ============================================================
# 27. Stage 2：
# 解凍最後兩個 feature blocks
# ============================================================

def unfreeze_last_two_blocks(
    model
):

    blocks = list(
        model.features.children()
    )


    for block in blocks[
        -2:
    ]:

        for parameter in (
            block.parameters()
        ):

            parameter.requires_grad = (
                True
            )


# ============================================================
# 28. Training
# ============================================================

def train_epochs(
    model,
    loader,
    criterion,
    optimizer,
    epochs
):

    for epoch in range(
        epochs
    ):

        model.train()

        total_loss = 0.0
        n_samples = 0


        for (
            images,
            labels,
            _
        ) in loader:

            images = (
                images.to(
                    DEVICE
                )
            )


            labels = (
                labels.to(
                    DEVICE
                )
            )


            optimizer.zero_grad()


            logits = (
                model(
                    images
                )
                .squeeze(
                    1
                )
            )


            loss = (
                criterion(
                    logits,
                    labels
                )
            )


            loss.backward()

            optimizer.step()


            total_loss += (
                loss.item()
                *
                len(
                    labels
                )
            )


            n_samples += len(
                labels
            )


        mean_loss = (
            total_loss
            /
            n_samples
        )


        print(
            f"      epoch "
            f"{epoch + 1}/{epochs} "
            f"loss={mean_loss:.4f}"
        )


# ============================================================
# 29. Prediction
# ============================================================

def predict(
    model,
    loader,
    dataframe
):

    model.eval()


    records = []


    with torch.no_grad():

        for (
            images,
            labels,
            indices
        ) in loader:

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


            probabilities = (
                torch.sigmoid(
                    logits
                )
                .cpu()
                .numpy()
            )


            predictions = (
                probabilities
                >=
                0.5
            ).astype(
                int
            )


            labels_numpy = (
                labels
                .numpy()
                .astype(
                    int
                )
            )


            indices_numpy = (
                indices
                .numpy()
            )


            for (
                row_index,
                true_value,
                pred_value,
                probability
            ) in zip(
                indices_numpy,
                labels_numpy,
                predictions,
                probabilities
            ):

                original = (
                    dataframe.iloc[
                        int(
                            row_index
                        )
                    ]
                )


                records.append({

                    "image_id":
                        original[
                            "image_id"
                        ],

                    "region":
                        original[
                            "region"
                        ],

                    "positive_votes":
                        int(
                            original[
                                "positive_votes"
                            ]
                        ),

                    "agreement_detail":
                        original[
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

                    "probability":
                        float(
                            probability
                        ),
                })


    return pd.DataFrame(
        records
    )


# ============================================================
# 30. Metrics
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
# 31. 主分析
# ============================================================

prediction_frames = []

fold_metric_records = []


for color in PRIMARY_COLORS:

    print("\n")
    print("=" * 100)
    print(
        f"PRIMARY COLOR：{color}"
    )
    print("=" * 100)


    # --------------------------------------------------------
    # 該顏色的醫師結果
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # 該顏色的 patient-level fold
    # --------------------------------------------------------

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
    # 兩種 segmentation
    # ========================================================

    for segmentation_method in (
        SEGMENTATION_METHODS
    ):

        print("\n")
        print(
            segmentation_method
        )


        region_images = (

            image_index[
                image_index[
                    "segmentation_method"
                ]
                ==
                segmentation_method
            ]
            .copy()
        )


        # ----------------------------------------------------
        # 圖片 + 醫師標註
        # ----------------------------------------------------

        data = (

            region_images
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


        if len(
            data
        ) != 500:

            raise ValueError(

                f"{segmentation_method} / {color}: "
                f"merge 後只有 {len(data)} rows，"
                "理論上應為 500。"
            )


        # ====================================================
        # 五個 fold
        # ====================================================

        for fold_id in sorted(

            color_folds[
                "fold"
            ]
            .unique()
        ):

            fold_id = int(
                fold_id
            )


            print("\n")
            print(
                f"  Fold {fold_id}"
            )


            # ------------------------------------------------
            # Test patient IDs
            # ------------------------------------------------

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


            # ------------------------------------------------
            # Train patient IDs
            # ------------------------------------------------

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
                    f"Patient leakage detected："
                    f"{overlap}"
                )


            # =================================================
            # TRAIN：
            #
            # 只使用 held-in patients
            # 且四位醫師完全一致
            #
            # 4:0 positive
            # 0:4 negative
            # =================================================

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
                .reset_index(
                    drop=True
                )
            )


            # =================================================
            # TEST：
            #
            # held-out patients
            #
            # 只測 3:1 disagreement
            #
            # 3 positive → Ground Truth 1
            # 1 positive → Ground Truth 0
            # =================================================

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
                .reset_index(
                    drop=True
                )
            )


            # =================================================
            # Train / Test counts
            # =================================================

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


            print(
                f"    Train={len(train_df)} "
                f"(pos={train_positive}, "
                f"neg={train_negative})"
            )


            print(
                f"    Test={len(test_df)} "
                f"(pos={test_positive}, "
                f"neg={test_negative})"
            )


            # ------------------------------------------------
            # Sanity check
            # ------------------------------------------------

            if (
                train_positive == 0
                or
                train_negative == 0
            ):

                raise ValueError(
                    f"{color} / "
                    f"{segmentation_method} / "
                    f"fold {fold_id}: "
                    "Training set 缺少其中一類。"
                )


            if (
                test_positive == 0
                or
                test_negative == 0
            ):

                raise ValueError(
                    f"{color} / "
                    f"{segmentation_method} / "
                    f"fold {fold_id}: "
                    "Test set 缺少其中一類。"
                )


            # =================================================
            # Dataset
            # =================================================

            train_dataset = (
                TongueDataset(
                    train_df,
                    train_transform
                )
            )


            test_dataset = (
                TongueDataset(
                    test_df,
                    test_transform
                )
            )


            # =================================================
            # DataLoader
            # =================================================

            generator = (
                torch.Generator()
            )


            generator.manual_seed(
                SEED
                +
                fold_id
            )


            train_loader = (
                DataLoader(

                    train_dataset,

                    batch_size=
                        BATCH_SIZE,

                    shuffle=
                        True,

                    num_workers=
                        0,

                    generator=
                        generator
                )
            )


            test_loader = (
                DataLoader(

                    test_dataset,

                    batch_size=
                        BATCH_SIZE,

                    shuffle=
                        False,

                    num_workers=
                        0
                )
            )


            # =================================================
            # Class imbalance：
            #
            # pos_weight = negative / positive
            # =================================================

            pos_weight_value = (

                train_negative
                /
                train_positive
            )


            print(
                f"    pos_weight="
                f"{pos_weight_value:.4f}"
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
                    pos_weight=
                        pos_weight
                )
            )


            # =================================================
            # Model
            # =================================================

            set_seed(
                SEED
                +
                fold_id
            )


            model = (
                build_model()
            )


            # =================================================
            # Stage 1
            #
            # 只訓練 classifier head
            # =================================================

            print(
                "    Stage 1：classifier head"
            )


            optimizer = (
                torch.optim.Adam(

                    filter(
                        lambda p:
                        p.requires_grad,

                        model.parameters()
                    ),

                    lr=
                        STAGE1_LR
                )
            )


            train_epochs(

                model,

                train_loader,

                criterion,

                optimizer,

                STAGE1_EPOCHS
            )


            # =================================================
            # Stage 2
            #
            # 解凍最後兩個 feature blocks
            # =================================================

            print(
                "    Stage 2：last 2 feature blocks"
            )


            unfreeze_last_two_blocks(
                model
            )


            optimizer = (
                torch.optim.Adam(

                    filter(
                        lambda p:
                        p.requires_grad,

                        model.parameters()
                    ),

                    lr=
                        STAGE2_LR
                )
            )


            train_epochs(

                model,

                train_loader,

                criterion,

                optimizer,

                STAGE2_EPOCHS
            )


            # =================================================
            # Prediction
            # =================================================

            fold_predictions = (
                predict(

                    model,

                    test_loader,

                    test_df
                )
            )


            metrics = (
                calculate_metrics(

                    fold_predictions[
                        "true"
                    ],

                    fold_predictions[
                        "pred"
                    ]
                )
            )


            print(
                f"    F1="
                f"{metrics['f1']:.4f}"
                f" | "
                f"BA="
                f"{metrics['balanced_accuracy']:.4f}"
                f" | "
                f"Accuracy="
                f"{metrics['accuracy']:.4f}"
            )


            # =================================================
            # Prediction metadata
            # =================================================

            fold_predictions[
                "color"
            ] = (
                color
            )


            fold_predictions[
                "analysis_type"
            ] = (
                "primary"
            )


            fold_predictions[
                "segmentation_method"
            ] = (
                segmentation_method
            )


            fold_predictions[
                "model"
            ] = (
                "CNN"
            )


            fold_predictions[
                "fold"
            ] = (
                fold_id
            )


            prediction_frames.append(
                fold_predictions
            )


            # =================================================
            # Fold metrics
            # =================================================

            fold_metric_records.append({

                "color":
                    color,

                "analysis_type":
                    "primary",

                "segmentation_method":
                    segmentation_method,

                "model":
                    "CNN",

                "fold":
                    fold_id,

                "train_items":
                    len(
                        train_df
                    ),

                "train_positive":
                    train_positive,

                "train_negative":
                    train_negative,

                "test_items":
                    len(
                        test_df
                    ),

                "test_positive":
                    test_positive,

                "test_negative":
                    test_negative,

                **metrics,
            })


            # =================================================
            # 清除模型記憶體
            # =================================================

            del model


            if DEVICE.type == "mps":

                torch.mps.empty_cache()


            elif DEVICE.type == "cuda":

                torch.cuda.empty_cache()


# ============================================================
# 32. 合併 Predictions
# ============================================================

predictions_df = pd.concat(
    prediction_frames,
    ignore_index=True
)


fold_metrics_df = pd.DataFrame(
    fold_metric_records
)


# ============================================================
# 33. 檢查 OOF Prediction 重複
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
        subset=
            duplicate_columns,
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
        .to_string(
            index=False
        )
    )


    raise ValueError(
        "CNN OOF predictions 出現重複。"
    )


# ============================================================
# 34. Color-level OOF Metrics
# ============================================================

color_metric_records = []


for (
    color,
    segmentation_method
), group in predictions_df.groupby(

    [
        "color",
        "segmentation_method",
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
            "primary",

        "segmentation_method":
            segmentation_method,

        "model":
            "CNN",

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


cnn_color_metrics = pd.DataFrame(
    color_metric_records
)


# ============================================================
# 35. CNN Primary Summary
# ============================================================

cnn_primary_summary = (

    cnn_color_metrics
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


# ============================================================
# 36. CNN 輸出檔
# ============================================================

PREDICTION_OUTPUT = (
    OUTPUT_DIR
    /
    "unanimous_to_disagreement_cnn_predictions.csv"
)


FOLD_OUTPUT = (
    OUTPUT_DIR
    /
    "unanimous_to_disagreement_cnn_fold_metrics.csv"
)


COLOR_OUTPUT = (
    OUTPUT_DIR
    /
    "unanimous_to_disagreement_cnn_color_metrics.csv"
)


CNN_SUMMARY_OUTPUT = (
    OUTPUT_DIR
    /
    "unanimous_to_disagreement_cnn_primary_summary.csv"
)


predictions_df.to_csv(
    PREDICTION_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


fold_metrics_df.to_csv(
    FOLD_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


cnn_color_metrics.to_csv(
    COLOR_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


cnn_primary_summary.to_csv(
    CNN_SUMMARY_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 37. 合併 SVM + RF + CNN
# ============================================================

ALL_COLOR_OUTPUT = None
ALL_SUMMARY_OUTPUT = None


if SVM_RF_COLOR_FILE.exists():

    svm_rf_color = pd.read_csv(
        SVM_RF_COLOR_FILE
    )


    # --------------------------------------------------------
    # 只取 primary：
    # 淡紅 + 鮮紅
    # --------------------------------------------------------

    svm_rf_primary = (

        svm_rf_color[
            svm_rf_color[
                "analysis_type"
            ]
            ==
            "primary"
        ]
        .copy()
    )


    # --------------------------------------------------------
    # 合併三種模型
    # --------------------------------------------------------

    all_color_metrics = pd.concat(

        [
            svm_rf_primary,
            cnn_color_metrics,
        ],

        ignore_index=True
    )


    # ========================================================
    # 三模型 macro summary
    # ========================================================

    all_primary_summary = (

        all_color_metrics
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


    # --------------------------------------------------------
    # F1 排名
    # --------------------------------------------------------

    all_primary_summary[
        "rank_by_macro_f1"
    ] = (

        all_primary_summary[
            "macro_f1"
        ]
        .rank(
            ascending=False,
            method="min"
        )
        .astype(
            int
        )
    )


    all_primary_summary = (

        all_primary_summary
        .sort_values(
            "rank_by_macro_f1"
        )
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # 輸出
    # ========================================================

    ALL_COLOR_OUTPUT = (
        OUTPUT_DIR
        /
        "unanimous_to_disagreement_all_models_color_metrics.csv"
    )


    ALL_SUMMARY_OUTPUT = (
        OUTPUT_DIR
        /
        "unanimous_to_disagreement_all_models_primary_summary.csv"
    )


    all_color_metrics.to_csv(
        ALL_COLOR_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )


    all_primary_summary.to_csv(
        ALL_SUMMARY_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 38. Terminal：CNN Color-level Results
# ============================================================

print("\n")
print("=" * 100)
print(
    "CNN Color-level OOF Results"
)
print("=" * 100)


display_color = (

    cnn_color_metrics[
        [
            "color",
            "segmentation_method",
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
# 39. Terminal：三模型 Primary Results
# ============================================================

print("\n")
print("=" * 100)
print(
    "ALL THREE MODELS：PRIMARY RESULTS"
)
print("=" * 100)


if SVM_RF_COLOR_FILE.exists():

    display_all = (
        all_primary_summary
        .copy()
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


else:

    print(
        "找不到 SVM/RF color metrics。"
    )

    print(
        "目前只完成 CNN。"
    )


# ============================================================
# 40. 輸出檔案
# ============================================================

print("\n")
print("=" * 100)
print(
    "輸出檔案"
)
print("=" * 100)


print(
    PREDICTION_OUTPUT
)

print(
    FOLD_OUTPUT
)

print(
    COLOR_OUTPUT
)

print(
    CNN_SUMMARY_OUTPUT
)


if ALL_COLOR_OUTPUT is not None:

    print(
        ALL_COLOR_OUTPUT
    )


if ALL_SUMMARY_OUTPUT is not None:

    print(
        ALL_SUMMARY_OUTPUT
    )


print("=" * 100)
