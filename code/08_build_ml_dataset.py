from pathlib import Path
import pandas as pd


# ============================================================
# 1. 路徑設定
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

CONSENSUS_FILE = OUTPUT_DIR / "consensus_long.csv"


# ============================================================
# 2. 兩種舌區分法
# ============================================================

SEGMENTATION_METHODS = [
    "tip_middle_root_1_2_2",
    "tip_middle_root_1_3_1",
]


# ============================================================
# 3. 五個舌區與六種舌色
# ============================================================

REGIONS = [
    "舌尖",
    "舌中",
    "舌左邊",
    "舌右邊",
    "舌根",
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
# 4. 讀取四位醫師共識
# ============================================================

consensus = pd.read_csv(
    CONSENSUS_FILE,
    dtype={"image_id": str}
)

consensus["image_id"] = (
    consensus["image_id"]
    .astype(str)
    .str.strip()
)

print("=" * 70)
print("四位醫師共識資料")
print("=" * 70)

print(
    "資料列數：",
    len(consensus)
)

print(
    "照片數：",
    consensus["image_id"].nunique()
)


# ============================================================
# 5. 將舊格式 label 拆成 region + color
#
# 例如：
# 舌尖_淡紅
#
# ↓
#
# region = 舌尖
# color  = 淡紅
# ============================================================

def parse_label(label):

    label = str(label).strip()

    for region in REGIONS:

        prefix = region + "_"

        if label.startswith(prefix):

            color = label[
                len(prefix):
            ]

            return pd.Series(
                [
                    region,
                    color
                ]
            )

    return pd.Series(
        [
            None,
            None
        ]
    )


consensus[
    ["region", "color"]
] = consensus[
    "label"
].apply(
    parse_label
)


# ============================================================
# 6. 只保留本研究六種舌質色
# ============================================================

consensus = consensus[
    consensus["color"].isin(
        COLORS
    )
].copy()


# ============================================================
# 7. 共識標籤轉成 Wide Format
#
# 每一列：
# image_id + region
#
# 六個輸出：
# y_淡紅
# y_淡白
# y_鮮紅
# y_暗紅
# y_青紫
# y_灰黑
#
# 1  = 至少 3 位醫師認為有
# 0  = 至少 3 位醫師認為沒有
# -1 = 2:2 無共識
# ============================================================

target_table = (
    consensus
    .pivot_table(
        index=[
            "image_id",
            "region"
        ],
        columns="color",
        values="consensus",
        aggfunc="first"
    )
    .reset_index()
)

target_table.columns.name = None


# 確保六種顏色都存在
for color in COLORS:

    if color not in target_table.columns:

        target_table[
            color
        ] = pd.NA


# 加上 y_ 前綴
target_table = (
    target_table.rename(
        columns={
            color: f"y_{color}"
            for color in COLORS
        }
    )
)


print("\n")
print("=" * 70)
print("整理完成的 Ground Truth")
print("=" * 70)

print(
    "資料列數：",
    len(target_table)
)

print(
    "照片數：",
    target_table["image_id"].nunique()
)


# ============================================================
# 8. 處理單一分區方法
# ============================================================

def build_dataset(
    method_name
):

    print("\n")
    print("=" * 70)
    print(
        "建立 ML Dataset：",
        method_name
    )
    print("=" * 70)


    # --------------------------------------------------------
    # 讀取該分區方法的色彩特徵
    # --------------------------------------------------------

    method_dir = (
        OUTPUT_DIR
        / method_name
    )

    feature_file = (
        method_dir
        / "color_features.csv"
    )


    if not feature_file.exists():

        print(
            "⚠ 找不到：",
            feature_file
        )

        return None


    features = pd.read_csv(
        feature_file,
        dtype={
            "image_id": str
        }
    )


    features["image_id"] = (
        features["image_id"]
        .astype(str)
        .str.strip()
    )


    features["region"] = (
        features["region"]
        .astype(str)
        .str.strip()
    )


    print(
        "影像特徵資料列：",
        len(features)
    )

    print(
        "照片數：",
        features[
            "image_id"
        ].nunique()
    )


    # ========================================================
    # 9. 合併影像特徵與醫師共識
    # ========================================================

    dataset = features.merge(

        target_table,

        on=[
            "image_id",
            "region"
        ],

        how="inner",

        validate="one_to_one"
    )


    print(
        "成功合併資料列：",
        len(dataset)
    )


    print(
        "成功合併照片：",
        dataset[
            "image_id"
        ].nunique()
    )


    # ========================================================
    # 10. 檢查是否有影像區域沒配到共識
    # ========================================================

    feature_keys = set(
        zip(
            features[
                "image_id"
            ],
            features[
                "region"
            ]
        )
    )


    dataset_keys = set(
        zip(
            dataset[
                "image_id"
            ],
            dataset[
                "region"
            ]
        )
    )


    missing_keys = sorted(
        feature_keys
        -
        dataset_keys
    )


    if missing_keys:

        print(
            "\n⚠ 有影像區域沒有配到醫師共識：",
            len(missing_keys)
        )

        for key in missing_keys[:20]:

            print(
                " -",
                key
            )

    else:

        print(
            "✓ 所有影像區域都有醫師共識資料"
        )


    # ========================================================
    # 11. 找影像特徵欄位
    # ========================================================

    target_columns = [
        f"y_{color}"
        for color in COLORS
    ]


    non_feature_columns = [

        "segmentation_method",

        "image_id",

        "region",

        *target_columns
    ]


    feature_columns = [

        column

        for column
        in dataset.columns

        if column
        not in non_feature_columns
    ]


    # ========================================================
    # 12. 檢查 Feature NaN
    # ========================================================

    feature_nan_count = int(

        dataset[
            feature_columns
        ]
        .isna()
        .sum()
        .sum()
    )


    # ========================================================
    # 13. Ground Truth 分布
    # ========================================================

    print("\n")
    print("-" * 70)
    print("六種舌質色分布")
    print("-" * 70)


    summary_records = []


    for color in COLORS:

        column = (
            f"y_{color}"
        )


        counts = (
            dataset[
                column
            ]
            .value_counts(
                dropna=False
            )
            .to_dict()
        )


        positive = int(
            counts.get(
                1,
                0
            )
        )

        negative = int(
            counts.get(
                0,
                0
            )
        )

        uncertain = int(
            counts.get(
                -1,
                0
            )
        )


        summary_records.append({

            "segmentation_method":
                method_name,

            "color":
                color,

            "positive":
                positive,

            "negative":
                negative,

            "uncertain":
                uncertain,

            "usable":
                positive
                +
                negative,
        })


        print(

            f"{color}: "

            f"陽性={positive}, "

            f"陰性={negative}, "

            f"2:2無共識={uncertain}, "

            f"可訓練={positive + negative}"
        )


    summary_df = pd.DataFrame(
        summary_records
    )


    # ========================================================
    # 14. 排序
    # ========================================================

    dataset = (
        dataset
        .sort_values(
            [
                "image_id",
                "region"
            ]
        )
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # 15. 輸出 ml_dataset.csv
    # ========================================================

    dataset_output = (
        method_dir
        / "ml_dataset.csv"
    )


    dataset.to_csv(

        dataset_output,

        index=False,

        encoding="utf-8-sig"
    )


    # ========================================================
    # 16. 輸出標籤統計
    # ========================================================

    summary_output = (
        method_dir
        / "ml_target_summary.csv"
    )


    summary_df.to_csv(

        summary_output,

        index=False,

        encoding="utf-8-sig"
    )


    # ========================================================
    # 17. 最終結果
    # ========================================================

    print("\n")
    print("-" * 70)

    print(
        "分區方法：",
        method_name
    )

    print(
        "資料列數：",
        len(dataset)
    )

    print(
        "照片數：",
        dataset[
            "image_id"
        ].nunique()
    )

    print(
        "影像特徵數：",
        len(
            feature_columns
        )
    )

    print(
        "Ground Truth 數：",
        len(
            target_columns
        )
    )

    print(
        "影像特徵 NaN：",
        feature_nan_count
    )

    print(
        "輸出：",
        dataset_output
    )

    print("-" * 70)


    return dataset


# ============================================================
# 18. 自動建立兩種分區方法的 ML Dataset
# ============================================================

all_datasets = []


for method_name in SEGMENTATION_METHODS:

    dataset = build_dataset(
        method_name
    )

    if dataset is not None:

        all_datasets.append(
            dataset
        )


# ============================================================
# 19. 額外建立兩種方法合併總表
# ============================================================

if all_datasets:

    combined_dataset = (
        pd.concat(
            all_datasets,
            ignore_index=True
        )
    )


    combined_output = (
        OUTPUT_DIR
        /
        "ml_dataset_all_methods.csv"
    )


    combined_dataset.to_csv(

        combined_output,

        index=False,

        encoding="utf-8-sig"
    )


    print("\n")
    print("=" * 70)
    print("兩種分區方法 ML Dataset 全部完成")
    print("=" * 70)

    print(
        "總資料列：",
        len(
            combined_dataset
        )
    )

    print(
        "分區方法：",
        combined_dataset[
            "segmentation_method"
        ].nunique()
    )

    print(
        "不同照片：",
        combined_dataset[
            "image_id"
        ].nunique()
    )

    print(
        "總表：",
        combined_output
    )

    print("=" * 70)