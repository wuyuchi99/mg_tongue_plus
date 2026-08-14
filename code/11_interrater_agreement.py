from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. 專案路徑
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"


# ============================================================
# 2. 四位醫師 Label 檔
# ============================================================

DOCTOR_FILES = {
    "D1": OUTPUT_DIR / "D1_labels.csv",
    "D2": OUTPUT_DIR / "D2_labels.csv",
    "D3": OUTPUT_DIR / "D3_labels.csv",
    "D4": OUTPUT_DIR / "D4_labels.csv",
}


# ============================================================
# 3. 五個舌區、六種舌色
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
# 4. 讀取四位醫師資料
# ============================================================

doctor_data = {}


for doctor, file_path in DOCTOR_FILES.items():

    if not file_path.exists():

        raise FileNotFoundError(
            f"找不到 {file_path}\n"
            "請先執行 02_prepare_labels.py"
        )

    df = pd.read_csv(
        file_path,
        dtype={"image_id": str}
    )

    df["image_id"] = (
        df["image_id"]
        .astype(str)
        .str.strip()
    )

    # 確認沒有重複病例
    if df["image_id"].duplicated().any():

        duplicates = (
            df[
                df["image_id"].duplicated(
                    keep=False
                )
            ]["image_id"]
            .tolist()
        )

        raise ValueError(
            f"{doctor} 有重複 image_id："
            f"{duplicates}"
        )

    doctor_data[doctor] = (
        df.set_index(
            "image_id"
        )
    )


print("=" * 75)
print("四位醫師舌質色一致性分析")
print("=" * 75)


for doctor, df in doctor_data.items():

    print(
        doctor,
        "病例數：",
        len(df)
    )


# ============================================================
# 5. 確認四位醫師病例完全一致
# ============================================================

reference_ids = set(
    doctor_data["D1"].index
)


for doctor in [
    "D2",
    "D3",
    "D4"
]:

    ids = set(
        doctor_data[
            doctor
        ].index
    )

    if ids != reference_ids:

        missing = (
            reference_ids
            -
            ids
        )

        extra = (
            ids
            -
            reference_ids
        )

        raise ValueError(
            f"{doctor} 病例編號與 D1 不一致\n"
            f"缺少：{sorted(missing)}\n"
            f"多出：{sorted(extra)}"
        )


print(
    "✓ 四位醫師病例編號完全一致"
)


# ============================================================
# 6. Fleiss' Kappa
#
# 每一個 item 有 4 位醫師
# 每位醫師判斷：
#
# 0 = 無此舌色
# 1 = 有此舌色
#
# 不直接使用 consensus，
# 而是使用四位醫師原始判斷。
# ============================================================

def fleiss_kappa_binary(
    ratings
):

    """
    ratings:
        shape = (N, 4)

    每一列：
        一個 image-region-color item

    每一欄：
        一位醫師

    值只能是 0 或 1
    """

    ratings = np.asarray(
        ratings,
        dtype=int
    )


    if ratings.ndim != 2:

        raise ValueError(
            "ratings 必須是二維矩陣"
        )


    n_items = (
        ratings.shape[0]
    )

    n_raters = (
        ratings.shape[1]
    )


    if n_items == 0:

        return np.nan


    if n_raters < 2:

        return np.nan


    # --------------------------------------------------------
    # 每個 item 有多少醫師投 0 / 1
    # --------------------------------------------------------

    n_positive = (
        ratings.sum(
            axis=1
        )
    )

    n_negative = (
        n_raters
        -
        n_positive
    )


    # --------------------------------------------------------
    # 每個 item 的 observed agreement
    #
    # P_i =
    # [n0(n0-1) + n1(n1-1)]
    # /
    # [n(n-1)]
    # --------------------------------------------------------

    p_item = (

        (
            n_negative
            *
            (
                n_negative
                -
                1
            )
        )

        +

        (
            n_positive
            *
            (
                n_positive
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
        p_item.mean()
    )


    # --------------------------------------------------------
    # 全體 positive / negative 比例
    # --------------------------------------------------------

    total_ratings = (
        n_items
        *
        n_raters
    )


    p_positive = (
        n_positive.sum()
        /
        total_ratings
    )


    p_negative = (
        n_negative.sum()
        /
        total_ratings
    )


    # --------------------------------------------------------
    # Chance agreement
    # --------------------------------------------------------

    expected_agreement = (

        p_positive ** 2

        +

        p_negative ** 2

    )


    # --------------------------------------------------------
    # Fleiss Kappa
    # --------------------------------------------------------

    denominator = (
        1
        -
        expected_agreement
    )


    if np.isclose(
        denominator,
        0
    ):

        kappa = np.nan

    else:

        kappa = (

            observed_agreement
            -
            expected_agreement

        ) / denominator


    return float(
        kappa
    )


# ============================================================
# 7. 單一 Label 的 Agreement 統計
# ============================================================

def calculate_agreement(
    ratings
):

    ratings = np.asarray(
        ratings,
        dtype=int
    )


    vote_count = (
        ratings.sum(
            axis=1
        )
    )


    n_items = len(
        vote_count
    )


    # --------------------------------------------------------
    # 四位完全一致
    #
    # 0:4 或 4:0
    # --------------------------------------------------------

    unanimous = int(
        np.sum(
            (vote_count == 0)
            |
            (vote_count == 4)
        )
    )


    # --------------------------------------------------------
    # 3:1 / 1:3
    # --------------------------------------------------------

    three_one = int(
        np.sum(
            (vote_count == 1)
            |
            (vote_count == 3)
        )
    )


    # --------------------------------------------------------
    # 2:2
    # --------------------------------------------------------

    two_two = int(
        np.sum(
            vote_count == 2
        )
    )


    # --------------------------------------------------------
    # positive vote distribution
    # --------------------------------------------------------

    vote_0 = int(
        np.sum(
            vote_count == 0
        )
    )

    vote_1 = int(
        np.sum(
            vote_count == 1
        )
    )

    vote_2 = int(
        np.sum(
            vote_count == 2
        )
    )

    vote_3 = int(
        np.sum(
            vote_count == 3
        )
    )

    vote_4 = int(
        np.sum(
            vote_count == 4
        )
    )


    # --------------------------------------------------------
    # Fleiss Kappa
    # --------------------------------------------------------

    kappa = (
        fleiss_kappa_binary(
            ratings
        )
    )


    return {

        "n_items":
            n_items,

        "unanimous_n":
            unanimous,

        "unanimous_rate":
            unanimous / n_items
            if n_items > 0
            else np.nan,

        "three_one_n":
            three_one,

        "three_one_rate":
            three_one / n_items
            if n_items > 0
            else np.nan,

        "two_two_n":
            two_two,

        "two_two_rate":
            two_two / n_items
            if n_items > 0
            else np.nan,

        "vote_0":
            vote_0,

        "vote_1":
            vote_1,

        "vote_2":
            vote_2,

        "vote_3":
            vote_3,

        "vote_4":
            vote_4,

        "fleiss_kappa":
            kappa,
    }


# ============================================================
# 8. 建立 image × region × color 原始醫師投票表
# ============================================================

vote_records = []


image_ids = sorted(
    reference_ids
)


for image_id in image_ids:

    for region in REGIONS:

        for color in COLORS:

            label_column = (
                f"{region}_{color}"
            )


            # ------------------------------------------------
            # 確認每位醫師都有這個欄位
            # ------------------------------------------------

            for doctor in [
                "D1",
                "D2",
                "D3",
                "D4"
            ]:

                if (
                    label_column
                    not in
                    doctor_data[
                        doctor
                    ].columns
                ):

                    raise KeyError(
                        f"{doctor} 找不到欄位："
                        f"{label_column}"
                    )


            d1 = int(
                doctor_data[
                    "D1"
                ].loc[
                    image_id,
                    label_column
                ]
            )

            d2 = int(
                doctor_data[
                    "D2"
                ].loc[
                    image_id,
                    label_column
                ]
            )

            d3 = int(
                doctor_data[
                    "D3"
                ].loc[
                    image_id,
                    label_column
                ]
            )

            d4 = int(
                doctor_data[
                    "D4"
                ].loc[
                    image_id,
                    label_column
                ]
            )


            ratings = [
                d1,
                d2,
                d3,
                d4
            ]


            # ------------------------------------------------
            # 安全檢查
            # ------------------------------------------------

            if not all(
                value in [
                    0,
                    1
                ]
                for value
                in ratings
            ):

                raise ValueError(
                    f"{image_id} "
                    f"{label_column} "
                    f"含有非 0/1 值："
                    f"{ratings}"
                )


            vote_count = sum(
                ratings
            )


            if vote_count in [
                0,
                4
            ]:

                agreement_type = (
                    "4:0"
                )

            elif vote_count in [
                1,
                3
            ]:

                agreement_type = (
                    "3:1"
                )

            else:

                agreement_type = (
                    "2:2"
                )


            vote_records.append({

                "image_id":
                    image_id,

                "region":
                    region,

                "color":
                    color,

                "label":
                    label_column,

                "D1":
                    d1,

                "D2":
                    d2,

                "D3":
                    d3,

                "D4":
                    d4,

                "positive_votes":
                    vote_count,

                "agreement_type":
                    agreement_type,
            })


vote_df = pd.DataFrame(
    vote_records
)


vote_output = (
    OUTPUT_DIR
    /
    "interrater_vote_details.csv"
)


vote_df.to_csv(
    vote_output,
    index=False,
    encoding="utf-8-sig"
)


print()
print(
    "原始評分 items：",
    len(
        vote_df
    )
)


# ============================================================
# 9. 每一種舌色的 Agreement
#
# 每個顏色：
# 所有照片 × 五區
# ============================================================

color_records = []


for color in COLORS:

    subset = (
        vote_df[
            vote_df[
                "color"
            ]
            ==
            color
        ]
        .copy()
    )


    ratings = (
        subset[
            [
                "D1",
                "D2",
                "D3",
                "D4"
            ]
        ]
        .to_numpy()
    )


    result = (
        calculate_agreement(
            ratings
        )
    )


    result[
        "color"
    ] = color


    color_records.append(
        result
    )


color_agreement_df = pd.DataFrame(
    color_records
)


# 調整欄位順序
color_agreement_df = (
    color_agreement_df[
        [
            "color",
            "n_items",
            "unanimous_n",
            "unanimous_rate",
            "three_one_n",
            "three_one_rate",
            "two_two_n",
            "two_two_rate",
            "vote_0",
            "vote_1",
            "vote_2",
            "vote_3",
            "vote_4",
            "fleiss_kappa",
        ]
    ]
)


color_output = (
    OUTPUT_DIR
    /
    "interrater_agreement_by_color.csv"
)


color_agreement_df.to_csv(
    color_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 10. 每個舌區 × 每種舌色
# ============================================================

region_color_records = []


for region in REGIONS:

    for color in COLORS:

        subset = (
            vote_df[
                (
                    vote_df[
                        "region"
                    ]
                    ==
                    region
                )
                &
                (
                    vote_df[
                        "color"
                    ]
                    ==
                    color
                )
            ]
            .copy()
        )


        ratings = (
            subset[
                [
                    "D1",
                    "D2",
                    "D3",
                    "D4"
                ]
            ]
            .to_numpy()
        )


        result = (
            calculate_agreement(
                ratings
            )
        )


        result[
            "region"
        ] = region

        result[
            "color"
        ] = color


        region_color_records.append(
            result
        )


region_color_df = pd.DataFrame(
    region_color_records
)


region_color_df = (
    region_color_df[
        [
            "region",
            "color",
            "n_items",
            "unanimous_n",
            "unanimous_rate",
            "three_one_n",
            "three_one_rate",
            "two_two_n",
            "two_two_rate",
            "vote_0",
            "vote_1",
            "vote_2",
            "vote_3",
            "vote_4",
            "fleiss_kappa",
        ]
    ]
)


region_color_output = (
    OUTPUT_DIR
    /
    "interrater_agreement_by_region_color.csv"
)


region_color_df.to_csv(
    region_color_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 11. 各舌區整體 Agreement
#
# 每區把六種顏色一起計算
# ============================================================

region_records = []


for region in REGIONS:

    subset = (
        vote_df[
            vote_df[
                "region"
            ]
            ==
            region
        ]
        .copy()
    )


    ratings = (
        subset[
            [
                "D1",
                "D2",
                "D3",
                "D4"
            ]
        ]
        .to_numpy()
    )


    result = (
        calculate_agreement(
            ratings
        )
    )


    result[
        "region"
    ] = region


    region_records.append(
        result
    )


region_df = pd.DataFrame(
    region_records
)


region_df = (
    region_df[
        [
            "region",
            "n_items",
            "unanimous_n",
            "unanimous_rate",
            "three_one_n",
            "three_one_rate",
            "two_two_n",
            "two_two_rate",
            "fleiss_kappa",
        ]
    ]
)


region_output = (
    OUTPUT_DIR
    /
    "interrater_agreement_by_region.csv"
)


region_df.to_csv(
    region_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 12. 全部資料整體 Fleiss Kappa
# ============================================================

all_ratings = (
    vote_df[
        [
            "D1",
            "D2",
            "D3",
            "D4"
        ]
    ]
    .to_numpy()
)


overall_result = (
    calculate_agreement(
        all_ratings
    )
)


overall_df = pd.DataFrame([
    {
        "analysis":
            "all_regions_all_colors",

        **overall_result
    }
])


overall_output = (
    OUTPUT_DIR
    /
    "interrater_agreement_overall.csv"
)


overall_df.to_csv(
    overall_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 13. Terminal 顯示結果
# ============================================================

print("\n")
print("=" * 75)
print("各舌質色醫師一致性")
print("=" * 75)


display_color = (
    color_agreement_df[
        [
            "color",
            "n_items",
            "unanimous_rate",
            "three_one_rate",
            "two_two_rate",
            "fleiss_kappa"
        ]
    ]
    .copy()
)


for column in [
    "unanimous_rate",
    "three_one_rate",
    "two_two_rate",
    "fleiss_kappa"
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


print("\n")
print("=" * 75)
print("各舌區醫師一致性")
print("=" * 75)


display_region = (
    region_df[
        [
            "region",
            "n_items",
            "unanimous_rate",
            "three_one_rate",
            "two_two_rate",
            "fleiss_kappa"
        ]
    ]
    .copy()
)


for column in [
    "unanimous_rate",
    "three_one_rate",
    "two_two_rate",
    "fleiss_kappa"
]:

    display_region[
        column
    ] = (
        display_region[
            column
        ]
        .round(
            4
        )
    )


print(
    display_region.to_string(
        index=False
    )
)


print("\n")
print("=" * 75)
print("全部資料整體結果")
print("=" * 75)


print(
    "Items：",
    overall_result[
        "n_items"
    ]
)

print(
    "四位完全一致率：",
    round(
        overall_result[
            "unanimous_rate"
        ],
        4
    )
)

print(
    "3:1 比例：",
    round(
        overall_result[
            "three_one_rate"
        ],
        4
    )
)

print(
    "2:2 比例：",
    round(
        overall_result[
            "two_two_rate"
        ],
        4
    )
)

print(
    "Fleiss' kappa：",
    round(
        overall_result[
            "fleiss_kappa"
        ],
        4
    )
)


print("\n輸出檔案：")

print(
    vote_output
)

print(
    color_output
)

print(
    region_output
)

print(
    region_color_output
)

print(
    overall_output
)

print("=" * 75)
