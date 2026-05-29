import pandas as pd


def daily_stock_std(
    df: pd.DataFrame,
    date_col: str = "date",
    stock_col: str = "stock",
    value_col: str = "mid",
) -> pd.DataFrame:
    """Compute unbiased sample standard deviation per day per stock."""
    if (
        date_col not in df.columns
        or stock_col not in df.columns
        or value_col not in df.columns
    ):
        raise ValueError(
            f"DataFrame must contain columns: {date_col}, {stock_col}, {value_col}"
        )

    grouped = df.groupby([date_col, stock_col], observed=True)[value_col]
    result = grouped.std(ddof=1).reset_index()
    result = result.rename(columns={value_col: f"{value_col}_std"})
    return result
