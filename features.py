"""Raw alert rows -> model-ready feature matrix.

The split into hard-to-fake and easy-to-fake features is the point of this file.
A customer writes their own complaint category; they do not write the delivery
scan, their own account age, or how many other accounts share their device. When
the model leans on the first group it is reading evidence. When it leans on the
second it is reading a claim, and a claim can be rewritten the moment someone
works out what the model wants to hear.
"""

import math
from pathlib import Path

import pandas as pd

ALERTS = Path("data/alerts.csv")

# Ground truth, analysis-only columns, identifiers, and raw hashes. A hash is an
# identity, not a pattern: given device_fingerprint the model would memorise the
# specific rings in this dataset and learn nothing about sharing behaviour.
FORBIDDEN = [
    "persona",
    "ring_archetype",
    "text_contradiction",
    "would_win_if_fought",
    "alert_id",
    "customer_id",
    "device_fingerprint",
    "address_hash",
]

PROOF_LEVELS = ("signed", "otp", "none")
STATUS_LEVELS = ("delivered", "in_transit", "lost")
CATEGORY_LEVELS = ("item_not_received", "not_as_described", "unauthorised", "duplicate_charge")


def _one_hot(col: pd.Series, levels: tuple[str, ...], prefix: str) -> pd.DataFrame:
    """Levels are pinned so a split that happens to contain no lost parcels still
    produces the same columns in the same order as every other split."""
    cat = pd.Categorical(col, categories=levels)
    assert not pd.isna(cat).any(), f"unknown {prefix} value: {set(col) - set(levels)}"
    return pd.get_dummies(pd.Series(cat, index=col.index), prefix=prefix).astype(int)


def share_counts(full: pd.DataFrame) -> dict[str, pd.Series]:
    """Distinct customers per device and per address, over the whole dataset.

    ponytail: computed on all 3000 rows, so a test row's count reflects ring
    siblings that only appear in train. That is mild leakage. On real data this
    has to be an as-of computation -- how many accounts shared this device
    *before this alert arrived* -- which needs an event timestamp the simulator
    does not currently emit. Upgrade path: add first_seen_at per alert, then
    compute the counts with an expanding window ordered by that timestamp.
    """
    return {
        "device": full.groupby("device_fingerprint")["customer_id"].nunique(),
        "address": full.groupby("address_hash")["customer_id"].nunique(),
    }


def build_features(
    df: pd.DataFrame,
    shares: dict[str, pd.Series] | None = None,
    include_easy: bool = True,
) -> pd.DataFrame:
    """Feature matrix for df. Pass `shares` computed from the full dataset when
    building a split on its own, otherwise train and test disagree about how
    many accounts share a device.

    include_easy=False drops the customer-supplied block. The money path runs
    without it: a feature the claimant writes is a feature the claimant can
    rewrite once they work out which wording pays."""
    if shares is None:
        shares = share_counts(df)

    # ---- HARD TO FAKE: evidence the customer does not author -----------------
    hard = pd.DataFrame(index=df.index)
    hard["account_age_days"] = df["account_age_days"]
    hard["prior_orders"] = df["prior_orders"]
    hard["prior_disputes"] = df["prior_disputes"]
    # A first-order customer with one dispute is a different animal from a
    # fifty-order customer with one dispute.
    hard["dispute_rate"] = df["prior_disputes"] / df["prior_orders"].clip(lower=1)
    hard["device_share_count"] = df["device_fingerprint"].map(shares["device"])
    hard["address_share_count"] = df["address_hash"].map(shares["address"])
    hard["amount"] = df["amount"]
    # Amounts are log-normal by construction; the linear column keeps the money
    # scale for the EV maths, the log column is what a linear model can use.
    hard["log_amount"] = df["amount"].apply(math.log1p)
    hard["days_since_purchase"] = df["days_since_purchase"]

    hard = pd.concat([
        hard,
        _one_hot(df["delivery_proof"], PROOF_LEVELS, "proof"),
        _one_hot(df["delivery_status"], STATUS_LEVELS, "status"),
    ], axis=1)

    # ---- EASY TO FAKE: whatever the customer told us -------------------------
    easy = _one_hot(df["complaint_category"], CATEGORY_LEVELS, "cat")

    x = pd.concat([hard, easy], axis=1) if include_easy else hard

    leaked = set(FORBIDDEN) & set(x.columns)
    assert not leaked, f"forbidden column in feature matrix: {sorted(leaked)}"
    assert len(x) == len(df), f"row count changed: {len(df)} -> {len(x)}"
    assert not x.isna().any().any(), f"NaNs in: {sorted(x.columns[x.isna().any()])}"
    assert x.dtypes.map(pd.api.types.is_numeric_dtype).all(), "non-numeric feature survived"
    return x


def main() -> None:
    df = pd.read_csv(ALERTS)
    x = build_features(df)

    assert not set(FORBIDDEN) & set(x.columns)
    assert not x.isna().any().any()
    assert len(x) == len(df) == 3000

    # A split built with the full-dataset counts must match the full build row
    # for row; built on its own it must not, which is the whole reason `shares`
    # is threaded through instead of recomputed per split.
    test = df.sample(frac=0.30, random_state=42)
    with_shares = build_features(test, shares=share_counts(df))
    assert with_shares.equals(x.loc[test.index]), "passing shares in changed the matrix"
    alone = build_features(test)
    assert alone["device_share_count"].sum() < with_shares["device_share_count"].sum()

    hard_only = build_features(df, include_easy=False)
    dropped = set(x.columns) - set(hard_only.columns)
    assert dropped == {c for c in x.columns if c.startswith("cat_")}, f"dropped wrong block: {dropped}"
    assert hard_only.equals(x[hard_only.columns]), "dropping the easy block changed the hard block"

    print(f"{x.shape[0]} rows x {x.shape[1]} features")
    print(f"hard to fake: {[c for c in x.columns if not c.startswith('cat_')]}")
    print(f"easy to fake: {[c for c in x.columns if c.startswith('cat_')]}")
    print(x[["dispute_rate", "device_share_count", "address_share_count", "log_amount"]]
          .describe().round(2).to_string())


if __name__ == "__main__":
    main()
