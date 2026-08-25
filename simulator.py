"""Synthetic pre-dispute alert generator with sampled ground-truth labels.

The label is drawn from a sigmoid over *evidence only*. Persona is never a term
in the score: a fraud ring holding a signed delivery receipt really is hard to
beat at arbitration, and an honest customer whose parcel was lost really does
lose. If persona leaked into the label, every downstream model would learn
"profile the customer" instead of "read the evidence", which is the exact
failure mode this project exists to avoid.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/alerts.csv")
N_ALERTS = 3000
N_CUSTOMERS = 1800

PERSONAS = ("honest", "opportunist", "ring")
PERSONA_MIX = (0.75, 0.20, 0.05)

# Ring clusters share infrastructure, but not all in the same shape. Bundling
# device and address sharing into one signal makes the two columns collinear and
# lets a model score "shares anything" as fraud; separating them forces it to
# learn which kind of sharing it is looking at.
RING_ARCHETYPES = ("tight", "dropship", "household")
RING_ARCHETYPE_MIX = (0.40, 0.35, 0.25)

# Benign sharing. A family ships to one address, a household shares one laptop.
# Without these, every shared identifier in the dataset is fraud and the model
# learns a rule that denies real customers their refund.
BENIGN_ADDRESS_SHARE = 0.03
BENIGN_DEVICE_SHARE = 0.02

# Evidence weights, in log-odds of winning a fought dispute. Tuned so the
# marginal win rate lands near 55% -- roughly what a merchant with decent
# delivery-proof coverage sees at arbitration.
BASE = -0.5
PROOF_W = {"signed": 1.6, "otp": 0.7, "none": -1.5}
DELIVERY_W = {"delivered": 0.8, "in_transit": -0.2, "lost": -1.8}
UNAUTHORISED_W = -0.9          # issuers side with the cardholder more often here
ORDER_W = 0.02                 # loyalty helps, but only mildly
DISPUTE_W = -0.25              # a repeat disputer looks worse to the issuer

CATEGORIES = ("item_not_received", "not_as_described", "unauthorised", "duplicate_charge")
CATEGORY_MIX = {
    "honest": (0.35, 0.35, 0.15, 0.15),
    "opportunist": (0.40, 0.30, 0.20, 0.10),
    "ring": (0.25, 0.10, 0.55, 0.10),
}

TEMPLATES = {
    "item_not_received": [
        "Order never arrived. Tracking shows delivered but nothing reached me.",
        "I have not received this package even after {days} days.",
        "Nothing was delivered to my address. Please refund the amount.",
    ],
    "not_as_described": [
        "Item is not what the listing showed. Quality is poor.",
        "Received the wrong variant, this is not what I ordered.",
        "Product does not match the description on your site.",
    ],
    "unauthorised": [
        "I did not authorise this transaction on my card.",
        "This charge is not mine, my card was used without permission.",
        "Unknown charge from your store, I never placed any order.",
    ],
    "duplicate_charge": [
        "Charged twice for the same order, please reverse one debit.",
        "Two payments went out for a single purchase.",
        "Duplicate debit of the same amount on the same day.",
    ],
}

# Product-describing sentences. Bolted onto a non-receipt or unauthorised claim
# they contradict it outright -- you cannot describe what you never received.
CONTRADICTIONS = [
    " The box was torn when I opened it.",
    " The colour is nothing like the photos.",
    " It stopped working after two days of use.",
    " The size is wrong, I want to send it back.",
]

PERSONA_SUFFIX = {
    "honest": ["", "", " Kindly help."],
    "opportunist": ["", " I raised this before and got no response.", " This keeps happening."],
    "ring": ["", " Refund immediately.", " Reverse this now."],
}


def _hex(n: int = 8) -> str:
    return f"{random.randrange(16 ** n):0{n}x}"


def _add_benign_sharing(customers: list[dict]) -> None:
    """Pair up a few honest customers on a shared address or device, in place.

    Sharing has to exist outside the rings or it is perfectly fraud-coded, and a
    model trained on that will deny a refund to a mother and daughter ordering to
    the same flat. These pairs are the counterexamples that stop share counts
    from being a proxy for guilt."""
    honest = [i for i, c in enumerate(customers) if c["persona"] == "honest"]
    for key, rate in (("addresses", BENIGN_ADDRESS_SHARE), ("device_fingerprint", BENIGN_DEVICE_SHARE)):
        picked = random.sample(honest, k=2 * round(rate * len(honest) / 2))
        for a, b in zip(picked[::2], picked[1::2]):
            value = customers[a][key]
            customers[b][key] = list(value) if isinstance(value, list) else value


def make_customers(n: int) -> list[dict]:
    """Customer pool built to the persona mix by count, not by weighted draw --
    ring accounts arrive in clusters, so sampling a persona per account would
    over-fill the pool with ring members. Ring clusters share infrastructure in
    one of three shapes (see RING_ARCHETYPES), and a small slice of honest
    customers share too, for entirely boring reasons."""
    customers: list[dict] = []
    for _ in range(round(PERSONA_MIX[0] * n)):
        customers.append({
            "persona": "honest",
            "ring_archetype": "",
            "account_age_days": random.randint(200, 1500),
            "prior_orders": random.randint(5, 50),
            "prior_disputes": random.randint(0, 1),
            "device_fingerprint": f"dev_{_hex()}",
            "addresses": [f"addr_{_hex()}"],
        })
    for _ in range(round(PERSONA_MIX[1] * n)):
        customers.append({
            "persona": "opportunist",
            "ring_archetype": "",
            "account_age_days": random.randint(60, 600),
            "prior_orders": random.randint(3, 20),
            "prior_disputes": random.randint(1, 4),
            "device_fingerprint": f"dev_{_hex()}",
            # An occasional address change: same person, fresh shipping address.
            "addresses": [f"addr_{_hex()}" for _ in range(random.randint(1, 3))],
        })
    # Archetypes are allocated by target account count, not by a weighted draw
    # per cluster: clusters vary in size, so drawing per cluster does not control
    # the account-level mix (see FAILURES.md, the same bug at persona level).
    target_ring = round(PERSONA_MIX[2] * n)
    for archetype, share in zip(RING_ARCHETYPES, RING_ARCHETYPE_MIX):
        made = 0
        while made < round(share * target_ring):
            device, address = f"dev_{_hex()}", f"addr_{_hex()}"
            for _ in range(random.randint(3, 6)):  # self plus 2 to 5 siblings
                customers.append({
                    "persona": "ring",
                    "ring_archetype": archetype,
                    "account_age_days": random.randint(5, 90),
                    "prior_orders": random.randint(0, 3),
                    "prior_disputes": random.randint(0, 3),
                    # dropship: one device, a fresh drop address per mule.
                    # household: one address, a different handset per account.
                    "device_fingerprint": device if archetype != "household" else f"dev_{_hex()}",
                    "addresses": [address if archetype != "dropship" else f"addr_{_hex()}"],
                })
                made += 1

    _add_benign_sharing(customers)
    # Shuffle before numbering so customer_id itself carries no persona signal.
    random.shuffle(customers)
    for i, c in enumerate(customers):
        c["customer_id"] = f"CUST{i:05d}"
    return customers


def draw_amount(persona: str, sigma: float = 0.85) -> float:
    """sigma widens or tightens the log-normal tail; sweep.py varies it, the
    default reproduces the committed dataset byte for byte."""
    if persona == "opportunist" and random.random() < 0.5:
        # Parks just under the round number merchants tend to auto-refund below.
        return round(random.uniform(1500, 1999), 2)
    return round(float(np.clip(np.random.lognormal(7.7, sigma), 200, 80000)), 2)


def draw_delivery() -> tuple[str, str]:
    """Proof is conditional on status -- nobody holds a signature for a parcel
    that never left the hub, and an in-transit parcel has no proof at all. The
    one exception, in_transit plus OTP, is minted only by the undecidable branch
    in generate(), so that combination stays a clean marker for coin-flip cases."""
    status = random.choices(
        ("delivered", "in_transit", "lost"), weights=(0.70, 0.20, 0.10)
    )[0]
    if status == "delivered":
        proof = random.choices(("signed", "otp", "none"), weights=(0.45, 0.40, 0.15))[0]
    else:
        proof = "none"
    return status, proof


def win_strength(row: dict) -> float:
    """Log-odds that we win this dispute if we fight it. Evidence terms only."""
    return (
        BASE
        + PROOF_W[row["delivery_proof"]]
        + DELIVERY_W[row["delivery_status"]]
        + ORDER_W * min(row["prior_orders"], 30)
        + DISPUTE_W * row["prior_disputes"]
        + (UNAUTHORISED_W if row["complaint_category"] == "unauthorised" else 0.0)
    )


def complaint_text(category: str, persona: str, days: int, contradict: bool) -> str:
    text = random.choice(TEMPLATES[category]).format(days=days)
    if contradict:
        text += random.choice(CONTRADICTIONS)
    return text + random.choice(PERSONA_SUFFIX[persona])


def _calibrate(scores: np.ndarray, base_rate: float, noise: float) -> tuple[float, float]:
    """Find (scale, shift) so that p = sigmoid(scale * score + shift) hits a
    target mean win rate and a target irreducible noise E[min(p, 1 - p)].

    Nested bisection: shift controls the mean at any scale (monotone), scale
    controls the noise once the mean is pinned (saturating the sigmoid drives
    noise to zero, flattening it drives noise to min(base, 1 - base)). Both are
    monotone on the actual score array, so 40 halvings each is plenty.
    """
    def stats(scale: float, shift: float) -> tuple[float, float]:
        p = 1.0 / (1.0 + np.exp(-(scale * scores + shift)))
        return float(p.mean()), float(np.minimum(p, 1 - p).mean())

    def solve_shift(scale: float) -> float:
        lo, hi = -10.0, 10.0
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if stats(scale, mid)[0] < base_rate else (lo, mid)
        return (lo + hi) / 2

    lo, hi = 0.05, 40.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if stats(mid, solve_shift(mid))[1] > noise else (lo, mid)
    scale = (lo + hi) / 2
    return scale, solve_shift(scale)


def generate(n_alerts: int = N_ALERTS, seed: int = 42, amount_sigma: float = 0.85,
             target_base_rate: float | None = None, target_noise: float | None = None,
             return_probs: bool = False):
    """Default arguments reproduce the committed dataset byte for byte: the
    relabelling pass below only runs when sweep targets are given, and it uses
    its own RNG so the primary streams are untouched either way."""
    random.seed(seed)
    np.random.seed(seed)

    customers = make_customers(N_CUSTOMERS)
    # Roughly 8% of alerts are built to be genuinely undecidable: in transit with
    # only an OTP, which sits at exactly zero log-odds before the mild terms.
    undecidable = set(random.sample(range(n_alerts), k=int(0.08 * n_alerts)))

    rows = []
    for i in range(n_alerts):
        c = random.choice(customers)
        ambiguous = i in undecidable
        if ambiguous:
            status, proof = "in_transit", "otp"
            category = random.choice([x for x in CATEGORIES if x != "unauthorised"])
        else:
            status, proof = draw_delivery()
            category = random.choices(CATEGORIES, weights=CATEGORY_MIX[c["persona"]])[0]

        days = random.randint(1, 45)
        contradict = category in ("item_not_received", "unauthorised") and random.random() < 0.12
        row = {
            "alert_id": f"ALT{i:05d}",
            "customer_id": c["customer_id"],
            "amount": draw_amount(c["persona"], amount_sigma),
            "account_age_days": c["account_age_days"],
            "prior_orders": c["prior_orders"],
            "prior_disputes": c["prior_disputes"],
            "delivery_status": status,
            "delivery_proof": proof,
            "device_fingerprint": c["device_fingerprint"],
            "address_hash": random.choice(c["addresses"]),
            "days_since_purchase": days,
            "complaint_category": category,
            "complaint_text": complaint_text(category, c["persona"], days, contradict),
            "persona": c["persona"],
            "ring_archetype": c["ring_archetype"],
            "text_contradiction": contradict,
        }
        score = win_strength(row)
        if ambiguous:
            # Clamp the loyalty/dispute terms so these stay coin flips by
            # construction rather than by luck.
            score = float(np.clip(score, -0.4, 0.4))
        row["_score"] = score
        p_win = 1.0 / (1.0 + np.exp(-score))
        # Sampled, not thresholded: two identical evidence packets can land
        # differently at arbitration, and that irreducible noise is the whole
        # reason a calibrated P(win) beats a hard rule.
        row["would_win_if_fought"] = int(np.random.random() < p_win)
        rows.append(row)

    df = pd.DataFrame(rows)
    scores = df.pop("_score").to_numpy()
    probs = 1.0 / (1.0 + np.exp(-scores))
    if target_base_rate is not None or target_noise is not None:
        # Sweep path: re-map the same evidence scores through a calibrated
        # sigmoid and resample labels from a dedicated RNG. Evidence, personas
        # and amounts are untouched -- only how forgiving the world is changes.
        scale, shift = _calibrate(scores,
                                  target_base_rate if target_base_rate is not None else 0.5,
                                  target_noise if target_noise is not None else 0.2)
        probs = 1.0 / (1.0 + np.exp(-(scale * scores + shift)))
        df["would_win_if_fought"] = (np.random.default_rng(seed).random(len(df)) < probs).astype(int)
    return (df, probs) if return_probs else df


def main() -> None:
    df = generate()
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)

    win_rate = df["would_win_if_fought"].mean()
    assert 0.40 <= win_rate <= 0.60, f"win rate {win_rate:.3f} outside 40-60%"

    by_persona = df.groupby("persona")["would_win_if_fought"].mean()
    assert by_persona.between(0.15, 0.85).all(), f"persona label leakage:\n{by_persona}"

    ambiguous = df[(df.delivery_status == "in_transit") & (df.delivery_proof == "otp")]
    assert 0.07 <= len(ambiguous) / len(df) <= 0.09, f"{len(ambiguous)} undecidable rows"

    mix = df["persona"].value_counts(normalize=True)
    assert abs(mix["honest"] - 0.75) < 0.05 and abs(mix["ring"] - 0.05) < 0.03, f"mix off:\n{mix}"

    # Same check one level down. Cluster-level draws do not control account-level
    # share, and this dataset has few enough clusters that eyeballing it lies.
    arch = df[df.persona == "ring"].drop_duplicates("customer_id")["ring_archetype"]
    arch_mix = arch.value_counts(normalize=True)
    for name, share in zip(RING_ARCHETYPES, RING_ARCHETYPE_MIX):
        assert abs(arch_mix[name] - share) < 0.10, f"ring archetype mix off:\n{arch_mix}"

    dev_share = df.groupby("device_fingerprint")["customer_id"].nunique()
    addr_share = df.groupby("address_hash")["customer_id"].nunique()
    dev = df["device_fingerprint"].map(dev_share)
    addr = df["address_hash"].map(addr_share)

    assert dev[df.persona == "ring"].max() >= 3, "ring accounts are not sharing devices"
    # The whole point of the archetypes: the two counts must come apart.
    assert (dev > addr).any() and (addr > dev).any(), "device and address sharing still bundled"
    assert dev.corr(addr) < 0.9, f"share counts still collinear: {dev.corr(addr):.2f}"

    honest = df.persona == "honest"
    assert (dev[honest] > 1).any() and (addr[honest] > 1).any(), "no benign sharing"

    assert len(pd.read_csv(OUT)) == 3000, "file is not 3000 rows"

    print(f"{len(df)} alerts -> {OUT}")
    print(f"win rate {win_rate:.1%} | undecidable {len(ambiguous) / len(df):.1%} | "
          f"contradictions {df.text_contradiction.mean():.1%}")
    print(by_persona.to_string())


if __name__ == "__main__":
    main()
