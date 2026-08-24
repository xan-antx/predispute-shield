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


def make_customers(n: int) -> list[dict]:
    """Customer pool built to the persona mix by count, not by weighted draw --
    ring accounts arrive in clusters, so sampling a persona per account would
    over-fill the pool with ring members. Each cluster shares one device
    fingerprint and address, the only structural signal a graph feature could
    pick up later. Everyone else gets an address pool of their own, sized so
    that an opportunist occasionally ships to a different address."""
    customers: list[dict] = []
    for _ in range(round(PERSONA_MIX[0] * n)):
        customers.append({
            "persona": "honest",
            "account_age_days": random.randint(200, 1500),
            "prior_orders": random.randint(5, 50),
            "prior_disputes": random.randint(0, 1),
            "device_fingerprint": f"dev_{_hex()}",
            "addresses": [f"addr_{_hex()}"],
        })
    for _ in range(round(PERSONA_MIX[1] * n)):
        customers.append({
            "persona": "opportunist",
            "account_age_days": random.randint(60, 600),
            "prior_orders": random.randint(3, 20),
            "prior_disputes": random.randint(1, 4),
            "device_fingerprint": f"dev_{_hex()}",
            "addresses": [f"addr_{_hex()}" for _ in range(random.randint(1, 3))],
        })
    target_ring = round(PERSONA_MIX[2] * n)
    ring = 0
    while ring < target_ring:
        device, address = f"dev_{_hex()}", f"addr_{_hex()}"
        for _ in range(random.randint(3, 6)):  # self plus 2 to 5 siblings
            customers.append({
                "persona": "ring",
                "account_age_days": random.randint(5, 90),
                "prior_orders": random.randint(0, 3),
                "prior_disputes": random.randint(0, 3),
                "device_fingerprint": device,
                "addresses": [address],
            })
            ring += 1
    # Shuffle before numbering so customer_id itself carries no persona signal.
    random.shuffle(customers)
    for i, c in enumerate(customers):
        c["customer_id"] = f"CUST{i:05d}"
    return customers


def draw_amount(persona: str) -> float:
    if persona == "opportunist" and random.random() < 0.5:
        # Parks just under the round number merchants tend to auto-refund below.
        return round(random.uniform(1500, 1999), 2)
    return round(float(np.clip(np.random.lognormal(7.7, 0.85), 200, 80000)), 2)


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


def generate(n_alerts: int = N_ALERTS) -> pd.DataFrame:
    random.seed(42)
    np.random.seed(42)

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
            "amount": draw_amount(c["persona"]),
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
            "text_contradiction": contradict,
        }
        score = win_strength(row)
        if ambiguous:
            # Clamp the loyalty/dispute terms so these stay coin flips by
            # construction rather than by luck.
            score = float(np.clip(score, -0.4, 0.4))
        p_win = 1.0 / (1.0 + np.exp(-score))
        # Sampled, not thresholded: two identical evidence packets can land
        # differently at arbitration, and that irreducible noise is the whole
        # reason a calibrated P(win) beats a hard rule.
        row["would_win_if_fought"] = int(np.random.random() < p_win)
        rows.append(row)

    return pd.DataFrame(rows)


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

    ring_devices = df[df.persona == "ring"].groupby("device_fingerprint")["customer_id"].nunique()
    assert ring_devices.max() >= 3, "ring accounts are not sharing devices"

    assert len(pd.read_csv(OUT)) == 3000, "file is not 3000 rows"

    print(f"{len(df)} alerts -> {OUT}")
    print(f"win rate {win_rate:.1%} | undecidable {len(ambiguous) / len(df):.1%} | "
          f"contradictions {df.text_contradiction.mean():.1%}")
    print(by_persona.to_string())


if __name__ == "__main__":
    main()
