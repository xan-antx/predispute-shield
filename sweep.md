# Simulator parameter sweep

27 configurations: label noise x base win rate x amount distribution. Each cell
regenerates 3,000 alerts (seed = 1000 + cell index), retrains the calibrated
model on that world's 60/20/20 split, and scores the EV strategy against
always_refund on the 600-row test slice. Δ = system minus always_refund in net
₹ per 1000 alerts; positive (bold) means the system wins. Cells at or below
the incumbent are listed, not excluded.

**System beats always_refund in 26 of 27 cells (96%).**
On a 600-row slice, one flipped fight outcome moves Δ by ₹6,000 per 1000
(the flat FP regret), so a cell within that band of zero is indistinguishable
from always_refund at this sample size -- not a loss.

### log-normal σ=0.85 (current)

| noise \ base rate | 0.35 | 0.50 | 0.65 |
|---|--:|--:|--:|
| 0.10 | **+180,804** | **+386,092** | **+398,138** |
| 0.20 | **+7,073** | **+158,958** | **+357,476** |
| 0.30 | **+80,679** | **+349,250** | **+554,238** |

### tight σ=0.45

| noise \ base rate | 0.35 | 0.50 | 0.65 |
|---|--:|--:|--:|
| 0.10 | **+32,576** | **+36,985** | **+76,554** |
| 0.20 | **+27,333** | **+19,011** | **+44,439** |
| 0.30 | -6,000 | **+680** | **+9,014** |

### heavy σ=1.30

| noise \ base rate | 0.35 | 0.50 | 0.65 |
|---|--:|--:|--:|
| 0.10 | **+1,192,119** | **+1,064,198** | **+1,436,924** |
| 0.20 | **+609,511** | **+1,050,770** | **+1,719,751** |
| 0.30 | **+850,300** | **+701,518** | **+1,322,838** |

## Cells at or below the incumbent

- tight σ=0.45, noise 0.30, base 0.35: Δ -6,000 (1 fight, lost) -- exactly one
  flipped outcome, indistinguishable from the incumbent

Cells where the system never fights (Δ exactly 0 by construction): 0.

## Detail

| Amounts | Noise tgt | Base tgt | Noise achieved | Base achieved | System ₹/1000 | Refund ₹/1000 | Δ | Fought | Lost | Seed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| log-normal σ=0.85 (current) | 0.10 | 0.35 | 0.100 | 0.350 | -1,484,441 | -1,665,245 | +180,804 | 50 | 11 | 1000 |
| log-normal σ=0.85 (current) | 0.10 | 0.50 | 0.100 | 0.500 | -1,628,627 | -2,014,719 | +386,092 | 74 | 7 | 1001 |
| log-normal σ=0.85 (current) | 0.10 | 0.65 | 0.100 | 0.650 | -1,323,014 | -1,721,152 | +398,138 | 82 | 7 | 1002 |
| log-normal σ=0.85 (current) | 0.20 | 0.35 | 0.200 | 0.350 | -1,882,158 | -1,889,231 | +7,073 | 35 | 18 | 1003 |
| log-normal σ=0.85 (current) | 0.20 | 0.50 | 0.200 | 0.500 | -1,520,163 | -1,679,121 | +158,958 | 53 | 15 | 1004 |
| log-normal σ=0.85 (current) | 0.20 | 0.65 | 0.200 | 0.650 | -1,641,624 | -1,999,099 | +357,476 | 96 | 19 | 1005 |
| log-normal σ=0.85 (current) | 0.30 | 0.35 | 0.300 | 0.350 | -1,865,396 | -1,946,075 | +80,679 | 29 | 16 | 1006 |
| log-normal σ=0.85 (current) | 0.30 | 0.50 | 0.300 | 0.500 | -1,499,143 | -1,848,393 | +349,250 | 54 | 14 | 1007 |
| log-normal σ=0.85 (current) | 0.30 | 0.65 | 0.300 | 0.650 | -1,464,854 | -2,019,092 | +554,238 | 89 | 25 | 1008 |
| tight σ=0.45 | 0.10 | 0.35 | 0.100 | 0.350 | -1,156,167 | -1,188,743 | +32,576 | 16 | 1 | 1009 |
| tight σ=0.45 | 0.10 | 0.50 | 0.100 | 0.500 | -1,166,042 | -1,203,027 | +36,985 | 27 | 2 | 1010 |
| tight σ=0.45 | 0.10 | 0.65 | 0.100 | 0.650 | -1,070,757 | -1,147,310 | +76,554 | 38 | 1 | 1011 |
| tight σ=0.45 | 0.20 | 0.35 | 0.200 | 0.350 | -1,210,512 | -1,237,845 | +27,333 | 15 | 3 | 1012 |
| tight σ=0.45 | 0.20 | 0.50 | 0.200 | 0.500 | -1,109,674 | -1,128,685 | +19,011 | 23 | 5 | 1013 |
| tight σ=0.45 | 0.20 | 0.65 | 0.200 | 0.650 | -1,110,746 | -1,155,185 | +44,439 | 22 | 2 | 1014 |
| tight σ=0.45 | 0.30 | 0.35 | 0.300 | 0.350 | -1,214,979 | -1,208,979 | -6,000 | 1 | 1 | 1015 |
| tight σ=0.45 | 0.30 | 0.50 | 0.300 | 0.500 | -1,169,863 | -1,170,543 | +680 | 19 | 6 | 1016 |
| tight σ=0.45 | 0.30 | 0.65 | 0.300 | 0.650 | -1,218,298 | -1,227,312 | +9,014 | 24 | 7 | 1017 |
| heavy σ=1.30 | 0.10 | 0.35 | 0.100 | 0.350 | -2,716,843 | -3,908,962 | +1,192,119 | 84 | 20 | 1018 |
| heavy σ=1.30 | 0.10 | 0.50 | 0.100 | 0.500 | -2,243,013 | -3,307,211 | +1,064,198 | 104 | 12 | 1019 |
| heavy σ=1.30 | 0.10 | 0.65 | 0.100 | 0.650 | -2,302,989 | -3,739,913 | +1,436,924 | 130 | 16 | 1020 |
| heavy σ=1.30 | 0.20 | 0.35 | 0.200 | 0.350 | -2,993,792 | -3,603,303 | +609,511 | 79 | 32 | 1021 |
| heavy σ=1.30 | 0.20 | 0.50 | 0.200 | 0.500 | -2,608,713 | -3,659,483 | +1,050,770 | 108 | 23 | 1022 |
| heavy σ=1.30 | 0.20 | 0.65 | 0.200 | 0.650 | -2,043,136 | -3,762,887 | +1,719,751 | 133 | 21 | 1023 |
| heavy σ=1.30 | 0.30 | 0.35 | 0.300 | 0.350 | -2,830,492 | -3,680,793 | +850,300 | 69 | 33 | 1024 |
| heavy σ=1.30 | 0.30 | 0.50 | 0.300 | 0.500 | -2,148,276 | -2,849,794 | +701,518 | 83 | 33 | 1025 |
| heavy σ=1.30 | 0.30 | 0.65 | 0.300 | 0.650 | -2,767,936 | -4,090,774 | +1,322,838 | 133 | 40 | 1026 |

Achieved noise/base come from the calibrated generative probabilities, not
the sampled labels; corner cells (low base + high noise) sit at the feasible
boundary and can miss their targets -- read the achieved columns first.
