# Ratio-penalty sensitivity sweep

27 penalty curves (floor x ceiling x exponent), each treated as the true cost
structure of its world: the EV strategy decides with that curve and fights are
charged with it at the ratio standing when they were filed (start 0.40%, every
fight counts from filing). One canonical seed-42 dataset and one trained model
throughout, so the curve is the only thing that varies. Δ = system minus
always_refund (-1,816,286) in net ₹ per 1000 alerts; positive (bold) wins.

**System beats always_refund in 15 of 27 cells, ties it in 9, loses in 3.** Every tie is a zero-fight cell: the curve prices all fights out and the system degenerates to always_refund exactly. Winning Δ ranges +7,840 to +149,907; the worst loss is -7,573.

### exponent 1 (linear)

| floor \ ceiling | 25,000 | 50,000 | 100,000 |
|---|--:|--:|--:|
| 250 | **+23,744** | tie (0 fights) | tie (0 fights) |
| 500 | **+23,246** | tie (0 fights) | tie (0 fights) |
| 1,000 | **+22,248** | tie (0 fights) | tie (0 fights) |

### exponent 2 (quadratic)

| floor \ ceiling | 25,000 | 50,000 | 100,000 |
|---|--:|--:|--:|
| 250 | **+46,477** | -3,388 | tie (0 fights) |
| 500 | **+40,309** | -4,783 | tie (0 fights) |
| 1,000 | **+38,783** | -7,573 | tie (0 fights) |

### exponent 3 (cubic)

| floor \ ceiling | 25,000 | 50,000 | 100,000 |
|---|--:|--:|--:|
| 250 | **+147,334** | **+71,037** | **+16,966** |
| 500 | **+149,907** | **+60,097** | **+13,647** |
| 1,000 | **+119,574** | **+45,662** | **+7,840** |

## Losing cells

- floor 1,000, ceiling 50,000, exponent 2: Δ -7,573 (4 fights)
- floor 500, ceiling 50,000, exponent 2: Δ -4,783 (4 fights)
- floor 250, ceiling 50,000, exponent 2: Δ -3,388 (4 fights)

## Shape versus magnitude

Mean Δ per axis value, and the mean spread of Δ when one axis moves across its
full range while the other two stay fixed:

| Axis | Values -> mean Δ | Spread when varied alone |
|---|---|--:|
| exponent | 1: +7,693, 2: +12,203, 3: +70,229 | 64,286 |
| floor | 250: +33,574, 500: +31,380, 1,000: +25,170 | 8,690 |
| ceiling | 25,000: +67,958, 50,000: +17,895, 100,000: +4,273 | 65,435 |

## Detail

| Floor | Ceiling | Exp | System ₹/1000 | Δ | Fought | Lost | End ratio |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 250 | 25,000 | 1 | -1,792,542 | +23,744 | 2 | 0 | 0.405% |
| 250 | 25,000 | 2 | -1,769,810 | +46,477 | 18 | 5 | 0.445% |
| 250 | 25,000 | 3 | -1,668,952 | +147,334 | 46 | 12 | 0.515% |
| 250 | 50,000 | 1 | -1,816,286 | +0 | 0 | 0 | 0.400% |
| 250 | 50,000 | 2 | -1,819,674 | -3,388 | 4 | 2 | 0.410% |
| 250 | 50,000 | 3 | -1,745,249 | +71,037 | 24 | 6 | 0.460% |
| 250 | 100,000 | 1 | -1,816,286 | +0 | 0 | 0 | 0.400% |
| 250 | 100,000 | 2 | -1,816,286 | +0 | 0 | 0 | 0.400% |
| 250 | 100,000 | 3 | -1,799,320 | +16,966 | 8 | 2 | 0.420% |
| 500 | 25,000 | 1 | -1,793,041 | +23,246 | 2 | 0 | 0.405% |
| 500 | 25,000 | 2 | -1,775,977 | +40,309 | 18 | 5 | 0.445% |
| 500 | 25,000 | 3 | -1,666,379 | +149,907 | 43 | 10 | 0.507% |
| 500 | 50,000 | 1 | -1,816,286 | +0 | 0 | 0 | 0.400% |
| 500 | 50,000 | 2 | -1,821,069 | -4,783 | 4 | 2 | 0.410% |
| 500 | 50,000 | 3 | -1,756,189 | +60,097 | 22 | 6 | 0.455% |
| 500 | 100,000 | 1 | -1,816,286 | +0 | 0 | 0 | 0.400% |
| 500 | 100,000 | 2 | -1,816,286 | +0 | 0 | 0 | 0.400% |
| 500 | 100,000 | 3 | -1,802,639 | +13,647 | 7 | 2 | 0.417% |
| 1,000 | 25,000 | 1 | -1,794,039 | +22,248 | 2 | 0 | 0.405% |
| 1,000 | 25,000 | 2 | -1,777,503 | +38,783 | 16 | 4 | 0.440% |
| 1,000 | 25,000 | 3 | -1,696,713 | +119,574 | 38 | 9 | 0.495% |
| 1,000 | 50,000 | 1 | -1,816,286 | +0 | 0 | 0 | 0.400% |
| 1,000 | 50,000 | 2 | -1,823,859 | -7,573 | 4 | 2 | 0.410% |
| 1,000 | 50,000 | 3 | -1,770,624 | +45,662 | 18 | 5 | 0.445% |
| 1,000 | 100,000 | 1 | -1,816,286 | +0 | 0 | 0 | 0.400% |
| 1,000 | 100,000 | 2 | -1,816,286 | +0 | 0 | 0 | 0.400% |
| 1,000 | 100,000 | 3 | -1,808,446 | +7,840 | 6 | 2 | 0.415% |
