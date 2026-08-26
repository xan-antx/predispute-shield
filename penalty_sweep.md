# Ratio-penalty sensitivity sweep

27 penalty curves (floor x ceiling x exponent), each treated as the true cost
structure of its world: the EV strategy decides with that curve and fights are
charged with it at the ratio standing when they were filed (start 0.40%, every
fight counts from filing). One canonical seed-42 dataset and one trained model
throughout, so the curve is the only thing that varies. Δ = system minus
always_refund (-1,816,286) in net ₹ per 1000 alerts; positive (bold) wins.

**System beats always_refund in 23 of 27 cells, ties it in 3, and lands below it in 1.** Every tie is a zero-fight cell: the curve prices all fights out and the system degenerates to always_refund exactly. Each below-incumbent cell sits within one to two flipped fight outcomes of zero (a flip moves Δ by ₹6,000 per 1000), which is indistinguishable from the incumbent at this sample size. Winning Δ ranges +628 to +319,330; the worst cell anywhere is -283.

### exponent 1 (linear)

| floor \ ceiling | 25,000 | 50,000 | 100,000 |
|---|--:|--:|--:|
| 250 | **+3,290** | **+12,487** | tie (0 fights) |
| 500 | -283 | **+11,877** | tie (0 fights) |
| 1,000 | **+9,239** | **+10,656** | tie (0 fights) |

### exponent 2 (quadratic)

| floor \ ceiling | 25,000 | 50,000 | 100,000 |
|---|--:|--:|--:|
| 250 | **+156,013** | **+62,903** | **+8,563** |
| 500 | **+146,172** | **+50,530** | **+4,490** |
| 1,000 | **+129,296** | **+39,814** | **+628** |

### exponent 3 (cubic)

| floor \ ceiling | 25,000 | 50,000 | 100,000 |
|---|--:|--:|--:|
| 250 | **+319,330** | **+201,291** | **+126,081** |
| 500 | **+292,031** | **+187,476** | **+120,040** |
| 1,000 | **+245,677** | **+166,821** | **+93,608** |

## Cells at or below the incumbent

- floor 500, ceiling 25,000, exponent 1: Δ -283 (8 fights)

## Shape versus magnitude

Mean Δ per axis value, and the mean spread of Δ when one axis moves across its
full range while the other two stay fixed:

| Axis | Values -> mean Δ | Spread when varied alone |
|---|---|--:|
| exponent | 1: +5,252, 2: +66,490, 3: +194,706 | 189,454 |
| floor | 250: +98,884, 500: +90,259, 1,000: +77,304 | 23,299 |
| ceiling | 25,000: +144,529, 50,000: +82,651, 100,000: +39,268 | 107,823 |

## Detail

| Floor | Ceiling | Exp | System ₹/1000 | Δ | Fought | Lost | End ratio |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 250 | 25,000 | 1 | -1,812,996 | +3,290 | 9 | 3 | 0.422% |
| 250 | 25,000 | 2 | -1,660,273 | +156,013 | 46 | 12 | 0.515% |
| 250 | 25,000 | 3 | -1,496,956 | +319,330 | 88 | 20 | 0.620% |
| 250 | 50,000 | 1 | -1,803,799 | +12,487 | 2 | 0 | 0.405% |
| 250 | 50,000 | 2 | -1,753,383 | +62,903 | 22 | 6 | 0.455% |
| 250 | 50,000 | 3 | -1,614,995 | +201,291 | 62 | 16 | 0.555% |
| 250 | 100,000 | 1 | -1,816,286 | +0 | 0 | 0 | 0.400% |
| 250 | 100,000 | 2 | -1,807,723 | +8,563 | 6 | 2 | 0.415% |
| 250 | 100,000 | 3 | -1,690,205 | +126,081 | 40 | 10 | 0.500% |
| 500 | 25,000 | 1 | -1,816,569 | -283 | 8 | 3 | 0.420% |
| 500 | 25,000 | 2 | -1,670,114 | +146,172 | 44 | 11 | 0.510% |
| 500 | 25,000 | 3 | -1,524,255 | +292,031 | 83 | 19 | 0.607% |
| 500 | 50,000 | 1 | -1,804,409 | +11,877 | 2 | 0 | 0.405% |
| 500 | 50,000 | 2 | -1,765,756 | +50,530 | 20 | 6 | 0.450% |
| 500 | 50,000 | 3 | -1,628,811 | +187,476 | 56 | 15 | 0.540% |
| 500 | 100,000 | 1 | -1,816,286 | +0 | 0 | 0 | 0.400% |
| 500 | 100,000 | 2 | -1,811,796 | +4,490 | 5 | 2 | 0.413% |
| 500 | 100,000 | 3 | -1,696,246 | +120,040 | 38 | 9 | 0.495% |
| 1,000 | 25,000 | 1 | -1,807,047 | +9,239 | 6 | 2 | 0.415% |
| 1,000 | 25,000 | 2 | -1,686,990 | +129,296 | 40 | 9 | 0.500% |
| 1,000 | 25,000 | 3 | -1,570,609 | +245,677 | 64 | 16 | 0.560% |
| 1,000 | 50,000 | 1 | -1,805,630 | +10,656 | 2 | 0 | 0.405% |
| 1,000 | 50,000 | 2 | -1,776,472 | +39,814 | 18 | 5 | 0.445% |
| 1,000 | 50,000 | 3 | -1,649,465 | +166,821 | 47 | 12 | 0.517% |
| 1,000 | 100,000 | 1 | -1,816,286 | +0 | 0 | 0 | 0.400% |
| 1,000 | 100,000 | 2 | -1,815,658 | +628 | 5 | 2 | 0.413% |
| 1,000 | 100,000 | 3 | -1,722,678 | +93,608 | 32 | 8 | 0.480% |
