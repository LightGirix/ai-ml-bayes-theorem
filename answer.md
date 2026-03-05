# Lab: Probabilistic Graphical Models & Naive Bayes
## Activity 1: Linear Chain
### Task 1: Coding (Colab)
We are given:
| A = 0  | A = 1 |
|---|---|
| 0.9 | 0.1 |

|       | A = 0 | A = 1 |
|-------|-----|-----|
| B = 0 | 0.8 | 0.1 |
| B = 1 | 0.2 | 0.9 |

|       | B = 0 | B = 1 |
|-------|-----|-----|
| C = 0 | 0.95 | 0.3 |
| C = 1 | 0.05 | 0.7 |


Since $P(A|C) = \dfrac{P(C|A)P(A)}{P(C)}$ 

To find $P(C)$, we need to find $P(B)$ first:
From the Law of Total Probability: \
$P(B) = P(B|A)P(A) + P(B|\neg A)P(\neg A)$ \
$P(B) = 0.9\times 0.1 + 0.2\times 0.9$ \
$P(B) = 0.27$

|   |   |
|---|---|
| B = 0 | 0.73 |
| B = 1 | 0.27 |

$P(C) = P(C|B)P(B) + P(C|\neg B)P(\neg B)$ \
$P(C) = 0.7\times 0.27 + 0.05\times 0.73$ \
$P(C) = 0.2255$

|   |   |
|---|---|
| C = 0 | 0.7745 |
| C = 1 | 0.2255 |

From the Chapman-Kolmogorov equation: \
$P(C|A) = P(C|B)P(B|A) + P(C|\neg B)P(\neg B|A)$ \
$P(C|A) = 0.7\times 0.9 + 0.05\times 0.1$ \
$P(C|A) = 0.635$

$P(A|C) = \dfrac{0.635\times 0.1}{0.2255}$ \
$P(A/C) \approx 0.2816$

## Activity 2: V-Structure and Diamond Network
### Task 2: Coding (V-Structure)
We are given:
| S = 0  | S = 1 |
|---|---|
| 0.8 | 0.2 |

| F = 0  | F = 1 |
|---|---|
| 0.9 | 0.1 |

|              | O = 0 | O = 1 |
|--------------|-----|-----|
| S = 0, F = 0 | 0.99 | 0.01 |
| S = 0, F = 1 | 0.30 | 0.70 |
| S = 1, F = 0 | 0.20 | 0.80 |
| S = 1, F = 1 | 0.05 | 0.95 |

We are finding $P(S|O) = \dfrac{P(O|S)P(S)}{P(O)}$

$P(O) = P(O|\neg S,\neg F)P(\neg S)P(\neg F) + P(O|\neg S, F)P(\neg S)P(F) + P(O|S,\neg F)P(S)P(\neg F) + P(O|S,F)P(S)P(F)$ \
$P(O) = (0.01\times 0.8\times 0.9) + (0.70\times 0.8\times 0.1) +(0.80\times 0.2\times 0.9) + (0.95\times 0.2\times 0.1)$ \
$P(O) = 0.2262$

$P(O|S) = P(O|S,\neg F)P(\neg F) + P(O|S,F)P(F)$ \ 
$P(O|S) = (0.80\times 0.9) + (0.95\times 0.1)$ \
$P(O|S) = 0.815$ 

$P(S|O) = \dfrac{0.815\times 0.2}{0.2262}$ \
$P(S|O) \approx 0.7206$
