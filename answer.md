# Lab: Probabilistic Graphical Models & Naive Bayes
## Activity 1: Linear Chain
We currently have:
|   |   |
|---|---|
| A = 0 | 0.9 |
| A = 1 | 0.1 |

|       | A = 0 | A = 1|
|-------|-----|-----|
| B = 0 | 0.8 | 0.1 |
| B = 1 | 0.2 | 0.9 |

|       | B = 0 | B = 1|
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
