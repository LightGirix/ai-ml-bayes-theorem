# Lab: Probabilistic Graphical Models & Naive Bayes
## Activity 1: Linear Chain
### Task 1: Coding (Colab)
![linear_chain.png](https://github.com/LightGirix/ai-ml-bayes-theorem/blob/main/linear_chain.png)

We are given:
| A = 0 | A = 1 |
| ----- | ----- |
| 0.9   | 0.1   |

| A     | B = 0 | B = 1 |
| ----- | ----- | ----- |
| A = 0 | 0.8   | 0.2   |
| A = 1 | 0.1   | 0.9   |

| B     | C = 0 | C = 1 |
| ----- | ----- | ----- |
| B = 0 | 0.95  | 0.05  |
| B = 1 | 0.30  | 0.70  |

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
![v_struc.png](https://github.com/LightGirix/ai-ml-bayes-theorem/blob/main/v_struc.png)

We are given:
| S = 0 | S = 1 |
| ----- | ----- |
| 0.8   | 0.2   |

| F = 0 | F = 1 |
| ----- | ----- |
| 0.9   | 0.1   |

| S     | F     | O = 0 | O = 1 |
| ----- | ----- | ----- | ----- |
| S = 0 | F = 0 | 0.99  | 0.01  |
| S = 0 | F = 1 | 0.30  | 0.70  |
| S = 1 | F = 0 | 0.20  | 0.80  |
| S = 1 | F = 1 | 0.05  | 0.95  |

We are finding $P(S|O) = \dfrac{P(O|S)P(S)}{P(O)}$

$P(O) = P(O|\neg S,\neg F)P(\neg S)P(\neg F) + P(O|\neg S, F)P(\neg S)P(F) + P(O|S,\neg F)P(S)P(\neg F) + P(O|S,F)P(S)P(F)$ \
$P(O) = (0.01\times 0.8\times 0.9) + (0.70\times 0.8\times 0.1) +(0.80\times 0.2\times 0.9) + (0.95\times 0.2\times 0.1)$ \
$P(O) = 0.2262$

$P(O|S) = P(O|S,\neg F)P(\neg F) + P(O|S,F)P(F)$ \ 
$P(O|S) = (0.80\times 0.9) + (0.95\times 0.1)$ \
$P(O|S) = 0.815$ 

$P(S|O) = \dfrac{0.815\times 0.2}{0.2262}$ \
$P(S|O) \approx 0.7206$

### Task 3: Coding (Diamond Network)
![diamond_netw.png](https://github.com/LightGirix/ai-ml-bayes-theorem/blob/main/diamond_netw.png)

Let's assume some probabilities!

So, let's say an AI model has a fair chance of being a quality one. We will assume, that around 55% of AI models are quality models.

| A = 0 | A = 1 |
| ----- | ----- |
| 0.45  | 0.55  |

Next, if an AI model is a quality one, then the chances for it to give an inaccurate prediction must be pretty low, let's say 10%

However, if it's not, then there's a fairly high chance it'll be inaccurate, let's say 60%

| A     | B = 0 | B = 1 |
| ----- | ----- | ----- |
| A = 0 | 0.60  | 0.40  |
| A = 1 | 0.10  | 0.90  |

Then, the latency, this is a tricky one, a quality AI model doesn't necessarily has to have a low latency, however latency does contribute towards quality, and therefore it's a nice-to-have feature. Let's say a quality AI model has a 35% chance of having low latency.

If an AI is not a quality one, then the chances of having low latency must be lower, let's say 15%.

| A     | D = 0 | D = 1 |
| ----- | ----- | ----- |
| A = 0 | 0.85  | 0.15  |
| A = 1 | 0.65  | 0.35  |

Now, let's move on to customer's satisfaction.

A customer would be at least likely to be unsatisfied by an AI model with accurate prediction and low latency, let's say around 2% does feel unsatisfied for some reason.

If the latency is not low, yet it's still accurate, the overall customer satisfaction wouldn't change much, let's say around 10% of the customers are unsatisfied because some of are impatient.

If a model is inaccurate even with low latency, the overall satisfaction would be real low, let's say around 15% of people is satisfied because it can gave out basic answers. 

If an AI model is neither accurate or has low latency, most people wouldn't be satisfied for sure, let's say around 5% of people is satisfied due to the future potential of the model.

| B     | D     | C = 0 | C = 1 |
| ----- | ----- | ----- | ----- |
| B = 0 | D = 0 | 0.95  | 0.05  |
| B = 0 | D = 1 | 0.85  | 0.15  |
| B = 1 | D = 0 | 0.10  | 0.90  |
| B = 1 | D = 1 | 0.02  | 0.98  |

Now let's find out "How likely for an AI to be a quality model if a customer is satisfied with it?" $P(A|C)$

$P(A|C) = \dfrac{P(C|A)P(A)}{P(C)}$

By applying probability 

$P(C|A) = P(C|\neg B, \neg D)P(\neg B|A)P(\neg D|A) + P(C|\neg B,D)P(\neg B|A)P(D|A) + P(C|B,\neg D)P(B|A)P(\neg D|A) + P(C|B,D)P(B|A)P(D|A)$
$P(C|A) = (0.05\times 0.10\times 0.65) + (0.15\times 0.10\times 0.35) + (0.90\times 0.90\times 0.65) + (0.98\times 0.90\times 0.35)$
$P(C∣A) \approx 0.844$
