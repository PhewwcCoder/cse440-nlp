# Machine Learning Essentials

> CSE440 — Natural Language Processing (Chapter 3)
> Classifiers, Bag of Words, probability, Naïve Bayes, neural networks, optimization, and evaluation.

## Class 02 — 16 Feb 2026

### Classifier

A **classifier** is a **function** that maps a feature representation to a category label:

```
h(x) → y
```

- **x** — *feature*: the representation of some object
- **y** — *category label*

Formally, `h ∈ O → Y`: a function that takes an element from set **O** and outputs an element in set **Y**.

### Two Types of Classifiers

1. **Generative classifier** → e.g. **Naïve Bayes**
   - Models the joint probability: `P(x, y) = P(x | y) · P(y)`
2. **Discriminative classifier** → e.g. **Logistic Regression**
   - Models the conditional probability directly: `P(y | x)`
   - Learns a **decision boundary** between classes.

### Data Layout

- **X matrix** — rows are objects, columns are features:

```
f₁(o₁)  f₂(o₁)  ...  fₘ(o₁)
  ⋮       ⋮            ⋮
f₁(oₙ)  f₂(oₙ)  ...  fₘ(oₙ)
```

- **Y vector** — the labels `y₁ ... yₙ`
- **Feature vector** for one object: `x = [f₁(o), f₂(o), ..., fₘ(o)]`

### Generative vs Discriminative

**Generative classifier** — models *how the input is generated*.

> If the class is fixed, what kind of feature patterns do we expect?

Example (spam detection):

- Instead of asking: *"Given this email, is it spam?"*
- Generative thinking asks: *"Suppose I tell you this email **is** spam — what kinds of words would you expect to see?"*
- That expectation is `P(x | y = spam)`.
- A generative model can actually **generate fake samples**.

**Discriminative classifier** — directly learns the **boundary** between classes.

---

## Bag of Words (BoW)

**Bag of Words (BoW)** → a **numerical representation** of text.

Each document becomes a vector of word counts over the vocabulary:

| Sentence | dog | bite | man | is | a | pet | animal |
|---|---|---|---|---|---|---|---|
| (i) dog bite man | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| (ii) man bite dog | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| (iii) dog is a pet animal | 1 | 0 | 0 | 1 | 1 | 1 | 1 |

Note: BoW **cannot capture context**, and the matrix is **sparse** (lots of zeros).

**Issues:**

1. Large + sparse matrices
2. Completely ignores word order (sentences (i) and (ii) look identical)

…but BoW is **often still useful**.

**Classwork example (spam corpus):**

- *Sorry I'll call later*
- *U can call me now*
- *U have won call now!!*
- *Sorry! U can not unsubscribe*

---

## Probability Review

**Addition rule:**

```
P(a ∨ b) = P(a) + P(b) − P(a ∧ b)
```

Key terms:

- **Prior** — probability of a class **before** observing any data.
- **Joint** — both the data and the class occur **together** (multiple events at the same time): `P(A, B)` or `P(A ∧ B)`.
- **Conditional** — one event **given** that another event is known.

### Conditional Probability

```
P(A | B) = P(A and B together) / P(B)
         = P(A ∩ B) / P(B)
```

Rearranging gives the **Product Rule**:

```
P(A ∩ B) = P(A | B) · P(B)
```

For events conditionally independent given C:

```
P(A ∩ B | C) = P(A | C) · P(B | C)
```

### Bayes' Rule

Starting from the product rule `P(A ∩ B) = P(A | B) · P(B)` — (1)

```
P(A | B) = P(A ∩ B) / P(B)
```

Swapping A and B:

```
P(B | A) = P(A ∩ B) / P(A)
```

Substituting (1):

```
P(B | A) = P(A | B) · P(B) / P(A)     ← Bayes' Rule
```

**Spam example:**

```
P(spam | "you won jackpot") = P("you won jackpot" | spam) · P(spam) / P("you won jackpot")
```

### Prediction Rule

```
h(o) = argmax_{y ∈ Y} P(y | o)
```

- Returns the **class** (index) with the highest probability.
- For `y ∈ {+, −}`, compare `P(+ | o)` and `P(− | o)` and take whichever is **max**.

Expanding with Bayes:

```
P(+ | o) = P(o | +) · P(+) / P(o)
P(− | o) = P(o | −) · P(−) / P(o)
```

The denominator `P(o)` is the **same** for both classes, so it can be **ignored** when comparing — it doesn't change which class wins.

### Naïve Bayes Assumption

Using the full chain rule is **computationally complex**. **Naïve Bayes** assumes **every feature is independent** of the others:

```
P(you, won, jackpot | y) = P(you | y) · P(won | y) · P(jackpot | y)
```

Word likelihood:

```
P(word | class) = count of 'word' in that class / count of all words in that class
```

---

## Naïve Bayes Worked Example — 18 Feb 2026

```
h(o) = argmax_{y ∈ Y} P(y) · ∏_{i=1}^{m} P(fᵢ(o) | y)
```

**Laplace (add-one) smoothing** avoids zero probabilities:

```
P(word | class) = (count(word) in that class + 1) / (count of all words in that class + |V|)
```

where **|V|** = vocabulary size (number of unique words in the training data).

**Positive class:**

```
P(+ | o) ∝ P(+) · P(! | +) · P(sorry | +) · P(U | +) · P(can | +)
         = 2/3 × (0+1)/(10+12) × (1+1)/(10+12) × (1+1)/(10+12) × (1+1)/(10+12)
         ≈ 0.000023
```

**Negative class:**

```
P(− | o) ∝ P(−) · P(! | −) · P(sorry | −) · P(U | −) · P(can | −)
         = 1/3 × (2+1)/(7+12) × (0+1)/(7+12) × (1+1)/(7+12) × (0+1)/(7+12)
         ≈ 0.000015
```

(vocabulary size |V| = 12)

**Decision:** `h(o) = +ve`, since `0.000023 > 0.000015`.

### Refining Features

- **Lexicon** — an extra count of positive / negative emotion words from a **pre-annotated list**.

1. **Binarization** — instead of using word **frequency**, only record whether a word is **present or not**. Equivalent to removing duplicate words in each document.
2. **Lexicon features** — add sentiment counts from a lexicon.

---

## Loss & Neural Networks — 23 Feb 2026

### Concept of Loss

**Binary Cross-Entropy (BCE) loss:**

```
Loss_BCE = −[ y·log(ŷ) + (1 − y)·log(1 − ŷ) ]
```

Example — actual `y = 0`, prediction `ŷ = 0.1`:

```
Loss_BCE = −[ 0·log(0.1) + (1−0)·log(1−0.1) ] = −log(0.9)
```

### Neural Network Structure

- **Input layer** — e.g. an image, or a bag-of-words **vectorized** input.
- **Hidden layer(s)**
- **Output layer**

### A Single Neuron

For inputs `x₁…x₄` with weights `ω₁…ω₄`:

```
a₁ = x₁ω₁ + x₂ω₂ + x₃ω₃ + x₄ω₄ + b
```

The result passes through an **activation function** G.

**Bias trick** — let `w₀ = b` and `x₀ = 1`:

```
a₁ = x₀w₀ + x₁ω₁ + x₂ω₂ + x₃ω₃ + x₄ω₄
```

Why? So it can be written as a **matrix product**:

```
a = [x₀ x₁ x₂ x₃ x₄]  ·  [w₀ w₁ w₂ w₃ w₄]ᵀ
        (1×5)                  (5×1)

output = G(X · W)
```

### Stacking Layers

For inputs `[x₀ x₁ x₂]` (1×3) and weights `W⁽¹⁾` (3×2):

```
[x₀ x₁ x₂] · [w₁₀ w₂₀;  w₁₁ w₂₁;  w₁₂ w₂₂] = [σ(a₁)  σ(a₂)]
```

- Output of **first** hidden layer: `G(X · W⁽¹⁾)`
- Output of **second** hidden layer: `G( G(X · W⁽¹⁾) · W⁽²⁾ )`

---

## Training Neural Networks — 25 Feb 2026

- **ReLU note:** it's better to use **multiple activation functions** in a network.

### How a NN Works

**Forward propagation:**

```
Dataset → NN Model → Output
```

**Backward propagation:**

```
Output → Loss calculate → Gradient → Weight update → (back to NN Model)
```

### Optimizers

An **optimizer** updates the weights (gradient descent):

- **Stochastic Gradient Descent (SGD)** — computes loss and updates weights for **each single row** of the dataset.
- **Gradient Descent (batch)** — uses the **entire dataset**; learning rate is fixed.
- **Mini-batch** — updates on small batches (often a power of two, like 2ⁿ rows).
- **ADAM (Adaptive Moment Estimation)** — the **default** optimizer we usually use.

### Underfitting vs Overfitting

- **Underfitting** — model is **not complex enough**.
- **Overfitting** — model is **too complex**; it fits the **noise** too, which hurts generalization.
  - Which features to drop to make it simpler? → **Regularization**.

There is an **optimal complexity** where bias and variance balance (the bias–variance trade-off vs complexity).

### Data Splitting

Split the data into **3 parts**:

- **Train**
- **Validation**
- **Test**

---

## Evaluation — 02 Mar 2026

**Accuracy is not always the best metric** → it fails on **imbalanced** data.

### Confusion Matrix

Example: 100 emails → 70 spam, 30 not spam. The model predicts 80 spam, 20 not spam (20 not-spam correct).

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | TN = 20 | FP = 10 |
| **Actual 1** | FN = 0 | TP = 70 |

### Precision vs Recall

- **Precision** is about **False Positives** (e.g. a real email marked as spam).
- **Recall** is about **False Negatives** (a spam email that is missed).
- Combine the two → **F1 score**.

> A "100% accurate" model is a warning sign — it has likely **learned the noise** (overfitting).
