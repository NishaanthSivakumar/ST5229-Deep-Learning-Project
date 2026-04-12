# ST5229 Deep Learning Project  
## Literature Review, Systematic Experimentation and Extension of Masked Autoencoders are Scalable Vision Learners

This repository contains the final project for the **ST5229 Deep Learning in Data Analytics** module at the National University of Singapore.

The project explores **Masked Autoencoders (MAE)** through a combination of:
1. Literature Review  
2. Model Understanding & Visualisation  
3. Systematic Experimentation  
4. Extended Applications  

---

## 📌 Project Overview

Masked Autoencoders (MAE), introduced in *"Masked Autoencoders Are Scalable Vision Learners"*, are a class of **self-supervised learning models** that learn representations by reconstructing masked portions of input images.

This project aims to:
- Understand the theoretical foundations of MAE
- Analyse its implementation and behaviour
- Evaluate its sensitivity to key hyperparameters
- Extend its application to new domains and masking strategies

---

## 📂 Repository Structure
```text
.
├── Literature Review & Explanation & Examples/
├── Semantic Masking on CIFAR 10/
├── MAE experimentation on CIFAR 10/
├── MAE on Medical Images/
└── .gitignore
```

---

# 🧠 Part 1: Literature Review

This section focuses on the paper:

> **"Masked Autoencoders Are Scalable Vision Learners"**

### Key contributions studied:
- Asymmetric encoder-decoder architecture
- High masking ratios (up to 75%)
- Efficient representation learning
- Scalability to large datasets

### What we did:
- Broke down the methodology and architecture
- Explained the intuition behind masking
- Analysed why MAE works effectively
- Provided simplified explanations and examples

---

# 🔍 Part 2: Model Understanding & Visualisation

This component focuses on understanding MAE **at an implementation level**.

### Key tasks:
- Analysed the MAE pipeline step-by-step
- Visualised:
  - Masked inputs
  - Reconstructed outputs
  - Learned representations


### Objective:
To gain intuition on:
- What the model learns
- How reconstruction improves over training
- The effect of masking on input structure

---

# 📊 Part 3: Experimentation on CIFAR-10

We conducted systematic experiments on CIFAR-10 by varying key hyperparameters.

### Experiments performed:

#### 1. Masking Ratio
- Tested different masking levels (e.g., 50%, 75%, 90%)
- Observed trade-offs between:
  - Reconstruction quality
  - Representation learning

#### 2. Patch Size
- Smaller vs larger patches
- Impact on spatial information retention

#### 3. Encoder Size
- Varied model capacity
- Evaluated performance vs computational cost

---

### Evaluation method:
- Reconstruction loss
- Visual inspection
- Linear probing (downstream performance)

---

### Key insights:
- High masking ratios still allow meaningful learning
- Patch size significantly affects reconstruction quality
- Larger encoders improve performance but with diminishing returns

---

# 🧪 Part 4: Extensions & Applications

## 4.1 MAE on Medical Images

We extended MAE to a different domain:
- Medical imaging data

### Objective:
- Test generalisation capability of MAE
- Observe behaviour on structured, high-detail images

### Observations:
- MAE can reconstruct medical images reasonably well
- Domain complexity affects reconstruction fidelity

---

## 4.2 Semantic Masking on CIFAR-10

Instead of random masking, we explored **semantic masking**.

### Idea:
- Mask meaningful regions (instead of random patches)
- Force the model to learn more structured representations

### Implementation:
- Custom masking strategy applied to CIFAR-10 images

### Results:
- Improved reconstruction in some cases
- Better preservation of important structures
- Trade-offs depending on masking strategy

---

# 📈 Summary of Contributions

This project provides:

- A **clear explanation** of MAE theory  
- A **hands-on implementation and visual analysis**  
- A **systematic experimental study** of key hyperparameters  
- **Extensions to new domains and masking strategies**  

---

# 🔮 Future Work

- Apply MAE to larger datasets (e.g., ImageNet)
- Explore transformer scaling
- Combine semantic masking with adaptive masking
- Fine-tune for classification tasks

---

# 👤 Author

**Gao Demin,** **Kong Moon Hyeong,** **Mahapatra Onkar,** **Sivakumar Nishaanth.**  
MSc Statistics, National University of Singapore  

---

# 📜 Notes

- Dataset is not included in this repository
- This project is part of academic coursework for ST5229