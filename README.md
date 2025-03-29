# Domain Adaptation with DANN for ISIC Dataset Classification

## Introduction
This repository focuses on domain adaptation for skin lesion classification using the **ISIC Dataset**. We employ **Domain-Adversarial Neural Networks (DANN)** to improve model performance across different domains. The repository includes scripts to train and evaluate models with and without DANN.

---

## 1. ISIC Dataset
The **International Skin Imaging Collaboration (ISIC) Dataset** is a large-scale dataset used for skin lesion classification. It contains images labeled with various skin disease types, helping in the development of computer-aided diagnosis tools.

### Sample Images from ISIC Dataset
| ![Sample 1](https://content.isic-archive.com/5d9fef95-eb18-4b85-9abe-01bd2c2669e4/ISIC_0053462_thumbnail_256.jpg?Expires=1743811200&Signature=lCooqtzotHP61H1jkEEo4JzTKsf742f32PHRhe6teFbOn3DQ-qo0LZaIAzxAmMb4EzQw9Ee57RJsXXtagpIP4VFFAqkJq6hVAqaJjr7tjVMZX3yOvONc9rHhBZYc3MWLJXEwGc4dNLC3Th35I0pRaU~ygozrn~2gb5Ytpw--97jjw2mbMmdbupPkPT8i9GUIYfTMB7FRwC6thd0D3MFH5hl1HHsqo6EDzRmH6L4aD1syGo4sX0bR1gddwkcizqBLABgo~d3HvsUVBTwLAFy2-4AMy51Tld6B0CMr3yyXaSUS7BIiEdUThBBp8N1defyogrgglGfw6fvaREbDBJmUWw__&Key-Pair-Id=K3KFHCM130RXTL) | ![Sample 2](https://content.isic-archive.com/2662a321-988d-4aca-9f20-111a0c70c175/ISIC_0053484_thumbnail_256.jpg?Expires=1743811200&Signature=b954ejBU87~OSBkC1scPaOgprfQjRi6dqVPE3zlrsBpUfRtK-zcOt6r6Ut9L7joKu7bfkrs6~HVA72bCpr6q7sUqpGM~S0sI9wmNT07UOYlRXy-Msw4d15Ex-Op1wqD3RXbdnCQMKjNYF-SJcVcB4tkppS8U5Rj4FpjxEOpxuRD~RaOQOyedKIIyGLDgkMVWlRCg0Y3oXT4pxt6o6erm0yAg4NQTASIftN1FaBKyVVLI0kX366EwIRUVhc8K73ABznO1ixM~c-ERCNSDvL~bCJI4fKf3t0J6h~Xyq5~ws0z7JXIBZyBejoTAkcBddU8k1ke6G9pZOR4stPPOUXSIPg__&Key-Pair-Id=K3KFHCM130RXTL) | ![Sample 3](https://content.isic-archive.com/0220db7b-05b3-482b-9898-939acb9f7d60/ISIC_0073243_thumbnail_256.jpg?Expires=1743811200&Signature=f0MbYSgV4sXZUMkcSzbzgjPRQUhx-oeQwHjNAT7sSlVrQ2LHKpA3yM~P5bRfUEZtdzrhct3UbFnV5oE-tbKGJ5fu6Q7yjs3UyBORARHTAMb4~vMitoRw~SGku7D0soLzoDd~K0EzJiBB6reOCG2xwIyCaW~CYfVvvINjq4vOZ4H4xHa~rWx2DvUlXmj5pWQBhWqD9xXYUuTRYWVarO2lFUKtFX8~nXyhdhwFSg5s33NVFL~luSMpGwLrdRgD5paRzA3Yq~bbQK2rjtdgn0JcpwO0KF2PK6bMTt-uSe7Zwxv01z3Z3H0Sy5uiPXkwj9fn47b-xFyDHN3GEnQfteKbMg__&Key-Pair-Id=K3KFHCM130RXTL) |
|:--------------------------------:|:--------------------------------:|:--------------------------------:|

## 2. Domain-Adversarial Neural Networks (DANN)
DANN is a transfer learning technique that minimizes domain shift between source and target datasets by introducing an adversarial loss. This method enhances model generalization across different distributions.

### Benefits of DANN in Transfer Learning
- Reduces domain discrepancy
- Enhances classification accuracy for unseen target domains
- Improves generalization of deep learning models

---

## 3. Accuracy Comparison: Without DANN vs. With DANN
We compare the accuracy of our model across **7 different classification tasks** before and after applying DANN:

| Task | Accuracy Without DANN | Accuracy With DANN |
|------|----------------------|-------------------|
| BA vs HA   | 49.01% | 95.50% |
| HA vs MA   | 91.02% | 92.02% |
| MA vs BA   | 85.89% | 91.90% |
| BLH vs HLH | 60.45% | 45.00% |
| BLH vs MLH | 59.25% | 76.80% |
| HLH vs MLH | 53.08% | 59.90% |
| BLP vs HLP | 17.89% | 74.70% |

---

## 4. How DANN Improves Performance
To demonstrate DANN's impact, we visualize t-SNE plots comparing feature distributions **before and after applying DANN**.

### t-SNE Plots:
Below are the t-SNE plots for different comparisons along with their A-Distance values:

| Task Name | A-Distance | Image Link |
|-----------|------------|-----------|
| BA vs HA | 1.07 | [View Image](https://ibb.co/MxZqvCyp) |
| HA vs MA | 0.01 | [View Image](https://ibb.co/8nGN1JSH) |
| MA vs BA | 1.01 | [View Image](https://ibb.co/sdfDmmTN) |
| BLH vs HLH | 1.05 | [View Image](https://ibb.co/DfDSfqCd) |
| BLH vs MLH | 1.05 | [View Image](https://ibb.co/LDDjbrm2) |
| HLH vs MLH | 0.2 | [View Image](https://ibb.co/BVpBj6YL) |
| BLP vs HLP | 1.25 | [View Image](https://ibb.co/j96GyRzz) |

Each plot provides a t-SNE visualization for different biological factors and comparisons.


Observations:
- Without DANN, feature distributions of different domains are scattered, leading to poor generalization.
- With DANN, feature distributions align better, improving classification accuracy.

---

## 5. Installation and Setup
### Prerequisites
Ensure the following packages are installed:
```bash
pip install requirements.txt
```

### Cloning the Repository
```bash
git clone https://github.com/planr2025/ISIC_DANN.git
cd ISIC_DANN
```

---

## 6. Running DANN for ISIC Dataset
The script to run DANN is provided in `run_dann.sh`. Follow these steps:

### 1. Make the script executable:
```bash
chmod +x run_dann.sh
```

### 2. Run the script:
```bash
./run_dann.sh
```

This script executes the DANN model on the **ISIC dataset**, adapting the classifier from the source domain **ham_loc_head_neck** to the target domain **msk_loc_head_neck**.

---

## Conclusion
By applying DANN, we successfully mitigate domain shift in the ISIC dataset, significantly improving classification accuracy across multiple tasks. The provided scripts can be further customized for different dataset partitions and tasks.


