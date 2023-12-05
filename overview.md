# ⚙️ MiDas 🔎
*by Truong Giang Nguyen, Thanh Le-Cong, Hong Jin Kang, Ratnadira Widyasari, Chengran Yang, Zhipeng Zhao, Bowen Xu, Jiayuan Zhou, Xin Xia, Ahmed E. Hassan, Xuan-Bach D. Le, David Lo*
<p align="center">
    <a href="https://ieeexplore.ieee.org/document/10138621"><img src="https://img.shields.io/badge/Journal-IEEE TSE Volume 49 (2023)-green?style=for-the-badge">
    <a href="https://arxiv.org/pdf/2305.13884.pdf"><img src="https://img.shields.io/badge/arXiv-2305.13884-b31b1b.svg?style=for-the-badge">
    <a href="https://hub.docker.com/r/thanhlecong/midas"><img src="https://img.shields.io/badge/docker-thanhlecong%2Fmidas-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white"></a>
</p>

## 💥 Approach

With the increasing reliance on Open Source Software, users are exposed to third-party library vulnerabilities. Software Composition Analysis (SCA) tools have been created to alert users of such vulnerabilities. SCA requires the identification of vulnerability-fixing commits. Prior works have proposed methods that can automatically identify such vulnerability-fixing commits. However, identifying such commits is highly challenging, as only a very small minority of commits are vulnerability fixing. Moreover, code changes can be noisy and difficult to analyze. We observe that noise can occur at different levels of detail, making it challenging to detect vulnerability fixes accurately. 

To address these challenges and boost the effectiveness of prior works, we propose MiDas (Multi-Granularity Detector for Vulnerability Fixes). Unique from prior works, MiDas constructs different neural networks for each level of code change granularity, corresponding to commit-level, file-level, hunk-level, and line-level, following their natural organization. It then utilizes an ensemble model that combines all base models to generate the final prediction. This design allows MiDas to better handle the noisy and highly imbalanced nature of vulnerability-fixing commit data. Additionally, to reduce the human effort required to inspect code changes, we have designed an effort-aware adjustment for MiDas’s outputs based on commit length.

<p align="center">
<img width="750" alt="Screenshot 2023-12-05 at 9 32 17 pm" src="https://github.com/soarsmu/midas/assets/43113794/8be4b9f5-30d3-443f-bf9a-e2fa67ed85e9">
</p>

## 📈 Experimental Results

### Effectiveness
The evaluation results demonstrate that MiDas outperforms the current state-of-the-art baseline in terms of AUC by 4.9% and 13.7% on Java and Python-based datasets, respectively. Furthermore, in terms of two effort-aware metrics, EffortCost@L and Popt@L, MiDas also outperforms the state-of-the-art baseline, achieving improvements of up to 28.2% and 15.9% on Java, and 60% and 51.4% on Python, respectively. 
<p align="center">
<img width="750" alt="Screenshot 2023-12-05 at 9 35 09 pm" src="https://github.com/soarsmu/midas/assets/43113794/e984c86c-f4f1-42b9-a095-36e9ed4f22fb">
</p>

### Effectiveness on real-world project (TensorFlow)
MiDas can effectively classify vulnerability-fixing commits in the TensorFlow framework with an AUC of 0.88. These experimental results also suggest that MiDas can significantly reduce human efforts in identifying vulnerability-fixing commits. For instance, MiDas can detect 81% of vulnerabilities by examining just 5% of the lines of code, and this figure increases to 94% when examining 20% of the code. The results show that MiDas is promising on reducing human efforts on monitoring vulnerability-fixing commits from real projects.
<p align="center">
<img width="550" alt="Screenshot 2023-12-05 at 9 37 24 pm" src="https://github.com/soarsmu/midas/assets/43113794/acc76132-790d-4a67-aa9f-62521819baef">
</p>
