# CompeteSMoE - Statistically Guaranteed Mixture of Experts Training via Competition

**Authors:** Nam V. Nguyen, Huy Nguyen, Quang Pham, Van Nguyen, Savitha Ramasamy, Nhat Ho

<p align="center">
  <a href="https://arxiv.org/abs/2505.13380">
    <img src="https://img.shields.io/badge/arXiv-2505.13380-red?style=flat&label=arXiv">
  </a>
  <a href="https://huggingface.co/collections/Fsoft-AIC/competesmoe-682b3801d431764f000429d1">
    <img src="https://img.shields.io/badge/HuggingFace-CompeteSMoE-blue?style=flat&logo=huggingface">
  </a>
</p>

![image](https://github.com/user-attachments/assets/4a7ac43e-0c8c-4720-8dec-5554196f5e86)


## 📌 About

Sparse mixture of experts (SMoE) offers an appealing solution to scale up the model complexity beyond the mean of increasing the network's depth or width. 
However, we argue that effective SMoE training remains challenging because of the suboptimal routing process where experts that perform computation do not directly contribute to the routing process. In this work, we propose  competition, a novel mechanism to route tokens to experts with the highest neural response. Theoretically, we show that the competition mechanism enjoys a better sample efficiency than the traditional softmax routing. Furthermore, we develop CompeteSMoE, a simple yet effective algorithm to train large language models by deploying a router to learn the competition policy, thus enjoying strong performances at a low training overhead. Our extensive empirical evaluations on both the visual instruction tuning and language pre-training tasks demonstrate the efficacy, robustness, and scalability of CompeteSMoE compared to state-of-the-art SMoE strategies.
## 🏆 Performance & Comparisons
![CompeteSMoE Performance](https://github.com/user-attachments/assets/67b54a45-9835-4fc7-8a49-cee70c25153d)

**Highlights:**  
- **CompeteSMoE-5.1B** demonstrates strong performance across a range of MoE routing strategies, including both standard and state-of-the-art routing methods.
- Achieves competitive results compared to recent MoE architectures such as **SharedE-V2** and **SharedE-V3** (inspired by DeepSeek).
- Despite architectural innovations of these models (e.g. shared experts), **CompeteSMoE-5.1B** consistently delivers superior or comparable results.


## 🗓️ Release Notes

| Date       | Release Notes                                                                 |
|------------|--------------------------------------------------------------------------------|
| 2025-05-20 | - Released CompeteSMoE 5.1B, trained on the LLAVA665K dataset. ✅              |
| 2025-05-20 | - Published CompeteSMoE paper and open-source code. ✅                         |

## 📌 Citation
If you find this repository useful, please consider citing our paper:

```bibtex
@misc{nguyen2025competesmoestatisticallyguaranteed,
      title={CompeteSMoE -- Statistically Guaranteed Mixture of Experts Training via Competition}, 
      author={Nam V. Nguyen and Huy Nguyen and Quang Pham and Van Nguyen and Savitha Ramasamy and Nhat Ho},
      year={2025},
      eprint={2505.13380},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2505.13380}
}
```












