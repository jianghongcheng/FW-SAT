# FAW-SAT

### Flexible Window-based Self-attention Transformer in Thermal Image Super-Resolution  
📄 [[Paper Link (CVPR 2024 Workshop)]](https://openaccess.thecvf.com/content/CVPR2024W/PBVS/html/Jiang_Flexible_Window-based_Self-attention_Transformer_in_Thermal_Image_Super-Resolution_CVPRW_2024_paper.html)

**Authors:**  
[Hongcheng Jiang](https://github.com/jianghongcheng)  
[Zhiqiang Chen](https://sse.umkc.edu/profiles/zhiqiang-chen.html)

---

## 📚 Citation

If you find this work helpful in your research, please cite:

```bibtex
@InProceedings{Jiang_2024_CVPR,
    author    = {Jiang, Hongcheng and Chen, Zhiqiang},
    title     = {Flexible Window-based Self-attention Transformer in Thermal Image Super-Resolution},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
    pages     = {3076--3085}
}
```

---

## 🛠️ Environment Setup

This project is implemented using [PyTorch](https://pytorch.org/) and [BasicSR](https://github.com/XPixelGroup/BasicSR).

**Dependencies:**
- PyTorch >= 2.0  
- BasicSR == 1.3.4.9  

### 🔧 Installation

Clone the repository and install the required packages:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### 🔍 Inference

To perform inference with a trained model:

```bash
python predict.py
```

### 🏋️‍♀️ Training

To train the model:

1. Set the dataset path in `dataloader.py`.  
2. Adjust model and training configuration in `option.yaml`.  
3. Start training:

```bash
python train.py
```

---

## 📬 Contact

If you have any questions, feedback, or collaboration ideas, feel free to reach out:

- 💻 Website: [jianghongcheng.github.io](https://jianghongcheng.github.io/)
- 📧 Email: [hjq44@mail.umkc.edu](mailto:hjq44@mail.umkc.edu)
- 🏫 Affiliation: University of Missouri–Kansas City (UMKC)
