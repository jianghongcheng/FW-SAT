# FAW-SAT

### Flexible Window-based Self-attention Transformer in Thermal Image Super-Resolution  
📄 [[Paper Link (CVPR 2024 Workshop)]](https://openaccess.thecvf.com/content/CVPR2024W/PBVS/html/Jiang_Flexible_Window-based_Self-attention_Transformer_in_Thermal_Image_Super-Resolution_CVPRW_2024_paper.html)

**Authors:**  
[Hongcheng Jiang](https://jianghongcheng.github.io/)  
[Zhiqiang Chen](https://sse.umkc.edu/profiles/zhiqiang-chen.html)

---
---

## 🔍 Overview

Thermal imaging plays a crucial role in various applications such as surveillance, agriculture, and medical diagnostics due to its ability to capture infrared radiation emitted by objects. However, thermal images often suffer from low spatial resolution, limiting their effectiveness. High-resolution thermal sensors are expensive and constrained by hardware limitations, making software-based super-resolution (SR) approaches a cost-effective alternative.

**FAW-SAT** (Flexible Window-based Self-attention Transformer) is proposed to enhance the resolution of low-quality thermal images. The model integrates multiple attention mechanisms to capture information at different scales:

- **Channel and Spatial Attention**: Captures global contextual information.
- **Window-based Self-Attention**: Focuses on local features within fixed-size windows.
- **Flexible Window-based Self-Attention**: Aggregates regional features by adapting window sizes based on the content.

By combining these mechanisms, FAW-SAT effectively reconstructs high-resolution thermal images from low-resolution inputs. The model has demonstrated superior performance in both qualitative and quantitative evaluations, surpassing state-of-the-art techniques in the PBVS-2024 Thermal Image Super-Resolution Challenge (GTISR) - Track2.


---

## 📈 Performance Gains

We propose **FW-SAT**, a Flexible Window-based Self-attention Transformer for thermal image super-resolution.

✅ **Results on Validation Set**

**×8 Upscaling:**
- FW-SAT achieves **27.80 dB / 0.8815** in PSNR/SSIM, outperforming all competitors:
  - **+2.82 dB / +0.0645** vs. SwinIR
  - **+1.94 dB / +0.0385** vs. HAN
  - **+2.21 dB / +0.0410** vs. GRL
  - **+2.14 dB / +0.0421** vs. EDSR

**×16 Upscaling:**
- FW-SAT achieves **24.61 dB / 0.8116**, again setting a new benchmark:
  - **+3.39 dB / +0.0839** vs. SwinIR
  - **+1.92 dB / +0.0525** vs. HAN
  - **+2.23 dB / +0.0616** vs. GRL
  - **+2.02 dB / +0.0554** vs. EDSR

These consistent improvements across scales and metrics validate FW-SAT’s strong generalization and superior spatial-spectral learning capabilities.



---

## 🧠 Network Architecture

<p align="center">
  <img src="https://github.com/jianghongcheng/FW-SAT/blob/main/Figures/Network.png" width="800"/>
</p>


---

## 🧩 Flexible Window Attention Module

<p align="center">
  <img src="https://github.com/jianghongcheng/FW-SAT/blob/main/Figures/Flexible_Window_Att.png" width="800"/>
</p>


---

## 🖼️ Visual Results

<p align="center"><strong>Comparison with State-of-the-Art Methods</strong></p>
<p align="center">
  <img src="https://github.com/jianghongcheng/FW-SAT/blob/main/Figures/Visual_Result.png" width="800"/>
</p>


---




## 📊 Quantitative Results

<p align="center"><b>Table: Quantitative comparison with state-of-the-art methods on the validation dataset (PSNR/SSIM)</b></p>

<div align="center">

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="2">×8</th>
      <th colspan="2">×16</th>
    </tr>
    <tr>
      <th>PSNR</th>
      <th>SSIM</th>
      <th>PSNR</th>
      <th>SSIM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>EDSR</td>
      <td>25.66</td>
      <td>0.8394</td>
      <td>22.59</td>
      <td>0.7562</td>
    </tr>
    <tr>
      <td>SwinIR</td>
      <td>24.98</td>
      <td>0.8170</td>
      <td>21.22</td>
      <td>0.7277</td>
    </tr>
    <tr>
      <td>HAN</td>
      <td>25.86</td>
      <td>0.8430</td>
      <td>22.69</td>
      <td>0.7591</td>
    </tr>
    <tr>
      <td>GRL</td>
      <td>25.59</td>
      <td>0.8405</td>
      <td>22.38</td>
      <td>0.7500</td>
    </tr>
    <tr>
      <td><b>FW-SAT (Ours)</b></td>
      <td><b>27.80</b></td>
      <td><b>0.8815</b></td>
      <td><b>24.61</b></td>
      <td><b>0.8116</b></td>
    </tr>
  </tbody>
</table>

</div>



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


## 📬 Contact

If you have any questions, feedback, or collaboration ideas, feel free to reach out:

- 💻 Website: [jianghongcheng.github.io](https://jianghongcheng.github.io/)
- 📧 Email: [hjq44@mail.umkc.edu](mailto:hjq44@mail.umkc.edu)
- 🏫 Affiliation: University of Missouri–Kansas City (UMKC)


