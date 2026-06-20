# 图像分割实验报告：黄色毛绒玩偶

本仓库包含一次图像分割实际案例实验的代码、实验报告和结果文件。实验对象为自行拍摄的商店货架黄色毛绒玩偶图片，目标是通过颜色阈值、形态学处理、连通域分析和距离变换分水岭方法完成玩偶区域分割与实例候选统计。

## 文件结构

- `src/build_segmentation_report.py`：生成早期颜色/分水岭 baseline 报告。
- `src/advanced_watershed_segmentation.py`：使用 OpenCV、scipy、skimage、pandas 进行颜色辅助分水岭实例分割，并导出统计表。
- `src/shape_seed_watershed.py`：改进版结构种子实例分割，避免单纯按颜色分割。
- `src/build_shape_seed_report.py`：生成改进版 Word 实验报告。
- `src/sam_prompt_segmentation.py`：使用 Segment Anything Model（SAM）进行框+点提示实例分割。
- `src/build_sam_report.py`：生成 SAM 改进版实验报告。
- `figures/`：原图和实验过程图，包括颜色量化、掩膜、距离变换、分水岭实例分割结果。
- `report/图像分割实验报告_SAM改进版.docx`：推荐提交的最终 Word 实验报告。
- `report/图像分割实验报告_SAM改进版.pdf`：推荐提交的最终 PDF。
- `report/图像分割实验报告_改进版_结构种子.docx`：结构种子方法对照版 Word 报告。
- `report/图像分割实验报告_改进版_结构种子.pdf`：结构种子方法对照版 PDF。
- `report/图像分割实验报告_黄色毛绒玩偶.docx`：早期颜色分割 baseline 报告。
- `report/图像分割实验报告_黄色毛绒玩偶.pdf`：早期 baseline PDF。
- `report/黄色玩偶分水岭实例分割统计.xlsx`：实例分割统计表。

## 运行环境

图像分割部分使用 conda 环境 `codex-tools`：

```powershell
conda run -n codex-tools python src/advanced_watershed_segmentation.py
```

该环境已验证可用：

- `openpyxl`
- `pandas`
- `numpy`
- `opencv-python`
- `scipy`
- `scikit-image`

Word 报告生成依赖 `python-docx` 和 `Pillow`。如果 `codex-tools` 中未安装 `python-docx`，可在已有 Python 环境中运行：

```powershell
python src/build_segmentation_report.py
```

脚本会优先读取仓库内的 `figures/source_photo.jpg`，因此 clone 仓库后可直接复现实验结果。

## 实验结果摘要

早期颜色阈值方法在黄色玩偶货架上效果较差，因为玩偶、货架和灯光都处于黄橙色范围。最终版使用 `facebook/sam-vit-base` 的 SAM 框+点提示分割，得到 19 个主要实例候选区域，平均预测置信度约 0.872，比颜色阈值和结构种子近似法更适合作为最终提交版本。
