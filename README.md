# 图像分割实验报告：黄色毛绒玩偶

本仓库包含一次图像分割实际案例实验的代码、实验报告和结果文件。实验对象为自行拍摄的商店货架黄色毛绒玩偶图片，目标是通过颜色阈值、形态学处理、连通域分析和距离变换分水岭方法完成玩偶区域分割与实例候选统计。

## 文件结构

- `src/build_segmentation_report.py`：生成基础分割结果图和 Word 实验报告。
- `src/advanced_watershed_segmentation.py`：使用 OpenCV、scipy、skimage、pandas 进行进一步分水岭实例分割，并导出统计表。
- `figures/`：原图和实验过程图，包括颜色量化、掩膜、距离变换、分水岭实例分割结果。
- `report/图像分割实验报告_黄色毛绒玩偶.docx`：最终 Word 实验报告。
- `report/图像分割实验报告_黄色毛绒玩偶.pdf`：由 Word 导出的 PDF 版本。
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

优化后的 OpenCV + skimage 分水岭方法得到 47 个实例候选区域。由于真实货架场景中存在遮挡、同色背景和玩偶相互接触，实例候选数量不等于精确商品数量，但能反映进一步分割对粘连区域的拆分效果。
