# POGD 临床预测模型 (v5.1) - 可复现性包

本项目遵循 TRIPOD+AI 指南，提供了一个用于预测术后胃功能障碍 (POGD) 的机器学习模型、完整的可复现分析代码以及一个交互式的在线风险计算器。

## 项目结构
. ├── output/ # 存放所有分析结果 (图表, CSV, 模型文件) ├── calculator_app.py # 在线风险计算器的Streamlit应用代码 ├── ml_pogd_final_v5_1.py # 主分析脚本 ├── requirements.txt # Python依赖环境 └── README.md # 本说明文件

## 如何使用

### 1. 环境设置

我们强烈建议使用虚拟环境 (如 venv 或 conda) 来隔离项目依赖。

```bash
# 创建一个新的虚拟环境 (可选但推荐)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# 安装所有必需的库
pip install -r requirements.txt
```

### 2. 运行主分析脚本

此脚本将执行完整的数据预处理、模型训练、评估和可视化流程。最重要的是，它会在 `output/` 文件夹中生成部署在线计算器所需的模型文件 (`ensemble_model_components.pkl`)。

```bash
python "ml_pogd_final_v5_1.py"
```

运行成功后，您将在 `output/` 目录下看到所有的性能指标、SHAP分析图以及保存好的模型文件。

### 3. 启动在线风险计算器

确保您已经成功运行了主分析脚本，因为计算器需要加载生成的模型文件。

在终端中运行以下命令：

```bash
streamlit run calculator_app.py
```

该命令会启动一个本地Web服务器，并在您的浏览器中自动打开一个新标签页，显示POGD风险计算器。您现在可以输入患者指标并实时查看风险预测结果。

### 4. 部署到云端 (可选)

如果您希望生成一个公开的在线链接，您可以轻松地将此应用部署到 [Streamlit Community Cloud](https://streamlit.io/cloud) (提供免费套餐)。

1.  将您的整个项目文件夹上传到 GitHub/Gitee 上的一个新仓库。
2.  登录 Streamlit Community Cloud。
3.  点击 "New app"，连接到您的 GitHub/Gitee 仓库，并选择 `calculator_app.py` 作为主文件。
4.  点击 "Deploy!"。几分钟后，您的在线计算器就会上线，并拥有一个公开的URL。

## TRIPOD+AI 合规性清单

-   [x] **模型的完整预测公式**: 通过提供训练好的模型文件 (`.pkl`) 和加载/预测代码 (`calculator_app.py`) 来实现。
-   [x] **可运行的分析代码**: `ml_pogd_final_v5_1.py` 提供了完整的分析流程。
-   [x] **在线计算器**: `calculator_app.py` 提供了一个交互式的风险计算器实现。
-   [x] **依赖与环境**: `requirements.txt` 确保了环境的可复现性。



