# LongGangTrader 全项目说明文档

本文档详细介绍了 LongGangTrader 量化交易项目的整体架构、数据流和核心模块。

## 1. 数据处理部分

数据是整个量化策略的基石。本项目的数据处理流程涵盖了从原始数据获取、特征计算到数据存储和共享的全过程。

- **数据来源**: 项目的主要原始数据来源于 [baostock](http://baostock.com/)，一个提供免费、开源的A股历史日K线数据的平台。

- **特征构建 (Alpha 158)**: 项目利用 `notebooks/1_LoadAlpha158.ipynb` 脚本计算了经典的 Alpha 158 因子。这些是基于历史量价数据构建的传统技术因子，作为后续深度学习模型的基础输入特征。

- **数据格式与加载**:
    - 特征和标签数据被处理成高效的二进制格式（`.bin`），并附带一个 `.json` 元数据文件。该元数据文件描述了数据的维度（如 `(di, ii, n_feat)` 分别代表日期、股票和特征数量）、数据类型等信息。
    - `longgang_trader/alpha/dataloader.py` 中的 `FactorDataset_di_ii` 类负责高效地读取这些数据。它使用 `numpy.memmap` 进行内存映射，可以处理远超内存大小的大型数据集，并支持按需加载时间序列数据。

- **数据共享 (DVC)**:
    - 项目采用 [DVC (Data Version Control)](https://dvc.org/) 来管理和版本化大型数据文件（如特征和模型文件）。
    - 配置文件（`.dvc` 文件）表明，这些大型文件通过 DVC 连接到对象存储服务（OSS），实现了团队成员之间高效、可复现的数据共享。

## 2. 因子构建部分

本项目的核心是利用深度学习模型，从基础特征中自动学习和构建有效的预测因子。

- **模型架构**: 采用 `GRU (Gated Recurrent Unit)` 结合 `Attention` 机制的深度学习模型。该模型定义在 `longgang_trader/alpha/gru_attention.py` 中，并已预训练为 `notebooks/AttentionGRU.pt`。GRU擅长捕捉时间序列依赖关系，而Attention机制则能帮助模型关注输入序列中最重要的部分。

- **因子生成（模型推理）**:
    - 因子生成过程即为模型推理过程，由 `longgang_trader/alpha/dl_model_factor.py` 中的 `inference_on_test_set` 函数执行。
    - 该函数加载预训练的 `AttentionGRU.pt` 模型，在测试集数据（`data/test_features.bin`）上进行滚动推理，生成每日对每只股票的预测值（即AI因子）。
    - 该过程是全自动化工作流（`run.py`）的核心步骤之一。

- **因子测试**:
    - `longgang_trader/alpha/factor_testing.py` 提供了一个强大的因子分析工具 `FactorTester`。
    - 它可以对生成的AI因子进行全面的有效性检验，包括：
        - **IC (信息系数)** 分析：计算因子的IC和Rank IC序列，评估其预测能力。
        - **分层回测**: 将股票按因子值分组，回测各组的收益表现和多空组合的收益。
        - **因子收益率分析**: 计算因子自身的收益率序列。
        - **可视化**: 绘制IC序列图、分层累计收益图、IC衰减图等，直观展示因子质量。

## 3. 权重优化部分

权重优化模块负责将原始的AI因子信号转化为实际的投资组合权重，同时满足各种风险和约束。

- **优化器架构**:
    - 优化器位于 `longgang_trader/optimize/optimizer.py`，采用面向对象的设计，定义了 `BasePortfolioOptimizer` 抽象基类。
    - 项目内置了多种具体的优化器实现，例如：
        - `EqualWeightOptimizer`：等权重优化器。
        - `MeanVarianceOptimizer`：经典的均值-方差优化器。
        - `RiskParityOptimizer`：风险平价优化器（当前版本为简化实现）。
        - `TopNOptimizer`：选取因子值最高的N只股票进行投资。
    - 该模块大量使用 [Polars](https://pola.rs/) 库进行数据处理，以实现高性能的向量化计算。

- **分层优化策略**:
    - `LayeredOptimizer` 是一个高级优化器，它将因子分组与具体的权重优化相结合。
    - 其流程为：
        1. **筛选**: （可选）筛选出因子值最高的前 n% 的股票。
        2. **分组**: 将股票按因子值高低分为不同的层次（Group）。
        3. **优化**: 在每个分组内部，调用一个具体的优化器（如 `EqualWeightOptimizer`）来计算权重。
    - `run.py` 工作流中正是采用此分层优化策略来生成最终的投资组合。

- **约束管理**:
    - `longgang_trader/optimize/constraints.py` 模块定义了约束管理的框架，如权重约束、换手率约束、行业中性约束等。这为未来扩展更复杂的优化场景提供了基础。

## 4. 回测部分

回测是检验整个策略有效性的最终环节。本项目提供了多种语言实现的回测引擎，以平衡开发效率和运行性能。

- **Python 回测引擎**:
    - 主要的回测逻辑位于 `longgang_trader/backtesting/backtester.py`。
    - `Backtester` 类是回测框架的核心，负责事件循环、资产管理、交易执行和绩效计算。
    - `GroupedFactorStrategy` 等策略类负责根据优化器生成的权重信号，在回测框架中产生具体的交易指令。
    - `run.py` 使用此Python回测引擎进行最终的策略回测和评估。

- **高性能回测引擎 (C++/Rust)**:
    - 为了追求极致的回测速度，项目同时提供了 C++ 和 Rust 实现的高性能回测模块。
    - **C++ 模块**: 位于 `cpp_core/` 目录，通过 `cpp_core/src/bindings.cpp` 文件利用 `pybind11` 等技术封装成Python可调用的接口 (`longgang_trader/backtesting/backtester_cpp.py`)。
    - **Rust 模块**: 位于 `rust_core/` 目录，同样可以编译为Python扩展，提供内存安全和高性能的计算能力。
    - 这些高性能模块可用于替代纯Python回测引擎，在进行大规模、高频率的回测时能大幅缩短等待时间。

## 5. 自动化运行脚本

`run.py` 是整个项目的入口和总指挥，它串联起了从因子生成到回测的全过程，实现了端到端的自动化交易策略流程。

其主要步骤如下：

1.  **环境设置**: 初始化路径，加载项目配置。
2.  **因子生成**: 调用 `inference_on_test_set` 函数，使用预训练的深度学习模型在最新的特征数据上进行推理，生成AI因子预测值。如果已有预测结果，则跳过此步。
3.  **组合优化**:
    - 加载AI因子和市场行情数据。
    - 使用 `LayeredOptimizer` 对因子进行分层优化，计算出每日的股票持仓权重。
    - 将优化后的权重保存到结果文件中 (`results/optimized_weights.parquet`)。
4.  **策略回测**:
    - 初始化 `Backtester` 回测框架和 `GroupedFactorStrategy` 策略。
    - 运行分组回测，即对在优化步骤中划分的每个股票组分别进行独立的回测。
    - 计算每个组的详细绩效指标（年化收益、夏普比率、最大回撤等），并生成可视化图表。
5.  **结果保存**: 将各组的回测历史和绩效指标保存到 `data/results/` 目录下，供后续分析。
