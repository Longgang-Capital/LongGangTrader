# Rust 和 Maturin 安装使用指南

本文档详细说明如何在 LongGangTrader 项目中安装 Rust、Maturin，并编译使用高性能回测模块。

## 1. 系统要求

- **操作系统**: Windows 10/11, macOS, 或 Linux
- **Python**: 3.8 或更高版本
- **内存**: 至少 4GB RAM
- **磁盘空间**: 至少 2GB 可用空间

## 2. 安装 Rust

### Windows 系统

1. **下载并安装 Rust**
   - 访问 [Rust 官网](https://www.rust-lang.org/tools/install)
   - 下载并运行 `rustup-init.exe`
   - 按照安装向导完成安装
   - 选择默认安装选项（推荐）

2. **验证安装**
   ```bash
   rustc --version
   cargo --version
   ```

### macOS 和 Linux 系统

1. **使用 rustup 安装**
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **重新加载 shell 配置**
   ```bash
   source $HOME/.cargo/env
   ```

3. **验证安装**
   ```bash
   rustc --version
   cargo --version
   ```

## 3. 安装 Python 依赖

### 创建虚拟环境（推荐）

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 安装项目依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 或者单独安装 maturin
pip install maturin
```

## 4. 编译 Rust 回测模块

### 编译步骤

1. **进入项目根目录**
   ```bash
   cd LongGangTrader
   ```

2. **使用 maturin develop 编译**
   ```bash
   # 开发模式编译（调试版本）
   maturin develop

   # 生产模式编译（优化版本）- 推荐用于回测
   maturin develop --release
   ```

### 编译选项说明

- `maturin develop`: 开发模式，编译速度快，适合调试
- `maturin develop --release`: 生产模式，代码优化，运行性能最佳

## 5. 验证安装

### 测试 Rust 模块

```python
import rust_core
print("Rust 模块导入成功!")

# 测试回测配置
from rust_core import BacktestConfig
config = BacktestConfig(
    initial_capital=1000000,
    transaction_cost_pct=0.001,
    symbol_col="symbol",
    date_col="date",
    close_col="close"
)
print("回测配置创建成功!")
```

### 测试完整回测流程

```python
import polars as pl
import pandas as pd
from longgang_trader.backtesting.backtester import Backtester, BaseStrategy

# 创建测试策略
class TestStrategy(BaseStrategy):
    def generate_signals_for_all_dates(self):
        dates = ['2023-01-01', '2023-01-02', '2023-01-03']
        symbols = ['AAPL', 'GOOGL']
        data = []
        for date in dates:
            for symbol in symbols:
                data.append({
                    'date': date,
                    'symbol': symbol,
                    'target_weight': 0.5
                })
        return pd.DataFrame(data)

# 创建市场数据
dates = ['2023-01-01', '2023-01-02', '2023-01-03']
symbols = ['AAPL', 'GOOGL']
data = []
for date in dates:
    for symbol in symbols:
        data.append({
            'date': date,
            'symbol': symbol,
            'close': 100.0
        })

market_data = pd.DataFrame(data)

# 运行回测
strategy = TestStrategy(None)
config = {
    'initial_capital': 100000,
    'transaction_cost': 0.001,
    'symbol_col': 'symbol',
    'date_col': 'date',
    'close_col': 'close'
}

backtester = Backtester(strategy, market_data, config)
backtester.run_backtest()

if backtester.portfolio_history is not None:
    print('回测成功完成!')
    print(backtester.portfolio_history.head())
else:
    print('回测失败')
```

## 6. 故障排除

### 常见问题

#### 1. "无法导入 'rust_core' 模块"

**原因**: Rust 模块未正确编译
**解决方案**:
```bash
# 确保在项目根目录执行
cd LongGangTrader
maturin develop --release
```

#### 2. "找不到 cargo 命令"

**原因**: Rust 未正确安装或环境变量未设置
**解决方案**:
- 重新安装 Rust
- 重启命令行终端
- 检查环境变量 PATH 是否包含 `~/.cargo/bin`

#### 3. "Python 版本不兼容"

**原因**: Python 版本过低
**解决方案**:
- 升级到 Python 3.8 或更高版本
- 使用 conda 或 pyenv 管理 Python 版本

#### 4. "编译错误: 找不到 polars 库"

**原因**: Rust 依赖版本问题
**解决方案**:
```bash
# 清理并重新编译
cargo clean
maturin develop --release
```

#### 5. "内存不足"

**原因**: 编译过程中内存不足
**解决方案**:
- 关闭其他应用程序释放内存
- 增加系统虚拟内存
- 使用 `cargo build --release` 单独编译

### 调试技巧

1. **检查编译日志**
   ```bash
   maturin develop -v  # 详细输出
   ```

2. **单独测试 Rust 编译**
   ```bash
   cd rust_core
   cargo build --release
   ```

3. **检查 Python 环境**
   ```bash
   python -c "import sys; print(sys.executable)"
   ```

## 7. 性能优化建议

1. **使用 release 模式编译**
   ```bash
   maturin develop --release
   ```

2. **优化数据输入**
   - 使用 Polars LazyFrame 减少内存使用
   - 预处理数据减少传输开销

3. **批量处理**
   - 避免频繁调用 Rust 函数
   - 使用向量化操作

## 8. 开发工作流程

### 修改 Rust 代码后

```bash
# 重新编译
maturin develop --release

# 或者使用增量编译（开发时）
maturin develop
```

### 更新依赖后

```bash
# 更新 Cargo.toml 后
cargo update
maturin develop --release
```

## 9. 技术支持

如果遇到问题：

1. 检查本文档的故障排除部分
2. 查看项目 GitHub Issues
3. 联系开发团队

---

**注意**: 首次编译可能需要较长时间（5-15分钟），因为需要下载和编译 Rust 依赖项。后续编译会快很多。