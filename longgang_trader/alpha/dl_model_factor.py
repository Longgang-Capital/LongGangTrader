# longgang_trader/alpha/dl_model_factor.py

import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from .factor_calculator import BaseFactor
from .gru_attention import AttentionGRURes

class DLModelFactor(BaseFactor):
    def __init__(self, model_path: str, input_dim: int = 158, seq_len: int = 60, 
                 hidden_dim: int = 128, num_layers: int = 1, dropout: float = 0.1,
                 batch_size: int = 2048):
        """
        初始化DL模型因子。
        :param model_path: 预训练模型的路径。
        :param input_dim: 模型输入的特征维度。
        :param seq_len: 模型输入的序列长度。
        :param hidden_dim: GRU隐藏层维度。
        :param num_layers: GRU层数。
        :param dropout: Dropout比例。
        :param batch_size: 推理时的批处理大小。
        """
        super().__init__(name=f"DLFactor_seq{seq_len}")
        self.model_path = model_path
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        
        # 在初始化时就加载模型到内存，避免每次compute都加载
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DLModelFactor 使用的设备: {self.device}")
        
        self.model = AttentionGRURes(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        except FileNotFoundError:
            print(f"错误：模型文件未找到于 {model_path}")
            print("请提供正确的 'model_path' 参数来实例化 DLModelFactor。")
            raise
            
        self.model.eval()

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算DL模型的预测值作为因子。
        :param data: 输入的DataFrame，应包含所有模型需要的原始特征，
                     且索引为 MultiIndex(date, instrument)。
        :return: 返回一个DataFrame，索引为 (date, instrument)，值为模型预测分。
        """
        print("开始计算深度学习因子...")

        # 确保数据按 instrument 和 date 排序
        data = data.sort_index(level=['instrument', 'date'])
        
        # 获取所有特征列
        feature_cols = [col for col in data.columns if col not in ['date', 'instrument']]
        if len(feature_cols) != self.model.gru.input_size:
            raise ValueError(f"输入数据特征数 {len(feature_cols)} 与模型要求 {self.model.gru.input_size} 不符")

        all_sequences = []
        all_indices = []

        # 按股票代码分组，为每只股票创建时间序列
        for instrument_code, group in tqdm(data.groupby(level='instrument'), desc="为每支股票创建序列"):
            if len(group) < self.seq_len:
                continue

            feature_array = group[feature_cols].values

            # 使用 numpy 的 stride_tricks 高效创建滚动窗口
            shape = (len(group) - self.seq_len + 1, self.seq_len, feature_array.shape[1])
            strides = (feature_array.strides[0], feature_array.strides[0], feature_array.strides[1])
            sequences = np.lib.stride_tricks.as_strided(feature_array, shape=shape, strides=strides)
            
            all_sequences.append(sequences)

            # 记录每个序列对应的索引（即窗口的最后一个时间点）
            indices = group.index[self.seq_len - 1:]
            all_indices.extend(indices)

        if not all_sequences:
            print("警告：没有生成任何有效的时间序列。")
            return pd.DataFrame()

        sequences_np = np.vstack(all_sequences)
        sequences_tensor = torch.tensor(sequences_np, dtype=torch.float32)

        # --- 模型推理 ---
        predictions = []
        with torch.inference_mode():
            for i in tqdm(range(0, len(sequences_tensor), self.batch_size), desc="模型推理中"):
                batch = sequences_tensor[i:i+self.batch_size].to(self.device)
                pred_batch = self.model(batch).cpu().numpy()
                predictions.append(pred_batch)

        predictions_np = np.concatenate(predictions).flatten()

        # --- 格式化输出 ---
        result_df = pd.DataFrame(
            predictions_np,
            index=pd.MultiIndex.from_tuples(all_indices, names=['date', 'instrument']),
            columns=['factor_value']
        )

        print(f"深度学习因子计算完成，生成 {len(result_df)} 个因子值。")
        return result_df
