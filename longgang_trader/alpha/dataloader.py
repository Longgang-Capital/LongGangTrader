import json
import numpy as np
import torch
from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset, DataLoader

import json
import os
from pathlib import Path
from typing import Union, List
import numpy as np

PathLike = Union[str, Path]


def _conv_dtype(_dtype):
    if isinstance(_dtype, str):
        return _dtype
    elif isinstance(_dtype, list):
        if len(_dtype) == 2:
            if isinstance(_dtype[1], int):
                return (_conv_dtype(_dtype[0]), _dtype[1])
            elif isinstance(_dtype[1], list) and _dtype[1] and isinstance(_dtype[1][0], int):
                return (_conv_dtype(_dtype[0]), tuple(_dtype[1]))
        ret = []
        for field in _dtype:
            if len(field) == 3:
                ret.append(( 
                    field[0],
                    _conv_dtype(field[1]),
                    field[2] if isinstance(field[2], int) else tuple(field[2])
                ))
            elif len(field) == 2:
                ret.append((field[0], _conv_dtype(field[1])))
            else:
                raise Exception(f"failed to parse dtype: {_dtype}")
        return ret
    else:
        raise Exception(f"failed to parse dtype: {_dtype}")


class _JsonLoader:

    def __init__(self) -> None:
        self._loaded = set()

    def load(self, jsonfile: PathLike) -> dict:
        p = Path(jsonfile)
        with p.open('r', encoding='utf8') as f:
            data = json.load(f)
        self._loaded.add(str(p.resolve()))

        if '!include' in data:
            inc = data.pop('!include')
            inc_path = Path(inc) if os.path.isabs(inc) else p.parent / inc
            inc_abs = str(inc_path.resolve())

            if inc_abs not in self._loaded:
                self._loaded.add(inc_abs)
                parent_data = self.load(inc_path)
                parent_data.update(data)
                data = parent_data

        return data


def load_json(filename: PathLike) -> dict:
    loader = _JsonLoader()
    return loader.load(filename)


def load_sarray_data(datafile: PathLike,
                     copy_on_write: bool = True,
                     offset: int = 0) -> np.ndarray:
    datafile = Path(datafile)
    jsonfile = Path(str(datafile) + '.json')

    if not jsonfile.exists():
        raise FileNotFoundError(f"json file not found: {jsonfile}")

    attr = load_json(jsonfile)

    raw_dtype = attr.get('dtype')
    if raw_dtype is None:
        raise ValueError(f"`dtype` not found in {jsonfile}")
    dtype_align = attr.get('dtype_align', True)
    dtype = np.dtype(_conv_dtype(raw_dtype), align=dtype_align)

    shape: List[int] = attr.get('shape', [])
    real_offset = offset or attr.get('offset', 0)

    if not shape:
        fsize = datafile.stat().st_size - real_offset
        if fsize % dtype.itemsize != 0:
            raise ValueError(f"file size {fsize} not divisible by itemsize {dtype.itemsize}")
        shape = [fsize // dtype.itemsize]

    mode = 'c' if copy_on_write else 'r'
    arr = np.memmap(datafile, dtype=dtype, mode=mode,
                    shape=tuple(shape), offset=real_offset)
    return arr



class FactorDataset_di_ii(Dataset):
    """
    PyTorch 数据集，用于基于sarray读取单因子特征和标签（日频数据，支持时间序列）。
    """
    def __init__(self, feature_path: Union[str, Path], label_path: Union[str, Path,None] = None, universe: Optional[str] = None, seq_len: int = 1):
        print(f"正在读取特征: {feature_path}")
        print(f"正在读取标签: {label_path}")
        self.seq_len = seq_len

        # 加载特征元数据
        self.meta = load_json(str(Path(feature_path)) + '.json')
        
        # 加载特征数据 (di, ii, n_feat)
        self.X_3d = load_sarray_data(feature_path, copy_on_write=True)
        
        if label_path is None:
            self.y_2d = np.zeros((self.X_3d.shape[0], self.X_3d.shape[1]), dtype=np.float32)
        else:
            # 加载标签数据 (di, ii)
            self.y_2d = load_sarray_data(label_path, copy_on_write=True)
        
        # 获取维度信息
        self.di, self.ii, self.n_feat = self.X_3d.shape
        print(f"原始数据维度: di={self.di}, ii={self.ii}, n_feat={self.n_feat}, seq_len={self.seq_len}")
        
        # 保存原始股票索引
        self.indices = None
        self.orig_ii = self.ii
        
        # 如果需要universe筛选
        if universe == "None":
            universe = None
        if universe is not None:
             raise NotImplementedError("当前版本暂不支持 universe 筛选功能，请检查调用参数。")
            
        
        # 获取有效样本索引（考虑seq_len约束）
        self.valid_di_ii_pairs = self._get_valid_di_ii_pairs()
        
        # 预计算有效样本的di和ii索引
        self.di_arr = self.valid_di_ii_pairs[:, 0].astype(np.int32)
        self.ii_arr = self.valid_di_ii_pairs[:, 1].astype(np.int32)
        
        # 如果有universe筛选，orig_ii_arr需要映射回原始索引
        if self.indices is not None:
            self.orig_ii_arr = self.indices[self.ii_arr]
        else:
            self.orig_ii_arr = self.ii_arr
        
        self.length = len(self.valid_di_ii_pairs)
        print(f"有效样本数量: {self.length}")

    def _get_valid_di_ii_pairs(self):

        D, I = self.di, self.ii
        valid_pairs = []
        
        # 遍历所有可能的(di, ii)对
        for di in range(self.seq_len - 1, D):  # di必须 >= seq_len-1
            for ii in range(I):
                # 获取序列：从 (di-seq_len+1) 到 di，共seq_len天
                seq_features = self.X_3d[di-self.seq_len+1:di+1, ii, :]  # (seq_len, n_feat)
                seq_label = self.y_2d[di, ii]  # 标签是最后一天的
                
                # 检查序列特征是否有效
                # 1. 不是所有样本都全为0
                not_all_zero = ~(seq_features == 0).all()
                # 2. 所有特征都有限
                all_finite = np.isfinite(seq_features).all()
                # 3. 标签有限
                label_finite = np.isfinite(seq_label)
                
                if not_all_zero and all_finite and label_finite:
                    valid_pairs.append([di, ii])
        
        pairs = np.array(valid_pairs, dtype=np.int32)
        
        if len(pairs) == 0:
            print("警告: 没有找到有效样本!")
            return np.array([]).reshape(0, 2)
        
        print(f"有效的(di, ii)对数量: {len(pairs)} / {(D-self.seq_len+1)*I} ({100*len(pairs)/((D-self.seq_len+1)*I):.2f}%)")
        
        return pairs
    


    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):

        if idx >= self.length:
            raise IndexError(f"索引 {idx} 超出范围，总样本数为 {self.length}")
        
        # 获取当前样本的(di, ii)
        di = self.di_arr[idx]
        ii = self.ii_arr[idx]
        
        # 获取序列特征: 从 (di-seq_len+1) 到 di，共seq_len天
        seq_features = self.X_3d[di-self.seq_len+1:di+1, ii, :]  # (seq_len, n_feat)
        
        # 读取标签（最后一天）
        y_val = self.y_2d[di, ii]
        
        # 转换为tensor
        x = torch.from_numpy(np.ascontiguousarray(seq_features).astype(np.float32, copy=False))
        y = torch.as_tensor(y_val, dtype=torch.float32)
        
        return x, y, di, self.orig_ii_arr[idx]


def get_test_dataloader(feature_path: Union[str, Path],
                        label_path: Union[str, Path, None]=None,
                        batch_size: int = 64,
                        num_workers: int = 8,
                        universe: Optional[str] = None,
                        seq_len: int = 1):
    """
    生成 test集的 DataLoader。

    参数：
        feature_path (str or Path): 特征文件 (.bin) 的直接路径.
        label_path (str or Path): 标签文件 (.bin) 的直接路径.
        batch_size (int): 批次大小
        num_workers (int): 数据加载器工作进程数
        universe (str): 股票池筛选 (可选)
        seq_len (int): 序列长度

    返回：
        test_loader
    """
    import torch
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    # 创建 test dataset
    try:
        test_ds = FactorDataset_di_ii(
            feature_path=feature_path,
            label_path=label_path,
            universe=universe, 
            seq_len=seq_len
        )
    except FileNotFoundError as e:
        print(f"警告: 找不到数据文件: {e}。返回 None。")
        return None

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True, persistent_workers=False
    )

    return test_loader
