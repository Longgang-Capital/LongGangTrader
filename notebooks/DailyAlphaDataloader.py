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
        data = json.load(p.open('r', encoding='utf8'))
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
    def __init__(self, path_root: str, factor_name: str, split: str, universe: Optional[str] = None, seq_len: int = 1):
        print(f"正在读取{path_root}下的数据")
        self.path_root = Path(path_root)
        self.split = split
        self.seq_len = seq_len
        
        # 读取特征元数据
        feat_meta_path = self.path_root / "features" / f"{split}_features.bin.json"
        self.meta = load_json(feat_meta_path)
        
        # 加载特征数据 (di, ii, n_feat)
        feat_path = self.path_root / "features" / f"{split}_features.bin"
        self.X_3d = load_sarray_data(feat_path, copy_on_write=True)
        
        # 加载标签数据 (di, ii)
        label_path = self.path_root / "labels" / f"{split}_labels.bin"
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


def get_dataloaders(path_root: str,
                    batch_size: int = 64,
                    num_workers: int = 8,
                    universe: Optional[str] = None,
                    seq_len: int = 1):
    """
    生成 train/val/test 的 DataLoader。

    参数：
        path_root (str): 数据根路径
        batch_size (int): 批次大小
        num_workers (int): 数据加载器工作进程数
        universe (str): 股票池筛选，如 "univ_hs300/valid"（可选）
        seq_len (int): 序列长度，每个样本包含过去seq_len天的数据

    返回：
        train_loader, val_loader, test_loader
        
    注意：
        所有DataLoader都返回：(X, y, di_tensor, ii_tensor)
        - X.shape=(batch_size, seq_len, n_feat)
        - y.shape=(batch_size,)  
        - di_tensor.shape=(batch_size,)  # 日期索引（最后一天）
        - ii_tensor.shape=(batch_size,)  # 股票索引
    """
    import torch
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    path_root = Path(path_root)
    
    train_ds = FactorDataset_di_ii(path_root, 'features', 'train', universe, seq_len)
    val_ds   = FactorDataset_di_ii(path_root, 'features', 'val', universe, seq_len)
    test_ds  = FactorDataset_di_ii(path_root, 'features', 'test', universe, seq_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True, persistent_workers=False
    )
    
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True, persistent_workers=False
    )

    return train_loader, val_loader, test_loader


def get_infer_dataloaders(path_root: str,
                    batch_size: int = 64,
                    num_workers: int = 8,
                    universe: Optional[str] = None,
                    seq_len: int = 1):
    """
    生成 train/val/test 的 DataLoader。

    参数：
        path_root (str): 数据根路径
        batch_size (int): 批次大小
        num_workers (int): 数据加载器工作进程数
        universe (str): 股票池筛选，如 "univ_hs300/valid"（可选）
        seq_len (int): 序列长度，每个样本包含过去seq_len天的数据

    返回：
        train_loader, val_loader, test_loader
        
    注意：
        所有DataLoader都返回：(X, y, di_tensor, ii_tensor)
        - X.shape=(batch_size, seq_len, n_feat)
        - y.shape=(batch_size,)  
        - di_tensor.shape=(batch_size,)  # 日期索引（最后一天）
        - ii_tensor.shape=(batch_size,)  # 股票索引
    """
    import torch
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    path_root = Path(path_root)
    
    train_ds = FactorDataset_di_ii(path_root, 'features', 'train', universe, seq_len)
    val_ds   = FactorDataset_di_ii(path_root, 'features', 'val', universe, seq_len)
    test_ds  = FactorDataset_di_ii(path_root, 'features', 'test', universe, seq_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True, persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True, persistent_workers=False
    )
    
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True, persistent_workers=False
    )

    return train_loader, val_loader, test_loader


def test_dataloaders(path_root: str,
                     factor_name: str,
                     selected_dataset: str = "val",
                     batch_size: int = 64,
                     universe: Optional[str] = None,
                     seq_len: int = 1):
    """
    生成测试用的 DataLoader。

    参数：
        path_root (str): 数据根路径
        factor_name (str): 因子名称（保持接口兼容，实际未使用）
        selected_dataset (str): 选择的数据集 ('train', 'val', 或 'test')
        batch_size (int): 批次大小
        universe (str): 股票池筛选（可选）
        seq_len (int): 序列长度

    返回：
        dataloader
    """
    path_root = Path(path_root)
    data_set = FactorDataset_di_ii(path_root, factor_name, selected_dataset, universe, seq_len)
    dataloader = DataLoader(
        data_set, batch_size=batch_size, shuffle=False, 
        num_workers=8, pin_memory=True,
    )
    return dataloader
from torch.utils.data import Dataset, DataLoader
if __name__ == '__main__':
    # 示例：测试dataloader
    path_root = '/local/yjhuang/qlib_data'
    factor_name = 'features'
    batch_size = 32
    universe = None  # 不使用universe筛选
    selected_dataset = "train"
    seq_len = 5  # 使用过去5天的数据
    
    print("="*60)
    print("测试 DataLoader (带序列长度)")
    print("="*60)
    
    my_loader = test_dataloaders(
        path_root, 
        factor_name,
        selected_dataset, 
        batch_size, 
        universe=universe,
        seq_len=seq_len
    )
    
    print(f"数据集: {selected_dataset}")
    print(f"总样本数: {len(my_loader.dataset)}")
    print(f"特征维度: {my_loader.dataset.n_feat}")
    print(f"序列长度: {my_loader.dataset.seq_len}")
    print(f"日期数: {my_loader.dataset.di}")
    print(f"股票数: {my_loader.dataset.ii}")
    print(f"Universe: {universe}")
    print("="*60)
    
    # 测试迭代
    print("测试数据迭代...")
    for batch_idx, (X_batch, y_batch, di_batch, ii_batch) in enumerate(my_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  X_batch.shape: {X_batch.shape}  # 应该是 (batch_size, seq_len, n_feat)")
        print(f"  y_batch.shape: {y_batch.shape}")
        print(f"  di_batch.shape: {di_batch.shape}")
        print(f"  ii_batch.shape: {ii_batch.shape}")
        print(f"  di range: [{di_batch.min()}, {di_batch.max()}]")
        print(f"  ii range: [{ii_batch.min()}, {ii_batch.max()}]")
        print(f"  y range: [{y_batch.min():.4f}, {y_batch.max():.4f}]")
        print(f"  X有NaN: {torch.isnan(X_batch).any().item()}")
        print(f"  y有NaN: {torch.isnan(y_batch).any().item()}")
        
        # 验证序列长度
        assert X_batch.shape[1] == seq_len, f"序列长度不匹配: {X_batch.shape[1]} != {seq_len}"
        print(f"  ✓ 序列长度验证通过")
        
        if batch_idx >= 2:  # 只测试前3个batch
            break
    
    print("\n测试完成！")
