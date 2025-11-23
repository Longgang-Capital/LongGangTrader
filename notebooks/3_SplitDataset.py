from py3lib import taskbase, sarray, config, dlog, multiproc
import sys
import pandas as pd
import pathlib
sys.path.append('/nas02/sap/ver/stable')  # stable是最新的稳定版本，也可以指定具体版本号例如 4.1.0
import sap
task = taskbase.TaskBase()
task.get_options({'sap_cache_dir': '/local/sap/sapcache_1m_2020_rs'}) 

rdr = sap.reader('/local/yjhuang/mysapcache/') 
save_data_path = pathlib.Path('/local/yjhuang/qlib_data')

train_idx = (rdr.date2di(task.options.train_startdate, round_to_smaller=False), rdr.date2di(task.options.train_enddate, round_to_smaller=True))
val_idx = (rdr.date2di(task.options.val_startdate, round_to_smaller=False), rdr.date2di(task.options.val_enddate, round_to_smaller=True))
test_idx = (rdr.date2di(task.options.test_startdate, round_to_smaller=False), rdr.date2di(task.options.test_enddate, round_to_smaller=True))
task.log.info(f"train_idx is :{train_idx}, val_idx is :{val_idx}, test_idx is :{test_idx}")
# process_features
full_features = sarray.load('/nas02/home/yjhuang/qlib/对接本地数据源/alpha158_14tonow_3d.bin').data

train_features = full_features[train_idx[0]:train_idx[1]+1]
val_features = full_features[val_idx[0]:val_idx[1]+1]
test_features = full_features[test_idx[0]:test_idx[1]+1]

# process_labels
full_labels = sarray.load('/local/yjhuang/mysapcache/labels/v2v_ret_5d.bin').data

train_labels = full_labels[train_idx[0]:train_idx[1]+1]
val_labels = full_labels[val_idx[0]:val_idx[1]+1]
test_labels = full_labels[test_idx[0]:test_idx[1]+1]

# save

sarray.save_ndarray(train_features, save_data_path / 'features' / 'train_features.bin',extra_json={'start_date':task.options.train_startdate, 'end_date':task.options.train_enddate})
sarray.save_ndarray(val_features, save_data_path / 'features' / 'val_features.bin',extra_json={'start_date':task.options.val_startdate, 'end_date':task.options.val_enddate})
sarray.save_ndarray(test_features, save_data_path / 'features' / 'test_features.bin',extra_json={'start_date':task.options.test_startdate, 'end_date':task.options.test_enddate})

sarray.save_ndarray(train_labels, save_data_path / 'labels' / 'train_labels.bin',extra_json={'start_date':task.options.train_startdate, 'end_date':task.options.train_enddate})
sarray.save_ndarray(val_labels, save_data_path / 'labels' / 'val_labels.bin',extra_json={'start_date':task.options.val_startdate, 'end_date':task.options.val_enddate})
sarray.save_ndarray(test_labels, save_data_path / 'labels' / 'test_labels.bin',extra_json={'start_date':task.options.test_startdate, 'end_date':task.options.test_enddate})
