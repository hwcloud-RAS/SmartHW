# -*- coding: utf-8 -*-
import builtins
import collections
import functools
import gc
import io
import multiprocessing
import os
import pickle
import stat
from collections import namedtuple
from typing import List, Dict, Union, Tuple, Optional, IO

import attr
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from tqdm import tqdm

DIMM_LOCATION = namedtuple("DIMM_LOCATION", ["node", "channel", "slot"])
DIMM_PHY_ADDR = namedtuple("DIMM_PHY_ADDR", ["rank", "dev", "bankgroup", "bank", "row", "column"])
MEM_ERR_EVENT = namedtuple("MEM_ERR_EVENT",
                           ["timestamp", "err_type", "err_count", "dimm_location", "phy_addr", "paras", "bitmask"])
InputItem = namedtuple("InputItem", ["dimm_key", "timestamp", "mesg_type", "mesg"])
OutputItem = namedtuple("OutputItem", ["dimm_key", "timestamp", "result", "probability", "detail"])
BitMaskFeature = namedtuple("BitMaskFeature", ['timestamp', 'err_bits_per_parity', 'min_err_dq_dist', 'adj_err_dq_cnt',
                                               'max_err_bits_per_burst', 'err_bursts_per_parity',
                                               'adj_err_bit_cnt_per_parity', 'max_err_burst_dist_per_parity',
                                               'min_err_burst_dist_per_parity', 'dq_cnt'])
ListPointers = namedtuple('ListPointers', ['start', 'end'])

# 单个Device共有4种dq报错
DEVICE_DQ = 4
# 每个parity对应单个Device上的8个Burst
DEVICE_BURST = 8
# 最大bitmask特征列表长度
MAX_BIT_MASK_FEATURE_LIST_LEN = 100
# bitmask特征列表长度
BIT_MASK_FEATURE_WINDOW_SIZE = 120 * 3600
# 滑动时间窗口大小
WINDOW_SIZE_120H = 120 * 60 * 60
# 训练集和验证集分界线: 2024年5月1日零点
TRAIN_TEST_SPLIT_TIMESTAMP = 1714492800
# 预测为正样本的阈值
threshold = 0.88

CNT_FEATURE_READ_WINDOW_SIZE_6M = 6 * 60
CNT_FEATURE_READ_WINDOW_SIZE_3H = 3 * 60 * 60
CNT_FEATURE_READ_WINDOW_SIZE_6H = 6 * 60 * 60
CNT_FEATURE_READ_WINDOW_SIZE_120H = 120 * 60 * 60

RPT_CNT_READ_FEATURE_NAME_6M = 'RptCnt(CE.READ,6m)'
RPT_CNT_READ_FEATURE_NAME_3H = 'RptCnt(CE.READ,3h)'
RPT_CNT_READ_FEATURE_NAME_6H = 'RptCnt(CE.READ,6h)'
RPT_CNT_READ_FEATURE_NAME_120H = 'RptCnt(CE.READ,120h)'

ERR_CNT_READ_FEATURE_NAME_6M = 'ErrCnt(CE.READ,6m)'
ERR_CNT_READ_FEATURE_NAME_3H = 'ErrCnt(CE.READ,3h)'
ERR_CNT_READ_FEATURE_NAME_6H = 'ErrCnt(CE.READ,6h)'
ERR_CNT_READ_FEATURE_NAME_120H = 'ErrCnt(CE.READ,120h)'

CE_STORM_INTERVAL = 60
CE_STORM_THRESHOLD = 10
CE_STORM_CNT_FEATURE_WINDOW_SIZE = WINDOW_SIZE_120H
CE_STORM_CNT_CURRENT_FEATURE_NAME = 'CE_storm_Cnt(All,current)'
CE_STORM_CNT_ALL_FEATURE_NAME = 'CE_storm_Cnt(All,120h)'

# 第4位表示出现相邻DQ出错的组合数量
CEBitMaskStatsMap = [
    (0, 4, 0, 0),  # 0: 0000
    (1, 4, 0, 0),  # 1: 0001
    (1, 4, 0, 0),  # 2: 0010
    (2, 1, 1, 1),  # 3: 0011
    (1, 4, 0, 0),  # 4: 0100
    (2, 2, 2, 0),  # 5: 0101
    (2, 1, 1, 1),  # 6: 0110
    (3, 1, 2, 2),  # 7: 0111
    (1, 4, 0, 0),  # 8: 1000
    (2, 3, 3, 0),  # 9: 1001
    (2, 2, 2, 0),  # A: 1010
    (3, 1, 3, 1),  # B: 1011
    (2, 1, 1, 1),  # C: 1100
    (3, 1, 3, 1),  # D: 1101
    (3, 1, 2, 2),  # E: 1110
    (4, 1, 3, 3),  # F: 1111
]


class RestrictedUnpickler(pickle.Unpickler):
    SAFE_BUILTINS = {'range', 'complex', 'set', 'frozenset', 'slice', 'bytearray'}
    SAFE_COLLECTIONS = {'OrderedDict', 'defaultdict'}
    SAFE_NUMPY = {'_reconstruct', 'ndarray', 'dtype', 'scalar'}
    SAFE_PANDAS = {'DataFrame', 'BlockManager', 'new_block', '_unpickle_block', '_new_Index', 'Index', 'RangeIndex'}
    SAFE_MODEL = {'LGBMClassifier', 'Booster', 'LabelEncoder'}
    SAFE_OTHERS = {'partial'}

    def find_class(self, module: str, name: str) -> attr:
        """
        反序列化类型检查，只有在白名单中的类型才可正常使用
        :param module:包的类型
        :param name:具体包名
        :return:如果反序列化后的类型在白名单中则返回对应类，否则抛出异常
        """
        builtin_info = {"builtins": builtins}
        collection_info = {"collections": collections}
        numpy_info = {"numpy.core.multiarray": np.core.multiarray,
                      "numpy": np}
        pandas_info = {"pandas.core.frame": pd.core.frame,
                       "pandas.core.internals.managers": pd.core.internals.managers,
                       "pandas.core.internals.blocks": pd.core.internals.blocks,
                       "pandas._libs.internals": pd._libs.internals,
                       "pandas.core.indexes.base": pd.core.indexes.base,
                       "pandas.core.indexes.range": pd.core.indexes.range}
        model_info = {"lightgbm.sklearn": lightgbm.sklearn,
                      "lightgbm.basic": lightgbm.basic,
                      "sklearn.preprocessing._label": sklearn.preprocessing._label,
                      "sklearn.preprocessing._data": sklearn.preprocessing._data}
        others_info = {"functools": functools}

        for class_info, check_set in [(builtin_info, RestrictedUnpickler.SAFE_BUILTINS),
                                      (collection_info, RestrictedUnpickler.SAFE_COLLECTIONS),
                                      (numpy_info, RestrictedUnpickler.SAFE_NUMPY),
                                      (pandas_info, RestrictedUnpickler.SAFE_PANDAS),
                                      (model_info, RestrictedUnpickler.SAFE_MODEL),
                                      (others_info, RestrictedUnpickler.SAFE_OTHERS)]:
            if module in class_info and name in check_set:
                return getattr(class_info.get(module), name)

        raise pickle.UnpicklingError(f"global '{module}-{name}' is forbidden")

    @staticmethod
    def open_file_to_write(
            file_path: str,
            is_binary: bool = False,
            write_flags: int = os.O_CREAT | os.O_TRUNC,
            open_modes: int = stat.S_IRUSR | stat.S_IWUSR,
            encoding: Optional[str] = None) -> IO:
        """
        创建/覆盖文件，并以只写方式打开(用于写json、pickle等文件时, 替换open函数)
        :param file_path: 待打开文件路径
        :param is_binary: 是否以打开二进制方式打开
        :param write_flags: 打开文件附加模式，可选 os.O_CREAT, os.O_TRUNC, os.O_APPEND 中的一个或多个组合（按位或）
        :param open_modes: 文件权限设置
        :param encoding: 文本文件编码，默认值为 None
        :return: 打开后的文件 IO
        """
        write_mask = os.O_CREAT | os.O_TRUNC | os.O_APPEND
        open_flags = os.O_WRONLY | (write_flags & write_mask)
        mode = 'wb' if is_binary else 'w'
        return os.fdopen(os.open(file_path, open_flags, open_modes), mode, encoding=encoding)

    @staticmethod
    def write_pkl_data(saved_data, data_path: str, **kwargs) -> None:
        """
        save pkl文件
        :param saved_data: 要保存的数据
        :param data_path: 数据保存路径
        """
        with RestrictedUnpickler.open_file_to_write(data_path, **kwargs) as f:
            pickle.dump(saved_data, f)

    @staticmethod
    def read_pkl_file(file_path: str):
        """
        读取pkl文件
        :param file_path: 数据读取路径
        :return: 加载的数据
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The pkl file not found in path: {file_path}")
        with open(file_path, 'rb') as f:
            data = f.read()
        return RestrictedUnpickler(io.BytesIO(data)).load()


class InputAdapter:
    @staticmethod
    def __get_dimm_loc_info(mesg: Dict) -> Tuple[str, DIMM_LOCATION]:
        """
        获取内存条的序列号及位置信息，如sn, CpuId, DimmId等
        :param mesg: 输入特征信息，格式为dict
        :return: 内存sn, DIMM_LOCATION 信息
        """
        dimm_key = mesg["SN"]
        dimm_loc = DIMM_LOCATION(mesg["CpuId"], mesg["ChannelId"], mesg["DimmId"])
        return dimm_key, dimm_loc

    @staticmethod
    def __get_paras(mesg: Dict) -> Dict:
        """
        重新解析输入特征信息，并解析出需要的格式
        :param mesg: 输入特征信息，格式为dict
        :return: 提取需要信息并解析出特定格式，格式为{
                "host_ip":xx,
                "dimm_type": xx,
                "manufacturer": xx,
                "PN": xx,
            }
        """
        return {
            "host_ip": mesg.get("IP", "UNKNOWN_IP"),
            "dimm_type": mesg.get("memory_type", "UNKNOWN_DIMM_TYPE"),
            "manufacturer": mesg.get("Manufacturer", "UNKNOWN_MANUFACTURER"),
            "PN": mesg.get("PN", "UNKNOWN_IP"),
        }

    @staticmethod
    def __process_input_data(input_data: Optional[Union[List, Dict]], dimm_sn: str) -> \
            Tuple[Optional[str], Optional[DIMM_LOCATION], Optional[Dict], Optional[List]]:
        """
        对输入特定的故障特征数据进行解析，并提取需要的格式数据
        :param input_data:输入特征数据
        :param dimm_sn: 内存序列号
        :return: 解析后的新格式数据，(内存序列号, 内存位置信息, 内存属性信息, 故障日志信息)
        """
        dimm_key = None
        dimm_loc = None
        ce_log = []
        paras = None
        if isinstance(input_data, list):
            for seg in input_data:
                seg['SN'] = dimm_sn
                ce_log.append(seg)
                if dimm_key is None or dimm_loc is None:
                    dimm_key, dimm_loc = InputAdapter.__get_dimm_loc_info(seg)
                if paras is None:
                    paras = InputAdapter.__get_paras(seg)
        return dimm_key, dimm_loc, paras, ce_log

    @staticmethod
    def __make_item(log_item: Dict, dimm_key: str, dimm_loc: DIMM_LOCATION, paras: Dict) -> InputItem:
        """
        对输入的当前内存日志及属性信息进行格式化，转为InputItem格式供后续推理及训练使用
        :param log_item: 输入特征信息，格式为dict
        :param dimm_key: 内存序列号
        :param dimm_loc: 内存位置信息
        :param paras: 内存静态属性信息
        :return: 解析后特定的数据格式，格式为InputItem类别
        """
        parity_value = log_item.get("RetryRdErrLogParity")
        _bitmask = 0 if parity_value is None or np.isnan(float(parity_value)) else int(parity_value)
        _phy_addr = DIMM_PHY_ADDR(
            rank=int(log_item.get("RankId")) if not np.isnan(log_item.get("RankId")) else -1,
            dev=int(log_item.get("deviceID")) if not np.isnan(log_item.get("deviceID")) else -1,
            bankgroup=int(log_item.get("BankgroupId")) if not np.isnan(log_item.get("BankgroupId")) else -1,
            bank=int(log_item.get("BankId")) if not np.isnan(log_item.get("BankId")) else -1,
            row=int(log_item.get("RowId")) if not np.isnan(log_item.get("RowId")) else -1,
            column=int(log_item.get("ColumnId")) if not np.isnan(log_item.get("ColumnId")) else -1,
        )
        err_count = 0 if log_item.get("Count") is None else log_item.get("Count", 0)
        input_item = InputItem(dimm_key, log_item["LogTime"], "MEM_ERR_EVENT",
                               mesg=MEM_ERR_EVENT(
                                   timestamp=log_item.get("LogTime", 0),
                                   err_count=err_count,
                                   err_type=log_item.get("error_type_full_name", "CE"),
                                   dimm_location=dimm_loc,
                                   phy_addr=_phy_addr,
                                   bitmask=_bitmask,
                                   paras=paras,
                               ))
        return input_item

    @staticmethod
    def convert(input_data: Union[List, Dict], dimm_sn: str, sample_count: int = 0) -> List[InputItem]:
        """
        输入特征转换
        :param input_data: 输入特征数据
        :param dimm_sn: 内存序列号信息
        :param sample_count: 采样数量
        :return: 转换后内存特征信息
        """
        std_input = []
        dimm_key, dimm_loc, paras, ce_log = InputAdapter.__process_input_data(input_data, dimm_sn)
        ce_log.sort(key=lambda x: x["LogTime"])
        ce_log = ce_log[-sample_count:]

        for log_item in ce_log:
            input_item = InputAdapter.__make_item(log_item, dimm_key, dimm_loc, paras)
            if input_item is not None:
                std_input.append(input_item)
        return std_input


class FeatureExtractor:
    def __init__(self):
        self.history_event_list = []
        self.addr_features = {'RankCnt': set(), 'BankCnt': set(), 'DeviceCnt': set()}
        self.addr_ce_bit_mask_stat = self.__default_stat_of_bit_mask_feature
        self.bitmask_feature_pointer = ListPointers(start=0, end=0)
        self.bitmask_feature_list = []

        self.current_features = {'ce_bit_mask': self.__default_feature_of_bit_mask(),
                                 'ce_addr': self.__default_feature_of_addr,
                                 'ce_err_cnt': self.__default_feature_of_err_cnt,
                                 'ce_rpt_cnt': self.__default_feature_of_rpt_cnt,
                                 'ce_storm_cnt': self.__default_feature_of_ce_storm_cnt_feature}

    @staticmethod
    def __default_feature_of_bit_mask(stat_features: bool = True,
                                      compare_features: bool = True) -> Dict:
        """
        根据bit信息获取的默认特征
        :param stat_features: 是否使用stat_feature
        :param compare_features: 是否使用compare_features
        :return: 默认bit_mask相关特征
        """
        ret = {}
        if stat_features:
            ret.update({
                'AvgMinErrDqDistPerParity': 0.0,
                'AvgErrBitsPerErrBurst': 0.0,
                'AvgAdjErrDqCntPerParity': 0.0,
                'AvgErrBurstsPerParity': 0.0,
                'AvgMaxErrBurstsDistPerParity': 0.0,
                'AvgMinErrBurstsDistPertParity': 0.0,
                "DqCnt=1": 0,
                "DqCnt=2": 0,
                "DqCnt=3": 0,
                "DqCnt=4": 0,
                "BurstCnt=1": 0,
                "BurstCnt=2": 0,
                "BurstCnt=3": 0,
                "BurstCnt=4": 0,
                "BurstCnt=5": 0,
                "BurstCnt=6": 0,
                "BurstCnt=7": 0,
                "BurstCnt=8": 0,
            })
        if compare_features:
            ret.update({
                "MaxErrBitsPerBurst": 0,
                "MinErrDqDistance": 4,
                "MinErrBurstDist": 8,
                "MaxErrBurstDist": 0,
                "MaxErrBurstCnt": 0,
                "MaxAdjErrBitCnt": 0,
                "MaxAdjErrDqCnt": 0,
            })
        return ret

    @property
    def __default_stat_of_bit_mask_feature(self) -> Dict:
        """
        默认bit之间统计特征
        """
        ret = {
            "bitmask_count": 0,
            "total_err_burst_count": 0,
            "total_err_bits_count": 0,
            "total_min_bit_dist": 0,
            "total_min_burst_dist": 0,
            "total_max_burst_dist": 0,
            "total_adj_dq_count": 0,
            "DqCnt=1": 0,
            "DqCnt=2": 0,
            "DqCnt=3": 0,
            "DqCnt=4": 0,
            "BurstCnt=1": 0,
            "BurstCnt=2": 0,
            "BurstCnt=3": 0,
            "BurstCnt=4": 0,
            "BurstCnt=5": 0,
            "BurstCnt=6": 0,
            "BurstCnt=7": 0,
            "BurstCnt=8": 0,
        }
        return ret

    @property
    def __default_single_bitmask_feature(self) -> Dict:
        """
        默认单bit相关特征
        """
        ret = {
            'timestamp': 0,
            "err_bits_per_parity": 0,
            "err_bursts_per_parity": 0,
            "max_err_bits_per_burst": 0,
            "min_err_dq_dist": 4,
            "min_err_burst_dist_per_parity": 8,
            "max_err_burst_dist_per_parity": 0,
            "adj_err_bit_cnt_per_parity": 0,
            "adj_err_dq_cnt": 0,
            "dq_cnt": 0,
        }
        return ret

    @property
    def __default_feature_of_addr(self) -> Dict:
        """
        默认addr相关特征
        """
        ret = {"AddrCnt(Rank)": 0,
               "AddrCnt(Bank)": 0,
               "AddrCnt(Device)": 0}
        return ret

    @property
    def __default_feature_of_err_cnt(self) -> Dict:
        """
        故障类型统计特征
        """
        ret = {
            ERR_CNT_READ_FEATURE_NAME_6M: 1,
            ERR_CNT_READ_FEATURE_NAME_3H: 1,
            ERR_CNT_READ_FEATURE_NAME_6H: 1,
            ERR_CNT_READ_FEATURE_NAME_120H: 1,
        }
        return ret

    @property
    def __default_feature_of_rpt_cnt(self) -> Dict:
        """
        默认重复统计特征
        """
        ret = {
            RPT_CNT_READ_FEATURE_NAME_120H: 1,
            RPT_CNT_READ_FEATURE_NAME_6H: 1,
            RPT_CNT_READ_FEATURE_NAME_3H: 1,
            RPT_CNT_READ_FEATURE_NAME_6M: 1,
        }
        return ret

    @property
    def __default_feature_of_ce_storm_cnt_feature(self) -> Dict:
        """
        默认ce风暴统计特征
        """
        ret = {
            CE_STORM_CNT_CURRENT_FEATURE_NAME: 0,
            CE_STORM_CNT_ALL_FEATURE_NAME: 0,
        }
        return ret

    @staticmethod
    def __prepare_bit_mask(bit_mask: int) -> Union[str, None]:
        """
        获取bit mask信息
        :param bit_mask: bit mask 信息
        :return: 返回16进制筛选后的bit_mask信息，如'0x4000000'
        """
        if isinstance(bit_mask, int):
            bit_mask = hex(bit_mask)
            return bit_mask[2:]
        elif isinstance(bit_mask, str) and (len(bit_mask) != 10 or not bit_mask.lower().startswith("0x")):
            return bit_mask[2:]
        return None

    @staticmethod
    def __update_inc_dec(stats: dict, operate: str, bm_feature: BitMaskFeature) -> None:
        """
        增加或减少特征统计量函数
        """
        if operate == '+':
            stats["bitmask_count"] += 1
            stats["total_err_bits_count"] += bm_feature.err_bits_per_parity
            stats["total_err_burst_count"] += bm_feature.err_bursts_per_parity
            stats["total_min_bit_dist"] += bm_feature.min_err_dq_dist
            stats["total_adj_dq_count"] += bm_feature.adj_err_dq_cnt
            stats["total_max_burst_dist"] += bm_feature.max_err_burst_dist_per_parity
            stats["total_min_burst_dist"] += bm_feature.min_err_burst_dist_per_parity
            if bm_feature.dq_cnt > 0:
                stats[f"DqCnt={bm_feature.dq_cnt}"] += 1
            if bm_feature.err_bursts_per_parity > 0:
                stats[f'BurstCnt={bm_feature.err_bursts_per_parity}'] += 1
        else:
            stats["bitmask_count"] -= 1
            stats["total_err_bits_count"] -= bm_feature.err_bits_per_parity
            stats["total_err_burst_count"] -= bm_feature.err_bursts_per_parity
            stats["total_min_bit_dist"] -= bm_feature.min_err_dq_dist
            stats["total_adj_dq_count"] -= bm_feature.adj_err_dq_cnt
            stats["total_max_burst_dist"] -= bm_feature.max_err_burst_dist_per_parity
            stats["total_min_burst_dist"] -= bm_feature.min_err_burst_dist_per_parity
            if bm_feature.dq_cnt > 0:
                stats[f"DqCnt={bm_feature.dq_cnt}"] -= 1
            if bm_feature.err_bursts_per_parity > 0:
                stats[f"BurstCnt={bm_feature.err_bursts_per_parity}"] -= 1

    @staticmethod
    def __update_compare(stats: dict, bm_feature: BitMaskFeature) -> None:
        """
        根据当前bit mask特征进行比较并获取相关统计特征
        :param stats: stats特征
        :param bm_feature: bit_mask 特征
        """
        if bm_feature.max_err_bits_per_burst > stats["MaxErrBitsPerBurst"]:
            stats["MaxErrBitsPerBurst"] = bm_feature.max_err_bits_per_burst
        if bm_feature.err_bursts_per_parity > stats["MaxErrBurstCnt"]:
            stats["MaxErrBurstCnt"] = bm_feature.err_bursts_per_parity
        if bm_feature.min_err_dq_dist < stats["MinErrDqDistance"]:
            stats["MinErrDqDistance"] = bm_feature.min_err_dq_dist
        if bm_feature.adj_err_dq_cnt > stats["MaxAdjErrDqCnt"]:
            stats["MaxAdjErrDqCnt"] = bm_feature.adj_err_dq_cnt
        if bm_feature.adj_err_bit_cnt_per_parity > stats["MaxAdjErrBitCnt"]:
            stats["MaxAdjErrBitCnt"] = bm_feature.adj_err_bit_cnt_per_parity
        if bm_feature.max_err_burst_dist_per_parity > stats["MaxErrBurstDist"]:
            stats["MaxErrBurstDist"] = bm_feature.max_err_burst_dist_per_parity
        if bm_feature.min_err_burst_dist_per_parity < stats["MinErrBurstDist"]:
            stats["MinErrBurstDist"] = bm_feature.min_err_burst_dist_per_parity

    @staticmethod
    def __filter_events(event_list: List[MEM_ERR_EVENT], current_event: MEM_ERR_EVENT, window_size: int = 3600) -> List[
        MEM_ERR_EVENT]:
        """
        对事件进行过滤，只取窗口内的事件
        :param event_list: 事件列表
        :param current_event: 当前事件
        :param window_size: 事件过滤时间窗大小
        :return: 过滤后的事件列表
        """
        cur_timestamp = current_event.timestamp
        ts_start = cur_timestamp - window_size
        _filter_events = []
        for _event in event_list[::-1]:
            if _event.timestamp < ts_start:
                break
            _filter_events.append(_event)
        return _filter_events[::-1]

    @staticmethod
    def get_dimm_feature_dict(current_features: Dict) -> Dict:
        """
        获取当前内存的所有统计特征
        """
        feature_dict = {}
        for _group, _feature_values in current_features.items():
            suffix = None
            if _group == 'ce_bit_mask':
                suffix = '(120h,100)'
            if suffix is None:
                feature_dict.update(_feature_values)
            else:
                for feature_name, value in _feature_values.items():
                    feature_dict[feature_name + suffix] = value
        return feature_dict

    def __calc_current_bit_mask_feature(self, timestamp: int, bit_mask: str) -> BitMaskFeature:
        """
        统计当前时刻bit mask相关特征
        :param timestamp: 当前事件事件
        :param bit_mask: bit_mask数据信息
        :return: BitMaskFeature 特征信息
        """
        single_features = self.__default_single_bitmask_feature
        single_features['timestamp'] = timestamp
        err_bursts = []
        dq_mask = 0
        for sn, bit in enumerate(bit_mask[::-1]):
            bv = int(bit, 16)
            err_bit_count, min_bit_distance, _, adj_dq_num = CEBitMaskStatsMap[bv]
            single_features['err_bits_per_parity'] += err_bit_count
            single_features['adj_err_bit_cnt_per_parity'] += adj_dq_num
            if err_bit_count > single_features['max_err_bits_per_burst']:
                single_features['max_err_bits_per_burst'] = err_bit_count
            if bv > 0:
                err_bursts.append(sn)
                dq_mask |= bv
        single_features["err_bursts_per_parity"] = len(err_bursts)
        if len(err_bursts) > 0:
            single_features['max_err_burst_dist_per_parity'] = err_bursts[-1] - err_bursts[0]
        burst_dist = []
        if len(err_bursts) > 1:
            for i in range(0, len(err_bursts) - 1):
                burst_dist.append(err_bursts[i + 1] - err_bursts[i])
            single_features['min_err_burst_dist_per_parity'] = min(burst_dist)
        err_bit_count, min_bit_distance, _, adj_dq_num = CEBitMaskStatsMap[dq_mask]
        single_features['min_err_dq_dist'] = min_bit_distance
        single_features['adj_err_dq_cnt'] = adj_dq_num
        single_features["dq_cnt"] = err_bit_count
        return BitMaskFeature(**single_features)

    def __update_feature_of_addr(self, current_event: MEM_ERR_EVENT) -> None:
        """
        更新addr类统计特征
        :param current_event: 当前事件
        """
        self.addr_features['RankCnt'].add(current_event.phy_addr.rank)
        self.addr_features['BankCnt'].add(current_event.phy_addr.bank)
        self.addr_features['DeviceCnt'].add(current_event.phy_addr.dev)
        self.current_features['ce_addr'].update({
            'AddrCnt(Rank)': len(self.addr_features['RankCnt']),
            'AddrCnt(Bank)': len(self.addr_features['BankCnt']),
            'AddrCnt(Device)': len(self.addr_features['DeviceCnt'])
        })

    def __update_feature_of_bit_mask(self, current_event: MEM_ERR_EVENT,
                                     window_size=BIT_MASK_FEATURE_WINDOW_SIZE) -> None:
        """
        更新bit mask相关统计特征
        :param current_event: 当前事件
        :param window_size: 处理事件的时间窗大小
        """
        updated_bitmask_features = self.current_features['ce_bit_mask']
        bitmask_stat = self.addr_ce_bit_mask_stat

        if current_event.err_type == "CE.SCRUB":
            return
        bitmask = self.__prepare_bit_mask(current_event.bitmask)
        if bitmask is None:
            return
        current_features = self.__calc_current_bit_mask_feature(current_event.timestamp, bitmask)
        start, end = self.bitmask_feature_pointer.start, self.bitmask_feature_pointer.end
        self.__update_inc_dec(bitmask_stat, '+', current_features)
        self.__update_bitmask_stats(bitmask_stat, current_event, current_features, start, end, updated_bitmask_features,
                                    window_size)

        if bitmask_stat["bitmask_count"] > 0:
            updated_bitmask_features["AvgMaxErrBurstsDistPerParity"] = \
                bitmask_stat["total_max_burst_dist"] / bitmask_stat["bitmask_count"]
            updated_bitmask_features["AvgMinErrBurstsDistPertParity"] = \
                bitmask_stat["total_min_burst_dist"] / bitmask_stat["bitmask_count"]
            updated_bitmask_features["AvgMinErrDqDistPerParity"] = \
                bitmask_stat["total_min_bit_dist"] / bitmask_stat["bitmask_count"]
            updated_bitmask_features["AvgAdjErrDqCntPerParity"] = \
                bitmask_stat["total_adj_dq_count"] / bitmask_stat["bitmask_count"]
            updated_bitmask_features["AvgErrBurstsPerParity"] = \
                bitmask_stat["total_err_burst_count"] / bitmask_stat["bitmask_count"]
        if bitmask_stat["total_err_burst_count"] > 0:
            updated_bitmask_features["AvgErrBitsPerErrBurst"] = \
                bitmask_stat["total_err_bits_count"] / bitmask_stat["total_err_burst_count"]
        for i in range(1, DEVICE_DQ + 1):
            updated_bitmask_features[f"DqCnt={i}"] = bitmask_stat[f"DqCnt={i}"]
        for i in range(1, DEVICE_BURST + 1):
            updated_bitmask_features[f"BurstCnt={i}"] = bitmask_stat[f"BurstCnt={i}"]

    def __update_bitmask_stats(self, bitmask_stat, current_event, current_features, start, end,
                               updated_bitmask_features,
                               window_size) -> None:
        window_start = current_event.timestamp - window_size
        need_compare_all = True
        if len(self.bitmask_feature_list) < MAX_BIT_MASK_FEATURE_LIST_LEN:
            self.bitmask_feature_list.append(current_features)
            if self.bitmask_feature_list[start].timestamp >= window_start:
                self.__update_compare(updated_bitmask_features, current_features)
                need_compare_all = False
        else:
            if start == end:
                self.__update_inc_dec(bitmask_stat, '-', self.bitmask_feature_list[start])
                start = (start + 1) % MAX_BIT_MASK_FEATURE_LIST_LEN
            self.bitmask_feature_list[end] = current_features
        end = (end + 1) % MAX_BIT_MASK_FEATURE_LIST_LEN
        if need_compare_all:
            updated_bitmask_features.update(
                self.__default_feature_of_bit_mask(stat_features=False, compare_features=True))
            add_idx = 0
            end_tmp = end
            if end <= start:
                end_tmp = end + MAX_BIT_MASK_FEATURE_LIST_LEN
            for i in range(start, end_tmp):
                _idx = i % MAX_BIT_MASK_FEATURE_LIST_LEN
                _feature = self.bitmask_feature_list[_idx]
                if _feature.timestamp < window_start:
                    self.__update_inc_dec(bitmask_stat, '-', _feature)
                    add_idx += 1
                else:
                    self.__update_compare(updated_bitmask_features, _feature)
            start = (start + add_idx) % MAX_BIT_MASK_FEATURE_LIST_LEN
        self.bitmask_feature_pointer = ListPointers(start=start, end=end)

    def __update_feature_of_rpt_cnt(self, current_event: MEM_ERR_EVENT, evt_items: List[MEM_ERR_EVENT]) -> None:
        """
        更新默认重复统计特征
        :param current_event: 当前事件
        :param evt_items: 滑动窗口内事件列表（120小时内所有CE事件）
        """
        _rpt_cnt_feature = self.__default_feature_of_rpt_cnt
        cur_ts = current_event.timestamp
        for _event in evt_items[::-1]:
            _time_window = cur_ts - _event.timestamp
            if _event.err_type in {"CE.READ", "CE"}:
                if _time_window <= CNT_FEATURE_READ_WINDOW_SIZE_6M:
                    _rpt_cnt_feature[RPT_CNT_READ_FEATURE_NAME_6M] += 1
                elif _time_window <= CNT_FEATURE_READ_WINDOW_SIZE_3H:
                    _rpt_cnt_feature[RPT_CNT_READ_FEATURE_NAME_3H] += 1
                elif _time_window <= CNT_FEATURE_READ_WINDOW_SIZE_6H:
                    _rpt_cnt_feature[RPT_CNT_READ_FEATURE_NAME_6H] += 1
                elif _time_window <= CNT_FEATURE_READ_WINDOW_SIZE_120H:
                    _rpt_cnt_feature[RPT_CNT_READ_FEATURE_NAME_120H] += 1
        _rpt_cnt_feature[RPT_CNT_READ_FEATURE_NAME_3H] += _rpt_cnt_feature[RPT_CNT_READ_FEATURE_NAME_6M]
        _rpt_cnt_feature[RPT_CNT_READ_FEATURE_NAME_6H] += _rpt_cnt_feature[RPT_CNT_READ_FEATURE_NAME_3H]
        _rpt_cnt_feature[RPT_CNT_READ_FEATURE_NAME_120H] += _rpt_cnt_feature[RPT_CNT_READ_FEATURE_NAME_6H]
        self.current_features['ce_rpt_cnt'].update(_rpt_cnt_feature)

    def __update_feature_of_err_cnt(self, current_event: MEM_ERR_EVENT, evt_items: List[MEM_ERR_EVENT]) -> None:
        """
        更新故障类型统计特征
        :param current_event: 当前事件
        :param evt_items: 滑动窗口内事件列表（120小时内所有CE事件）
        """
        _err_cnt_feature = self.__default_feature_of_err_cnt
        cur_ts = current_event.timestamp
        for _event in evt_items[::-1]:
            _time_window = cur_ts - _event.timestamp
            if _event.err_type in {"CE.READ", "CE"}:
                if _time_window <= CNT_FEATURE_READ_WINDOW_SIZE_6M:
                    _err_cnt_feature[ERR_CNT_READ_FEATURE_NAME_6M] += _event.err_count
                elif _time_window <= CNT_FEATURE_READ_WINDOW_SIZE_3H:
                    _err_cnt_feature[ERR_CNT_READ_FEATURE_NAME_3H] += _event.err_count
                elif _time_window <= CNT_FEATURE_READ_WINDOW_SIZE_6H:
                    _err_cnt_feature[ERR_CNT_READ_FEATURE_NAME_6H] += _event.err_count
                elif _time_window <= CNT_FEATURE_READ_WINDOW_SIZE_120H:
                    _err_cnt_feature[ERR_CNT_READ_FEATURE_NAME_120H] += _event.err_count
        _err_cnt_feature[ERR_CNT_READ_FEATURE_NAME_3H] += _err_cnt_feature[ERR_CNT_READ_FEATURE_NAME_6M]
        _err_cnt_feature[ERR_CNT_READ_FEATURE_NAME_6H] += _err_cnt_feature[ERR_CNT_READ_FEATURE_NAME_3H]
        _err_cnt_feature[ERR_CNT_READ_FEATURE_NAME_120H] += _err_cnt_feature[ERR_CNT_READ_FEATURE_NAME_6H]
        self.current_features['ce_err_cnt'].update(_err_cnt_feature)

    def __update_feature_of_ce_storm_cnt_feature(self, current_event: MEM_ERR_EVENT,
                                                 evt_items: List[MEM_ERR_EVENT]) -> None:
        """
        更新ce风暴统计特征
        :param current_event: 当前事件
        :param evt_items: 滑动窗口内事件列表（120小时内所有CE事件）
        """
        ce_storm_cnt_feature = self.__default_feature_of_ce_storm_cnt_feature
        cur_ts = current_event.timestamp

        ce_continuous = 0
        ce_storm_trigger = False
        for i, _event in enumerate(evt_items):
            _time_window = cur_ts - _event.timestamp
            if i != 0 and (_event.err_type in {"CE.READ", "CE", "CE.SCRUB"}) and \
                    (_time_window <= CE_STORM_CNT_FEATURE_WINDOW_SIZE):
                if _event.timestamp - evt_items[i - 1].timestamp <= CE_STORM_INTERVAL:
                    ce_continuous += 1
                else:
                    ce_continuous = 0
                    ce_storm_trigger = False

                if not ce_storm_trigger and ce_continuous > CE_STORM_THRESHOLD:
                    ce_storm_cnt_feature[CE_STORM_CNT_ALL_FEATURE_NAME] += 1
                    ce_storm_trigger = True
                    if _event.timestamp == cur_ts:
                        ce_storm_cnt_feature[CE_STORM_CNT_CURRENT_FEATURE_NAME] = 1

                elif _event.timestamp == cur_ts:
                    ce_storm_cnt_feature[CE_STORM_CNT_CURRENT_FEATURE_NAME] = 1

        self.current_features['ce_storm_cnt'].update(ce_storm_cnt_feature)

    def update_all_features(self, current_event: MEM_ERR_EVENT) -> Dict:
        self.history_event_list.append(current_event)

        max_window_size = WINDOW_SIZE_120H
        evt_items = self.__filter_events(self.history_event_list, current_event, window_size=max_window_size)

        self.__update_feature_of_addr(current_event)
        self.__update_feature_of_bit_mask(current_event)
        self.__update_feature_of_rpt_cnt(current_event, evt_items)
        self.__update_feature_of_err_cnt(current_event, evt_items)
        self.__update_feature_of_ce_storm_cnt_feature(current_event, evt_items)
        return self.current_features

    @staticmethod
    def make_feature_vector(feature_dict: Dict, feature_name_dict: Dict) -> List:
        """
        生成特征向量
        :param feature_dict: 特征字典
        :param feature_name_dict: 训练需要使用到的特征名称
        :return: 特征向量
        """
        _feature_vector = list()
        for _feature_name in feature_name_dict.values():
            if _feature_name in feature_dict:
                _feature_vector.append(feature_dict[_feature_name])
            else:
                raise RuntimeError(f"feature: {_feature_name} not found in feature input.")
        return _feature_vector


class Baseline:
    def __init__(self, competition_data_path: str, sn_type: str, sample_count: int = 0, worker_count: int = 12,
                 filter_by_register: bool = False):
        self.competition_data_path = competition_data_path
        self.sn_type = sn_type
        self.sample_count = sample_count
        self.worker_count = worker_count
        self.filter_by_register = filter_by_register

    feature_dict = {
        # 特征组1 addr特征
        "AddrCnt_1": "AddrCnt(Rank)",
        "AddrCnt_2": "AddrCnt(Bank)",
        "AddrCnt_3": "AddrCnt(Device)",

        # 特征组2 CE风暴特征
        "CE_storm_1": "CE_storm_Cnt(All,current)",
        "CE_storm_2": "CE_storm_Cnt(All,120h)",

        # 特征组3 重复次数统计特征
        "RptCnt_1": "RptCnt(CE.READ,120h)",
        "RptCnt_2": "RptCnt(CE.READ,6h)",
        "RptCnt_3": "RptCnt(CE.READ,3h)",
        "RptCnt_4": "RptCnt(CE.READ,6m)",

        # 特征组4 错误统计特征（仅在 filter_by_register=True 时使用）
        "ErrCnt_1": "ErrCnt(CE.READ,120h)",
        "ErrCnt_2": "ErrCnt(CE.READ,6h)",
        "ErrCnt_3": "ErrCnt(CE.READ,3h)",
        "ErrCnt_4": "ErrCnt(CE.READ,6m)",

        # 特征组5 bit mask特征
        "Bitmask_minmax_1": "MinErrDqDistance(120h,100)",
        "Bitmask_minmax_2": "MinErrBurstDist(120h,100)",
        "Bitmask_minmax_3": "MaxErrBitsPerBurst(120h,100)",
        "Bitmask_minmax_4": "MaxErrBurstCnt(120h,100)",
        "Bitmask_minmax_5": "MaxErrBurstDist(120h,100)",
        "Bitmask_minmax_6": "MaxAdjErrDqCnt(120h,100)",
        "Bitmask_minmax_7": "MaxAdjErrBitCnt(120h,100)",
        "Bitmask_avg_1": "AvgErrBitsPerErrBurst(120h,100)",
        "Bitmask_avg_2": "AvgErrBurstsPerParity(120h,100)",
        "Bitmask_avg_3": "AvgMinErrDqDistPerParity(120h,100)",
        "Bitmask_avg_4": "AvgMinErrBurstsDistPertParity(120h,100)",
        "Bitmask_avg_5": "AvgMaxErrBurstsDistPerParity(120h,100)",
        "Bitmask_avg_6": "AvgAdjErrDqCntPerParity(120h,100)",
        "Bitmask_dq_1": "DqCnt=1(120h,100)",
        "Bitmask_dq_2": "DqCnt=2(120h,100)",
        "Bitmask_dq_3": "DqCnt=3(120h,100)",
        "Bitmask_dq_4": "DqCnt=4(120h,100)",
        "Bitmask_burst_1": "BurstCnt=1(120h,100)",
        "Bitmask_burst_2": "BurstCnt=2(120h,100)",
        "Bitmask_burst_3": "BurstCnt=3(120h,100)",
        "Bitmask_burst_4": "BurstCnt=4(120h,100)",
        "Bitmask_burst_5": "BurstCnt=5(120h,100)",
        "Bitmask_burst_6": "BurstCnt=6(120h,100)",
        "Bitmask_burst_7": "BurstCnt=7(120h,100)",
        "Bitmask_burst_8": "BurstCnt=8(120h,100)",
    }
    LGB_MODEL_PARAMS = {"objective": 'binary', "metric": 'f1score', "n_estimators": 1000, "max_depth": 8,
                        "num_leaves": 100, "learning_rate": 0.05, "feature_fraction": 1.0, "num_iteration": 50,
                        "importance_type": 'gain', "is_unbalance": True}

    def process_chunk(self, args) -> None:
        """
        处理数据块
        :param args: (csv_files, thread, pkl_directory)
        """
        csv_files, thread, pkl_directory = args[0], args[1], args[2]
        cur_event_dict = {}
        if not csv_files:
            return
        ticket_csv_path = csv_files[0].split("type_")[0] + "ticket.csv"
        ticket_df = pd.read_csv(ticket_csv_path, index_col=0)
        condition = (ticket_df['alarm_time'] <= TRAIN_TEST_SPLIT_TIMESTAMP) & (
                ticket_df['sn_type'] == csv_files[0].split("type_")[1][0])
        train_ticket_df = ticket_df.loc[condition]
        train_ticket_dict = train_ticket_df.set_index('sn_name')['alarm_time'].to_dict()
        for key in list(cur_event_dict.keys()):
            sn_name = key.split("\\")[-1]
            if sn_name in train_ticket_dict:
                alarm_time = train_ticket_dict[sn_name]
                cur_event_dict[key] = [event for event in cur_event_dict[key] if event['LogTime'] <= alarm_time]

        for index, csv_file in tqdm(enumerate(csv_files), desc=f"Processing chunk {thread + 1}"):
            with open(csv_file, 'r', encoding='utf-8') as f:
                df = Baseline.reduce_memory_usage(pd.read_csv(f))
            cur_event_list = df.to_dict(orient='records')
            sn_name = csv_file.split(".")[0].split("\\")[-1]
            cur_event_dict[sn_name] = cur_event_list
            if sn_name in train_ticket_dict:
                alarm_time = train_ticket_dict[sn_name]
                cur_event_dict[sn_name] = [event for event in cur_event_dict[sn_name] if
                                           event['LogTime'] <= alarm_time]
            if (index + 1) % 20 == 0 or index == len(csv_files) - 1:
                if self.filter_by_register:
                    cur_event_dict = Baseline.filter_data_by_register(cur_event_dict)
                train_feature = self.light_feature_extraction(cur_event_dict)
                train_feature['SN'] = train_feature["SN"].apply(lambda x: x.split('\\')[-1])
                RestrictedUnpickler.write_pkl_data(train_feature,
                                                   '{}/train_feature_{}_{}.pkl'.format(pkl_directory, thread + 1,
                                                                                       index + 1), is_binary=True)
                cur_event_dict.clear()
                gc.collect()
        return

    def light_feature_extraction(self, feature_data: Dict) -> pd.DataFrame:
        """
        从原始数据中提取特征（轻量化版本）
        :param feature_data: 原始特征的Dict
        :return: 提取出训练可用特征的Dataframe
        """
        all_features = []
        for dimm_sn, log in feature_data.items():
            extractor = FeatureExtractor()
            input_data = InputAdapter.convert(log, dimm_sn, self.sample_count)
            single_sn_features = []
            for _evt_id, _item in enumerate(input_data):
                current_features = extractor.update_all_features(_item.mesg)
                new_features = extractor.get_dimm_feature_dict(current_features)
                new_feature_vector = FeatureExtractor.make_feature_vector(new_features, Baseline.feature_dict)
                new_feature_vector = [_item.dimm_key, 0, _item.timestamp] + new_feature_vector
                single_sn_features.append(new_feature_vector)
            all_features.extend(single_sn_features)
        return pd.DataFrame(all_features, columns=['SN', 'WithUCE', 'TimeStamp'] + list(Baseline.feature_dict.values()))

    @staticmethod
    def filter_data_by_register(input_log_data: Dict, filter_time_period: int = 21600) -> Dict:
        """
        对输入的数据进行去重,默认使用最近6小时数据进行去重
        :param input_log_data: 输入日志数据：{sn:[{log1},{log2},...]}
        :param filter_time_period: 去重使用的时间窗大小，默认6小时即 21600秒（单位秒）
        :return: 去重之后的日志数据
        """
        filtered_data = dict()
        for sn, log_data_list in input_log_data.items():
            log_data_list.sort(key=lambda x: x['LogTime'])
            for log_data in log_data_list:
                rd_parity = str(log_data.get('RetryRdErrLogParity', ''))
                row_id, col_id, bank_id, rank_id = log_data.get('RowId', -1), log_data.get('ColumnId', -1), \
                    log_data.get('BankId', -1), log_data.get('RankId', -1)
                logtime_revised = log_data.get('LogTime') // filter_time_period
                tag = f"{rd_parity}_{logtime_revised}_{row_id}_{col_id}_{bank_id}_{rank_id}"
                log_data['Count'] = 1
                if sn not in filtered_data:
                    filtered_data[sn] = {tag: log_data}
                elif tag not in filtered_data[sn]:
                    filtered_data[sn][tag] = log_data
                else:
                    filtered_data[sn][tag]['Count'] += 1
        # 展开，去重tag标签
        for sn in filtered_data:
            filtered_data[sn] = list(filtered_data[sn].values())
        return filtered_data

    @staticmethod
    def ticket_labeling(train_feature: pd.DataFrame, ticket_data: pd.DataFrame) -> pd.DataFrame:
        """
        基于维修单为数据集创建标签
        :param train_feature: 用于训练的特征文件
        :param ticket_data: 维修单数据
        :return: 重新标签后的数据集
        """
        # 合并训练特征和维修单标签
        trainset_map = train_feature.merge(ticket_data,
                                           left_on='SN', right_on='factory_serial_number', how='left').fillna(-1)
        # 基于告警时间对训练特征文件打标签
        trainset_map['alarm_time'] = trainset_map['alarm_time'].astype(int)
        trainset_map['gap'] = trainset_map['alarm_time'] - trainset_map['TimeStamp']
        trainset_map.loc[(trainset_map['gap'] <= WINDOW_SIZE_120H) & (trainset_map['alarm_time'] > 0), 'WithUCE'] = \
            trainset_map['alarm_time']
        # 去掉不必要的错误事件和多余的特征
        training_set = trainset_map.drop(trainset_map[(trainset_map['WithUCE'] != 0)
                                                      & (trainset_map['TimeStamp'] > trainset_map['WithUCE'])].index)
        faulty_dimm = training_set['SN'][training_set['WithUCE'] != 0].drop_duplicates()
        training_set = training_set.drop(training_set[(training_set['SN'].isin(faulty_dimm))
                                                      & (training_set['WithUCE'] == 0)].index)
        training_set.drop(columns=['factory_serial_number', 'alarm_time', 'gap'], inplace=True)
        return training_set

    @staticmethod
    def make_label(train_feature: pd.DataFrame) -> pd.DataFrame:
        """
        为数据集创建UCE标签，原"WithUCE"列的值将被复制到"UCETOI"列.
        :param train_feature: 原始生成数据集
        :return: 创建标签以后的训练集
        """
        uce_samples = train_feature.loc[train_feature["WithUCE"] > 0].copy()
        non_uce_samples = train_feature.loc[train_feature["WithUCE"] == 0].copy()
        uce_samples["UCETOI"] = uce_samples["WithUCE"]
        non_uce_samples["UCETOI"] = non_uce_samples["WithUCE"]
        uce_samples["WithUCE"] = 1
        non_uce_samples["WithUCE"] = 0
        train_set = pd.concat([uce_samples, non_uce_samples], axis=0)
        return train_set

    @staticmethod
    def plot_feature_importance(feature_importances, feature_names):
        # 创建 DataFrame 以便于展示
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        # 打印特征重要性
        print(importance_df)
        plt.figure(figsize=(16, 8))
        plt.barh(importance_df['Feature'], importance_df['Importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.gca().invert_yaxis()
        plt.show()

    @staticmethod
    def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
        """
        优化内存使用
        :param df: 默认数据类型的Dataframe
        :return: 优化内存占用后的Dataframe
        """
        for col in df.columns:
            col_type = df[col].dtype
            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()

                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)

        return df

    def step1_feature_extraction(self) -> None:
        """
        将原始特征转换为训练可用特征
        """
        competition_data_path = self.competition_data_path
        sn_type = self.sn_type
        pkl_directory = f'processed_features_{sn_type}'
        if not os.path.exists(pkl_directory):
            os.makedirs(pkl_directory)
        csv_files = [os.path.join(competition_data_path, f'type_{sn_type}', i) for i in
                     os.listdir(os.path.join(competition_data_path, f'type_{sn_type}')) if i.endswith('.csv')]
        remaining_csv_files = [f for f in csv_files if f"type_{sn_type}" in f]
        finished_csv_files = []
        for filename in os.listdir(pkl_directory):
            if filename.endswith('.pkl'):
                file_path = os.path.join(pkl_directory, filename)
                df_chunks = RestrictedUnpickler.read_pkl_file(file_path)
                finished_csv_files.extend(df_chunks['SN'].unique())
        print("Finished csv files: ", len(finished_csv_files))
        remaining_csv_files = [i for i in remaining_csv_files if i.split("\\")[-1][:-4] not in finished_csv_files]
        print("Remaining csv files: ", len(remaining_csv_files))

        worker_count = self.worker_count

        if worker_count > 1:
            chunk_size = len(remaining_csv_files) // worker_count + 1
            chunk_files = [remaining_csv_files[i:i + chunk_size] for i in
                           range(0, len(remaining_csv_files), chunk_size)]
            pool = multiprocessing.Pool()
            pool.map(self.process_chunk,
                     zip(chunk_files, list(range(len(finished_csv_files), len(finished_csv_files) + worker_count)),
                         [pkl_directory] * worker_count))
            pool.close()
            pool.join()
        else:
            self.process_chunk((remaining_csv_files, len(finished_csv_files), pkl_directory))
        return

    def step2_create_datasets(self) -> None:
        """
        将pkl文件切片合并成Dataframe，划分训练和验证集，并添加UCE标签
        """
        competition_data_path = self.competition_data_path
        sn_type = self.sn_type
        pkl_directory = f'processed_features_{sn_type}'
        df_chunks = []
        for filename in tqdm(os.listdir(pkl_directory)):
            if filename.endswith('.pkl'):
                file_path = os.path.join(pkl_directory, filename)
                df_chunk = Baseline.reduce_memory_usage(RestrictedUnpickler.read_pkl_file(file_path))
                df_chunks.append(df_chunk)
        combined_df = pd.concat(df_chunks, ignore_index=True)
        combined_df.to_csv("combined_df.csv", index=False)
        print("Combined df shape: ", combined_df.shape)

        ticket_df = pd.read_csv(os.path.join(competition_data_path, "ticket.csv"), index_col=0)
        ticket_df.rename(columns={'sn_name': 'factory_serial_number'}, inplace=True)

        train_df = combined_df[combined_df['TimeStamp'] <= TRAIN_TEST_SPLIT_TIMESTAMP]
        train_condition = (ticket_df['alarm_time'] <= TRAIN_TEST_SPLIT_TIMESTAMP) & (ticket_df['sn_type'] == sn_type)
        train_ticket_df = ticket_df.loc[train_condition]
        train_feature = Baseline.ticket_labeling(train_df, train_ticket_df)
        labelled_train_df = Baseline.make_label(train_feature)
        labelled_train_df.to_csv("train_df.csv", index=False)
        print("Train df shape: ", labelled_train_df.shape)
        del labelled_train_df, train_df

        val_df = combined_df[combined_df['TimeStamp'] > TRAIN_TEST_SPLIT_TIMESTAMP]
        val_condition = (ticket_df['alarm_time'] > TRAIN_TEST_SPLIT_TIMESTAMP) & (ticket_df['sn_type'] == sn_type)
        val_ticket_df = ticket_df.loc[val_condition]
        val_feature = Baseline.ticket_labeling(val_df, val_ticket_df)
        labelled_val_df = Baseline.make_label(val_feature)
        labelled_val_df.to_csv("val_df.csv", index=False)
        print("Val df shape: ", labelled_val_df.shape)
        del labelled_val_df, val_df
        return

    def step3_train_lgbm_model(self) -> None:
        """
        训练LightGBM模型并保存
        """
        chunk_size = 1000000
        chunks = []
        for chunk in tqdm(pd.read_csv("train_df.csv", chunksize=chunk_size, low_memory=False)):
            chunks.append(Baseline.reduce_memory_usage(chunk))
        train_df = pd.concat(chunks, ignore_index=True)

        model = lightgbm.LGBMClassifier(**Baseline.LGB_MODEL_PARAMS)
        model.fit(
            train_df[
                train_df.drop(columns=['SN', 'WithUCE', 'TimeStamp', 'UCETOI', 'sn_type']).columns].values,
            train_df["WithUCE"])
        RestrictedUnpickler.write_pkl_data(model, 'model.pkl', is_binary=True)

        show_feature_importance = False
        if show_feature_importance:
            feature_importances = model.feature_importances_
            feature_names = train_df[
                train_df.drop(columns=['SN', 'WithUCE', 'TimeStamp', 'UCETOI', 'sn_type']).columns].columns
            self.plot_feature_importance(feature_importances, feature_names)

        return

    def step4_val_and_inf(self) -> None:
        """
        验证模型效果，最后绘制p-r曲线
        """
        competition_data_path = self.competition_data_path
        sn_type = self.sn_type
        model = RestrictedUnpickler.read_pkl_file('model.pkl')

        chunk_size = 1000000
        chunks = []
        for chunk in tqdm(pd.read_csv("val_df.csv", chunksize=chunk_size, low_memory=False)):
            chunks.append(Baseline.reduce_memory_usage(chunk))
        val_df = pd.concat(chunks, ignore_index=True)
        result = model.predict_proba(
            val_df[
                val_df.drop(columns=['SN', 'WithUCE', 'TimeStamp', 'UCETOI', 'sn_type']).columns].values)

        ticket_df = pd.read_csv(os.path.join(competition_data_path, "ticket.csv"), index_col=0)
        ticket_df.rename(columns={'sn_name': 'factory_serial_number'}, inplace=True)
        val_condition = (ticket_df['alarm_time'] > TRAIN_TEST_SPLIT_TIMESTAMP) & (ticket_df['sn_type'] == sn_type)
        val_ticket_df = ticket_df.loc[val_condition]

        inf_df = pd.DataFrame({
            'SN': val_df['SN'],
            'TimeStamp': val_df['TimeStamp'],
            'label': result[:, 1]
        })

        precisions = []
        recalls = []
        thresholds = [i / 100 for i in range(60, 100, 2)]
        for threshold in thresholds:
            label = inf_df['label'].apply(lambda x: 1 if x >= threshold else 0)
            origin_df = inf_df.copy()
            origin_df['label'] = inf_df['label'].apply(lambda x: 1 if x >= threshold else 0)
            origin_df = origin_df[
                (origin_df['label'] == 1) & (origin_df['SN'].isin(val_ticket_df['factory_serial_number'].values))]
            filtered_df = inf_df[label == 1].groupby('SN')['TimeStamp'].min()
            sn_min_timestamp = filtered_df.to_dict()
            time_window_start = 15 * 60  # 15 minutes in seconds
            time_window_end = 7 * 24 * 60 * 60  # 7 days in seconds
            precision_tp, recall_tp = 0, 0
            for sn, min_timestamp in sn_min_timestamp.items():
                if sn in val_ticket_df['factory_serial_number'].values:
                    alarm_time = val_ticket_df[val_ticket_df['factory_serial_number'] == sn]['alarm_time'].values[0]
                    if alarm_time - time_window_end <= min_timestamp <= alarm_time - time_window_start:
                        precision_tp += 1
            for sn in origin_df['SN'].unique():
                sn_df = origin_df[origin_df['SN'] == sn]
                for timestamp in sn_df['TimeStamp']:
                    if sn in val_ticket_df['factory_serial_number'].values:
                        alarm_time = val_ticket_df[val_ticket_df['factory_serial_number'] == sn]['alarm_time'].values[0]
                        if alarm_time - time_window_end <= timestamp <= alarm_time - time_window_start:
                            recall_tp += 1
                            break
            precision = precision_tp / len(sn_min_timestamp) if len(sn_min_timestamp) > 0 else 0
            recall = recall_tp / len(val_ticket_df) if len(val_ticket_df) > 0 else 0
            print("threshold: {}, precision: {:.3f}, recall: {:.3f}".format(threshold, precision, recall))
            precisions.append(precision)
            recalls.append(recall)

        # Plot the PR curve
        plt.figure()
        print(recalls, precisions)
        plt.plot(recalls, precisions, marker='o')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.show()
        return

    def step4_inf(self) -> None:
        """
        验证模型效果并进行推理，最后绘制p-r曲线
        """
        sn_type = self.sn_type

        model = RestrictedUnpickler.read_pkl_file('model.pkl')

        chunk_size = 1000000
        chunks = []
        for chunk in tqdm(pd.read_csv("val_df.csv", chunksize=chunk_size, low_memory=False)):
            chunks.append(Baseline.reduce_memory_usage(chunk))
        val_df = pd.concat(chunks, ignore_index=True)
        result = model.predict_proba(
            val_df[
                val_df.drop(columns=['SN', 'WithUCE', 'TimeStamp', 'UCETOI', 'sn_type']).columns].values)

        inf_df = pd.DataFrame({
            'sn_name': val_df['SN'],
            'alarm_time': val_df['TimeStamp'],
            'label': result[:, 1]
        })
        label = inf_df['label'].apply(lambda x: 1 if x >= threshold else 0)
        filtered_df = inf_df[label == 1].groupby('sn_name')['alarm_time'].min()
        result_df = pd.DataFrame(list(filtered_df.items()), columns=['sn_name', 'alarm_time'])
        result_df["sn_type"] = sn_type
        result_df.to_csv(f"result_df_{sn_type}.csv", index=False)
        return


if __name__ == "__main__":
    for sn_type in ["A", "B"]:
        baseline = Baseline(competition_data_path="competition_data", sn_type=sn_type, sample_count=1000,
                            worker_count=12, filter_by_register=False)
        baseline.step1_feature_extraction()
        baseline.step2_create_datasets()
        baseline.step3_train_lgbm_model()
        baseline.step4_inf()
    result_df = pd.concat([pd.read_csv(f'result_df_{sn_type}.csv') for sn_type in ["A", "B"]])
    result_df.to_csv("submission.csv", index=False)
    print()


