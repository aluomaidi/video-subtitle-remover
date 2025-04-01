import torch
import shutil
import subprocess
import os
from pathlib import Path
import threading
import cv2
import sys
import numpy as np
import time
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from backend.tools.common_tools import is_video_or_image, is_image_file
from backend.scenedetect import scene_detect
from backend.scenedetect.detectors import ContentDetector
from backend.inpaint.sttn_inpaint import STTNInpaint, STTNVideoInpaint
from backend.inpaint.lama_inpaint import LamaInpaint
from backend.inpaint.video_inpaint import VideoInpaint
from backend.tools.inpaint_tools import create_mask, batch_generator
import importlib
import platform
import tempfile
import multiprocessing
from shapely.geometry import Polygon
from tqdm import tqdm
from tools.infer import utility
from tools.infer.predict_det import TextDetector


class SubtitleDetect:
    """
    文本框检测类，用于检测视频帧中是否存在文本框
    """

    def __init__(self, video_path, sub_area=None):
        # 获取参数对象
        importlib.reload(config)
        args = utility.parse_args()
        args.det_algorithm = 'DB'
        args.det_model_dir = config.DET_MODEL_PATH
        self.text_detector = TextDetector(args)
        self.video_path = video_path
        self.sub_area = sub_area

    def detect_subtitle(self, img):
        dt_boxes, elapse = self.text_detector(img)
        return dt_boxes, elapse

    @staticmethod
    def get_coordinates(dt_box):
        """
        从返回的检测框中获取坐标
        :param dt_box 检测框返回结果
        :return list 坐标点列表
        """
        coordinate_list = list()
        if isinstance(dt_box, list):
            for i in dt_box:
                i = list(i)
                (x1, y1) = int(i[0][0]), int(i[0][1])
                (x2, y2) = int(i[1][0]), int(i[1][1])
                (x3, y3) = int(i[2][0]), int(i[2][1])
                (x4, y4) = int(i[3][0]), int(i[3][1])
                xmin = max(x1, x4)
                xmax = min(x2, x3)
                ymin = max(y1, y2)
                ymax = min(y3, y4)
                coordinate_list.append((xmin, xmax, ymin, ymax))
        return coordinate_list

    def find_subtitle_frame_no(self, sub_remover=None):
        video_cap = cv2.VideoCapture(self.video_path)
        frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        current_frame_no = 0
        subtitle_frame_no_box_dict = {}
        print('[Processing] start finding subtitles...')
        
        # 如果指定了字幕区域，直接使用该区域
        if self.sub_area is not None:
            s_ymin, s_ymax, s_xmin, s_xmax = self.sub_area
            # 为每一帧使用相同的字幕区域
            while video_cap.isOpened():
                ret, frame = video_cap.read()
                if not ret:
                    break
                current_frame_no += 1
                subtitle_frame_no_box_dict[current_frame_no] = [(s_xmin, s_xmax, s_ymin, s_ymax)]
        else:
            # 如果没有指定字幕区域，则进行文本检测
            while video_cap.isOpened():
                ret, frame = video_cap.read()
                if not ret:
                    break
                current_frame_no += 1
                dt_boxes, elapse = self.detect_subtitle(frame)
                coordinate_list = self.get_coordinates(dt_boxes.tolist())
                if coordinate_list:
                    subtitle_frame_no_box_dict[current_frame_no] = coordinate_list
                if sub_remover:
                    sub_remover.update_progress(current_frame_no, int(frame_count), "Finding Subtitles")
                    sub_remover.progress_total = (100 * float(current_frame_no) / float(frame_count)) // 2
        
        subtitle_frame_no_box_dict = self.unify_regions(subtitle_frame_no_box_dict)
        print('[Finished] Finished finding subtitles...')
        new_subtitle_frame_no_box_dict = dict()
        for key in subtitle_frame_no_box_dict.keys():
            if len(subtitle_frame_no_box_dict[key]) > 0:
                new_subtitle_frame_no_box_dict[key] = subtitle_frame_no_box_dict[key]
        return new_subtitle_frame_no_box_dict

    @staticmethod
    def split_range_by_scene(intervals, points):
        # 确保离散值列表是有序的
        points.sort()
        # 用于存储结果区间的列表
        result_intervals = []
        # 遍历区间
        for start, end in intervals:
            # 在当前区间内的点
            current_points = [p for p in points if start <= p <= end]

            # 遍历当前区间内的离散点
            for p in current_points:
                # 如果当前离散点不是区间的起始点，添加从区间开始到离散点前一个数字的区间
                if start < p:
                    result_intervals.append((start, p - 1))
                # 更新区间开始为当前离散点
                start = p
            # 添加从最后一个离散点或区间开始到区间结束的区间
            result_intervals.append((start, end))
        # 输出结果
        return result_intervals

    @staticmethod
    def get_scene_div_frame_no(v_path):
        """
        获取发生场景切换的帧号
        """
        scene_div_frame_no_list = []
        scene_list = scene_detect(v_path, ContentDetector())
        for scene in scene_list:
            start, end = scene
            if start.frame_num == 0:
                pass
            else:
                scene_div_frame_no_list.append(start.frame_num + 1)
        return scene_div_frame_no_list

    @staticmethod
    def are_similar(region1, region2):
        """判断两个区域是否相似。"""
        xmin1, xmax1, ymin1, ymax1 = region1
        xmin2, xmax2, ymin2, ymax2 = region2

        return abs(xmin1 - xmin2) <= config.PIXEL_TOLERANCE_X and abs(xmax1 - xmax2) <= config.PIXEL_TOLERANCE_X and \
            abs(ymin1 - ymin2) <= config.PIXEL_TOLERANCE_Y and abs(ymax1 - ymax2) <= config.PIXEL_TOLERANCE_Y

    def unify_regions(self, raw_regions):
        """将连续相似的区域统一，保持列表结构。"""
        if len(raw_regions) > 0:
            keys = sorted(raw_regions.keys())  # 对键进行排序以确保它们是连续的
            unified_regions = {}

            # 初始化
            last_key = keys[0]
            unify_value_map = {last_key: raw_regions[last_key]}

            for key in keys[1:]:
                current_regions = raw_regions[key]

                # 新增一个列表来存放匹配过的标准区间
                new_unify_values = []

                for idx, region in enumerate(current_regions):
                    last_standard_region = unify_value_map[last_key][idx] if idx < len(unify_value_map[last_key]) else None

                    # 如果当前的区间与前一个键的对应区间相似，我们统一它们
                    if last_standard_region and self.are_similar(region, last_standard_region):
                        new_unify_values.append(last_standard_region)
                    else:
                        new_unify_values.append(region)

                # 更新unify_value_map为最新的区间值
                unify_value_map[key] = new_unify_values
                last_key = key

            # 将最终统一后的结果传递给unified_regions
            for key in keys:
                unified_regions[key] = unify_value_map[key]
            return unified_regions
        else:
            return raw_regions

    @staticmethod
    def find_continuous_ranges(subtitle_frame_no_box_dict):
        """
        获取字幕出现的起始帧号与结束帧号
        """
        numbers = sorted(list(subtitle_frame_no_box_dict.keys()))
        ranges = []
        start = numbers[0]  # 初始区间开始值

        for i in range(1, len(numbers)):
            # 如果当前数字与前一个数字间隔超过1，
            # 则上一个区间结束，记录当前区间的开始与结束
            if numbers[i] - numbers[i - 1] != 1:
                end = numbers[i - 1]  # 则该数字是当前连续区间的终点
                ranges.append((start, end))
                start = numbers[i]  # 开始下一个连续区间
        # 添加最后一个区间
        ranges.append((start, numbers[-1]))
        return ranges

    @staticmethod
    def find_continuous_ranges_with_same_mask(subtitle_frame_no_box_dict):
        numbers = sorted(list(subtitle_frame_no_box_dict.keys()))
        ranges = []
        start = numbers[0]  # 初始区间开始值
        for i in range(1, len(numbers)):
            # 如果当前帧号与前一个帧号间隔超过1，
            # 则上一个区间结束，记录当前区间的开始与结束
            if numbers[i] - numbers[i - 1] != 1:
                end = numbers[i - 1]  # 则该数字是当前连续区间的终点
                ranges.append((start, end))
                start = numbers[i]  # 开始下一个连续区间
            # 如果当前帧号与前一个帧号间隔为1，且当前帧号对应的坐标点与上一帧号对应的坐标点不一致
            # 记录当前区间的开始与结束
            if numbers[i] - numbers[i - 1] == 1:
                if subtitle_frame_no_box_dict[numbers[i]] != subtitle_frame_no_box_dict[numbers[i - 1]]:
                    end = numbers[i - 1]  # 则该数字是当前连续区间的终点
                    ranges.append((start, end))
                    start = numbers[i]  # 开始下一个连续区间
        # 添加最后一个区间
        ranges.append((start, numbers[-1]))
        return ranges

    @staticmethod
    def sub_area_to_polygon(sub_area):
        """
        xmin, xmax, ymin, ymax = sub_area
        """
        s_xmin = sub_area[0]
        s_xmax = sub_area[1]
        s_ymin = sub_area[2]
        s_ymax = sub_area[3]
        return Polygon([[s_xmin, s_ymin], [s_xmax, s_ymin], [s_xmax, s_ymax], [s_xmin, s_ymax]])

    @staticmethod
    def expand_and_merge_intervals(intervals, expand_size=config.STTN_NEIGHBOR_STRIDE*config.STTN_REFERENCE_LENGTH, max_length=config.STTN_MAX_LOAD_NUM):
        # 初始化输出区间列表
        expanded_intervals = []

        # 对每个原始区间进行扩展
        for interval in intervals:
            start, end = interval

            # 扩展至至少 'expand_size' 个单位，但不超过 'max_length' 个单位
            expansion_amount = max(expand_size - (end - start + 1), 0)

            # 在保证包含原区间的前提下尽可能平分前后扩展量
            expand_start = max(start - expansion_amount // 2, 1)  # 确保起始点不小于1
            expand_end = end + expansion_amount // 2

            # 如果扩展后的区间超出了最大长度，进行调整
            if (expand_end - expand_start + 1) > max_length:
                expand_end = expand_start + max_length - 1

            # 对于单点的处理，需额外保证有至少 'expand_size' 长度
            if start == end:
                if expand_end - expand_start + 1 < expand_size:
                    expand_end = expand_start + expand_size - 1

            # 检查与前一个区间是否有重叠并进行相应的合并
            if expanded_intervals and expand_start <= expanded_intervals[-1][1]:
                previous_start, previous_end = expanded_intervals.pop()
                expand_start = previous_start
                expand_end = max(expand_end, previous_end)

            # 添加扩展后的区间至结果列表
            expanded_intervals.append((expand_start, expand_end))

        return expanded_intervals

    @staticmethod
    def filter_and_merge_intervals(intervals, target_length=config.STTN_REFERENCE_LENGTH):
        """
        合并传入的字幕起始区间，确保区间大小最低为STTN_REFERENCE_LENGTH
        """
        expanded = []
        # 首先单独处理单点区间以扩展它们
        for start, end in intervals:
            if start == end:  # 单点区间
                # 扩展到接近的目标长度，但保证前后不重叠
                prev_end = expanded[-1][1] if expanded else float('-inf')
                next_start = float('inf')
                # 查找下一个区间的起始点
                for ns, ne in intervals:
                    if ns > end:
                        next_start = ns
                        break
                # 确定新的扩展起点和终点
                new_start = max(start - (target_length - 1) // 2, prev_end + 1)
                new_end = min(start + (target_length - 1) // 2, next_start - 1)
                # 如果新的扩展终点在起点前面，说明没有足够空间来进行扩展
                if new_end < new_start:
                    new_start, new_end = start, start  # 保持原样
                expanded.append((new_start, new_end))
            else:
                # 非单点区间直接保留，稍后处理任何可能的重叠
                expanded.append((start, end))
        # 排序以合并那些因扩展导致重叠的区间
        expanded.sort(key=lambda x: x[0])
        # 合并重叠的区间，但仅当它们之间真正重叠且小于目标长度时
        merged = [expanded[0]]
        for start, end in expanded[1:]:
            last_start, last_end = merged[-1]
            # 检查是否重叠
            if start <= last_end and (end - last_start + 1 < target_length or last_end - last_start + 1 < target_length):
                # 需要合并
                merged[-1] = (last_start, max(last_end, end))  # 合并区间
            elif start == last_end + 1 and (end - last_start + 1 < target_length or last_end - last_start + 1 < target_length):
                # 相邻区间也需要合并的场景
                merged[-1] = (last_start, end)
            else:
                # 如果没有重叠且都大于目标长度，则直接保留
                merged.append((start, end))
        return merged

    def compute_iou(self, box1, box2):
        box1_polygon = self.sub_area_to_polygon(box1)
        box2_polygon = self.sub_area_to_polygon(box2)
        intersection = box1_polygon.intersection(box2_polygon)
        if intersection.is_empty:
            return -1
        else:
            union_area = (box1_polygon.area + box2_polygon.area - intersection.area)
            if union_area > 0:
                intersection_area_rate = intersection.area / union_area
            else:
                intersection_area_rate = 0
            return intersection_area_rate

    def get_area_max_box_dict(self, sub_frame_no_list_continuous, subtitle_frame_no_box_dict):
        _area_max_box_dict = dict()
        for start_no, end_no in sub_frame_no_list_continuous:
            # 寻找面积最大文本框
            current_no = start_no
            # 查找当前区间矩形框最大面积
            area_max_box_list = []
            while current_no <= end_no:
                for coord in subtitle_frame_no_box_dict[current_no]:
                    # 取出每一个文本框坐标
                    xmin, xmax, ymin, ymax = coord
                    # 计算当前文本框坐标面积
                    current_area = abs(xmax - xmin) * abs(ymax - ymin)
                    # 如果区间最大框列表为空，则当前面积为区间最大面积
                    if len(area_max_box_list) < 1:
                        area_max_box_list.append({
                            'area': current_area,
                            'xmin': xmin,
                            'xmax': xmax,
                            'ymin': ymin,
                            'ymax': ymax
                        })
                    # 如果列表非空，判断当前文本框是与区间最大文本框在同一区域
                    else:
                        has_same_position = False
                        # 遍历每个区间最大文本框，判断当前文本框位置是否与区间最大文本框列表的某个文本框位于同一行且交叉
                        for area_max_box in area_max_box_list:
                            if (area_max_box['ymin'] - config.THRESHOLD_HEIGHT_DIFFERENCE <= ymin
                                    and ymax <= area_max_box['ymax'] + config.THRESHOLD_HEIGHT_DIFFERENCE):
                                if self.compute_iou((xmin, xmax, ymin, ymax), (
                                        area_max_box['xmin'], area_max_box['xmax'], area_max_box['ymin'],
                                        area_max_box['ymax'])) != -1:
                                    # 如果高度差异不一样
                                    if abs(abs(area_max_box['ymax'] - area_max_box['ymin']) - abs(
                                            ymax - ymin)) < config.THRESHOLD_HEIGHT_DIFFERENCE:
                                        has_same_position = True
                                    # 如果在同一行，则计算当前面积是不是最大
                                    # 判断面积大小，若当前面积更大，则将当前行的最大区域坐标点更新
                                    if has_same_position and current_area > area_max_box['area']:
                                        area_max_box['area'] = current_area
                                        area_max_box['xmin'] = xmin
                                        area_max_box['xmax'] = xmax
                                        area_max_box['ymin'] = ymin
                                        area_max_box['ymax'] = ymax
                        # 如果遍历了所有的区间最大文本框列表，发现是新的一行，则直接添加
                        if not has_same_position:
                            new_large_area = {
                                'area': current_area,
                                'xmin': xmin,
                                'xmax': xmax,
                                'ymin': ymin,
                                'ymax': ymax
                            }
                            if new_large_area not in area_max_box_list:
                                area_max_box_list.append(new_large_area)
                                break
                current_no += 1
            _area_max_box_list = list()
            for area_max_box in area_max_box_list:
                if area_max_box not in _area_max_box_list:
                    _area_max_box_list.append(area_max_box)
            _area_max_box_dict[f'{start_no}->{end_no}'] = _area_max_box_list
        return _area_max_box_dict

    def get_subtitle_frame_no_box_dict_with_united_coordinates(self, subtitle_frame_no_box_dict):
        """
        将多个视频帧的文本区域坐标统一
        """
        subtitle_frame_no_box_dict_with_united_coordinates = dict()
        frame_no_list = self.find_continuous_ranges_with_same_mask(subtitle_frame_no_box_dict)
        area_max_box_dict = self.get_area_max_box_dict(frame_no_list, subtitle_frame_no_box_dict)
        for start_no, end_no in frame_no_list:
            current_no = start_no
            while True:
                area_max_box_list = area_max_box_dict[f'{start_no}->{end_no}']
                current_boxes = subtitle_frame_no_box_dict[current_no]
                new_subtitle_frame_no_box_list = []
                for current_box in current_boxes:
                    current_xmin, current_xmax, current_ymin, current_ymax = current_box
                    for max_box in area_max_box_list:
                        large_xmin = max_box['xmin']
                        large_xmax = max_box['xmax']
                        large_ymin = max_box['ymin']
                        large_ymax = max_box['ymax']
                        box1 = (current_xmin, current_xmax, current_ymin, current_ymax)
                        box2 = (large_xmin, large_xmax, large_ymin, large_ymax)
                        res = self.compute_iou(box1, box2)
                        if res != -1:
                            new_subtitle_frame_no_box = (large_xmin, large_xmax, large_ymin, large_ymax)
                            if new_subtitle_frame_no_box not in new_subtitle_frame_no_box_list:
                                new_subtitle_frame_no_box_list.append(new_subtitle_frame_no_box)
                subtitle_frame_no_box_dict_with_united_coordinates[current_no] = new_subtitle_frame_no_box_list
                current_no += 1
                if current_no > end_no:
                    break
        return subtitle_frame_no_box_dict_with_united_coordinates

    def prevent_missed_detection(self, subtitle_frame_no_box_dict):
        """
        添加额外的文本框，防止漏检
        """
        frame_no_list = self.find_continuous_ranges_with_same_mask(subtitle_frame_no_box_dict)
        for start_no, end_no in frame_no_list:
            current_no = start_no
            while True:
                current_box_list = subtitle_frame_no_box_dict[current_no]
                if current_no + 1 != end_no and (current_no + 1) in subtitle_frame_no_box_dict.keys():
                    next_box_list = subtitle_frame_no_box_dict[current_no + 1]
                    if set(current_box_list).issubset(set(next_box_list)):
                        subtitle_frame_no_box_dict[current_no] = subtitle_frame_no_box_dict[current_no + 1]
                current_no += 1
                if current_no > end_no:
                    break
        return subtitle_frame_no_box_dict

    @staticmethod
    def get_frequency_in_range(sub_frame_no_list_continuous, subtitle_frame_no_box_dict):
        sub_area_with_frequency = {}
        for start_no, end_no in sub_frame_no_list_continuous:
            current_no = start_no
            while True:
                current_box_list = subtitle_frame_no_box_dict[current_no]
                for current_box in current_box_list:
                    if str(current_box) not in sub_area_with_frequency.keys():
                        sub_area_with_frequency[f'{current_box}'] = 1
                    else:
                        sub_area_with_frequency[f'{current_box}'] += 1
                current_no += 1
                if current_no > end_no:
                    break
        return sub_area_with_frequency

    def filter_mistake_sub_area(self, subtitle_frame_no_box_dict, fps):
        """
        过滤错误的字幕区域
        """
        sub_frame_no_list_continuous = self.find_continuous_ranges_with_same_mask(subtitle_frame_no_box_dict)
        sub_area_with_frequency = self.get_frequency_in_range(sub_frame_no_list_continuous, subtitle_frame_no_box_dict)
        correct_sub_area = []
        for sub_area in sub_area_with_frequency.keys():
            if sub_area_with_frequency[sub_area] >= (fps // 2):
                correct_sub_area.append(sub_area)
            else:
                print(f'drop {sub_area}')
        correct_subtitle_frame_no_box_dict = dict()
        for frame_no in subtitle_frame_no_box_dict.keys():
            current_box_list = subtitle_frame_no_box_dict[frame_no]
            new_box_list = []
            for current_box in current_box_list:
                if str(current_box) in correct_sub_area and current_box not in new_box_list:
                    new_box_list.append(current_box)
            correct_subtitle_frame_no_box_dict[frame_no] = new_box_list
        return correct_subtitle_frame_no_box_dict


# 配置日志
def setup_logger():
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    log_file = f'subtitle_remover_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('SubtitleRemover')

class SubtitleRemover:
    def __init__(self, video_path, subtitle_area, preview=False, progress_callback=None):
        importlib.reload(config)
        # 设置日志记录器
        self.logger = setup_logger()
        self.logger.info(f"初始化字幕去除器，处理文件: {video_path}")
        self.logger.info(f"使用配置: MODE={config.MODE}, SKIP_TEXT_DETECTION={config.SKIP_TEXT_DETECTION}")
        
        # 记录开始时间
        self.start_time = time.time()
        self.step_times = {}
        
        # 线程锁
        self.lock = threading.RLock()
        self.video_path = video_path
        self.sub_area = subtitle_area
        self.gui_mode = preview
        self.progress_callback = progress_callback
        
        # 记录字幕区域
        if subtitle_area:
            self.logger.info(f"指定字幕区域: {subtitle_area}")
        
        # 判断是否为图片
        self.is_picture = False
        if is_image_file(str(video_path)):
            self.sub_area = None
            self.is_picture = True
        # 视频路径
        self.video_path = video_path
        self.video_cap = cv2.VideoCapture(video_path)
        # 通过视频路径获取视频名称
        self.vd_name = Path(self.video_path).stem
        # 视频帧总数
        self.frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5)
        # 视频帧率
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        # 视频尺寸
        self.size = (int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.mask_size = (int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # 创建字幕检测对象
        self.sub_detector = SubtitleDetect(self.video_path, self.sub_area)
        # 创建视频临时对象，windows下delete=True会有permission denied的报错
        self.video_temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        # 创建视频写对象 - 修改编码器为更兼容的选项
        if platform.system() == 'Darwin':  # macOS
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264编码
        else:  # Windows和Linux
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MPEG-4编码
        
        self.video_writer = cv2.VideoWriter(
            self.video_temp_file.name, 
            fourcc, 
            self.fps, 
            self.size
        )
        
        # 检查视频写入器是否成功初始化
        if not self.video_writer.isOpened():
            print("警告：视频写入器初始化失败，尝试备用编码器")
            # 尝试备用编码器
            backup_fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_temp_file = tempfile.NamedTemporaryFile(suffix='.avi', delete=False)
            self.video_writer = cv2.VideoWriter(
                self.video_temp_file.name,
                backup_fourcc,
                self.fps,
                self.size
            )
            if not self.video_writer.isOpened():
                print("错误：所有视频写入器初始化都失败")
        
        # 获取模型名称用于输出文件命名
        model_name = self._get_model_name()
        
        self.video_out_name = os.path.join(os.path.dirname(self.video_path), f'{self.vd_name}_no_sub_{model_name}.mp4')
        self.video_inpaint = None
        self.lama_inpaint = None
        self.ext = os.path.splitext(video_path)[-1]
        if self.is_picture:
            pic_dir = os.path.join(os.path.dirname(self.video_path), 'no_sub')
            if not os.path.exists(pic_dir):
                os.makedirs(pic_dir)
            self.video_out_name = os.path.join(pic_dir, f'{self.vd_name}_no_sub_{model_name}{self.ext}')
        if torch.cuda.is_available():
            print('use GPU for acceleration')
        # 总处理进度
        self.progress_total = 0
        self.progress_remover = 0
        self.isFinished = False
        # 预览帧
        self.preview_frame = None
        # 是否将原音频嵌入到去除字幕后的视频
        self.is_successful_merged = False

    def _get_model_name(self):
        """根据当前配置获取模型名称"""
        if config.MODE == config.InpaintMode.PROPAINTER:
            return "propainter"
        elif config.MODE == config.InpaintMode.STTN:
            return "sttn"
        else:
            if config.LAMA_SUPER_FAST:
                return "lama_fast"
            else:
                return "lama"

    def _time_it(self, step_name):
        """记录步骤耗时"""
        current_time = time.time()
        if hasattr(self, 'last_step_time'):
            elapsed = current_time - self.last_step_time
            self.step_times[step_name] = elapsed
            self.logger.info(f"步骤 '{step_name}' 耗时: {elapsed:.2f}秒")
        self.last_step_time = current_time

    def update_progress(self, current, total, desc=""):
        """更新进度的统一接口"""
        if self.progress_callback:
            self.progress_callback({
                'current': current,
                'total': total,
                'desc': desc
            })
        else:
            # 非GUI模式下使用tqdm
            print(f"{desc}: {current}/{total}")

    def propainter_mode(self):
        print('use propainter mode')
        sub_list = self.sub_detector.find_subtitle_frame_no(sub_remover=self)
        continuous_frame_no_list = self.sub_detector.find_continuous_ranges_with_same_mask(sub_list)
        scene_div_points = self.sub_detector.get_scene_div_frame_no(self.video_path)
        continuous_frame_no_list = self.sub_detector.split_range_by_scene(continuous_frame_no_list,
                                                                          scene_div_points)
        self.video_inpaint = VideoInpaint(config.PROPAINTER_MAX_LOAD_NUM)
        print('[Processing] start removing subtitles...')
        index = 0
        while True:
            ret, frame = self.video_cap.read()
            if not ret:
                break
            index += 1
            # 如果当前帧没有水印/文本则直接写
            if index not in sub_list.keys():
                self.video_writer.write(frame)
                print(f'write frame: {index}')
                self.update_progress(index, len(sub_list), "Processing")
                continue
            # 如果有水印，判断该帧是不是开头帧
            else:
                # 如果是开头帧，则批推理到尾帧
                if self.is_current_frame_no_start(index, continuous_frame_no_list):
                    # print(f'No 1 Current index: {index}')
                    start_frame_no = index
                    print(f'find start: {start_frame_no}')
                    # 找到结束帧
                    end_frame_no = self.find_frame_no_end(index, continuous_frame_no_list)
                    # 判断当前帧号是不是字幕起始位置
                    # 如果获取的结束帧号不为-1则说明
                    if end_frame_no != -1:
                        print(f'find end: {end_frame_no}')
                        # ************ 读取该区间所有帧 start ************
                        temp_frames = list()
                        # 将头帧加入处理列表
                        temp_frames.append(frame)
                        inner_index = 0
                        # 一直读取到尾帧
                        while index < end_frame_no:
                            ret, frame = self.video_cap.read()
                            if not ret:
                                break
                            index += 1
                            temp_frames.append(frame)
                        # ************ 读取该区间所有帧 end ************
                        if len(temp_frames) < 1:
                            # 没有待处理，直接跳过
                            continue
                        elif len(temp_frames) == 1:
                            inner_index += 1
                            single_mask = create_mask(self.mask_size, sub_list[index])
                            if self.lama_inpaint is None:
                                self.lama_inpaint = LamaInpaint()
                            inpainted_frame = self.lama_inpaint(frame, single_mask)
                            self.video_writer.write(inpainted_frame)
                            print(f'write frame: {start_frame_no + inner_index} with mask {sub_list[start_frame_no]}')
                            self.update_progress(start_frame_no + inner_index, len(sub_list), "Processing")
                            continue
                        else:
                            # 将读取的视频帧分批处理
                            # 1. 获取当前批次使用的mask
                            mask = create_mask(self.mask_size, sub_list[start_frame_no])
                            for batch in batch_generator(temp_frames, config.PROPAINTER_MAX_LOAD_NUM):
                                # 2. 调用批推理
                                if len(batch) == 1:
                                    single_mask = create_mask(self.mask_size, sub_list[start_frame_no])
                                    if self.lama_inpaint is None:
                                        self.lama_inpaint = LamaInpaint()
                                    inpainted_frame = self.lama_inpaint(frame, single_mask)
                                    self.video_writer.write(inpainted_frame)
                                    print(f'write frame: {start_frame_no + inner_index} with mask {sub_list[start_frame_no]}')
                                    inner_index += 1
                                    self.update_progress(start_frame_no + inner_index, len(sub_list), "Processing")
                                elif len(batch) > 1:
                                    inpainted_frames = self.video_inpaint.inpaint(batch, mask)
                                    for i, inpainted_frame in enumerate(inpainted_frames):
                                        self.video_writer.write(inpainted_frame)
                                        print(f'write frame: {start_frame_no + inner_index} with mask {sub_list[index]}')
                                        inner_index += 1
                                        if self.gui_mode:
                                            self.preview_frame = cv2.hconcat([batch[i], inpainted_frame])
                                self.update_progress(start_frame_no + inner_index, len(sub_list), "Processing")

    def sttn_mode_with_no_detection(self):
        """使用sttn对选中区域进行重绘，不进行字幕检测"""
        print('use sttn mode with no detection')
        print('[Processing] start removing subtitles...')
        
        if self.sub_area is not None:
            ymin, ymax, xmin, xmax = self.sub_area
        else:
            print('[Info] No subtitle area has been set. Video will be processed in full screen.')
            ymin, ymax, xmin, xmax = 0, self.frame_height, 0, self.frame_width
            
        mask_area_coordinates = [(xmin, xmax, ymin, ymax)]
        mask = create_mask(self.mask_size, mask_area_coordinates)
        
        # 创建STTN视频修复实例
        sttn_video_inpaint = STTNVideoInpaint(self.video_path)
        
        # 处理视频
        current_frame = 0
        total_frames = self.frame_count
        
        def progress_update(frame_num):
            nonlocal current_frame
            current_frame = frame_num
            self.update_progress(current_frame, total_frames, "Processing")
            self.progress_total = (100 * float(current_frame) / float(total_frames))
        
        # 调用STTN处理，传入进度更新回调
        sttn_video_inpaint(
            input_mask=mask,
            input_sub_remover=self,
            progress_callback=progress_update
        )

    def sttn_mode(self):
        """STTN模式的入口方法"""
        # 移除tbar参数，因为我们现在使用progress_callback
        self.sttn_mode_with_no_detection()

    def lama_mode(self):
        self.logger.info("使用LAMA模式进行字幕去除")
        self._time_it("初始化")
        
        # 检查是否使用用户指定的字幕区域
        if self.sub_area is not None and config.SKIP_TEXT_DETECTION:
            self.logger.info('使用用户指定的字幕区域，跳过文本检测')
            # 创建一个包含所有帧的字典，每帧都使用相同的字幕区域
            sub_list = {}
            # 获取视频总帧数
            total_frames = int(self.frame_count)
            # 将用户指定的区域应用到所有帧
            for i in range(1, total_frames + 1):
                # 将sub_area转换为[(xmin, xmax, ymin, ymax)]格式
                s_ymin, s_ymax, s_xmin, s_xmax = self.sub_area
                sub_list[i] = [(s_xmin, s_xmax, s_ymin, s_ymax)]
            self.logger.info(f"为{total_frames}帧创建了字幕区域映射")
        else:
            # 原有的自动文本检测逻辑
            self.logger.info("开始自动文本检测")
            detect_start = time.time()
            sub_list = self.sub_detector.find_subtitle_frame_no(sub_remover=self)
            self.logger.info(f"文本检测完成，检测到{len(sub_list)}帧包含字幕，耗时: {time.time() - detect_start:.2f}秒")
        
        self._time_it("文本检测")
        
        if self.lama_inpaint is None:
            self.logger.info("初始化LAMA修复模型")
            model_start = time.time()
            self.lama_inpaint = LamaInpaint()
            self.logger.info(f"LAMA模型加载完成，耗时: {time.time() - model_start:.2f}秒")
        
        self._time_it("模型加载")
        
        index = 0
        frames_with_subtitle = 0
        inpaint_time_total = 0
        
        self.logger.info('[Processing] 开始去除字幕...')
        
        while True:
            ret, frame = self.video_cap.read()
            if not ret:
                break
            original_frame = frame.copy()
            index += 1
            
            if index in sub_list.keys():
                frames_with_subtitle += 1
                mask = create_mask(self.mask_size, sub_list[index])
                
                inpaint_start = time.time()
                if config.LAMA_SUPER_FAST:
                    frame = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
                else:
                    frame = self.lama_inpaint(frame, mask)
                inpaint_time = time.time() - inpaint_start
                inpaint_time_total += inpaint_time
                
                if index % 100 == 0:
                    self.logger.info(f"处理第{index}帧，修复耗时: {inpaint_time:.4f}秒")
            
            if self.gui_mode:
                self.preview_frame = cv2.hconcat([original_frame, frame])
            
            if self.is_picture:
                cv2.imencode(self.ext, frame)[1].tofile(self.video_out_name)
            else:
                self.video_writer.write(frame)
            
            self.update_progress(index, self.frame_count, "Processing")
            self.progress_remover = 100 * float(index) / float(self.frame_count) // 2
            self.progress_total = 50 + self.progress_remover
        
        self._time_it("帧处理")
        
        avg_inpaint_time = inpaint_time_total / frames_with_subtitle if frames_with_subtitle > 0 else 0
        self.logger.info(f"处理完成: 总帧数={index}, 包含字幕帧数={frames_with_subtitle}")
        self.logger.info(f"平均每帧修复耗时: {avg_inpaint_time:.4f}秒")

    def merge_audio_to_video(self):
        # 创建音频临时对象，windows下delete=True会有permission denied的报错
        temp = tempfile.NamedTemporaryFile(suffix='.aac', delete=False)
        audio_extract_command = [config.FFMPEG_PATH,
                                 "-y", "-i", self.video_path,
                                 "-acodec", "copy",
                                 "-vn", "-loglevel", "error", temp.name]
        use_shell = True if os.name == "nt" else False
        try:
            subprocess.check_output(audio_extract_command, stdin=open(os.devnull), shell=use_shell)
        except Exception as e:
            print(f'音频提取失败: {str(e)}')
            return
        else:
            if os.path.exists(self.video_temp_file.name) and os.path.getsize(self.video_temp_file.name) > 0:
                # 确保视频写入器已关闭
                if self.video_writer and self.video_writer.isOpened():
                    self.video_writer.release()
                    
                # 使用ffmpeg直接合并视频和音频
                audio_merge_command = [
                    config.FFMPEG_PATH,
                    "-y", 
                    "-i", self.video_temp_file.name,
                    "-i", temp.name,
                    "-c:v", "copy",  # 直接复制视频流，不重新编码
                    "-c:a", "aac",   # 使用AAC编码音频
                    "-strict", "experimental",
                    "-shortest",     # 使用最短的流长度
                    "-loglevel", "error", 
                    self.video_out_name
                ]
                
                try:
                    subprocess.check_output(audio_merge_command, stdin=open(os.devnull), shell=use_shell)
                    if os.path.exists(self.video_out_name) and os.path.getsize(self.video_out_name) > 0:
                        self.is_successful_merged = True
                        print(f'音频合并成功，输出文件: {self.video_out_name}')
                    else:
                        print('音频合并失败，输出文件为空或不存在')
                        # 尝试不同的合并方法
                        self._try_alternative_merge(temp.name)
                except Exception as e:
                    print(f'音频合并失败: {str(e)}')
                    # 尝试不同的合并方法
                    self._try_alternative_merge(temp.name)
            else:
                print('临时视频文件不存在或为空，无法合并音频')
            
            # 清理临时音频文件
            if os.path.exists(temp.name):
                try:
                    os.remove(temp.name)
                except Exception:
                    if platform.system() != 'Windows':
                        print(f'无法删除临时文件 {temp.name}')
                
            # 如果所有合并方法都失败，直接复制临时视频文件
            if not self.is_successful_merged and os.path.exists(self.video_temp_file.name) and os.path.getsize(self.video_temp_file.name) > 0:
                try:
                    print('音频合并失败，尝试直接复制视频文件')
                    shutil.copy2(self.video_temp_file.name, self.video_out_name)
                    if os.path.exists(self.video_out_name) and os.path.getsize(self.video_out_name) > 0:
                        print(f'视频文件复制成功: {self.video_out_name}')
                except IOError as e:
                    print(f"无法复制文件: {str(e)}")
            
            # 关闭临时文件
            self.video_temp_file.close()

    def _try_alternative_merge(self, audio_file):
        """尝试替代方法合并音频和视频"""
        try:
            # 尝试使用不同的编码参数
            alt_merge_command = [
                config.FFMPEG_PATH,
                "-y", 
                "-i", self.video_temp_file.name,
                "-i", audio_file,
                "-c:v", "libx264",  # 使用H.264重新编码视频
                "-c:a", "aac",      # 使用AAC编码音频
                "-strict", "experimental",
                "-b:a", "192k",     # 设置音频比特率
                "-shortest",        # 使用最短的流长度
                "-loglevel", "error", 
                self.video_out_name
            ]
            
            use_shell = True if os.name == "nt" else False
            subprocess.check_output(alt_merge_command, stdin=open(os.devnull), shell=use_shell)
            
            if os.path.exists(self.video_out_name) and os.path.getsize(self.video_out_name) > 0:
                self.is_successful_merged = True
                print(f'使用替代方法成功合并音频，输出文件: {self.video_out_name}')
            else:
                print('替代合并方法失败')
        except Exception as e:
            print(f'替代合并方法失败: {str(e)}')

    def run(self):
        self.logger.info("开始执行字幕去除任务")
        start_time = time.time()
        self.progress_total = 0
        
        try:
            if self.is_picture:
                self._process_image()
            else:
                if config.MODE == config.InpaintMode.PROPAINTER:
                    self.propainter_mode()
                elif config.MODE == config.InpaintMode.STTN:
                    self.sttn_mode()
                else:
                    self.lama_mode()
                
                if not self.is_picture:
                    merge_start = time.time()
                    self.merge_audio_to_video()
                    self.logger.info(f"音频合并耗时: {time.time() - merge_start:.2f}秒")
                    self.logger.info(f"[Finished] 字幕成功去除，视频生成于: {self.video_out_name}")
            
            total_time = time.time() - start_time
            self.logger.info(f'总耗时: {total_time:.2f}秒')
            
            # 打印各步骤耗时统计
            self.logger.info("各步骤耗时统计:")
            for step, elapsed in self.step_times.items():
                self.logger.info(f"  - {step}: {elapsed:.2f}秒 ({elapsed/total_time*100:.1f}%)")
            
        except Exception as e:
            self.logger.error(f"处理过程中出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
        finally:
            self.cleanup()
            self.isFinished = True
            self.progress_total = 100

    def cleanup(self):
        """清理资源"""
        if self.video_cap:
            self.video_cap.release()
        if self.video_writer:
            self.video_writer.release()
        if os.path.exists(self.video_temp_file.name):
            try:
                os.remove(self.video_temp_file.name)
            except Exception:
                if platform.system() != 'Windows':
                    print(f'Failed to delete temp file {self.video_temp_file.name}')

    def _process_image(self):
        sub_list = self.sub_detector.find_subtitle_frame_no(sub_remover=self)
        self.lama_inpaint = LamaInpaint()
        original_frame = cv2.imread(self.video_path)
        if len(sub_list):
            mask = create_mask(original_frame.shape[0:2], sub_list[1])
            inpainted_frame = self.lama_inpaint(original_frame, mask)
        else:
            inpainted_frame = original_frame
        if self.gui_mode:
            self.preview_frame = cv2.hconcat([original_frame, inpainted_frame])
        cv2.imencode(self.ext, inpainted_frame)[1].tofile(self.video_out_name)
        self.update_progress(1, 1, "Processing")

    @staticmethod
    def get_coordinates(dt_box):
        """
        从返回的检测框中获取坐标
        :param dt_box 检测框返回结果
        :return list 坐标点列表
        """
        coordinate_list = list()
        if isinstance(dt_box, list):
            for i in dt_box:
                i = list(i)
                (x1, y1) = int(i[0][0]), int(i[0][1])
                (x2, y2) = int(i[1][0]), int(i[1][1])
                (x3, y3) = int(i[2][0]), int(i[2][1])
                (x4, y4) = int(i[3][0]), int(i[3][1])
                xmin = max(x1, x4)
                xmax = min(x2, x3)
                ymin = max(y1, y2)
                ymax = min(y3, y4)
                coordinate_list.append((xmin, xmax, ymin, ymax))
        return coordinate_list

    @staticmethod
    def is_current_frame_no_start(frame_no, continuous_frame_no_list):
        """
        判断给定的帧号是否为开头，是的话返回结束帧号，不是的话返回-1
        """
        for start_no, end_no in continuous_frame_no_list:
            if start_no == frame_no:
                return True
        return False

    @staticmethod
    def find_frame_no_end(frame_no, continuous_frame_no_list):
        """
        判断给定的帧号是否为开头，是的话返回结束帧号，不是的话返回-1
        """
        for start_no, end_no in continuous_frame_no_list:
            if start_no <= frame_no <= end_no:
                return end_no
        return -1


if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
    # 1. 提示用户输入视频路径
    video_path = input(f"Please input video or image file path: ").strip()
    # 判断视频路径是不是一个目录，是目录的化，批量处理该目录下的所有视频文件
    # 2. 按以下顺序传入字幕区域
    subtitle_area = None
    area_input = input("是否需要指定字幕区域? (y/n): ").strip().lower()
    if area_input == 'y':
        try:
            print("请按以下格式输入字幕区域坐标 (ymin ymax xmin xmax):")
            print("例如: 400 450 100 500")
            coords = input().strip().split()
            if len(coords) == 4:
                ymin, ymax, xmin, xmax = map(int, coords)
                subtitle_area = (ymin, ymax, xmin, xmax)
                print(f"设置字幕区域为: {subtitle_area}")
            else:
                print("坐标格式错误，将使用自动检测模式")
        except ValueError:
            print("输入格式错误，将使用自动检测模式")
    # 3. 新建字幕提取对象
    if is_video_or_image(video_path):
        sd = SubtitleRemover(video_path, subtitle_area)
        sd.run()
    else:
        print(f'Invalid video path: {video_path}')
