import copy
import datetime
import logging
import logging.handlers
import os
import platform
import sys

import colorlog
import numpy as np
import yaml


def maybe_create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_platform_specific_value(platform_specific_values):
    if isinstance(platform_specific_values, dict):
        current_platform = platform.system()
        value = platform_specific_values[current_platform]
    else:
        value = platform_specific_values
    return value


def load_config(config_filename: str) -> dict:
    with open(config_filename, 'r', encoding='utf8') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config_filename: str, config: dict):
    maybe_create_path(os.path.dirname(config_filename))
    with open(config_filename, 'w', encoding='utf8') as f:
        yaml.safe_dump(config, f, sort_keys=False)


def set_logger(logging_folder=None, verbose=False, logging_file_prefix=None):
    if verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.getLogger('PIL').setLevel(logging.INFO)  # prevent PIL logging many debug msgs
    logging.getLogger('matplotlib').setLevel(logging.INFO)  # prevent matplotlib logging many debug msgs
    logging.getLogger('pytorch_lightning').setLevel(level)

    # root logger to log everything
    root_logger = logging.root
    root_logger.setLevel(level)
    if not root_logger.handlers:
        format_str = '%(asctime)s [%(threadName)s] %(levelname)s [%(name)s] - %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
        cformat = '%(log_color)s' + format_str
        colors = {'DEBUG': 'cyan',
                  'INFO': 'green',
                  'WARNING': 'bold_yellow',
                  'ERROR': 'red',
                  'CRITICAL': 'bold_red', }
        color_formatter = colorlog.ColoredFormatter(cformat, date_format, log_colors=colors)
        plain_formatter = logging.Formatter(format_str, date_format)
        # Logging to console
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(color_formatter)
        root_logger.addHandler(stream_handler)
        # Logging to file
        if logging_folder is not None:
            maybe_create_path(logging_folder)
            logging_filename = datetime.datetime.now().strftime('%Y-%m-%d#%H-%M-%S') + '.log'
            if logging_file_prefix is not None:
                logging_filename = logging_file_prefix + '_' + logging_filename
            logging_filename = os.path.join(logging_folder, logging_filename)
            file_handler = logging.handlers.RotatingFileHandler(
                logging_filename, maxBytes=5 * 1024 * 1024, encoding='utf8')  # 5MB per file
            file_handler.setFormatter(plain_formatter)
            root_logger.addHandler(file_handler)


def config_merge(src_config: dict, dst_config: dict) -> dict:
    """
    deep merge src_config to dst_config
    :param src_config:
    :param dst_config:
    :return:
    """
    merged_config = copy.deepcopy(dst_config)
    for k, v in src_config.items():
        if k not in dst_config:
            merged_config[k] = copy.deepcopy(v)
        else:
            if isinstance(v, dict):
                merged_config[k] = config_merge(v, dst_config[k])
            else:
                merged_config[k] = copy.deepcopy(v)
    return merged_config


def process_config(ori_config):
    current_tasks_config = copy.deepcopy(ori_config['current_tasks'])
    all_tasks_datasets_config = copy.deepcopy(ori_config['all_tasks_datasets'])
    common_config = copy.deepcopy(ori_config['common'])

    # complete all tasks conf
    for i in range(len(all_tasks_datasets_config)):
        all_tasks_datasets_config[i] = config_merge(all_tasks_datasets_config[i], common_config['dataset'])

    # complete current tasks conf
    output_current_tasks_config = []
    for task_id in current_tasks_config['task_ids']:
        current_task_dataset_conf = None
        for task_dataset_conf in all_tasks_datasets_config:
            if task_dataset_conf['kwargs']['task_id'] == task_id:
                current_task_dataset_conf = task_dataset_conf
        merge_1 = config_merge(current_tasks_config['task_confs'][task_id], {'dataset': current_task_dataset_conf})
        merge_2 = config_merge(merge_1, common_config)
        merge_3 = config_merge(merge_2, {'test_datasets': all_tasks_datasets_config})
        output_current_tasks_config.append(merge_3)
    return output_current_tasks_config


def random_split_samples(num_samples, num_splits, at_least_one=False):
    if at_least_one:
        assert num_splits <= num_samples
        if num_samples == num_splits:
            return [1 for _ in range(num_splits)]
        num_samples -= num_splits
    x = np.round((num_splits - 1) * np.random.random([num_samples])).astype(int)
    num_samples_split = [int(np.sum((x == i).astype(int))) for i in range(num_splits)]
    if at_least_one:
        num_samples_split = [i + 1 for i in num_samples_split]
    return num_samples_split
