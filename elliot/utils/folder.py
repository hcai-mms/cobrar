"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import os
import shutil
import datetime


def manage_directories(path_output_rec_result, path_output_rec_weight, path_output_rec_performance, path_output_performance_user_level):
    if os.path.exists(path_output_rec_result):
        return
    os.makedirs(path_output_rec_result)

    if os.path.exists(path_output_rec_weight):
        return
    os.makedirs(path_output_rec_weight)

    if os.path.exists(path_output_rec_performance):
        return
    os.makedirs(path_output_rec_performance)

    if os.path.exists(path_output_performance_user_level):
        return
    os.makedirs(path_output_performance_user_level)
    # if os.path.exists(os.path.dirname(path_output_rec_result)):
    #     return
    # os.makedirs(os.path.dirname(path_output_rec_result))
    #
    # if os.path.exists(os.path.dirname(path_output_rec_weight)):
    #     return
    # os.makedirs(os.path.dirname(path_output_rec_weight))
    #
    # if os.path.exists(os.path.dirname(path_output_rec_performance)):
    #     return
    # os.makedirs(os.path.dirname(path_output_rec_performance))


def build_model_folder(path_output_rec_weight, model):
    if not os.path.exists(os.path.abspath(os.sep.join([path_output_rec_weight, model]))):
        try:
            os.makedirs(os.path.abspath(os.sep.join([path_output_rec_weight, model])))
        except OSError as e:
            print("model folder too long! shortening it with timestamp...be aware.")
            os.makedirs(os.path.abspath(os.sep.join([path_output_rec_weight, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")])))
    # if not os.path.exists(os.path.dirname(f'{path_output_rec_weight}{model}/')):
    #     os.makedirs(os.path.dirname(f'{path_output_rec_weight}{model}/'))


def build_log_folder(path_log_folder):
    if not os.path.exists(os.path.abspath(path_log_folder)):
        os.makedirs(os.path.abspath(path_log_folder))


def create_folder_by_index(path, index):
    if os.path.exists(os.path.abspath(os.sep.join([path, index]))):
        shutil.rmtree(os.path.abspath(os.sep.join([path, index])))
    os.makedirs(os.path.abspath(os.sep.join([path, index])))
    return os.path.abspath(os.sep.join([path, index]))
