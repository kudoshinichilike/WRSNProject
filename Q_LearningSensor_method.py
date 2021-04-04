import math
import numpy as np
from scipy.spatial import distance

import Parameter as para
import Fuzzy
from Node_Method import find_receiver


def reward_function(sensor, network, receive_func=find_receiver):
    """
    calculate reward function
    :param sensor:
    :param network:
    :param energy_charge_to_sensitive:
    :param receive_func:
    :return: reward
    """
    second = para.max_default
    if sensor.weight != 0:
        second = sensor.weight_sensitive / sensor.weight

    reward = second
    return reward


def init_q_table_function(nb_action=81):
    """
    init q table
    :param nb_action:
    :return:
    """
    return np.zeros((para.state_dimension1 + 1, para.state_dimension2 + 1, para.number_action), dtype=float)


def calc_state_function(sensor):
    percent_lack_sensitive_sensors = sensor.get_percent_lack_sensitive_sensors()
    percent_residual_energy = sensor.get_percent_residual_energy()
    return [percent_residual_energy, percent_lack_sensitive_sensors]
