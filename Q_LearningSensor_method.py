import math
import numpy as np
from scipy.spatial import distance

import Parameter as para
import Fuzzy
from Network_Method import get_all_path
from Node_Method import find_receiver


def reward_function(sensor, network, receive_func=find_receiver):
    """
    calculate reward function
    :param sensor:
    :param network:
    :param receive_func:
    :return: reward
    """
    # get_weight
    all_path = get_all_path(network)
    weight_sensor = sensor.get_weight(network, all_path)

    if weight_sensor == 0:
        return para.max_default

    reward = 0
    for node in sensor.list_request:
        reward += node.receive_energy[sensor.id] * node.get_weight(network, all_path) / weight_sensor

    return reward


def init_q_table_function(nb_action=81):
    """
    init q table
    :param nb_action:
    :return:
    """
    q_table = np.zeros((para.state_dimension1 + 5, para.state_dimension2 + 5, para.number_action + 5), dtype=float)
    # for state_sensor in range (101):
    #         q_table[state_sensor][0][0] = 1000

    return q_table


def calc_state_function(sensor):
    percent_lack_sensitive_sensors = sensor.get_percent_lack_sensitive_sensors()
    percent_residual_energy = sensor.get_percent_residual_energy()
    return [percent_residual_energy, percent_lack_sensitive_sensors]
