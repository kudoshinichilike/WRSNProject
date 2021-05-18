import math
import numpy as np
from scipy.spatial import distance

import Parameter as para
import Fuzzy
from Network_Method import get_all_path
from Node_Method import find_receiver


def reward_function(sensor, network):
    """
    calculate reward function
    :param sensor:
    :param network:
    :param receive_func:
    :return: reward
    """
    all_path = get_all_path(network)
    weight_sensor = sensor.get_weight(network, all_path)

    if weight_sensor == 0:
        weight_sensor = 0.5
    reward = sensor.charging_to_sensor.get_weight(network, all_path) / weight_sensor

    rang_buoc_i = 0.5
    if sensor.id in network.target:
        rang_buoc_i = 1
    for neighbor_id in sensor.neighbor:
        neighbor = network.node[neighbor_id]
        if neighbor.level > sensor.level and neighbor.is_active and neighbor.neighbor_low_level == 1:
            rang_buoc_i += 1

    rang_buoc_j = 0
    for neighbor_id in sensor.charging_to_sensor.neighbor:
        neighbor = network.node[neighbor_id]
        if neighbor.level > sensor.charging_to_sensor.level and neighbor.is_active and neighbor.neighbor_low_level == 1:
            rang_buoc_j += 1

    # print("reward", reward)
    reward += rang_buoc_j / rang_buoc_i
    reward += sensor.charging_to_sensor.calE_charge_by_sensor(sensor, 1) *100

    if sensor.average_used != 0:
        reward += sensor.charging_to_sensor.average_used / sensor.average_used
        # print(sensor.charging_to_sensor.average_used / sensor.average_used)
    else:
        reward += sensor.charging_to_sensor.average_used * 10000

    # print(rang_buoc_j / rang_buoc_i, sensor.charging_to_sensor.calE_charge_by_sensor(sensor, 1) * 10000)

    if sensor.get_lack_energy() > 0:
        reward -= sensor.get_lack_energy()

    return reward


def init_q_table_function():
    """
    init q table
    :param nb_action:
    :return:
    """
    q_table = np.zeros((para.state_dimension1 + 2, para.state_dimension2 + 2, para.number_action + 2), dtype=float)
    # for state_sensor in range (101):
    #         q_table[state_sensor][0][0] = 1000

    return q_table


def calc_state_function(sensor):
    percent_lack_sensitive_sensors = round(sensor.get_percent_lack_sensitive_sensors() / 2.0)
    percent_residual_energy = round(sensor.get_percent_residual_energy() / 2.0)

    return [percent_residual_energy, percent_lack_sensitive_sensors]
