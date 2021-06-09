import math
import random

import numpy as np
from Node_Method import find_receiver
from Q_LearningSensor_method import init_q_table_function, reward_function
import Parameter as para
from Network_Method import get_all_path

class Q_LearningSensor:
    def __init__(self, sensor, init_q_table_func=init_q_table_function, alpha=0.5, gamma=0.5):
        self.sensor = sensor
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # scale factor
        self.state = None  # current state
        self.action = 0  # current action
        self.q_table = init_q_table_func()  # q table
        self.mark = np.zeros((para.state_dimension1 + 2, 12), dtype=float)
        self.energy_share = 0

    def update_charge(self, node):
        """
        update q table and all attribute of q learning
        :param node:
        :return: charging time at the location, if = -1 mean no charge
        """
        self.state = self.calc_state_function(node)
        temp = self.choose_next_action(self.state, node)
        self.action = temp[0]
        self.energy_share = temp[1]

        # print("update_charge", node.id, temp)

        if self.energy_share == 0:
            self.sensor.charging_time = para.sensor_no_charge
            node.request_to_sensor = 0
        else:
            self.sensor.charging_to_sensor = node
            self.sensor.charging_time = self.sensor.get_time_charging(self.energy_share, node)
            self.sensor.charged_energy = 0
            node.request_to_neighbor = -2
            # print("update_charge", self.state, self.sensor.id, "charge_to_another_sensor", node.id, "time", self.sensor.charging_time)

    def set_reward(self, network=None, reward_func=reward_function):
        """
        update reward for state and action, goi khi sensor dung sac
        :param reward_func: function to calc reward
        :param network:
        :return:
        """
        newState = self.calc_state_function(self.sensor.charging_to_sensor)
        reward = reward_func(self.sensor, network)
        new_value_q = (1 - self.alpha) * self.q_table[self.state[0]][self.state[1]][self.action] + self.alpha \
                      * (reward + self.gamma * self.q_max(newState))
        self.q_table[self.state[0]][self.state[1]][self.action] = new_value_q

        # print("set_reward id", self.sensor.id, "state", self.state, "action", self.action, "q_value", new_value_q)

        # reset chi so sau khi set reward
        self.sensor.charging_to_sensor.request_to_sensor = -1
        self.sensor.charging_to_sensor = None
        self.sensor.charging_time = para.sensor_no_charge

    def q_max(self, state):
        """
        get q_max of action in state (s t+1)
        :param state: (s t+1)
        :return: q_max
        """
        return np.amax(self.q_table[state[0]][state[1]])

    def choose_next_action(self, state, node):
        """
        choose next action
        :param state:
        :return: action is percent energy will charge in max q_table
        """
        if self.sensor.get_residual_energy() <= 0:
            return [0, 0]

        eps = (0.95**self.mark[self.state[0]][self.state[1]]) * para.learning_rate0

        # eps = 1

        if random.uniform(0, 1) <= eps:
            residual_energy_i = self.sensor.get_residual_energy()
            t_to_thresh = residual_energy_i / (self.sensor.average_used + node.calE_charge_by_sensor(self.sensor, 1))
            max_energy_can_share = t_to_thresh * node.calE_charge_by_sensor(self.sensor, 1)

            energy_can_receive = node.get_lack_energy()
            energy_share = random.uniform(0, min(max_energy_can_share, energy_can_receive))
            action = round((energy_share / self.sensor.energy_max)*100)
        else:
            action = np.argmax(self.q_table[state[0]][state[1]])
            energy_share = (self.sensor.energy_max/100.0) * (action + random.uniform(0, 0.99999))

        self.mark[self.state[0]][self.state[1]] += 1
        return [action, energy_share]

    def calc_state_function(self, node):
        """
        state0: năng lượng thiếu của j
        staet1: mức năng lượng sử dụng trung bình của j so với i chia theo các khoảng:
        <1/5, 1/4, 1/3, 1/2, 1, 2/1, 3/1, 4/1, 5/1, >5/1
        0,    1,    2,  3,   4,  5,  6,   7,    8,   9
        :param node:
        :return:
        """
        state1 = node.get_percent_lack_energy()

        if self.sensor.average_used == 0:
            balance_j_i = 6
        else:
            balance_j_i = node.average_used / self.sensor.average_used
        if balance_j_i < 1/5:
            state2 = 0
        elif balance_j_i < 1/4:
            state2 = 1
        elif balance_j_i < 1/3:
            state2 = 2
        elif balance_j_i < 1/2:
            state2 = 3
        elif balance_j_i < 1:
            state2 = 4
        elif balance_j_i < 2:
            state2 = 5
        elif balance_j_i < 3:
            state2 = 6
        elif balance_j_i < 4:
            state2 = 7
        elif balance_j_i < 5:
            state2 = 8
        else:
            state2 = 9

        state = [state1, state2]
        return state
