import math
import random

import numpy as np
from Node_Method import find_receiver
from Q_LearningSensor_method import init_q_table_function, calc_state_function, reward_function
import Parameter as para
from Network_Method import get_all_path

class Q_LearningSensor:
    def __init__(self, sensor, init_q_table_func=init_q_table_function, init_state_func=calc_state_function, alpha=0.5,
                 gamma=0.5):
        self.sensor = sensor
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # scale factor
        self.state = init_state_func(self.sensor)  # current state
        self.action = 0  # current action
        self.q_table = init_q_table_func()  # q table
        self.mark = np.zeros((para.state_dimension1 + 2, para.state_dimension2 + 2), dtype=float)

    def update_charge(self, node):
        """
        update q table and all attribute of q learning
        :param node:
        :return: charging time at the location, if = -1 mean no charge
        """
        self.state = self.calc_state_function(node)
        self.action = self.choose_next_action(self.state, node)

        # print("update_charge", self.state, self.action)

        if self.action == 0:
            self.sensor.charging_time = para.sensor_no_charge
            node.request_to_sensor = 0
        else:
            self.sensor.charging_to_sensor = node
            self.sensor.charging_time = self.sensor.get_time_charging(self.action, node)
            self.sensor.charged_energy = 0
            node.request_to_neighbor = -2
            # print("update_charge", self.state, self.sensor.id, "charge_to_another_sensor", node.id, "time", self.sensor.charging_time)

    def get_weight_change(self, node, network):
        all_path = get_all_path(network)
        weight_self = self.sensor.get_weight(network, all_path)
        weight_node = node.get_weight(network, all_path)
        if weight_self == 0:
            return weight_node
        return weight_node / weight_self

    def set_reward(self, network=None, reward_func=reward_function):
        """
        update reward for state and action, goi khi sensor dung sac
        :param reward_func: function to calc reward
        :param network:
        :return:
        """
        newState = self.calc_state_function(self.sensor.charging_to_sensor)
        reward = reward_func(self.sensor, network)
        new_value_q = (1 - self.alpha) * self.q_table[self.state[0]][self.state[1]][self.action] + self.alpha * (reward + self.gamma * self.q_max(newState))
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
            return 0

        if self.mark[self.state[0]][self.state[1]] <= 5:
            eps = 1
        elif self.mark[self.state[0]][self.state[1]] <= 10:
            eps = 0.8
        elif self.mark[self.state[0]][self.state[1]] <= 15:
            eps = 0.5
        else:
            eps = 0.3

        eps = 1

        if random.uniform(0, 1) <= eps:
            residual_energy = self.sensor.get_residual_energy()
            t = residual_energy / (self.sensor.average_used + node.calE_charge_by_sensor(self.sensor, 1))
            thresh1 = math.ceil(t * node.calE_charge_by_sensor(self.sensor, 1) / self.sensor.energy_max * 100)
            thresh1 = min(thresh1, state[0])

            action = round(random.uniform(0, min(thresh1+1, state[1]+1, 100)))
        else:
            action = np.argmax(self.q_table[state[0]][state[1]])

        self.mark[self.state[0]][self.state[1]] += 1
        return action

    def calc_state_function(self, node):
        state = [self.sensor.get_percent_residual_energy(), node.get_percent_lack_energy()]
        return state
