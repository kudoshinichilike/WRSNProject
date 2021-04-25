import random

import numpy as np
from Node_Method import find_receiver
from Q_LearningSensor_method import init_q_table_function, calc_state_function, reward_function
import Parameter as para


class Q_LearningSensor:
    def __init__(self, sensor, init_q_table_func=init_q_table_function, init_state_func=calc_state_function, alpha=0.5,
                 gamma=0.5):
        self.sensor = sensor
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # scale factor
        self.state = init_state_func(self.sensor)  # current state
        self.action = 0  # current action
        self.q_table = init_q_table_func()  # q table

    def update(self, network, reward_func=reward_function):
        """
        update q table and all attribute of q learning
        :param network:
        :param reward_func:
        :return: charging time at the location, if = -1 mean no charge
        """
        self.state = calc_state_function(self.sensor)
        self.action = self.choose_next_action(self.state)
        # print("update q_sensor", self.sensor.id, self.state, self.action)

        if self.action == 0:
            self.sensor.charging_time = para.sensor_no_charge
        else:
            self.sensor.get_time_charging(self.action)
            self.sensor.charging_energy_original = self.sensor.get_residual_energy() / 100.0 * self.action
            # print("update q_sensor", self.sensor.id, self.state, self.action)
            self.sensor.residual_energy = self.sensor.get_residual_energy()
            self.sensor.charged_energy = 0
            for node in self.sensor.list_request:
                node.receive_energy[self.sensor.id] = 0

    def set_reward(self, reward_func=reward_function, network=None):
        """
        update reward for state and action, goi khi sensor dung sac
        :param reward_func: function to calc reward
        :param network:
        :return:
        """
        state = calc_state_function(self.sensor)
        reward = reward_func(self.sensor, network, receive_func=find_receiver)

        self.action = round(self.sensor.charged_energy / self.sensor.residual_energy * 100)

        # print("set reward action", self.action, self.sensor.charged_energy, self.sensor.residual_energy)
        # print("set reward state", self.state)
        new_value_q = (1 - self.alpha) * self.q_table[self.state[0]][self.state[1]][self.action] + self.alpha * (reward + self.gamma * self.q_max(state))
        # print("new_value_q", new_value_q)
        self.q_table[self.state[0]][self.state[1]][self.action] = new_value_q

        # reset chi so sau khi set reward
        self.sensor.charged_energy = 0
        self.sensor.charging_time_original = 0
        self.sensor.residual_energy = 0
        for node in self.sensor.list_request:
            node.receive_energy[self.sensor.id] = 0

        self.sensor.list_request = []

    def q_max(self, state):
        """
        get q_max of action in state (s t+1)
        :param state: (s t+1)
        :return: q_max
        """
        return np.amax(self.q_table[state[0]][state[1]])

    def choose_next_action(self, state):
        """
        choose next action
        :param state:
        :return: action is percent energy will charge in max q_table
        """
        if state[0] == 0 or state[1] == 0:
            return 0

        if random.uniform(0, 1) < para.epsilon:
            action = round(random.uniform(0, 100))
        else:
            action = np.argmax(self.q_table[state[0]][state[1]])

        return action
