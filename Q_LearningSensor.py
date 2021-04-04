import numpy as np
from Node_Method import find_receiver
from Q_LearningSensor_method import init_q_table_function, calc_state_function, reward_function


class Q_LearningSensor:
    def __init__(self, sensor, init_q_table_func=init_q_table_function, init_state_func=calc_state_function, alpha=0.5, gamma=0.5):
        self.sensor = sensor
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # scale factor
        self.state = init_state_func(self.sensor) # current state
        self.action = 0 # current action
        self.q_table = init_q_table_func(sensor)  # q table

    def update(self, network, reward_func=reward_function):
        """
        update q table and all attribute of q learning
        :param network:
        :param alpha: learning rate
        :param gamma: learning rate
        :param q_max_func:
        :param reward_func:
        :return: charging time at the location, if = -1 mean no charge
        """
        state = calc_state_function(self.sensor)
        self.set_reward(state, reward_func, network)
        action = self.choose_next_action(state)
        self.sensor.get_time_charging(action)
        self.sensor.update_weight(network)
        return self.sensor.charging_time

    def set_reward(self, state, reward_func=reward_function, network=None):
        """
        update reward for state and action, goi khi sensor dung sac
        :param state: state need to update
        :param action: action choose
        :param reward_func: function to calc reward
        :param network:
        :return:
        """
        reward = reward_func(self.sensor, network, receive_func=find_receiver)
        new_value_q = (1 - self.alpha) * self.q_table[self.state[0]][self.state[1]][self.action] + self.alpha * (reward + self.gamma * self.q_max(state))
        # print("new_value_q", new_value_q)
        self.q_table[self.state[0]][self.state[1]][self.action] = new_value_q

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
        action = np.argmax(self.q_table[state[0]][state[1]])
        return action

