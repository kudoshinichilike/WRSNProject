import math

import numpy as np
from scipy.spatial import distance

import Parameter as para
from Node_Method import to_string, find_receiver, request_function, estimate_average_energy


class Node:
    def __init__(self, location=None, com_ran=None, sen_ran=None, energy=None, prob=para.prob, avg_energy=0.0,
                 len_cp=10, id=None, is_active=True, energy_max=None, energy_thresh=None):
        self.location = location  # location of sensor
        self.com_ran = com_ran  # communication range
        self.sen_ran = sen_ran  # sensing range
        self.energy = float(energy)  # energy of sensor
        self.energy_max = energy_max  # capacity of sensor
        self.energy_thresh = energy_thresh  # threshold to sensor send request for mc
        self.prob = prob  # probability of sending data
        self.check_point = [{"E_current": self.energy, "time": 0, "avg_e": 0.0}]  # check point of information of sensor
        self.used_energy = 0.0  # energy was used from last check point to now
        self.just_used_energy = 0.0
        self.avg_energy = avg_energy  # average energy of sensor
        self.len_cp = len_cp  # length of check point list
        self.id = id  # identify of sensor
        self.neighbor = []  # neighborhood of sensor
        self.is_active = is_active  # statement of sensor. If sensor dead, state is False
        self.is_request = False  # if node requested, is_request = True
        self.level = 0  # the distance from node to base

        self.list_request = []  # List sensor energy receive per second form this node > used per second
        self.charging_time = para.sensor_no_charge
        self.charged_energy = 0
        self.charging_energy_original = 0
        self.residual_energy = 0

        self.receive_energy = np.zeros(605, dtype=float)
        self.is_receive_from_sensor = False
        self.list_just_request = []
        self.is_need_update = None
        self.average_used = 0.0
        self.needSetReward = False

    def set_average_energy(self, func=estimate_average_energy):
        """
        calculate average energy of sensor
        :param func: function to calculate
        :return: set value for average energy with estimate function is func
        """
        self.avg_energy = func(self)

    def set_check_point(self, t):
        """
        add new check point in check_point list
        :param t: time stem
        :return: if queue of check point is not full, add new check point
        """
        if len(self.check_point) >= self.len_cp:
            self.check_point.pop(0)
        self.check_point.append(
            {"E_current": self.energy, "time": t, "avg_e": self.used_energy / (t - self.check_point[-1]["time"])})
        self.avg_energy = self.check_point[-1]["avg_e"]
        self.used_energy = 0.0

    def charge(self, mc):
        """
        charging to sensor
        :param mc: mobile charger
        :return: the amount of energy mc charges to this sensor
        """
        if self.energy <= self.energy_thresh + para.delta and mc.is_stand and self.is_active:
            d = distance.euclidean(self.location, mc.current)
            p_theory = para.alpha / (d + para.beta) ** 2
            p_actual = min(self.energy_thresh - self.energy, p_theory)
            self.energy = self.energy + p_actual
            # print("charge by MC", p_actual, self.energy)
            return p_actual
        else:
            return 0

    def charge_by_sensor(self, sensor, time=1):
        """
        charging to self by sensor
        :param sensor: sensor charge to this
        :param time: time charging to this
        :return: the amount of energy mc charges to this sensor
        """
        if self.energy <= self.energy_thresh + para.delta and self.is_active:
            self.is_receive_from_sensor = True
            d = distance.euclidean(self.location, sensor.location)
            p_theory = (para.alpha_sensor / (d + para.beta_sensor) ** 2) * time / 5.0
            p_actual = min(self.energy_thresh - self.energy, p_theory)
            self.energy = self.energy + p_actual
            # print("charge_by_sensor", self.id, p_actual, self.energy)
            return p_actual
        else:
            return 0

    def calE_charge_by_sensor(self, sensor, time=1):
        """
        charging to self by sensor
        :param sensor: sensor charge to this
        :param time: time charging to this
        :return: the amount of energy mc charges to this sensor
        """
        if self.energy <= self.energy_max - para.delta:
            d = distance.euclidean(self.location, sensor.location)
            p_theory = (para.alpha_sensor / (d + para.beta_sensor) ** 2) * time / 5.0
            p_actual = min(self.energy_max - self.energy, p_theory)
            return p_actual
        else:
            return 0

    def send(self, net=None, package=None, receiver=find_receiver, is_energy_info=False):
        """
        send package
        :param package:
        :param net: the network
        :param receiver: the function calculate receiver node
        :param is_energy_info: if this package is energy package, is_energy_info will be true
        :return: send package to the next node and reduce energy of this node
        """
        d0 = math.sqrt(para.EFS / para.EMP)
        package.update_path(self.id)
        if distance.euclidean(self.location, para.base) > self.com_ran:
            receiver_id = receiver(self, net)
            if receiver_id != -1:
                d = distance.euclidean(self.location, net.node[receiver_id].location)
                e_send = para.ET + para.EFS * d ** 2 if d <= d0 else para.ET + para.EMP * d ** 4
                self.energy -= e_send * package.size
                self.used_energy += e_send * package.size
                self.just_used_energy += e_send * package.size
                net.node[receiver_id].receive(package)
                net.node[receiver_id].send(net, package, receiver, is_energy_info)
        else:
            package.is_success = True
            d = distance.euclidean(self.location, para.base)
            e_send = para.ET + para.EFS * d ** 2 if d <= d0 else para.ET + para.EMP * d ** 4
            self.energy -= e_send * package.size
            self.used_energy += e_send * package.size
            self.just_used_energy += e_send * package.size
            package.update_path(-1)
        self.check_active(net)
        # print("send id", self.id, "e_send", e_send, "energy", self.energy)

    def receive(self, package):
        """
        receive package from other node
        :param package: size of package
        :return: reduce energy of this node
        """
        self.energy -= para.ER * package.size
        self.used_energy += para.ER * package.size
        self.just_used_energy += para.ER * package.size

    def check_active(self, net):
        """
        check if the node is alive
        :param net: the network
        :return: None
        """
        if self.energy < 0 or len(self.neighbor) == 0:
            self.is_active = False
        else:
            a = [1 for neighbor in self.neighbor if net.node[neighbor].is_active]
            self.is_active = True if len(a) > 0 else False

    def request(self, network, mc, t, request_func=request_function):
        """
        send a message to mc if the energy is below a threshold
        :param network:
        :param mc: mobile charger
        :param t: time to send request
        :param request_func: structure of message
        :return: None
        """
        self.set_check_point(t)
        # print(self.check_point)
        if not self.is_request:
            request_func(self, network, mc, t)
            self.is_request = True

    def print_node(self, func=to_string):
        """
        print node information
        :param func: print function
        :return:
        """
        func(self)

    def get_percent_residual_energy(self):
        if self.energy > self.energy_thresh:
            return round((self.energy - self.energy_thresh) / self.energy_max * 100)

        return 0

    def get_percent_lack_energy(self):
        if self.energy < self.energy_thresh:
            return round((self.energy_thresh - self.energy) / self.energy_max * 100)

        return 0

    def get_lack_energy(self):
        if self.energy < self.energy_thresh:
            return self.energy_thresh - self.energy

        return 0

    def get_energy_max(self):
        return self.energy_max

    def get_percent_lack_sensitive_sensors(self):
        sum_lack_energy = 0
        sum_energy_max = 0
        for sensor in self.list_request:
            sum_lack_energy += sensor.get_lack_energy()
            sum_energy_max += sensor.get_energy_max()

        if sum_energy_max:
            return round(sum_lack_energy / sum_energy_max * 100)
        else:
            return 0

    def get_weight(self, network, all_path):
        self.weight = 0

        for path in all_path:
            if self.id in path:
                self.weight += 1

        return self.weight

    def is_lack_energy(self):
        # if self.energy < self.energy_thresh:
        # print("Node is_lack_energy", self.energy < self.energy_thresh)
        return self.energy < self.energy_thresh

    def charge_to_another_sensor(self, time):
        """
        mc charge per second in network
        :param time: time need to charge
        :return:
        """
        self.needSetReward = True
        for sensor in self.list_request:
            if self.charged_energy + sensor.calE_charge_by_sensor(self, time) > self.residual_energy:
                continue

            charged_energy = sensor.charge_by_sensor(self, time)
            self.energy -= charged_energy
            self.charged_energy += charged_energy
            sensor.receive_energy[self.id] += charged_energy
            if charged_energy != 0:
                self.needSetReward = False

        self.charging_time -= time

    def get_residual_energy(self):
        if self.energy > self.energy_thresh:
            return self.energy - self.energy_thresh

        return 0

    def get_time_charging(self, action):
        self.charging_energy_original = self.get_residual_energy() / 100.0 * action

        charging_time_list = []

        for sensitive_effect_sensor in self.list_request:
            energy_need_charge = sensitive_effect_sensor.energy_thresh - sensitive_effect_sensor.energy
            energy_charge_per_second = sensitive_effect_sensor.calE_charge_by_sensor(self)
            charging_time_list.append([energy_need_charge / energy_charge_per_second, energy_charge_per_second])

        charging_time_list = sorted(charging_time_list, key=lambda x: x[1], reverse=True)
        energy_charge = np.zeros(len(charging_time_list), dtype=float)

        for idx in range(len(charging_time_list)):
            if not idx == 0:
                energy_charge[idx] = energy_charge[idx - 1] + charging_time_list[idx][1]
            else:
                energy_charge[idx] = charging_time_list[idx][1]

        for idx in range(len(charging_time_list) - 1, -1, -1):
            if idx != 0 and (charging_time_list[idx][0] - para.delta <= charging_time_list[idx - 1][0] <= charging_time_list[idx][0] + para.delta):
                continue

            energy_charge_sum = energy_charge[idx] * charging_time_list[idx][0]
            if energy_charge_sum >= self.charging_energy_original - para.delta:
                energy_charge_sum = 0 if idx == (len(charging_time_list) - 1) else energy_charge[idx + 1] * charging_time_list[idx + 1][0]
                self.charging_time = 0 if idx == (len(charging_time_list) - 1) else charging_time_list[idx + 1][0]
                energy_charge_sum = self.charging_energy_original - energy_charge_sum
                self.charging_time += energy_charge_sum / energy_charge[idx]
                return self.charging_time

        self.charging_time = charging_time_list[0][0]
        return self.charging_time

    def update_energy_thresh(self):
        self.average_used = self.just_used_energy / 20
        self.energy_thresh = self.average_used * 2000
        print("update_energy_thresh", self.energy_thresh)
