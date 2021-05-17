import math

import numpy as np
from scipy.spatial import distance

import Parameter as para
from Node_Method import to_string, find_receiver, request_function, estimate_average_energy, request_to_neighbor_function


class Node:
    def __init__(self, location=None, com_ran=None, sen_ran=None, energy=None, prob=para.prob, avg_energy=0.0,
                 len_cp=10, id=None, is_active=True, energy_max=None, energy_thresh=None):
        self.location = location  # location of sensor
        self.com_ran = com_ran  # communication range
        self.sen_ran = sen_ran  # sensing range
        self.energy = float(energy)  # energy of sensor
        self.energy_max = energy_max  # capacity of sensor
        self.energy_thresh = energy_thresh  # threshold to sensor send request for mc
        self.energy_thresh_weight = energy_thresh  # threshold to sensor send request for mc
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

        self.list_request = []  # list sensor just request
        self.charging_to_sensor = None
        self.charging_time = para.sensor_no_charge
        self.charged_energy = 0
        self.charging_energy_original = 0
        self.residual_energy = 0

        self.charging_from_sensor = False
        self.request_to_sensor = -1 # -2 - dang sac, -1 - haven't request, # time to next request
        self.neighbor_charge = []
        self.optimizer = None
        self.average_used = 0


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
        if self.energy <= self.energy_max and mc.is_stand and self.is_active:
            d = distance.euclidean(self.location, mc.current)
            p_theory = para.alpha / (d + para.beta) ** 2
            p_actual = min(self.energy_max - self.energy, p_theory)
            self.energy = self.energy + p_actual
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
        if self.energy <= self.energy_max + para.delta:
            d = distance.euclidean(self.location, sensor.location)
            p_theory = (para.alpha_sensor / (d + para.beta_sensor) ** 2) * time * para.charge_rate
            p_actual = min(self.energy_max - self.energy, p_theory)
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
            p_theory = (para.alpha_sensor / (d + para.beta_sensor) ** 2) * time * para.charge_rate
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
        if self.energy <= 0 or len(self.neighbor) == 0:
            self.is_active = False
        else:
            a = [1 for neighbor in self.neighbor if net.node[neighbor].is_active]
            self.is_active = True if len(a) > 0 else False

    def request(self, mc, t, request_func=request_function):
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
            request_func(self, mc, t)
            self.is_request = True

    def update_request(self, network, request_to_neighbor_func = request_to_neighbor_function):
        if self.energy <= 0:
            return

        if self.request_to_sensor == 0:
            request_to_neighbor_func(node=self, network=network)
        elif self.request_to_sensor > 0:
            self.request_to_sensor = self.request_to_sensor - 1

    def print_node(self, func=to_string):
        """
        print node information
        :param func: print function
        :return:
        """
        func(self)

    def get_percent_residual_energy(self):
        if self.energy > self.energy_thresh_weight:
            return round((self.energy - self.energy_thresh_weight) / self.energy_max * 100)
        return 0

    def get_residual_energy(self):
        if self.energy > self.energy_thresh_weight:
            return self.energy - self.energy_thresh_weight
        return 0

    def get_percent_lack_energy(self):
        if self.energy < self.energy_thresh_weight:
            return round((self.energy_thresh_weight - self.energy) / self.energy_max * 100)
        return 0

    def get_lack_energy(self):
        if self.energy < self.energy_thresh_weight:
            return self.energy_thresh_weight - self.energy
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
        return self.energy < self.energy_thresh_weight

    def charge_to_another_sensor(self, time):
        """
        mc charge per second in network
        :param time: time need to charge
        :return:
        """
        charged_energy = self.charging_to_sensor.charge_by_sensor(self, time)
        # print("sensor_charge charge_to_another_sensor", charged_energy)
        self.energy -= charged_energy
        self.charged_energy += charged_energy

    def get_time_charging(self, action, node):
        energy = self.energy_max / 100 * action
        time = energy / node.calE_charge_by_sensor(self, 1)
        return time

    def update_energy_thresh(self):
        self.average_used = self.just_used_energy / 20
        self.energy_thresh_weight = min(self.average_used * 1000, self.energy_max*0.8)
