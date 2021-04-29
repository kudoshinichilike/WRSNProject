import csv
from scipy.spatial import distance

import Parameter as para
from Network_Method import uniform_com_func, to_string, count_package_function, get_all_path

write_file_log = "log"
write_name_log = "log/" + write_file_log + ".csv"
open_file_log = open(write_name_log, "w")
result_log = csv.DictWriter(open_file_log, fieldnames=["time", "energy"])
result_log.writeheader()

class Network:
    def __init__(self, list_node=None, mc=None, target=None):
        self.node = list_node  # list of node in network
        self.set_neighbor()  # find neighbor of each node
        self.set_level()  # set the distance in graph from each node to base
        self.mc = mc  # mobile charger
        self.target = target  # list of target. each item is index of sensor where target is located
        self.nb_pack = 0
        self.nb_pack_sent = 0

    def set_neighbor(self):
        """
        find the neighbor of every node
        :return:
        """
        for node in self.node:
            for other in self.node:
                if other.id != node.id and distance.euclidean(node.location, other.location) <= node.com_ran:
                    node.neighbor.append(other.id)

    def set_level(self):
        """
        find the distance in graph from every node to base
        :return:
        """
        queue = []
        for node in self.node:
            if distance.euclidean(node.location, para.base) < node.com_ran:
                node.level = 1
                queue.append(node.id)
        while queue:
            for neighbor_id in self.node[queue[0]].neighbor:
                if not self.node[neighbor_id].level:
                    self.node[neighbor_id].level = self.node[queue[0]].level + 1
                    queue.append(neighbor_id)
            queue.pop(0)

    def communicate(self, communicate_func=uniform_com_func):
        """
        communicate each time in network
        :param communicate_func: the function used to communicating
        :return:
        """
        return communicate_func(self)

    def run_per_second(self, t, optimizer=None, list_optimizer_sensor = None):
        """
        simulate network per second
        :param list_optimizer_sensor:
        :param t: current time
        :param optimizer: the optimizer used to calculate the next location of mc
        :return:
        """

        self.nb_pack = 0
        self.nb_pack_sent = 0

        if t < 200:
            para.epsilon = 0.8
        elif t < 500:
            para.epsilon = 0.5
        elif t < 1000:
            para.epsilon = 0.3
        else:
            para.epsilon = 0.1

        state = self.communicate()

        request_id = []
        for index, node in enumerate(self.node):
            node.is_receive_from_sensor = False
            node.is_need_update = None

            node.list_just_request = []
            if t == 0:
                node.average_used = node.just_used_energy
                node.update_energy_thresh()
                node.just_used_energy = 0.0
            elif t % 20 == 0:
                node.update_energy_thresh()
                node.just_used_energy = 0.0

            if node.energy < node.energy_thresh:
                node.request(network=self, mc=self.mc, t=t)
                request_id.append(index)
            else:
                node.is_request = False

        if request_id:
            # print("request_id", request_id)
            for index, node in enumerate(self.node):
                if index in request_id:
                    if node.list_just_request:
                        # sensor_charge = None
                        # max_charge_E = 0.0
                        for sensor in node.list_just_request:
                            if sensor.is_need_update is not None:
                                sensor.is_need_update.append(node)
                            else:
                                sensor.is_need_update = []
                                sensor.is_need_update.append(node)

                        # for sensor in node.list_just_request:
                        #     print("request_id", max_charge_E, node.calE_charge_by_sensor(sensor))
                        #     if max_charge_E < node.calE_charge_by_sensor(sensor):
                        #         max_charge_E = node.calE_charge_by_sensor(sensor)
                        #         sensor_charge = sensor

                        # if sensor_charge is not None:
                        #     if sensor_charge.is_need_update is not None:
                        #         sensor_charge.is_need_update.append(node)
                        #     else:
                        #         sensor_charge.is_need_update = []
                        #         sensor_charge.is_need_update.append(node)
                        #
                            # for idx in range(len(sensor_charge.is_need_update)):
                            #     print("xxxxx", sensor_charge.id, sensor_charge.is_need_update[idx].id)

                elif index not in request_id and (t - node.check_point[-1]["time"]) > 50:
                    node.set_check_point(t)

        if list_optimizer_sensor:
            for idx, optimizer_sensor in enumerate(list_optimizer_sensor):
                if optimizer_sensor.sensor.is_need_update is not None:
                    # print("is_need_update", idx)
                    if optimizer_sensor.sensor.charging_time != para.sensor_no_charge:
                        optimizer_sensor.set_reward(network=self)

                    for sensor in optimizer_sensor.sensor.is_need_update:
                        optimizer_sensor.sensor.list_request.append(sensor)
                    optimizer_sensor.sensor.is_need_update = None
                    optimizer_sensor.update(self, optimizer.state)

                if optimizer_sensor.sensor.charging_time >= para.delta:
                    optimizer_sensor.sensor.charge_to_another_sensor(1)
                    optimizer_sensor.sensor.charging_time -= 1
                    if optimizer_sensor.sensor.needSetReward:
                        optimizer_sensor.sensor.charging_time = para.sensor_no_charge

                if optimizer_sensor.sensor.charging_time < para.delta and optimizer_sensor.sensor.charging_time != para.sensor_no_charge:
                    optimizer_sensor.set_reward(network=self)
                    optimizer_sensor.sensor.charging_time = para.sensor_no_charge
                    optimizer_sensor.sensor.list_request = []

        if optimizer:
            self.mc.run(network=self, time_stem=t, optimizer=optimizer)

        # for node in self.node:
        #     print("run_per_second", node.id, node.energy)

        return state

    def simulate_lifetime(self, optimizer=None, list_optimizer_sensor = None, file_name="log/energy_log.csv"):
        """
        simulate process finish when energy of any node is less than 0
        :param list_optimizer_sensor:
        :param optimizer:
        :param file_name: log file
        :return:
        """
        energy_log = open(file_name, "w")
        writer = csv.DictWriter(energy_log, fieldnames=["time", "mc energy", "min energy"])
        writer.writeheader()
        t = 0
        while self.node[self.find_min_node()].energy >= 0 and t <= 1000:
            print("simulate_lifetime time", t)
            t = t + 1
            if (t - 1) % 1000 == 0:
                print("simulate_lifetime", t, self.mc.current, self.node[self.find_min_node()].energy)
            state = self.run_per_second(t, optimizer, list_optimizer_sensor)
            # if not (t - 1) % 50:
            #     writer.writerow(
            #         {"time": t, "mc energy": self.mc.energy, "min energy": self.node[self.find_min_node()].energy})
            writer.writerow({"time": t, "mc energy": self.mc.energy, "min energy": self.node[self.find_min_node()].energy})

        energy_log.close()
        return t

    def simulate_max_time(self, optimizer=None, list_optimizer_sensor=None, max_time=50, file_name="log/information_log.csv"):
        """
        simulate process finish when current time is more than the max_time
        :param optimizer:
        :param list_optimizer_sensor:
        :param max_time:
        :param file_name:
        :return:
        """
        information_log = open(file_name, "w")
        # writer = csv.DictWriter(information_log, fieldnames=["time", "nb dead", "nb package"])
        writer = csv.DictWriter(information_log, fieldnames=["time", "mc location", "mc energy", "min energy", "max energy", "max charge", "nb_dead", "nb_pack"])
        writer.writeheader()
        nb_dead = 0
        nb_package = len(self.target)
        t = 0
        while t <= max_time and nb_package > 0:
            print("simulate_max_time time", t)
            t += 1
            state = self.run_per_second(t, optimizer, list_optimizer_sensor)
            #
            # eee = []
            # for node in self.node:
            #     eee.append(node.energy)
            # print("enery", eee)

            current_dead = self.count_dead_node()
            nb_dead = current_dead
            # current_package = self.count_package()
            # if current_dead != nb_dead or current_package != nb_package:
            #     nb_dead = current_dead
            #     nb_package = current_package
            #     writer.writerow({"time": t, "nb dead": nb_dead, "nb package": nb_package})
            node_min_energy = self.node[self.find_min_node()]
            node_max_energy = self.node[self.find_max_node()]
            print("min_energy", node_min_energy.id, node_min_energy.energy, "max_energy", node_max_energy.id, node_max_energy.energy, "current_dead", current_dead, "nb_pack", self.nb_pack-self.nb_pack_sent)
            writer.writerow(
                {"time": t, "mc location": self.mc.current, "mc energy": self.mc.energy, "min energy": node_min_energy.energy, "max energy": node_max_energy.energy, "max charge": self.node[self.find_max_node_charging()].charging_time, "nb_dead": current_dead, "nb_pack": self.nb_pack-self.nb_pack_sent})
        information_log.close()
        return t

    def simulate(self, optimizer=None, list_optimizer_sensor=None, max_time=None, file_name="log/energy_log.csv"):
        """
        simulate in general. if max_time is not none, simulate_max_time will be called
        :param optimizer:
        :param max_time:
        :param file_name:
        :return:
        """
        if max_time:
            t = self.simulate_max_time(optimizer=optimizer, list_optimizer_sensor=list_optimizer_sensor, max_time=max_time)
        else:
            t = self.simulate_lifetime(optimizer=optimizer, list_optimizer_sensor=list_optimizer_sensor, file_name=file_name)
        return t

    def print_net(self, func=to_string):
        """
        print information of network
        :param func:
        :return:
        """
        func(self)

    def find_min_node(self):
        """
        find id of node which has minimum energy in network
        :return:
        """
        min_energy = 10 ** 10
        min_id = -1
        for node in self.node:
            # print("find_min_node", node.id, node.energy)
            if node.energy <= min_energy:
                min_energy = node.energy
                min_id = node.id
        return min_id

    def find_max_node(self):
        """
        find id of node which has minimum energy in network
        :return:
        """
        max_energy = -10 ** 10
        max_id = -1
        for node in self.node:
            # print("find_min_node", node.id, node.energy)
            if node.energy >= max_energy:
                max_energy = node.energy
                max_id = node.id
        return max_id

    def find_max_used(self):
        max_energy = -10 ** 10
        max_id = -1
        for node in self.node:
            # print("find_min_node", node.id, node.energy)
            if node.just_used_energy >= max_energy:
                max_energy = node.just_used_energy
                max_id = node.id
        return max_id

    def find_max_node_charging(self):
        """
        find id of node which has minimum energy in network
        :return:
        """
        max_charging = -1
        max_id = -1
        for node in self.node:
            if node.charging_time > max_charging:
                max_charging = node.charging_time
                max_id = node.id
        return max_id

    def count_dead_node(self):
        """
        count the number of node which dead
        :return:
        """
        count = 0
        for node in self.node:
            if node.energy < 0:
                count += 1
        return count

    def count_package(self, count_func=count_package_function):
        """
        count the number of package which can go to base
        :param count_func:
        :return:
        """
        count = count_func(self)
        return count
