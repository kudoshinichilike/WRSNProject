from scipy.spatial import distance

from Node import Node
import random
from Network import Network
import pandas as pd
from ast import literal_eval
from MobileCharger import MobileCharger
from Q__Learning import Q_learning
from Q_LearningSensor import Q_LearningSensor
from Inma import Inma
from GSA import GSA
from scipy.stats import sem, t
import numpy as np
import csv
import sys
import Parameter as para

read_file = "thaydoisonode"
write_file = "try"
data_range = 5
data_start = 0
run_range = 4
learning_rate = 0.5
scale_factor = 0.5
learning_rate_sensor = 0.5
scale_factor_sensor = 0.5
read_name = "data/" + read_file + ".csv"
try:
    opt = sys.argv[8]
except:
    opt = "qlearning"
try:
    max_time = 1000000
except:
    max_time = None

# read_file = sys.argv[1]
# write_file = sys.argv[2]
# data_range = int(sys.argv[3])
# data_start = int(sys.argv[4])
# run_range = int(sys.argv[5])
# learning_rate = float(sys.argv[6])
# scale_factor = float(sys.argv[7])
# read_name = "data/" + read_file + ".csv"
# try:
#     opt = sys.argv[8]
# except:
#     opt = "qlearning"
# try:
#     max_time = int(sys.argv[9])
# except:
#     max_time = None

df = pd.read_csv(read_name)
for id_data in range(4, 5):
    index = id_data + data_start
    print("nb data rand = ", index)
    for nb_run in range(6, 7):
        write_name = "log/test_" + str(index) + "_" + str(nb_run) + ".csv"
        open_file = open(write_name, "w")
        result = csv.DictWriter(open_file, fieldnames=["idx", "state0", "state1", "time"])
        result.writeheader()

        print("nb run = ", nb_run)
        random.seed((index + 5)*7)
        node_pos = list(literal_eval(df.node_pos[index]))
        list_node = []
        list_optimizer_sensor = []
        for i in range(len(node_pos)):
            location = node_pos[i]
            com_ran = df.commRange[index]
            energy = df.energy[index]
            energy_max = 5.0
            prob = df.freq[index]
            energy = 3.0
            node = Node(location=location, com_ran=com_ran, energy=energy, energy_max=energy_max, id=i,
                        energy_thresh=0.4 * energy_max, prob=0.6)  # TODO: energy_thresh=0.4 * energy
            list_node.append(node)
            q_sensor = Q_LearningSensor(sensor=list_node[i], alpha=learning_rate, gamma=scale_factor)
            list_optimizer_sensor.append(q_sensor)
            node.optimizer = q_sensor

        for node1 in list_node:
            for node2 in list_node:
                if node1.id != node2.id:
                    if distance.euclidean(node1.location, node2.location) <= node1.com_ran:
                        node1.neighbor_charge.append(node2)

        mc = MobileCharger(energy=df.E_mc[index], capacity=df.E_max[index]/2, e_move=df.e_move[index],
                           e_self_charge=df.e_mc[index], velocity=df.velocity[index])
        target = [int(item) for item in df.target[index].split(',')]
        net = Network(list_node=list_node, mc=mc, target=target)
        print("test", len(net.node), len(net.target), max(net.target))
        q_learning = Q_learning(alpha=learning_rate, gamma=scale_factor)

        inma = Inma()
        gsa = GSA()
        if opt == "qlearning":
            optimizer = q_learning
        elif opt == "inma":
            optimizer = inma
        elif opt == "gsa":
            optimizer = gsa
        elif opt == "none":
            optimizer = None
        file_name = "log/q_learning_" + str(index) + ".csv"
        temp = net.simulate(optimizer=optimizer, list_optimizer_sensor=list_optimizer_sensor, file_name=file_name,
                            max_time=max_time, index=index, nb_run=nb_run)

        for sensor_optimizer in list_optimizer_sensor:
            for state0 in range(0, 100):
                for state1 in range(0, 100):
                    result.writerow({"idx": sensor_optimizer.sensor.id, "state0": state0, "state1": state1, "time": sensor_optimizer.mark[state0][state1]})

        print("done run = ", nb_run)

    # confidence = 0.95
    # h = sem(life_time) * t.ppf((1 + confidence) / 2, len(life_time) - 1)
    # result.writerow({"nb run": np.mean(life_time), "lifetime": h})
    open_file.close()
    print("done data = ", index)
print("done all")
