import random
from Package import Package
from scipy.spatial import distance
from Node_Method import find_receiver
import Parameter as para


def uniform_com_func(net):
    """
    communicate function
    :param net:
    :return:
    """
    if random.random() <= 0.1:
        prob = 1
    else:
        prob = None

    for node in net.node:
        prob_save = node.prob
        if prob is not None:
            node.prob = prob

        if node.id in net.target and random.random() <= node.prob:
            package = Package()
            if node.is_active:
                node.send(net, package)
            net.nb_pack += 1
            if len(package.path) and package.path[-1] == -1:
                net.nb_pack_sent += 1
            # print(package.path)

        node.prob = prob_save
    return True


def to_string(net):
    """
    print information of network
    :param net:
    :return:
    """
    min_energy = 10 ** 10
    min_node = -1
    for node in net.node:
        if node.energy < min_energy:
            min_energy = node.energy
            min_node = node
    min_node.print_node()


def count_package_function(net):
    """
    count the number of package which can go to base
    :param net:
    :return:
    """
    count = 0
    for target_id in net.target:
        package = Package(is_energy_info=True)
        net.node[target_id].send(net, package)
        if package.path[-1] == -1:
            count += 1
    return count


def get_path(net, sensor_id, receive_func=find_receiver):
    """
    getting path from sensor_id to base
    :param net:
    :param sensor_id:
    :param receive_func:
    :return:
    """
    path = [sensor_id]
    if distance.euclidean(net.node[sensor_id].location, para.base) <= net.node[sensor_id].com_ran:
        path.append(para.base)
    else:
        receive_id = receive_func(net=net, node=net.node[sensor_id])
        if receive_id != -1:
            path.extend(get_path(net, receive_id, receive_func))
    return path


def get_all_path(net, receive_func=find_receiver):
    """
    getting all paths from every target to base
    :param net:
    :param receive_func:
    :return:
    """
    list_path = []
    for sensor_id, target_id in enumerate(net.target):
        list_path.append(get_path(net, sensor_id, receive_func))
    return list_path
