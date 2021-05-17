import random
from Package import Package


def uniform_com_func(net):
    """
    communicate function
    :param net:
    :return:
    """
    for node in net.node:
        if node.id in net.target and random.random() <= node.prob:
            package = Package()
            if node.is_active:
                node.send(net, package)
            net.nb_pack += 1
            if len(package.path) and package.path[-1] == -1:
                net.nb_pack_sent += 1
            # print(package.path)
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
