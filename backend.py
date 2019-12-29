import copy
import json
import math
import random
import threading
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from PIL import Image


class AnnealLoss(threading.Thread):
    def __init__(self, optimizer, pi, mode):
        threading.Thread.__init__(self)
        self.optimizer = optimizer
        self.mode = mode
        self.pi = pi

    def run(self):
        print("Starting annealing in mode {}".format(self.mode))
        loss = self.optimizer.func_cel(self.pi, mode=self.mode)
        self.optimizer.set_loss(loss, self.mode)
        print("Exited annealing in mode {}".format(self.mode))


# optimal_path = self.path_optim.anneal(t0, tk, nodes, whole_pi_func)
#                 optimal_path = self.get_full_path(optimal_path)

class InnerLoss(threading.Thread):
    def __init__(self, optimizer, mode, nodes, whole_pi_func):
        threading.Thread.__init__(self)
        self.optimizer = optimizer
        self.nodes = nodes
        self.whole_pi_func = whole_pi_func
        self.mode = mode

    def run(self):
        optimal_path = self.optimizer.path_optim.anneal(1000, 900, self.nodes, self.whole_pi_func)
        optimal_path = self.optimizer.get_full_path(optimal_path)
        cost = self.optimizer.get_path_length(optimal_path)
        self.optimizer.add_cost(cost, mode=self.mode)
        print("Exited inner annealing in mode {}".format(self.mode))

class PathOptimizer:
    def __init__(self, graph):
        self.graph = graph
        self.get_whole_pi = None

    def func_cel(self, pi):
        pi_whole = self.get_whole_pi(pi)
        cost = 0
        for i in range(len(pi_whole) - 1):
            cost += self.get_path_length(nx.astar_path(self.graph, pi_whole[i], pi_whole[i + 1]))
        return cost

    def choose_random_neighb(self, pi, mode='interchange'):
        pi_new = copy.deepcopy(pi)
        if mode == 'interchange':
            idx1, idx2 = random.choices(range(len(pi)), k=2)
            temp = pi_new[idx1]
            pi_new[idx1] = pi_new[idx2]
            pi_new[idx2] = temp
        if mode == 'insert':
            idx1, idx2 = random.choices(range(len(pi)), k=2)
            pi_new.insert(idx1, pi_new[idx2])
            if idx2 > idx1:
                pi_new.pop(idx2 + 1)
            else:
                pi_new.pop(idx2)
        return pi_new

    def get_path_length(self, path):
        cost = 0
        for i in range(len(path) - 1):
            cost += self.graph[path[i]][path[i + 1]]["weight"]
        return cost

    def anneal(self, T0, Tk, pi, get_whole_pi_func, lam=0.9995, early_stopping=100, printing=False):
        T = T0
        self.get_whole_pi = get_whole_pi_func
        func_cel = self.func_cel
        random_neighb = lambda x: self.choose_random_neighb(x, mode='interchange')
        pi_star = copy.deepcopy(pi)
        loss_prev = func_cel(pi_star)
        steps = 0
        while (T >= Tk):
            pi_prim = random_neighb(pi)
            loss = func_cel(pi_star)
            if func_cel(pi_prim) < loss:
                pi_star = copy.deepcopy(pi_prim)
            if func_cel(pi_prim) <= func_cel(pi):
                pi = copy.deepcopy(pi_prim)
            else:
                delta = func_cel(pi_prim) - func_cel(pi)
                p = math.exp(-delta / T)
                z = random.random()
                if z <= p:
                    pi = copy.deepcopy(pi_prim)
            if printing:
                print("{}: {}".format(T, loss))

            if loss_prev == loss:
                steps += 1
            else:
                steps = 0
            if steps >= early_stopping:
                break
            loss_prev = loss
            T = lam * T
        return pi_star


class Optimizer:
    def __init__(self):
        self.data_path = Path("data")
        self.orders_mapping = self.load_mapping(self.data_path.joinpath("order2graf.txt"))
        self.graph = self.load_graph(self.data_path.joinpath("Graf.txt"))
        self.img2graphdict = self.load_img2graphdict(self.data_path.joinpath("graf2img.txt"))
        self.orders = self.load_orders(self.data_path.joinpath("orders.csv"))
        self.resources = self.load_resources(self.data_path.joinpath("resources.csv"))
        self.starting_pi = self.get_starting_pi()
        self.img = Image.open(self.data_path.joinpath("def.jpg"))
        self.path_optim = PathOptimizer(self.graph)

        self.loss = None
        self.loss_pi = None
        self.loss_prim = None
        self.loss_prev = None

        self.cost = 0
        self.cost_pi = 0
        self.cost_prim = 0
        self.cost_prev = 0

    def load_mapping(self, filename):
        order2graph = {}
        with open(filename, "r") as file:
            for line in file:
                ordp, node = line.strip().split(":")
                ordp = ordp.strip()
                node = int(node.strip())
                order2graph[ordp] = node
        return order2graph

    def add_cost(self, cost, mode):
        if mode == 1:
            self.cost_prev += cost
        elif mode == 2:
            self.cost += cost
        elif mode == 3:
            self.cost_prim += cost
        else:
            self.cost_pi += cost

    def set_loss(self, loss, mode):
        if mode == 1:
            self.loss_prev = loss
        elif mode == 2:
            self.loss = loss
        elif mode == 3:
            self.loss_prim = loss
        else:
            self.loss_pi = loss

    def load_graph(self, filename):
        graph = nx.Graph()

        with open(filename) as file:
            for line in file:
                ident, rest = line.split(":")
                ident = int(ident[1:].strip())
                if rest.strip()[-1] == ";":
                    rest = rest.strip()[:-1]
                rest = rest.split(";")
                idents = []
                weights = []
                for cont in rest:
                    iden_2, weight = cont.split("x")
                    iden_2 = int(iden_2.strip())
                    weight = int(weight.strip())
                    idents.append(iden_2)
                    weights.append(weight)
                graph.add_weighted_edges_from([(ident, idents[i], weights[i]) for i in range(len(idents))])
                graph.add_weighted_edges_from([(idents[i], ident, weights[i]) for i in range(len(idents))])
        return graph

    def load_img2graphdict(self, filename):
        i2gdict = {}
        with open(filename, mode="r") as file:
            for line in file:
                idx, rest = line.strip().strip(";").split(":")
                idx = int(idx)
                x, y = rest.split(",")
                x = int(x)
                y = int(y)
                i2gdict[idx] = (x, y)
        return i2gdict

    def load_orders(self, filename):
        dataframe = pd.read_csv(filename).set_index("order_id")
        return dataframe

    def load_resources(self, filename):
        dataframe = pd.read_csv(filename).set_index("uniq_id")
        return dataframe

    def get_full_path(self, solution):
        full_path = []
        for i in range(len(solution) - 1):
            if len(full_path) != 0:
                full_path += nx.astar_path(self.graph, solution[i], solution[i + 1])[1:]
            else:
                full_path += nx.astar_path(self.graph, solution[i], solution[i + 1])
        return full_path

    def draw_solution(self, solution):
        full_path = self.get_full_path(solution)
        xs = [self.img2graphdict[node][0] for node in full_path]
        ys = [self.img2graphdict[node][1] for node in full_path]
        plt.imshow(self.img)
        plt.plot(xs, ys)
        plt.show()

    def get_path_length(self, path):
        cost = 0
        for i in range(len(path) - 1):
            cost += self.graph[path[i]][path[i + 1]]["weight"]
        return cost

    def get_whole_route(self, pi, delivery_options, start=99):
        options = ["postal_service", "DHL", "InPost", "UPS", "TNT", "DPD"]
        end = []
        for delivery_option in delivery_options:
            if delivery_option in options:
                end.append(200 + options.index(delivery_option))
        end.sort()
        return [start] + pi + end

    def orders_to_graph(self, orders):
        nodes = []
        postals = []
        for order in orders:
            series = self.orders.loc[order]
            order_dict = json.loads(series.get("items_ids").replace("\'", "\""))
            service = series.get("delivery_option")
            for key in order_dict:
                location = self.resources.loc[key, "location"]
                location = location[:4]
                node = self.orders_mapping[location]
                if node not in nodes:
                    nodes.append(node)
            if service not in postals:
                postals.append(service)
        return nodes, postals

    def func_cel(self, pi, mode):
        if mode == 1:
            self.cost_prev = 0
        elif mode == 2:
            self.cost = 0
        elif mode == 3:
            self.cost_prim = 0
        else:
            self.cost_pi = 0
        pi_new = pi[1:]
        order_stack = []
        threads = []
        for order in pi_new:
            if order != 0:
                order_stack.append(order)
            else:
                nodes, postals = self.orders_to_graph(order_stack)
                whole_pi_func = lambda x: self.get_whole_route(x, postals)
                # t0 = 1000
                # tk = 700
                # optimal_path = self.path_optim.anneal(t0, tk, nodes, whole_pi_func)
                # optimal_path = self.get_full_path(optimal_path)
                # cost += self.get_path_length(optimal_path)
                threads.append(InnerLoss(self, mode, nodes, whole_pi_func))
                threads[-1].start()
                order_stack = []
        if len(order_stack) != 0:
            nodes, postals = self.orders_to_graph(order_stack)
            whole_pi_func = lambda x: self.get_whole_route(x, postals)
            # t0 = 1000
            # tk = 700
            # optimal_path = self.path_optim.anneal(t0, tk, nodes, whole_pi_func)
            # cost += self.get_path_length(optimal_path)
            threads.append(InnerLoss(self, mode, nodes, whole_pi_func))
            threads[-1].start()

        for thread in threads:
            thread.join()

        if mode == 1:
            return self.cost_prev
        elif mode == 2:
            return self.cost
        elif mode == 3:
            return self.cost_prim
        else:
            return self.cost_pi

    def choose_random_neighb(self, pi, mode='interchange'):
        zeroes = pi.count(0)
        pi_new = copy.deepcopy(pi)
        pi_len = len(pi) - zeroes
        if mode == 'interchange':
            idx1, idx2 = random.choices(range(pi_len), k=2)
            idx1 = idx1 + int(pi_len / 5)
            idx2 = idx2 + int(pi_len / 5)
            temp = pi_new[idx1]
            pi_new[idx1] = pi_new[idx2]
            pi_new[idx2] = temp
        if mode == 'insert':
            idx1, idx2 = random.choices(range(pi_len), k=2)
            idx1 = idx1 + int(pi_len / 5)
            idx2 = idx2 + int(pi_len / 5)
            pi_new.insert(idx1, pi_new[idx2])
            if idx2 > idx1:
                pi_new.pop(idx2 + 1)
            else:
                pi_new.pop(idx2)
        return pi_new

    def get_starting_pi(self):
        out = [0]
        for i in range(len(self.orders.index)):
            out.append(self.orders.index[i])
            if i % 5 == 4:
                out.append(0)
        return out

    def anneal(self, T0, Tk, pi, lam=0.9995, early_stopping=250, printing=True):
        T = T0
        random_neighb = lambda x: self.choose_random_neighb(x, mode='interchange')
        pi_star = copy.deepcopy(pi)
        # self.loss_prev = func_cel(pi_star)
        thread_prev = AnnealLoss(self, pi_star, 1)
        thread_prev.start()
        steps = 0
        calculate_loss = True
        calculate_loss_prim = True
        calculate_loss_pi = True
        while (T >= Tk):
            pi_prim = random_neighb(pi)
            calculate_loss_prim = True
            thread_1 = AnnealLoss(self, pi_star, 2)
            thread_2 = AnnealLoss(self, pi_prim, 3)
            thread_3 = AnnealLoss(self, pi, 4)
            if calculate_loss:
                # self.loss = func_cel(pi_star)
                thread_1.start()
            if calculate_loss_prim:
                thread_2.start()
            if calculate_loss_pi:
                thread_3.start()

            thread_1.join()
            thread_2.join()
            thread_3.join()

            calculate_loss = False
            calculate_loss_prim = False
            calculate_loss_pi = False

            if self.loss_prim < self.loss:
                pi_star = copy.deepcopy(pi_prim)
                calculate_loss = True
            if self.loss_prim <= self.loss_pi:
                pi = copy.deepcopy(pi_prim)
                calculate_loss_pi = True
            else:
                delta = self.loss_prim - self.loss_pi
                p = math.exp(-delta / T)
                z = random.random()
                if z <= p:
                    pi = copy.deepcopy(pi_prim)
                    calculate_loss_pi = True
            if printing:
                print("{}: {}".format(T, self.loss))

            if thread_prev.isAlive():
                thread_prev.join()

            if self.loss_prev == self.loss:
                steps += 1
            else:
                steps = 0
            if steps >= early_stopping:
                break
            self.loss_prev = self.loss
            T = lam * T
        return pi_star

    def format_output(self, optimal):
        out = []
        single = []
        for i in optimal:
            if i != 0:
                single.append(i)
            else:
                if len(single) != 0:
                    out.append(single)
                    single = []
        if len(single) != 0:
            out.append(single)
        return out

    def get_orders_node_dict(self, orders):
        nodes_dict = {}
        delivery_dict = {}
        for order in orders:
            series = self.orders.loc[order]
            order_dict = json.loads(series.get("items_ids").replace("\'", "\""))
            for key in order_dict:
                ser = self.resources.loc[key]
                location = ser.get("location")
                part_location = location[:4]
                node = self.orders_mapping[part_location]
                delivery_dict = {
                    "order_id": series.name,
                    "location": location,
                    "item": {
                        "product_name": ser.get("product_name"),
                        "product_description": ser.get("product_description"),
                        "product_information": ser.get("product_information"),
                        "description": ser.get("description"),
                        "manufacturer": ser.get("manufacturer")
                    },
                    "delivery_option": series.get("delivery_option")
                }

                if node in nodes_dict:
                    nodes_dict[node][ser.name] = delivery_dict
                else:
                    nodes_dict[node] = {
                        ser.name: delivery_dict
                    }

        return nodes_dict

    def optimize(self):
        optimal = self.anneal(1000, 500, self.starting_pi)
        order_stack = []
        output_list = []
        for order in optimal:
            if order != 0:
                order_stack.append(order)
            else:
                nodes, postals = self.orders_to_graph(order_stack)
                whole_pi_func = lambda x: self.get_whole_route(x, postals)
                t0 = 1000
                tk = 700
                optimal_path = self.path_optim.anneal(t0, tk, nodes, whole_pi_func)
                output_list.append({
                    "orders_ids": order_stack,
                    "optimal_path": optimal_path,
                    "node_dict": self.get_orders_node_dict(order_stack)
                })
                order_stack = []
        if len(order_stack) != 0:
            nodes, postals = self.orders_to_graph(order_stack)
            whole_pi_func = lambda x: self.get_whole_route(x, postals)
            t0 = 1000
            tk = 700
            optimal_path = self.path_optim.anneal(t0, tk, nodes, whole_pi_func)
            output_list.append({
                "orders_ids": order_stack,
                "optimal_path": optimal_path,
                "node_dict": self.get_orders_node_dict(order_stack)
            })
        return output_list


if __name__ == "__main__":
    optimizer = Optimizer()
    output = optimizer.optimize()