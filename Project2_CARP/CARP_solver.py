import copy
import os
import random
from queue import PriorityQueue
import sys
import time
import numpy as np


class Vertex:

    def __init__(self, idx):
        self.idx = idx
        self.neighbors_idx = []
        self.to_neighbor_cost_list = []
        self.to_neighbor_demand_list = []


termination_time = 0
random_seed = 0

instance_name = ''
vertices_num = 0
depot_vertex_id = 0
required_edges_num = 0
non_required_edges_num = 0
vehicles_num = 0
capacity = 0
total_cost_of_required_edges = 0
total_demand_of_tasks = 0

vertices: list[Vertex] = []
cost_map = {}
demand_map = {}

task_list = []

distance_matrix = []

GE_routes = []
GE_batch_size = 20


def initiation():
    global termination_time, random_seed, instance_name, vertices_num, depot_vertex_id, required_edges_num, non_required_edges_num, vehicles_num, capacity, total_cost_of_required_edges, distance_matrix, total_demand_of_tasks
    console = sys.argv

    # # for debug use
    # console = console[2:8]

    input_file = console[1]
    termination_time = console[3]
    random_seed = int(console[5])
    random_seed = random_seed % (2 ** 32)

    with open(input_file, 'r') as file:
        line = file.readline()
        while line:
            line_info = line.split()
            if line_info[0] == 'NAME':
                instance_name = line_info[2]
            elif line_info[0] == 'VERTICES':
                vertices_num = int(line_info[2])
                distance_matrix = np.full((vertices_num, vertices_num), sys.maxsize)
                for i in range(vertices_num):
                    vertices.append(Vertex(i))
            elif line_info[0] == 'DEPOT':
                depot_vertex_id = int(line_info[2]) - 1
            elif line_info[0] == 'REQUIRED':
                required_edges_num = int(line_info[3])
            elif line_info[0] == 'NON-REQUIRED':
                non_required_edges_num = int(line_info[3])
            elif line_info[0] == 'VEHICLES':
                vehicles_num = int(line_info[2])
            elif line_info[0] == 'CAPACITY':
                capacity = int(line_info[2])
            elif line_info[0] == 'TOTAL':
                total_cost_of_required_edges = int(line_info[6])
            elif line_info[0] == 'NODES':
                pass
            elif line_info[0] == 'END':
                break
            elif line_info[0].isdigit():
                id_1 = int(line_info[0]) - 1
                id_2 = int(line_info[1]) - 1
                cost = int(line_info[2])
                demand = int(line_info[3])

                vertex_1 = vertices[id_1]
                vertex_2 = vertices[id_2]
                vertex_1.neighbors_idx.append(id_2)
                vertex_1.to_neighbor_cost_list.append(cost)
                vertex_1.to_neighbor_demand_list.append(demand)
                vertex_2.neighbors_idx.append(id_1)
                vertex_2.to_neighbor_cost_list.append(cost)
                vertex_2.to_neighbor_demand_list.append(demand)

                cost_map[(id_1, id_2)] = cost
                cost_map[(id_2, id_1)] = cost
                demand_map[(id_1, id_2)] = demand
                demand_map[(id_2, id_1)] = demand
                total_demand_of_tasks += demand

                if demand != 0:
                    task_list.append((id_1, id_2))
                    task_list.append((id_2, id_1))

            line = file.readline()


# dijkstra to find the shortest path between start_vertex to any other vertex
def dijkstra(start_vertex_idx):
    global distance_matrix
    priorityQueue = PriorityQueue()
    priorityQueue.put((0, start_vertex_idx))
    distance_matrix[start_vertex_idx][start_vertex_idx] = 0
    visited_vertices_id = []

    while not priorityQueue.empty():
        current_cost, current_vertex_idx = priorityQueue.get()
        current_vertex = vertices[current_vertex_idx]
        if current_vertex.idx not in visited_vertices_id:
            visited_vertices_id.append(current_vertex.idx)
            for i in range(len(current_vertex.neighbors_idx)):
                neighbor_idx = current_vertex.neighbors_idx[i]
                to_cost = current_vertex.to_neighbor_cost_list[i]
                new_cost = current_cost + to_cost

                if distance_matrix[start_vertex_idx][neighbor_idx] > new_cost:
                    distance_matrix[start_vertex_idx][neighbor_idx] = new_cost
                    priorityQueue.put((new_cost, neighbor_idx))


'''
Determine Rule:
 1) maximize the distance from the task to the depot; 
 2) minimize the distance from the task to the depot; 
 3) maximize the term dem(t)/sc(t), where dem(t) and sc(t) are demand and serving cost of task t, respectively;
 4) minimize the term dem(t)/sc(t); 
 5) use rule 1) if the vehicle is less than half- full, otherwise use rule 2) 
'''


def path_scanning(routes, rule):
    total_quality = 0
    route_quality = 0
    current_load = 0
    end_of_path_ptr = depot_vertex_id
    route = [(0, 0)]

    task_list_copy = copy.deepcopy(task_list)
    while task_list_copy:
        min_distance = sys.maxsize
        best_task = None
        best_task_src_vertex, best_task_dst_vertex = -1, -1
        best_task_demand = -1
        # Determine rule 5
        load_half_full = (current_load >= capacity / 2)

        # check all the tasks in list and get the best task to take
        for task in task_list_copy:
            task_demand = demand_map[task]
            task_src_vertex, task_dst_vertex = task

            if task_demand + current_load <= capacity:
                distance = distance_matrix[end_of_path_ptr][task_src_vertex]
                # select the closest task to current vertex(end of the path)
                if distance < min_distance:
                    min_distance = distance
                    best_task = task
                    best_task_src_vertex, best_task_dst_vertex = task
                    best_task_demand = task_demand
                # Addition Rules to judge the best task when they're both the closest task
                elif distance == min_distance:
                    # Determine Rule 1:
                    # MAXIMIZE the distance from the task dst vertex(next end of the path) to the depot
                    if rule == 0:
                        if distance_matrix[task_dst_vertex][depot_vertex_id] > distance_matrix[best_task_dst_vertex][
                            depot_vertex_id]:
                            best_task = task
                            best_task_src_vertex, best_task_dst_vertex = task
                            best_task_demand = task_demand
                    # Determine Rule 2:
                    # MINIMIZE the distance from the task dst vertex(next end of the path) to the depot
                    elif rule == 1:
                        if distance_matrix[task_dst_vertex][depot_vertex_id] < distance_matrix[best_task_dst_vertex][
                            depot_vertex_id]:
                            best_task = task
                            best_task_src_vertex, best_task_dst_vertex = task
                            best_task_demand = task_demand
                    # Determine Rule 3:
                    # MAXIMIZE the term dem(t)/sc(t), where dem(t) and sc(t) are demand and serving cost of task t
                    elif rule == 2:
                        frac_1 = demand_map[task] / cost_map[task]
                        frac_2 = demand_map[best_task] / cost_map[best_task]
                        if frac_1 > frac_2:
                            best_task = task
                            best_task_src_vertex, best_task_dst_vertex = task
                            best_task_demand = task_demand
                    # Determine Rule 4:
                    # MINIMIZE the term dem(t)/sc(t), where dem(t) and sc(t) are demand and serving cost of task t
                    elif rule == 3:
                        frac_1 = demand_map[task] / cost_map[task]
                        frac_2 = demand_map[best_task] / cost_map[best_task]
                        if frac_1 < frac_2:
                            best_task = task
                            best_task_src_vertex, best_task_dst_vertex = task
                            best_task_demand = task_demand
                    elif rule == 4:
                        # Determine Rule 5: use rule 1) if the vehicle is less than half full, otherwise use rule 2)
                        if not load_half_full:
                            if distance_matrix[task_dst_vertex][depot_vertex_id] > \
                                    distance_matrix[best_task_dst_vertex][
                                        depot_vertex_id]:
                                best_task = task
                                best_task_src_vertex, best_task_dst_vertex = task
                                best_task_demand = task_demand
                        else:
                            if distance_matrix[task_dst_vertex][depot_vertex_id] < \
                                    distance_matrix[best_task_dst_vertex][
                                        depot_vertex_id]:
                                best_task = task
                                best_task_src_vertex, best_task_dst_vertex = task
                                best_task_demand = task_demand

        # return to the depot when no task can do
        if best_task is None:
            route.append((0, 0))
            route_quality += distance_matrix[end_of_path_ptr][depot_vertex_id]
            end_of_path_ptr = depot_vertex_id
            routes.append((route, route_quality, current_load))
            total_quality += route_quality
            # reset information about route
            route = [(0, 0)]
            route_quality = 0
            current_load = 0
        # Do the best task
        else:
            route_quality += min_distance + cost_map[(best_task_src_vertex, best_task_dst_vertex)]
            route.append(best_task)
            current_load += best_task_demand
            end_of_path_ptr = best_task_dst_vertex
            task_list_copy.remove(best_task)
            task_list_copy.remove((best_task_dst_vertex, best_task_src_vertex))

        # After doing the best task, all tasks are done -> back to depot
        if not task_list_copy:
            route_quality += distance_matrix[end_of_path_ptr][depot_vertex_id]
            route.append((0, 0))
            routes.append((route, route_quality, current_load))
            total_quality += route_quality
            route = [(0, 0)]
            route_quality = 0
            current_load = 0

    return total_quality


def path_scanning_ER(routes, rule):
    total_quality = 0
    route_quality = 0
    current_load = 0
    end_of_path_ptr = depot_vertex_id
    route = [(0, 0)]

    task_list_copy = copy.deepcopy(task_list)
    while task_list_copy:
        min_distance = sys.maxsize
        best_task = None
        best_task_src_vertex, best_task_dst_vertex = -1, -1
        best_task_demand = -1
        random_tasks_set = []

        for task in task_list_copy:
            task_demand = demand_map[task]
            task_src_vertex, task_dst_vertex = task
            distance = distance_matrix[end_of_path_ptr][task_src_vertex]

            if task_demand + current_load <= capacity:
                remaining_capacity = capacity - current_load
                avg_demand_with_para = 1.5 * total_demand_of_tasks / required_edges_num
                if remaining_capacity > avg_demand_with_para:
                    if distance <= min_distance:
                        min_distance = distance
                        random_tasks_set.append(task)
                else:
                    left_hand_side = distance_matrix[end_of_path_ptr][task_src_vertex] + cost_map[task] \
                                     + distance_matrix[task_dst_vertex][depot_vertex_id]
                    right_hand_side = distance_matrix[end_of_path_ptr][depot_vertex_id] \
                                      + total_cost_of_required_edges / required_edges_num
                    if left_hand_side <= right_hand_side:
                        if distance <= min_distance:
                            min_distance = distance
                            random_tasks_set.append(task)

        if len(random_tasks_set) != 0:
            best_task = random_tasks_set[np.random.randint(0, len(random_tasks_set))]
            best_task_src_vertex, best_task_dst_vertex = best_task
            best_task_demand = demand_map[best_task]

        # return to the depot when no task can do
        if best_task is None:
            route.append((0, 0))
            route_quality += distance_matrix[end_of_path_ptr][depot_vertex_id]
            end_of_path_ptr = depot_vertex_id
            routes.append((route, route_quality, current_load))
            total_quality += route_quality
            # reset information about route
            route = [(0, 0)]
            route_quality = 0
            current_load = 0
        # Do the best task
        else:
            route_quality += distance_matrix[end_of_path_ptr][best_task_src_vertex] \
                             + cost_map[best_task]
            route.append(best_task)
            current_load += best_task_demand
            end_of_path_ptr = best_task_dst_vertex
            task_list_copy.remove(best_task)
            task_list_copy.remove((best_task_dst_vertex, best_task_src_vertex))

        # After doing the best task, all tasks are done -> back to depot
        if not task_list_copy:
            route_quality += distance_matrix[end_of_path_ptr][depot_vertex_id]
            route.append((0, 0))
            routes.append((route, route_quality, current_load))
            total_quality += route_quality
            route = [(0, 0)]
            route_quality = 0
            current_load = 0

    return total_quality


def flip(routes, total_quality):
    for j in range(len(routes)):
        route, route_quality, route_load = routes[j]
        for i in range(len(route)):
            if route[i] == (0, 0):
                continue

            prev_arc = route[i - 1]
            next_arc = route[i + 1]
            prev_arc_end = prev_arc[1]
            next_arc_start = next_arc[0]
            now_arc_start, now_arc_end = route[i]
            cost_before_change = distance_matrix[prev_arc_end][now_arc_start] + distance_matrix[now_arc_end][
                next_arc_start]
            cost_after_change = distance_matrix[prev_arc_end][now_arc_end] + distance_matrix[now_arc_start][
                next_arc_start]

            if cost_after_change < cost_before_change:
                route_quality = route_quality - cost_before_change + cost_after_change
                total_quality = total_quality - cost_before_change + cost_after_change
                route[i] = (now_arc_end, now_arc_start)

        routes[j] = (route, route_quality, route_load)

    return total_quality


def find_in_route(cur_arc_idx, num, cur_route, cur_route_quality):
    target_arc_idx = None
    target_route_cost = None
    most_decrease_delta = 0

    prev_arc_end = (cur_route[cur_arc_idx - 1])[1]
    cur_arc_start = (cur_route[cur_arc_idx])[0]
    cur_arc_end = (cur_route[cur_arc_idx + num])[1]
    next_arc_start = (cur_route[cur_arc_idx + num + 1])[0]
    new_cost = cur_route_quality - distance_matrix[prev_arc_end][cur_arc_start] - \
               distance_matrix[cur_arc_end][next_arc_start] + distance_matrix[prev_arc_end][next_arc_start]

    available_arc_idx = []
    for idx in range(1, len(cur_route)):
        if idx == cur_arc_idx or idx == cur_arc_idx + num or idx == cur_arc_idx + num + 1:
            continue
        available_arc_idx.append(idx)

    new_pos = available_arc_idx[np.random.randint(0, len(available_arc_idx))]
    new_prev_arc_end = (cur_route[new_pos - 1])[1]
    new_next_arc_start = (cur_route[new_pos])[0]
    new_cost = new_cost - distance_matrix[new_prev_arc_end][new_next_arc_start] + \
               distance_matrix[new_prev_arc_end][cur_arc_start] + distance_matrix[cur_arc_end][new_next_arc_start]

    if cur_route_quality - new_cost > most_decrease_delta:
        most_decrease_delta = cur_route_quality - new_cost
        target_arc_idx = new_pos
        target_route_cost = new_cost

    return most_decrease_delta, target_arc_idx, target_route_cost


def find_across_routes(routes, cur_arc_start, cur_arc_end, cur_inside_cost, cur_route_cost, cur_route_quality,
                       new_route_idx):
    target_arc_idx = None
    target_route_cost = None
    delta = 0

    new_route, new_route_quality, new_route_load = routes[new_route_idx]
    new_arc_idx = np.random.randint(1, len(new_route))

    new_prev_arc_end = (new_route[new_arc_idx - 1])[1]
    new_next_arc_start = (new_route[new_arc_idx])[0]
    new_route_cost = new_route_quality - distance_matrix[new_prev_arc_end][new_next_arc_start] + \
                     distance_matrix[new_prev_arc_end][cur_arc_start] + cur_inside_cost \
                     + distance_matrix[cur_arc_end][new_next_arc_start]
    if (cur_route_quality + new_route_quality) - (cur_route_cost + new_route_cost) > delta:
        delta = (cur_route_quality + new_route_quality) - (cur_route_cost + new_route_cost)
        target_arc_idx = new_arc_idx
        target_route_cost = new_route_cost

    return delta, target_arc_idx, target_route_cost


def single_arc_insertion(routes, cur_route_idx, cur_arc_idx):
    cur_route_info = routes[cur_route_idx]
    cur_route, cur_route_quality, cur_route_load = cur_route_info
    prev_arc_end = (cur_route[cur_arc_idx - 1])[1]
    cur_arc = cur_route[cur_arc_idx]
    cur_arc_demand = demand_map[cur_arc]
    next_arc_start = (cur_route[cur_arc_idx + 1])[0]
    cur_arc_start, cur_arc_end = cur_arc
    cur_inside_cost = cost_map[cur_arc]
    cur_route_cost = cur_route_quality - distance_matrix[prev_arc_end][cur_arc_start] - cur_inside_cost - \
                     distance_matrix[cur_arc_end][next_arc_start] + distance_matrix[prev_arc_end][next_arc_start]

    in_route = True
    most_delta = 0

    available_routes_idx = []
    for route_idx in range(0, len(routes)):
        route, route_quality, route_load = routes[route_idx]
        if (route_load + cur_arc_demand > capacity and route_idx != cur_route_idx) or (
                route_idx == cur_route_idx and len(route) == 3):
            continue
        available_routes_idx.append(route_idx)

    if len(available_routes_idx) == 0:
        return most_delta

    new_route_idx = available_routes_idx[np.random.randint(0, len(available_routes_idx))]
    target_route_idx = new_route_idx

    if new_route_idx == cur_route_idx:
        most_delta, target_arc_idx, target_route_cost \
            = find_in_route(cur_arc_idx, 0, cur_route, cur_route_quality)
    else:
        in_route = False
        most_delta, target_arc_idx, target_route_cost \
            = find_across_routes(routes, cur_arc_start, cur_arc_end, cur_inside_cost, cur_route_cost, cur_route_quality,
                                 new_route_idx)

    # find a better place
    if in_route:
        addition_load = 0
        origin_route_cost = target_route_cost
    else:
        origin_route_cost = cur_route_cost
        addition_load = cur_arc_demand

    if most_delta != 0:
        cur_route.pop(cur_arc_idx)
        cur_route_quality = origin_route_cost
        cur_route_load -= addition_load
        if len(cur_route) == 2:
            routes.pop(cur_route_idx)
        else:
            routes[cur_route_idx] = (cur_route, cur_route_quality, cur_route_load)
        target_route, target_route_quality, target_route_load = routes[target_route_idx]
        # 当插入同一数组中时，如果索引前的其他元素pop掉，则索引需要减少
        if in_route and target_arc_idx > cur_arc_idx:
            target_route.insert(target_arc_idx - 1, cur_arc)
        else:
            target_route.insert(target_arc_idx, cur_arc)
        target_route_quality = target_route_cost
        target_route_load += addition_load
        routes[target_route_idx] = (target_route, target_route_quality, target_route_load)

    return most_delta


def double_arc_insertion(routes, cur_route_idx, cur_arc_idx):
    cur_route_info = routes[cur_route_idx]
    cur_route, cur_route_quality, cur_route_load = cur_route_info
    prev_arc_end = (cur_route[cur_arc_idx - 1])[1]
    cur_arc_first = cur_route[cur_arc_idx]
    cur_arc_start, cur_arc_mid_1 = cur_arc_first
    cur_arc_second = cur_route[cur_arc_idx + 1]
    cur_arc_mid_2, cur_arc_end = cur_arc_second
    next_arc_start = (cur_route[cur_arc_idx + 2])[0]
    cur_arc_demand = demand_map[cur_arc_first] + demand_map[cur_arc_second]
    cur_inside_cost = cost_map[cur_arc_first] + distance_matrix[cur_arc_mid_1][cur_arc_mid_2] + cost_map[cur_arc_second]
    cur_route_cost = cur_route_quality - distance_matrix[prev_arc_end][cur_arc_start] - cur_inside_cost - \
                     distance_matrix[cur_arc_end][next_arc_start] + distance_matrix[prev_arc_end][next_arc_start]

    in_route = True
    most_delta = 0

    available_routes_idx = []
    for route_idx in range(0, len(routes)):
        route, route_quality, route_load = routes[route_idx]
        if (route_load + cur_arc_demand > capacity and route_idx != cur_route_idx) or (
                route_idx == cur_route_idx and len(route) == 4):
            continue
        available_routes_idx.append(route_idx)

    if len(available_routes_idx) == 0:
        return most_delta

    new_route_idx = available_routes_idx[np.random.randint(0, len(available_routes_idx))]
    target_route_idx = new_route_idx
    # insert in route
    if new_route_idx == cur_route_idx:
        most_delta, target_arc_idx, target_route_cost \
            = find_in_route(cur_arc_idx, 1, cur_route, cur_route_quality)
    else:
        in_route = False
        most_delta, target_arc_idx, target_route_cost \
            = find_across_routes(routes, cur_arc_start, cur_arc_end, cur_inside_cost, cur_route_cost, cur_route_quality,
                                 new_route_idx)

    # find a better place
    if in_route:
        addition_load = 0
        origin_route_cost = target_route_cost
    else:
        origin_route_cost = cur_route_cost
        addition_load = cur_arc_demand

    if most_delta != 0:
        cur_route.pop(cur_arc_idx + 1)
        cur_route.pop(cur_arc_idx)
        cur_route_quality = origin_route_cost
        cur_route_load -= addition_load
        if len(cur_route) == 2:
            routes.pop(cur_route_idx)
        else:
            routes[cur_route_idx] = (cur_route, cur_route_quality, cur_route_load)
        target_route, target_route_quality, target_route_load = routes[target_route_idx]
        # 当插入同一数组中时，如果索引前的其他元素pop掉，则索引需要减少
        if in_route and target_arc_idx > cur_arc_idx:
            target_route.insert(target_arc_idx - 2, cur_arc_first)
            target_route.insert(target_arc_idx - 1, cur_arc_second)
        else:
            target_route.insert(target_arc_idx, cur_arc_first)
            target_route.insert(target_arc_idx + 1, cur_arc_second)
        target_route_quality = target_route_cost
        target_route_load += addition_load
        routes[target_route_idx] = (target_route, target_route_quality, target_route_load)

    return most_delta


def single_insertion(routes, total_quality):
    cur_route_idx = np.random.randint(0, len(routes))
    cur_route, cur_route_quality, cur_route_load = routes[cur_route_idx]
    cur_arc_idx = np.random.randint(1, len(cur_route) - 1)
    delta = single_arc_insertion(routes, cur_route_idx, cur_arc_idx)
    total_quality -= delta

    return total_quality


def double_insertion(routes, total_quality):
    available_routes_idx = []
    for idx in range(len(routes)):
        route, route_quality, route_load = routes[idx]
        if len(route) <= 3:
            continue
        available_routes_idx.append(idx)

    if len(available_routes_idx) == 0:
        return routes, total_quality

    cur_route_idx = available_routes_idx[np.random.randint(0, len(available_routes_idx))]
    cur_route, cur_route_quality, cur_route_load = routes[cur_route_idx]

    cur_arc_idx = np.random.randint(1, len(cur_route) - 2)
    delta = double_arc_insertion(routes, cur_route_idx, cur_arc_idx)
    total_quality -= delta

    return total_quality

def arc_swap_across_routes_global(routes, cur_route_idx, cur_arc_idx):
    target_route_idx = None
    target_arc_idx = None
    target_route_cost = None
    swap_cur_route_cost = None
    most_decrease_delta = 0

    cur_route, cur_route_quality, cur_route_load = routes[cur_route_idx]
    prev_arc_end = (cur_route[cur_arc_idx - 1])[1]
    cur_arc = cur_route[cur_arc_idx]
    cur_arc_demand = demand_map[cur_arc]
    next_arc_start = (cur_route[cur_arc_idx + 1])[0]
    cur_arc_start, cur_arc_end = cur_arc
    cur_arc_cost = cost_map[cur_arc]
    cur_route_cost = cur_route_quality - distance_matrix[prev_arc_end][cur_arc_start] - cur_arc_cost -\
        distance_matrix[cur_arc_end][next_arc_start]

    for i in range(len(routes)):
        if i == cur_route_idx:
            continue
        new_route, new_route_quality, new_route_load = routes[i]
        for j in range(1, len(new_route) - 1):
            swap_arc = new_route[j]
            swap_arc_start, swap_arc_end = swap_arc
            new_prev_arc_end = (new_route[j - 1])[1]
            new_next_arc_start = (new_route[j + 1])[0]
            swap_arc_demand = demand_map[swap_arc]
            swap_arc_cost = cost_map[swap_arc]
            if cur_route_load - cur_arc_demand + swap_arc_demand > capacity \
                    or new_route_load - swap_arc_demand + cur_arc_demand > capacity:
                continue
            new_route_cost = new_route_quality - distance_matrix[new_prev_arc_end][swap_arc_start] - swap_arc_cost -\
                distance_matrix[swap_arc_end][new_next_arc_start] + distance_matrix[new_prev_arc_end][cur_arc_start] +\
                cur_arc_cost + distance_matrix[cur_arc_end][new_next_arc_start]
            new_cur_route_cost = cur_route_cost + distance_matrix[prev_arc_end][swap_arc_start] + swap_arc_cost +\
                distance_matrix[swap_arc_end][next_arc_start]
            if (cur_route_quality + new_route_quality) - (new_route_cost + new_cur_route_cost) > most_decrease_delta:
                most_decrease_delta = (cur_route_quality + new_route_quality) - (new_route_cost + new_cur_route_cost)
                target_route_idx = i
                target_arc_idx = j
                target_route_cost = new_route_cost
                swap_cur_route_cost = new_cur_route_cost

    if most_decrease_delta != 0:
        target_route, target_route_quality, target_route_load = routes[target_route_idx]
        cur_route.pop(cur_arc_idx)
        swap_arc = target_route.pop(target_arc_idx)
        cur_route.insert(cur_arc_idx, swap_arc)
        target_route.insert(target_arc_idx, cur_arc)
        cur_route_quality = swap_cur_route_cost
        target_route_quality = target_route_cost
        cur_route_load = cur_route_load - cur_arc_demand + demand_map[swap_arc]
        target_route_load = target_route_load - demand_map[swap_arc] + cur_arc_demand
        routes[cur_route_idx] = (cur_route, cur_route_quality, cur_route_load)
        routes[target_route_idx] = (target_route, target_route_quality, target_route_load)

    return most_decrease_delta


def arc_swap_across_routes(routes, cur_route_idx, cur_arc_idx):
    target_route_idx = None
    target_arc_idx = None
    target_route_cost = None
    swap_cur_route_cost = None
    most_delta = 0

    cur_route, cur_route_quality, cur_route_load = routes[cur_route_idx]
    prev_arc_end = (cur_route[cur_arc_idx - 1])[1]
    cur_arc = cur_route[cur_arc_idx]
    cur_arc_demand = demand_map[cur_arc]
    next_arc_start = (cur_route[cur_arc_idx + 1])[0]
    cur_arc_start, cur_arc_end = cur_arc
    cur_arc_cost = cost_map[cur_arc]
    cur_route_cost = cur_route_quality - distance_matrix[prev_arc_end][cur_arc_start] - cur_arc_cost - \
                     distance_matrix[cur_arc_end][next_arc_start]

    find_target = False
    visited_route = set()
    while not find_target:
        new_route_idx = np.random.randint(0, len(routes))
        new_route, new_route_quality, new_route_load = routes[new_route_idx]
        swap_arc_idx = np.random.randint(1, len(new_route) - 1)
        swap_arc = new_route[swap_arc_idx]
        swap_arc_demand = demand_map[swap_arc]
        if new_route_idx == cur_route_idx or cur_route_load - cur_arc_demand + swap_arc_demand > capacity or \
                new_route_load - swap_arc_demand + cur_arc_demand > capacity:
            visited_route.add(new_route_idx)
            if len(visited_route) == len(routes):
                return most_delta
            continue
        find_target = True

        swap_arc_start, swap_arc_end = swap_arc
        new_prev_arc_end = (new_route[swap_arc_idx - 1])[1]
        new_next_arc_start = (new_route[swap_arc_idx + 1])[0]
        swap_arc_cost = cost_map[swap_arc]

        new_route_cost = new_route_quality - distance_matrix[new_prev_arc_end][swap_arc_start] - swap_arc_cost - \
                         distance_matrix[swap_arc_end][new_next_arc_start] + distance_matrix[new_prev_arc_end][
                             cur_arc_start] + \
                         cur_arc_cost + distance_matrix[cur_arc_end][new_next_arc_start]
        cur_route_cost = cur_route_cost + distance_matrix[prev_arc_end][swap_arc_start] + swap_arc_cost + \
                         distance_matrix[swap_arc_end][next_arc_start]

        if (cur_route_quality + new_route_quality) - (new_route_cost + cur_route_cost) > most_delta:
            most_delta = (cur_route_quality + new_route_quality) - (new_route_cost + cur_route_cost)
            target_route_idx = new_route_idx
            target_arc_idx = swap_arc_idx
            target_route_cost = new_route_cost
            swap_cur_route_cost = cur_route_cost

    if most_delta != 0:
        target_route, target_route_quality, target_route_load = routes[target_route_idx]
        cur_route.pop(cur_arc_idx)
        swap_arc = target_route.pop(target_arc_idx)
        cur_route.insert(cur_arc_idx, swap_arc)
        target_route.insert(target_arc_idx, cur_arc)
        cur_route_quality = swap_cur_route_cost
        target_route_quality = target_route_cost
        cur_route_load = cur_route_load - cur_arc_demand + demand_map[swap_arc]
        target_route_load = target_route_load - demand_map[swap_arc] + cur_arc_demand
        routes[cur_route_idx] = (cur_route, cur_route_quality, cur_route_load)
        routes[target_route_idx] = (target_route, target_route_quality, target_route_load)

    return most_delta


def swap(routes, total_quality):
    choice = np.random.random()
    cur_route_idx = np.random.randint(0, len(routes))
    cur_route, cur_route_quality, cur_route_load = routes[cur_route_idx]
    cur_arc_idx = np.random.randint(1, len(cur_route) - 1)
    if choice < 0.7:
        delta = arc_swap_across_routes(routes, cur_route_idx, cur_arc_idx)
    else:
        delta = arc_swap_across_routes_global(routes, cur_route_idx, cur_arc_idx)

    total_quality -= delta

    return total_quality


# single 随机选择一条route中的两个task，选择其中的节点断开与task与task中间路径的连接 然后将中间路径180度反转 与原来路径比较
# double 将两条route切成四条sub route，然后两两匹配 与原来相比

def two_opt_single(routes, total_quality):
    available_route_idx = []
    for idx in range(len(routes)):
        if len(routes[idx]) > 4:
            available_route_idx.append(idx)

    if len(available_route_idx) == 0:
        return total_quality

    cur_route_idx = available_route_idx[np.random.randint(0, len(available_route_idx))]
    cur_route, cur_route_quality, cur_route_load = routes[cur_route_idx]
    cur_route_copy = copy.deepcopy(cur_route)

    first_arc_idx = np.random.randint(1, len(cur_route) - 2)
    if first_arc_idx + 2 > len(cur_route) - 2:
        return total_quality
    second_arc_idx = np.random.randint(first_arc_idx + 2, len(cur_route) - 1)

    left_ptr = first_arc_idx + 1
    right_ptr = second_arc_idx - 1
    while left_ptr < right_ptr:
        vertex_1_src, vertex_1_dst = cur_route[left_ptr]
        vertex_2_src, vertex_2_dst = cur_route[right_ptr]
        cur_route_copy[left_ptr] = (vertex_2_dst, vertex_2_src)
        cur_route_copy[right_ptr] = (vertex_1_dst, vertex_1_src)
        left_ptr += 1
        right_ptr -= 1
    if left_ptr == right_ptr:
        vertex_src, vertex_dst = cur_route[left_ptr]
        cur_route_copy[left_ptr] = (vertex_dst, vertex_src)

    new_cost = calculate_cost(cur_route_copy)
    if cur_route_quality > new_cost:
        routes[cur_route_idx] = (cur_route_copy, new_cost, cur_route_load)
        total_quality = total_quality - cur_route_quality + new_cost

    return total_quality


def two_opt_double(routes, total_quality):
    route_1_idx, route_2_idx = 0, 0

    available_route_idx = []
    for idx in range(len(routes)):
        if len(routes[idx]) > 4:
            available_route_idx.append(idx)

    if len(available_route_idx) < 2:
        return total_quality

    while route_1_idx == route_2_idx:
        route_1_idx, route_2_idx = np.random.randint(0, len(available_route_idx)), np.random.randint(0, len(available_route_idx))

    route_1, route_1_quality, route_1_load = routes[available_route_idx[route_1_idx]]
    route_2, route_2_quality, route_2_load = routes[available_route_idx[route_2_idx]]

    cnt = 0
    while True:
        arc_idx_1 = np.random.randint(2, len(route_1) - 2)
        arc_idx_2 = np.random.randint(2, len(route_2) - 2)

        demand_1, demand_2 = 0, 0
        for idx in range(arc_idx_1 + 1, len(route_1) - 1):
            demand_1 += demand_map[route_1[idx]]
        for idx in range(arc_idx_2 + 1, len(route_2) - 1):
            demand_2 += demand_map[route_2[idx]]

        new_demand_route_1 = route_1_load - demand_1 + demand_2
        new_demand_route_2 = route_2_load - demand_2 + demand_1
        cnt += 1
        if (new_demand_route_1 < capacity and new_demand_route_2 < capacity) \
                or cnt >= len(route_1) + len(route_2):
            break

    if cnt >= len(route_1) + len(route_2):
        return total_quality

    new_route_1 = route_1[:(arc_idx_1 + 1)] + route_2[(arc_idx_2 + 1):]
    new_route_2 = route_2[:(arc_idx_2 + 1)] + route_1[(arc_idx_1 + 1):]

    new_cost_1 = calculate_cost(new_route_1)
    new_cost_2 = calculate_cost(new_route_2)

    if route_1_quality + route_2_quality > new_cost_1 + new_cost_2:
        routes[route_1_idx] = (new_route_1, new_cost_1, new_demand_route_1)
        routes[route_2_idx] = (new_route_2, new_cost_2, new_demand_route_2)
        total_quality = total_quality - route_1_quality - route_2_quality + new_cost_1 + new_cost_2

    return total_quality


def calculate_cost(route):
    cost = 0
    now_pos = depot_vertex_id
    for i in range(1, len(route) - 1):
        task = route[i]
        cost += distance_matrix[now_pos][task[0]] + cost_map[task]
        now_pos = task[1]
    cost += distance_matrix[now_pos][depot_vertex_id]

    return cost


def arcs_to_string(routes):
    s = 's '
    for route, route_quality, route_load in routes:
        for arc in route:
            if arc == (0, 0):
                s += '0,'
            else:
                s += '(' + str(arc[0] + 1) + ',' + str(arc[1] + 1) + '),'

    s = s.removesuffix(',')
    return s


if __name__ == '__main__':
    start_time = time.time()
    # read the input and construct the graph
    initiation()
    np.random.seed(random_seed)

    # find the shortest path between any two vertices using dijkstra
    for i in range(vertices_num):
        dijkstra(i)

    best_routes = []
    best_quality = sys.maxsize

    # Original Path Scanning
    for i in range(5):
        start_time = time.time()
        routes = []
        quality = path_scanning_ER(routes, rule=i)
        GE_routes.append((routes, quality))
        if quality < best_quality:
            best_quality = quality
            best_routes = routes
        print(time.time() - start_time)

    # Ellipse Path Scanning
    # for i in range(int(GE_batch_size - 5)):
    #     routes = []
    #     quality = path_scanning_ER(routes, i)
    #     GE_routes.append((routes, quality))
    #     if quality < best_quality:
    #         best_quality = quality
    #         best_routes = routes

    generation_cnt = 0
    while time.time() - start_time < float(termination_time) - 0.2:
        generation_cnt += 1
        for idx in range(len(GE_routes)):
            routes, quality = GE_routes[idx]
            choice_weight = [0.2, 0.5, 0.7, 0.85]
            choice = np.random.random()

            routes_copy = copy.deepcopy(routes)
            if choice < choice_weight[0]:
                quality = single_insertion(routes_copy, quality)
            elif choice < choice_weight[1]:
                quality = double_insertion(routes_copy, quality)
            elif choice < choice_weight[2]:
                quality = swap(routes_copy, quality)
            elif choice < choice_weight[3]:
                quality = two_opt_single(routes_copy, quality)
            else:
                quality = two_opt_double(routes_copy, quality)

            quality = flip(routes_copy, quality)

            possibility = np.random.random()
            if possibility < 0.8:
                GE_routes[idx] = (routes_copy, quality)
                if quality < best_quality:
                    best_quality = quality
                    best_routes = routes

    print(generation_cnt)

    s_line = arcs_to_string(best_routes)
    quality_line = 'q ' + str(best_quality)
    output_info = s_line + '\n' + quality_line
    print(output_info)

    # # local check output
    # f = open("output.txt", "w")
    # f.write(output_info + os.linesep)
    # f.close()

    # print(generation_cnt)
    # for routes, quality in GE_routes:
    #     routes_quality = 0
    #     for route, route_cost, route_load in routes:
    #         cost = calculate_cost(route)
    #         routes_quality += cost
    #         if cost != route_cost:
    #             print(route)
    #             print("correct cost: " + str(route_cost))
    #             print("now cost: " + str(cost))
    #
    #     if routes_quality != quality:
    #         print("routes quality wrong")
    #         print(routes)
    #         break
