from __future__ import division, print_function
import numpy as np
import pandas as pd

from highway_env.logger import Loggable
from highway_env.road.lane import LineType, StraightLane
from highway_env.vehicle.control import ControlledVehicle
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.dynamics import Obstacle,RedLight
from gym import logger


class RoadNetwork(object):

    def __init__(self):
        self.graph = {} # roadnetwork 的节点图
        self.LANES_NUMBER = 0 # roadnetwork 中有多少lane
        self.LANES = [] # 保存了每个lane的index: ("from","to","index:0/1/2/....")
        self.decoration_lanes = [] # 无需加入到graph里，起修饰作用的道路。起点和终点均用" "表示

    def add_node(self, node):
        """
            A node represents an symbolic intersection in the road network.
        :param node: the node label.
        """
        if node not in self.graph:
            self.graph[node] = []

    def add_lane(self, _from, _to, lane):
        """
            A lane is encoded as an edge in the road network.
        :param _from: the node at which the lane starts.
        :param _to: the node at which the lane ends.
        :param AbstractLane lane: the lane geometry.
        """
        if _from == " " or _to == " ":
            self.decoration_lanes.append(lane)
            return
        if _from not in self.graph:
            self.graph[_from] = {}
        if _to not in self.graph[_from]:
            self.graph[_from][_to] = []
        self.graph[_from][_to].append(lane)
        self.LANES_NUMBER += 1
        index = len(self.graph[_from][_to])
        lane = (_from, _to, index - 1)
        self.LANES.append(lane)

    def get_lane(self, index):
        """
            Get the lane geometry corresponding to a given index in the road network.
        :param index: a tuple (origin node, destination node, lane id on the road).
        :return: the corresponding lane geometry.
        """
        _from, _to, _id = index
        return self.graph[_from][_to][_id]

    def get_closest_lane_index(self, position):
        """
            Get the index of the lane closest to a world position.

        :param position: a world position [m].
        :return: the index of the closest lane.
        """
        indexes, distances = [], []
        for _from, to_dict in self.graph.items():
            for _to, lanes in to_dict.items():
                for _id, l in enumerate(lanes):
                    distances.append(l.distance(position))
                    indexes.append((_from, _to, _id))
        return indexes[int(np.argmin(distances))]

    def next_lane(self, current_index, route=None, position=None, np_random=np.random):
        """
            Get the index of the next lane that should be followed after finishing the current lane.

            If a plan is available and matches with current lane, follow it.
            Else, pick next road randomly.
            If it has the same number of lanes as current road, stay in the same lane.
            Else, pick next road's closest lane.
        :param current_index: the index of the current lane.
        :param route: the planned route, if any.
        :param position: the vehicle position.
        :param np_random: a source of randomness.
        :return: the index of the next lane to be followed when current lane is finished.
        """
        _from, _to, _id = current_index
        next_to = None
        # Pick next road according to planned route
        if route:
            if route[0][:2] == current_index[:2]:  # We just finished the first step of the route, drop it.
                route.pop(0)
            if route and route[0][0] == _to:  # Next road in route is starting at the end of current road.
                # print("route[0]:",route[0])
                _, next_to, route_id = route[0]
                # print("route[0] next_to:", next_to)
            elif route:
                # print("route:\n",route)
                logger.warn("Route {} does not start after current road {}.".format(route[0], current_index))
        # Randomly pick next road
        if not next_to:
            try:
                next_to = list(self.graph[_to].keys())[np_random.randint(len(self.graph[_to]))]
            except KeyError:
                # logger.warn("End of lane reached.")
                return current_index

        # If next road has same number of lane, stay on the same lane
        if len(self.graph[_from][_to]) == len(self.graph[_to][next_to]):
            next_id = _id
        # Else, pick closest lane
        else:
            lanes = range(len(self.graph[_to][next_to]))
            next_id = min(lanes,
                          key=lambda l: self.get_lane((_to, next_to, l)).distance(position))

        return _to, next_to, next_id

    def bfs_paths(self, start, goal):
        """
            Breadth-first search of all routes from start to goal.

        :param start: starting node
        :param goal: goal node
        :return: list of paths from start to goal.
        """
        # print("enter bfs_path")
        queue = [(start, [start])]
        while queue:
            (node, path) = queue.pop(0)
            if node not in self.graph:
                yield []
            for _next in set(self.graph[node].keys()) - set(path):
                if _next == goal:
                    yield path + [_next]
                elif _next in self.graph:
                    queue.append((_next, path + [_next]))

    def safest_path(self, lane_index, destination):
        """
            Breadth-first search of shortest path from start to goal.

        :param start: starting node
        :param goal: goal node
        :return: safest path from start to goal.
        * only run across side lanes(left/right) for achieving safest path, which may not occur some accidents (like some meek drivers in real world).
        """
        start = lane_index[0]
        sidelanes = self.side_lanes(lane_index)
        sidenext =[sidelane[1] for sidelane in sidelanes]
        queue = [(start, [start])]
        while queue:
            (node, path) = queue.pop(0)
            if node not in self.graph:
                yield []
            for _next in set(self.graph[node].keys()) - set(path):
                if _next == goal:
                    yield path + [_next]
                elif _next in self.graph and _next == sidelanes:
                    queue.append((_next, path + [_next]))

    def shortest_path(self, start, goal):
        """
            Breadth-first search of shortest path from start to goal.

        :param start: starting node
        :param goal: goal node
        :return: shortest path from start to goal.
        * may run across many lanes for achieving this shortest path, which may occur some accidents (like some brutal drivers in real world).
        """
        try:
            return next(self.bfs_paths(start, goal))
        except StopIteration:
            return None

    def all_side_lanes(self, lane_index):
        """
        :param lane_index: the index of a lane.
        :return: all indexes of lanes belonging to the same road.
        """
        return self.graph[lane_index[0]][lane_index[1]]

    def side_lanes(self, lane_index):
        """
                :param lane_index: the index of a lane.
                :return: indexes of lanes next to a an input lane, to its right or left.
                """
        _from, _to, _id = lane_index
        lanes = []
        if _id > 0:
            lanes.append((_from, _to, _id - 1))
        if _id < len(self.graph[_from][_to]) - 1:
            lanes.append((_from, _to, _id + 1))
        return lanes

    def side_lanes_bigmap(self, ego_vehicle):
        ego_lane = ego_vehicle.lane_index
        _from, _to, _id = ego_lane
        lanes = []
        if _id > 0:
            lanes.append((_from, _to, _id - 1))
        if _id < len(self.graph[_from][_to]) - 1:
            lanes.append((_from, _to, _id + 1))
        for i in ego_vehicle.lanes_around:
            l = self.get_lane(i)
            s,lat = l.local_coordinates(ego_vehicle.position)
            if -4 <= lat <= 4 and -4 <= s <= l.length + 4:
                lanes.append(i)
        # print("side_lanes_bigmap:",lanes)
        return lanes

    @staticmethod
    def is_same_road(lane_index_1, lane_index_2, same_lane=False):
        """
            Is lane 1 in the same road as lane 2?
        """
        return lane_index_1[:2] == lane_index_2[:2] and (not same_lane or lane_index_1[2] == lane_index_2[2])

    @staticmethod
    def is_leading_to_road(lane_index_1, lane_index_2, same_lane=False):
        """
            Is lane 1 leading to of lane 2?
        """
        return lane_index_1[1] == lane_index_2[0] and (not same_lane or lane_index_1[2] == lane_index_2[2])

    def is_connected_road(self, lane_index_1, lane_index_2, route=None, same_lane=False, depth=0):
        """
            Is the lane 2 leading to a road within lane 1's route?

            Vehicles on these lanes must be considered for collisions.
        :param lane_index_1: origin lane
        :param lane_index_2: target lane
        :param route: route from origin lane, if any
        :param same_lane: compare lane id
        :param depth: search depth from lane 1 along its route
        :return: whether the roads are connected
        """
        if RoadNetwork.is_same_road(lane_index_2, lane_index_1, same_lane) \
                or RoadNetwork.is_leading_to_road(lane_index_2, lane_index_1, same_lane):
            return True
        if depth > 0:
            if route and route[0][:2] == lane_index_1[:2]:
                # Route is starting at current road, skip it
                return self.is_connected_road(lane_index_1, lane_index_2, route[1:], same_lane, depth)
            elif route and route[0][0] == lane_index_1[1]:
                # Route is continuing from current road, follow it
                return self.is_connected_road(route[0], lane_index_2, route[1:], same_lane, depth - 1)
            else:
                # Recursively search all roads at intersection
                _from, _to, _id = lane_index_1
                return any([self.is_connected_road((_to, l1_to, _id), lane_index_2, route, same_lane, depth - 1)
                            for l1_to in self.graph.get(_to, {}).keys()])
        return False

    @staticmethod
    def straight_road_network(lanes=4, length=10000):
        """
        build road network automatically
        :param lanes: number of lanes of this road
        :param length: road's length
        :return: class Roadnetwork
        """
        net = RoadNetwork()
        for lane in range(lanes):
            origin = [0, lane * StraightLane.DEFAULT_WIDTH]
            end = [length, lane * StraightLane.DEFAULT_WIDTH]
            line_types = [LineType.CONTINUOUS_LINE if lane == 0 else LineType.STRIPED,
                          LineType.CONTINUOUS_LINE if lane == lanes - 1 else LineType.NONE]
            net.add_lane(0, 1, StraightLane(origin, end, line_types=line_types))
        return net


class Road(Loggable):
    """
        A road is a set of lanes, and a set of vehicles driving on these lanes
    """

    def __init__(self, vehicles=None, network=None, np_random=None):
        """
            New road.

        :param network: the network of road
        :param vehicles: the vehicles driving on the road
        """
        self.vehicles = vehicles or []
        self.np_random = np_random if np_random else np.random.RandomState()
        self.network = network or []

    @classmethod
    def create_random_road(cls,
                           lanes_count,
                           lane_width=3.6,
                           vehicles_count=50,
                           vehicles_type=ControlledVehicle,
                           np_random=None):
        """
            Create a road composed of straight adjacent lanes with randomly located vehicles on it.

        :param lanes_count: number of lanes
        :param lane_width: lanes width [m]
        :param vehicles_count: number of vehicles on the road
        :param vehicles_type: type of vehicles on the road
        :param np.random.RandomState np_random: a random number generator
        :return: the created road
        """
        lanes = []
        for lane in range(lanes_count):
            origin = np.array([0, lane * lane_width])
            heading = 0
            line_types = [LineType.CONTINUOUS_LINE if lane == 0 else LineType.STRIPED,
                          LineType.CONTINUOUS_LINE if lane == lanes_count - 1 else LineType.NONE]
            lanes.append(StraightLane(origin, heading, lane_width, line_types))
        r = Road(lanes)
        r.add_random_vehicles(vehicles_count, vehicles_type, np_random)
        return r

    def add_random_vehicles(self, vehicles_count=50, vehicles_type=ControlledVehicle, np_random=None):
        """
            Create some new random vehicles of a given type, and add them on the road.

        :param vehicles_count: number of vehicles to create
        :param vehicles_type: type of vehicles to create
        :param np.random.RandomState np_random: a random number generator
        """
        for _ in range(vehicles_count):
            self.vehicles.append(vehicles_type.create_random(self, np_random=np_random))

    def close_vehicles_to(self, vehicle, distances):
        return [v for v in self.vehicles if (distances[0] < vehicle.lane_distance_to(v) < distances[1]
                                             and v is not vehicle)]

    def closest_vehicles_to(self, vehicle, count):
        sorted_v = sorted([v for v in self.vehicles
                           if v is not vehicle
                           and -2 * vehicle.LENGTH < vehicle.lane_distance_to(v)],
                          key=lambda v: abs(vehicle.lane_distance_to(v)))
        return sorted_v[:count]

    def act(self,vehicle_actions = None):
        """
            Decide the actions of each entity on the road.
        """
        for vehicle in self.vehicles:
            if vehicle.id == 0 and vehicle_actions:
                vehicle.act(vehicle_actions=vehicle_actions)
            else:
                vehicle.act()

    def step(self, dt):
        """
            Step the dynamics of each entity on the road.

        :param dt: timestep [s]
        """
        for vehicle in self.vehicles:
            # if isinstance(vehicle,IDMVehicle):
            # continue
            vehicle.step(dt)
            for other in self.vehicles:
                if vehicle.check_collision(other):
                    # if hasattr(vehicle,"state") or hasattr(other,"state"):
                        # pass
                    # else:
                    self.vehicles.remove(vehicle)
                    self.vehicles.remove(other)
    """
    def get_lane(self, position):
    
            Get the lane closest to a world position.

        :param position: a world position [m]
        :return: the closest lane
        
        return self.lanes[self.get_lane_index(position)]

    def get_lane_index(self, position):
        
            Get the index of the lane closest to a world position.

        :param position: a world position [m]
        :return: the index of the closest lane
        
        # lateral = [abs(l.local_coordinates(position)[1]) for l in self.lanes]
        lateral = []
        for l in self.lanes:
            parameters = l.local_coordinates(position)
            # print("parameters[0]:", parameters[0], "\n")
            # print("parameters[1]:", parameters[1], "\n")
            lateral.append(np.linalg.norm(parameters))
        print("argmin lateral:", int(np.argmin(lateral)), "\n")
        return int(np.argmin(lateral))
    """

    def neighbour_vehicles_bigmap(self, vehicle, lane_index=None):
        """
            Find the preceding and following vehicles of a given vehicle.
        :param vehicle: the vehicle whose neighbours must be found
        :param lane: the lane on which to look for preceding and following vehicles.
                     It doesn't have to be the current vehicle lane but can also be another lane, in which case the
                     vehicle is projected on it considering its local coordinates in the lane.
        :return: its preceding vehicle, its following vehicle
        """
        lane_index = lane_index or vehicle.lane_index
        lanes = {}
        if not lane_index:
            return None, None
        lane = self.network.get_lane(lane_index)
        lanes["current"] = lane
        try:
            if len(lane.before_lane[0]):
                # print("lane_before",lane.before_lane[0])
                lanes["before"] = self.network.get_lane(lane.before_lane[0])
            if len(lane.after_lane[0]):
                lanes["after"] = self.network.get_lane(lane.after_lane[0])
        except IndexError:
            pass
        s = lane.local_coordinates(vehicle.position)[0]
        s_front = s_rear = None
        v_front = v_rear = None
        is_on_cur_lane = False
        is_on_before_lane = False
        is_on_after_lane = False
        for v in self.vehicles:
            if v is not vehicle and self.network.is_connected_road(v.lane_index, lane_index, same_lane=True):
                s_v, lat_v = lane.local_coordinates(v.position)
                if not lane.on_lane(v.position, s_v, lat_v):
                    is_on_cur_lane = False
                else:
                    is_on_cur_lane = True
                if "before" in lanes.keys():
                    s_v_before,lat_v_before = lanes["before"].local_coordinates(v.position)
                    if not lanes["before"].on_lane(v.position, s_v_before, lat_v_before):
                        is_on_before_lane = False
                    else:
                        is_on_before_lane = True
                if "after" in lanes.keys():
                    s_v_after,lat_v_after = lanes["after"].local_coordinates(v.position)
                    if not lanes["after"].on_lane(v.position, s_v_after, lat_v_after):
                        is_on_after_lane = False
                    else:
                        is_on_after_lane = True
                if not is_on_cur_lane and not is_on_before_lane and not is_on_after_lane:
                    continue
                if is_on_cur_lane:
                    if s <= s_v and (s_front is None or s_v <= s_front):
                        s_front = s_v
                        v_front = v
                    if s_v < s and (s_rear is None or s_v > s_rear):
                        s_rear = s_v
                        v_rear = v
                if is_on_before_lane:
                    if s_v_before - lanes["before"].length < s and (s_rear is None or s_v_before - lanes["before"].length > s_rear):
                        s_rear = s_v_before - lanes["before"].length
                        v_rear = v
                if is_on_after_lane:
                    if s <= s_v_after + lanes["current"].length and (s_front is None or s_v_after + lanes["current"].length <= s_front):
                        s_front = s_v_after + lanes["current"].length
                        v_front = v
        return v_front, v_rear

    def neighbour_vehicles(self, vehicle, lane_index=None):
        """
            Find the preceding and following vehicles of a given vehicle.
        :param vehicle: the vehicle whose neighbours must be found
        :param lane: the lane on which to look for preceding and following vehicles.
                     It doesn't have to be the current vehicle lane but can also be another lane, in which case the
                     vehicle is projected on it considering its local coordinates in the lane.
        :return: its preceding vehicle, its following vehicle
        """
        lane_index = lane_index or vehicle.lane_index

        if not lane_index:
            return None, None
        lane = self.network.get_lane(lane_index)

        s = lane.local_coordinates(vehicle.position)[0]
        s_front = s_rear = None
        v_front = v_rear = None

        for v in self.vehicles:
            if v is not vehicle and self.network.is_connected_road(v.lane_index, lane_index, same_lane=True):
                s_v, lat_v = lane.local_coordinates(v.position)
                if not lane.on_lane(v.position, s_v, lat_v):
                    continue
                if s <= s_v and (s_front is None or s_v <= s_front):
                    s_front = s_v
                    v_front = v
                if s_v < s and (s_rear is None or s_v > s_rear):
                    s_rear = s_v
                    v_rear = v
        return v_front, v_rear

    def dump(self):
        """
            Dump the data of all entities on the road
        """
        for v in self.vehicles:
            if not isinstance(v, Obstacle):
                v.dump()

    def get_log(self):
        """
            Concatenate the logs of all entities on the road.
        :return: the concatenated log.
        """
        return pd.concat([v.get_log() for v in self.vehicles])

    def __repr__(self):
        return self.vehicles.__repr__()
