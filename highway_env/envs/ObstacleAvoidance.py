from __future__ import division, print_function, absolute_import
import numpy as np

from highway_env import utils
from highway_env.envs.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane, CircularLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle,MPCControlledVehicle
from highway_env.vehicle.dynamics import RedLight, Obstacle
from highway_env.envs.graphics import EnvViewer

import threading
import time
# from highway_env import _SOCKET


def rad(deg):
    return deg * np.pi / 180


class ObstacleAvoidanceEnv(AbstractEnv):
    """
        A highway merge negotiation environment.

        The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
        It is rewarded for maintaining a high velocity and avoiding collisions, but also making room for merging
        vehicles.
    """

    DEFAULT_CONFIG = {"other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                      "incoming_vehicle_destination": None,
                      "other_vehicles_destination": None}

    def __init__(self):
        super(ObstacleAvoidanceEnv, self).__init__()
        self.SIMULATION_FREQUENCY = 100
        self.config = self.DEFAULT_CONFIG.copy()
        self.steps = 0
        self.traffic_lights = {}
        self.obstacles = []
        EnvViewer.SCREEN_HEIGHT = 500
        EnvViewer.SCREEN_WIDTH = 500
        self.exit = ["inter_n", "inter_e", "inter_w"]
        self.birth_place = [("se", "inters", 0), ("se", "inters", 1), ("se", "inters", 2),
                            ("inters","sel",0),("inters","sem",0),("inters","ser",0)]
        self.destination_trafficlight = None
        self.destination = None
        # self.server = _SOCKET.socket_init()
        self.vehicle_actions = None
        self.mission_completed = True
        self.vehicle_prelane = None

    def configure(self, config):
        self.config.update(config)

    def _observation(self):
        return super(ObstacleAvoidanceEnv, self)._observation()

    # def _reward(self, action):
    #     """
    #         The vehicle is rewarded for driving with high velocity on lanes to the right and avoiding collisions, but
    #         an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low velocity.
    #     :param action: the action performed
    #     :return: the reward of the state-action transition
    #     """
    #     action_reward = {0: self.LANE_CHANGE_REWARD,
    #                      1: 0,
    #                      2: self.LANE_CHANGE_REWARD,
    #                      3: 0,
    #                      4: 0}
    #     reward = self.COLLISION_REWARD * self.vehicle.crashed \
    #              + self.RIGHT_LANE_REWARD * self.vehicle.lane_index / (len(self.road.lanes) - 2) \
    #              + self.HIGH_VELOCITY_REWARD * self.vehicle.velocity_index / (self.vehicle.SPEED_COUNT - 1)
    #
    #     # Altruistic penalty
    #     for vehicle in self.road.vehicles:
    #         if vehicle.lane_index == len(self.road.lanes) - 1 and isinstance(vehicle, ControlledVehicle):
    #             reward += self.MERGING_VELOCITY_REWARD * \
    #                       (vehicle.target_velocity - vehicle.velocity) / vehicle.target_velocity
    #     return reward + action_reward[action]

    def _is_terminal(self):
        """
            The episode is over when a collision occurs or when the access ramp has been passed.
        """
        return self.vehicle.crashed

    def reset(self):
        self.make_roads()
        self.make_vehicles()
        self._observation()
        # self.socket_thread = threading.Thread(target=_SOCKET.socket_thread, args=(self,), name="socket")
        # self.socket_thread.start()

    def make_roads(self):
        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        """
        e:entrance
        before_inter:a node before inter_node
        inter: stripe lines --> solid lines
        """

        net.add_lane("se", "inters", StraightLane(np.array([0, 0]), np.array([0, -70]), line_types=[c, s]))
        net.add_lane("se", "inters", StraightLane(np.array([4, 0]), np.array([4, -70]), line_types=[s, s]))
        net.add_lane("se", "inters", StraightLane(np.array([8, 0]), np.array([8, -70]), line_types=[s, c]))

        # net.add_lane("before_inters", "inters", StraightLane(np.array([0, -60]), np.array([0, -70]), line_types=[c, s]))
        # net.add_lane("before_inters", "inters", StraightLane(np.array([4, -60]), np.array([4, -70]), line_types=[s, s]))
        # net.add_lane("before_inters", "inters", StraightLane(np.array([8, -60]), np.array([8, -70]), line_types=[s, c]))

        net.add_lane("inters", "sel",
                     StraightLane(np.array([0, -70]), np.array([0, -100]), line_types=[c, c], forbidden=True))
        net.add_lane("inters", "sem",
                     StraightLane(np.array([4, -70]), np.array([4, -100]), line_types=[c, c], forbidden=True))
        net.add_lane("inters", "ser",
                     StraightLane(np.array([8, -70]), np.array([8, -100]), line_types=[c, c], forbidden=True))

        net.add_lane("nxl", "inter_n", StraightLane(np.array([0, -130]), np.array([0, -150]), line_types=[c, s]))
        net.add_lane("nxm", "inter_n", StraightLane(np.array([4, -130]), np.array([4, -150]), line_types=[s, s]))
        net.add_lane("nxr", "inter_n", StraightLane(np.array([8, -130]), np.array([8, -150]), line_types=[s, c]))

        net.add_lane("interw", "wel", StraightLane(np.array([-30, -110]), np.array([-6, -110]), line_types=[c, c]))
        net.add_lane("interw", "wer", StraightLane(np.array([-30, -106]), np.array([-6, -106]), line_types=[c, c]))
        net.add_lane("wxl", "inter_w", StraightLane(np.array([-6, -118]), np.array([-30, -118]), line_types=[c, s]))
        net.add_lane("wxr", "inter_w", StraightLane(np.array([-6, -122]), np.array([-30, -122]), line_types=[s, c]))

        net.add_lane("intere", "eel", StraightLane(np.array([40, -118]), np.array([14, -118]), line_types=[c, c]))
        net.add_lane("intere", "eer", StraightLane(np.array([40, -122]), np.array([14, -122]), line_types=[c, c]))
        net.add_lane("exl", "inter_e", StraightLane(np.array([14, -110]), np.array([40, -110]), line_types=[c, s]))
        net.add_lane("exr", "inter_e", StraightLane(np.array([14, -106]), np.array([40, -106]), line_types=[s, c]))

        net.add_lane("sel", "wxl", StraightLane(np.array([0, -100]), np.array([-6, -118]), line_types=[n, n]))
        net.add_lane("sem", "nxm", StraightLane(np.array([4, -100]), np.array([4, -130]), line_types=[n, n]))
        net.add_lane("ser", "exr", StraightLane(np.array([8, -100]), np.array([14, -106]), line_types=[n, n]))

        road = Road(network=net, np_random=self.np_random)

        green_time = 5
        red_time = 8
        green_flash_time = 2
        yellow_time = 1
        """
        southwest crossroad traffic lights
        """
        self.traffic_lights = [
            RedLight(road, [0, -100], red_time, green_time, green_flash_time, yellow_time, 1),
            RedLight(road, [4, -100], red_time, green_time, green_flash_time, yellow_time, 1),
            RedLight(road, [8, -100], 0, 10, 0, 0, 1),
        ]

        self.road = road
        self.road.vehicles = self.traffic_lights

    def make_vehicles(self):
        """
            Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        """
        # road = self.road
        ego_birth = [("se", "inters", 0), ("se", "inters", 1), ("se", "inters", 2)]
        ego_lane_index = ego_birth[0]
        ego_lane = self.road.network.get_lane(ego_lane_index)
        position = ego_lane.position(0, 0)
        self.destination = self.exit[2]
        ego_vehicle = MPCControlledVehicle(self.road,
                                 position,
                                 velocity=10,
                                 heading=ego_lane.heading_at(position)).plan_route_to(self.destination)
        if self.destination == "inter_w":
            self.destination_trafficlight = self.traffic_lights[0]
        elif self.destination == "inter_n":
            self.destination_trafficlight = self.traffic_lights[1]
        elif self.destination == "inter_e":
            self.destination_trafficlight = self.traffic_lights[2]
        ego_vehicle.id = 0
        # ego_vehicle1.myimage = pygame.image.load("../red_alpha_resize.png")
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle
        # print("vehicle lane_index:", self.vehicle.lane_index)
        lanes = [("se", "inters", 0), ("se", "inters", 1), ("se", "inters", 2)]
        lanes.remove(ego_lane_index)
        lane_1 = ego_lane
        lane_2 = self.road.network.get_lane(lanes[0])
        lane_3 = self.road.network.get_lane(lanes[1])
        obstacle_1 = Obstacle(self.road, lane_1.position(np.random.randint(10, ego_lane.length-5), -1), LENGTH=2.0,
                             WIDTH=4.0)
        # obstacle_2 = Obstacle(self.road, lane_2.position(np.random.randint(10, ego_lane.length-5), 0))
        # obstacle_3 = Obstacle(self.road, lane_3.position(np.random.randint(10, ego_lane.length-5), 0))
        self.road.vehicles.extend([obstacle_1])
        self.obstacles.extend([ obstacle_1])

    def fake_step(self):
        """
        :return:
        """
        active_vehicles = 0

        flag = 1
        while flag:
            for i in range(len(self.road.vehicles)):
                if i == len(self.road.vehicles) - 1:
                    flag = 0
                if hasattr(self.road.vehicles[i], "state"):
                    continue
                else:
                    if self.road.vehicles[i].lane_index[1] in self.exit:
                        # print("lane_index",self.road.vehicles[i].lane_index[1])
                        lane = self.road.network.get_lane(self.road.vehicles[i].lane_index)
                        s, _ = lane.local_coordinates(self.road.vehicles[i].position)
                        if s >= lane.length * 0.75:
                            self.road.vehicles.remove(self.road.vehicles[i])
                            break
                    else:
                        active_vehicles += 1

        # self.get_traffic_lights()

        for k in range(int(self.SIMULATION_FREQUENCY // self.POLICY_FREQUENCY)):
            # self._socket(connection)

            self.road.act(vehicle_actions=self.vehicle_actions)
            self.road.step(1 / self.SIMULATION_FREQUENCY)

            # Automatically render intermediate simulation steps if a viewer has been launched
            self._automatic_rendering()

            # Stop at terminal states
            if self.done or self._is_terminal():
                break
        self.enable_auto_render = False
        self.steps += 1
        # from highway_env.extractors import Extractor
        # extractor = Extractor()
        # extractor_features = extractor.FeatureExtractor(self.road.vehicles, 0, 1)

        # flag = 1
        # while flag:
        #     for i in range(len(birth_place)):
        #         if i == len(birth_place) - 1:
        #             flag = 0
        #         if birth_place[i][1].find("inter") == -1:
        #             birth_place.remove(birth_place[i])
        #             break
        pre_birth = None
        for i in range(3):
            try:
                velocity_deviation = 1.0
                velocity = 5 + np.random.randint(1, 5) * velocity_deviation
                other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

                birth = self.birth_place[np.random.randint(0, len(self.birth_place))]
                lane = self.road.network.get_lane(birth)
                if pre_birth != birth and active_vehicles <= 10 and np.linalg.norm(
                        lane.start - self.vehicle.position) >= self.vehicle.LENGTH + self.vehicle.DISTANCE_WANTED + self.vehicle.TIME_WANTED * velocity:
                    car = other_vehicles_type.make_on_lane(self.road, birth,
                                                           longitudinal=0,
                                                           velocity=velocity)
                    # if self.config["incoming_vehicle_destination"] is not None:
                    #     destination = self.end[self.config["incoming_vehicle_destination"]]
                    # else:
                    #     destination = self.end[np.random.randint(0, len(self.end))]
                    # car.plan_route_to(destination)
                    # car.randomize_behavior()
                    self.road.vehicles.append(car)
                    lane.vehicles.append(car)
                    pre_birth = birth
            except:
                pass
        # obs = self._observation()
        # reward = self._reward(action)
        terminal = self._is_terminal()
        # info = {}
        return terminal  # , extractor_features


if __name__ == '__main__':
    pass
