from __future__ import division, print_function, absolute_import
import numpy as np

from highway_env import utils
from highway_env.envs.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane, LanesConcatenation
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.control import ControlledVehicle, MDPVehicle, CarSim, FreeControl
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.dynamics import RedLight
import time
import random


class MergeEnvOut(AbstractEnv):
    """
        A highway merge negotiation environment.

        The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
        It is rewarded for maintaining a high velocity and avoiding collisions, but also making room for merging
        vehicles.
    """

    COLLISION_REWARD = -1
    RIGHT_LANE_REWARD = 0.1
    HIGH_VELOCITY_REWARD = 0.2
    MERGING_VELOCITY_REWARD = -0.5
    LANE_CHANGE_REWARD = -0.05

    DEFAULT_CONFIG = {"other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                      "incoming_vehicle_destination": None,
                      "other_vehicles_destination": None}

    def __init__(self):
        super(MergeEnvOut, self).__init__()
        self.config = self.DEFAULT_CONFIG.copy()
        self.steps = 0
        # self.make_road()
        # self.reset()
        # self.double_merge()
        # self.make_vehicles()

    def configure(self, config):
        self.config.update(config)

    def _observation(self):
        return super(MergeEnvOut, self)._observation()

    def _reward(self, action):
        """
            The vehicle is rewarded for driving with high velocity on lanes to the right and avoiding collisions, but
            an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low velocity.
        :param action: the action performed
        :return: the reward of the state-action transition
        """
        action_reward = {0: self.LANE_CHANGE_REWARD,
                         1: 0,
                         2: self.LANE_CHANGE_REWARD,
                         3: 0,
                         4: 0}
        reward = self.COLLISION_REWARD * self.vehicle.crashed \
                 + self.RIGHT_LANE_REWARD * self.vehicle.lane_index / (len(self.road.lanes) - 2) \
                 + self.HIGH_VELOCITY_REWARD * self.vehicle.velocity_index / (self.vehicle.SPEED_COUNT - 1)

        # Altruistic penalty
        for vehicle in self.road.vehicles:
            if vehicle.lane_index == len(self.road.lanes) - 1 and isinstance(vehicle, ControlledVehicle):
                reward += self.MERGING_VELOCITY_REWARD * \
                          (vehicle.target_velocity - vehicle.velocity) / vehicle.target_velocity
        return reward + action_reward[action]

    def _is_terminal(self):
        """
            The episode is over when a collision occurs or when the access ramp has been passed.
        """
        return self.vehicle.crashed or self.vehicle.position[0] > 300

    def reset(self):
        # self.make_road()
        print("enter reset")
        self.make_roads()
        self.make_vehicles()
        return self._observation()

    def make_roads(self):
        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED

        net.add_lane("s1", "inter1", StraightLane(np.array([0, 0]), np.array([100, 0]), line_types=[c, s]))
        net.add_lane("inter1", "inter2", StraightLane(np.array([100, 0]), np.array([150, 0]), line_types=[c, s]))
        net.add_lane("inter2", "inter3", StraightLane(np.array([150, 0]), np.array([200, 0]), line_types=[c, s]))
        net.add_lane("inter3", "x1", StraightLane(np.array([200, 0]), np.array([300, 0]), line_types=[c, s]))

        net.add_lane("s1", "inter1", StraightLane(np.array([0, 4]), np.array([100, 4]), line_types=[s, s]))
        net.add_lane("inter1", "inter2", StraightLane(np.array([100, 4]), np.array([150, 4]), line_types=[s, s]))
        net.add_lane("inter2", "inter3", StraightLane(np.array([150, 4]), np.array([200, 4]), line_types=[s, s]))
        net.add_lane("inter3", "x1", StraightLane(np.array([200, 4]), np.array([300, 4]), line_types=[s, s]))

        net.add_lane("s1", "inter1", StraightLane(np.array([0, 8]), np.array([100, 8]), line_types=[s, s]))
        net.add_lane("inter1", "inter2", StraightLane(np.array([100, 8]), np.array([150, 8]), line_types=[s, s]))
        net.add_lane("inter2", "inter3", StraightLane(np.array([150, 8]), np.array([200, 8]), line_types=[s, c]))
        net.add_lane("inter3", "x1", StraightLane(np.array([200, 8]), np.array([300, 8]), line_types=[s, c]))

        amplitude = 4.5
        net.add_lane("s1", "inter1", StraightLane(np.array([0, 12]), np.array([100, 12]), line_types=[s, c]))
        net.add_lane("inter1", "inter2", StraightLane(np.array([100, 12]), np.array([150, 12]), line_types=[s, c]))
        net.add_lane("inter2", "ee", StraightLane(np.array([150, 12]), np.array([200, 12]), line_types=[s, c],forbidden=True))
        net.add_lane("ee", "ex",
                     SineLane(np.array([200, 12 + amplitude]), np.array([250, 12 + amplitude]), -amplitude,
                              2 * np.pi / (2 * 50), np.pi / 2, line_types=[c, c], forbidden=True))
        net.add_lane("ex", "x2",
                     StraightLane(np.array([250, 17 + amplitude]), np.array([300, 17 + amplitude]), line_types=[c, c],
                                  forbidden=True))

        road = Road(network=net, np_random=self.np_random)
        # road.vehicles.append(RedLight(road, [150, 0]))
        # road.vehicles.append(RedLight(road, [150, 4]))
        # road.vehicles.append(RedLight(road, [150, 8]))
        self.road = road

    def make_vehicles(self):
        """
            Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        """
        max_l = 300
        road = self.road
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        car_number_each_lane = 2
        # reset_position_range = (30, 40)
        # reset_lane = random.choice(road.lanes)
        reset_lane = ("s1", "inter1", 1)
        ego_vehicle = None
        birth_place = [("s1", "inter1", 0), ("s1", "inter1", 1), ("s1", "inter1", 2), ("s1", "inter1", 3)]
        destinations = ["x1", "x2"]
        position_deviation = 10
        velocity_deviation = 2
        # print("graph:", self.road.network.graph, "\n")
        for l in self.road.network.LANES:

            lane = road.network.get_lane(l)
            cars_on_lane = car_number_each_lane
            reset_position = None
            if l == reset_lane:
                # print("enter l==reset_lane")
                cars_on_lane += 1
                reset_position = random.choice(range(1, car_number_each_lane))
                # reset_position = 2
            for i in range(cars_on_lane):
                if i == reset_position and not ego_vehicle:
                    ego_lane = self.road.network.get_lane(("s1", "inter1", 1))
                    ego_vehicle = IDMVehicle(self.road,
                                             ego_lane.position(0, 1),
                                             velocity=10,
                                             heading=ego_lane.heading_at(0)).plan_route_to("x2")
                    # print("ego_route:", ego_vehicle.route, "\n")
                    # print("ego_relative_offset:",ego_vehicle.lane.local_coordinates(ego_vehicle.position)[1])
                    ego_vehicle.id = 0
                    road.vehicles.append(ego_vehicle)
                    self.vehicle = ego_vehicle
                else:
                    car = other_vehicles_type.make_on_lane(road, birth_place[np.random.randint(0, 4)],
                                                           longitudinal=5 + np.random.randint(1,
                                                                                              10) * position_deviation,
                                                           velocity=5 + np.random.randint(1, 5) * velocity_deviation)
                    if self.config["other_vehicles_destination"] is not None:
                        destination = destinations[self.config["other_vehicles_destination"]]
                    else:
                        destination = destinations[np.random.randint(0, 2)]
                    # print("destination:",destination)
                    car.plan_route_to(destination)
                    car.randomize_behavior()
                    road.vehicles.append(car)
                    lane.vehicles.append(car)
                    # road.vehicles.append(
                    # other_vehicles_type(road, l.position((i + 1) * np.random.randint(*reset_position_range), 0),
                    # velocity=np.random.randint(18, 25), dst=3, max_length=max_l))
        # for l in road.lanes[3:]:
        #     cars_on_lane = car_number_each_lane
        #     reset_position = None
        #     if l is reset_lane:
        #         cars_on_lane+=1
        #         reset_position = random.choice(range(1,car_number_each_lane))
        #     for i in range(cars_on_lane):
        #         if i == reset_position:
        #             ego_vehicle = ControlledVehicle(road, l.position((i+1) * np.random.randint(*reset_position_range), 0), velocity=20,max_length=max_l)
        #             road.vehicles.append(ego_vehicle)
        #             self.vehicle = ego_vehicle
        #         else:
        #             road.vehicles.append(other_vehicles_type(road, l.position((i+1) * np.random.randint(*reset_position_range), 0), velocity=np.random.randint(18,25),dst=2,rever=True,max_length=max_l))
        for i in range(self.road.network.LANES_NUMBER):
            lane = road.network.get_lane(self.road.network.LANES[i])
            # print("lane:", lane.LANEINDEX, "\n")
            lane.vehicles = sorted(lane.vehicles, key=lambda x: lane.local_coordinates(x.position)[0])
            # print("len of lane.vehicles:", len(lane.vehicles), "\n")
            for j, v in enumerate(lane.vehicles):
                # print("i:",i,"\n")
                v.vehicle_index_in_line = j
    def fake_step(self):
        """
        :return:
        """
        for k in range(int(self.SIMULATION_FREQUENCY // self.POLICY_FREQUENCY)):
            self.road.act()
            self.road.step(1 / self.SIMULATION_FREQUENCY)

            # Automatically render intermediate simulation steps if a viewer has been launched
            self._automatic_rendering()

            # Stop at terminal states
            if self.done or self._is_terminal():
                break

        self.enable_auto_render = False
        self.steps += 1
        from highway_env.extractors import Extractor
        extractor = Extractor()
        extractor_features = extractor.FeatureExtractor(self.road.vehicles, 0, 1)

        for i in range(2):
            birth_place = [("s1", "inter1", 0), ("s1", "inter1", 1), ("s1", "inter1", 2), ("s1", "inter1", 3)]
            destinations = ["x1", "x2"]
            # position_deviation = 5
            velocity_deviation = 1.5
            other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
            birth = birth_place[np.random.randint(0, 4)]
            lane = self.road.network.get_lane(birth)
            car = other_vehicles_type.make_on_lane(self.road, birth,
                                                   longitudinal=0,
                                                   velocity=5 + np.random.randint(1, 10) * velocity_deviation)
            if self.config["incoming_vehicle_destination"] is not None:
                destination = destinations[self.config["incoming_vehicle_destination"]]
            else:
                destination = destinations[np.random.randint(0, 2)]
            car.plan_route_to(destination)
            car.randomize_behavior()
            self.road.vehicles.append(car)
            lane.vehicles.append(car)

        # obs = self._observation()
        # reward = self._reward(action)
        terminal = self._is_terminal()
        info = {}
        return terminal,extractor_features


if __name__ == '__main__':
    pass
