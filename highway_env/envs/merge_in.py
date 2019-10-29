from __future__ import division, print_function, absolute_import
import numpy as np

from highway_env import utils
from highway_env.envs.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane, LanesConcatenation
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.control import ControlledVehicle, MDPVehicle, CarSim, FreeControl
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.dynamics import Obstacle
import time
import random


class MergeEnvIn(AbstractEnv):
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
        super(MergeEnvIn, self).__init__()
        self.config = self.DEFAULT_CONFIG.copy()
        self.steps = 0
        # self.make_road()
        # self.make_roads()
        # self.double_merge()
        # self.make_vehicles()

    def configure(self, config):
        self.config.update(config)

    def _observation(self):
        return super(MergeEnvIn, self)._observation()

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
        self.make_roads()
        self.make_vehicles()
        return self._observation()

    def make_roads(self):
        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        net.add_lane("s1", "ex", StraightLane(np.array([0, 0]), np.array([100, 0]), line_types=[c, s]))
        net.add_lane("ex", "em", StraightLane(np.array([100, 0]), np.array([200, 0]), line_types=[c, s]))
        net.add_lane("em", "x1", StraightLane(np.array([200, 0]), np.array([300, 0]), line_types=[c, s]))
        # lm10 = StraightLane(np.array([0, 0]), 0, 4.0, [LineType.CONTINUOUS_LINE, LineType.STRIPED],bounds=[0,300])
        # l1 = LanesConcatenation([lm10])
        net.add_lane("s1", "ex", StraightLane(np.array([0, 4]), np.array([100, 4]), line_types=[s, s]))
        net.add_lane("ex", "em", StraightLane(np.array([100, 4]), np.array([200, 4]), line_types=[s, s]))
        net.add_lane("em", "x1", StraightLane(np.array([200, 4]), np.array([300, 4]), line_types=[s, s]))
        # lm20 = StraightLane(l1.position(0,4), 0, 4.0, [LineType.STRIPED, LineType.STRIPED],bounds=[0,300])
        # l2 = LanesConcatenation([lm20])
        net.add_lane("s1", "ex", StraightLane(np.array([0, 8]), np.array([100, 8]), line_types=[s, c]))
        # lm30 = StraightLane(l2.position(0,4), 0, 4.0, [LineType.STRIPED, LineType.CONTINUOUS_LINE],bounds=[0,100])
        net.add_lane("ex", "em", StraightLane(np.array([100, 8]), np.array([200, 8]), line_types=[s, n]))
        net.add_lane("em", "x1", StraightLane(np.array([200, 8]), np.array([300, 8]), line_types=[s, c]))
        # lm31 = StraightLane(lm30.position(0,0), 0, 4.0, [LineType.STRIPED, LineType.STRIPED],bounds=[0,300])
        # l3 = LanesConcatenation([lm30,lm31])
        amplitude = 4.5
        net.add_lane("s2", "ee", StraightLane(np.array([0, 8 + 3 * amplitude]), np.array([50, 8 + 3 * amplitude]),
                                              line_types=[c, c], forbidden=True))
        # lm40 = StraightLane(l3.position(0,2*amplitude+4), 0, 4.0, [LineType.CONTINUOUS_LINE, LineType.CONTINUOUS_LINE],bounds=[0,50])
        net.add_lane("ee", "ex",
                     SineLane(np.array([50, 8 + 2 * amplitude]), np.array([100, 8 + 2 * amplitude]), amplitude,
                              2 * np.pi / (2 * 50), np.pi / 2, line_types=[c, c], forbidden=True))
        # lm41 = SineLane(lm40.position(50, -amplitude), 0, 4.0, amplitude, 2 * np.pi / (2*50), np.pi / 2,
        # [LineType.CONTINUOUS, LineType.CONTINUOUS], bounds=[0, 50], forbidden=True)
        net.add_lane("ex", "over",
                     StraightLane(np.array([100, 8 + amplitude]), np.array([200, 8 + amplitude]), line_types=[s, c],
                                  forbidden=True))
        # net.add_lane("over", "x1",
                     # SineLane(np.array([200, 8 +  amplitude/2]), np.array([220, 8 + amplitude/2]), amplitude/2,
                              # 2 * np.pi / (2 * 50), np.pi / 2, line_types=[c, c], forbidden=True))
        # lm42 = StraightLane(lm41.position(50,0), 0, 4.0, [LineType.STRIPED, LineType.CONTINUOUS_LINE],bounds=[0,150],forbidden=True)
        # l4 = LanesConcatenation([lm40,lm41,lm42])
        # road = Road([l1,l2,l3,l4])
        # road = Road([ l3])

        # road = Road([lm0,lm2])
        road = Road(network=net, np_random=self.np_random)
        road.vehicles.append(Obstacle(road, [200, 8 + amplitude]))
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
        # reset_position_range = (20,40)
        # reset_lane = random.choice(road.lanes)
        reset_lane = ("s2", "ee", 0)
        ego_vehicle = None
        birth_place = [("s1", "ex", 0), ("s1", "ex", 1), ("s1", "ex", 2), ("s2", "ee", 0)]
        destinations = ["x1"]
        position_deviation = 10
        velocity_deviation = 2
        for l in self.road.network.LANES:
            lane = road.network.get_lane(l)
            cars_on_lane = car_number_each_lane
            reset_position = None
            if l == reset_lane:
                cars_on_lane += 1
                reset_position = random.choice(range(1, car_number_each_lane))
            for i in range(cars_on_lane):
                if i == reset_position and not ego_vehicle:
                    ego_lane = self.road.network.get_lane(("s2", "ee", 0))
                    ego_vehicle = ControlledVehicle(self.road,
                                             ego_lane.position(0, 0),
                                             velocity=20,
                                             heading=ego_lane.heading_at(0)).plan_route_to("x1")
                    ego_vehicle.id = 0
                    road.vehicles.append(ego_vehicle)
                    self.vehicle = ego_vehicle
                else:
                    car = other_vehicles_type.make_on_lane(road, birth_place[np.random.randint(0, 4)],
                                                           longitudinal=5 + np.random.randint(1,
                                                                                              7) * position_deviation,
                                                           velocity=5 + np.random.randint(1, 5) * velocity_deviation)
                    destination = destinations[0]
                    # print("destination:",destination)
                    car.plan_route_to(destination)
                    car.randomize_behavior()
                    road.vehicles.append(car)
                    lane.vehicles.append(car)
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
            birth_place = [("s1", "ex", 0), ("s1", "ex", 1), ("s1", "ex", 2), ("s2", "ee", 0)]
            destinations = ["x1"]
            # position_deviation = 5
            velocity_deviation = 1.5
            other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
            birth = birth_place[np.random.randint(0, 4)]
            lane = self.road.network.get_lane(birth)
            car = other_vehicles_type.make_on_lane(self.road, birth,
                                                   longitudinal=0,
                                                   velocity=5 + np.random.randint(1, 10) * velocity_deviation)

            destination = destinations[0]
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
