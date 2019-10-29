from __future__ import division, print_function, absolute_import
import numpy as np

from highway_env import utils
from highway_env.envs.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane, LanesConcatenation
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.control import ControlledVehicle, MDPVehicle, CarSim, FreeControl
from highway_env.envs.graphics import EnvViewer
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.dynamics import Obstacle
import time
import random


class MergeEnv(AbstractEnv):
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

    DURATION = 15

    DEFAULT_CONFIG = {"other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                      "incoming_vehicle_destination": None,
                      "other_vehicles_destination": None}

    def __init__(self):
        super(MergeEnv, self).__init__()
        self.config = self.DEFAULT_CONFIG.copy()
        EnvViewer.SCREEN_WIDTH = 2400
        # self.make_road()
        # self.make_road()
        # self.make_vehicles()
        self.steps = 0

    def configure(self, config):
        self.config.update(config)

    def _observation(self):
        return super(MergeEnv, self)._observation()

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
        # print(self.vehicle.position[0],self.vehicle.max_length)
        return self.vehicle.crashed or self.steps >= self.DURATION

    def reset(self):
        self.make_road()
        self.make_vehicles()
        return self._observation()

    # def make_road_from_file(self):
    # pass

    def make_road_from_dict(self, d):
        lanes = []
        for k, v in d.items():
            clz = eval(k)
            lanes.append(clz(**v))
        return LanesConcatenation(lanes)

    def generate_merge(self, begin, amplitude_1, sin_length_1, str_length_1, amplitude_2, sin_length_2, width, reverse):
        l_in_1_sin = SineLane([0, begin], 0, width, -amplitude_1 * reverse, np.pi / (sin_length_1), np.pi / 2,
                              [LineType.CONTINUOUS, LineType.CONTINUOUS], bounds=[0, sin_length_1], forbidden=False)
        l_in_1_str = StraightLane(l_in_1_sin.position(sin_length_1, 0), 0, width,
                                  [LineType.CONTINUOUS_LINE, LineType.CONTINUOUS], bounds=[0, str_length_1],
                                  name='lm12')

        l_in_out_sin = SineLane(l_in_1_str.position(str_length_1, -amplitude_2 * reverse), 0, width,
                                amplitude_2 * reverse, np.pi / (sin_length_2), np.pi / 2,
                                [LineType.CONTINUOUS, LineType.CONTINUOUS], bounds=[0, sin_length_2], forbidden=False)
        l = LanesConcatenation([l_in_1_sin, l_in_1_str, l_in_out_sin])
        return l

    """
    def double_merge(self):
        amp = 3.25
        str_len = 150
        sin_len = 100
        width = 4
        l1 = self.generate_merge(0, amp, sin_len, str_len, amp, sin_len, width, 1)
        l2 = self.generate_merge(width, amp, sin_len, str_len, amp, sin_len, width, 1)
        l3 = self.generate_merge(2 * amp + 2 * width, amp, sin_len, str_len, amp, sin_len, width, -1)
        l4 = self.generate_merge(2 * amp + 3 * width, amp, sin_len, str_len, amp, sin_len, width, -1)
        # l2 = self.generate_merge(0,amp,sin_len,str_len,amp,sin_len,width,1)
        # l3 = self.generate_merge(8+10,amp,50,100,amp,50,4,-1)
        # l4 = self.generate_merge(22,amp,50,100,amp,50,4,-1)
        # amplitude = 3.25
        # l_in_1_sin = SineLane([0,0], 0, 4.0, -amplitude, 2 * np.pi / (200), np.pi / 2,
        #                [LineType.CONTINUOUS, LineType.CONTINUOUS], bounds=[0, 100], forbidden=True)
        # l_in_1_str = StraightLane(l_in_1_sin.position(100, 0), 0, 4.0,
        #                     [LineType.CONTINUOUS_LINE, LineType.STRIPED], bounds=[0, 200],name='lm12')
        #
        # l_in_out_sin = SineLane(l_in_1_str.position(200, -3.25), 0, 4.0, amplitude, 2 * np.pi / (200), np.pi / 2,
        #                [LineType.CONTINUOUS, LineType.CONTINUOUS], bounds=[0, 100], forbidden=True)
        # l1 = LanesConcatenation([l_in_1_sin,l_in_1_str,l_in_out_sin])
        # lm_out_1 = SineLane([0,0], 0, 4.0, -amplitude, 2 * np.pi / (200), np.pi / 2,
        #                    [LineType.CONTINUOUS, LineType.CONTINUOUS], bounds=[0, 100], forbidden=True)
        # lm_out_2 = StraightLane(lm_in_1.position(100, 0), 0, 4.0,
        #                        [LineType.CONTINUOUS_LINE, LineType.STRIPED], bounds=[0, np.inf],name='lm12')
        # lm3 = SineLane([0,4], 0, 4.0, -amplitude, 2 * np.pi / (200), np.pi / 2,
        #                [LineType.CONTINUOUS, LineType.CONTINUOUS], bounds=[0, 100], forbidden=True)
        # lm4 = StraightLane(lm3.position(100, 0), 0, 4.0,
        #                    [LineType.STRIPED, LineType.STRIPED], bounds=[0, np.inf],name='lm12')
        #
        # lc1 = SineLane((0, 14.5), 0, 4.0, amplitude, 2 * np.pi / (200), np.pi / 2,
        #                [LineType.CONTINUOUS, LineType.CONTINUOUS], bounds=[0, 100], forbidden=True)
        # lc2 = StraightLane(lc1.position(100, 0), 0, 4.0,
        #                    [LineType.STRIPED, LineType.STRIPED], bounds=[0, np.inf],name='lc2')
        #
        # lc1 = SineLane((0, 14.5), 0, 4.0, amplitude, 2 * np.pi / (200), np.pi / 2,
        #                [LineType.CONTINUOUS, LineType.CONTINUOUS], bounds=[0, 100], forbidden=True)
        # lc2 = StraightLane(lc1.position(100, 0), 0, 4.0,
        #                    [LineType.STRIPED, LineType.STRIPED], bounds=[0, np.inf],name='lc2')
        # l2 = LanesConcatenation([lm3,lm4])
        # l3 = LanesConcatenation([lc1,lc2])
        self.road = Road([l1, l2, l3, l4])
    """

    def make_road(self):
        """
            Make a road composed of a straight highway and a merging lane.
        :return: the road
        """
        amp = 3.25
        str_len = 150
        sin_len = 100
        width = 4
        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        net.add_lane("s1", "inter1", SineLane([0, 0], [sin_len, 0], -amp * 1, np.pi / sin_len, np.pi / 2,
                                              line_types=[c, c],
                                              forbidden=False))
        net.add_lane("s1", "inter1", SineLane([0, width], [sin_len, width], -amp * 1, np.pi / sin_len, np.pi / 2,
                                              line_types=[c, c],
                                              forbidden=False))
        net.add_lane("s2", "inter1",
                     SineLane([0, 2 * amp + 2 * width], [sin_len, 2 * amp + 2 * width], -amp * -1, np.pi / sin_len,
                              np.pi / 2,
                              line_types=[c, c],
                              forbidden=False))
        net.add_lane("s2", "inter1",
                     SineLane([0, 2 * amp + 3 * width], [sin_len, 2 * amp + 3 * width], -amp * -1, np.pi / sin_len,
                              np.pi / 2,
                              line_types=[c, c],
                              forbidden=False))
        net.add_lane("inter1", "inter2",
                     StraightLane([sin_len, amp], [sin_len + str_len, amp],
                                  line_types=[c, s]))
        net.add_lane("inter1", "inter2",
                     StraightLane([sin_len, amp + width], [sin_len + str_len, amp + width],
                                  line_types=[s, s]))
        net.add_lane("inter1", "inter2",
                     StraightLane([sin_len, amp + 2 * width], [sin_len + str_len, amp + 2 * width],
                                  line_types=[s, s]))
        net.add_lane("inter1", "inter2",
                     StraightLane([sin_len, amp + 3 * width], [sin_len + str_len, amp + 3 * width],
                                  line_types=[s, c]))
        net.add_lane("inter2", "e1",
                     SineLane([sin_len + str_len, 0], [2 * sin_len + str_len, 0], amp * 1, np.pi / sin_len, np.pi / 2,
                              line_types=[c, c],
                              forbidden=False))
        net.add_lane("inter2", "e1",
                     SineLane([sin_len + str_len, width], [2 * sin_len + str_len, width], amp * 1, np.pi / sin_len,
                              np.pi / 2,
                              line_types=[c, c],
                              forbidden=False))
        net.add_lane("inter2", "e2",
                     SineLane([sin_len + str_len, 2 * width + 2 * amp], [2 * sin_len + str_len, 2 * width + 2 * amp],
                              amp * -1, np.pi / sin_len,
                              np.pi / 2,
                              line_types=[c, c],
                              forbidden=False))
        net.add_lane("inter2", "e2",
                     SineLane([sin_len + str_len, 3 * width + 2 * amp], [2 * sin_len + str_len, 3 * width + 2 * amp],
                              amp * -1, np.pi / sin_len,
                              np.pi / 2,
                              line_types=[c, c],
                              forbidden=False))

        road = Road(network=net, np_random=self.np_random)
        # road.vehicles.append(Obstacle(road, [0, 12]))
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
        birth_place = [("s1", "inter1", 0), ("s1", "inter1", 1), ("s2", "inter1", 0), ("s2", "inter1", 1)]
        destinations = ["e1", "e2"]
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
                    ego_lane = self.road.network.get_lane(("s1", "inter1", 1))
                    ego_vehicle = IDMVehicle(self.road,
                                             ego_lane.position(20, 0),
                                             velocity=10,
                                             heading=ego_lane.heading_at(0)).plan_route_to("e2")
                    # print("ego_route:", ego_vehicle.route, "\n")
                    # print("ego_relative_offset:",ego_vehicle.lane.local_coordinates(ego_vehicle.position)[1])
                    ego_vehicle.id = 0
                    road.vehicles.append(ego_vehicle)
                    self.vehicle = ego_vehicle
                else:
                    car = other_vehicles_type.make_on_lane(road, birth_place[np.random.randint(0, 4)],
                                                           longitudinal=0 + np.random.randint(1,
                                                                                              5) * position_deviation,
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
        for i in range(self.road.network.LANES_NUMBER):
            lane = road.network.get_lane(self.road.network.LANES[i])
            # print("lane:", lane.LANEINDEX, "\n")
            lane.vehicles = sorted(lane.vehicles, key=lambda x: lane.local_coordinates(x.position)[0])
            # print("len of lane.vehicles:", len(lane.vehicles), "\n")
            for j, v in enumerate(lane.vehicles):
                # print("i:",i,"\n")
                v.vehicle_index_in_line = j
        #
        # merging_v.target_velocity = 30
        # road.vehicles.append(merging_v)

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

        for i in range(1):
            birth_place = [("s1", "inter1", 0), ("s1", "inter1", 1), ("s2", "inter1", 0), ("s2", "inter1", 1)]
            destinations = ["e1", "e2"]
            position_deviation = 5
            velocity_deviation = 1.5
            other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
            birth = birth_place[np.random.randint(0, 4)]
            lane = self.road.network.get_lane(birth)
            car = other_vehicles_type.make_on_lane(self.road, birth,
                                                   longitudinal=0 + np.random.randint(1,
                                                                                       5) * position_deviation,
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
        return terminal, extractor_features


if __name__ == '__main__':
    pass
