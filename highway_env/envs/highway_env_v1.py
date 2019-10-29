from __future__ import division, print_function, absolute_import
import numpy as np
from gym import logger
from gym import spaces
from highway_env import utils
from highway_env.envs.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle, Control_IDMVehicle
from highway_env.vehicle.control import MDPVehicle, ControlledVehicle, CarSim
from highway_env.utils import DIFFICULTY_LEVELS
from highway_env.vehicle.dynamics import Vehicle
from highway_env.extractors import Extractor
import pygame

ext = Extractor()


class HighwayEnv_v1(AbstractEnv):
    """
        A highway driving environment.For Reinforcement learning testing.

        The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high velocity,
        staying on the rightmost lanes and avoiding collisions.
    """

    COLLISION_REWARD = -10
    """ The reward received when colliding with a vehicle."""
    # RIGHT_LANE_REWARD = 0.1
    """ The reward received when driving on the right-most lanes, linearly mapped to zero for other lanes."""
    HIGH_VELOCITY_REWARD = 10.0
    """ The reward received when driving at full speed, linearly mapped to zero for lower speeds."""
    LANE_CHANGE_REWARD = 1.0
    """ The reward received at each lane change action."""
    DISTANCE_REWARD = 0.1
    """egocar tend to keep away from other cars to keep safe"""
    MOVE_REWARD = 0.1
    """ The reward received at each time step for how far the ego-car move in the last time step."""
    RESTRICT_REWARD = -1.0
    """ The reward received at each time step for the decision of ego-car move cannot be execute."""
    diff_lane_discount = 0.1
    road_end = 8000
    observation_size = 66
    # set egocar type
    vehicle_type = Control_IDMVehicle

    # vehicle_type = IDMVehicle

    def __init__(self):
        super(HighwayEnv_v1, self).__init__()
        # for distance reward
        self.distance = 0
        # action and state space
        self.action_space = spaces.Discrete(3)  # 0, 1, 2
        self.observation_space = spaces.Box(low=0, high=self.road_end, shape=(self.observation_size,))

        self.DIFFICULTY_LEVELS = DIFFICULTY_LEVELS
        self.config = self.DIFFICULTY_LEVELS["ORIGIN"].copy()
        # self.steps = 0
        self.reset()
        Vehicle.ID = 1

    def reset(self):
        Vehicle.ID = 1
        print('this egocar is', self.vehicle_type)
        self._create_road()
        self._create_vehicles()
        # print(m.move_carsim(
        #     {
        #         "command": "move_object",
        #         "object_id": str(len(self.road.vehicles)-1),
        #         "location": self.vehicle.position.tolist() + [0]
        #     }
        # ))
        self.steps = 0
        return self._observation()

    # def step(self, action):
    #     # self.steps += 1
    #     return super(HighwayEnv, self).step(action)

    def step(self, action):
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        # Forward action to the Control_IDMVehicle
        if isinstance(self.vehicle, Control_IDMVehicle):
            # print('action:', action)
            self.vehicle.act_control(self.ACTIONS[action])

        # Simulate
        for k in range(int(self.SIMULATION_FREQUENCY // self.POLICY_FREQUENCY)):
            self.road.act()
            self.road.step(1 / self.SIMULATION_FREQUENCY)

            # Automatically render intermediate simulation steps if a viewer has been launched
            self._automatic_rendering()

            # Stop at terminal states
            if self.done or self._is_terminal():
                break

        self.enable_auto_render = False

        obs = self._observation()
        reward = self._reward(action)
        terminal = self._is_terminal()
        info = {}
        self.steps += 1

        return obs, reward, terminal, info

    def set_difficulty_level(self, level):
        if level in self.DIFFICULTY_LEVELS:
            logger.info("Set difficulty level to: {}".format(level))
            self.config.update(self.DIFFICULTY_LEVELS[level])
            self.reset()
        else:
            raise ValueError("Invalid difficulty level, choose among {}".format(str(self.DIFFICULTY_LEVELS.keys())))

    def configure(self, config):
        self.config.update(config)

    def _create_road(self):
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"]),
                         np_random=self.np_random)
        # # IDMegocar
        # vehicle = IDMVehicle.create_random(road, 30)
        # control IDM egocar
        # vehicle = Control_IDMVehicle.create_random(road, 30)

        # set road end

        # road.vehicles[len(road.vehicles) // 2].velocity=50
        # return road, road.vehicles[len(road.vehicles) // 2]

    def _create_vehicles(self):
        """
            Create some new random vehicles of a given type, and add them on the road.
        """
        vehicle = self.vehicle_type.create_random(self.road, 30)
        vehicle.myimage = pygame.image.load("../red_alpha_resize.png")
        vehicle.id = 0
        vehicle.target_velocity = 50
        # vehicle.position = [0,0]

        self.road.vehicles.append(vehicle)
        # self.vehicle = MDPVehicle.create_random(self.road, 25, spacing=self.config["initial_spacing"])
        # self.road.vehicles.append(self.vehicle)

        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        for _ in range(self.config["vehicles_count"]):
            self.road.vehicles.append(vehicles_type.create_random(self.road))

        for v in self.road.vehicles:
            v.max_length = self.road_end
        self.vehicle = vehicle

    def _reward(self, action):
        # print('self.vehicle.out_of_restrict', self.vehicle.out_of_restrict)
        # assert not self.vehicle.crashed
        state_reward = self.MOVE_REWARD * (self.vehicle.position[0] - self.vehicle.last_position[
            0]) + self.COLLISION_REWARD * self.vehicle.crashed
        if isinstance(self.vehicle_type, Control_IDMVehicle):
            state_reward += self.RESTRICT_REWARD * self.vehicle.out_of_restrict \
                            + self.LANE_CHANGE_REWARD * self.vehicle.lane_change
        state_reward += self.vehicle.velocity / self.vehicle.target_velocity * self.HIGH_VELOCITY_REWARD

        self.vehicle.last_position[0] = self.vehicle.position[0]
        self.vehicle.last_position[1] = self.vehicle.position[1]

        # state_reward = self.HIGH_VELOCITY_REWARD * (self.vehicle.velocity - self.vehicle.target_velocity)\
        #     + self.COLLISION_REWARD * self.vehicle.crashed \
        #     # + self.DISTANCE_REWARD * self.distance

        return state_reward

    """
    def _observation(self):
        features = np.zeros(self.observation_size)
        num_neighbours = 0
        distance = 0
        #egocar
        features[0] = self.vehicle.velocity
        features[1] = self.vehicle.position[0]
        features[2] = self.vehicle.position[1]
        features[33] = self.vehicle.lane_index
        if 0 <= features[33] - 1 < len(self.road.lanes):
            features[34] = 1
        if 0 <= features[33] + 1 < len(self.road.lanes):
            features[35] = 1
        #neighbour on the same lane
        v_front, v_rear = self.vehicle.road.neighbour_vehicles(self.vehicle)
        if v_front:
            num_neighbours += 1
            #v_front exist
            features[3] = 1
            features[4] = v_front.velocity
            features[5] = v_front.position[0]
            features[6] = v_front.position[1]
            features[7] = np.sqrt((features[5] - features[1]) ** 2 + (features[6] - features[2]) ** 2)
            distance += features[7]
        if v_rear:
            num_neighbours += 1
            #v_rear exist
            features[8] = 1
            features[9] = v_rear.velocity
            features[10] = v_rear.position[0]
            features[11] = v_rear.position[1]
            features[12] = np.sqrt((features[10] - features[1]) ** 2 + (features[11] - features[2]) ** 2)
            distance += features[12]

        #cars on left and right lanes contribute less on distance
        left_lane = self.vehicle.lane_index - 1
        right_lane = self.vehicle.lane_index + 1
        if 0 <= left_lane < len(self.road.lanes):
            l_front, l_rear = self.road.neighbour_vehicles(self.vehicle, self.vehicle.road.lanes[left_lane])
            if l_front:
                num_neighbours += 1
                # v_front exist
                features[13] = 1
                features[14] = l_front.velocity
                features[15] = l_front.position[0]
                features[16] = l_front.position[1]
                features[17] = np.sqrt((features[15] - features[1]) ** 2 + (features[16] - features[2]) ** 2)
                distance += self.diff_lane_discount * features[17]
            if l_rear:
                num_neighbours += 1
                # v_rear exist
                features[18] = 1
                features[19] = l_rear.velocity
                features[20] = l_rear.position[0]
                features[21] = l_rear.position[1]
                features[22] = np.sqrt((features[20] - features[1]) ** 2 + (features[21] - features[2]) ** 2)
                distance += self.diff_lane_discount * features[22]
        if 0 <= right_lane < len(self.road.lanes):
            r_front, r_rear = self.road.neighbour_vehicles(self.vehicle, self.vehicle.road.lanes[right_lane])
            if r_front:
                num_neighbours += 1
                # v_front exist
                features[23] = 1
                features[24] = r_front.velocity
                features[25] = r_front.position[0]
                features[26] = r_front.position[1]
                features[27] = np.sqrt((features[25] - features[1]) ** 2 + (features[26] - features[2]) ** 2)
                distance += self.diff_lane_discount * features[27]
            if r_rear:
                num_neighbours += 1
                # v_rear exist
                features[28] = 1
                features[29] = r_rear.velocity
                features[30] = r_rear.position[0]
                features[31] = r_rear.position[1]
                features[32] = np.sqrt((features[30] - features[1]) ** 2 + (features[31] - features[2]) ** 2)
                distance += self.diff_lane_discount * features[32]
        if num_neighbours == 0:
            self.distance = 1
        else:
            self.distance = distance / self.road_end# / num_neighbours

        return features
    """

    def _observation(self):
        from highway_env.extractors import Extractor
        extractor = Extractor()
        extractor_features = extractor.FeatureExtractor(self.road.vehicles, 0, 1)
        return extractor_features

    def _is_terminal(self):
        """
            The episode is over if the ego vehicle crashed or the time is out.
        """
        # fixed distance
        # return self.vehicle.crashed or self.vehicle.position[0] > self.road_end

        # fixed time
        return self.vehicle.crashed or self.vehicle.out or self.steps > self.config["duration"]
