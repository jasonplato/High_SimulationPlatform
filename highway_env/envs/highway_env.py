from __future__ import division, print_function, absolute_import
import numpy as np
from gym import logger

from highway_env import utils
from highway_env.envs.abstract import AbstractEnv
from highway_env.road.road import Road,RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.control import MDPVehicle,ControlledVehicle,CarSim
from highway_env.utils import DIFFICULTY_LEVELS
from highway_env.vehicle.dynamics import Vehicle
from highway_env.extractors import Extractor

ext = Extractor()
class HighwayEnv(AbstractEnv):
    """
        A highway driving environment.

        The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high velocity,
        staying on the rightmost lanes and avoiding collisions.
    """

    COLLISION_REWARD = -1
    """ The reward received when colliding with a vehicle."""
    RIGHT_LANE_REWARD = 0.1
    """ The reward received when driving on the right-most lanes, linearly mapped to zero for other lanes."""
    HIGH_VELOCITY_REWARD = 0.2
    """ The reward received when driving at full speed, linearly mapped to zero for lower speeds."""
    LANE_CHANGE_REWARD = -0
    """ The reward received at each lane change action."""


    def __init__(self):
        super(HighwayEnv, self).__init__()
        self.DIFFICULTY_LEVELS = DIFFICULTY_LEVELS
        self.config = self.DIFFICULTY_LEVELS["ORIGIN"].copy()
        self.steps = 0
        self.reset()
        Vehicle.ID = 0

    def reset(self):
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

    def step(self, action):
        self.steps += 1
        return super(HighwayEnv, self).step(action)

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
        """
            Create a road composed of straight adjacent lanes.
        """
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"]),
                         np_random=self.np_random)

    def _create_vehicles(self):
        """
            Create some new random vehicles of a given type, and add them on the road.
        """
        self.vehicle = IDMVehicle.create_random(self.road, 25, spacing=self.config["initial_spacing"])
        self.road.vehicles.append(self.vehicle)

        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        for _ in range(self.config["vehicles_count"]):
            self.road.vehicles.append(vehicles_type.create_random(self.road))
    """
    def _create_road(self):
        road = Road.create_random_road(lanes_count=self.config["lanes_count"],
                                       vehicles_count=self.config["vehicles_count"],
                                       vehicles_type=utils.class_from_path(self.config["other_vehicles_type"]),
                                       np_random=self.np_random)
        vehicle = MDPVehicle.create_random(road, 25, spacing=self.config["initial_spacing"], np_random=self.np_random)
        #vehicle = ControlledVehicle.create_random(road, 25, spacing=self.config["initial_spacing"], np_random=self.np_random)
        #vehicle = CarSim.create_random(road, 10, spacing=self.config["initial_spacing"], np_random=self.np_random)
        #vehicle.position[0] =100
        road.vehicles.append(vehicle)
        #road.vehicles[len(road.vehicles) // 2].velocity=50
        # return road, road.vehicles[len(road.vehicles) // 2]
        return road, vehicle
    """

    def _reward(self, action):
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        action_reward = {0: self.LANE_CHANGE_REWARD, 1: 0, 2: self.LANE_CHANGE_REWARD, 3: 0, 4: 0}
        state_reward = \
            + self.COLLISION_REWARD * self.vehicle.crashed \
            + self.RIGHT_LANE_REWARD * self.vehicle.target_lane_index / (len(self.road.network.LANES) - 1) \
            + self.HIGH_VELOCITY_REWARD * self.vehicle.velocity_index / (self.vehicle.SPEED_COUNT - 1)
        return action_reward[action] + state_reward


    def _observation(self):
        # return ext.FeatureExtractor(self.road.vehicles,self.vehicle.id)
        return super(HighwayEnv, self)._observation()

    def _is_terminal(self):
        """
            The episode is over if the ego vehicle crashed or the time is out.
        """
        return self.vehicle.crashed or self.steps > self.config["duration"]

        # return self.vehicle.crashed  or self.vehicle.out or self.steps > self.config["duration"]

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
        for i in range(4):
            birth_place = [(0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 1, 3)]
            # destinations = ["e1", "e2"]
            position_deviation = 5
            velocity_deviation = 1.5
            other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
            birth = birth_place[np.random.randint(0, 4)]
            lane = self.road.network.get_lane(birth)
            car = other_vehicles_type.make_on_lane(self.road, birth,
                                                   longitudinal=100 + np.random.randint(1,
                                                                                       10) * position_deviation,
                                                   velocity=5 + np.random.randint(1, 10) * velocity_deviation)

            destination = 1
            car.plan_route_to(destination)
            car.randomize_behavior()
            self.road.vehicles.append(car)
            lane.vehicles.append(car)

        #obs = self._observation()
        #reward = self._reward(action)
        terminal = self._is_terminal()
        info = {}
        return terminal,extractor_features
