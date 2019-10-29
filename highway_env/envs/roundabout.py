from __future__ import division, print_function, absolute_import
import numpy as np

from highway_env import utils
from highway_env.envs.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane, LanesConcatenation, CircularLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.control import ControlledVehicle, MDPVehicle, CarSim, FreeControl
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.dynamics import Obstacle
from highway_env.envs.graphics import EnvViewer
import random


def rad(deg):
    return deg * np.pi / 180


class RoundaboutEnv(AbstractEnv):
    COLLISION_REWARD = -1
    HIGH_VELOCITY_REWARD = 0.2
    RIGHT_LANE_REWARD = 0
    LANE_CHANGE_REWARD = -0.05

    DURATION = 11

    DEFAULT_CONFIG = {"other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                      "incoming_vehicle_destination": None,
                      "centering_position": [0.5, 0.6]}

    def __init__(self):
        super(RoundaboutEnv, self).__init__()
        self.config = self.DEFAULT_CONFIG.copy()
        self.steps = 0
        self.switch = False
        self.reset()
        EnvViewer.SCREEN_HEIGHT = 1200

    def configure(self, config):
        self.config.update(config)

    def _observation(self):
        return super(RoundaboutEnv, self)._observation()

    def _reward(self, action):
        reward = self.COLLISION_REWARD * self.vehicle.crashed \
                 + self.HIGH_VELOCITY_REWARD * self.vehicle.velocity_index / max(self.vehicle.SPEED_COUNT - 1, 1) \
                 + self.LANE_CHANGE_REWARD * (action in [0, 2])
        # return utils.remap(reward, [self.COLLISION_REWARD+self.LANE_CHANGE_REWARD, self.HIGH_VELOCITY_REWARD], [0, 1])
        return reward

    def _is_terminal(self):
        """
            The episode is over when a collision occurs or when the access ramp has been passed.
        """
        return self.vehicle.crashed or self.steps >= self.DURATION

    def reset(self):
        self._make_road()
        self._make_vehicles()
        self.steps = 0
        # print("len of vehicles:", len(self.vehicle), "\n")
        return self._observation()

    def step(self, action):
        self.steps += 1
        return super(RoundaboutEnv, self).step(action)

    def _make_road(self):
        # Circle lanes: (s)outh/(e)ast/(n)orth/(w)est (e)ntry/e(x)it.
        center = [0, 0]  # [m]
        radius = 30  # [m]
        alpha = 20  # [deg]
        net = RoadNetwork()
        radii = [radius, radius + 4]
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        line = [[c, s], [n, c]]
        # r = []
        # bounds = [30 * 50 * np.pi / 180, 30 * 40 * np.pi / 180, 34 * 50 * np.pi / 180, 34 * 40 * np.pi / 180]
        # bounds = [30 * 2 * np.pi, 34 * 2 * np.pi ]

        for lane in [0, 1]:
            net.add_lane("se", "ex",
                         CircularLane(center, radii[lane], rad(90 - alpha), rad(alpha), line_types=line[lane]))
            net.add_lane("ex", "ee", CircularLane(center, radii[lane], rad(alpha), rad(-alpha), line_types=line[lane]))
            net.add_lane("ee", "nx",
                         CircularLane(center, radii[lane], rad(-alpha), rad(-90 + alpha), line_types=line[lane]))
            net.add_lane("nx", "ne",
                         CircularLane(center, radii[lane], rad(-90 + alpha), rad(-90 - alpha), line_types=line[lane]))
            net.add_lane("ne", "wx",
                         CircularLane(center, radii[lane], rad(-90 - alpha), rad(-180 + alpha), line_types=line[lane]))
            net.add_lane("wx", "we",
                         CircularLane(center, radii[lane], rad(-180 + alpha), rad(-180 - alpha), line_types=line[lane]))
            net.add_lane("we", "sx",
                         CircularLane(center, radii[lane], rad(180 - alpha), rad(90 + alpha),
                                      line_types=line[lane]))
            net.add_lane("sx", "se",
                         CircularLane(center, radii[lane], rad(90 + alpha), rad(90 - alpha),
                                      line_types=line[lane]))
        """
        lane = 0
        net.add_lane("se", "ex",
                     CircularLane(center, radii[lane], rad(90 - alpha), rad(alpha), line_types=line[lane]))
        net.add_lane("ex", "ee", CircularLane(center, radii[lane], rad(alpha), rad(-alpha), line_types=line[lane]))
        net.add_lane("ee", "nx",
                     CircularLane(center, radii[lane], rad(-alpha), rad(-90 + alpha), line_types=line[lane]))
        net.add_lane("nx", "ne",
                     CircularLane(center, radii[lane], rad(-90 + alpha), rad(-90 - alpha), line_types=line[lane]))
        net.add_lane("ne", "wx",
                     CircularLane(center, radii[lane], rad(-90 - alpha), rad(-180 + alpha), line_types=line[lane]))
        net.add_lane("wx", "we",
                     CircularLane(center, radii[lane], rad(-180 + alpha), rad(-180 - alpha), line_types=line[lane]))
        net.add_lane("we", "sx",
                     CircularLane(center, radii[lane], rad(180 - alpha), rad(90 + alpha), line_types=line[lane]))
        net.add_lane("sx", "se",
                     CircularLane(center, radii[lane], rad(90 + alpha), rad(90 - alpha),
                                  line_types=line[lane]))
        lane = 1
        # net.add_lane("se", "ex",
        # CircularLane(center, radii[lane], rad(90 - alpha), rad(alpha), line_types=line[lane]))
        net.add_lane("ser", "exb",
                     CircularLane(center, radii[lane], rad(90 - alpha + 8), rad(alpha - 2), line_types=line[lane]))
        net.add_lane("exb", "exl",
                     CircularLane(center, radii[lane], rad(alpha - 2), rad(alpha - 7),
                                  line_types=line[lane]))
        net.add_lane("eel", "exr",
                     CircularLane(center, radii[lane], rad(-180 + alpha + 165), rad(-180 + alpha + 155),
                                  line_types=line[lane]))
        # net.add_lane("ex", "ee", CircularLane(center, radii[lane], rad(alpha), rad(-alpha), line_types=line[lane]))
        # net.add_lane("ee", "nx",
        # CircularLane(center, radii[lane], rad(-alpha), rad(-90 + alpha), line_types=line[lane]))
        net.add_lane("eer", "nxb",
                     CircularLane(center, radii[lane], rad(-180 + alpha + 147), rad(-180 + alpha + 65),
                                  line_types=line[lane]))
        net.add_lane("nxb", "nxl",
                     CircularLane(center, radii[lane], rad(-180 + alpha + 65), rad(-180 + alpha + 60),
                                  line_types=line[lane]))
        # net.add_lane("nx", "ne",
        # CircularLane(center, radii[lane], rad(-90 + alpha), rad(-90 - alpha), line_types=line[lane]))
        net.add_lane("nxr", "nel",
                     CircularLane(center, radii[lane], rad(-180 + alpha + 58), rad(-180 + alpha + 48),
                                  line_types=line[lane]))
        net.add_lane("ner", "wxb",
                     CircularLane(center, radii[lane], rad(-180 + alpha + 45), rad(-180 + alpha + 5),
                                  line_types=line[lane]))
        # net.add_lane("ner", "wxb",
        # CircularLane(center, radii[lane], rad(-90 - alpha), rad(-180 + alpha), line_types=line[lane]))
        net.add_lane("wxb", "wxl",
                     CircularLane(center, radii[lane], rad(-180 + alpha + 5), rad(-180 + alpha - 7),
                                  line_types=line[lane]))
        # net.add_lane("wx", "we",
        # CircularLane(center, radii[lane], rad(-180 + alpha), rad(-180 - alpha), line_types=line[lane]))
        net.add_lane("wxr", "wel",
                     CircularLane(center, radii[lane], rad(-180 + alpha - 15), rad(-180 - alpha + 15),
                                  line_types=line[lane]))
        # net.add_lane("we", "sx",
        # CircularLane(center, radii[lane], rad(180 - alpha), rad(90 + alpha - 5), line_types=line[lane]))
        net.add_lane("wer", "sxb",
                     CircularLane(center, radii[lane], rad(180 - alpha + 7), rad(90 + alpha + 5),
                                  line_types=line[lane]))
        net.add_lane("sxb", "sxl",
                     CircularLane(center, radii[lane], rad(95 + alpha), rad(90 + alpha - 7), line_types=line[lane]))
        net.add_lane("sxr", "sel",
                     CircularLane(center, radii[lane], rad(90 + alpha - 15), rad(90 - alpha + 15),
                                  line_types=line[lane]))
        """
        # Access lanes: (r)oad/(s)ine
        access = 200  # [m]
        dev = 120  # [m]
        a = 5  # [m]
        delta_st = 0.20 * dev  # [m]

        delta_en = dev - delta_st
        w = 2 * np.pi / dev

        net.add_lane("ser", "ses", StraightLane([2, access], [2, dev / 2], line_types=[s, c]))
        net.add_lane("ses", "se",
                     SineLane([2 + a, dev / 2], [2 + a, dev / 2 - delta_st], a, w, -np.pi / 2, line_types=[c, c]))
        net.add_lane("sx", "sxs",
                     SineLane([-2 - a, -dev / 2 + delta_en], [-2 - a, dev / 2], a, w, -np.pi / 2 + w * delta_en,
                              line_types=[c, c]))
        net.add_lane("sxs", "sxr", StraightLane([-2, dev / 2], [-2, access], line_types=[n, c]))

        net.add_lane("eer", "ees", StraightLane([access, -2], [dev / 2, -2], line_types=[s, c]))
        net.add_lane("ees", "ee",
                     SineLane([dev / 2, -2 - a], [dev / 2 - delta_st, -2 - a], a, w, -np.pi / 2, line_types=[c, c]))
        net.add_lane("ex", "exs",
                     SineLane([-dev / 2 + delta_en, 2 + a], [dev / 2, 2 + a], a, w, -np.pi / 2 + w * delta_en,
                              line_types=[c, c]))
        net.add_lane("exs", "exr", StraightLane([dev / 2, 2], [access, 2], line_types=[n, c]))

        net.add_lane("ner", "nes", StraightLane([-2, -access], [-2, -dev / 2], line_types=[s, c]))
        net.add_lane("nes", "ne",
                     SineLane([-2 - a, -dev / 2], [-2 - a, -dev / 2 + delta_st], a, w, -np.pi / 2, line_types=[c, c]))
        net.add_lane("nx", "nxs",
                     SineLane([2 + a, dev / 2 - delta_en], [2 + a, -dev / 2], a, w, -np.pi / 2 + w * delta_en,
                              line_types=[c, c]))
        net.add_lane("nxs", "nxr", StraightLane([2, -dev / 2], [2, -access], line_types=[n, c]))

        net.add_lane("wer", "wes", StraightLane([-access, 2], [-dev / 2, 2], line_types=[s, c]))
        net.add_lane("wes", "we",
                     SineLane([-dev / 2, 2 + a], [-dev / 2 + delta_st, 2 + a], a, w, -np.pi / 2, line_types=[c, c]))
        net.add_lane("wx", "wxs",
                     SineLane([dev / 2 - delta_en, -2 - a], [-dev / 2, -2 - a], a, w, -np.pi / 2 + w * delta_en,
                              line_types=[c, c]))
        net.add_lane("wxs", "wxr", StraightLane([-dev / 2, -2], [-access, -2], line_types=[n, c]))

        road = Road(network=net, np_random=self.np_random)
        self.road = road

        # print("lanes:\n", self.road.lanes)

    def _make_vehicles(self):
        """
            Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        """
        """
        road = self.road
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        car_number_each_lane = 5
        # reset_position_range = (30, 40)
        reset_position_range = (100, 0)
        # reset_lane = random.choice(road.lanes)

        for l in road.lanes[:3]:
            reset_lane = road.lanes[2]
            cars_on_lane = car_number_each_lane
            reset_position = (32, 0)
            if l is reset_lane:
                cars_on_lane += 1
                # reset_position = random.choice(range(5, 6))

        # github-version
        position_deviation = 2
        velocity_deviation = 2

        # Ego-vehicle
        # ego_lane = self.road.network.get_lane(("ser", "ses", 0))
        ego_lane = self.road.get_lane((100, 0))
        print("\nego_lane.position:", ego_lane.position(140, 0), "\n")
        # ego_vehicle = MDPVehicle(self.road,
        # ego_lane.position(140, 0),
        # velocity=5,
        # heading=ego_lane.heading_at(140)).plan_route_to("nxs")
        ego_vehicle = IDMVehicle(road, (60, 0), np.pi / 2, velocity=10, max_length=500)
        # MDPVehicle.SPEED_MIN = 0
        # MDPVehicle.SPEED_MAX = 15
        # MDPVehicle.SPEED_COUNT = 4

        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        # Incoming vehicle
        # destinations = ["exr", "sxr", "nxr"]
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = other_vehicles_type.make_on_lane(self.road,
                                                   (0, 30),
                                                   longitudinal=5 + self.np_random.randn() * position_deviation,
                                                   velocity=16 + self.np_random.randn() * velocity_deviation)
        # if self.config["incoming_vehicle_destination"] is not None:
        # destination = destinations[self.config["incoming_vehicle_destination"]]
        # else:
        # destination = self.np_random.choice(destinations)

        # vehicle.plan_route_to(destination)
        # vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)

        # Other vehicles
        for i in list(range(1, 2)) + list(range(-1, 0)):
            vehicle = other_vehicles_type.make_on_lane(self.road,
                                                       (30 * i, 0),
                                                       longitudinal=20 * i + self.np_random.randn() * position_deviation,
                                                       velocity=16 + self.np_random.randn() * velocity_deviation)
            # vehicle.plan_route_to(self.np_random.choice(destinations))
            # vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

        # Entering vehicle
        # vehicle = other_vehicles_type.make_on_lane(self.road,
                                                   # ("eer", "ees", 0),
                                                   # longitudinal=50 + self.np_random.randn() * position_deviation,
                                                   # velocity=16 + self.np_random.randn() * velocity_deviation)
        # vehicle.plan_route_to(self.np_random.choice(destinations))
        # vehicle.randomize_behavior()
        # self.road.vehicles.append(vehicle)
    
        vehicle = MDPVehicle.create_random(self.road, 25, spacing=3, np_random=self.np_random)
        self.road.vehicles.append(vehicle)
        for v in self.road.vehicles:
            lane_index = v.lane_index
            self.road.lanes[lane_index].vehicles.append(v)
        """
        position_deviation = 5
        velocity_deviation = 1.5
        other_vehicles_mandatory = False
        max_l = 700
        road = self.road
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        car_number_each_lane = 1
        # reset_position_range = (30, 40)
        # reset_lane = random.choice(road.lanes)
        reset_lane = ("ser", "ses", 0)
        ego_vehicle = None
        num = 0

        print("lanes:", self.road.network.LANES, "\n")
        # self.road.network.LANES_NUMBER = 28
        for l in self.road.network.LANES:
            # print("l:",l,"\n")
            lane = road.network.get_lane(l)
            cars_on_lane = car_number_each_lane
            reset_position = None
            if l == reset_lane:
                cars_on_lane += 1
                reset_position = random.choice(range(0, 3))
                # reset_position = 2
            for j in range(cars_on_lane):
                if not ego_vehicle:
                    ego_lane = self.road.network.get_lane(("ser", "ses", 0))
                    if self.switch:
                        ego_vehicle = MDPVehicle(self.road,
                                                 ego_lane.position(110, 0),
                                                 velocity=15,
                                                 heading=ego_lane.heading_at(110)).plan_route_to("nxs")
                    else:
                        ego_vehicle = IDMVehicle(self.road,
                                                 ego_lane.position(110, 0),
                                                 velocity=15,
                                                 heading=ego_lane.heading_at(110)).plan_route_to("wxs")
                        # ego_vehicle.destination = 1
                        ego_vehicle.id = 0
                    self.road.vehicles.append(ego_vehicle)
                    self.vehicle = ego_vehicle
                    lane.vehicles.append(ego_vehicle)
                else:
                    destinations = ["wxr", "sxr", "nxr", "exr"]
                    birth_place = [("ee", "nx", 0), ("ne", "wx", 0),
                                   ("we", "sx", 0), ("se", "ex", 0), ("ee", "nx", 1), ("ne", "wx", 1),
                                   ("we", "sx", 1), ("se", "ex", 1),
                                   ("wer", "wes", 0), ("eer", "ees", 0), ("ner", "nes", 0),
                                   ("ser", "ses", 0),
                                   ("wxs", "wxr", 0), ("exs", "exr", 0), ("nxs", "nxr", 0),
                                   ("sxs", "sxr", 0)]
                    car = other_vehicles_type.make_on_lane(self.road, birth_place[np.random.randint(0, 16)],
                                                           longitudinal=70 + np.random.randint(1,
                                                                                               5) * position_deviation,
                                                           velocity=5 + np.random.randint(1, 10) * velocity_deviation)
                    if self.config["incoming_vehicle_destination"] is not None:
                        destination = destinations[self.config["incoming_vehicle_destination"]]
                    else:
                        destination = destinations[np.random.randint(0, 4)]
                        # destination = self.np_random.choice(destinations)
                    """
                    if  0<= num <=6:
                        destinations = ["wxr", "sxr", "nxr"]
                        car = other_vehicles_type.make_on_lane(self.road, ("ee", "nx", 1),
                                                               longitudinal=5 + self.np_random.randn() * position_deviation,
                                                               velocity=5 + self.np_random.randn() * velocity_deviation)

                        if self.config["incoming_vehicle_destination"] is not None:
                            destination = destinations[self.config["incoming_vehicle_destination"]]
                        else:
                            destination = self.np_random.choice(destinations)
                        # if other_vehicles_mandatory:
                        # car.destination = 1
                        # car.plan_route_to(destination)
                        # car.randomize_behavior()
                        # self.road.vehicles.append(car)
                        # road.vehicles.append(car)
                        # lane.vehicles.append(car)
                    elif 7<=num<=13:
                        destinations = ["exr", "sxr", "wxr"]
                        car = other_vehicles_type.make_on_lane(self.road, ("ne", "wx", 1),
                                                               longitudinal=5 + self.np_random.randn() * position_deviation,
                                                               velocity=5 + self.np_random.randn() * velocity_deviation)

                        if self.config["incoming_vehicle_destination"] is not None:
                            destination = destinations[self.config["incoming_vehicle_destination"]]
                        else:
                            destination = self.np_random.choice(destinations)
                        # if other_vehicles_mandatory:
                        # car.destination = 1
                        # car.plan_route_to(destination)
                        # car.randomize_behavior()
                        # self.road.vehicles.append(car)
                        # road.vehicles.append(car)
                        # lane.vehicles.append(car)

                    elif 14<=num<=20:
                        destinations = ["exr", "sxr", "nxr"]
                        car = other_vehicles_type.make_on_lane(self.road, ("we", "sx", 1),
                                                               longitudinal=5 + self.np_random.randn() * position_deviation,
                                                               velocity=5 + self.np_random.randn() * velocity_deviation)

                        if self.config["incoming_vehicle_destination"] is not None:
                            destination = destinations[self.config["incoming_vehicle_destination"]]
                        else:
                            destination = self.np_random.choice(destinations)
                        # if other_vehicles_mandatory:
                        # car.destination = 1
                        # car.plan_route_to(destination)
                        # car.randomize_behavior()
                        # self.road.vehicles.append(car)
                        # road.vehicles.append(car)
                        # lane.vehicles.append(car)
                    else :
                        destinations = ["exr", "wxr", "nxr"]
                        car = other_vehicles_type.make_on_lane(self.road, ("se", "ex", 1),
                                                               longitudinal=5 + self.np_random.randn() * position_deviation,
                                                               velocity=5 + self.np_random.randn() * velocity_deviation)

                        if self.config["incoming_vehicle_destination"] is not None:
                            destination = destinations[self.config["incoming_vehicle_destination"]]
                        else:
                            destination = self.np_random.choice(destinations)
                        # if other_vehicles_mandatory:
                        # car.destination = 1
                        # car.plan_route_to(destination)
                        # car.randomize_behavior()
                        # self.road.vehicles.append(car)
                        # road.vehicles.append(car)
                        # lane.vehicles.append(car)
                    """
                    car.plan_route_to(destination)
                    car.randomize_behavior()
                    self.road.vehicles.append(car)
                    # road.vehicles.append(car)
                    lane.vehicles.append(car)
                    # if other_vehicles_mandatory:
                    # car.destination = 1
                    # road.vehicles.append(car)
                    # l.vehicles.append(car)
            num += 1
        # print("make_vehicle finish")

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

        for i in range(1):
            destinations = ["wxr", "sxr", "nxr", "exr"]
            birth_place = [
                ("wer", "wes", 0), ("eer", "ees", 0), ("ner", "nes", 0),
                ("ser", "ses", 0)]
            position_deviation = 5
            velocity_deviation = 1.5
            other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
            birth = birth_place[np.random.randint(0, 4)]
            lane = self.road.network.get_lane(birth)
            car = other_vehicles_type.make_on_lane(self.road, birth,
                                                   longitudinal=70 + np.random.randint(1,
                                                                                       5) * position_deviation,
                                                   velocity=5 + np.random.randint(1, 10) * velocity_deviation)
            if self.config["incoming_vehicle_destination"] is not None:
                destination = destinations[self.config["incoming_vehicle_destination"]]
            else:
                destination = destinations[np.random.randint(0, 4)]
            car.plan_route_to(destination)
            car.randomize_behavior()
            self.road.vehicles.append(car)
            lane.vehicles.append(car)

        # obs = self._observation()
        # reward = self._reward(action)
        terminal = self._is_terminal()
        info = {}
        return terminal, extractor_features
