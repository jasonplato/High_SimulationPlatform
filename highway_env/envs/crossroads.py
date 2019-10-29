from __future__ import division, print_function, absolute_import
import numpy as np

from highway_env import utils
from highway_env.envs.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane, CircularLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.control import ControlledVehicle, MDPVehicle, CarSim, FreeControl
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.dynamics import RedLight
import pygame
import random
import math
from highway_env.envs.graphics import EnvViewer


def rad(deg):
    return deg * np.pi / 180


class CrossroadEnv(AbstractEnv):
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
        super(CrossroadEnv, self).__init__()
        self.config = self.DEFAULT_CONFIG.copy()
        self.steps = 0
        self.traffic_lights = {}
        self.have_traffic_lights = False
        self.entrance = ["swwe", "swse", "wmwe", "wmee", "nwwe", "nwne", "nene", "neee", "emwe", "emee", "sese", "seee"]
        self.end = ["swwx", "swsx", "wmwx", "nwwx", "nwnx", "nenx",
                    "neex",
                    "emex",
                    "seex", "sesx"]
        EnvViewer.SCREEN_HEIGHT = 700
        EnvViewer.SCREEN_WIDTH = 700
        self.SIMULATION_FREQUENCY = 50

    def configure(self, config):
        self.config.update(config)

    def _observation(self):
        return super(CrossroadEnv, self)._observation()

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
        return self.vehicle.crashed

    def reset(self):
        self.make_roads()
        self.make_vehicles()
        return self._observation()

    def make_roads(self):
        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        """
        crossroad of southwest
        sw:southwest
        w:west n:north e:east s:south
        e:entrance x:exit
        r:right of the ego_car  l:left of the ego_car
                
                
                                   swne swnx
                                    inter4
                            swner swnel swnxl swnxr
        swwe         swwxr                          sweer          swee
        swwx  inter1 swnxl                          sweel  inter3  swex
                     swnel                          swexl
                     swner                          swexr
                            swsxr swsxl swsel swser
                                    inter2
                                   swsx swse
        """

        net.add_lane("swwe", "intersw1", StraightLane(np.array([0, 0]), np.array([100, 0]), line_types=[c, s]))
        net.add_lane("intersw1", "swwel",
                     StraightLane(np.array([100, 0]), np.array([150, 0]), line_types=[c, c], forbidden=True))

        net.add_lane("swwe", "intersw1", StraightLane(np.array([0, 4]), np.array([100, 4]), line_types=[s, c]))
        net.add_lane("intersw1", "swwer",
                     StraightLane(np.array([100, 4]), np.array([150, 4]), line_types=[c, c], forbidden=True))

        net.add_lane("inter_sw_1", "swwx", StraightLane(np.array([100, -8]), np.array([0, -8]), line_types=[s, c]))
        net.add_lane("swwxr", "inter_sw_1", StraightLane(np.array([150, -8]), np.array([100, -8]), line_types=[s, c]))

        net.add_lane("inter_sw_1", "swwx", StraightLane(np.array([100, -4]), np.array([0, -4]), line_types=[c, s]))
        net.add_lane("swwxl", "inter_sw_1", StraightLane(np.array([150, -4]), np.array([100, -4]), line_types=[c, s]))

        net.add_lane("swse", "intersw2", StraightLane(np.array([167, 158]), np.array([167, 58]), line_types=[c, s]))
        net.add_lane("intersw2", "swsel",
                     StraightLane(np.array([167, 58]), np.array([167, 8]), line_types=[c, c], forbidden=True))

        net.add_lane("swse", "intersw2", StraightLane(np.array([171, 158]), np.array([171, 58]), line_types=[s, c]))
        net.add_lane("intersw2", "swser",
                     StraightLane(np.array([171, 58]), np.array([171, 8]), line_types=[c, c], forbidden=True))

        net.add_lane("inter_sw_2", "swsx", StraightLane(np.array([159, 58]), np.array([159, 158]), line_types=[s, c]))
        net.add_lane("swsxr", "inter_sw_2", StraightLane(np.array([159, 8]), np.array([159, 58]), line_types=[s, c]))

        net.add_lane("inter_sw_2", "swsx", StraightLane(np.array([163, 58]), np.array([163, 158]), line_types=[c, s]))
        net.add_lane("swsxl", "inter_sw_2", StraightLane(np.array([163, 8]), np.array([163, 58]), line_types=[c, s]))

        # net.add_lane("swee", "intersw3", StraightLane(np.array([328, -8]), np.array([228, -8]), line_types=[s, c]))
        net.add_lane("intersw3", "sweer",
                     StraightLane(np.array([228, -8]), np.array([178, -8]), line_types=[c, c], forbidden=True))

        # net.add_lane("swee", "intersw3", StraightLane(np.array([328, -4]), np.array([228, -4]), line_types=[c, s]))
        net.add_lane("intersw3", "sweel",
                     StraightLane(np.array([228, -4]), np.array([178, -4]), line_types=[c, c], forbidden=True))

        # net.add_lane("intersw_3", "swex", StraightLane(np.array([228, 0]), np.array([328, 0]), line_types=[c, s]))
        net.add_lane("swexl", "inter_sw_3", StraightLane(np.array([178, 0]), np.array([228, 0]), line_types=[c, s]))

        # net.add_lane("intersw_3", "swex", StraightLane(np.array([228, 4]), np.array([328, 4]), line_types=[s, c]))
        net.add_lane("swexr", "inter_sw_3", StraightLane(np.array([178, 4]), np.array([228, 4]), line_types=[s, c]))

        net.add_lane("intersw4", "swner",
                     StraightLane(np.array([159, -62]), np.array([159, -12]), line_types=[c, c], forbidden=True))

        net.add_lane("intersw4", "swnel",
                     StraightLane(np.array([163, -62]), np.array([163, -12]), line_types=[c, c], forbidden=True))

        net.add_lane("swnxl", "inter_sw_4", StraightLane(np.array([167, -12]), np.array([167, -62]), line_types=[c, s]))

        net.add_lane("swnxr", "inter_sw_4", StraightLane(np.array([171, -12]), np.array([171, -62]), line_types=[s, c]))

        # bellow: fulfill the turning lanes for vehicles to turn
        # center = [152, 10]
        # radii = [6, 10]
        # alpha = math.degrees(math.asin(math.sqrt(97) / radii[0] / 2))
        net.add_lane("swwer", "swsxr",
                     StraightLane(np.array([150, 4]), np.array([159, 8]), line_types=[n, n], forbidden=True))
        # center = [152, -13]
        net.add_lane("swner", "swwxr",
                     StraightLane(np.array([159, -12]), np.array([150, -8]), line_types=[n, n], forbidden=True))
        net.add_lane("swnel", "swsxl",
                     StraightLane(np.array([163, -12]), np.array([163, 8]), line_types=[n, n], forbidden=True))
        net.add_lane("swnel", "swexl",
                     StraightLane(np.array([163, -12]), np.array([178, 0]), line_types=[n, n], forbidden=True))

        # center = [178, -13]
        net.add_lane("sweer", "swnxr",
                     StraightLane(np.array([178, -8]), np.array([171, -12]), line_types=[n, n], forbidden=True))
        net.add_lane("sweel", "swsxl",
                     StraightLane(np.array([178, -4]), np.array([163, 8]), line_types=[n, n], forbidden=True))
        net.add_lane("sweel", "swwxl",
                     StraightLane(np.array([178, -4]), np.array([150, -4]), line_types=[n, n], forbidden=True))

        # center = [178, 10]
        net.add_lane("swser", "swexr",
                     StraightLane(np.array([171, 8]), np.array([178, 4]), line_types=[n, n], forbidden=True))
        net.add_lane("swsel", "swwxl",
                     StraightLane(np.array([167, 8]), np.array([150, -4]), line_types=[n, n], forbidden=True))
        net.add_lane("swsel", "swnxl",
                     StraightLane(np.array([167, 8]), np.array([167, -12]), line_types=[n, n], forbidden=True))
        net.add_lane("swwel", "swnxl",
                     StraightLane(np.array([150, 0]), np.array([167, -12]), line_types=[n, n], forbidden=True))
        net.add_lane("swwel", "swexl",
                     StraightLane(np.array([150, 0]), np.array([178, 0]), line_types=[n, n], forbidden=True))

        """
        straight road of west
        m:middle
        """
        net.add_lane("inter_sw_4", "interwm2",
                     StraightLane(np.array([167, -62]), np.array([167, -162]), line_types=[c, c]))
        # net.add_lane(" ", " ",
        #              StraightLane(np.array([66, -37]), np.array([66, -42]), line_types=[n, c]))
        # net.add_lane(" ", " ",
        #              StraightLane(np.array([66, -42]), np.array([62, -49]), line_types=[n, c]))
        # net.add_lane(" ", " ",
        #              StraightLane(np.array([62, -62]), np.array([66, -67]), line_types=[n, c]))
        # net.add_lane("intersw_mergein", "interwm_mergeout",
        #              StraightLane(np.array([62, -47]), np.array([62, -55]), line_types=[c, c]))
        net.add_lane("inter_wm_2", "intersw4",
                     StraightLane(np.array([163, -162]), np.array([163, -62]), line_types=[c, c]))

        # net.add_lane("interwm_mergeout", "interwm2l",
        #              StraightLane(np.array([62, -55]), np.array([62, -62]), line_types=[c, s]))
        # net.add_lane("interwm_mergeout", "interwm2r",
        #              StraightLane(np.array([62, -55]), np.array([66, -62]), line_types=[n, c]))
        net.add_lane("interwm2", "wmsel",
                     StraightLane(np.array([167, -162]), np.array([167, -212]), line_types=[c, c], forbidden=True))
        net.add_lane("interwm2", "wmser",
                     StraightLane(np.array([171, -162]), np.array([171, -212]), line_types=[c, c], forbidden=True))

        net.add_lane("wmsxr", "inter_wm_2",
                     StraightLane(np.array([159, -212]), np.array([159, -162]), line_types=[s, c]))
        net.add_lane("wmsxl", "inter_wm_2",
                     StraightLane(np.array([163, -212]), np.array([163, -162]), line_types=[c, s]))

        net.add_lane("wmwe", "interwm1", StraightLane(np.array([50, -216]), np.array([100, -216]), line_types=[s, c]))
        net.add_lane("wmwe", "interwm1", StraightLane(np.array([50, -220]), np.array([100, -220]), line_types=[c, s]))

        net.add_lane("interwm1", "wmwel",
                     StraightLane(np.array([100, -216]), np.array([150, -216]), line_types=[c, c], forbidden=True))
        net.add_lane("interwm1", "wmwer",
                     StraightLane(np.array([100, -220]), np.array([150, -220]), line_types=[c, c], forbidden=True))

        net.add_lane("inter_wm_1", "wmwx", StraightLane(np.array([100, -224]), np.array([50, -224]), line_types=[c, s]))
        net.add_lane("inter_wm_1", "wmwx", StraightLane(np.array([100, -228]), np.array([50, -228]), line_types=[s, c]))

        net.add_lane("wmwxl", "inter_wm_1",
                     StraightLane(np.array([150, -224]), np.array([100, -224]), line_types=[c, s]))
        net.add_lane("wmwxr", "inter_wm_1",
                     StraightLane(np.array([150, -228]), np.array([100, -228]), line_types=[s, c]))

        net.add_lane("wmee", "interwm3",
                     StraightLane(np.array([271, -228]), np.array([221, -228]), line_types=[c, s]))
        net.add_lane("wmee", "interwm3",
                     StraightLane(np.array([271, -224]), np.array([221, -224]), line_types=[s, c]))
        net.add_lane("interwm3", "wmeer",
                     StraightLane(np.array([221, -228]), np.array([179, -228]), line_types=[c, c], forbidden=True))
        net.add_lane("interwm3", "wmeel",
                     StraightLane(np.array([221, -224]), np.array([179, -224]), line_types=[c, c], forbidden=True))

        net.add_lane("wmexr", "inter_wm_3",
                     StraightLane(np.array([179, -216]), np.array([221, -216]), line_types=[s, c]))
        net.add_lane("wmexl", "inter_wm_3",
                     StraightLane(np.array([179, -220]), np.array([221, -220]), line_types=[c, s]))

        net.add_lane("inter_wm_3", "wmex",
                     StraightLane(np.array([221, -216]), np.array([271, -216]), line_types=[s, c]))
        net.add_lane("inter_wm_3", "wmex",
                     StraightLane(np.array([221, -220]), np.array([271, -220]), line_types=[c, s]))

        net.add_lane("interwm4", "wmner",
                     StraightLane(np.array([159, -274]), np.array([159, -232]), line_types=[c, c], forbidden=True))
        net.add_lane("interwm4", "wmnel",
                     StraightLane(np.array([163, -274]), np.array([163, -232]), line_types=[c, c], forbidden=True))

        net.add_lane("wmnxr", "inter_wm_4",
                     StraightLane(np.array([171, -232]), np.array([171, -274]), line_types=[s, c]))
        net.add_lane("wmnxl", "inter_wm_4",
                     StraightLane(np.array([167, -232]), np.array([167, -274]), line_types=[c, s]))

        # bellow: fulfill the turning lanes for vehicles to turn
        # center = [152, -210]
        # radii = [6, 10]
        # alpha = math.degrees(math.asin(math.sqrt(97) / radii[0] / 2))
        net.add_lane("wmwer", "wmsxr",
                     StraightLane(np.array([150, -220]), np.array([159, -212]), line_types=[n, n], forbidden=True))
        net.add_lane("wmwel", "wmnxl",
                     StraightLane(np.array([150, -216]), np.array([167, -232]), line_types=[n, n], forbidden=True))
        net.add_lane("wmwel", "wmexl",
                     StraightLane(np.array([150, -216]), np.array([179, -220]), line_types=[n, n], forbidden=True))

        # center = [152, -233]
        net.add_lane("wmner", "wmwxr",
                     StraightLane(np.array([159, -232]), np.array([150, -228]), line_types=[n, n], forbidden=True))
        net.add_lane("wmnel", "wmexl",
                     StraightLane(np.array([163, -232]), np.array([179, -220]), line_types=[n, n], forbidden=True))
        net.add_lane("wmnel", "wmsxl",
                     StraightLane(np.array([163, -232]), np.array([163, -212]), line_types=[n, n], forbidden=True))

        # center = [178, -233]
        net.add_lane("wmeer", "wmnxr",
                     StraightLane(np.array([179, -228]), np.array([171, -232]), line_types=[n, n], forbidden=True))
        net.add_lane("wmeel", "wmsxl",
                     StraightLane(np.array([179, -224]), np.array([163, -212]), line_types=[n, n], forbidden=True))
        net.add_lane("wmeel", "wmwxl",
                     StraightLane(np.array([179, -224]), np.array([150, -224]), line_types=[n, n], forbidden=True))

        # center = [178, -210]
        net.add_lane("wmser", "wmexr",
                     StraightLane(np.array([171, -212]), np.array([179, -216]), line_types=[n, n], forbidden=True))
        net.add_lane("wmsel", "wmnxl",
                     StraightLane(np.array([167, -212]), np.array([167, -232]), line_types=[n, n], forbidden=True))
        net.add_lane("wmsel", "wmwxl",
                     StraightLane(np.array([167, -212]), np.array([150, -224]), line_types=[n, n], forbidden=True))

        """
        straight road of west
        m:middle
        """
        net.add_lane("inter_wm_4", "internw2",
                     StraightLane(np.array([167, -274]), np.array([167, -374]), line_types=[c, c]))
        net.add_lane("inter_nw_2", "interwm4",
                     StraightLane(np.array([163, -374]), np.array([163, -274]), line_types=[c, c]))

        # net.add_lane(" ", " ",
        #              StraightLane(np.array([66, -136]), np.array([62, -146]), line_types=[n, c]))
        # net.add_lane(" ", " ",
        #              StraightLane(np.array([62, -161]), np.array([66, -165]), line_types=[n, c]))

        net.add_lane("internw2", "nwsel",
                     StraightLane(np.array([167, -374]), np.array([167, -424]), line_types=[c, c], forbidden=True))
        net.add_lane("internw2", "nwser",
                     StraightLane(np.array([171, -374]), np.array([171, -424]), line_types=[c, c], forbidden=True))

        net.add_lane("nwsxr", "inter_nw_2",
                     StraightLane(np.array([159, -424]), np.array([159, -374]), line_types=[s, c]))
        net.add_lane("nwsxl", "inter_nw_2",
                     StraightLane(np.array([163, -424]), np.array([163, -374]), line_types=[c, s]))

        net.add_lane("nwwe", "internw1", StraightLane(np.array([50, -428]), np.array([100, -428]), line_types=[s, c]))
        net.add_lane("nwwe", "internw1", StraightLane(np.array([50, -432]), np.array([100, -432]), line_types=[c, s]))

        net.add_lane("internw1", "nwwel",
                     StraightLane(np.array([100, -428]), np.array([150, -428]), line_types=[c, c], forbidden=True))
        net.add_lane("internw1", "nwwer",
                     StraightLane(np.array([100, -432]), np.array([150, -432]), line_types=[c, c], forbidden=True))

        net.add_lane("inter_nw_1", "nwwx", StraightLane(np.array([100, -436]), np.array([50, -436]), line_types=[c, s]))
        net.add_lane("inter_nw_1", "nwwx", StraightLane(np.array([100, -440]), np.array([50, -440]), line_types=[s, c]))

        net.add_lane("nwwxl", "inter_nw_1",
                     StraightLane(np.array([150, -436]), np.array([100, -436]), line_types=[c, s]))
        net.add_lane("nwwxr", "inter_nw_1",
                     StraightLane(np.array([150, -440]), np.array([100, -440]), line_types=[s, c]))

        net.add_lane("internw3", "nweer",
                     StraightLane(np.array([221, -448]), np.array([179, -448]), line_types=[c, c], forbidden=True))
        net.add_lane("internw3", "nweel",
                     StraightLane(np.array([221, -444]), np.array([179, -444]), line_types=[c, c], forbidden=True))

        net.add_lane("nwexr", "inter_nw_3",
                     StraightLane(np.array([179, -428]), np.array([221, -428]), line_types=[s, c]))
        net.add_lane("nwexl", "inter_nw_3",
                     StraightLane(np.array([179, -440]), np.array([221, -440]), line_types=[c, s]))
        net.add_lane("nwexmr", "inter_nw_3",
                     StraightLane(np.array([179, -432]), np.array([221, -432]), line_types=[s, s]))
        net.add_lane("nwexml", "inter_nw_3",
                     StraightLane(np.array([179, -436]), np.array([221, -436]), line_types=[s, s]))

        net.add_lane("nwne", "internw4",
                     StraightLane(np.array([159, -544]), np.array([159, -494]), line_types=[s, c]))
        net.add_lane("nwne", "internw4",
                     StraightLane(np.array([163, -544]), np.array([163, -494]), line_types=[c, s]))
        net.add_lane("internw4", "nwner",
                     StraightLane(np.array([159, -494]), np.array([159, -452]), line_types=[c, c], forbidden=True))
        net.add_lane("internw4", "nwnel",
                     StraightLane(np.array([163, -494]), np.array([163, -452]), line_types=[c, c], forbidden=True))

        net.add_lane("nwnxr", "inter_nw_4",
                     StraightLane(np.array([171, -452]), np.array([171, -498]), line_types=[s, c]))
        net.add_lane("nwnxl", "inter_nw_4",
                     StraightLane(np.array([167, -452]), np.array([167, -498]), line_types=[c, s]))

        net.add_lane("inter_nw_4", "nwnx",
                     StraightLane(np.array([171, -498]), np.array([171, -548]), line_types=[s, c]))
        net.add_lane("inter_nw_4", "nwnx",
                     StraightLane(np.array([167, -498]), np.array([167, -548]), line_types=[c, s]))

        # bellow: fulfill the turning lanes for vehicles to turn
        net.add_lane("nwner", "nwwxr",
                     StraightLane(np.array([159, -452]), np.array([150, -440]), line_types=[n, n], forbidden=True))

        net.add_lane("nwwer", "nwsxr",
                     StraightLane(np.array([150, -432]), np.array([159, -424]), line_types=[n, n], forbidden=True))
        net.add_lane("nwwel", "nwnxl",
                     StraightLane(np.array([150, -428]), np.array([167, -452]), line_types=[n, n], forbidden=True))
        net.add_lane("nwwel", "nwexmr",
                     StraightLane(np.array([150, -428]), np.array([179, -232]), line_types=[n, n], forbidden=True))

        net.add_lane("nweer", "nwnxr",
                     StraightLane(np.array([179, -448]), np.array([171, -452]), line_types=[n, n], forbidden=True))
        net.add_lane("nweel", "nwsxl",
                     StraightLane(np.array([179, -444]), np.array([163, -424]), line_types=[n, n], forbidden=True))
        net.add_lane("nweel", "nwwxl",
                     StraightLane(np.array([179, -444]), np.array([150, -436]), line_types=[n, n], forbidden=True))

        net.add_lane("nwser", "nwexr",
                     StraightLane(np.array([171, -424]), np.array([179, -428]), line_types=[n, n], forbidden=True))
        net.add_lane("nwsel", "nwwxl",
                     StraightLane(np.array([167, -424]), np.array([150, -436]), line_types=[n, n], forbidden=True))
        net.add_lane("nwsel", "nwnxl",
                     StraightLane(np.array([167, -424]), np.array([167, -452]), line_types=[n, n], forbidden=True))

        net.add_lane("nwnel", "nwexl",
                     StraightLane(np.array([163, -452]), np.array([179, -440]), line_types=[n, n], forbidden=True))
        net.add_lane("nwnel", "nwsxl",
                     StraightLane(np.array([163, -452]), np.array([163, -424]), line_types=[n, n], forbidden=True))

        """
        straight road of north
        internw3                         <-----                         interne_1
        internw3                                                        interne_1

                                         ----->
        internw_3       internm1        internm2        internm3        interne1
        internw_3       internm1        internm2        internm3        interne1
        internw_3       internm1        internm2        internm3        interne1
        internw_3       internm1        internm2                        interne1
                        internm1
        """
        net.add_lane("inter_nw_3", "internm1",
                     StraightLane(np.array([221, -440]), np.array([271, -440]), line_types=[c, s]))
        net.add_lane("inter_nw_3", "internm1",
                     StraightLane(np.array([221, -436]), np.array([271, -436]), line_types=[s, s]))
        net.add_lane("inter_nw_3", "internm1",
                     StraightLane(np.array([221, -432]), np.array([271, -432]), line_types=[s, s]))
        net.add_lane("inter_nw_3", "internm1",
                     StraightLane(np.array([221, -428]), np.array([271, -428]), line_types=[s, s]))
        net.add_lane("inter_nw_3", "internm1",
                     StraightLane(np.array([221, -424]), np.array([271, -424]), line_types=[s, c]))

        net.add_lane("internm1", "internm2",
                     StraightLane(np.array([271, -440]), np.array([321, -440]), line_types=[c, s]))
        net.add_lane("internm1", "internm2",
                     StraightLane(np.array([271, -436]), np.array([321, -436]), line_types=[s, s]))
        net.add_lane("internm1", "internm2",
                     StraightLane(np.array([271, -432]), np.array([321, -432]), line_types=[s, s]))
        net.add_lane("internm1", "internm2",
                     StraightLane(np.array([271, -428]), np.array([321, -428]), line_types=[s, c]))

        # net.add_lane(" ", " ",
        #              StraightLane(np.array([149, -190]), np.array([156, -190]), line_types=[n, c]))
        # net.add_lane(" ", " ",
        #              StraightLane(np.array([155, -190]), np.array([162, -194]), line_types=[n, c]))

        net.add_lane("internm2", "internm3",
                     StraightLane(np.array([321, -440]), np.array([371, -440]), line_types=[c, s]))
        net.add_lane("internm2", "internm3",
                     StraightLane(np.array([321, -436]), np.array([371, -436]), line_types=[s, s]))
        net.add_lane("internm2", "internm3",
                     StraightLane(np.array([321, -432]), np.array([371, -432]), line_types=[s, c]))

        # net.add_lane(" ", " ",
        #              StraightLane(np.array([174, -193]), np.array([176, -189]), line_types=[n, c]))
        #
        # net.add_lane(" ", " ",
        #              StraightLane(np.array([197, -190]), np.array([200, -186]), line_types=[n, c]))
        net.add_lane("internm3", "interne1",
                     StraightLane(np.array([371, -440]), np.array([421, -440]), line_types=[c, s]))
        net.add_lane("internm3", "interne1",
                     StraightLane(np.array([371, -436]), np.array([421, -436]), line_types=[s, s]))
        net.add_lane("internm3", "interne1",
                     StraightLane(np.array([371, -432]), np.array([421, -432]), line_types=[s, s]))
        net.add_lane("internm3", "interne1",
                     StraightLane(np.array([371, -428]), np.array([421, -428]), line_types=[s, c]))

        net.add_lane("inter_ne_1", "internw3",
                     StraightLane(np.array([421, -448]), np.array([221, -448]), line_types=[s, c]))
        net.add_lane("inter_ne_1", "internw3",
                     StraightLane(np.array([421, -444]), np.array([221, -444]), line_types=[c, s]))

        """
        crossroad of northeast
        """

        net.add_lane("interne1", "newel",
                     StraightLane(np.array([421, -440]), np.array([471, -440]), line_types=[c, c], forbidden=True))
        net.add_lane("interne1", "neweml",
                     StraightLane(np.array([421, -436]), np.array([471, -436]), line_types=[c, c], forbidden=True))
        net.add_lane("interne1", "newem",
                     StraightLane(np.array([421, -432]), np.array([471, -432]), line_types=[c, c], forbidden=True))
        net.add_lane("interne1", "newemr",
                     StraightLane(np.array([421, -428]), np.array([471, -428]), line_types=[c, c], forbidden=True))
        net.add_lane("interne1", "newer",
                     StraightLane(np.array([421, -424]), np.array([471, -424]), line_types=[c, c], forbidden=True))
        net.add_lane("newxr", "inter_ne_1",
                     StraightLane(np.array([471, -448]), np.array([421, -448]), line_types=[s, c]))
        net.add_lane("newxl", "inter_ne_1",
                     StraightLane(np.array([471, -444]), np.array([421, -444]), line_types=[c, s]))

        net.add_lane("nesxr", "inter_ne_2",
                     StraightLane(np.array([479, -420]), np.array([479, -370]), line_types=[s, c]))
        net.add_lane("nesxm", "inter_ne_2",
                     StraightLane(np.array([483, -420]), np.array([483, -370]), line_types=[s, s]))
        net.add_lane("nesxl", "inter_ne_2",
                     StraightLane(np.array([487, -420]), np.array([487, -370]), line_types=[c, s]))
        net.add_lane("interne2", "nesel",
                     StraightLane(np.array([491, -370]), np.array([491, -420]), line_types=[c, c], forbidden=True))
        net.add_lane("interne2", "neser",
                     StraightLane(np.array([495, -370]), np.array([495, -420]), line_types=[c, c], forbidden=True))

        net.add_lane("neexl", "inter_ne_3",
                     StraightLane(np.array([503, -440]), np.array([553, -440]), line_types=[c, s]))
        net.add_lane("neexr", "inter_ne_3",
                     StraightLane(np.array([503, -436]), np.array([553, -436]), line_types=[s, c]))

        net.add_lane("inter_ne_3", "neex",
                     StraightLane(np.array([553, -440]), np.array([603, -440]), line_types=[c, s]))
        net.add_lane("inter_ne_3", "neex",
                     StraightLane(np.array([553, -436]), np.array([603, -436]), line_types=[s, c]))

        net.add_lane("interne3", "neeel",
                     StraightLane(np.array([553, -444]), np.array([503, -444]), line_types=[c, c], forbidden=True))
        net.add_lane("interne3", "neeer",
                     StraightLane(np.array([553, -448]), np.array([503, -448]), line_types=[c, c], forbidden=True))

        net.add_lane("neee", "interne3",
                     StraightLane(np.array([603, -444]), np.array([553, -444]), line_types=[c, s]))
        net.add_lane("neee", "interne3",
                     StraightLane(np.array([603, -448]), np.array([553, -448]), line_types=[s, c]))

        net.add_lane("nene", "interne4",
                     StraightLane(np.array([479, -556]), np.array([479, -506]), line_types=[s, c]))
        net.add_lane("nene", "interne4",
                     StraightLane(np.array([483, -556]), np.array([483, -506]), line_types=[c, s]))

        net.add_lane("interne4", "nener",
                     StraightLane(np.array([479, -506]), np.array([479, -456]), line_types=[c, c], forbidden=True))
        net.add_lane("interne4", "nenel",
                     StraightLane(np.array([483, -506]), np.array([483, -456]), line_types=[c, c], forbidden=True))

        net.add_lane("nenxl", "inter_ne_4",
                     StraightLane(np.array([487, -456]), np.array([487, -506]), line_types=[c, s], forbidden=True))
        net.add_lane("nenxr", "inter_ne_4",
                     StraightLane(np.array([491, -456]), np.array([491, -506]), line_types=[s, c], forbidden=True))

        net.add_lane("inter_ne_4", "nenx",
                     StraightLane(np.array([487, -506]), np.array([487, -556]), line_types=[c, s]))
        net.add_lane("inter_ne_4", "nenx",
                     StraightLane(np.array([491, -506]), np.array([491, -556]), line_types=[s, c]))

        # bellow: fulfill the turning lanes for vehicles to turn
        net.add_lane("nener", "newxr",
                     StraightLane(np.array([479, -456]), np.array([471, -448]), line_types=[n, n], forbidden=True))
        net.add_lane("nenel", "neexl",
                     StraightLane(np.array([483, -456]), np.array([503, -440]), line_types=[n, n], forbidden=True))
        net.add_lane("nenel", "nesxm",
                     StraightLane(np.array([483, -456]), np.array([483, -420]), line_types=[n, n], forbidden=True))

        net.add_lane("newer", "nesxr",
                     StraightLane(np.array([471, -424]), np.array([479, -420]), line_types=[n, n], forbidden=True))
        net.add_lane("newel", "nenxl",
                     StraightLane(np.array([471, -440]), np.array([487, -456]), line_types=[n, n], forbidden=True))
        net.add_lane("neweml", "nenxl",
                     StraightLane(np.array([471, -436]), np.array([487, -456]), line_types=[n, n], forbidden=True))
        net.add_lane("newem", "neexl",
                     StraightLane(np.array([471, -432]), np.array([503, -440]), line_types=[n, n], forbidden=True))
        net.add_lane("newemr", "neexl",
                     StraightLane(np.array([471, -428]), np.array([503, -440]), line_types=[n, n], forbidden=True))

        net.add_lane("neser", "neexr",
                     StraightLane(np.array([495, -420]), np.array([503, -436]), line_types=[n, n], forbidden=True))
        net.add_lane("nesel", "nenxl",
                     StraightLane(np.array([491, -420]), np.array([487, -456]), line_types=[n, n], forbidden=True))
        net.add_lane("nesel", "newxl",
                     StraightLane(np.array([491, -420]), np.array([471, -444]), line_types=[n, n], forbidden=True))

        net.add_lane("neeer", "nenxr",
                     StraightLane(np.array([503, -448]), np.array([491, -456]), line_types=[n, n], forbidden=True))
        net.add_lane("neeel", "nesxl",
                     StraightLane(np.array([503, -444]), np.array([487, -420]), line_types=[n, n], forbidden=True))
        net.add_lane("neeel", "newxl",
                     StraightLane(np.array([503, -444]), np.array([471, -444]), line_types=[n, n], forbidden=True))

        """
        straight road of east
        """
        net.add_lane("inter_ne_2", "interem4",
                     StraightLane(np.array([479, -370]), np.array([479, -270]), line_types=[s, c]))
        net.add_lane("inter_ne_2", "interem4",
                     StraightLane(np.array([483, -370]), np.array([483, -270]), line_types=[s, s]))
        net.add_lane("inter_ne_2", "interem4",
                     StraightLane(np.array([487, -370]), np.array([487, -270]), line_types=[c, s]))
        net.add_lane("inter_em_4", "interne2",
                     StraightLane(np.array([491, -270]), np.array([491, -370]), line_types=[c, s]))
        net.add_lane("inter_em_4", "interne2",
                     StraightLane(np.array([495, -270]), np.array([495, -370]), line_types=[s, c]))

        """
        crossroad of east middle
        """
        # net.add_lane(" ", " ",
        #              StraightLane(np.array([224, -140]), np.array([220, -136]), line_types=[c, n]))
        net.add_lane("interem4", "emner",
                     StraightLane(np.array([475, -270]), np.array([475, -220]), line_types=[c, c], forbidden=True))
        net.add_lane("interem4", "emnemr",
                     StraightLane(np.array([479, -270]), np.array([479, -220]), line_types=[c, c], forbidden=True))
        net.add_lane("interem4", "emneml",
                     StraightLane(np.array([483, -270]), np.array([483, -220]), line_types=[c, c], forbidden=True))
        net.add_lane("interem4", "emnel",
                     StraightLane(np.array([487, -270]), np.array([487, -220]), line_types=[c, c], forbidden=True))
        net.add_lane("emnxl", "inter_em_4",
                     StraightLane(np.array([491, -220]), np.array([491, -270]), line_types=[c, s]))
        net.add_lane("emnxr", "inter_em_4",
                     StraightLane(np.array([495, -220]), np.array([495, -270]), line_types=[s, c]))

        net.add_lane("emsxr", "inter_em_2",
                     StraightLane(np.array([475, -196]), np.array([475, -146]), line_types=[s, c]))
        net.add_lane("emsxm", "inter_em_2",
                     StraightLane(np.array([479, -196]), np.array([479, -146]), line_types=[s, s]))
        net.add_lane("emsxl", "inter_em_2",
                     StraightLane(np.array([483, -196]), np.array([483, -146]), line_types=[c, s]))
        net.add_lane("interem2", "emsel",
                     StraightLane(np.array([487, -146]), np.array([487, -196]), line_types=[c, c], forbidden=True))
        net.add_lane("interem2", "emser",
                     StraightLane(np.array([491, -146]), np.array([491, -196]), line_types=[c, c], forbidden=True))

        net.add_lane("emwxr", "inter_em_1",
                     StraightLane(np.array([467, -214]), np.array([417, -214]), line_types=[s, c]))
        net.add_lane("emwxl", "inter_em_1",
                     StraightLane(np.array([467, -210]), np.array([417, -210]), line_types=[c, s]))

        net.add_lane("inter_em_1", "emwx",
                     StraightLane(np.array([417, -214]), np.array([367, -214]), line_types=[s, c]))
        net.add_lane("inter_em_1", "emwx",
                     StraightLane(np.array([417, -210]), np.array([367, -210]), line_types=[c, s]))
        net.add_lane("emwe", "interem1",
                     StraightLane(np.array([367, -206]), np.array([417, -206]), line_types=[c, s]))
        net.add_lane("emwe", "interem1",
                     StraightLane(np.array([367, -202]), np.array([417, -202]), line_types=[s, c]))
        net.add_lane("interem1", "emwel",
                     StraightLane(np.array([417, -206]), np.array([467, -206]), line_types=[c, c], forbidden=True))
        net.add_lane("interem1", "exwer",
                     StraightLane(np.array([417, -202]), np.array([467, -202]), line_types=[c, c], forbidden=True))

        net.add_lane("emee", "interem3",
                     StraightLane(np.array([601, -214]), np.array([551, -214]), line_types=[s, c]))
        net.add_lane("emee", "interem3",
                     StraightLane(np.array([601, -210]), np.array([551, -210]), line_types=[c, s]))
        net.add_lane("interem3", "emeer",
                     StraightLane(np.array([551, -214]), np.array([501, -214]), line_types=[c, c], forbidden=True))
        net.add_lane("interem3", "emeel",
                     StraightLane(np.array([551, -210]), np.array([501, -210]), line_types=[c, c], forbidden=True))

        net.add_lane("emexl", "interem_3",
                     StraightLane(np.array([501, -206]), np.array([551, -206]), line_types=[c, s]))
        net.add_lane("emexr", "interem_3",
                     StraightLane(np.array([501, -202]), np.array([551, -202]), line_types=[s, c]))

        net.add_lane("inter_em_3", "emex",
                     StraightLane(np.array([551, -206]), np.array([601, -206]), line_types=[c, s]))
        net.add_lane("inter_em_3", "emex",
                     StraightLane(np.array([551, -202]), np.array([601, -202]), line_types=[s, c]))

        # bellow: fulfill the turning lanes for vehicles to turn
        net.add_lane("emner", "emwxr",
                     StraightLane(np.array([475, -220]), np.array([467, -214]), line_types=[n, n], forbidden=True))
        net.add_lane("emnemr", "emsxm",
                     StraightLane(np.array([479, -220]), np.array([479, -196]), line_types=[n, n], forbidden=True))
        net.add_lane("emneml", "emsxm",
                     StraightLane(np.array([483, -220]), np.array([479, -196]), line_types=[n, n], forbidden=True))
        net.add_lane("emnel", "emexl",
                     StraightLane(np.array([487, -220]), np.array([501, -206]), line_types=[n, n], forbidden=True))

        net.add_lane("emwer", "emsxr",
                     StraightLane(np.array([467, -202]), np.array([475, -196]), line_types=[n, n], forbidden=True))
        net.add_lane("emwel", "emnxl",
                     StraightLane(np.array([467, -206]), np.array([491, -220]), line_types=[n, n], forbidden=True))
        net.add_lane("emwel", "emexl",
                     StraightLane(np.array([467, -206]), np.array([501, -206]), line_types=[n, n], forbidden=True))

        net.add_lane("emser", "emexr",
                     StraightLane(np.array([491, -196]), np.array([501, -202]), line_types=[n, n], forbidden=True))
        net.add_lane("emsel", "emwxl",
                     StraightLane(np.array([487, -196]), np.array([467, -210]), line_types=[n, n], forbidden=True))
        net.add_lane("emsel", "emnxl",
                     StraightLane(np.array([487, -196]), np.array([491, -220]), line_types=[n, n], forbidden=True))

        net.add_lane("emeer", "emnxr",
                     StraightLane(np.array([501, -214]), np.array([495, -220]), line_types=[n, n], forbidden=True))
        net.add_lane("emeel", "emsxl",
                     StraightLane(np.array([501, -210]), np.array([483, -196]), line_types=[n, n], forbidden=True))
        net.add_lane("emeel", "emwxl",
                     StraightLane(np.array([501, -210]), np.array([467, -210]), line_types=[n, n], forbidden=True))

        """
        straight road of east
        """
        net.add_lane("inter_em_2", "interse4",
                     StraightLane(np.array([475, -146]), np.array([475, -62]), line_types=[s, c]))
        net.add_lane("inter_em_2", "interse4",
                     StraightLane(np.array([479, -146]), np.array([479, -62]), line_types=[s, s]))
        net.add_lane("inter_em_2", "interse4",
                     StraightLane(np.array([483, -146]), np.array([483, -62]), line_types=[c, s]))
        net.add_lane("inter_se_4", "interem2",
                     StraightLane(np.array([487, -62]), np.array([487, -146]), line_types=[c, s]))
        net.add_lane("inter_se_4", "interem2",
                     StraightLane(np.array([491, -62]), np.array([491, -146]), line_types=[s, c]))

        """
        crossroad of southeast
        """
        net.add_lane("interse4", "sener",
                     StraightLane(np.array([475, -62]), np.array([475, -12]), line_types=[c, c], forbidden=True))
        net.add_lane("interse4", "senem",
                     StraightLane(np.array([479, -62]), np.array([479, -12]), line_types=[c, c], forbidden=True))
        net.add_lane("interse4", "senel",
                     StraightLane(np.array([483, -62]), np.array([483, -12]), line_types=[c, c], forbidden=True))
        net.add_lane("senxl", "inter_se_4",
                     StraightLane(np.array([487, -12]), np.array([487, -62]), line_types=[c, s]))
        net.add_lane("senxr", "inter_se_4",
                     StraightLane(np.array([491, -12]), np.array([491, -62]), line_types=[s, c]))

        net.add_lane("sesxr", "inter_se_2",
                     StraightLane(np.array([475, 12]), np.array([475, 62]), line_types=[s, c]))
        net.add_lane("sesxl", "inter_se_2",
                     StraightLane(np.array([479, 12]), np.array([479, 62]), line_types=[c, s]))

        net.add_lane("inter_se_2", "sesx",
                     StraightLane(np.array([475, 62]), np.array([475, 112]), line_types=[s, c]))
        net.add_lane("inter_se_2", "sesx",
                     StraightLane(np.array([479, 62]), np.array([479, 112]), line_types=[c, s]))

        net.add_lane("interse2", "sesel",
                     StraightLane(np.array([483, 62]), np.array([483, 12]), line_types=[c, c], forbidden=True))
        net.add_lane("interse2", "seser",
                     StraightLane(np.array([487, 62]), np.array([487, 12]), line_types=[c, c], forbidden=True))

        net.add_lane("sese", "interse2",
                     StraightLane(np.array([483, 112]), np.array([483, 62]), line_types=[c, s]))
        net.add_lane("sese", "interse2",
                     StraightLane(np.array([487, 112]), np.array([487, 62]), line_types=[s, c]))

        net.add_lane("interse1", "sewel",
                     StraightLane(np.array([418, 0]), np.array([468, 0]), line_types=[c, c], forbidden=True))
        net.add_lane("interse1", "sewer",
                     StraightLane(np.array([418, 4]), np.array([468, 4]), line_types=[c, c], forbidden=True))
        net.add_lane("sewxr", "inter_se_1",
                     StraightLane(np.array([468, -4]), np.array([418, -4]), line_types=[c, s]))
        net.add_lane("sewxl", "inter_se_1",
                     StraightLane(np.array([468, -8]), np.array([418, -8]), line_types=[s, c]))

        net.add_lane("seexl", "inter_se_3",
                     StraightLane(np.array([496, 0]), np.array([546, 0]), line_types=[c, s]))
        net.add_lane("seexr", "inter_se_3",
                     StraightLane(np.array([496, 4]), np.array([546, 4]), line_types=[s, c]))

        net.add_lane("inter_se_3", "seex",
                     StraightLane(np.array([546, 0]), np.array([596, 0]), line_types=[c, s]))
        net.add_lane("inter_se_3", "seex",
                     StraightLane(np.array([546, 4]), np.array([596, 4]), line_types=[s, c]))

        net.add_lane("seee", "interse3",
                     StraightLane(np.array([596, -4]), np.array([546, -4]), line_types=[c, s]))
        net.add_lane("seee", "interse3",
                     StraightLane(np.array([596, -8]), np.array([546, -8]), line_types=[s, c]))

        net.add_lane("interse3", "seeel",
                     StraightLane(np.array([546, -4]), np.array([496, -4]), line_types=[c, c], forbidden=True))
        net.add_lane("interse3", "seeer",
                     StraightLane(np.array([546, -8]), np.array([496, -8]), line_types=[c, c], forbidden=True))

        # bellow: fulfill the turning lanes for vehicles to turn
        net.add_lane("sewer", "sesxr",
                     StraightLane(np.array([468, 4]), np.array([475, 12]), line_types=[n, n], forbidden=True))
        net.add_lane("sewel", "senxl",
                     StraightLane(np.array([468, 0]), np.array([487, -12]), line_types=[n, n], forbidden=True))
        net.add_lane("sewel", "seexl",
                     StraightLane(np.array([468, 0]), np.array([496, 0]), line_types=[n, n], forbidden=True))

        net.add_lane("seser", "seexr",
                     StraightLane(np.array([487, 12]), np.array([496, 4]), line_types=[n, n], forbidden=True))
        net.add_lane("sesel", "sewxl",
                     StraightLane(np.array([483, 12]), np.array([468, -8]), line_types=[n, n], forbidden=True))
        net.add_lane("sesel", "senxl",
                     StraightLane(np.array([483, 12]), np.array([487, -12]), line_types=[n, n], forbidden=True))

        net.add_lane("seeer", "senxr",
                     StraightLane(np.array([496, -8]), np.array([491, -12]), line_types=[n, n], forbidden=True))
        net.add_lane("seeel", "sesxl",
                     StraightLane(np.array([496, -4]), np.array([479, 12]), line_types=[n, n], forbidden=True))
        net.add_lane("seeel", "sewxl",
                     StraightLane(np.array([496, -4]), np.array([468, -8]), line_types=[n, n], forbidden=True))

        net.add_lane("sener", "sewxr",
                     StraightLane(np.array([475, -12]), np.array([468, -4]), line_types=[n, n], forbidden=True))
        net.add_lane("senem", "sesxl",
                     StraightLane(np.array([479, -12]), np.array([479, 12]), line_types=[n, n], forbidden=True))
        net.add_lane("senel", "seexl",
                     StraightLane(np.array([483, -12]), np.array([496, 0]), line_types=[n, n], forbidden=True))

        """
        straight road of south
        """
        net.add_lane("intersw_3", "intersm_1",
                     StraightLane(np.array([228, 0]), np.array([258, 0]), line_types=[c, s]))
        net.add_lane("intersw_3", "intersm_1",
                     StraightLane(np.array([228, 4]), np.array([258, 4]), line_types=[s, c]))
        net.add_lane("intersm1", "intersw3",
                     StraightLane(np.array([258, -4]), np.array([228, -4]), line_types=[c, s]))
        net.add_lane("intersm1", "intersw3",
                     StraightLane(np.array([258, -8]), np.array([228, -8]), line_types=[s, c]))

        # net.add_lane(" ", " ",
        #              StraightLane(np.array([125, -12]), np.array([120, -12]), line_types=[n, c]))
        # net.add_lane(" ", " ",
        #              StraightLane(np.array([120, -12]), np.array([115, -8]), line_types=[n, c]))

        net.add_lane("inter_sm_1", "inter_sm_2",
                     StraightLane(np.array([258, 0]), np.array([288, 0]), line_types=[c, s]))
        net.add_lane("inter_sm_1", "inter_sm_2",
                     StraightLane(np.array([258, 4]), np.array([288, 4]), line_types=[s, c]))
        net.add_lane("intersm2", "intersm1",
                     StraightLane(np.array([288, -4]), np.array([258, -4]), line_types=[c, s]))
        net.add_lane("intersm2", "intersm1",
                     StraightLane(np.array([288, -8]), np.array([258, -8]), line_types=[s, s]))
        net.add_lane("intersm2", "intersm1",
                     StraightLane(np.array([288, -12]), np.array([258, -12]), line_types=[s, c]))

        # net.add_lane(" ", " ",
        #              StraightLane(np.array([155, -8]), np.array([152, -12]), line_types=[n, c]))

        net.add_lane("inter_sm_2", "inter_sm_3",
                     StraightLane(np.array([288, 0]), np.array([318, 0]), line_types=[c, s]))
        net.add_lane("inter_sm_2", "inter_sm_3",
                     StraightLane(np.array([288, 4]), np.array([318, 4]), line_types=[s, c]))
        net.add_lane("intersm3", "intersm2",
                     StraightLane(np.array([318, -4]), np.array([288, -4]), line_types=[c, s]))
        net.add_lane("intersm3", "intersm2",
                     StraightLane(np.array([318, -8]), np.array([288, -8]), line_types=[s, c]))

        # net.add_lane(" ", " ",
        #              StraightLane(np.array([170, -12]), np.array([165, -12]), line_types=[n, c]))
        # net.add_lane(" ", " ",
        #              StraightLane(np.array([165, -12]), np.array([160, -8]), line_types=[n, c]))

        net.add_lane("inter_sm_3", "inter_sm_4",
                     StraightLane(np.array([318, 0]), np.array([348, 0]), line_types=[c, s]))
        net.add_lane("inter_sm_3", "inter_sm_4",
                     StraightLane(np.array([318, 4]), np.array([348, 4]), line_types=[s, c]))
        net.add_lane("intersm4", "intersm3",
                     StraightLane(np.array([348, -4]), np.array([318, -4]), line_types=[c, s]))
        net.add_lane("intersm4", "intersm3",
                     StraightLane(np.array([348, -8]), np.array([318, -8]), line_types=[s, s]))
        net.add_lane("intersm4", "intersm3",
                     StraightLane(np.array([348, -12]), np.array([318, -12]), line_types=[s, c]))

        net.add_lane("inter_sm_4", "inter_sm_5",
                     StraightLane(np.array([348, 0]), np.array([378, 0]), line_types=[c, s]))
        net.add_lane("inter_sm_4", "inter_sm_5",
                     StraightLane(np.array([348, 4]), np.array([378, 4]), line_types=[s, c]))
        net.add_lane("intersm5", "intersm4",
                     StraightLane(np.array([378, -4]), np.array([348, -4]), line_types=[c, s]))
        net.add_lane("intersm5", "intersm4",
                     StraightLane(np.array([378, -8]), np.array([348, -8]), line_types=[s, c]))

        net.add_lane("inter_sm_5", "interse1",
                     StraightLane(np.array([378, 0]), np.array([418, 0]), line_types=[c, s]))
        net.add_lane("inter_sm_5", "interse1",
                     StraightLane(np.array([378, 4]), np.array([418, 4]), line_types=[s, c]))

        # net.add_lane(" ", " ",
        #              StraightLane(np.array([199, -8]), np.array([195, -12]), line_types=[n, c]))
        net.add_lane("inter_se_1", "intersm5",
                     StraightLane(np.array([418, -4]), np.array([378, -4]), line_types=[c, s]))
        net.add_lane("inter_se_1", "intersm5",
                     StraightLane(np.array([418, -8]), np.array([378, -8]), line_types=[s, s]))
        net.add_lane("inter_se_1", "intersm5",
                     StraightLane(np.array([418, -12]), np.array([378, -12]), line_types=[s, c]))

        road = Road(network=net, np_random=self.np_random)

        green_time = 5
        red_time = 8
        green_flash_time = 2
        yellow_time = 1
        """
        southwest crossroad traffic lights
        """
        self.traffic_lights["red_sw"] = [
            RedLight(road, [150, 0], red_time, green_time, green_flash_time, yellow_time, 1),
            RedLight(road, [150, 4], red_time, green_time, green_flash_time, yellow_time, 1),
            RedLight(road, [167, 8], red_time, green_time, green_flash_time, yellow_time, 0),
            RedLight(road, [171, 8], red_time, green_time, green_flash_time, yellow_time, 0),
            RedLight(road, [178, -8], red_time, green_time, green_flash_time, yellow_time, 1),
            RedLight(road, [178, -4], red_time, green_time, green_flash_time, yellow_time, 1),
            RedLight(road, [159, -12], red_time, green_time, green_flash_time, yellow_time, 0),
            RedLight(road, [163, -12], red_time, green_time, green_flash_time, yellow_time, 0),
        ]

        """
        west middle crossroad traffic lights
        """

        self.traffic_lights["red_wm"] = [
            RedLight(road, [150, -216], red_time, green_time, green_flash_time, yellow_time, 1),
            RedLight(road, [150, -220], red_time, green_time, green_flash_time, yellow_time, 1),
            RedLight(road, [167, -212], red_time, green_time, green_flash_time, yellow_time, 0),
            RedLight(road, [171, -212], red_time, green_time, green_flash_time, yellow_time, 0),
            RedLight(road, [180, -228], red_time, green_time, green_flash_time, yellow_time, 1),
            RedLight(road, [180, -224], red_time, green_time, green_flash_time, yellow_time, 1),
            RedLight(road, [159, -232], red_time, green_time, green_flash_time, yellow_time, 0),
            RedLight(road, [163, -232], red_time, green_time, green_flash_time, yellow_time, 0)
        ]

        """
        northwest crossroad traffic light
        """

        self.traffic_lights["red_nw"] = [
            RedLight(road, [150, -428], red_time, green_time, green_flash_time, yellow_time, 1),
            RedLight(road, [150, -432], red_time, green_time, green_flash_time, yellow_time, 1),
            RedLight(road, [167, -424], red_time, green_time, green_flash_time, yellow_time, 0),
            RedLight(road, [171, -424], red_time, green_time, green_flash_time, yellow_time, 0),
            RedLight(road, [180, -448], red_time, green_time, green_flash_time, yellow_time, 1),
            RedLight(road, [180, -444], red_time, green_time, green_flash_time, yellow_time, 1),
            RedLight(road, [159, -452], red_time, green_time, green_flash_time, yellow_time, 0),
            RedLight(road, [163, -452], red_time, green_time, green_flash_time, yellow_time, 0)
        ]

        """
        northeast crossroad traffic light
        """

        self.traffic_lights["red_ne"] = [
            RedLight(road, [471, -440], red_time, green_time, green_flash_time, yellow_time, 1),
            RedLight(road, [471, -436], red_time, green_time, green_flash_time, yellow_time, 1),
            RedLight(road, [471, -432], red_time, green_time, green_flash_time, yellow_time, 1),
            RedLight(road, [471, -428], red_time, green_time, green_flash_time, yellow_time, 1),
            RedLight(road, [471, -424], red_time, green_time, green_flash_time, yellow_time, 1),
            RedLight(road, [503, -448], red_time, green_time, green_flash_time, yellow_time, 1),
            RedLight(road, [503, -444], red_time, green_time, green_flash_time, yellow_time, 1),
            RedLight(road, [479, -456], red_time, green_time, green_flash_time, yellow_time, 0),
            RedLight(road, [483, -456], red_time, green_time, green_flash_time, yellow_time, 0),
            RedLight(road, [491, -420], red_time, green_time, green_flash_time, yellow_time, 0),
            RedLight(road, [495, -420], red_time, green_time, green_flash_time, yellow_time, 0)
        ]

        """
        east middle crossroad traffic light
        """

        self.traffic_lights["red_em"] = [
            RedLight(road, [467, -202], red_time, green_time, green_flash_time, yellow_time, 1),
            RedLight(road, [467, -206], red_time, green_time, green_flash_time, yellow_time, 1),
            RedLight(road, [501, -214], red_time, green_time, green_flash_time, yellow_time, 1),
            RedLight(road, [501, -210], red_time, green_time, green_flash_time, yellow_time, 1),
            RedLight(road, [475, -220], red_time, green_time, green_flash_time, yellow_time, 0),
            RedLight(road, [479, -220], red_time, green_time, green_flash_time, yellow_time, 0),
            RedLight(road, [483, -220], red_time, green_time, green_flash_time, yellow_time, 0),
            RedLight(road, [487, -220], red_time, green_time, green_flash_time, yellow_time, 0),
            RedLight(road, [491, -196], red_time, green_time, green_flash_time, yellow_time, 0),
            RedLight(road, [487, -196], red_time, green_time, green_flash_time, yellow_time, 0)
        ]

        """
        southeast crossroad traffic light
        """

        self.traffic_lights["red_se"] = [
            RedLight(road, [468, 0], red_time, green_time, green_flash_time, yellow_time, 1),
            RedLight(road, [468, 4], red_time, green_time, green_flash_time, yellow_time, 1),
            RedLight(road, [496, -8], red_time, green_time, green_flash_time, yellow_time, 1),
            RedLight(road, [496, -4], red_time, green_time, green_flash_time, yellow_time, 1),
            RedLight(road, [475, -12], red_time, green_time, green_flash_time, yellow_time, 0),
            RedLight(road, [479, -12], red_time, green_time, green_flash_time, yellow_time, 0),
            RedLight(road, [483, -12], red_time, green_time, green_flash_time, yellow_time, 0),
            RedLight(road, [483, 12], red_time, green_time, green_flash_time, yellow_time, 0),
            RedLight(road, [487, 12], red_time, green_time, green_flash_time, yellow_time, 0)
        ]

        self.road = road

        # for lane_index in road.network.LANES:
        #     _from, _to, _id = lane_index
        #     _before = None
        #     next_to = None
        #     lane = self.road.network.get_lane(lane_index)
        #     try:
        #         next_to = list(self.road.network.graph[_to].keys())[
        #             np.random.randint(len(self.road.network.graph[_to]))]
        #         if len(self.road.network.graph[_from][_to]) <= len(self.road.network.graph[_to][next_to]):
        #             next_id = _id
        #         else:
        #             if _id + 1 > len(self.road.network.graph[_to][next_to]):
        #                 next_id = len(self.road.network.graph[_to][next_to]) - 1
        #             else:
        #                 next_id = _id
        #         if (_to, next_to, next_id) in self.road.network.LANES:
        #             lane.after_lane.append((_to, next_to, next_id))
        #     except KeyError:
        #         pass
        #     try:
        #         for _key in (list(self.road.network.graph.keys())):
        #             if _from in list(self.road.network.graph[_key].keys()):
        #                 _before = _key
        #                 break
        #
        #         if _before:
        #             if len(self.road.network.graph[_before][_from]) < len(self.road.network.graph[_from][_to]):
        #                 if _id <= len(self.road.network.graph[_before][_from]) - 1:
        #                     before_id = _id
        #                 else:
        #                     before_id = len(self.road.network.graph[_before][_from]) - 1
        #             else:
        #                 before_id = _id
        #             if (_before, _from, before_id) in self.road.network.LANES:
        #                 lane.before_lane.append((_before, _from, before_id))
        #     except KeyError:
        #         pass

    def make_vehicles(self):
        """
            Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        """
        # road = self.road

        ego_lane = self.road.network.get_lane(("intersm1", "intersw3", 0))
        position = ego_lane.position(0, 0)
        ego_vehicle = IDMVehicle(self.road,
                                 position,
                                 velocity=10,
                                 heading=ego_lane.heading_at(position)).plan_route_to("intersm2")
        ego_vehicle.id = 0
        # ego_vehicle1.myimage = pygame.image.load("../red_alpha_resize.png")
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle
        # print("vehicle lane_index:", self.vehicle.lane_index)

    def fake_step(self):
        """
        :return:
        """
        self.vehicle.lanes_around = []
        active_vehicles = 0

        for i in self.road.network.LANES:
            if i == self.vehicle.lane_index:
                self.vehicle.lanes_around.append(i)
                continue
            l = self.road.network.get_lane(i)
            l_middle = (l.end - l.start) / 2 + l.start
            # s , _= l.local_coordinates(self.vehicle.position)
            if np.linalg.norm(l_middle - self.vehicle.position) < 75: # and -100 <= s <= 50:
                self.vehicle.lanes_around.append(i)

        flag = 1
        while flag:
            for i in range(len(self.road.vehicles)):
                if i == len(self.road.vehicles) - 1:
                    flag = 0
                if hasattr(self.road.vehicles[i], "state"):
                    continue
                else:
                    if self.road.vehicles[i].lane_index[1][-1] == "x" or (np.linalg.norm(self.road.vehicles[i].position - self.vehicle.position) >= 75 and not hasattr(self.road.vehicles[i],"state")):
                        # print("lane_index",self.road.vehicles[i].lane_index[1])
                        self.road.vehicles.remove(self.road.vehicles[i])
                        break
                    else:
                        active_vehicles += 1

        self.get_traffic_lights()
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
        # from highway_env.extractors import Extractor
        # extractor = Extractor()
        # extractor_features = extractor.FeatureExtractor(self.road.vehicles, 0, 1)

        birth_place = self.vehicle.lanes_around
        flag = 1
        while flag:
            for i in range(len(birth_place)):
                if i == len(birth_place) - 1:
                    flag = 0
                if birth_place[i][1].find("inter") == -1:
                    birth_place.remove(birth_place[i])
                    break
        pre_birth = None
        for i in range(2):
            try:
                velocity_deviation = 1.0
                velocity = 5 + np.random.randint(1, 5) * velocity_deviation
                other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

                birth = birth_place[np.random.randint(0, len(birth_place))]
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

    def get_traffic_lights(self):
        if self.have_traffic_lights:
            print("have traffic lights")
            if self.vehicle.lane_index[1].find("_") != -1:
                flag = 1
                while flag:
                    for i in range(len(self.road.vehicles)):
                        # print("i:",i)
                        if i == len(self.road.vehicles) - 1:
                            flag = 0
                        if hasattr(self.road.vehicles[i], "state"):
                            self.road.vehicles.remove(self.road.vehicles[i])
                            break
                self.have_traffic_lights = False
        elif not self.have_traffic_lights:
            print("no traffic lights")
            if self.vehicle.lane_index[1].find("intersw") != -1:
                for _red in self.traffic_lights["red_sw"]:
                    self.road.vehicles.append(_red)
                self.have_traffic_lights = True
            elif self.vehicle.lane_index[1].find("interwm") != -1:
                for _red in self.traffic_lights["red_wm"]:
                    self.road.vehicles.append(_red)
                self.have_traffic_lights = True
            elif self.vehicle.lane_index[1].find("internw") != -1:
                for _red in self.traffic_lights["red_nw"]:
                    self.road.vehicles.append(_red)
                self.have_traffic_lights = True
            elif self.vehicle.lane_index[1].find("interne") != -1:
                for _red in self.traffic_lights["red_ne"]:
                    self.road.vehicles.append(_red)
                self.have_traffic_lights = True
            elif self.vehicle.lane_index[1].find("interem") != -1:
                for _red in self.traffic_lights["red_em"]:
                    self.road.vehicles.append(_red)
                self.have_traffic_lights = True
            elif self.vehicle.lane_index[1].find("interse") != -1:
                for _red in self.traffic_lights["red_se"]:
                    self.road.vehicles.append(_red)
                self.have_traffic_lights = True
            else:
                pass
        else:
            return


if __name__ == '__main__':
    pass
