from __future__ import division, print_function

import pygame
from collections import deque
import numpy as np
from highway_env.vehicle.control import ControlledVehicle
from highway_env import utils
from highway_env.vehicle.dynamics import RedLight,Obstacle
import random


class IDMVehicle(ControlledVehicle):
    """
        A vehicle using both a longitudinal and a lateral decision policies.

        - Longitudinal: the IDM model computes an acceleration given the preceding vehicle's distance and velocity.
        - Lateral: the MOBIL model decides when to change lane by maximizing the acceleration of nearby vehicles.
    IDM = {
        'ACC_MAX': 6.0, # [m/s2]
        'COMFORT_ACC_MAX': 3.0, # [m/s2]
        'COMFORT_ACC_MIN': -5.0, # [m/s2]
        'VELOCITY_WANTED': 25.0, # [m/s]
        'DISTANCE_WANTED':  5, # [m]
        'TIME_WANTED': 1.5, # [s]
        'DELTA': 4.0, # []
    }
    MOBIL = {
        'POLITENESS': 0., # in [0, 1]
        'LANE_CHANGE_MIN_ACC_GAIN': 0.4, # [m/s2]
        'RIGHT_LANE_CHANGE_MIN_ACC_GAIN': 0.4,
        'LANE_CHANGE_MAX_BRAKING_IMPOSED': 1.0, # [m/s2]
        'LANE_CHANGE_DELAY': 1.0, # [s]
    }
    """
    myimage = pygame.image.load("../blue_alpha_resize.png")
    IDM = {
        'ACC_MAX': 8.0,  # [m/s2]
        'COMFORT_ACC_MAX': 3.0,  # [m/s2]
        'COMFORT_ACC_MIN': -5.0,  # [m/s2]
        'VELOCITY_WANTED': 20.0,  # [m/s]
        'DISTANCE_WANTED': 5,  # [m]
        'TIME_WANTED': 1.5,  # [s]
        'DELTA': 4.0,  # []
    }
    MOBIL = {
        'POLITENESS': 0.,  # in [0, 1]
        'LANE_CHANGE_MIN_ACC_GAIN': 0.1,  # [m/s2]
        'RIGHT_LANE_CHANGE_MIN_ACC_GAIN': 0.1,
        'LANE_CHANGE_MAX_BRAKING_IMPOSED': 1.0,  # [m/s2]
        'LANE_CHANGE_DELAY': 0.5,  # [s]
    }
    level = {
        0: (0, 0),
        1: (0, 0.05),
        2: (0.05, 0.1),
        3: (0.1, 0.15),
        -1: (-0.05, 0),
        -2: (-0.1, -0.05),
        -3: (-0.15, -0.1),
    }

    @classmethod
    def generate_style(cls, self):
        '''

        :param level: higher means more aggressive
        :return:
        '''
        l = IDMVehicle.level[self.level]
        adjust = np.random.uniform(*l)
        for k, v in IDMVehicle.IDM.items():
            setattr(self, k, v + v * adjust)
        for k, v in IDMVehicle.MOBIL.items():
            setattr(self, k, v + v * adjust)

    # Longitudinal policy parameters
    ACC_MAX = 8.0  # [m/s2]
    COMFORT_ACC_MAX = 5.0  # [m/s2]
    COMFORT_ACC_MIN = -5.0  # [m/s2]
    VELOCITY_WANTED = 25.0  # [m/s]
    DISTANCE_WANTED = 2  # [m]
    TIME_WANTED = 0.5  # [s]
    DELTA = 4.0  # []

    # Lateral policy parameters
    POLITENESS = 0.  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.1  # [m/s2]
    RIGHT_LANE_CHANGE_MIN_ACC_GAIN = LANE_CHANGE_MIN_ACC_GAIN
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 1.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]
    SIMULATION_FREQUENCY = 15

    def __init__(self, road, position,
                 heading=0,
                 velocity=0,
                 target_lane_index=None,
                 target_velocity=None,
                 enable_lane_change=True,
                 timer=None,
                 dst=None,
                 rever=False,
                 max_length=None,
                 route=None):
        super(IDMVehicle, self).__init__(road, position, heading, velocity, target_lane_index, target_velocity, route)
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position) * np.pi) % self.LANE_CHANGE_DELAY
        self.acc = 0.0
        self.pre_acc = self.acc
        self.level = np.random.randint(-3, 4)
        self.DELTA = 0.0  # []
        self.rever = rever
        self.dst_index = dst
        if dst is not None:
            if not rever:
                self.dst = deque(range(self.lane_index + 1, dst + 1))
            else:
                self.dst = deque(range(self.lane_index - 1, dst - 1, -1))
        else:
            self.dst = None
        IDMVehicle.generate_style(self)
        self.agent = 'idm'
        # self.lane.min_positon = min(self.lane.min_position,self.position[0])
        # self.position_reset = self.position.copy()
        # self.dst_reset = self.dst.copy()
        # self.lane_index_reset = self.lane_index

    def reset(self):
        cls = type(self)
        # lane_index = random.choice(range(len(self.road.lanes)))
        # lane = self.road.lanes[lane_index]
        lane = self.lane
        position = lane.position(np.random.randint(20, 40), 0)
        # if lane_index < 2:
        #     v = cls(self.road,position, velocity=np.random.randint(18,26),dst=self.dst,max_length=self.max_length,rever=self.rever)
        # else:
        #     v = cls(self.road,position, velocity=np.random.randint(18,26),dst=self.dst,max_length=self.max_length,rever=self.rever)
        v = cls(self.road, position, velocity=np.random.randint(18, 26), dst=self.dst_index, max_length=self.max_length,
                rever=self.rever)
        return v

    def randomize_behavior(self):
        pass

    @classmethod
    def create_from(cls, vehicle):
        """
            Create a new vehicle from an existing one.
            The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position, heading=vehicle.heading, velocity=vehicle.velocity,
                target_lane_index=vehicle.target_lane_index, target_velocity=vehicle.target_velocity,
                timer=getattr(vehicle, 'timer', None))
        return v

    def act(self, action=None,vehicle_actions = None):
        """
            Execute an action.

            For now, no action is supported because the vehicle takes all decisions
            of acceleration and lane changes on its own, based on the IDM and MOBIL models.

        :param vehicle_actions: test for CRUISE / LEFT_CHANGE / RIGHT_CHANGE / STOP / TAKEOVER
        :param action: the action
        """
        if self.crashed:
            return
        action = {}
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)

        # Lateral: MOBIL
        self.follow_road()
        if self.enable_lane_change and not self.lane.forbidden:
            self.change_lane_policy()
        action['steering'] = self.steering_control(self.target_lane_index)

        # Longitudinal: IDM
        self.pre_acc = 0.0 if self.acc is None else self.acc
        action['acceleration'] = self.acceleration(ego_vehicle=self,
                                                              front_vehicle=front_vehicle,
                                                              rear_vehicle=rear_vehicle)
        # action['acceleration'] = self.recover_from_stop(action['acceleration'])
        self.acc = action['acceleration'] = np.clip(action['acceleration'], -self.ACC_MAX, self.ACC_MAX)
        super(ControlledVehicle, self).act(action)

    def step(self, dt):
        """
            Step the simulation.

            Increases a timer used for decision policies, and step the vehicle dynamics.

        :param dt: timestep
        """
        self.timer += dt
        super(IDMVehicle, self).step(dt)
        # print(self.id,self.dst,self.lane_index)
        if self.dst is not None and len(self.dst) > 1 and self.dst[0] == self.lane_index:
            self.dst.popleft()

    @classmethod
    def acc_self_def(cls, ego_vehicle, front_vehicle=None, rear_vehicle=None):
        """
        self defined acc considering the vehicles of which the target lane index is the current lane
        """

        if not ego_vehicle:
            return 0
        # decelerate at the crossroad
        target_velocity = ego_vehicle.target_velocity
        if hasattr(front_vehicle, "state") and front_vehicle.state != "RED":

            # run with a given velocity at the crossroad
            if ego_vehicle.lane_distance_to(front_vehicle) < 40 and target_velocity > 30 / 3.6:
                target_velocity = 30 / 3.6
            front_vehicle = None
            # rear_vehicle = None
        acceleration = ego_vehicle.COMFORT_ACC_MAX * (
                1 - np.power(ego_vehicle.velocity / utils.not_zero(target_velocity), ego_vehicle.DELTA)) \
            if isinstance(ego_vehicle, IDMVehicle) else \
            cls.COMFORT_ACC_MAX * (
                    1 - np.power(ego_vehicle.velocity / utils.not_zero(target_velocity), cls.DELTA))
        if hasattr(front_vehicle, "state") and front_vehicle.state == "RED":
            if ego_vehicle.lane_distance_to(front_vehicle) < ego_vehicle.LENGTH / 2 + front_vehicle.LENGTH / 2:
                front_vehicle = None

        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle)
            acceleration -= ego_vehicle.COMFORT_ACC_MAX * \
                            np.power(cls.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2) \
                if isinstance(ego_vehicle, IDMVehicle) else \
                cls.COMFORT_ACC_MAX * \
                np.power(cls.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2)
        # the acceleration will not exceed COMFORT_ACC_MAX

        for lane_index in ego_vehicle.road.network.side_lanes_bigmap(ego_vehicle):
            # Is the candidate lane close enough?
            front_vehicle, rear_vehicle = ego_vehicle.road.neighbour_vehicles(ego_vehicle, lane_index)
            # if hasattr(front_vehicle, "state"):
            #     front_vehicle = None
            if hasattr(front_vehicle, "state") and front_vehicle.state != "RED":

                # run with a given velocity at the crossroad
                if ego_vehicle.lane_distance_to(front_vehicle) < 40 and target_velocity > 30 / 3.6:
                    target_velocity = 30 / 3.6
                front_vehicle = None
            # if hasattr(front_vehicle,"state"):
            #     print("state:",front_vehicle.state)
            if front_vehicle is not None and not isinstance(front_vehicle,RedLight) and front_vehicle.target_lane_index != ego_vehicle.lane_index:
                continue
            target_velocity = ego_vehicle.target_velocity

            # rear_vehicle = None
            acceleration_new = ego_vehicle.COMFORT_ACC_MAX * (
                    1 - np.power(ego_vehicle.velocity / utils.not_zero(target_velocity), ego_vehicle.DELTA)) \
                if isinstance(ego_vehicle, IDMVehicle) else \
                cls.COMFORT_ACC_MAX * (
                        1 - np.power(ego_vehicle.velocity / utils.not_zero(target_velocity), cls.DELTA))
            if hasattr(front_vehicle, "state") and front_vehicle.state == "RED":
                if ego_vehicle.lane_distance_to(front_vehicle) < ego_vehicle.LENGTH / 2 + front_vehicle.LENGTH / 2:
                    front_vehicle = None

            if front_vehicle:
                d = ego_vehicle.lane_distance_to(front_vehicle)
                acceleration_new -= ego_vehicle.COMFORT_ACC_MAX * \
                                    np.power(cls.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2) \
                    if isinstance(ego_vehicle, IDMVehicle) else \
                    cls.COMFORT_ACC_MAX * \
                    np.power(cls.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2)
            if acceleration_new < acceleration:
                acceleration = acceleration_new

        if ego_vehicle.acc < acceleration and ego_vehicle.acc < ego_vehicle.COMFORT_ACC_MAX:
            ego_vehicle.acc += ego_vehicle.COMFORT_ACC_MAX / ego_vehicle.SIMULATION_FREQUENCY
        elif ego_vehicle.acc > acceleration and ego_vehicle.acc > ego_vehicle.COMFORT_ACC_MIN:
            ego_vehicle.acc += ego_vehicle.COMFORT_ACC_MIN / ego_vehicle.SIMULATION_FREQUENCY

        return acceleration

    @classmethod
    def acceleration(cls, ego_vehicle, front_vehicle=None, rear_vehicle=None):
        """
            Compute an acceleration command with the Intelligent Driver Model.

            The acceleration is chosen so as to:
            - reach a target velocity;
            - maintain a minimum safety distance (and safety time) w.r.t the front vehicle.

        :param ego_vehicle: the vehicle whose desired acceleration is to be computed. It does not have to be an
                            IDM vehicle, which is why this method is a class method. This allows an IDM vehicle to
                            reason about other vehicles behaviors even though they may not IDMs.
        :param front_vehicle: the vehicle preceding the ego-vehicle
        :param rear_vehicle: the vehicle following the ego-vehicle
        :return: the acceleration command for the ego-vehicle [m/s2]
        """
        if not ego_vehicle:
            return 0
        # decelerate at the crossroad
        target_velocity = ego_vehicle.target_velocity
        if hasattr(front_vehicle, "state") and front_vehicle.state == "GREEN":

            # run with a given velocity at the crossroad
            if ego_vehicle.lane_distance_to(front_vehicle) < 35:
                target_velocity = 30 / 3.6
            front_vehicle = None
            # rear_vehicle = None
        acceleration = ego_vehicle.COMFORT_ACC_MAX * (
                1 - np.power(ego_vehicle.velocity / utils.not_zero(target_velocity), ego_vehicle.DELTA)) \
            if isinstance(ego_vehicle, IDMVehicle) else \
            cls.COMFORT_ACC_MAX * (
                    1 - np.power(ego_vehicle.velocity / utils.not_zero(target_velocity), cls.DELTA))
        if hasattr(front_vehicle, "state") and front_vehicle.state == "RED":
            if ego_vehicle.lane_distance_to(front_vehicle) < ego_vehicle.LENGTH / 2 + front_vehicle.LENGTH / 2:
                front_vehicle = None
        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle)
            acceleration -= ego_vehicle.COMFORT_ACC_MAX * \
                            np.power(cls.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2) \
                if isinstance(ego_vehicle, IDMVehicle) else \
                cls.COMFORT_ACC_MAX * \
                np.power(cls.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2)
        return acceleration

    @classmethod
    def desired_gap(cls, ego_vehicle, front_vehicle=None):
        """
            Compute the desired distance between a vehicle and its leading vehicle.

        :param ego_vehicle: the vehicle being controlled
        :param front_vehicle: its leading vehicle
        :return: the desired distance between the two [m]
        """
        if isinstance(ego_vehicle, IDMVehicle):
            d0 = ego_vehicle.DISTANCE_WANTED + ego_vehicle.LENGTH / 2 + front_vehicle.LENGTH / 2
            tau = ego_vehicle.TIME_WANTED
            ab = -ego_vehicle.COMFORT_ACC_MAX * ego_vehicle.COMFORT_ACC_MIN
            dv = ego_vehicle.velocity - front_vehicle.velocity
            d_star = d0 + ego_vehicle.velocity * tau + ego_vehicle.velocity * dv / (2 * np.sqrt(ab))
        else:
            d0 = cls.DISTANCE_WANTED + ego_vehicle.LENGTH / 2 + front_vehicle.LENGTH / 2
            tau = cls.TIME_WANTED
            ab = -cls.COMFORT_ACC_MAX * cls.COMFORT_ACC_MIN
            dv = ego_vehicle.velocity - front_vehicle.velocity
            d_star = d0 + ego_vehicle.velocity * tau + ego_vehicle.velocity * dv / (2 * np.sqrt(ab))
        return d_star

    def maximum_velocity(self, front_vehicle=None):
        """
            Compute the maximum allowed velocity to avoid Inevitable Collision States.

            Assume the front vehicle is going to brake at full deceleration and that
            it will be noticed after a given delay, and compute the maximum velocity
            which allows the ego-vehicle to brake enough to avoid the collision.

        :param front_vehicle: the preceding vehicle
        :return: the maximum allowed velocity, and suggested acceleration
        """
        if not front_vehicle:
            return self.target_velocity
        d0 = self.DISTANCE_WANTED
        a0 = self.COMFORT_ACC_MIN
        a1 = self.COMFORT_ACC_MIN
        tau = self.TIME_WANTED
        d = max(self.lane_distance_to(front_vehicle) - self.LENGTH / 2 - front_vehicle.LENGTH / 2 - d0, 0)
        v1_0 = front_vehicle.velocity
        delta = 4 * (a0 * a1 * tau) ** 2 + 8 * a0 * (a1 ** 2) * d + 4 * a0 * a1 * v1_0 ** 2
        v_max = -a0 * tau + np.sqrt(delta) / (2 * a1)

        # Velocity control
        self.target_velocity = min(self.maximum_velocity(front_vehicle), self.target_velocity)
        acceleration = self.velocity_control(self.target_velocity)

        return v_max, acceleration

    def change_lane_policy(self):
        """
            Decide when to change lane.

            Based on:
            - frequency;
            - closeness of the target lane;
            - MOBIL model.
        """
        # print("current_lane:",self.lane_index)
        # print("target_lane:", self.target_lane_index)
        # If a lane change already ongoing
        if self.lane_index != self.target_lane_index:
            # abort it if someone else is already changing into the same lane
            for v in self.road.vehicles:
                if v is not self \
                        and v.lane_index != self.target_lane_index \
                        and isinstance(v, ControlledVehicle) \
                        and v.target_lane_index == self.target_lane_index:
                    d = self.lane_distance_to(v)
                    d_star = self.desired_gap(self, v) * 0.5
                    if 0 < d < d_star:
                        self.target_lane_index = self.lane_index
                        break
            # todo: abort it if a possible collision may happen in the target line
            new_preceding, new_following = self.road.neighbour_vehicles(self, self.target_lane_index)
            if hasattr(new_preceding, "state") and new_preceding.state != "RED":
                new_preceding = None
            if new_preceding is not None:
                d_preceding = self.lane_distance_to(new_preceding)
                d_preceding_star = self.desired_gap(self, new_preceding) * 0.5
                if 0 < d_preceding < d_preceding_star:
                    # print("too close:",new_preceding)
                    self.target_lane_index = self.lane_index
            if new_following is not None:
                d_following = -self.lane_distance_to(new_following)
                d_following_star = self.desired_gap(new_following, self) * 0.5
                if 0 < d_following < d_following_star:
                    self.target_lane_index = self.lane_index
            return
        # else, at a given frequency,
        if not utils.do_every(self.LANE_CHANGE_DELAY, self.timer):
            return
        self.timer = 0

        # decide to make a lane change
        for lane_index in self.road.network.side_lanes_bigmap(self):
            # Is the candidate lane close enough?
            if not self.road.network.get_lane(lane_index).is_reachable_from(self.position):
                continue
            if abs(self.heading - self.lane.heading_at(self.lane.local_coordinates(self.position)[0])) > 0.05:
                continue
            # Does the MOBIL model recommend a lane change?
            if self.mobil(lane_index):
                self.target_lane_index = lane_index

    def mobil(self, lane_index):
        """
            MOBIL lane change model: Minimizing Overall Braking Induced by a Lane change

            The vehicle should change lane only if:
            - after changing it (and/or following vehicles) can accelerate more;
            - it doesn't impose an unsafe braking on its new following vehicle.

        :param lane_index: the candidate lane for the change
        :return: whether the lane change should be performed
        """
        # Is the maneuver unsafe for the new following vehicle?
        new_preceding, new_following = self.road.neighbour_vehicles(self, lane_index)
        if hasattr(new_preceding, "state") and new_preceding.state != "RED":
            new_preceding = None
        new_following_a = self.acceleration(ego_vehicle=new_following, front_vehicle=new_preceding)
        new_following_pred_a = self.acceleration(ego_vehicle=new_following, front_vehicle=self)
        if new_following_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False

        # Do I have a planned route for a specific lane which is safe for me to access?
        old_preceding, old_following = self.road.neighbour_vehicles(self)
        if hasattr(old_preceding, "state") and old_preceding.state != "RED":
            old_preceding = None
        self_pred_a = self.acceleration(ego_vehicle=self, front_vehicle=new_preceding)
        if self.route and self.route[0][2]:
            # Wrong direction
            if np.sign(lane_index[2] - self.target_lane_index[2]) != np.sign(
                    self.route[0][2] - self.target_lane_index[2]):
                return False
            # Unsafe braking required
            elif self_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
                return False

        # Is there an acceleration advantage for me and/or my followers to change lane?
        else:
            self_a = self.acceleration(ego_vehicle=self, front_vehicle=old_preceding)
            old_following_a = self.acceleration(ego_vehicle=old_following, front_vehicle=self)
            old_following_pred_a = self.acceleration(ego_vehicle=old_following, front_vehicle=old_preceding)
            jerk = self_pred_a - self_a + self.POLITENESS * (new_following_pred_a - new_following_a
                                                             + old_following_pred_a - old_following_a)
            if jerk < self.LANE_CHANGE_MIN_ACC_GAIN:
                return False

        # All clear, let's go!
        return True

    def recover_from_stop(self, acceleration):
        """
            If stopped on the wrong lane, try a reversing maneuver.

        :param acceleration: desired acceleration from IDM
        :return: suggested acceleration to recover from being stuck
        """
        stopped_velocity = 5
        safe_distance = 200
        # Is the vehicle stopped on the wrong lane?
        if self.target_lane_index != self.lane_index and self.velocity < stopped_velocity:
            _, rear = self.road.neighbour_vehicles(self)
            _, new_rear = self.road.neighbour_vehicles(self, self.road.lanes[self.target_lane_index])
            # Check for free room behind on both lanes
            if (not rear or rear.lane_distance_to(self) > safe_distance) and \
                    (not new_rear or new_rear.lane_distance_to(self) > safe_distance):
                # Reverse
                return -self.COMFORT_ACC_MAX / 2
        return acceleration


class LinearVehicle(IDMVehicle):
    """
        A Vehicle whose longitudinal and lateral controllers are linear with respect to parameters
    """
    ACCELERATION_PARAMETERS = [0.3, 0.14, 0.8]
    TIME_WANTED = 2.0

    def __init__(self, road, position,
                 heading=0,
                 velocity=0,
                 target_lane_index=None,
                 target_velocity=None,
                 enable_lane_change=True,
                 timer=None):
        super(LinearVehicle, self).__init__(road,
                                            position,
                                            heading,
                                            velocity,
                                            target_lane_index,
                                            target_velocity,
                                            enable_lane_change,
                                            timer)

    @classmethod
    def acceleration(cls, ego_vehicle, front_vehicle=None, rear_vehicle=None):
        """
            Compute an acceleration command with a Linear Model.

            The acceleration is chosen so as to:
            - reach a target velocity;
            - reach the velocity of the leading (resp following) vehicle, if it is lower (resp higher) than ego's;
            - maintain a minimum safety distance w.r.t the leading vehicle.

        :param ego_vehicle: the vehicle whose desired acceleration is to be computed. It does not have to be an
                            Linear vehicle, which is why this method is a class method. This allows a Linear vehicle to
                            reason about other vehicles behaviors even though they may not Linear.
        :param front_vehicle: the vehicle preceding the ego-vehicle
        :param rear_vehicle: the vehicle following the ego-vehicle
        :return: the acceleration command for the ego-vehicle [m/s2]
        """
        return np.dot(cls.ACCELERATION_PARAMETERS, cls.acceleration_features(ego_vehicle, front_vehicle, rear_vehicle))

    @classmethod
    def acceleration_features(cls, ego_vehicle, front_vehicle=None, rear_vehicle=None):
        vt, dv, dp = 0, 0, 0
        if ego_vehicle:
            vt = ego_vehicle.target_velocity - ego_vehicle.velocity
            d_safe = cls.DISTANCE_WANTED + np.max(ego_vehicle.velocity, 0) * cls.TIME_WANTED + ego_vehicle.LENGTH
            if front_vehicle:
                d = ego_vehicle.lane_distance_to(front_vehicle)
                dv = min(front_vehicle.velocity - ego_vehicle.velocity, 0)
                dp = min(d - d_safe, 0)
        return np.array([vt, dv, dp])

    def steering_control(self, target_lane_index):
        """
            Linear controller with respect to parameters.
            Overrides the non-linear controller ControlledVehicle.steering_control()
        :param target_lane_index: index of the lane to follow
        :return: a steering wheel angle command [rad]
        """
        steering_angle = np.dot(np.array(self.STEERING_GAIN), self.steering_features(target_lane_index))
        steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        return steering_angle

    def steering_features(self, target_lane_index):
        """
            A collection of features used to follow a lane
        :param target_lane_index: index of the lane to follow
        :return: a array of features
        """
        lane_coords = self.road.lanes[target_lane_index].local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.velocity * (self.TAU_DS + self.STEERING_TAU)
        lane_future_heading = self.road.lanes[target_lane_index].heading_at(lane_next_coords)
        features = np.array([-lane_coords[1] * self.LENGTH / (utils.not_zero(self.velocity) ** 2),
                             (lane_future_heading - self.heading) * self.LENGTH / utils.not_zero(self.velocity)])
        return features


class AggressiveVehicle(LinearVehicle):
    LANE_CHANGE_MIN_ACC_GAIN = 1.0  # [m/s2]
    MERGE_ACC_GAIN = 0.8
    MERGE_VEL_RATIO = 0.75
    MERGE_TARGET_VEL = 30
    ACCELERATION_PARAMETERS = [MERGE_ACC_GAIN / ((1 - MERGE_VEL_RATIO) * MERGE_TARGET_VEL),
                               MERGE_ACC_GAIN / (MERGE_VEL_RATIO * MERGE_TARGET_VEL),
                               0.5]


class DefensiveVehicle(LinearVehicle):
    LANE_CHANGE_MIN_ACC_GAIN = 1.0  # [m/s2]
    MERGE_ACC_GAIN = 1.2
    MERGE_VEL_RATIO = 0.75
    MERGE_TARGET_VEL = 30
    ACCELERATION_PARAMETERS = [MERGE_ACC_GAIN / ((1 - MERGE_VEL_RATIO) * MERGE_TARGET_VEL),
                               MERGE_ACC_GAIN / (MERGE_VEL_RATIO * MERGE_TARGET_VEL),
                               2.0]


class Control_IDMVehicle(IDMVehicle):

    def __init__(self, road, position, heading=0, velocity=0, target_lane_index=None, target_velocity=None,
                 enable_lane_change=False,
                 timer=None, dst=None, rever=False, max_length=None):
        super(Control_IDMVehicle, self).__init__(road, position, heading, velocity, target_lane_index, target_velocity,
                                                 enable_lane_change,
                                                 timer, dst, rever, max_length)
        self.last_position = position
        # if egocar make decisions that cannot be executed
        self.out_of_restrict = False
        self.lane_change = False

    def check_safe(self, target_lane_index):
        # Is the maneuver unsafe for the new following vehicle?
        new_preceding, new_following = self.road.neighbour_vehicles(self, target_lane_index)
        new_following_pred_a = IDMVehicle.acceleration(ego_vehicle=new_following, front_vehicle=self)
        self_pred_a = self.acceleration(ego_vehicle=self, front_vehicle=new_preceding)
        if new_following_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED or self_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False
        return True

    def act_control(self, action_op):
        action = {}
        self.out_of_restrict = False
        self.lane_change = False
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)

        if action_op == "LANE_RIGHT":
            target_lane_index = (self.lane_index[0], self.lane_index[1], self.lane_index[2] + 1)
            if 0 <= target_lane_index[2] < len(self.road.network.LANES) and self.check_safe(target_lane_index):
                self.target_lane_index = target_lane_index
                self.lane_change = True
            else:
                self.out_of_restrict = True
        elif action_op == "LANE_LEFT":
            target_lane_index = (self.lane_index[0], self.lane_index[1], self.lane_index[2] + 1)
            if 0 <= target_lane_index[2] < len(self.road.network.LANES) and self.check_safe(target_lane_index):
                self.target_lane_index = target_lane_index
                self.lane_change = True
            else:
                self.out_of_restrict = True
        action['steering'] = self.steering_control(self.target_lane_index)

        # Longitudinal: IDM
        self.acc = action['acceleration'] = self.acceleration(ego_vehicle=self,
                                                              front_vehicle=front_vehicle,
                                                              rear_vehicle=rear_vehicle)
        action['acceleration'] = np.clip(action['acceleration'], -self.ACC_MAX, self.ACC_MAX)

        super(ControlledVehicle, self).act(action)

    def act(self, action=None):
        action = {}
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)

        self.break_lane_change()

        action['steering'] = self.steering_control(self.target_lane_index)
        # Longitudinal: IDM
        self.acc = action['acceleration'] = self.acceleration(ego_vehicle=self,
                                                              front_vehicle=front_vehicle,
                                                              rear_vehicle=rear_vehicle)
        action['acceleration'] = np.clip(action['acceleration'], -self.ACC_MAX, self.ACC_MAX)

        super(ControlledVehicle, self).act(action)

    def break_lane_change(self):
        # If a lane change already ongoing
        if self.lane_index != self.target_lane_index:
            # abort it if someone else is already changing into the same lane
            for v in self.road.vehicles:
                if v is not self \
                        and v.lane_index != self.target_lane_index \
                        and isinstance(v, ControlledVehicle) \
                        and v.target_lane_index == self.target_lane_index:
                    d = self.lane_distance_to(v)
                    d_star = self.desired_gap(self, v)
                    if 0 < d < d_star:
                        self.target_lane_index = self.lane_index
                        break
            return


class MPCControlledVehicle(ControlledVehicle):
    """
        A vehicle using both a longitudinal and a lateral decision policies.

        - Longitudinal: the IDM model computes an acceleration given the preceding vehicle's distance and velocity.
        - Lateral: the MOBIL model decides when to change lane by maximizing the acceleration of nearby vehicles.
    IDM = {
        'ACC_MAX': 6.0, # [m/s2]
        'COMFORT_ACC_MAX': 3.0, # [m/s2]
        'COMFORT_ACC_MIN': -5.0, # [m/s2]
        'VELOCITY_WANTED': 25.0, # [m/s]
        'DISTANCE_WANTED':  5, # [m]
        'TIME_WANTED': 1.5, # [s]
        'DELTA': 4.0, # []
    }
    MOBIL = {
        'POLITENESS': 0., # in [0, 1]
        'LANE_CHANGE_MIN_ACC_GAIN': 0.4, # [m/s2]
        'RIGHT_LANE_CHANGE_MIN_ACC_GAIN': 0.4,
        'LANE_CHANGE_MAX_BRAKING_IMPOSED': 1.0, # [m/s2]
        'LANE_CHANGE_DELAY': 1.0, # [s]
    }
    """

    # Longitudinal policy parameters
    ACC_MAX = 8.0  # [m/s2]
    COMFORT_ACC_MAX = 5.0  # [m/s2]
    COMFORT_ACC_MIN = -5.0  # [m/s2]
    VELOCITY_WANTED = 25.0  # [m/s]
    DISTANCE_WANTED = 2  # [m]
    TIME_WANTED = 0.5  # [s]
    DELTA = 4.0  # []

    # Lateral policy parameters
    POLITENESS = 0.  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.1  # [m/s2]
    RIGHT_LANE_CHANGE_MIN_ACC_GAIN = LANE_CHANGE_MIN_ACC_GAIN
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 1.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]
    SIMULATION_FREQUENCY = 15

    def __init__(self, road, position,
                 heading=0,
                 velocity=0,
                 target_lane_index=None,
                 target_velocity=None,
                 enable_lane_change=True,
                 timer=None,
                 dst=None,
                 rever=False,
                 max_length=None,
                 route=None):
        super(MPCControlledVehicle, self).__init__(road, position, heading, velocity, target_lane_index, target_velocity, route)
        self.enable_lane_change = enable_lane_change
        self.acc = 0.0
        self.pre_acc = self.acc
        self.pre_lane = self.lane_index
        self.level = np.random.randint(-3, 4)
        self.DELTA = 0.0  # []
        self.rever = rever
        self.dst_index = dst
        self.mission_begin = True
        if dst is not None:
            if not rever:
                self.dst = deque(range(self.lane_index + 1, dst + 1))
            else:
                self.dst = deque(range(self.lane_index - 1, dst - 1, -1))
        else:
            self.dst = None
        IDMVehicle.generate_style(self)
        self.agent = 'idm'
        # self.lane.min_positon = min(self.lane.min_position,self.position[0])
        # self.position_reset = self.position.copy()
        # self.dst_reset = self.dst.copy()
        # self.lane_index_reset = self.lane_index

    def reset(self):
        cls = type(self)
        # lane_index = random.choice(range(len(self.road.lanes)))
        # lane = self.road.lanes[lane_index]
        lane = self.lane
        position = lane.position(np.random.randint(20, 40), 0)
        # if lane_index < 2:
        #     v = cls(self.road,position, velocity=np.random.randint(18,26),dst=self.dst,max_length=self.max_length,rever=self.rever)
        # else:
        #     v = cls(self.road,position, velocity=np.random.randint(18,26),dst=self.dst,max_length=self.max_length,rever=self.rever)
        v = cls(self.road, position, velocity=np.random.randint(18, 26), dst=self.dst_index, max_length=self.max_length,
                rever=self.rever)
        return v

    def randomize_behavior(self):
        pass

    @classmethod
    def create_from(cls, vehicle):
        """
            Create a new vehicle from an existing one.
            The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position, heading=vehicle.heading, velocity=vehicle.velocity,
                target_lane_index=vehicle.target_lane_index, target_velocity=vehicle.target_velocity,
                timer=getattr(vehicle, 'timer', None))
        return v

    def act(self, action=None,vehicle_actions = None):
        """
            Execute an action.

            For now, no action is supported because the vehicle takes all decisions
            of acceleration and lane changes on its own, based on the IDM and MOBIL models.

        :param vehicle_actions: test for CRUISE / LEFT_CHANGE / RIGHT_CHANGE / STOP / TAKEOVER
        :param action: the action
        """
        if self.crashed:
            return
        action = {}
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)

        # Lateral: MOBIL
        self.follow_road()
        # if self.id == 0:
        #     print("behavior/vehicle_actions",vehicle_actions," mission_begin:",self.mission_begin)
        if vehicle_actions == "RIGHT_CHANGE" and self.mission_begin:
            print("do right_change")
            _from, _to, _id = self.target_lane_index
            target_lane_index = _from, _to, np.clip(_id + 1, 0, len(self.road.network.graph[_from][_to]) - 1)
            if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                self.target_lane_index = target_lane_index
            self.mission_begin = False

        elif vehicle_actions == "LEFT_CHANGE" and self.mission_begin:
            print("do left_change")
            _from, _to, _id = self.target_lane_index
            target_lane_index = _from, _to, np.clip(_id - 1, 0, len(self.road.network.graph[_from][_to]) - 1)
            if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                self.target_lane_index = target_lane_index
            self.mission_begin = False
        # elif vehicle_actions == "STOP" and self.mission_begin:
        #     print("do stop")
        #     action["acceleration"] = -self.velocity
        else:
            pass

        # else:
        #     if self.enable_lane_change and not self.lane.forbidden:
        #         self.change_lane_policy()
        action['steering'] = self.delta_w
        action['acceleration'] = self.acc
        # Longitudinal: IDM
        self.pre_acc = 0.0 if self.acc is None else self.acc
        # if vehicle_actions != "STOP":
        #     self.acc = action['acceleration'] = self.acceleration(ego_vehicle=self,
        #                                                       front_vehicle=front_vehicle,
        #                                                       rear_vehicle=rear_vehicle)
        # action['acceleration'] = self.recover_from_stop(action['acceleration'])
        action['acceleration'] = np.clip(action['acceleration'], -self.ACC_MAX, self.ACC_MAX)
        super(ControlledVehicle, self).act(action)

    def step(self, dt):
        """
            Step the simulation.

            Increases a timer used for decision policies, and step the vehicle dynamics.

        :param dt: timestep
        """
        super(MPCControlledVehicle, self).step(dt)
        # print(self.id,self.dst,self.lane_index)
        if self.dst is not None and len(self.dst) > 1 and self.dst[0] == self.lane_index:
            self.dst.popleft()
    #
    # @classmethod
    # def acc_self_def(cls, ego_vehicle, front_vehicle=None, rear_vehicle=None):
    #     """
    #     self defined acc considering the vehicles of which the target lane index is the current lane
    #     """
    #
    #     if not ego_vehicle:
    #         return 0
    #     # decelerate at the crossroad
    #     target_velocity = ego_vehicle.target_velocity
    #     if hasattr(front_vehicle, "state") and front_vehicle.state != "RED":
    #
    #         # run with a given velocity at the crossroad
    #         if ego_vehicle.lane_distance_to(front_vehicle) < 40 and target_velocity > 30 / 3.6:
    #             target_velocity = 30 / 3.6
    #         front_vehicle = None
    #         # rear_vehicle = None
    #     acceleration = ego_vehicle.COMFORT_ACC_MAX * (
    #             1 - np.power(ego_vehicle.velocity / utils.not_zero(target_velocity), ego_vehicle.DELTA)) \
    #         if isinstance(ego_vehicle, IDMVehicle) else \
    #         cls.COMFORT_ACC_MAX * (
    #                 1 - np.power(ego_vehicle.velocity / utils.not_zero(target_velocity), cls.DELTA))
    #     if hasattr(front_vehicle, "state") and front_vehicle.state == "RED":
    #         if ego_vehicle.lane_distance_to(front_vehicle) < ego_vehicle.LENGTH / 2 + front_vehicle.LENGTH / 2:
    #             front_vehicle = None
    #
    #     if front_vehicle:
    #         d = ego_vehicle.lane_distance_to(front_vehicle)
    #         acceleration -= ego_vehicle.COMFORT_ACC_MAX * \
    #                         np.power(cls.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2) \
    #             if isinstance(ego_vehicle, IDMVehicle) else \
    #             cls.COMFORT_ACC_MAX * \
    #             np.power(cls.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2)
    #     # the acceleration will not exceed COMFORT_ACC_MAX
    #
    #     for lane_index in ego_vehicle.road.network.side_lanes_bigmap(ego_vehicle):
    #         # Is the candidate lane close enough?
    #         front_vehicle, rear_vehicle = ego_vehicle.road.neighbour_vehicles(ego_vehicle, lane_index)
    #         # if hasattr(front_vehicle, "state"):
    #         #     front_vehicle = None
    #         if hasattr(front_vehicle, "state") and front_vehicle.state != "RED":
    #
    #             # run with a given velocity at the crossroad
    #             if ego_vehicle.lane_distance_to(front_vehicle) < 40 and target_velocity > 30 / 3.6:
    #                 target_velocity = 30 / 3.6
    #             front_vehicle = None
    #         # if hasattr(front_vehicle,"state"):
    #         #     print("state:",front_vehicle.state)
    #         if front_vehicle is not None and not isinstance(front_vehicle,RedLight) and front_vehicle.target_lane_index != ego_vehicle.lane_index:
    #             continue
    #         target_velocity = ego_vehicle.target_velocity
    #
    #         # rear_vehicle = None
    #         acceleration_new = ego_vehicle.COMFORT_ACC_MAX * (
    #                 1 - np.power(ego_vehicle.velocity / utils.not_zero(target_velocity), ego_vehicle.DELTA)) \
    #             if isinstance(ego_vehicle, IDMVehicle) else \
    #             cls.COMFORT_ACC_MAX * (
    #                     1 - np.power(ego_vehicle.velocity / utils.not_zero(target_velocity), cls.DELTA))
    #         if hasattr(front_vehicle, "state") and front_vehicle.state == "RED":
    #             if ego_vehicle.lane_distance_to(front_vehicle) < ego_vehicle.LENGTH / 2 + front_vehicle.LENGTH / 2:
    #                 front_vehicle = None
    #
    #         if front_vehicle:
    #             d = ego_vehicle.lane_distance_to(front_vehicle)
    #             acceleration_new -= ego_vehicle.COMFORT_ACC_MAX * \
    #                                 np.power(cls.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2) \
    #                 if isinstance(ego_vehicle, IDMVehicle) else \
    #                 cls.COMFORT_ACC_MAX * \
    #                 np.power(cls.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2)
    #         if acceleration_new < acceleration:
    #             acceleration = acceleration_new
    #
    #     if ego_vehicle.acc < acceleration and ego_vehicle.acc < ego_vehicle.COMFORT_ACC_MAX:
    #         ego_vehicle.acc += ego_vehicle.COMFORT_ACC_MAX / ego_vehicle.SIMULATION_FREQUENCY
    #     elif ego_vehicle.acc > acceleration and ego_vehicle.acc > ego_vehicle.COMFORT_ACC_MIN:
    #         ego_vehicle.acc += ego_vehicle.COMFORT_ACC_MIN / ego_vehicle.SIMULATION_FREQUENCY
    #
    #     return acceleration

    # @classmethod
    # def acceleration(cls, ego_vehicle, front_vehicle=None, rear_vehicle=None):
    #     """
    #         Compute an acceleration command with the Intelligent Driver Model.
    #
    #         The acceleration is chosen so as to:
    #         - reach a target velocity;
    #         - maintain a minimum safety distance (and safety time) w.r.t the front vehicle.
    #
    #     :param ego_vehicle: the vehicle whose desired acceleration is to be computed. It does not have to be an
    #                         IDM vehicle, which is why this method is a class method. This allows an IDM vehicle to
    #                         reason about other vehicles behaviors even though they may not IDMs.
    #     :param front_vehicle: the vehicle preceding the ego-vehicle
    #     :param rear_vehicle: the vehicle following the ego-vehicle
    #     :return: the acceleration command for the ego-vehicle [m/s2]
    #     """
    #     if not ego_vehicle:
    #         return 0
    #     # decelerate at the crossroad
    #     target_velocity = ego_vehicle.target_velocity
    #     if hasattr(front_vehicle, "state") and front_vehicle.state == "GREEN":
    #
    #         # run with a given velocity at the crossroad
    #         if ego_vehicle.lane_distance_to(front_vehicle) < 35:
    #             target_velocity = 30 / 3.6
    #         front_vehicle = None
    #         # rear_vehicle = None
    #     acceleration = ego_vehicle.COMFORT_ACC_MAX * (
    #             1 - np.power(ego_vehicle.velocity / utils.not_zero(target_velocity), ego_vehicle.DELTA)) \
    #         if isinstance(ego_vehicle, IDMVehicle) else \
    #         cls.COMFORT_ACC_MAX * (
    #                 1 - np.power(ego_vehicle.velocity / utils.not_zero(target_velocity), cls.DELTA))
    #     if hasattr(front_vehicle, "state") and front_vehicle.state == "RED":
    #         if ego_vehicle.lane_distance_to(front_vehicle) < ego_vehicle.LENGTH / 2 + front_vehicle.LENGTH / 2:
    #             front_vehicle = None
    #
    #     if isinstance(front_vehicle,Obstacle):
    #         front_vehicle = None
    #     if front_vehicle:
    #         d = ego_vehicle.lane_distance_to(front_vehicle)
    #         acceleration -= ego_vehicle.COMFORT_ACC_MAX * \
    #                         np.power(cls.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2) \
    #             if isinstance(ego_vehicle, IDMVehicle) else \
    #             cls.COMFORT_ACC_MAX * \
    #             np.power(cls.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2)
    #     return acceleration

    @classmethod
    def desired_gap(cls, ego_vehicle, front_vehicle=None):
        """
            Compute the desired distance between a vehicle and its leading vehicle.

        :param ego_vehicle: the vehicle being controlled
        :param front_vehicle: its leading vehicle
        :return: the desired distance between the two [m]
        """
        if isinstance(ego_vehicle, IDMVehicle):
            d0 = ego_vehicle.DISTANCE_WANTED + ego_vehicle.LENGTH / 2 + front_vehicle.LENGTH / 2
            tau = ego_vehicle.TIME_WANTED
            ab = -ego_vehicle.COMFORT_ACC_MAX * ego_vehicle.COMFORT_ACC_MIN
            dv = ego_vehicle.velocity - front_vehicle.velocity
            d_star = d0 + ego_vehicle.velocity * tau + ego_vehicle.velocity * dv / (2 * np.sqrt(ab))
        else:
            d0 = cls.DISTANCE_WANTED + ego_vehicle.LENGTH / 2 + front_vehicle.LENGTH / 2
            tau = cls.TIME_WANTED
            ab = -cls.COMFORT_ACC_MAX * cls.COMFORT_ACC_MIN
            dv = ego_vehicle.velocity - front_vehicle.velocity
            d_star = d0 + ego_vehicle.velocity * tau + ego_vehicle.velocity * dv / (2 * np.sqrt(ab))
        return d_star

    def maximum_velocity(self, front_vehicle=None):
        """
            Compute the maximum allowed velocity to avoid Inevitable Collision States.

            Assume the front vehicle is going to brake at full deceleration and that
            it will be noticed after a given delay, and compute the maximum velocity
            which allows the ego-vehicle to brake enough to avoid the collision.

        :param front_vehicle: the preceding vehicle
        :return: the maximum allowed velocity, and suggested acceleration
        """
        if not front_vehicle:
            return self.target_velocity
        d0 = self.DISTANCE_WANTED
        a0 = self.COMFORT_ACC_MIN
        a1 = self.COMFORT_ACC_MIN
        tau = self.TIME_WANTED
        d = max(self.lane_distance_to(front_vehicle) - self.LENGTH / 2 - front_vehicle.LENGTH / 2 - d0, 0)
        v1_0 = front_vehicle.velocity
        delta = 4 * (a0 * a1 * tau) ** 2 + 8 * a0 * (a1 ** 2) * d + 4 * a0 * a1 * v1_0 ** 2
        v_max = -a0 * tau + np.sqrt(delta) / (2 * a1)

        # Velocity control
        self.target_velocity = min(self.maximum_velocity(front_vehicle), self.target_velocity)
        acceleration = self.velocity_control(self.target_velocity)

        return v_max, acceleration

    def change_lane_policy(self):
        """
            Decide when to change lane.

            Based on:
            - frequency;
            - closeness of the target lane;
            - MOBIL model.
        """
        # print("current_lane:",self.lane_index)
        # print("target_lane:", self.target_lane_index)
        # If a lane change already ongoing
        if self.lane_index != self.target_lane_index:
            # abort it if someone else is already changing into the same lane
            for v in self.road.vehicles:
                if v is not self \
                        and v.lane_index != self.target_lane_index \
                        and isinstance(v, ControlledVehicle) \
                        and v.target_lane_index == self.target_lane_index:
                    d = self.lane_distance_to(v)
                    d_star = self.desired_gap(self, v) * 0.5
                    if 0 < d < d_star:
                        self.target_lane_index = self.lane_index
                        break
            # todo: abort it if a possible collision may happen in the target line
            new_preceding, new_following = self.road.neighbour_vehicles(self, self.target_lane_index)
            if hasattr(new_preceding, "state") and new_preceding.state != "RED":
                new_preceding = None
            if new_preceding is not None:
                d_preceding = self.lane_distance_to(new_preceding)
                d_preceding_star = self.desired_gap(self, new_preceding) * 0.5
                if 0 < d_preceding < d_preceding_star:
                    # print("too close:",new_preceding)
                    self.target_lane_index = self.lane_index
            if new_following is not None:
                d_following = -self.lane_distance_to(new_following)
                d_following_star = self.desired_gap(new_following, self) * 0.5
                if 0 < d_following < d_following_star:
                    self.target_lane_index = self.lane_index
            return
        # else, at a given frequency,
        if not utils.do_every(self.LANE_CHANGE_DELAY, self.timer):
            return
        self.timer = 0

        # decide to make a lane change
        for lane_index in self.road.network.side_lanes_bigmap(self):
            # Is the candidate lane close enough?
            if not self.road.network.get_lane(lane_index).is_reachable_from(self.position):
                continue
            if abs(self.heading - self.lane.heading_at(self.lane.local_coordinates(self.position)[0])) > 0.05:
                continue
            # Does the MOBIL model recommend a lane change?
            if self.mobil(lane_index):
                self.target_lane_index = lane_index

    def mobil(self, lane_index):
        """
            MOBIL lane change model: Minimizing Overall Braking Induced by a Lane change

            The vehicle should change lane only if:
            - after changing it (and/or following vehicles) can accelerate more;
            - it doesn't impose an unsafe braking on its new following vehicle.

        :param lane_index: the candidate lane for the change
        :return: whether the lane change should be performed
        """
        # Is the maneuver unsafe for the new following vehicle?
        new_preceding, new_following = self.road.neighbour_vehicles(self, lane_index)
        if hasattr(new_preceding, "state") and new_preceding.state != "RED":
            new_preceding = None
        new_following_a = self.acceleration(ego_vehicle=new_following, front_vehicle=new_preceding)
        new_following_pred_a = self.acceleration(ego_vehicle=new_following, front_vehicle=self)
        if new_following_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False

        # Do I have a planned route for a specific lane which is safe for me to access?
        old_preceding, old_following = self.road.neighbour_vehicles(self)
        if hasattr(old_preceding, "state") and old_preceding.state != "RED":
            old_preceding = None
        self_pred_a = self.acceleration(ego_vehicle=self, front_vehicle=new_preceding)
        if self.route and self.route[0][2]:
            # Wrong direction
            if np.sign(lane_index[2] - self.target_lane_index[2]) != np.sign(
                    self.route[0][2] - self.target_lane_index[2]):
                return False
            # Unsafe braking required
            elif self_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
                return False

        # Is there an acceleration advantage for me and/or my followers to change lane?
        else:
            self_a = self.acceleration(ego_vehicle=self, front_vehicle=old_preceding)
            old_following_a = self.acceleration(ego_vehicle=old_following, front_vehicle=self)
            old_following_pred_a = self.acceleration(ego_vehicle=old_following, front_vehicle=old_preceding)
            jerk = self_pred_a - self_a + self.POLITENESS * (new_following_pred_a - new_following_a
                                                             + old_following_pred_a - old_following_a)
            if jerk < self.LANE_CHANGE_MIN_ACC_GAIN:
                return False

        # All clear, let's go!
        return True

    def recover_from_stop(self, acceleration):
        """
            If stopped on the wrong lane, try a reversing maneuver.

        :param acceleration: desired acceleration from IDM
        :return: suggested acceleration to recover from being stuck
        """
        stopped_velocity = 5
        safe_distance = 200
        # Is the vehicle stopped on the wrong lane?
        if self.target_lane_index != self.lane_index and self.velocity < stopped_velocity:
            _, rear = self.road.neighbour_vehicles(self)
            _, new_rear = self.road.neighbour_vehicles(self, self.road.lanes[self.target_lane_index])
            # Check for free room behind on both lanes
            if (not rear or rear.lane_distance_to(self) > safe_distance) and \
                    (not new_rear or new_rear.lane_distance_to(self) > safe_distance):
                # Reverse
                return -self.COMFORT_ACC_MAX / 2
        return acceleration
