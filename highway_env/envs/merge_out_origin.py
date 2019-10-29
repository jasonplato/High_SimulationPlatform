from __future__ import division, print_function, absolute_import
import numpy as np

from highway_env import utils
from highway_env.envs.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane, LanesConcatenation
from highway_env.road.road import Road
from highway_env.vehicle.control import ControlledVehicle, MDPVehicle, CarSim, FreeControl
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.dynamics import Obstacle
import time
import random


def mobil(self, lane_index, mandatory):

    """
        action_explain = ['left acc', 'left same', 'left dec', 'same acc', 'same same', 'same dec', 'right acc',
                      'right same', 'right dec']
        MOBIL lane change model: Minimizing Overall Braking Induced by a Lane change

        The vehicle should change lane only if:
        - after changing it (and/or following vehicles) can accelerate more;
        - it doesn't impose an unsafe braking on its new following vehicle.

    :param lane_index: the candidate lane for the change
    :param mandatory: if the lane change is mandatory
    :return: whether the lane change should be performed
    """

    def acceleration(ego_vehicle, front_vehicle=None):
        """
            Compute an acceleration command with the Intelligent Driver Model.

            The acceleration is chosen so as to:
            - reach a target velocity;
            - maintain a minimum safety distance (and safety time) w.r.t the front vehicle.

        :param ego_vehicle: the vehicle whose desired acceleration is to be computed. It does not have to be an
                            IDM vehicle, which is why this method is a class method. This allows an IDM vehicle to
                            reason about other vehicles behaviors even though they may not IDMs.
        :param front_vehicle: the vehicle preceding the ego-vehicle
        :return: the acceleration command for the ego-vehicle [m/s2]
        """
        COMFORT_ACC_MAX = 3.0
        COMFORT_ACC_MIN = -5.0
        TIME_WANTED = 1.5
        DISTANCE_WANTED = 10
        DELTA = 4.0

        def not_zero(x):
            EPSILON = 0.01
            if abs(x) > EPSILON:
                return x
            elif x > 0:
                return EPSILON
            else:
                return -EPSILON

        def desired_gap(ego_vehicle, front_vehicle):
            d0 = DISTANCE_WANTED + ego_vehicle.LENGTH / 2 + front_vehicle.LENGTH / 2
            tau = TIME_WANTED
            ab = -COMFORT_ACC_MAX * COMFORT_ACC_MIN
            dv = ego_vehicle.velocity - front_vehicle.velocity
            d_star = d0 + ego_vehicle.velocity * tau + ego_vehicle.velocity * dv / (2 * np.sqrt(ab))
            return d_star

        if not ego_vehicle:
            return 0

        acceleration = COMFORT_ACC_MAX * (
                1 - np.power(ego_vehicle.velocity / not_zero(ego_vehicle.target_velocity), DELTA))
        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle)
            acceleration -= COMFORT_ACC_MAX * np.power(
                desired_gap(ego_vehicle, front_vehicle) / not_zero(d), 2)
        return acceleration

    LANE_CHANGE_MAX_BRAKING_IMPOSED = 1.0
    LANE_CHANGE_MIN_ACC_GAIN = 0.1
    POLITENESS = 0.

    # Is the maneuver unsafe for the new following vehicle?
    new_preceding, new_following = self.road.neighbour_vehicles(self, self.road.lanes[lane_index])

    # todo: added mandatory part
    preceding_vehicle_ok = True
    if new_preceding:
        relative_x = new_preceding.position[0] - self.position[0]
        relative_v = self.velocity - new_preceding.velocity
        if relative_x < 5:
            preceding_vehicle_ok = False
        if relative_v == 0.0:
            pass
        else:
            t = relative_x / relative_v
            if 0 < t < 3:
                preceding_vehicle_ok = False
    following_vehicle_ok = True
    if new_following:
        relative_x = self.position[0] - new_following.position[0]
        relative_v = new_following.velocity - self.velocity
        if relative_x < 5:
            following_vehicle_ok = False
        if relative_v == 0.0:
            pass
        else:
            t = relative_x / relative_v
            if 0 < t < 3:
                following_vehicle_ok = False
    if mandatory:
        if preceding_vehicle_ok and following_vehicle_ok:
            return True
        else:
            return False
    # todo: part finish

    new_following_a = acceleration(ego_vehicle=new_following, front_vehicle=new_preceding)
    new_following_pred_a = acceleration(ego_vehicle=new_following, front_vehicle=self)
    if new_following_pred_a < -LANE_CHANGE_MAX_BRAKING_IMPOSED:
        return False

    # Is there an advantage for me and/or my followers to change lane?
    old_preceding, old_following = self.road.neighbour_vehicles(self)
    self_a = acceleration(ego_vehicle=self, front_vehicle=old_preceding)
    self_pred_a = acceleration(ego_vehicle=self, front_vehicle=new_preceding)
    old_following_a = acceleration(ego_vehicle=old_following, front_vehicle=self)
    old_following_pred_a = acceleration(ego_vehicle=old_following, front_vehicle=old_preceding)
    jerk = self_pred_a - self_a + POLITENESS * (
            new_following_pred_a - new_following_a + old_following_pred_a - old_following_a)
    if jerk < LANE_CHANGE_MIN_ACC_GAIN:
        return False

    # All clear, let's go!
    return True


def global_mobil(env, action):
    """
    :param env: environment
    :param action: action_explain = ['left acc', 'left same', 'left dec', 'same acc', 'same same', 'same dec',
    'right acc', 'right same', 'right dec']
    """
    vehicle = env.vehicle
    mandatory = False
    lane_index = vehicle.lane_index
    if action in [0, 1, 2]:
        lane_index -= 1
        mandatory = True
        if lane_index >= 0 and env.road.lanes[lane_index].is_reachable_from(vehicle.position):
            print('mandatory to left: {}'.format(mobil(vehicle, lane_index, mandatory)))
    elif action in [6, 7, 8]:
        lane_index += 1
        mandatory = True
        if lane_index < len(env.road.lanes) and env.road.lanes[lane_index].is_reachable_from(vehicle.position):
            print('mandatory to right: {}'.format(mobil(vehicle, lane_index, mandatory)))
    else:
        lane_offsets = [i for i in [-1, 1] if 0 <= vehicle.lane_index + i < len(env.road.lanes)]
        for lane_offset in lane_offsets:
            # Is the candidate lane close enough?
            if not env.road.lanes[vehicle.lane_index + lane_offset].is_reachable_from(vehicle.position):
                continue
            # Does the MOBIL model recommend a lane change?
            if mobil(vehicle, lane_index, mandatory):
                print("unmandatory to {}, True!".format(lane_offset))
            else:
                print("unmandatory to {}, False!".format(lane_offset))

# todo
# --------------------------------------------
# todo


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

    DEFAULT_CONFIG = {"other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle"}

    def __init__(self):
        super(MergeEnvOut, self).__init__()
        self.switch = False
        self.other_vehicles_mandatory = False
        self.config = self.DEFAULT_CONFIG.copy()
        # self.make_road()
        self.make()
        # self.double_merge()
        self.make_vehicles(self.other_vehicles_mandatory)
        self.success_cnt = 0

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
            if vehicle.lane_index == len(self.road.lanes)-1 and isinstance(vehicle, ControlledVehicle):
                reward += self.MERGING_VELOCITY_REWARD * \
                          (vehicle.target_velocity - vehicle.velocity) / vehicle.target_velocity
        return reward + action_reward[action]

    def ego_vehicle_switch(self):
        self.switch = not self.switch

    def _is_terminal(self):
        """
            The episode is over when a collision occurs or when the access ramp has been passed.
        """
        if self.vehicle.position[0] > 500:
            if self.vehicle.lane_index == 3:
                self.success_cnt += 0.5
        return self.vehicle.crashed or self.vehicle.position[0] > 500

    def reset(self):
        # self.make_road()
        self.make()
        self.make_vehicles(self.other_vehicles_mandatory)
        return self._observation()

    def make_straight(self):
        lm10 = StraightLane(np.array([0, 0]), 0, 4.0, [LineType.CONTINUOUS_LINE, LineType.STRIPED], bounds=[0, 500])
        l1 = LanesConcatenation([lm10])
        lm20 = StraightLane(l1.position(0, 4), 0, 4.0, [LineType.STRIPED, LineType.STRIPED], bounds=[0, 500])
        l2 = LanesConcatenation([lm20])
        # lm30 = StraightLane(l2.position(0,4), 0, 4.0, [LineType.STRIPED, LineType.STRIPED],bounds=[0,100])
        # lm31 = StraightLane(lm30.position(0,0), 0, 4.0, [LineType.STRIPED, LineType.STRIPED],bounds=[0,500])
        # l3 = LanesConcatenation([lm30,lm31])
        lm30 = StraightLane(l2.position(0, 4), 0, 4.0, [LineType.STRIPED, LineType.STRIPED], bounds=[0, 500])
        l3 = LanesConcatenation([lm30])
        amplitude = 4.5
        lm40 = StraightLane(l3.position(0, 4), 0, 4.0, [LineType.STRIPED, LineType.CONTINUOUS_LINE], bounds=[200, 400])
        lm41 = SineLane(lm40.position(400, amplitude), 0, 4.0, -amplitude, 2 * np.pi / (2 * 50), np.pi / 2,
                        [LineType.CONTINUOUS, LineType.CONTINUOUS], bounds=[0, 50], forbidden=True)
        lm42 = StraightLane(lm41.position(50, 0), 0, 4.0, [LineType.CONTINUOUS_LINE, LineType.CONTINUOUS_LINE],
                            bounds=[0, 50],
                            forbidden=True)
        l4 = LanesConcatenation([lm40, lm41, lm42])
        road = Road([l1, l2, l3, l4])
        # road = Road([ l3])

        # road = Road([lm0,lm2])
        # todo !!!!!!!!!!! how to do with Obstacle in lane.vehicles
        obstacle = Obstacle(road, lm40.position(0, 0))
        road.vehicles.append(obstacle)
        road.lanes[3].vehicles.append(obstacle)
        self.road = road

    def make_sin(self):
        # amplitude = 4.5
        amplitude = 9.0

        lm10 = StraightLane(np.array([0, 0]), 0, 5.0, [LineType.CONTINUOUS_LINE, LineType.STRIPED], bounds=[0, 400])
        lm11 = SineLane(lm10.position(400, amplitude), 0, 5.0, -amplitude, 2 * np.pi / (2 * 50), np.pi / 2,
                        [LineType.CONTINUOUS, LineType.STRIPED], bounds=[0, 250])
        lm12 = StraightLane(lm11.position(250, 0), 0, 5.0, [LineType.CONTINUOUS_LINE, LineType.STRIPED], bounds=[0, 50])
        l1 = LanesConcatenation([lm10, lm11, lm12])

        lm20 = StraightLane(lm10.position(0, 5), 0, 5.0, [LineType.STRIPED, LineType.STRIPED], bounds=[0, 400])
        lm21 = SineLane(lm20.position(400, amplitude), 0, 5.0, -amplitude, 2 * np.pi / (2 * 50), np.pi / 2,
                        [LineType.STRIPED, LineType.STRIPED], bounds=[0, 250])
        lm22 = StraightLane(lm21.position(250, 0), 0, 5.0, [LineType.STRIPED, LineType.STRIPED], bounds=[0, 50])
        l2 = LanesConcatenation([lm20, lm21, lm22])

        lm30 = StraightLane(lm20.position(0, 5), 0, 5.0, [LineType.STRIPED, LineType.STRIPED], bounds=[0, 400])
        lm31 = SineLane(lm30.position(400, amplitude), 0, 5.0, -amplitude, 2 * np.pi / (2 * 50), np.pi / 2,
                        [LineType.STRIPED, LineType.STRIPED], bounds=[0, 250])
        lm32 = StraightLane(lm31.position(250, 0), 0, 5.0, [LineType.STRIPED, LineType.STRIPED], bounds=[0, 50])
        l3 = LanesConcatenation([lm30, lm31, lm32])

        lm40 = StraightLane(lm30.position(0, 5), 0, 5.0, [LineType.STRIPED, LineType.CONTINUOUS_LINE], bounds=[0, 400])
        lm41 = SineLane(lm40.position(400, amplitude), 0, 5.0, -amplitude, 2 * np.pi / (2 * 50), np.pi / 2,
                        [LineType.STRIPED, LineType.CONTINUOUS], bounds=[0, 250])
        lm42 = StraightLane(lm41.position(250, 0), 0, 5.0, [LineType.STRIPED, LineType.CONTINUOUS_LINE],
                            bounds=[0, 50],)
        l4 = LanesConcatenation([lm40, lm41, lm42])
        road = Road([l1, l2, l3, l4])
        # road = Road([ l3])

        # road = Road([lm0,lm2])
        # todo !!!!!!!!!!! how to do with Obstacle in lane.vehicles
        obstacle = Obstacle(road, lm40.position(0, 0))
        road.vehicles.append(obstacle)
        road.lanes[3].vehicles.append(obstacle)
        self.road = road

    def make(self):
        self.make_straight()
        # self.make_sin()

    def make_vehicles(self, other_vehicles_mandatory=False):
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :param other_vehicles_mandatory: if the lane changing maneuvers of other vehicles are mandatory
        :return: None
        """
        max_l = 500
        road = self.road
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        car_number_each_lane = 15
        # reset_position_range = (30, 40)
        reset_position_range = (20, 30)
        # reset_lane = random.choice(road.lanes)
        reset_lane = road.lanes[0]
        for l in road.lanes[:3]:
            cars_on_lane = car_number_each_lane
            reset_position = None
            if l is reset_lane:
                cars_on_lane += 1
                reset_position = random.choice(range(5, 6))
                # reset_position = 2
            for i in range(cars_on_lane):
                if i == reset_position:
                    if self.switch:
                        ego_vehicle = MDPVehicle(road, l.position((i+1) * np.random.randint(*reset_position_range), 0),
                                                 velocity=20, max_length=max_l)
                    else:
                        ego_vehicle = IDMVehicle(road, l.position((i + 1) * np.random.randint(*reset_position_range), 0),
                                                 velocity=20, max_length=max_l)
                        ego_vehicle.destination = 1
                        ego_vehicle.id = 0
                    road.vehicles.append(ego_vehicle)
                    self.vehicle = ego_vehicle
                    l.vehicles.append(ego_vehicle)
                else:
                    car = other_vehicles_type(road, l.position((i+1) * np.random.randint(*reset_position_range), 0),
                                              velocity=np.random.randint(18, 25), dst=3, max_length=max_l)
                    if other_vehicles_mandatory:
                        car.destination = 1
                    road.vehicles.append(car)
                    l.vehicles.append(car)

        for l in [road.lanes[3]]:
            cars_on_lane = car_number_each_lane
            reset_position = None
            if l is reset_lane:
                cars_on_lane += 1
                reset_position = random.choice(range(5, 6))
                # reset_position = 2
            for i in range(cars_on_lane):
                if i < 8:
                    continue
                if i == reset_position:
                    # ego_vehicle = MDPVehicle(road, l.position((i+1) * np.random.randint(*reset_position_range), 0),
                    #                          velocity=20, max_length=max_l)
                    ego_vehicle = IDMVehicle(road, l.position((i + 1) * np.random.randint(*reset_position_range), 0),
                                             velocity=20, max_length=max_l)
                    ego_vehicle.destination = 1
                    ego_vehicle.id = 0
                    road.vehicles.append(ego_vehicle)
                    self.vehicle = ego_vehicle
                    l.vehicles.append(ego_vehicle)
                else:
                    car = other_vehicles_type(road, l.position((i+1) * np.random.randint(*reset_position_range), 0),
                                              velocity=np.random.randint(18, 25), dst=3, max_length=max_l)
                    if other_vehicles_mandatory:
                        car.destination = 1
                    road.vehicles.append(car)
                    l.vehicles.append(car)

        for lane in road.lanes:
            lane.vehicles = sorted(lane.vehicles, key=lambda x: lane.local_coordinates(x.position)[0])
            for i, v in enumerate(lane.vehicles):
                v.vehicle_index_in_line = i

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


if __name__ == '__main__':
    pass
