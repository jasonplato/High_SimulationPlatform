from __future__ import division, print_function

import pygame

import numpy as np
import copy
from highway_env import utils
from highway_env.vehicle.dynamics import Vehicle
#from client import m

class ControlledVehicle(Vehicle):
    """
        A vehicle piloted by two low-level controller, allowing high-level actions
        such as cruise control and lane changes.

        - The longitudinal controller is a velocity controller;
        - The lateral controller is a heading controller cascaded with a lateral position controller.
    """

    TAU_A = 0.5  # [s]
    TAU_DS = 0.2  # [s]
    PURSUIT_TAU = 1.5 * TAU_DS
    KP_A = 1 / TAU_A
    KP_HEADING = 1 / TAU_DS
    KP_LATERAL = 1 / 1.0  # [1/s]
    STEERING_GAIN = [KP_HEADING * KP_LATERAL, KP_HEADING]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]
    DELTA_VELOCITY = 5  # [m/s]
    myimage = pygame.image.load("../red_alpha_resize.png")

    def __init__(self,
                 road,
                 position,
                 heading=0,
                 velocity=0,
                 target_lane_index=None,
                 target_velocity=None,
                 max_length = None,
                 route = None):
        super(ControlledVehicle, self).__init__(road, position, heading, velocity, max_length)
        self.target_lane_index = target_lane_index or self.lane_index
        self.target_velocity = target_velocity or self.velocity
        self.agent = 'agent'
        self.out = False
        self.route = route

    @classmethod
    def create_from(cls, vehicle):
        """
            Create a new vehicle from an existing one.
            The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position, heading=vehicle.heading, velocity=vehicle.velocity,
                target_lane_index=vehicle.target_lane_index, target_velocity=vehicle.target_velocity)
        return v

    def plan_route_to(self, destination):
        """
            Plan a route to a destination in the road network

        :param destination: a node in the road network
        """
        # print("enter plan_route_to")
        path = self.road.network.shortest_path(self.lane_index[1], destination)
        if path:
            # for i in range(len(path)-1)
            self.route = [self.lane_index] + [(path[i], path[i + 1], None) for i in range(len(path) - 1)]
        else:
            self.route = [self.lane_index]
        return self

    def act(self,action=None):
        # return
        """
            Perform a high-level action to change the desired lane or velocity.

            - If a high-level action is provided, update the target velocity and lane;
            - then, perform longitudinal and lateral control.

        :param action: a high-level action
        """
        self.follow_road()
        if action == "FASTER":
            self.target_velocity += self.DELTA_VELOCITY
        elif action == "SLOWER":
            self.target_velocity -= self.DELTA_VELOCITY

        elif action == "LANE_RIGHT":
            _from, _to, _id = self.target_lane_index
            target_lane_index = _from, _to, np.clip(_id + 1, 0, len(self.road.network.graph[_from][_to]) - 1)
            if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                self.target_lane_index = target_lane_index
        elif action == "LANE_LEFT":
            _from, _to, _id = self.target_lane_index
            target_lane_index = _from, _to, np.clip(_id - 1, 0, len(self.road.network.graph[_from][_to]) - 1)
            if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                self.target_lane_index = target_lane_index
        """
        elif action == "LANE_RIGHT":
            points = self.road.lanes[self.lane_index].cut_points
            if len(points) == 0:
                target_lane_index = np.clip(self.lane_index + 1, 0, len(self.road.lanes) - 1)
                #print(self.road.lanes[self.lane_index].on_lane(self.position))
                if self.road.lanes[target_lane_index].is_reachable_from(self.position):
                    self.target_lane_index = target_lane_index
            else:
                for point in points:
                    if point.is_reachable_from(self.position):
                        self.target_lane_index = point.index
            if self.lane_index == len(self.road.lanes) - 1:
                self.out = True
        elif action == "LANE_LEFT":
            if self.lane_index == 0:
                self.out = True
            target_lane_index = np.clip(self.lane_index - 1, 0, len(self.road.lanes) - 1)
            if self.road.lanes[target_lane_index].is_reachable_from(self.position):
                self.target_lane_index = target_lane_index
        """
        action = {'steering': self.steering_control(self.target_lane_index),
                  'acceleration': self.velocity_control(self.target_velocity)}
        # print(action)
        super(ControlledVehicle, self).act(action)

    def follow_road(self):
        """
           At the end of a lane, automatically switch to a next one.
        """
        if self.road.network.get_lane(self.target_lane_index).after_end(self.position):
            self.target_lane_index = self.road.network.next_lane(self.target_lane_index,
                                                                 route=self.route,
                                                                 position=self.position,
                                                                 np_random=self.road.np_random)

    def steering_control(self, target_lane_index,dt= None):
        """
            Steer the vehicle to follow the center of an given lane.

        1. Lateral position is controlled by a proportional controller yielding a lateral velocity command
        2. Lateral velocity command is converted to a heading reference
        3. Heading is controlled by a proportional controller yielding a heading rate command
        4. Heading rate command is converted to a steering angle

        :param target_lane_index: index of the lane to follow
        :return: a steering wheel angle command [rad]
        """
        target_lane = self.road.network.get_lane(target_lane_index)
        lane_coords = target_lane.local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.velocity * 1
        lane_future_heading = target_lane.heading_at(lane_next_coords)

        # Lateral position control
        lateral_velocity_command = - self.KP_LATERAL * lane_coords[1]
        # Lateral velocity to heading
        heading_command = np.arcsin(np.clip(lateral_velocity_command / utils.not_zero(self.velocity), -1, 1))
        heading_ref = lane_future_heading + np.clip(heading_command, -np.pi / 4, np.pi / 4)
        # Heading control
        heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(heading_ref - self.heading)
        # Heading rate to steering angle
        #steering_angle = self.LENGTH / utils.not_zero(self.velocity) * heading_rate_command
        steering_angle = self.LENGTH / utils.not_zero(self.velocity) * np.arctan(heading_rate_command)
        steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        return steering_angle

    def velocity_control(self, target_velocity):
        """
            Control the velocity of the vehicle.

            Using a simple proportional controller.

        :param target_velocity: the desired velocity
        :return: an acceleration command [m/s2]
        """
        return self.KP_A * (target_velocity - self.velocity)

class FreeControl(ControlledVehicle):

    def __init__(self,*args,**kwargs):
        super(FreeControl,self).__init__(*args,**kwargs)

    def act(self, action=None):
        #return
        """
            Perform a high-level action to change the desired lane or velocity.

            - If a high-level action is provided, update the target velocity and lane;
            - then, perform longitudinal and lateral control.

        :param action: a high-level action
        """
        super(ControlledVehicle, self).act(action)

    def step(self, dt):
        """
            Propagate the vehicle state given its actions.

            Integrate a modified bicycle model with a 1st-order response on the steering wheel dynamics.
            If the vehicle is crashed, the actions are overridden with erratic steering and braking until complete stop.
            The vehicle's current lane is updated.

        :param dt: timestep of integration of the model [s]
        """
        if self.crashed:
            self.action['steering'] = 0
            self.action['acceleration'] = -self.velocity

        v = self.velocity * np.array([np.cos(self.heading), np.sin(self.heading)])
        self.position += v * dt
        self.heading += self.velocity * self.steering_angle / self.LENGTH * dt
        self.steering_angle += 1 / self.STEERING_TAU * (np.tan(self.action['steering']) - self.steering_angle) * dt
        #self.steering_angle = self.action['steering']
        self.velocity += self.action['acceleration'] * dt

        if self.road:
            self.lane_index = self.road.get_lane_index(self.position)
            self.lane = self.road.lanes[self.lane_index]
class CarSim(ControlledVehicle):

    def __init__(self,*args,**kwargs):
        super(CarSim,self).__init__(*args,**kwargs)

    def act(self, action=None):
        if not action:
            action = {
                "throttle": 0,
                "brake": 0.,
                "steering": 0
            }
        action = m.send_control(action)
        super(ControlledVehicle, self).act(action)

    def step(self, dt):
        """
            Propagate the vehicle state given its actions.

            Integrate a modified bicycle model with a 1st-order response on the steering wheel dynamics.
            If the vehicle is crashed, the actions are overridden with erratic steering and braking until complete stop.
            The vehicle's current lane is updated.

        :param dt: timestep of integration of the model [s]
        """
        position = self.action['location']
        rotation = self.action['rotation']
        self.position[0] = position[0]
        self.position[1] = position[1] + 15.8
        # self.position[0] = 40
        # self.position[1] = 2
        self.heading = rotation[2] / 57.3
        self.velocity = self.action['global_speed'][0]
        # if self.crashed:
        #     self.action['steering'] = 0
        #     self.action['acceleration'] = -self.velocity
        #
        # v = self.velocity * np.array([np.cos(self.heading), np.sin(self.heading)])
        # self.position += v * dt
        #self.heading += self.velocity * self.steering_angle / self.LENGTH * dt
        #self.steering_angle += 1 / self.STEERING_TAU * (np.tan(self.action['steering']) - self.steering_angle) * dt
        # self.velocity += self.action['acceleration'] * dt

        if self.road:
            self.lane_index = self.road.get_lane_index(self.position)
            self.lane = self.road.lanes[self.lane_index]

class MDPVehicle(ControlledVehicle):
    """
        A controlled vehicle with a specified discrete range of allowed target velocities.
    """

    SPEED_COUNT = 3  # []
    SPEED_MIN = 10  # [m/s]
    SPEED_MAX = 30  # [m/s]

    def __init__(self,
                 road,
                 position,
                 heading=0,
                 velocity=0,
                 target_lane_index=None,
                 target_velocity=None,
                 max_length = None,
                 route = None):
        super(MDPVehicle, self).__init__(road, position, heading, velocity, target_lane_index, target_velocity,max_length,route)
        self.velocity_index = self.speed_to_index(self.target_velocity)
        self.target_velocity = self.index_to_speed(self.velocity_index)
        self.id = 0

    def act(self, actions=None):
        """
            Perform a high-level action.

            If the action is a velocity change, choose velocity from the allowed discrete range.
            Else, forward action to the ControlledVehicle handler.

        :param action: a high-level action
        """
        if actions is not None:
            for action in actions.split('+'):
                if action == "FASTER":
                    self.velocity_index = self.speed_to_index(self.velocity) + 1
                elif action == "SLOWER":
                    self.velocity_index = self.speed_to_index(self.velocity) - 1
                else:
                    super(MDPVehicle, self).act(action)
                    # return
        else:
            self.velocity_index = np.clip(self.velocity_index, 0, self.SPEED_COUNT - 1)
            self.target_velocity = self.index_to_speed(self.velocity_index)
            super(MDPVehicle, self).act()

    @classmethod
    def index_to_speed(cls, index):
        """
            Convert an index among allowed speeds to its corresponding speed
        :param index: the speed index []
        :return: the corresponding speed [m/s]
        """
        if cls.SPEED_COUNT > 1:
            return cls.SPEED_MIN + index * (cls.SPEED_MAX - cls.SPEED_MIN) / (cls.SPEED_COUNT - 1)
        else:
            return cls.SPEED_MIN

    @classmethod
    def speed_to_index(cls, speed):
        """
            Find the index of the closest speed allowed to a given speed.
        :param speed: an input speed [m/s]
        :return: the index of the closest speed allowed []
        """
        x = (speed - cls.SPEED_MIN) / (cls.SPEED_MAX - cls.SPEED_MIN)
        return np.int(np.clip(np.round(x * (cls.SPEED_COUNT - 1)), 0, cls.SPEED_COUNT - 1))

    def speed_index(self):
        """
            The index of current velocity
        """
        return self.speed_to_index(self.velocity)

    def predict_trajectory(self, actions, action_duration, trajectory_timestep, dt):
        """
            Predict the future trajectory of the vehicle given a sequence of actions.

        :param actions: a sequence of future actions.
        :param action_duration: the duration of each action.
        :param trajectory_timestep: the duration between each save of the vehicle state.
        :param dt: the timestep of the simulation
        :return: the sequence of future states
        """
        states = []
        v = copy.deepcopy(self)
        t = 0
        for action in actions:
            v.act(action)  # High-level decision
            for _ in range(int(action_duration / dt)):
                t += 1
                v.act()  # Low-level control action
                v.step(dt)
                if (t % int(trajectory_timestep / dt)) == 0:
                    states.append(copy.deepcopy(v))
        return states
