from __future__ import division, print_function
import numpy as np
import pandas as pd
import time

from highway_env import utils
from highway_env.logger import Loggable


class Vehicle(Loggable):
    ID = 1
    """
        A moving vehicle on a road, and its dynamics.

        The vehicle is represented by a dynamical system: a modified bicycle model.
        It's state is propagated depending on its steering and acceleration actions.
    """
    COLLISIONS_ENABLED = True
    """ Enable collision detection between vehicles """

    LENGTH = 3.5
    """ Vehicle length [m] """
    WIDTH = 2.0
    """ Vehicle width [m] """
    STEERING_TAU = 0.2
    """ Steering wheel response time [s] """
    DEFAULT_VELOCITIES = [20, 40]
    reset_count = 0
    """ Range for random initial velocities [m/s] """

    def __init__(self, road, position, heading=0, velocity=0, max_length=None):
        self.road = road
        self.position = np.array(position).astype('float')
        self.heading = heading
        self.pre_heading = heading
        self.steering_angle = 0
        self.pre_steering_angle = self.steering_angle
        self.delta_w = 0.0
        self.mission_completed = True
        self.velocity = velocity
        self.lane_index = self.road.network.get_closest_lane_index(self.position) if self.road else np.nan
        self.pre_lane = self.lane_index
        self.lane = self.road.network.get_lane(self.lane_index) if self.road else None
        self.action = {'steering': 0, 'acceleration': 0}
        self.crashed = False
        self.log = []
        self.id = Vehicle.ID
        self.max_length = max_length
        self.agent = None
        self.lanes_around = []
        Vehicle.ID += 1

    def reset(self):
        pass

    @classmethod
    def make_on_lane(cls, road, lane_index, longitudinal, velocity=0):
        """
            Create a vehicle on a given lane at a longitudinal position.
        :param road: the road where the vehicle is driving
        :param lane_index: index of the lane where the vehicle is located
        :param longitudinal: longitudinal position along the lane
        :param velocity: initial velocity in [m/s]
        :return: A vehicle with at the specified position
        """
        lane = road.network.get_lane(lane_index)
        return cls(road, lane.position(longitudinal, 0), lane.heading_at(longitudinal), velocity)

    @classmethod
    def create_random(cls, road, velocity=None, spacing=1):
        """
            Create a random vehicle on the road.

            The lane and /or velocity are chosen randomly, while longitudinal position is chosen behind the last
            vehicle in the road with density based on the number of lanes.

        :param road: the road where the vehicle is driving
        :param velocity: initial velocity in [m/s]. If None, will be chosen randomly
        :param spacing: ratio of spacing to the front vehicle, 1 being the default
        :param np.random.RandomState np_random: a random number generator
        :return: A vehicle with random position and/or velocity
        """
        default_spacing = 40
        _from = road.np_random.choice(list(road.network.graph.keys()))
        _to = road.np_random.choice(list(road.network.graph[_from].keys()))
        _id = road.np_random.choice(len(road.network.graph[_from][_to]))
        offset = spacing * default_spacing * np.exp(-5 / 30 * len(road.network.graph[_from][_to]))
        x0 = np.max([v.position[0] for v in road.vehicles]) if len(road.vehicles) else 3 * offset
        x0 += offset
        velocity = velocity or road.np_random.randint(Vehicle.DEFAULT_VELOCITIES[0], Vehicle.DEFAULT_VELOCITIES[1])
        v = cls(road, road.network.get_lane((_from, _to, _id)).position(x0, 0), 0, velocity)
        return v

    @classmethod
    def create_from(cls, vehicle):
        """
            Create a new vehicle from an existing one.
            Only the vehicle dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position, vehicle.heading, vehicle.velocity)
        return v

    def act(self, action=None):
        """
            Store an action to be repeated.

        :param action: the input action
        """
        if action:
            self.action = action

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
            self.action['acceleration'] = - self.velocity

        v = self.velocity * np.array([np.cos(self.heading), np.sin(self.heading)])
        self.position += v * dt
        self.pre_heading = 0.0 if self.heading is None else self.heading
        # if hasattr(self,"mission_completed"):
        #     if self.action['steering'] >= 0:
        #         self.heading += self.velocity * np.tan(self.action['steering']) / self.LENGTH * dt
        #     else:
        #         self.heading -= self.velocity * np.tan(self.action['steering']) / self.LENGTH * dt
        # else:
        self.heading += self.velocity * np.tan(self.action['steering']) / self.LENGTH * dt

        self.pre_steering_angle = 0.0 if self.steering_angle is None else self.steering_angle
        # self.steering_angle += 1 / self.STEERING_TAU * (np.tan(self.action['steering']) - self.steering_angle) * dt
        self.velocity += self.action['acceleration'] * dt
        if self.velocity <= 0.0:
            self.velocity = 0.0

        if self.road:
            self.lane_index = self.road.network.get_closest_lane_index(self.position)
            # if self.id == 0:
            #     print("pre_lane:",self.pre_lane)
            #     print("current_lane:",self.lane_index)
            if self.lane_index != self.pre_lane:
                # if self.id == 0:
                #     print("lane_change")
                self.mission_completed = True
                self.pre_lane = self.lane_index
            self.lane = self.road.network.get_lane(self.lane_index)
        # if not self.lane.on_lane(self.position):
        if self.max_length and self.position[0] > self.max_length:
            # self.road.vehicles.append(type(self)(self.road, self.lane.position(0,0), velocity=np.random.randint(5,20)+50,dst=0,rever=True))
            if self.agent != 'agent':
                self.road.vehicles.append(self.reset())
                self.road.vehicles.remove(self)

    def lane_distance_to(self, vehicle):
        """
            Compute the signed distance to another vehicle along current lane.

        :param vehicle: the other vehicle
        :return: the distance to the other vehicle [m]
        """
        if not vehicle:
            return np.nan
        return self.lane.local_coordinates(vehicle.position)[0] - self.lane.local_coordinates(self.position)[0]

    def check_collision(self, other):
        """
            Check for collision with another vehicle.

        :param other: the other vehicle
        """
        if not self.COLLISIONS_ENABLED or not other.COLLISIONS_ENABLED or self.crashed or other is self:
            return False
        if hasattr(other, "state") or hasattr(self, "state"):
            if np.linalg.norm(other.position - self.position) <= self.LENGTH:
                return False
        # Fast spherical pre-check
        if np.linalg.norm(other.position - self.position) > self.LENGTH:
            return False

        # Accurate elliptic check
        if utils.point_in_ellipse(other.position, self.position, self.heading, self.LENGTH, self.WIDTH):
            self.velocity = other.velocity = min(self.velocity, other.velocity)
            self.crashed = other.crashed = True
            return True

    def to_dict(self, origin_vehicle=None):
        d = {
            'presence': 1,
            'x': self.position[0],
            'y': self.position[1],
            'vx': self.velocity * np.cos(self.heading),
            'vy': self.velocity * np.sin(self.heading)
        }
        if origin_vehicle:
            origin_dict = origin_vehicle.to_dict()
            for key in ['x', 'y', 'vx', 'vy']:
                d[key] -= origin_dict[key]
        return d

    def dump(self):
        """
            Update the internal log of the vehicle, containing:
                - its kinematics;
                - some metrics relative to its neighbour vehicles.
        """
        data = {
            'x': self.position[0],
            'y': self.position[1],
            'psi': self.heading,
            'vx': self.velocity * np.cos(self.heading),
            'vy': self.velocity * np.sin(self.heading),
            'v': self.velocity,
            'acceleration': self.action['acceleration'],
            'steering': self.action['steering']}

        if self.road:
            for lane_index in range(len(self.road.lanes)):
                lane_coords = self.road.lanes[lane_index].local_coordinates(self.position)
                data.update({
                    'dy_lane_{}'.format(lane_index): lane_coords[1],
                    'psi_lane_{}'.format(lane_index): self.road.lanes[lane_index].heading_at(lane_coords[0])
                })
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
            if front_vehicle:
                data.update({
                    'front_v': front_vehicle.velocity,
                    'front_distance': self.lane_distance_to(front_vehicle)
                })
            if rear_vehicle:
                data.update({
                    'rear_v': rear_vehicle.velocity,
                    'rear_distance': rear_vehicle.lane_distance_to(self)
                })

        self.log.append(data)

    def get_log(self):
        """
            Cast the internal log as a DataFrame.

        :return: the DataFrame of the Vehicle's log.
        """
        return pd.DataFrame(self.log)

    def __str__(self):
        return "{} #{}: {} id: {} ".format(self.__class__.__name__, id(self) % 1000, self.position, self.id)

    def __repr__(self):
        return self.__str__()


class Obstacle(Vehicle):
    """
        A motionless obstacle at a given position.
    """

    # myimage = pygame.image.load("../red_light.jpg")
    def __init__(self, road, position, LENGTH=2.0, WIDTH=2.0):
        super(Obstacle, self).__init__(road, position, velocity=0)
        self.target_velocity = 0
        self.LENGTH = LENGTH
        self.WIDTH = WIDTH
        self.acc = 0
        self.pre_acc = 0


class RedLight(Vehicle):
    """
        A motionless obstacle at a given position.
    """

    def __init__(self, road, position, red_time, green_time, greenflash_time, yellow_time, red_or_green):
        super(RedLight, self).__init__(road, position, velocity=0)
        self.target_velocity = 0
        self.WIDTH = 2.0
        self.LENGTH = self.WIDTH
        self.acc = 0
        self.pre_acc = 0
        self.red_time = red_time
        self.green_time = green_time
        self.green_flash_time = greenflash_time
        self.yellow_time = yellow_time
        self.timer = time.time()
        self.pre_timer = self.timer

        if red_or_green == 0:
            self.state = "RED"
            # self.myimage = pygame.image.load("../red_light.png")
            self.COLLISIONS_ENABLED = True
        else:
            self.state = "GREEN"
            # self.myimage = pygame.image.load("../green_light.png")
            self.COLLISIONS_ENABLED = False
            self.pre_timer = self.pre_timer + 2
        # self.state = "RED" if red_or_green == 0 else "GREEN"

    def act(self, action=None):
        self.timer = time.time()
        if self.state == "RED":
            if self.red_time == 0:
                return
            if (self.timer - self.pre_timer) < self.red_time + 4:
                pass
            else:
                self.COLLISIONS_ENABLED = False
                self.state = "GREEN"
                # self.myimage = pygame.image.load("../green_light.png")
                self.pre_timer = self.timer
        elif self.state == "GREEN":
            if self.green_time == 0:
                return
            if (self.timer - self.pre_timer) < self.green_time:
                pass
            else:
                self.COLLISIONS_ENABLED = False
                self.state = "GREEN_FLASH"
                # self.myimage = pygame.image.load("../red_light.png")
                self.pre_timer = self.timer
        elif self.state == "GREEN_FLASH":
            if self.green_flash_time == 0:
                return
            if (self.timer - self.pre_timer) < self.green_flash_time:
                pass
            else:
                self.COLLISIONS_ENABLED = False
                self.state = "YELLOW"
                self.pre_timer = self.timer
        elif self.state == "YELLOW":
            if self.yellow_time == 0:
                return
            if (self.timer - self.pre_timer) < self.yellow_time:
                pass
            else:
                self.COLLISIONS_ENABLED = True
                self.state = "RED"
                self.pre_timer = self.timer

    def step(self, dt):
        pass
