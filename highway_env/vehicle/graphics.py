from __future__ import division, print_function
import numpy as np
import pygame

from highway_env.vehicle.dynamics import Vehicle,RedLight
from highway_env.vehicle.control import ControlledVehicle, MDPVehicle,CarSim
from highway_env.vehicle.behavior import IDMVehicle, LinearVehicle,MPCControlledVehicle
#from client import m

class VehicleGraphics(object):

    RED = (255, 100, 100)
    GREEN = (50, 200, 0)
    BLUE = (100, 200, 255)
    YELLOW = (200, 200, 0)
    BLACK = (60, 60, 60)
    PURPLE = (200, 0, 150)
    WHITE = (255,255,255)
    DEFAULT_COLOR = YELLOW
    EGO_COLOR = GREEN
    FONT = None

    CONTROL = {
        "throttle": 0,
        "brake": 0.,
        "steering": 0
    }

    @classmethod
    def display(cls, vehicle, surface, transparent=False,command_dict = None):
        """
            Display a vehicle on a pygame surface.

            The vehicle is represented as a colored rotated rectangle.

        :param vehicle: the vehicle to be drawn
        :param surface: the surface to draw the vehicle on
        :param transparent: whether the vehicle should be drawn slightly transparent
        """
        v = vehicle
        s = pygame.Surface((surface.pix(v.LENGTH), surface.pix(v.WIDTH)), pygame.SRCALPHA)  # per-pixel alpha
        rect = (0, surface.pix(v.LENGTH) / 2 - surface.pix(v.WIDTH) / 2, surface.pix(v.LENGTH), surface.pix(v.WIDTH))

        pygame.draw.rect(s, cls.get_color(v, transparent), rect, 0)
        pygame.draw.rect(s, cls.WHITE, rect, 1)

        #s = pygame.Surface.convert_alpha(s)
        h = v.heading if abs(v.heading) > 2 * np.pi / 180 else 0
        # if v.id == 1:
        #     print(v.position[0])
        sr = pygame.transform.rotate(s, -h * 180 / np.pi)
        # print("v.position_x:",v.position[0],"\n")
        # print("v.position_y:",v.position[1],"\n")
        surface.blit(sr, (surface.pos2pix(v.position[0] - v.LENGTH / 2, v.position[1] - v.LENGTH / 2)))

        if not hasattr(vehicle,"state"):
            surface.blit(VehicleGraphics.FONT.render(str(vehicle.id), True, (255, 255, 255)), (surface.pos2pix(v.position[0] - v.LENGTH / 2, v.position[1] - v.LENGTH / 2)))
            surface.blit(VehicleGraphics.FONT.render('{:.2f}'.format(vehicle.velocity), True, (255, 255, 255)), (surface.pos2pix(v.position[0] - v.LENGTH / 2, v.position[1] + v.LENGTH / 4)))
        # if v.id == 49:
        #     print(49,v.position)
        #surface.blit(VehicleGraphics.FONT.render(str(vehicle.id), True, (0, 0, 0)), (surface.pos2pix(100,100)))
        if isinstance(vehicle,IDMVehicle):
            command_dict[v.id] = {'location':[v.position[0],v.position[1]-15.8,0],'rotation':[0,0,h * 57.3]}
        # else:
        #     surface.blit(VehicleGraphics.FONT.render(str(vehicle.id), True, (0, 0, 0)), (surface.origin[0] + 100 , surface.origin[1]+ 100))

    @classmethod
    def display_trajectory(cls, states, surface):
        """
            Display the whole trajectory of a vehicle on a pygame surface.

        :param states: the list of vehicle states within the trajectory to be displayed
        :param surface: the surface to draw the vehicle future states on
        """
        for vehicle in states:
            cls.display(vehicle, surface, transparent=True)

    @classmethod
    def get_color(cls, vehicle, transparent=False):
        color = cls.DEFAULT_COLOR
        if vehicle.crashed:
            color = cls.RED

        elif isinstance(vehicle, LinearVehicle):
            color = cls.PURPLE
        elif isinstance(vehicle, MPCControlledVehicle):
            color = cls.RED
        elif isinstance(vehicle, IDMVehicle):
            if vehicle.id ==0 :
                color = cls.RED
            else :
                color = cls.BLUE
        elif isinstance(vehicle, MDPVehicle):
            color = cls.EGO_COLOR
        elif isinstance(vehicle,RedLight):
            if vehicle.state == "GREEN":
                color = cls.GREEN
            elif vehicle.state == "RED":
                color = cls.RED
            elif vehicle.state == "YELLOW":
                color = cls.YELLOW
            else:
                color = cls.GREEN
                transparent = True
        if transparent:
            color = (color[0], color[1], color[2], 50)
        return color

    @classmethod
    def handle_event(cls, vehicle, event):
        """
            Handle a pygame event depending on the vehicle type

        :param vehicle: the vehicle receiving the event
        :param event: the pygame event
        """
        if isinstance(vehicle, CarSim):
            return cls.carsim_event(vehicle, event)
        if isinstance(vehicle, ControlledVehicle):
            cls.control_event(vehicle, event)
        elif isinstance(vehicle, Vehicle):
            cls.dynamics_event(vehicle, event)

    @classmethod
    def carsim_event(cls, vehicle, event):
        """
            Map the pygame keyboard events to control decisions

        :param vehicle: the vehicle receiving the event
        :param event: the pygame event
        """
        #for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                VehicleGraphics.CONTROL['throttle'] += 0.1 if VehicleGraphics.CONTROL['throttle'] <= 1 else 1
            # else:
            #     VehicleGraphics.CONTROL['throttle'] -= 0.1 if VehicleGraphics.CONTROL['throttle'] >= 0 else 0
            if event.key == pygame.K_LEFT:
                VehicleGraphics.CONTROL['brake'] += 0.1 if VehicleGraphics.CONTROL['brake'] <= 1 else 1
            # else:
            #     VehicleGraphics.CONTROL['brake'] -= 0.1 if VehicleGraphics.CONTROL['brake'] >= 0 else 0
            if event.key == pygame.K_DOWN:
                VehicleGraphics.CONTROL['steering'] += 0.1 if VehicleGraphics.CONTROL['steering'] <= 1 else 1
            # else:
            #     VehicleGraphics.CONTROL['steering'] -= 0.1 if VehicleGraphics.CONTROL['steering'] >= 0 else 0
            if event.key == pygame.K_UP:
                VehicleGraphics.CONTROL['steering'] -= 0.1 if VehicleGraphics.CONTROL['steering'] >= -1 else -1
            # else:
            #     VehicleGraphics.CONTROL['steering'] += 0.1 if VehicleGraphics.CONTROL['steering'] < 0 else 0
        return VehicleGraphics.CONTROL
        # print(VehicleGraphics.CONTROL)
        # car_sim_data = m.send_control(VehicleGraphics.CONTROL)
        # #t = m.send_control(VehicleGraphics.CONTROL)
        # position = car_sim_data['location']
        # rotation = car_sim_data['rotation']
        # vehicle.position[0] = position[0]
        # vehicle.position[1] = position[1]
        # vehicle.heading = rotation[2] / 57.3
        #print('carsim',position)
        #print(t)

    @classmethod
    def control_event(cls, vehicle, event):
        """
            Map the pygame keyboard events to control decisions

        :param vehicle: the vehicle receiving the event
        :param event: the pygame event
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                vehicle.act("FASTER")
            if event.key == pygame.K_LEFT:
                vehicle.act("SLOWER")
            if event.key == pygame.K_DOWN:
                vehicle.act("LANE_RIGHT")
            if event.key == pygame.K_UP:
                vehicle.act("LANE_LEFT")

    @classmethod
    def dynamics_event(cls, vehicle, event):
        """
            Map the pygame keyboard events to dynamics actuation

        :param vehicle: the vehicle receiving the event
        :param event: the pygame event
        """
        action = vehicle.action
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                action['acceleration'] = 4
            if event.key == pygame.K_LEFT:
                action['acceleration'] = -6
            if event.key == pygame.K_DOWN:
                action['steering'] = 20 * np.pi / 180
            if event.key == pygame.K_UP:
                action['steering'] = -20 * np.pi / 180
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_RIGHT:
                action['acceleration'] = 0
            if event.key == pygame.K_LEFT:
                action['acceleration'] = 0
            if event.key == pygame.K_DOWN:
                action['steering'] = 0
            if event.key == pygame.K_UP:
                action['steering'] = 0
        if action != vehicle.action:
            vehicle.act(action)
