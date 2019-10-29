from __future__ import division, print_function, absolute_import

import os

import numpy as np
import pygame

from highway_env.road.graphics import WorldSurface, RoadGraphics
from highway_env.vehicle.graphics import VehicleGraphics

CONTROL = {
    "throttle": 0,
    "brake": 0.,
    "steering": 0
}
ControlledVehicle_CONTROL = {
    'acceleration': 0,
    'steering': 0
}
class EnvViewer(object):
    """
        A viewer to render a highway driving environment.
    """
    screen = None
    SCREEN_WIDTH = 1800
    SCREEN_HEIGHT = 500
    pygame.init()
    VehicleGraphics.FONT = pygame.font.SysFont('Arial', 16)

    def __init__(self, env):
        self.env = env

        pygame.display.set_caption("Highway-env")
        panel_size = (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT),0,32)
        # self.screen.fill ((0,0,0))
        self.sim_surface = WorldSurface(panel_size, 0, pygame.Surface(panel_size))
        self.clock = pygame.time.Clock()

        self.enabled = True
        if "SDL_VIDEODRIVER" in os.environ and os.environ["SDL_VIDEODRIVER"] == "dummy":
            self.enabled = False

        self.agent_display = None
        self.agent_surface = None

    def set_agent_display(self, agent_display):
        if self.agent_display is None:
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, 2 * self.SCREEN_HEIGHT))
            self.agent_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.agent_display = agent_display

    def handle_control(self):
        pygame.event.pump()
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_UP]:
            ControlledVehicle_CONTROL['steering'] = ControlledVehicle_CONTROL['steering'] - 0.001 if ControlledVehicle_CONTROL['steering'] > -0.2 else -0.2
        if pressed[pygame.K_DOWN]:
            ControlledVehicle_CONTROL['steering'] = ControlledVehicle_CONTROL['steering'] + 0.001 if ControlledVehicle_CONTROL['steering'] < 0.2 else 0.2
        if pressed[pygame.K_RIGHT]:
            ControlledVehicle_CONTROL['acceleration'] = ControlledVehicle_CONTROL['acceleration'] + 0.001 if ControlledVehicle_CONTROL['acceleration'] < 1 else 1
        if pressed[pygame.K_LEFT]:
            ControlledVehicle_CONTROL['acceleration'] = ControlledVehicle_CONTROL['acceleration'] - 0.001 if ControlledVehicle_CONTROL['acceleration'] > -1 else -1
        # if not pressed[pygame.K_UP] and not pressed[pygame.K_DOWN]:
        #     if ControlledVehicle_CONTROL['steering'] > 0:
        #         ControlledVehicle_CONTROL['steering'] -= 0.1
        #     elif ControlledVehicle_CONTROL['steering'] < 0:
        #         ControlledVehicle_CONTROL['steering'] += 0.1
        #     if -0.1 < ControlledVehicle_CONTROL['steering'] < 0.1:
        #         ControlledVehicle_CONTROL['steering'] = 0
        # if not pressed[pygame.K_LEFT]:
        #     ControlledVehicle_CONTROL['brake'] = ControlledVehicle_CONTROL['brake'] - 0.1 if ControlledVehicle_CONTROL['brake'] > 0 else 0
        # if not pressed[pygame.K_RIGHT]:
        #     ControlledVehicle_CONTROL['throttle'] = ControlledVehicle_CONTROL['throttle'] - 0.1 if ControlledVehicle_CONTROL['throttle'] > 0 else 0
        #print(ControlledVehicle_CONTROL)
        self.env.vehicle.act(ControlledVehicle_CONTROL)
    def handle_carsim(self):
        pygame.event.pump()
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_UP]:
            CONTROL['steering'] = CONTROL['steering'] + 0.05 if CONTROL['steering'] < 0.3 else 0.3
        if pressed[pygame.K_DOWN]:
            CONTROL['steering'] = CONTROL['steering'] - 0.05 if CONTROL['steering'] > -0.3 else -0.3
        if pressed[pygame.K_LEFT]:
            CONTROL['brake'] = CONTROL['brake'] + 0.05 if CONTROL['brake'] < 1 else 1
        if pressed[pygame.K_RIGHT]:
            CONTROL['throttle'] = CONTROL['throttle'] + 0.05 if CONTROL['throttle'] < 1 else 1
        if not pressed[pygame.K_UP] and not pressed[pygame.K_DOWN]:
            if CONTROL['steering'] > 0:
                CONTROL['steering'] -= 0.1
            elif CONTROL['steering'] < 0:
                CONTROL['steering'] += 0.1
            if -0.1 < CONTROL['steering'] < 0.1:
                CONTROL['steering'] = 0
        if not pressed[pygame.K_LEFT]:
            CONTROL['brake'] = CONTROL['brake'] - 0.1 if CONTROL['brake'] > 0 else 0
        if not pressed[pygame.K_RIGHT]:
            CONTROL['throttle'] = CONTROL['throttle'] - 0.1 if CONTROL['throttle'] > 0 else 0
        self.env.vehicle.act(CONTROL)
        # action = m.send_request_other()
        # if 'error' not in action:
        #     print(action)
        #print(CONTROL['brake'])

    def handle_events(self):
        """
            Handle pygame events by forwarding them to the display and environment vehicle.
        """
        #self.handle_carsim()
        #self.handle_control()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.env.close()
            if self.env.vehicle:
                CONTROL = VehicleGraphics.handle_event(self.env.vehicle, event)
            self.sim_surface.handle_event(event)

        # car_sim_data = m.send_control(VehicleGraphics.CONTROL)
        # #t = m.send_control(VehicleGraphics.CONTROL)
        # position = car_sim_data['location']
        # rotation = car_sim_data['rotation']
        # self.env.vehicle.position[0] = position[0]
        # self.env.vehicle.position[1] = position[1]
        #self.env.vehicle.heading = -rotation[2] / 57.3

    def display(self):
        """
            Display the road and vehicles on a pygame window.
        """
        if not self.enabled:
            return

        self.sim_surface.move_display_window_to(self.window_position())
        RoadGraphics.display(self.env.road, self.sim_surface)
        # RoadGraphics.display_bigmap(self.env, self.env.road, self.sim_surface)
        RoadGraphics.display_traffic(self.env.road, self.sim_surface)
        # RoadGraphics.display_traffic_bigmap(self.env,self.env.road, self.sim_surface)

        if self.agent_display:
            self.agent_display(self.agent_surface, self.sim_surface)
            self.screen.blit(self.agent_surface, (0, self.SCREEN_HEIGHT))

        self.screen.blit(self.sim_surface, (0, 0))
        # pygame.display.update()
        # self.clock.tick(20)
        pygame.display.flip()

    def get_image(self):
        """
        :return: the rendered image as a rbg array
        """
        data = pygame.surfarray.array3d(self.screen)
        return np.moveaxis(data, 0, 1)

    def window_position(self):
        """
        :return: the world position of the center of the displayed window.
        """
        if self.env.vehicle:
            if False:
                return self.env.vehicle.position
            else:
                # return np.array([self.env.vehicle.position[0], self.env.road.network.LANES_NUMBER / 2 * 4 - 2])
                return np.array([self.env.vehicle.position[0], self.env.vehicle.position[1]])
        else:
            return np.array([0, self.env.road.network.LANES_NUMBER / 2 * 4])

    def close(self):
        """
            Close the pygame window.
        """
        pygame.quit()

