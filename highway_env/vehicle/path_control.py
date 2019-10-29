from math import sin, cos, degrees, radians


def plan_path_based_on_dynamics(start_kinematics, steering_angles, longitudinal_velocities, dt):
    """
    Input: A start kinematics and a series of commands(steering angles and longitudinal velocities)
    Output: A series of coordinates(in fixed coordinate system) and yaw angles
    """
    if len(steering_angles) != len(longitudinal_velocities):
        return None, None, None

    m = start_kinematics.weight
    I = start_kinematics.rotational_inertia
    Lf = start_kinematics.distance_from_centroid_to_front_axle
    Lr = start_kinematics.distance_from_centroid_to_rear_axle
    Cf = start_kinematics.front_tire_cornering_stiffness
    Cr = start_kinematics.rear_tire_cornering_stiffness

    Xs, Ys, vXs, vYs = [0.0], [0.0], [longitudinal_velocities[0]], [0.0]
    latitudinal_velocities, thetas, dthetadts, latitudinal_accelerations, d2thetadt2s = [0.0], [0.0], [0.0], [], []

    for i in range(len(steering_angles)):
        delta = radians(steering_angles[i])
        vx = longitudinal_velocities[i]
        vy = latitudinal_velocities[i]
        theta = thetas[i]
        dthetadt = dthetadts[i]
        X = Xs[i]
        Y = Ys[i]

        a = -(Cf * cos(delta) + Cr) / (m * vx)
        b = (-Lf * Cf * cos(delta) + Lr * Cr) / (m * vx) - vx
        c = (-Lf * Cf * cos(delta) + Lr * Cr) / (I * vx)
        d = -(Lf * Lf * Cf * cos(delta) + Lr * Lr * Cr) / (I * vx)
        e = Cf * cos(delta) / m
        f = Lf * Cf * cos(delta) / I

        ay = a * vy + b * dthetadt + e * delta
        vy = vy + ay * dt
        d2thetadt2 = c * vy + d * dthetadt + f * delta
        dthetadt = dthetadt + d2thetadt2 * dt
        theta = theta + dthetadt * dt

        latitudinal_accelerations.append(ay)
        latitudinal_velocities.append(vy)
        d2thetadt2s.append(d2thetadt2)
        dthetadts.append(dthetadt)
        thetas.append(theta)

        vX = vx * cos(theta) - vy * sin(theta)
        vY = vx * sin(theta) + vy * cos(theta)
        vXs.append(vX)
        vYs.append(vY)

        X = X + vX * dt
        Y = Y + vY * dt
        Xs.append(X)
        Ys.append(Y)

    for i in range(len(thetas)):
        thetas[i] = degrees(thetas[i])

    return Xs, Ys, thetas
if __name__ == '__main__':
    import pygame
    import time

    pygame.init()
    screen = pygame.display.set_mode((300, 300))
    done = False

    happy = pygame.image.load('/home/zj/zj/projects/highway-env/red_alpha.png')  # our happy blue protagonist
    #checkers = pygame.image.load('background.png')  # 32x32 repeating checkered image

    while not done:
        start = time.time()
        # pump those events!
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                done = True

        # here comes the protagonist
        screen.blit(happy, (100, 100))

        pygame.display.flip()

        # yeah, I know there's a pygame clock method
        # I just like the standard threading sleep
        end = time.time()
        diff = end - start
        framerate = 30
        delay = 1.0 / framerate - diff
        if delay > 0:
            time.sleep(delay)
