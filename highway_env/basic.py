import gym
import highway_env.envs
from client import signal_handler
import sys, traceback, time
import numpy as np

np.set_printoptions(suppress=True, precision=4)

try:
    for i in range(30):
        done = False
        # env = gym.make('highway-v0')
        # env = gym.make("highway-roundabout-v0")
        # env = gym.make("highway-merge-v0")
        # env = gym.make("highway-mergein-v0")
        # env = gym.make("highway-mergeout-v0")
        env = gym.make("ObsAvoid-v1")
        # env = gym.make("M4U-v2")
        # env.ego_vehicle_switch()
        env.reset()
        # print('Task: blue vehicle named 0 trying to get out of the road!')
        # env = highway_env.envs.highway_env.HighwayEnv()
        env.render()

        # env.vehicle.lane_index = 1
        # env.vehicle.target_velocity = 18
        # print(env.vehicle.position)
        # cnt = 0
        # absolute_time = time.time()
        # profile = LineProfiler(env.step)
        # profile = LineProfiler(env.fake_step)
        # profile.enable()
        t_old = time.time()
        features = []
        while not done:
            # cnt += 1
            # if cnt % 10 == 0:
            #     print('iteration: {} using time: {}'.format(cnt, time.time() - absolute_time))
            #     absolute_time = time.time()
            # ob, r, done, info = env.step(1)
            # global_mobil(env, 6)
            done = env.fake_step()
            # features.append(feature)
            t_new = time.time()
            # time = t_new - t_old
            print('time: {}'.format(t_new - t_old))
            t_old = t_new
            # profile.disable()
            # profile.print_stats(sys.stdout)
            # cProfile.run('env.fake_step()')
            # cProfile.run('ob, r, done, info = env.step(1)')
            # print(r)
            # ext.FeatureExtractor(env.road.vehicles,1)
            # print(env.road.vehicles[-2].position,env.road.vehicles[-1].position)
            env.render()
            # print(env.vehicle.position[0])
        # features = np.array(features)
        # with open("features.txt", "w") as f:
        #     f.write(features)
        # np.savetxt("features.txt", features)

except Exception as e:
    traceback.print_exc(file=sys.stdout)
    signal_handler()
