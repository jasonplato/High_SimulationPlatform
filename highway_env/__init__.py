from gym.envs.registration import register

register(
    id='highway-v0',
    entry_point='highway_env.envs:HighwayEnv',
)
register(
    id='highway-v1',
    entry_point='highway_env.envs:HighwayEnv_v1',
)
register(
    id='highway-roundabout-v0',
    entry_point='highway_env.envs:RoundaboutEnv',
)

register(
    id='highway-merge-v0',
    entry_point='highway_env.envs:MergeEnv',
)
register(
    id='highway-mergein-v0',
    entry_point='highway_env.envs:MergeEnvIn',
)
register(
    id='highway-mergeout-v0',
    entry_point='highway_env.envs:MergeEnvOut',
)
register(
    id='M4U-v2',
    entry_point='highway_env.envs:CrossroadEnv',
)
register(
    id='ObsAvoid-v1',
    entry_point='highway_env.envs:ObstacleAvoidanceEnv',
)
