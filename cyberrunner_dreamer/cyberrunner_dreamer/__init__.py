from gym.envs.registration import register


register(
    id="cyberrunner-ros-v0",
    entry_point="cyberrunner_dreamer.env:CyberrunnerGym",
    # max_episode_steps=6000,
)

# register(
#    id='cyberrunner-ros-v1',
#    entry_point='cyberrunner_dreamer.env:CyberrunnerGymV2',
#    #max_episode_steps=6000,
# )
