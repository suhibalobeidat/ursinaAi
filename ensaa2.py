import gym

env = gym.make("CartPole-v1")

done = False
obs = env.reset()

while not done:
    action = env.action_space.sample()
    obs,reward,done,infor = env.step(action)
    env.render()