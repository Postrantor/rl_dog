from sac import SAC2Agent, train_loop
import mdoger7_gym_env as e

env = e.mdoger7BulletEnv(render=True, drift_weight=0.625, shake_weight=0.375, energy_weight=0.005)
agent = SAC2Agent(env)
train_loop(env, agent, 500000, 1000, 256, intermediate_policies=True, verbose=True)
agent.save_policy("finalPolicy.pth")
