import pprint

from SULI.src.tensorForceEnv import CustomEnvironment


def test_done():
    env = CustomEnvironment()

    actions = env.actions()
    print(f"actions: {actions}")

    states, reward, done = env.execute(actions=0)
    print(f"states: {states}")
    print(f"reward: {reward}")
    print(f"done: {done}")

    assert done is False


def test_seven_steps():
    env = CustomEnvironment()

    state_reward_done = []
    for step in range(7):
        state_reward_done.append(env.execute(actions=0))

    pprint.pprint(state_reward_done)

    assert env.extraCounter == 9
    assert state_reward_done[6][2] is True
