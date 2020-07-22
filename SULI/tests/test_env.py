import pprint

from SULI.src.tensorForceEnv import CustomEnvironment


def test_done():
    env = CustomEnvironment()


    actions = env.actions()
    print(f"actions: {actions}")

    assert env.extraCounter == 3
    print(f"extra Count: {env.extraCounter}")

    states, reward, done = env.execute(actions=0)
    print(f"states: {states}")
    print(f"reward: {reward}")
    print(f"done: {done}")

    assert done is False

    states, reward, done = env.execute(actions=0)
    assert done is False

    states, reward, done = env.execute(actions=0)
    assert done is False

    states, reward, done = env.execute(actions=0)
    assert done is False

    states, reward, done = env.execute(actions=0)
    assert done is False

    states, reward, done = env.execute(actions=0)
    assert done is False

    states, reward, done = env.execute(actions=0)
    assert done is True

    # states, reward, done = env.execute(actions=0)
    print(f"states: {states}")
    print(f"reward: {reward}")
    print(f"done: {done}")

    # assert done is False


def test_reset():
    env = CustomEnvironment()

    assert env.extraCounter == 3
    assert env.agent_pos == 3
    assert len(env.GRID) == env.SAMPLES
    assert len(env.GRID[0]) == env.TRIALS


    env.execute(actions=0)
    env.reset()

    assert env.extraCounter == 3
    assert env.agent_pos == 3
    assert len(env.GRID) == env.SAMPLES
    assert len(env.GRID[0]) == env.TRIALS

    env.execute(actions=0)
    env.reset()

    assert env.extraCounter == 3
    assert env.agent_pos == 3
    assert len(env.GRID) == env.SAMPLES
    assert len(env.GRID[0]) == env.TRIALS


def test_seven_steps():
    env = CustomEnvironment()

    state_reward_done = []
    for step in range(7):
        state_reward_done.append(env.execute(actions=0))

    pprint.pprint(state_reward_done)

    assert env.extraCounter == 10
    assert state_reward_done[6][2] is True


def test_stepthru_reset():
    env = CustomEnvironment()

    assert env.agent_pos == env.startingPoint
    assert env.extraCounter == env.startingPoint

    state_reward_done = []
    for step in range(7):
        state_reward_done.append(env.execute(actions=0))

    env.reset()

    for step in range(7):
        state_reward_done.append(env.execute(actions=0))

    pprint.pprint(state_reward_done)

    assert env.extraCounter == 10
    assert state_reward_done[6][2] is True
    assert state_reward_done[-1][2] is True