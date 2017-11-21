import gym

env = gym.make('CartPole-v0')
initial_games = 1000
score_requirement = 50
goal_steps = 500


def some_random_games_first():
    for i_episode in range(20):
        # resetting the enc and getting a random initial state
        observation = env.reset()

        # taking maximum of 100 steps if note gameover mean time
        for t in range(goal_steps):
            env.render()
            print(observation)

            # selecting a random action
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            if done:
                print("Episode finished after {} fimesteps".format(t + 1))
                break


def initial_population():
    training_data = []
    scores = []
    accepted_score = []
