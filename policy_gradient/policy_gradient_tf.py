import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym

from td_lambda.q_learning import plot_cost_to_go, plot_running_avg


class HiddenLayer:
    def __init__(self, M1, M2, f=tf.nn.tanh):
        self.W = tf.Variable(tf.random_normal(shape=(M1, M2)) * np.sqrt(2. / M1, dtype=np.float32))
        self.bias = tf.Variable(np.zeros(M2).astype(np.float32))
        self.f = f

    def forward(self, X):
        return self.f(tf.matmul(X, self.W) + self.bias)


class PolicyModel:
    def __init__(self, D, K, hidden_layer_sizes=[]):
        """
        :param D: input dimension (observation dimension)
        :param K: number of action, output dimension
        :param hidden_layer_sizes: size of neurons in hidden layers
        """
        self.layers = []
        M1 = D
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2)
            self.layers.append(layer)
            M1 = M2

        # last layer
        self.layers.append(HiddenLayer(M1, K, tf.nn.softmax))

        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')
        self.advantages = tf.placeholder(tf.float32, shape=(None,), name='advantage')

        # Calculate the output
        Z = self.X
        for layer in self.layers:
            Z = layer.forward(Z)
        p_a_given_s = Z
        self.predict_op = p_a_given_s

        # keeping only the prob for the action taken
        selected_prob = tf.log(
            tf.reduce_sum(p_a_given_s * tf.one_hot(self.actions, K), reduction_indices=[1]))
        cost = -tf.reduce_sum(self.advantages * selected_prob)

        self.train_op = tf.train.AdagradOptimizer(10e-2).minimize(cost)

    def set_session(self, sesstion):
        self.session = sesstion

    def partial_fit(self, X, actions, advantages):
        # Check dimension
        X = np.atleast_2d(X)
        actions = np.atleast_1d(actions)
        advantages = np.atleast_1d(advantages)

        self.session.run(self.train_op, feed_dict={
            self.X: X,
            self.actions: actions,
            self.advantages: advantages
        })

    def predict(self, X):
        X = np.atleast_2d(X)
        return self.session.run(self.predict_op, feed_dict={self.X: X})

    def sample_action(self, X):
        p = self.predict(X)[0]
        return np.random.choice(len(p), p=p)


class ValueModel:
    def __init__(self, D, hidden_layers_sizes):
        self.layers = []
        M1 = D
        for M2 in hidden_layers_sizes:
            layer = HiddenLayer(M1, M2)
            self.layers.append(layer)
            M1 = M2

        # final layer
        layer = HiddenLayer(M1, 1, lambda x: x)
        self.layers.append(layer)

        # inputs and targets
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.Y = tf.placeholder(tf.float32, shape=(None, ), name='Y')

        # Calculate output and cost
        Z = self.X
        for layer in self.layers:
            Z = layer.forward(Z)
        y_hat = tf.reshape(Z, [-1])
        self.predict_op = y_hat

        cost = tf.reduce_sum(tf.square(self.Y - y_hat))
        self.train_op = tf.train.GradientDescentOptimizer(10e-5).minimize(cost)

    def set_session(self, sesstion):
        self.session = sesstion

    def partial_fit(self, X, Y):
        X = np.atleast_2d(X)
        Y = np.atleast_1d(Y)
        self.session.run(self.train_op, feed_dict={self.X: X, self.Y: Y})

    def predict(self, X):
        X = np.atleast_2d(X)
        return self.session.run(self.predict_op, feed_dict={self.X: X})


def play_one_mc(env, pmodel, vmodel, gamma):
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0

    states = []
    actions = []
    rewards = []

    while not done and iters < 2000:
        action = pmodel.sample_action(observation)
        prev_observatoin = observation
        observation, reward, done , info = env.step(action)

        if done:
            reward = -200

        states.append(prev_observatoin)
        actions.append(action)
        rewards.append(reward)

        if reward == 1:
            totalreward += reward
        iters += 1

    returns = []
    advantages = []
    G = 0
    for s, r in zip(reversed(states), reversed(rewards)):
        returns.append(G)
        advantages.append(G - vmodel.predict(s)[0])
        G = r + gamma*G
    returns.reverse()
    advantages.reverse()

    # update the models
    pmodel.partial_fit(states, actions, advantages)
    vmodel.partial_fit(states, returns)

    return iters, totalreward


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    D = env.observation_space.shape[0]  # 4
    K = env.action_space.n  # 2
    pmodel = PolicyModel(D, K, [])
    vmodel = ValueModel(D, [10])
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)
    pmodel.set_session(session)
    vmodel.set_session((session))
    gamma = 0.99

    N = 500
    totalrewards = np.empty(N)
    costs = np.empty(N)
    for n in range(N):
        num_steps, totalreward = play_one_mc(env, pmodel, vmodel, gamma)
        totalrewards[n] = totalreward
        if n % 1 == 0:
            print("episode:", n, "total reward: %.1f" % totalreward, "num steps: %d" % num_steps,
                  "avg reward (last 100): %.1f" % totalrewards[max(0, n - 100):(n + 1)].mean())

    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(totalrewards)