# Classic Reinforcement learning

Implementation for Reinforcement learning algorithm


### Dynamic programming
- [Iterative policy evaluation](classic_reinforcement_learning/iterative_policy_evaluation.py)
- [Policy iteration](classic_reinforcement_learning/policy_iteration.py)
- [Value iteration](classic_reinforcement_learning/value_iteration.py)

### Monte carlo
- [Policy evaluation](classic_reinforcement_learning/monte_carlo.py)
- [Policy iteration - Epsilon greedy](classic_reinforcement_learning/monte_carlo_es.py)
- [Policy iteration - no ES (epsilon soft)](classic_reinforcement_learning/monte_carlo_no_es.py)

### TD(λ)
- [n-step](td_lambda/n_step.py)

# Deep Reinforcement learning

### [Deep Q learning](q_learning)
- DQN : Playing Atari with Deep Reinforcement Learning [[Paper]](https://arxiv.org/abs/1312.5602)  [[Code]](q_learning/1.dqn)

We present the first deep learning model to successfully learn control policies di-rectly from high-dimensional sensory input using reinforcement learning. The model is a convolutional neural network, trained with a variant of Q-learning, whose input is raw pixels and whose output is a value function estimating future rewards. We apply our method to seven Atari 2600 games from the Arcade Learn-ing Environment, with no adjustment of the architecture or learning algorithm. We find that it outperforms all previous approaches on six of the games and surpasses a human expert on three of them.


- Double DQN: Deep Reinforcement Learning with Double Q-Learning [[Paper]](http://arxiv.org/abs/1509.06461)  [[Code]](q_learning/2.double_dqn)

The popular Q-learning algorithm is known to overestimate action values under certain conditions. It was not previously known whether, in practice, such overestimations are com- mon, whether they harm performance, and whether they can generally be prevented. In this paper, we answer all these questions affirmatively. In particular, we first show that the recent DQN algorithm, which combines Q-learning with a deep neural network, suffers from substantial overestimations in some games in the Atari 2600 domain.We then show that the idea behind the Double Q-learning algorithm, which was introduced in a tabular setting, can be generalized to work with large-scale function approximation. We propose a spe- cific adaptation to the DQN algorithm and show that the re- sulting algorithm not only reduces the observed overestima- tions, as hypothesized, but that this also leads to much better performance on several games.


### Policy Gradient
- CartPole [Tensorflow](policy_gradient/policy_gradient_tf.py), [Keras](policy_gradient/cart-pole%20keras.py)
