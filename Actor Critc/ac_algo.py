"""
Actor-Critic with continuous action using TD-error as the Advantage, Reinforcement Learning.

Using:
tensorflow r1.3
gym 0.8.0
"""
import time
import tensorflow as tf
import numpy as np
from UAV_env import UAVEnv
from state_normalization import StateNormalization
import matplotlib.pyplot as plt


# np.random.seed(2)
# tf.set_random_seed(2)  # reproducible


class Actor(object):
    def __init__(self, sess, a_dim, n_features, action_bound, lr=0.001):
        self.sess = sess
        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        # self.a = tf.placeholder(tf.float32, None, name="act")
        self.a = tf.placeholder(tf.float32, shape=[1, a_dim], name="act")
        self.td_error = tf.placeholder(tf.float32, None, name="td_error")  # TD_error

        l1 = tf.layers.dense(
            inputs=self.s,
            units=400,  # number of hidden units
            activation=tf.nn.relu6,
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
            name='l1'
        )
        # l2 = tf.layers.dense(
        #     inputs=l1,
        #     units=300,  # number of hidden units
        #     activation=tf.nn.relu6,
        #     kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
        #     bias_initializer=tf.constant_initializer(0.1),  # biases
        #     name='l2'
        # )
        l3 = tf.layers.dense(
            inputs=l1,
            units=10,  # number of hidden units
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
            name='l3'
        )

        mu = tf.layers.dense(
            inputs=l3,
            # units=1,  # number of hidden units
            units=a_dim,  # number of hidden units
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
            name='mu'
        )

        sigma = tf.layers.dense(
            inputs=l3,
            units=a_dim,  # output units
            activation=tf.nn.softplus,  # get action probabilities
            # activation=tf.nn.tanh,  # get action probabilities
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(1.),  # biases
            name='sigma'
        )
        global_step = tf.Variable(0, trainable=False)
        # self.e = epsilon = tf.train.exponential_decay(2., global_step, 1000, 0.9)
        ##squeeze 去掉值为1的维度[[0]] ----> 0
        # self.mu, self.sigma = tf.squeeze(mu * 2), tf.squeeze(sigma + 0.1)
        self.mu, self.sigma = tf.squeeze(mu), tf.squeeze(sigma + 0.1)
        # tf.distributions.normal可以生成一个均值为loc，方差为scale的正态分布。
        self.normal_dist = tf.distributions.Normal(self.mu, self.sigma)

        ##tf.clip_by_value:输入一个张量A，把A中的每一个元素的值都压缩在min和max之间。小于min的让它等于min，大于max的元素的值等于max。
        # sample输出value值，在正态分布中概率的log值，作为loss值
        # self.action = tf.clip_by_value(self.normal_dist.sample(1), action_bound[0], action_bound[1])
        self.action = tf.clip_by_value(self.normal_dist.sample(), action_bound[0], action_bound[1])

        with tf.name_scope('exp_v'):
            log_prob = self.normal_dist.log_prob(self.a)  # loss without advantage
            self.exp_v = log_prob * self.td_error  # advantage (TD_error) guided loss
            # Add cross entropy cost to encourage exploration
            self.exp_v += 0.01 * self.normal_dist.entropy()

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v, global_step)  # min(v) = max(-v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        return self.sess.run(self.action, {self.s: s})  # get probabilities for all actions


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess
        with tf.name_scope('inputs'):
            self.s = tf.placeholder(tf.float32, [1, n_features], "state")
            self.v_ = tf.placeholder(tf.float32, [1, 1], name="v_next")
            self.r = tf.placeholder(tf.float32, name='r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=400,  # number of hidden units
                activation=tf.nn.relu6,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )
            # l2 = tf.layers.dense(
            #     inputs=l1,
            #     units=300,  # number of hidden units
            #     activation=tf.nn.relu6,
            #     kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            #     # bias_initializer=tf.constant_initializer(0.1),  # biases
            #     name='l2'
            # )
            l3 = tf.layers.dense(
                inputs=l1,
                units=10,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l3'
            )

            self.v = tf.layers.dense(
                inputs=l3,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = tf.reduce_mean(self.r + GAMMA * self.v_ - self.v)
            self.loss = tf.square(self.td_error)  # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op], {self.s: s, self.v_: v_, self.r: r})
        return td_error


OUTPUT_GRAPH = False
MAX_EPISODE = 1000
# DISPLAY_REWARD_THRESHOLD = -100  # renders environment if total episode reward is greater then this threshold
GAMMA = 0.001
LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic

env = UAVEnv()
Normal = StateNormalization()
MAX_EP_STEPS = env.slot_num

N_S = env.state_dim
A_BOUND = env.action_bound[1]
a_dim = env.action_dim

sess = tf.Session()

actor = Actor(sess, a_dim=a_dim, n_features=N_S, lr=LR_A, action_bound=[-A_BOUND, A_BOUND])
critic = Critic(sess, n_features=N_S, lr=LR_C)

sess.run(tf.global_variables_initializer())

if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)

t1 = time.time()
ep_reward_list = []
for i in range(MAX_EPISODE):
    s = env.reset()
    ep_reward = 0
    j = 0
    while j < MAX_EP_STEPS:
        a = actor.choose_action(Normal.state_normal(s))

        s_, r, is_terminal, step_redo = env.step(a)
        td_error = critic.learn(Normal.state_normal(s), r,
                                Normal.state_normal(s_))  # gradient = grad[r + gamma * V(s_) - V(s)]
        actor.learn(s, [a], td_error)  # true_gradient = grad[logPi(s,a) * td_error]
        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS - 1 or is_terminal:
            print('Episode:', i, ' Steps: %2d' % j, ' Reward: %7.2f' % ep_reward)
            ep_reward_list = np.append(ep_reward_list, ep_reward)
            break
        j = j + 1
print('Running time: ', time.time() - t1)
plt.plot(ep_reward_list)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()
