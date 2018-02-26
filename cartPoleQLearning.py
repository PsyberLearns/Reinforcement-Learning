import gym
import math
import numpy as np

class CartPoleQLearner:
    def __init__(self, buckets=(1, 1, 6, 12), episodes=1000, timeSteps=250, minLearningRate=0.1, minEpsilon=0.1, discount=1):
        self.buckets = buckets #Number of finite states (bucket) for each dimension
        self.episodes = episodes
        self.timeSteps = timeSteps
        self.minLearningRate = minLearningRate
        self.minEpsilon = minEpsilon
        self.discount = discount
        self.totalScore = []

        self.env = gym.make("CartPole-v0")
        self.qValues = np.zeros(self.buckets + (self.env.action_space.n,)) #dimensions of the qValue

    def discretize(self, obs):
        upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50)]
        lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50)]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)
    
    def getLearningRate(self, t):
        return max(self.minLearningRate, min(1, 1.0 - math.log10(t+1/25)))

    def getEpsilon(self, t):
        return max(self.minEpsilon, min(1, 1 - math.log10(t+1/25)))

    def chooseAction(self, state, epsilon):
        #generates a random number between 0 and 1, if random number is less than epsilon choose random action, otherwise choose action that will yield the best Q-Value
        if np.random.random() < epsilon:
            return self.env.action_space.sample()
        
        return np.argmax(self.qValues[state])
    
    def update(self, state, action, reward, nextState, learningRate):
        difference = (reward + self.discount * np.max(self.qValues[nextState])) - self.qValues[state + (action,)]
        self.qValues[state + (action,)] += learningRate * difference

    def run(self):

        f = open("score", "w")

        for episode in range(1, self.episodes + 1):
            observation = self.discretize(self.env.reset())
            a, e = self.getLearningRate(episode), self.getEpsilon(episode)

            done = False
            score = 0
            for timeStep in range(self.timeSteps):
                #self.env.render()
                action = self.chooseAction(observation, e)
                newObservation, reward, done, info = self.env.step(action)
                newObservation = self.discretize(newObservation)
                if done: break

                self.update(observation, action, reward, newObservation, a)

                observation = newObservation
                score += reward
            
            f.write("Score on episode {}: {}".format(episode, score))
            f.write("\n")
            self.totalScore.append(score)
            if episode % 100 == 0:
                mean = np.mean(self.totalScore)
                print ("mean score of {} after {} episodes for 100 episodes ".format(mean, episode))
                self.totalScore = []

        f.close()


if __name__ == "__main__":
    agent = CartPoleQLearner()
    agent.run()