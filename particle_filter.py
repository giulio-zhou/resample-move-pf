import numpy as np
import random

class ParticleFilter(object):
    def __init__(self, init_fn, trans_fn, obs_fn, query_fn, resample_fn):
        self.init_fn = init_fn
        self.trans_fn = trans_fn
        self.obs_fn = obs_fn
        self.query_fn = query_fn
        self.resample_fn = resample_fn
        self.particles = []
        self.num_particles = 0

    def resample(self, curr_obs):
        p_likeli = self.obs_fn(self.particles, curr_obs)
        # Perform weighted sampling
        idx = np.random.choice(np.arange(self.num_particles),
                               p=p_likeli, size=self.num_particles)
        self.particles = self.resample_fn(self.particles, idx)
        self.particles = self.trans_fn(self.particles)

    def query(self, curr_obs):
        p_likeli = self.obs_fn(self.particles, curr_obs)
        return self.query_fn(self.particles, p_likeli)

    def run(self, obs, num_particles=1000, num_iter=100):
        assert len(obs) == num_iter + 1
        self.particles = self.init_fn(num_particles)
        self.num_particles = num_particles
        queries = []
        for i in range(num_iter):
            self.resample(obs[i])
            queries.append(self.query(obs[i]))
            print(i, queries[-1])
        return queries

# ===============================================
# Supporting functions for simple linear example
# with Gaussian noise observations 
# ===============================================
from scipy.stats import norm
class GaussianLinear(object):
    def gen_obs(num_iter):
        theta = 0.7
        curr_val = 0
        obs = [np.random.normal(curr_val, 1)]
        for i in range(num_iter):
            curr_val = np.random.normal(theta * curr_val, 1)
            obs.append(np.random.normal(curr_val, 1))
        # return obs
        return np.array(obs)

    def init_fn(num_particles):
        """
        State consists of a 2-by-num_particles NumPy array with (1) the current
        hypothesis for the state value and (2) the guess for theta.
        """
        return np.vstack((np.zeros(num_particles),
                          np.random.normal(0, 0.1, num_particles)))

    def obs_fn(particles, obs):
        """
        Given the particle hypotheses and current observation, calculate the
        deviations and compute the pdf using zero-mean, unit variance Gaussian.
        """
        devs = obs - particles[1]
        likeli = norm(0, 1).pdf(devs)
        likeli /= np.sum(likeli)
        return likeli

    def query_fn(particles, p_likeli):
        """
        For now, compute answer as weighted sum of particle hypotheses
        """
        return np.sum(particles[1] * p_likeli)

    def resample_fn(particles, idx):
        """
        Given next set of indices, get resampled particles
        """
        return particles[:, idx]

    def trans_fn(particles):
        """
        Multiply hidden state by theta, then add Gaussian noise for transition.
        """
        trans_particles = particles[0] * particles[1] + \
                          np.random.normal(0, 1, particles.shape[1])
        return np.vstack((trans_particles, particles[1]))

    gen_obs = staticmethod(gen_obs)
    init_fn = staticmethod(init_fn)
    obs_fn = staticmethod(obs_fn)
    query_fn = staticmethod(query_fn)
    resample_fn = staticmethod(resample_fn)
    trans_fn = staticmethod(trans_fn)

def main():
    num_iter = 1000
    num_particles = 100000
    init_fn, obs_fn = GaussianLinear.init_fn, GaussianLinear.obs_fn
    query_fn, trans_fn = GaussianLinear.query_fn, GaussianLinear.trans_fn
    resample_fn = GaussianLinear.resample_fn
    obs = GaussianLinear.gen_obs(num_iter)
    pf = ParticleFilter(init_fn, trans_fn, obs_fn, query_fn, resample_fn)
    pf.run(obs, num_particles, num_iter)

if __name__ == '__main__':
    main()
