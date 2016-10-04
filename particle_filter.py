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
        self.particles = self.trans_fn(self.particles)
        p_likeli = self.obs_fn(self.particles, curr_obs)
        # Perform weighted sampling
        idx = np.random.choice(np.arange(self.num_particles),
                               p=p_likeli, size=self.num_particles)
        self.particles = self.resample_fn(self.particles, idx)

    def run(self, obs, num_particles=1000, num_iter=100):
        assert len(obs) == num_iter
        self.particles = self.init_fn(num_particles)
        self.num_particles = num_particles
        queries = []
        for i in range(num_iter):
            self.resample(obs[i])
            queries.append(self.query_fn(self.particles))
            print(i, queries[-1])
        return queries

