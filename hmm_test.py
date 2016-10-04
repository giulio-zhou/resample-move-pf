import numpy as np
import random
from particle_filter import ParticleFilter

# ===============================================
# Supporting functions for HMM DNA example
# ===============================================
class hmmDNA(object):
    def gen_obs():
        return np.array([1, 0, 0, 0, 2])

    def init_fn(num_particles):
        priors = np.array([0.3, 0.2, 0.1, 0.4])
        return np.random.choice(np.arange(4), p=priors, size=num_particles)

    def obs_fn(particles, obs):
        obs_matrix = np.array([[0.85, 0.05, 0.05, 0.05],
                               [0.05, 0.85, 0.05, 0.05],
                               [0.05, 0.05, 0.85, 0.05],
                               [0.05, 0.05, 0.05, 0.85]])
        likeli = obs_matrix[particles, obs]
        likeli /= np.sum(likeli)
        return likeli

    def query_fn(particles):
        num_particles = float(particles.shape[0])
        output = {}
        classes = ['A', 'C', 'G', 'T']
        for val in range(4):
            idx = np.where(particles == val)
            output[val] = len(idx[0]) / num_particles
        return output 

    def resample_fn(particles, idx):
        return particles[idx]

    def trans_fn(particles):
        trans_matrix = np.array([[0.1, 0.3, 0.3, 0.3],
                                 [0.3, 0.1, 0.3, 0.3],
                                 [0.3, 0.3, 0.1, 0.3],
                                 [0.3, 0.3, 0.3, 0.1]])
        trans_particles = particles.copy() 
        for val in range(4):
            idx = np.where(particles == val)
            trans_particles[idx] = np.random.choice(
                np.arange(4), p=trans_matrix[val], size=len(idx[0]))
        return trans_particles
        

    gen_obs = staticmethod(gen_obs)
    init_fn = staticmethod(init_fn)
    obs_fn = staticmethod(obs_fn)
    query_fn = staticmethod(query_fn)
    resample_fn = staticmethod(resample_fn)
    trans_fn = staticmethod(trans_fn)

def main():
    num_iter = 5
    num_particles = 10000
    init_fn, obs_fn = hmmDNA.init_fn, hmmDNA.obs_fn
    query_fn, trans_fn = hmmDNA.query_fn, hmmDNA.trans_fn
    resample_fn = hmmDNA.resample_fn
    obs = hmmDNA.gen_obs()
    pf = ParticleFilter(init_fn, trans_fn, obs_fn, query_fn, resample_fn)
    pf.run(obs, num_particles, num_iter)

if __name__ == '__main__':
    main()
