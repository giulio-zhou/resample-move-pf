import numpy as np
import random
from scipy.stats import norm
from particle_filter import ParticleFilter

# ===============================================
# Supporting functions for simple linear example
# with Gaussian noise observations 
# ===============================================
OBS_STDDEV = 1
TRANS_STDDEV = 1
class GaussianLinear(object):
    def gen_obs(num_iter):
        theta = 0.7
        curr_val = 0
        obs = [np.random.normal(curr_val, OBS_STDDEV)]
        for i in range(num_iter - 1):
            curr_val = np.random.normal(theta * curr_val, TRANS_STDDEV)
            obs.append(np.random.normal(curr_val, OBS_STDDEV))
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
        likeli = norm(0, OBS_STDDEV).pdf(devs)
        likeli /= np.sum(likeli)
        return likeli

    def query_fn(particles):
        """
        For now, compute answer as average of particle hypotheses
        """
        return np.sum(particles[1]) / float(particles.shape[1])

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
                          np.random.normal(0, TRANS_STDDEV, particles.shape[1])
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
