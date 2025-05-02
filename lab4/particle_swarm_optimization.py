import numpy as np
import random
from math import inf


class Particle:
    """
    Represents a particle of the Particle Swarm Optimization algorithm.
    """

    def __init__(self, lower_bound, upper_bound):
        """
        Creates a particle of the Particle Swarm Optimization algorithm.

        :param lower_bound: lower bound of the particle position.
        :type lower_bound: numpy array.
        :param upper_bound: upper bound of the particle position.
        :type upper_bound: numpy array.
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        delta = upper_bound - lower_bound

        self.x = np.random.uniform(lower_bound, upper_bound)
        self.bi = self.x.copy()
        self.bi_fitness = -inf
        self.v = np.random.uniform(-delta, delta)


class ParticleSwarmOptimization:
    """
    Represents the Particle Swarm Optimization algorithm.
    Hyperparameters:
        inertia_weight: inertia weight.
        cognitive_parameter: cognitive parameter.
        social_parameter: social parameter.

    :param hyperparams: hyperparameters used by Particle Swarm Optimization.
    :type hyperparams: Params.
    :param lower_bound: lower bound of particle position.
    :type lower_bound: numpy array.
    :param upper_bound: upper bound of particle position.
    :type upper_bound: numpy array.
    """

    def __init__(self, hyperparams, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.hyperparams = hyperparams

        self.particles = []
        for _ in range(hyperparams.num_particles):
            particle = Particle(lower_bound, upper_bound)
            self.particles.append(particle)

        self.bg = None
        self.bg_fitness = -inf

        self.evaluation = 0

    def get_best_position(self):
        """
        Obtains the best position so far found by the algorithm.

        :return: the best position.
        :rtype: numpy array.
        """
        return self.bg

    def get_best_value(self):
        """
        Obtains the value of the best position so far found by the algorithm.

        :return: value of the best position.
        :rtype: float.
        """
        return self.bg_fitness

    def get_position_to_evaluate(self):
        """
        Obtains a new position to evaluate.

        :return: position to evaluate.
        :rtype: numpy array.
        """
        return self.particles[self.evaluation].x

    def advance_generation(self):
        """
        Advances the generation of particles. Auxiliary method to be used by notify_evaluation().
        """
        w = self.hyperparams.inertia_weight
        phip = self.hyperparams.cognitive_parameter
        phig = self.hyperparams.social_parameter

        for particle in self.particles:
            rp = random.uniform(0.0, 1.0)
            rg = random.uniform(0.0, 1.0)
            particle.v = (
                w * particle.v
                + phip * rp * (particle.bi - particle.x)
                + phig * rg * (self.bg - particle.x)
            )
            vmax = self.upper_bound - self.lower_bound
            particle.v = np.clip(particle.v, -vmax, vmax)
            particle.x += particle.v
            particle.x = np.clip(particle.x, self.lower_bound, self.upper_bound)

    def notify_evaluation(self, value):
        """
        Notifies the algorithm that a particle position evaluation was completed.

        :param value: quality of the particle position.
        :type value: float.
        """

        p = self.particles[self.evaluation]

        if value > p.bi_fitness:
            p.bi_fitness = value
            p.bi = p.x.copy()
        if value > self.bg_fitness:
            self.bg_fitness = value
            self.bg = p.x.copy()

        self.evaluation += 1

        if self.evaluation >= self.hyperparams.num_particles:
            self.advance_generation()
            self.evaluation = 0
