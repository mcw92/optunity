#! /usr/bin/env python

# Copyright (c) 2014 KU Leuven, ESAT-STADIUS
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# 3. Neither name of copyright holders nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math
import numpy as np
import operator as op
import random
import array
import functools

from .solver_registry import register_solver
from .util import Solver, _copydoc, uniform_in_bounds
from . import util
from .Sobol import Sobol
from . import ParticleSwarm                                                             # import normal PSO from optunity
# Classes required for dynamic PSO can then inherit from base classes implemented in normal PSO.

@register_solver('dynamic particle swarm',                                                          # name to register solver with
                 'dynamic particle swarm optimization',                                             # one-line description of solver
                 ['Optimizes the function using a dynamic variant of particle swarm optimization.', # extensive description and manual of solver
                  'Parameters of the objective function are adapted after each iteration',
                  'according to the current state of knowledge.',
                  ' ',
                  'This is a two-phase approach:',
                  '1. Initialization: Randomly initialize num_particles particles.',
                  '   Particles are randomized uniformly within the box constraints.',
                  '2. Iteration: Particles move during num_generations iterations.',
                  '   Movement is based on their velocities and mutual attractions.',
                  ' ',
                  'This function requires the following arguments:',
                  '- num_particles: number of particles to use in the swarm',
                  '- num_generations: number of iterations used by the swarm',
                  '- max_speed: maximum speed of the particles in each direction (in (0, 1])',
                  '- box constraints via key words: constraints are lists [lb, ub]', ' ',
                  'This solver performs num_particles*num_generations function evaluations.'
                  ])

class DynamicPSO(ParticleSwarm)
    """
    Dynamic particle swarm optimization solver class.
    """

    class DynamicParticle(ParticleSwarm.Particle):
        def __init__(self, position, speed, best, fitness, best_fitness, fargs, fparams):
            """Construct a dynamic particle"""
            super().__init__(position, speed, best, fitness, best_fitness)
            """
            fargs is a vector containing different unweighted terms of objective function.
            fparams is a vector containing different parameters of objective function.
            """
            self.fargs = fargs
            self.fparams = fparams
        
        def clone(self):
        """Clone this dynamic particle."""
            return DynamicPSO.DynamicParticle(position=self.position[:], speed=self.speed[:],
                                              best=self.best[:], fitness=self.fitness,
                                              best_fitness=self.best_fitness,fargs=self.fargs[:],
                                              self.fparams[:])
    """    
    The DynamicPSO class definition does not have an .__init__() because it inherits from ParticleSwarm and
    .__init__() does not really do anything differently for DynamicPSO than it already does for ParticleSwarm.
    This is why one can skip defining it and the .__init() of the superclass will be called automatically.
    def __init__(self, num_particles, num_generations, max_speed=None, phi1=1.5, phi2=2.0, **kwargs):
        super().__init__(self, num_particles, num_generations, max_speed=None, phi1=1.5, phi2=2.0, **kwargs)
    """
    def generate(self):
        """Generate a new Particle."""
        if len(self.bounds) < Sobol.maxdim():
            sobol_vector, self.sobolseed = Sobol.i4_sobol(len(self.bounds), self.sobolseed)
            vector = util.scale_unit_to_bounds(sobol_vector, self.bounds.values())
        else: vector = uniform_in_bounds(self.bounds)

        part = ParticleSwarm.Particle(position=array.array('d', vector),
                                      speed=array.array('d', map(random.uniform,
                                                                 self.smin, self.smax)),
                                      best=None, fitness=None, best_fitness=None)
        return part

    def updateParticle(self, part, best, phi1, phi2):
        """Update the particle."""
        u1 = (random.uniform(0, phi1) for _ in range(len(part.position)))
        u2 = (random.uniform(0, phi2) for _ in range(len(part.position)))
        v_u1 = map(op.mul, u1,
                    map(op.sub, part.best, part.position))
        v_u2 = map(op.mul, u2,
                    map(op.sub, best.position, part.position))
        part.speed = array.array('d', map(op.add, part.speed,
                                          map(op.add, v_u1, v_u2)))
        for i, speed in enumerate(part.speed):
            if speed < self.smin[i]:
                part.speed[i] = self.smin[i]
            elif speed > self.smax[i]:
                part.speed[i] = self.smax[i]
        part.position[:] = array.array('d', map(op.add, part.position, part.speed))

    def particle2dict(self, particle):                          # Convert particle to dict format {"hyperparameter": particle_position}.
        return dict([(k, v) for k, v in zip(self.bounds.keys(), # self.bound.keys() returns hyperparameter names.
                                            particle.position)])

    @_copydoc(Solver.optimize)
    def optimize(self, f, maximize=True, pmap=map):             # f is objective function to be optimized.
        
        # map(function,iterable,...): Return an iterator that applies function to every item
        # of iterable, yielding the results. If additional iterable arguments are passed,
        # function must take that many arguments and is applied to the items from all iterables
        # in parallel. With multiple iterables, the interator stops when the shortest iterable
        # is exhausted.

        @functools.wraps(f)                                     # wrapper function evaluating f
        def evaluate(d):
            return f(**d)

        # Determine whether optimization problem is maximization or minimization problem.
        # The 'optimize' function is a maximizer, so if we want to minimze, we basically
        # maximize -f.

        if maximize:
            fit = 1.0
        else:
            fit = -1.0

        pop = [self.generate() for _ in range(self.num_particles)]          # Randomly generate list of num_particle new particles. 
        # "_" is common usage in python, meaning that the iterator is not needed. Like, running a list of int,
        # using range, what matters is the times the range shows not its value. It is just another variable,
        # but conventionally used to show that one does not care about its value.
        
        # ? HOW TO GET IN NUM_ARGS = number of numerical inputs of objective function and NUM_PARAMS = number of parameters (= weights) in objective function?
        # Initialize PSO history as numpy array. 
        # Particle history: Each line gives data from one iteration and corresponds to a particular particle in a
        # particular generation, i.e. a specific combination of hyperparameters tested.
        # generation index | particle index | parameters tested | numerical inputs of objective function    <= These quantities are determined once for each iteration.
        # current score | current personal best | current global best                        <= These quantities have to be updated after each iteration.
        particle_history = np.empty(self.num_generations*self.num_particles,len(self.bounds+NUM_ARGS+NUM_PARAMS+3))
        weights_history = np.empty(self.num_generations*self.num_particles, NUM_PARAMS)
        best = None                                                         # Initialize particle storing global best.
        
        for g in range(self.num_generations):                               # Loop over generations.
            fitnesses = pmap(evaluate, list(map(self.particle2dict, pop)))  # Evaluate fitnesses for all particles in current generation.
            for part, fitness in zip(pop, fitnesses):                       # Loop over pairs of particles and individual fitnesses.
                part.fitness = fit * util.score(fitness)                    # util.score: wrapper around objective function evaluations to get score.
                if not part.best or part.best_fitness < part.fitness:       # Update personal best if required.
                    part.best = part.position
                    part.best_fitness = part.fitness
                if not best or best.fitness < part.fitness:                 # Update global best if required.
                    best = part.clone()
            for part in pop:                                                # Update particle for next generation loop.
                self.updateParticle(part, best, self.phi1, self.phi2)

        return dict([(k, v)                                                 # Return best position for each hyperparameter.
                        for k, v in zip(self.bounds.keys(), best.position)]), None
