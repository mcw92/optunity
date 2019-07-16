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

#def determine_params(func=None, *args, **kwargs):
#    """
#    Update weights according to function func specified by user. If no function is specified, nothing will happen.
#    """
#    if func==None: pass                             # If no function to adapt weights is specified by the user, do nothing.
#    else:                                           # Otherwise, adapt/update weights according to specified function func
#        adapted_weights = func(*args, **kwargs)
#        return adapted_weights
#
#def combine_obj(part, func, curr_weights):
#    """
#    Re-evaluate fitness for all particles in history using most recent weights according to current state of knowledge.
#    :param part: particle whose score is to be updated
#    :param func: function specifying mathematical form of objective function, i.e. how terms and weights are to be combined to yield score
#    :param curr_weights: current set of weights to be used in evaluation
#    """
#    part.fitness = func(part.fargs, curr_weights)
#    if part.fitness < part.best_fitness: part.best_fitness = part.fitness # Update personal best if required.
#    '''I have to compare each historical particle's best fitness with the particle's personal best fitness from the current generation.'''
#    pass

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

class DynamicPSO(ParticleSwarm):
    """
    Dynamic particle swarm optimization solver class.
    """

    class DynamicParticle(ParticleSwarm.Particle):
        def __init__(self, position, speed, best, fitness, best_fitness, fargs):
            """Construct a dynamic particle"""
            super().__init__(position, speed, best, fitness, best_fitness)
            """
            :param position: current particle position corresponding to hyperparameter combination to be tested
            :param speed: current particle speed giving its direction of movement in hyperparameter space
            :param best: best position of this particle so far (i.e. considering current and all previous generations)
            :param fitness: current particle fitness (i.e. in generation considered)
            :param best_fitness: best fitness of this particle so far (i.e. considering current and all previous generations)
            :param fargs: vector containing different unweighted terms of objective function for this particle
            :param fparams: vector containing different parameters of objective function for this particle
            """
            self.fargs = fargs
        
        def clone(self):
            """Clone this dynamic particle."""
            return DynamicPSO.DynamicParticle(position=self.position[:], speed=self.speed[:],
                                              best=self.best[:], fitness=self.fitness,
                                              best_fitness=self.best_fitness,fargs=self.fargs[:])
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
        
        """
        array.array(typecode[,initializer]) creates a new array whose items are restricted by typecode and
        initialized from optional initializer value. 'd' means C doubles, i.e. Python floats.
        """
        part = DynamicPSO.DynamicParticle(position=array.array('d', vector),
                                      speed=array.array('d', map(random.uniform,
                                                                 self.smin, self.smax)),
                                      best=None, fitness=None, best_fitness=None,
                                      fargs=None, fparams=None)
        return part

    def updateParticle(self, part, best, phi1, phi2):
        """Propagate the particle, i.e. update its speed and position according to current personal and global best."""
        u1 = (random.uniform(0, phi1) for _ in range(len(part.position)))
        u2 = (random.uniform(0, phi2) for _ in range(len(part.position)))
        v_u1 = map(op.mul, u1,                                              # operator.mul(a,b) returns a*b for numbers a and b.
                    map(op.sub, part.best, part.position))                  # operator.sub(a,b) returns a-b.
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

    """
    particle2dict is inherited from ParticleSwarm parent class without changes.
    Convert particle to dict format {"hyperparameter": particle position}
    """
    
    @_copydoc(Solver.optimize)
    def optimize(self, f, maximize=False, pmap=map):             # f is objective function to be optimized.
        """
        map(function,iterable,...): Return an iterator that applies function to every item
        of iterable, yielding the results. If additional iterable arguments are passed,
        function must take that many arguments and is applied to the items from all iterables
        in parallel. With multiple iterables, the interator stops when the shortest iterable
        is exhausted.
        """
        @functools.wraps(f)                                     # wrapper function evaluating f
        def evaluate(d):
            """
            wrapper function evaluating objective function f accepting a dict {"hyperparameter": particle position}
            """
            return f(**d)
        
        #Determine whether optimization problem is maximization or minimization problem.
        #The 'optimize' function is a maximizer, i.e. to minimze, basically maximize -f.
        
        if maximize:
            fit = 1.0
        else:
            fit = -1.0

        pop = [self.generate() for _ in range(self.num_particles)]          # Randomly generate list of num_particle new particles. 
        pop_history = []                                                    # Initialize particle history as list.
        fparams_history = []                                                # Initialize obj. func. param. history as list.
        """
        "_" is common use in python, when the iterator is not needed. E.g., running a list using range(), it is the times range 
        shows up what matters, not its value. "_" is a normal variable conventionally used to show that its value is irrelevant.
        """
        best = None                                                         # Initialize particle storing global best.
        """
        !!! With this loop structure, weights can only be updated once for each generation and not after each particle iteration.
        Otherwise, objective function evaluations (simulations) could not be run in parallel for particles from one generation.
        """
        for g in range(self.num_generations):                               # Loop over generations.
            Fargs = [my_loss_function(self.particle2dict(particle) for particle in pop]
            for part, fargs in zip(pop, Fargs):
                part.fargs = fargs 
            """
            for part in pop:                                                # Evaluate unweighted terms for all particles in current generation.
                part.fargs = my_loss_function(self.particle2dict)           # Here: my_loss_function => evaluate/f?
            """
            pop_history.append(pop)                                         # Append current particle generation to history.
            fparams = determine_params(self.update_param, pop_history)
            fparams_history.append(fparams)
            for pops in pop_history:
                for part in pops:
                    part.fitness = fit * combine_obj(part.fargs, fparams)
                    if not part.best or part.best_fitness < part.fitness:
                        part.best = part.position
                        part.best_fitness = part.fitness
                    if not best or best.fitness < part.fitness:
                        best = part.clone()
            for part in pop:
                self.updateParticle(part, best, self.phi1, self.phi2)
         return dict([(k, v)                                                 # Return best position for each hyperparameter.
                         for k, v in zip(self.bounds.keys(), best.position)]), None           
            """
            #flat_pop_history = [part for pop in pop_history for part in pop]# Flatten particle history list.
            fitnesses = pmap(evaluate, list(map(self.particle2dict, pop)))  # Evaluate fitnesses for all particles in current generation.
            for part, fitness in zip(pop, fitnesses):                       # Loop over pairs of particles and individual fitnesses.
                part.fitness = fit * util.score(fitness)                    # util.score: wrapper around objective function evaluations to get score.
        """
