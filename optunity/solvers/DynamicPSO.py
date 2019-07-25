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

#########
# TO DO #
#########
# Check which implementation in dynamic PSO loop is faster.
# Implement default functions for determine_params and combine_obj.
# Implement dynamic PSO loop.

import math
import numpy
import operator as op
import random
import array
import functools    # higher-order functions and operations on callable objects

from .solver_registry import register_solver
from .util import Solver, _copydoc, uniform_in_bounds
from . import util
from .Sobol import Sobol
from . import ParticleSwarm # import normal PSO from optunity, dynamic PSO classes can then inherit from ParticleSwarm classes.

def updateParam(pop_history, num_args=1, num_params=0, func=None, **kwargs):
    """Update/determine objective function parameters."""
    #if func is not None:
    #    return func(pop_history)
    #else:
    #    return numpy.ones(num_params)
    return numpy.ones(num_params)
#    Update weights according to function func specified by user. If no function is specified, nothing will happen.
#    else:                                           # Otherwise, adapt/update weights according to specified function func
#        adapted_weights = func(*args, **kwargs)
#        return adapted_weights

def evaluateObjFunc(args, params=None, func=None, **kwargs):
    """Calculate scalar fitness according to objective function, given its arguments and parameters.
    :param args:   [vector] (unweighted) arguments of objective function
    :param params: [vector] parameters of objective function
    :param func:   [function] function specifying functional form of objective function, i.e.
                   how to combine arguments and parameters to obtain scalar fitness
    :returns:      objective function value, i.e. scalar fitness
    """

    if func is not None and params is not None:
        return func(args, params)
    else:
        if params is not None:
            assert len(args) == len(params), "If `combine_obj` is not specified, arguments and parameters vectors need to have same length."
            return sum([ param*arg for param, arg in zip(params, args) ])
        else:
            return sum(args)

@register_solver('dynamic particle swarm',                                                          # name to register solver with
                 'dynamic particle swarm optimization',                                             # one-line description
                 ['Optimizes the function using a dynamic variant of particle swarm optimization.', # extensive description and manual
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
                  '- update_param: function specifying how to determine parameters according',
                  '                to current state of knowledge',
                  '- eval_obj: function specifying how to combine unweighted contributions',
                  '            and parameters of objective function to obtain scalar fitness',
                  '- box constraints via key words: constraints are lists [lb, ub]', ' ',
                  'This solver performs num_particles*num_generations function evaluations.'
                  ])

class DynamicPSO(ParticleSwarm):
    """Dynamic particle swarm optimization solver class."""
    class DynamicParticle(ParticleSwarm.Particle):
        def __init__(self, position, speed, best, fitness, best_fitness, fargs):
            """Construct a dynamic particle.
            :param position: current particle position corresponding to hyperparameter combination to be tested
            :param speed: current particle speed giving its direction of movement in hyperparameter space
            :param best: best position of this particle so far (considering current and all previous generations)
            :param fitness: current particle fitness (in generation considered)
            :param best_fitness: best fitness of this particle so far (considering current and all previous generations)
            :param fargs: vector containing different unweighted terms of objective function for this particle
            """
            super().__init__(position, speed, best, fitness, best_fitness)
            self.fargs = fargs
        
        def clone(self):
            """Clone this dynamic particle."""
            return DynamicPSO.DynamicParticle(position=self.position[:], speed=self.speed[:],
                                              best=self.best[:], fitness=self.fitness,
                                              best_fitness=self.best_fitness,fargs=self.fargs[:])

    def __init__(self, num_particles, num_generations, max_speed=None, phi1=1.5, phi2=2.0, update_param=None, eval_obj=None, **kwargs):
        """ Initialize a dynamic PSO solver."""
        assert all([len(v) == 2 and v[0] <= v[1]        # Check format of bounds given for each hyperparameter.
            for v in kwargs.values()]), 'kwargs.values() are not [lb, ub] pairs'
        self._bounds = kwargs                           # len(self.bounds) gives number of hyperparameters considered.
        self._num_particles = num_particles
        self._num_generations = num_generations

        self._sobolseed = random.randint(100,2000)      # random.randint(a,b) returns random integer N such that a <= N <= b.

        if max_speed is None: max_speed = 0.7 / num_generations
        self._max_speed = max_speed
        self._smax = [self.max_speed * (b[1] - b[0]) for _, b in self.bounds.items()]
        # dictionary.items() returns view object displaying (key,value) tuple pair list.

        self._smin = list(map(op.neg, self.smax))       # operator.neg(obj) returns obj negated (-obj).

        self._phi1 = phi1
        self._phi2 = phi2

        #super().__init__(num_particles, num_generations, max_speed, phi1, phi2, **kwargs)
        if update_param is None: 
            self._update_param = updateParam
        else:
            self._update_param = update_param
        if eval_obj is None: 
            self._eval_obj = evaluateObjFunc
        else:
            self._eval_obj = eval_obj

    def generate(self):
        """Generate new particle."""
        if len(self.bounds) < Sobol.maxdim():
            sobol_vector, self.sobolseed = Sobol.i4_sobol(len(self.bounds), self.sobolseed)
            vector = util.scale_unit_to_bounds(sobol_vector, self.bounds.values())
        else: vector = uniform_in_bounds(self.bounds)
        
        # array.array(typecode[,initializer]) creates a new array whose items are restricted by typecode and
        # initialized from optional initializer value. 'd' means C doubles, i.e. Python floats.
        
        part = DynamicPSO.DynamicParticle(position=array.array('d', vector),
                                      speed=array.array('d', map(random.uniform,
                                                                 self.smin, self.smax)),
                                      best=None, fitness=0, best_fitness=0,
                                      fargs=None)
        return part

    # updateParticle is inherited from ParticleSwarm class without changes.
    # Propagate particle, i.e. update its speed and position according to current personal and global best.
    # particle2dict is inherited from ParticleSwarm class without changes.
    # Convert particle to dict format {"hyperparameter": particle position}
    
    @_copydoc(Solver.optimize)
    def optimize(self, f, num_args_obj, num_params_obj, maximize=False, pmap=map):  # f is objective function to be optimized.
    
    # map(function,iterable,...): Return an iterator that applies function to every item
    # of iterable, yielding the results. If additional iterable arguments are passed,
    # function must take that many arguments and is applied to the items from all iterables
    # in parallel. With multiple iterables, the interator stops when the shortest iterable
    # is exhausted.
       
        # functools.wraps(wrapped) is a convenience function for invoking update_wrapper() as a function decorator when
        # defining a wrapper function. functools.update_wrapper(wrapper, wrapped) updates a wrapper function to look 
        # like the wrapped function.

        @functools.wraps(f)                                     # wrapper function evaluating f
        def evaluate(d):
            """Wrapper function evaluating objective function f accepting a dict {"hyperparameter": particle position}."""
            return f(**d)
        
        # Maximization or minimization problem?
        # 'optimize' function is a maximizer, i.e. to minimze, maximize -f.
        
        if maximize:
            fit = 1.0
        else:
            fit = -1.0

        pop = [self.generate() for _ in range(self.num_particles)]  # Randomly generate list of num_particle new particles. 
        pop_history = []                                            # Initialize particle history as list.
        fparams_history = []                                        # Initialize obj. func. param. history as list.
        
        # "_" is common use in python, when the iterator is not needed. E.g., running a list using range(), it is the times range 
        # shows up what matters, not its value. "_" is a normal variable conventionally used to show that its value is irrelevant.
        
        best = None                                                 # Initialize particle storing global best.
        
        # With this loop structure, parameters can only be updated once for each generation and not after each particle iteration.
        # In exchange, calculations of obj. func. contributions (simulations) can be run in parallel within one generation.
        
        for g in range(self.num_generations):                               # Loop over generations.
            print("Generation",repr(int(g)+1))
            Fargs = [ evaluate(self.particle2dict(part)) for part in pop ]  # Calculate obj. func. contributions for all particles in generation.
            #Fargs = pmap(evaluate, list(map(self.particle2dict, pop)))     # alternativel calculation method
            for idx, (part, fargs) in enumerate(zip(pop, Fargs)):                             # Set objective function arguments as particle attributes.
                part.fargs = fargs 
                print("Particle", repr(int(idx)+1),": position:", repr(numpy.around(part.position,2)),", arguments:", repr(numpy.around(numpy.asarray(part.fargs),2)))
            pop_history.append(pop)                                 # Append current particle generation to history.
            # Calculate obj. func. param.s according to current state of knowledge.
            fparams = updateParam(pop_history, num_args=num_args_obj, num_params=num_params_obj, func=self._update_param)  
            print("Parameters: ", repr(fparams))
            fparams_history.append(fparams)                         # Append current obj. func. parameter set to history.
            for pops in pop_history:                                # Update fitnesses using most recent obj. param.s.
                #for part_curr, part in zip(pop, pops):
                for idx, part in zip(range(int(self.num_particles)), pops):
                    #print(repr(part.fargs[:]))
                    fitness = fit * util.score(evaluateObjFunc(args=part.fargs[:], params=fparams[:], func=self._eval_obj))
                    if not part.best or pop[idx].best_fitness < fitness:
                        pop[idx].best = part.position
                        pop[idx].best_fitness = fitness
                    #if not part.best or part_curr.best_fitness < fitness:
                        #part_curr.best = part.position
                        #part_curr.best_fitness = fitness
                    if not best or best.fitness < fitness:
                        best = part.clone()
            for part in pop:
                self.updateParticle(part, best, self.phi1, self.phi2)
        print(fparams_history)
        print("Particle history: ", repr(pop_history))
        print(len(pop_history))
        return dict([(k, v) for k, v in zip(self.bounds.keys(), best.position)]), None # Return best position for each hyperparameter.
