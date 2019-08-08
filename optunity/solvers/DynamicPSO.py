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
# Update solver manual.

import math             # mathematical functions
import numpy            # scientific computing package
import operator as op   # standard operators as functions
import random           # generate pseudo-random numbers
import array            # efficient arrays of numeric values
import functools        # higher-order functions and operations on callable objects
import os

# optunity imports
from .solver_registry import register_solver
from .util import Solver, _copydoc, uniform_in_bounds, loguniform_in_bounds
from . import util
from .Sobol import Sobol
from . import ParticleSwarm

def updateParam(pop_history, num_params=0, func=None, **kwargs):
    """Update/determine objective function parameters according to user-specified function.
    If function is not specified, all parameters are set to 1.
    :param pop_history: [list] list of dynamic particle lists from all previous generations
    :param num_params:  [int] number of objective function parameters
    :param func:        [function] how to update objective function parameters
    :returns:           [list] list of objective function parameters
    """
    if func is not None:
        fparams = func(pop_history, num_params, **kwargs)
        #print("User-specified updateParam() evaluates to", fparams,".")
        return fparams 
    else:
        #print("Default objective function parameters are used.")
        return numpy.ones(num_params)

def evaluateObjFunc(args, params=None, func=None, **kwargs):
    """Calculate scalar fitness according to objective function, given its arguments and parameters.
    If `func` is None, the  scalar product `args[i]*params[i]` is returned.

    :param args:   [vector] (unweighted) arguments of/contributions to objective function
    :param params: [vector] parameters of objective function
    :param func:   [function] function specifying functional form of objective function, i.e.
                   how to combine arguments and parameters to obtain scalar fitness
    :returns:      [float] objective function value (scalar fitness)
    """

    if func is not None and params is not None:
        #print("User-specified combineObj() evaluates to", numpy.around(func(args, params, **kwargs), 2), ".")
        return func(args, params, **kwargs)
    else:
        if params is not None:
            if op.ne(len(args), len(params)):
                raise ValueError("If `combine_obj` is not specified, arguments and parameters vectors need to have same length.")
            return numpy.dot(params, args)
        else:
            return sum(args)

@register_solver('dynamic particle swarm',                                                          # name to register solver with
                 'dynamic particle swarm optimization',                                             # one-line description
                 ['Optimizes the function using a dynamic variant of particle swarm optimization.', # extensive description and manual
                  'Parameters of the objective function are adapted after each generation',
                  'according to the current state of knowledge. To make use of this func-',
                  'tionality, the user has to specify two additional functions `update_param`',
                  'and `combine_obj` as well as the number of arguments and parameters in the',
                  'objective function.'
                  ' ',
                  'This is a two-phase approach:',
                  '1. Initialization: Randomly initialize num_particles particles uniformly',
                  '                   within the box constraints.',
                  '2. Iteration: Particles move during num_generations iterations based on',
                  '              their velocities and mutual attractions derived from',
                  '              individual and global best fitnesses.',
                  ' ',
                  'This function requires the following arguments:',
                  '- num_particles: number of particles to use in the swarm',
                  '- num_generations: number of generations',
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
        """Dynamic particle class."""
        def __init__(self, position, speed, best, fitness, best_fitness, fargs):
            """Construct a dynamic particle.
            :param position: particle position corresponding to hyperparameter combination to be tested
            :param speed: particle speed giving its direction of movement in hyperparameter space
            :param best: best particle position so far (considering current and all previous generations)
            :param fitness: particle fitness (according to its original generation)
            :param best_fitness: (personal) best particle fitness so far (considering current and all previous generations)
            :param fargs: vector of unweighted objective function terms for this particle
            """
            super().__init__(position, speed, best, fitness, best_fitness)
            self.fargs = fargs
        
        def clone(self):
            """Clone this dynamic particle."""
            return DynamicPSO.DynamicParticle(position=self.position[:], speed=self.speed[:],
                                              best=self.best[:], fitness=self.fitness,
                                              best_fitness=self.best_fitness,fargs=self.fargs[:])

    def __init__(self, num_particles, num_generations, max_speed=None, phi1=1.5, phi2=2.0, update_param=None, eval_obj=None, **kwargs):
        """ Initialize a dynamic PSO solver.
        :param num_particles: [int] number of particles in a generation
        :param num_generations: [int] number of generations
        :param max_speed: [float] upper bound for particle velocity
        :param phi1: [float] acceleration coefficient determining impact of each particle's historical best on its movement
        :param phi2: [float] acceleration coefficient determining impact of global best on movement of each particle
        :param update_param: [function] how to determine obj. func. param.s according to current state of knowledge
        :param eval_obj: [function] how to combine obj. func. arg.s and param.s to obtain scalar fitness
        :param **kwargs: box constraints for each hyperparameter as key-worded arguments
        """
        # Check format of bounds given for each hyperparameter.
        assert all([len(v) == 2 and v[0] <= v[1] for v in kwargs.values()]), 'kwargs.values() are not [lb, ub] pairs'
        self._bounds = kwargs                           # len(self.bounds) gives number of hyperparameters considered.
        self._num_particles = num_particles
        self._num_generations = num_generations

        self._sobolseed = random.randint(100,2000)      # random.randint(a,b) gives random integer N with a <= N <= b.

        if max_speed is None: 
            max_speed = 0.7/num_generations
        self._max_speed = max_speed
        # Calculate min. and max. velocities for each hyperparameter considered.
        self._smax = [self.max_speed * (b[1] - b[0]) for _, b in self.bounds.items()]
        # dictionary.items() returns view object displaying (key,value) tuple pair list.
        self._smin = list(map(op.neg, self.smax))       # operator.neg(obj) returns obj negated (-obj).

        self._phi1 = phi1
        self._phi2 = phi2

        self._update_param = update_param
        self._eval_obj = eval_obj

    def generate(self):
        """Generate new dynamic particle."""
        if len(self.bounds) < Sobol.maxdim():                                               # Sobol.maxdim() gives maximum dimensionality supported.
            sobol_vector, self.sobolseed = Sobol.i4_sobol(len(self.bounds), self.sobolseed) # Sobol.i4_sobol generates new quasi-random Sobol vector with each call.
            vector = util.scale_unit_to_bounds(sobol_vector, self.bounds.values())          
            # util.scale_unit_to_bounds(seq, bounds) scales seq elements (unit hypercube) to box constraints in bounds
        else: 
            vector = uniform_in_bounds(self.bounds)
            #vector = loguniform_in_bounds(self.bounds)
            # util.uniform_in_bounds(bounds) generates random uniform sample between `bounds`.

        # array.array(typecode[,initializer]) creates a new array whose items are restricted by typecode and
        # initialized from optional initializer value. 'd' means C doubles, i.e. Python floats.
        
        part = DynamicPSO.DynamicParticle(position=array.array('d', vector),                # random.uniform(a, b) returns a random floating point number N such that
                                      speed=array.array('d', map(random.uniform,            # a <= N <= b for a <= b and vice versa.
                                                                 self.smin, self.smax)),
                                      best=None, fitness=0, best_fitness=0,
                                      fargs=None)
        return part

    def updateParticle(self, part, best, phi1, phi2):
        """Propagate particle, i.e. update its speed and position according to current personal and global best."""
        u1 = (random.uniform(0, phi1) for _ in range(len(part.position)))           # Generate phi1 and phi2 random number coeffiecents
        u2 = (random.uniform(0, phi2) for _ in range(len(part.position)))           # for each hyperparameter
        v_u1 = map(op.mul, u1, map(op.sub, part.best, part.position))               # Calculate phi1 and phi2 velocity contributions.      
        v_u2 = map(op.mul, u2, map(op.sub, best.position, part.position))
        part.speed = array.array('d', map(op.add, part.speed,                       # Add up velocity contributions.
                                          map(op.add, v_u1, v_u2)))
        for i, speed in enumerate(part.speed):                                      # Constrain particle speed to range (smin, smax).
            if speed < self.smin[i]:
                part.speed[i] = self.smin[i]
            elif speed > self.smax[i]:
                part.speed[i] = self.smax[i]
        #print("Old position:", part.position[:])
        #print("Speed:", part.speed[:])
        part.position[:] = array.array('d', map(op.add, part.position, part.speed)) # Add velocity to position to propagate particle.
        #print("New position:", part.position[:])
    # updateParticle is inherited from ParticleSwarm class without changes.
    # => Propagate particle, i.e. update its speed and position according to current personal and global best.
    # particle2dict is inherited from ParticleSwarm class without changes.
    # => Convert particle to dict format {"hyperparameter": particle position}
    
    @_copydoc(Solver.optimize)
    def optimize(self, f, domains, num_args_obj, num_params_obj, maximize=False, pmap=map):  # f is objective function to be optimized.
        """Actual solver implementing dynamic particle swarm optimization.""" 
    # map(function,iterable,...): Return iterator that applies function to every item
    # of iterable, yielding the results. If additional iterable arguments are passed,
    # function must take that many arguments and is applied to the items from all iterables
    # in parallel. With multiple iterables, the interator stops when the shortest iterable
    # is exhausted.
       
        # functools.wraps(wrapped) is a convenience function for invoking update_wrapper() as function decorator when
        # defining a wrapper function. functools.update_wrapper(wrapper, wrapped) updates a wrapper function to look 
        # like the wrapped function.

        @functools.wraps(f)                                     # wrapper function evaluating f
        def evaluate(d):
            """Wrapper function evaluating objective function f accepting a dict {"hyperparameter": particle position}."""
            return f(**d)
        
        if maximize:    # Maximization or minimization problem?
            fit = 1.0   # `optimize` function is a maximizer,
        else:           # i.e. to minimize, maximize -f.
            fit = -1.0
        
        pop = [self.generate() for _ in range(self.num_particles)]  # Randomly generate list of num_particle new particles. 
        pop_history = []                                            # Initialize particle history as list.
        fparams_history = []                                        # Initialize obj. func. param. history as list.
        
        # "_" is common use in python, when the iterator is not needed. E.g., running a list using range(), it is the times range 
        # shows up what matters, not its value. "_" is a normal variable conventionally used to show that its value is irrelevant.
        
        best = None                                                 # Initialize particle storing global best.
        
        # With this loop structure, parameters can only be updated once for each generation and not after each particle iteration.
        # In exchange, calculations of obj. func. contributions (simulations) can be run in parallel within one generation.
        print(self.bounds)
        for g in range(self.num_generations):                                       # Loop over generations.
            print("---------------------------------------------------------")
            print("---------------------------------------------------------")
            print("Generation",repr(int(g)+1))
            print("---------------------------------------------------------")
            print("---------------------------------------------------------")
            Fargs = [ evaluate(self.particle2dict(part)) for part in pop ]          # Calculate obj. func. contributions for all particles in generation.
            for idx, (part, fargs) in enumerate(zip(pop, Fargs)):                   # Set objective function arguments as particle attributes.
                part.fargs = fargs 
            pop_history.append(pop)                                                 # Append current particle generation to history.
            fparams = updateParam(pop_history, num_params_obj, self._update_param)  # Update obj. func. param.s.
            fparams_history.append(fparams)                                         # Append current obj. func. parameter set to history.
            for idg, pops in enumerate(pop_history):                                                # Update fitnesses using most recent obj. param.s.
                assert len(pops) == self.num_particles, "Number of particles in generation does not match `num_particles`."
                print("--------------------\n", "Generation", str(idg+1), "\n--------------------")
                with open("../log.log", "a") as log:
                    log.writelines("--------------------\n" + "Generation " + str(idg+1) + "\n--------------------\n")
                for idx, part in enumerate(pops):
                    fitness = fit * util.score(evaluateObjFunc(part.fargs[:], fparams[:], self._eval_obj))
                    line = "Particle " + repr(int(idx)+1) +" at " + repr(part.position) + " with obj. func. args " + repr(part.fargs) + ", fitness " + repr(fitness)
                    print(line)
                    with open("../log.log", "a") as log:
                        log.writelines(line + "\n")
                    if not part.best or pop[idx].best_fitness < fitness:
                        #print("Personal best is updated in particle history.")
                        pop[idx].best = part.position
                        pop[idx].best_fitness = fitness
                    if not best or best.best_fitness < fitness:
                        #print("Global best is updated.")
                        best = part.clone()
                        #print(self.particle2dict(best))
            for part in pop:
                self.updateParticle(part, best, self.phi1, self.phi2)
            print("Current obj. func. parameters: ", repr(numpy.around(fparams, 2)) + "\n")
            print("Best position so far:", best.position, "with fitness", best.best_fitness)
            with open("../log.log", "a") as log:
                log.writelines("Current obj. func. parameters: " + repr(numpy.around(fparams, 2)) + "\n")
                log.writelines("Best position so far: " + repr(best.position) + " with fitness " + repr(best.best_fitness) + "\n--------------------\n")
        #print(fparams_history)
        return dict([(k, v) for k, v in zip(self.bounds.keys(), best.position)]), None # Return best position for each hyperparameter.
