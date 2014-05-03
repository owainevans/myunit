import time
import random
import numpy as np
from venture.ripl.ripl import _strip_types
from venture.venturemagics.ip_parallel import *
from venture.shortcuts import make_puma_church_prime_ripl
from history import History, Run, Series

## FIXME eliminatino
execfile('/home/owainevans/myunit/unit/history.py')


# ASANA

## GET SEED GET SEED!!!


# [assume x (binomial 1 .00001)]
# [observe (poisson x) 1]
# [infer 1]
# --core dump in puma

# v.assume('dir_mult','(make_dir_mult (array 1 1))')
# == 'unknown'
# -- should be something like (see sym_dir_mult value)
# {'simplex': [1.0,1.0], 'counts': [0, 0],  'type': 'dir_mult'}


# v.assume('sym_dir','(make_sym_dir_mult 1 2)')
# v.observe('(sym_dir)','atom<0>')
# v.list_directives()[-1]
# {'directive_id': 17,
#  'expression': ['s'],
#  'instruction': 'observe',
#  'value': 0.0}

# v.assume('atom_or_num','(lambda ()(if (flip theta) atom<0> 0))')
# v.observe('(atom_or_num)','0')
# v.infer(1)
# RuntimeError: Cannot constrain a deterministic value.
# --not sure if there's a way to observe an atom noisily. want sp
#   where input and output are atoms. 

# --can't recover fact that value we conditioned on was atom


# # analytics vs. old unit
# 1. why don't we get exactly same results? (what's up with seeds)
# 2. add ability to query ripl with arbitrary function

# 1. get rid of sweep notion. take string or parsed thing or procedure that takes a ripl (which
# could implement count assumes and observes or count current number of random choices).
# 2. scale parameter and so on can live in sweeps. 



## TODO
# when we generate data from prior, want to store both 
# the data, all assumes symbols and all query_exps. (and if we plot it, it should be 
# indicated as the true value). verbose should give print out
# 

# add method to history: nameValues (gives all values)
# add method to history: nameSnapshot (gives snapshot at end of inference)
# for mripl, need good snaphshot overtime plots (bayesDB?)

# add method for viewing observes and expressions.

# plots: can't plot 100 runs. need to select some. should 
# if you have enough runs (so variance in your estimates
# of the within and between chain variances is low), can annotate at probe points
# with gelman-rubin variances. 


# todo:

def directive_split(d):
    'Splits directive in list_directives form to components'
    ## FIXME: replace symbols, calls build_exp from ip_para
    if d['instruction']=='assume':
        return (d['symbol'], build_exp(d['expression']) ) 
    elif d['instruction']=='observe':
        return (build_exp(d['expression']), d['value']) 
    elif d['instruction']=='predict':
        return build_exp(d['expression'])

# whether to record a value returned from the ripl
def record(value):
    return value['type'] in {'boolean', 'real', 'number', 'atom', 'count', 'array', 'simplex'}

parseValue = _strip_types

class VentureUnit(object):
    '''class BatchExperiment, for running analytics on all parameter
    settings specified by a parameter dict'''
    ripl = None
    parameters = {}
    assumes = []
    observes = []

    # Register an assume.
    def assume(self, symbol, expression):
        self.assumes.append((symbol, expression))

    # Override to create generative model.
    def makeAssumes(self): pass

    # Register an observe.
    def observe(self, expression, literal):
        self.observes.append((expression, literal))

    # Override to constrain model on data.
    def makeObserves(self): pass

    # Masquerade as a ripl.
    def clear(self):
        self.assumes = []
        self.observes = []

    # Initializes parameters, generates the model, and prepares the ripl.
    def __init__(self, ripl, parameters=None):
        if parameters is None: parameters = {}
        self.ripl = ripl

        # FIXME: Should the random seed be stored, or re-initialized?
        self.parameters = parameters.copy()
        if 'venture_random_seed' not in self.parameters:
            self.parameters['venture_random_seed'] = self.ripl.get_seed()
        else:
            self.ripl.set_seed(self.parameters['venture_random_seed'])

        # FIXME: automatically assume parameters (and omit them from history)?
        self.assumes = []
        self.makeAssumes()

        self.observes = []
        self.makeObserves()
        
        self.analyticsArgs = (self.ripl,)
        self.analyticsKwargs = dict(assumes=self.assumes, observes=self.observes,
                           parameters=self.parameters)

    def nameObserve(self,index):
        pass

    def sampleFromJoint(self,*args,**kwargs):
        a = Analytics(*self.analyticsArgs, **self.analyticsKwargs)
        return a.sampleFromJoint(*args,**kwargs)

    def runFromJoint(self,*args,**kwargs):
        a = Analytics(*self.analyticsArgs, **self.analyticsKwargs)
        return a.runFromJoint(*args,**kwargs)
    
    def computeJointKL(self,*args,**kwargs):
        a = Analytics(*self.analyticsArgs, **self.analyticsKwargs)
        return a.computeJointKL(*args,**kwargs)

    def runFromConditional(self, *args, **kwargs):
        a = Analytics(*self.analyticsArgs, **self.analyticsKwargs)
        return a.runFromConditional(*args,**kwargs)

    def runConditionedFromPrior(self, *args, **kwargs):
        a = Analytics(*self.analyticsArgs, **self.analyticsKwargs)
        return a.runConditionedFromPrior(*args,**kwargs)


class Analytics(object):


    def __init__(self, ripl, assumes=None,observes=None,queryExps=None,
                 parameters=None):
        '''Methods for analyzing and debugging inference on a model.

        Arguments
        ---------
        ripl :: ripl (Puma or Lite)
            Inference is done on a fresh ripl with same backend. If no
            *assumes* are specified, assumes are extracted from ripl.
        
        assumes :: [(sym,exp)]
            List of assume pairs. If not None, replace ripl assumes.

        observes  :: [(exp,literal)]
            List of observe pairs. Values are used by *runFromConditional* as
            data. Expressions are used by *geweke* and *runConditionedFromPrior*.

        queryExps :: [exp]
            List of expressions which are evaluated and recorded at every sweep
            of inference (in addition to symbols in assumes).

        parameters :: {string: a}
            FIXME explain'''
 
        assert not(assumes is None and observes is not None),'No *observes* without *assumes*.'

        self.ripl = ripl
        directives_list = self.ripl.list_directives()
        
        if assumes is not None:
            self.ripl.clear()
            self.assumes = assumes
            self.observes = observes if observes is not None else []
        else:
            assumes = [d for d in directives_list if d['instruction']=='assume']
            self.assumes = map(directive_split,assumes)
            observes = [d for d in directives_list if d['instruction']=='observe']
            self.observes = map(directive_split,observes)

        self.queryExps=[] if queryExps is None else queryExps

        if parameters is None: parameters = {}
        
        # FIXME: Should the random seed be stored, or re-initialized?
        self.parameters = parameters.copy()
        if 'venture_random_seed' not in self.parameters:
            self.parameters['venture_random_seed'] = self.ripl.get_seed()
            print 'Ripl seed was set to zero due to ripl.get_seed() bug'
        else:
            self.ripl.set_seed(self.parameters['venture_random_seed'])


    def updateObserves(self,newObserves=None,removeAllObserves=False):
        '''Extend list of observes or empty it.
           Input: newObserves :: [(exp,literal)], removeAllObserves :: bool.'''
        if removeAllObserves:
            self.observes = []
        if newObserves is not None:
            self.observes.extend( newObserves )
        #self.observes.extend( map(directive_split,newObserves) )
        
    def updateQueryExps(self,newQueryExps=None,removeAllQueryExps=False):
        '''Extend list of query expressions or empty it.
           Input: newQueryExps :: [(exp)], removeAllQueryExps :: bool.'''
        if removeAllQueryExps:
            self.queryExps = []
        if newQueryExps is not None:
            self.queryExps.extend( newQueryExps )
    

    def _loadAssumes(self, prune=True):
        
        assumeToDirective = {}
        # since we extracted from ripl, exception would only
        # be caused by bug in extraction or run-time type error
        for (symbol, expression) in self.assumes:
            from venture.exception import VentureException
            try:
                value = self.ripl.assume(symbol, expression, label=symbol, type=True)
            except VentureException as e:
                print expression
                raise e
            if (not prune) or record(value):
                assumeToDirective[symbol] = symbol
        return assumeToDirective


    def _assumesFromRipl(self):
        'For assumes with values allowed by *record*, return {sym:did}'
        assumeToDirective = {}
        for directive in self.ripl.list_directives(type=True):
            ## FIXME record used here to filter certain values
            if directive["instruction"] == "assume" and record(directive["value"]):
                assumeToDirective[directive["symbol"]] = directive["directive_id"]
        return assumeToDirective

    def _loadObservesAsPredicts(self, track=-1, prune=True):
        predictToDirective = {}
        for (index, (expression, _)) in enumerate(self.observes):
            #print("self.ripl.predict('%s', label='%d')" % (expression, index))
            label = 'observe_%d' % index
            value = self.ripl.predict(expression, label=label, type=True)
            if (not prune) or record(value):
                predictToDirective[index] = label

        # choose a random subset to track; by default all are tracked
        if track >= 0:
            track = min(track, len(predictToDirective))
            # FIXME: need predictable behavior from RNG
            random.seed(self.parameters['venture_random_seed'])
            predictToDirective = dict(random.sample(predictToDirective.items(), track))

        return predictToDirective

    def _loadObserves(self, data=None):
        'If not data, then use observes'
        for (index, (expression, literal)) in enumerate(self.observes):
            datum = literal if data is None else data[index]
            self.ripl.observe(expression, datum)

    # Loads the assumes and changes the observes to predicts.
    # Also picks a subset of the predicts to track (by default all are tracked).
    # Prunes non-scalar values, unless prune=False.
    # Does not reset engine RNG.
    def loadModelWithPredicts(self, track=-1, prune=True):
        self.ripl.clear()

        assumeToDirective = self._loadAssumes(prune=prune)
        predictToDirective = self._loadObservesAsPredicts(track=track, prune=prune)

        return (assumeToDirective, predictToDirective)

    # Updates recorded values after an iteration of the ripl.
    def updateValues(self, keyedValues, keyToDirective=None):
        for (key, values) in keyedValues.items():

            if keyToDirective is None: # queryExps are sampled and have no dids
                value = self.ripl.sample(key,type=True)
                #value = f(self.ripl) --add ability to query with function
            else:
                if key not in keyToDirective: # we aren't interested in this series
                    del keyedValues[key]
                    continue

                value = self.ripl.report(keyToDirective[key], type=True)
                
            if len(values) > 0:
                if value['type'] == values[0]['type']:
                    values.append(value)
                else: # directive has returned a different type; discard the series
                    del keyedValues[key]
            elif record(value):
                values.append(value)
            else: # directive has returned a non-scalar type; discard the series
                del keyedValues[key]

    # Gives a name to an observe directive.
    def nameObserve(self, index):
        return 'observe[' + str(index) + '] ' + self.observes[index][0]

    # Provides independent samples from the joint distribution (observes turned into predicts).
    # A random subset of the predicts are tracked along with the assumed variables.
    # Returns a History object that always represents exactly one Run.
    def sampleFromJoint(self, samples, track=5, verbose=False, name=None):
        assumedValues = {symbol:  [] for (symbol, _) in self.assumes}
        predictedValues = {index: [] for index in range(len(self.observes))}
        queryExpsValues = {exp: [] for exp in self.queryExps}

        logscores = []

        for i in range(samples):
            if verbose:
                print "Generating sample " + str(i) + " of " + str(samples)

            (assumeToDirective, predictToDirective) = self.loadModelWithPredicts(track)

            logscores.append(self.ripl.get_global_logscore())

            self.updateValues(assumedValues,keyToDirective=assumeToDirective)
            self.updateValues(predictedValues,keyToDirective=predictToDirective)
            self.updateValues(queryExpsValues, keyToDirective=None)

        tag = 'sample_from_joint' if name is None else name + '_sample_from_joint'
        history = History(tag, self.parameters)

        history.addSeries('logscore', 'i.i.d.', logscores)

        for (symbol, values) in assumedValues.iteritems():
            history.addSeries(symbol, 'i.i.d.', map(parseValue, values))

        for (index, values) in predictedValues.iteritems():
            history.addSeries(self.nameObserve(index), 'i.i.d.',
                              map(parseValue, values))

        for (exp, values) in queryExpsValues.iteritems():
            history.addSeries(exp, 'i.i.d.', map(parseValue, values))

        return history

    # iterates until (approximately) all random choices have been resampled
    def sweep(self,infer=None):
        iterations = 0

        #FIXME: use a profiler method here
        get_entropy_info = self.ripl.sivm.core_sivm.engine.get_entropy_info

        while iterations < get_entropy_info()['unconstrained_random_choices']:
            step = get_entropy_info()['unconstrained_random_choices']
            if infer is None:
                self.ripl.infer(step)
            # TODO Incoming infer string or procedure may touch more
            # than "step" choices; how to count sweeps right?
            elif isinstance(infer, str):
                self.ripl.infer(infer)
            else:
                infer(self.ripl, step)
            iterations += step

        return iterations
    
    # TODO: run in parallel?
    def _runRepeatedly(self, f, tag, runs=3, verbose=False, profile=False, **kwargs):
        history = History(tag, self.parameters)

        for run in range(runs):
            if verbose:
                print "Starting run " + str(run) + " of " + str(runs)
            res = f(label="run %s" % run, verbose=verbose, **kwargs)
            history.addRun(res)

        if profile:
            history.profile = Profile(self.ripl)
        return history

    # Runs inference on the joint distribution (observes turned into predicts).
    # A random subset of the predicts are tracked along with the assumed variables.
    # If profiling is enabled, information about random choices is recorded.
    def runFromJoint(self, sweeps, name=None, **kwargs):
        tag = 'run_from_joint' if name is None else name + '_run_from_joint'
        return self._runRepeatedly(self.runFromJointOnce, tag, sweeps=sweeps, **kwargs)

    def runFromJointOnce(self, track=5, **kwargs):
        (assumeToDirective, predictToDirective) = self.loadModelWithPredicts(track)
        return self._collectSamples(assumeToDirective, predictToDirective, **kwargs)

    # Returns a History reflecting exactly one Run.
    def collectStateSequence(self, name=None, profile=False, **kwargs):
        assumeToDirective = self._assumesFromRipl()
        tag = 'run_from_conditional' if name is None else name + '_run_from_conditional'
        history = History(tag, self.parameters)
        history.addRun(self._collectSamples(assumeToDirective, {}, label="interactive", **kwargs))
        if profile:
            history.profile = Profile(self.ripl)
        return history

    def _collectSamples(self, assumeToDirective, predictToDirective, sweeps=100, label=None, verbose=False, infer=None):
        answer = Run(label, self.parameters)

        assumedValues = {symbol : [] for symbol in assumeToDirective}
        predictedValues = {index: [] for index in predictToDirective}
        queryExpsValues = {exp: [] for exp in self.queryExps}

        sweepTimes = []
        sweepIters = []
        logscores = []

        for sweep in range(sweeps):
            if verbose:
                print "Running sweep " + str(sweep) + " of " + str(sweeps)

            # FIXME: use timeit module for better precision
            start = time.time()
            iterations = self.sweep(infer=infer)
            end = time.time()

            sweepTimes.append(end-start)
            sweepIters.append(iterations)
            logscores.append(self.ripl.get_global_logscore())

            self.updateValues(assumedValues,keyToDirective=assumeToDirective)
            self.updateValues(predictedValues,keyToDirective=predictToDirective)
            self.updateValues(queryExpsValues,keyToDirective=None)
            
        answer.addSeries('sweep time (s)', Series(label, sweepTimes))
        answer.addSeries('sweep_iters', Series(label, sweepIters))
        answer.addSeries('logscore', Series(label, logscores))

        for (symbol, values) in assumedValues.iteritems():
            answer.addSeries(symbol, Series(label, map(parseValue, values)))

        for (index, values) in predictedValues.iteritems():
            answer.addSeries(self.nameObserve(index),
                             Series(label, map(parseValue, values)))

        for (exp, values) in queryExpsValues.iteritems():
            answer.addSeries(exp, Series(label, map(parseValue, values)))

        return answer


    def runFromConditional(self, sweeps, name=None, data=None, **kwargs):
        '''Runs inference on the model conditioned on data.
           By default data is values of self.observes.

           Arguments
           ---------
           sweeps :: int
               Total number of iterations of inference program. Values are
               recorded after every sweep.
           runs :: int
               Number of parallel chains for inference. Default is 3.
           infer :: string | function on ripl
               Inference program
           name :: string
               Label this particular set of runs. Added to history and plots.
           data :: [values]
               List of values that replace values in self.observes for this
               inference run only.
           verbose :: bool
               Print when initiating runs and sweeps.

           Returns
           -------
           history :: History
               history.History with nameToSeries dictionary of runs*Series
               for each recorded expression.
           ripl :: ripl
               Ripl with same backend as given to constructor, mutated by
               assumes,observes (with values given by data) and inference.'''

        tag = 'run_from_conditional' if name is None else name + '_run_from_conditional'
        
        history = self._runRepeatedly(self.runFromConditionalOnce,
                                      tag, data=data, sweeps=sweeps, **kwargs)

        if data is not None: # data specified by user or by other method
            ## FIXME this branch has types, other one doesn't
            data = [(exp,datum) for (exp,_),datum in zip(self.observes,data)]
        else:
            data = self.observes
        history.addData(data)

        return history,self.ripl
  
    def runFromConditionalOnce(self, data=None, **kwargs):
        self.ripl.clear()
        assumeToDirective = self._loadAssumes()
        self._loadObserves(data)
##FIXME remove print
        print 'runFCO data:', data
        
        # note: we loadObserves, but predictToDirective arg = {}
        # so we are not collecting sample of the observes here. 
        return self._collectSamples(assumeToDirective, {}, **kwargs)

    
    def testFromPrior(self,no_datasets,sweeps,verbose=False,**kwargs):
        # we don't need sweeps for generateDataFromPrior, because
        # sweeps are given just for plotting the data on time series

        # FIXME this method will block certain types of data via *record*
        typed_datasets = [self.generateDataFromPrior(1)[0] for reps in range(no_datasets)]
        datasets = [[datum['value'] for datum in dataset] for dataset in typed_datasets]
        
        histories = []
        parameters = {'venture_random_seed': 0} # FIXME: Lite compatible

        # problem that the seed does not appear to fix the initial state
        # of the chain. one thing to do is get the ripl to print it's state
        # and see where the states diverge and whether we can intervene there
        for i,dataset in enumerate(datasets):
            observes = [(exp,datum) for (exp,_),datum in zip(self.observes,dataset)]
            a = Analytics( ripl=make_puma_church_prime_ripl(),
                           assumes=self.assumes, observes=observes,
                           queryExps=self.queryExps, parameters=parameters )

            histories.append(a.runFromConditional( sweeps,runs=1, **kwargs))
        return histories

    # Run inference conditioned on data generated from the prior.
    def runConditionedFromPrior(self, sweeps, verbose=False, **kwargs):
        
        (data, prior_run, groundTruth) = self.generateDataFromPrior(sweeps, verbose=verbose)
        
        history = self.runFromConditional(sweeps, data=data, verbose=verbose, **kwargs)
        history.addRun(prior_run)
        history.addGroundTruth(groundTruth,sweeps)
        history.label = 'run_conditioned_from_prior'
        return history

    # The "sweeps" argument specifies the number of times to repeat
    # the values collected from the prior, so that they are parallel
    # to the samples one intends to compare against them.
    def generateDataFromPrior(self, sweeps, verbose=False):
        if verbose:
            print 'Generating data from prior'

        (assumeToDirective, predictToDirective) = self.loadModelWithPredicts(prune=False)

        data = [self.ripl.report(predictToDirective[index],type=True) for index in range(len(self.observes))]

        prior_run = Run('run_conditioned_from_prior', self.parameters)
        assumedValues = {}
        for (symbol, directive) in assumeToDirective.iteritems():
            value = self.ripl.report(directive, type=True)
            if record(value):
                assumedValues[symbol] = value
        
        queryExpsValues = {}
        for exp in self.queryExps:
            value = self.ripl.sample(exp,type=True)
            if record(value):
                queryExpsValues[exp] = value
                
        logscore = self.ripl.get_global_logscore()
        prior_run.addSeries('logscore', Series('prior',
                                               [logscore]*sweeps, hist=False))
        
        groundTruth = assumedValues.copy() # store groundTruth as {exp:value}
        groundTruth.update(queryExpsValues.copy())

        return (data, prior_run, groundTruth)




    # Computes the KL divergence on i.i.d. samples from the prior and inference on the joint.
    # Returns the sampled history, inferred history, and history of KL divergences.
    def computeJointKL(self, sweeps, samples, track=5, runs=3, verbose=False, name=None, infer=None):
        sampledHistory = self.sampleFromJoint(samples, track, verbose, name=name)
        inferredHistory = self.runFromJoint(sweeps, track=track, runs=runs, verbose=verbose, name=name, infer=infer)

        tag = 'kl_divergence' if name is None else name + '_kl_divergence'
        klHistory = History(tag, self.parameters)

        for (name, seriesList) in inferredHistory.nameToSeries.iteritems():
            if name not in sampledHistory.nameToSeries: continue

            for inferredSeries in seriesList:
                sampledSeries = sampledHistory.nameToSeries[name][0]

# KL between all fwd samples and inferredSeries samples up to index
# why not combine inferred series into snapshots?
## FIXME: note that currently one of runs is not appearing
                klValues = [computeKL(sampledSeries.values, inferredSeries.values[:index+1]) for index in range(sweeps)]

                klHistory.addSeries('KL_' + name, inferredSeries.label, klValues, hist=False)

        return (sampledHistory, inferredHistory, klHistory)




# Reads the profile data from the ripl.
# Returns a map from (random choice) addresses to info objects.
# The info contains the trials, successes, acceptance_rate, proposal_time, and source_location.
class Profile(object):
    def __init__(self, ripl):
        random_choices = ripl.profiler_list_random_choices()
        self.addressToInfo = {}
        self.locationToInfo = {}

        for address in random_choices:
            info = object()
            info.address = address

            acceptance = self.ripl.profiler_get_acceptance_rate(address)
            info.trials = acceptance[0]
            info.successes = acceptance[1]
            info.acceptance_rate = info.successes / info.trials

            info.proposal_time = self.ripl.profiler_get_proposal_time(address)

            info.source_location = self.ripl.profiler_address_to_source_code_location()

            self.addressToInfo[address] = info

            if info.proposal_time not in self.locationToAddress:
                self.locationToAddress[info.proposal_time] = []

            self.locationToAddress[info.proposal_time].append(info)

        # aggregates multiple info objects into one
        def aggregate(infos):
            agg = object()

            for attr in ['trials', 'successes', 'proposal_time']:
                setattr(agg, attr, sum([getattr(info, attr) for info in infos]))

            agg.acceptance_rate = agg.successes / agg.trials

            return agg

        self.locationToAggregate = dict([(location, aggregate(infos)) for (location, infos) in self.locationToInfo.items()])

    # The [5] longest
    def hotspots(self, num=5):
        hot = sorted(self.addressToInfo.values(), key=lambda info: info.proposal_time, reverse=True)
        return hot[:num]

    def coldspots(self, num=5):
        cold = sorted(self.addressToInfo.values(), key=lambda info: info.acceptance_rate)
        return cold[:num]

import math

# Approximates the KL divergence between samples from two distributions.
# 'reference' is the "true" distribution
# 'approx' is an approximation to 'reference'
def computeKL(reference, approx, numbins=20):

    # smooths out a probability distribution function
    def smooth(pdf, amt=0.1):
        return [(p + amt / len(pdf)) / (1.0 + amt) for p in pdf]

    mn = min(reference + approx)
    mx = max(reference + approx)

    refHist = np.histogram(reference, bins=numbins, range = (mn, mx), density=True)[0]
    apxHist = np.histogram(approx, bins=numbins, range = (mn, mx), density=True)[0]

    refPDF = smooth(refHist)
    apxPDF = smooth(apxHist)

    kl = 0.0

    for (p, q) in zip(refPDF, apxPDF):
        kl += math.log(p/q) * p * (mx-mn) / numbins

    return kl

















