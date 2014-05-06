from venture.venturemagics.ip_parallel import *
from venture.unit import *
execfile('/home/owainevans/Venturecxx/python/lib/unit/history.py')

xs = [4.17811131241034, 3.8207451562269097, 3.8695630629179485,
      0.22006118284977338, 2.210199033799397, 3.783908156611711,
      2.837419867371207, 1.317835790137246, -0.16712980626716778,
      2.9172052420088024, 2.510820987230155, 3.8160095647125587,
      2.1845237960891737, 4.857767012696541, 3.575666111020788,
      0.5826540416187078, 4.911935337633685, 1.6865857289172699,
      2.096957795256201, 3.962559707705782, 2.0649737290837695,
      4.447773338208195, 3.0441473773992254, 1.9403530443371844,
      2.149892339815339, 1.8535027835799924, 1.3764327100611682,
      2.787737100652772, 4.605218953213757, 4.3600668534442955,]
xs[3:7] = [-x for x in xs[5:10]]
xs[17:23] = [-x for x in xs[17:23]]

hmmModel = """
[ASSUME observation_noise (scope_include (quote hypers) 0 (gamma 1.0 1.0))]

[ASSUME get_state
  (mem (lambda (t)
	 (if (= t 0) 
	     (scope_include (quote state) 0 (bernoulli 0.3))
	     (transition_fn (get_state (- t 1)) t))))]

[ASSUME get_observation
  (mem (lambda (t)
	 (observation_fn (get_state t))))]

[ASSUME transition_fn
  (lambda (state t)
    (if state
        (scope_include (quote state) t (bernoulli 0.7))
        (scope_include (quote state) t (bernoulli 0.3))))]

[ASSUME observation_fn
  (lambda (state)
    (normal (if state 3 -3) observation_noise))]

"""
hmmObserves=[("(observation_fn (get_state %d))"%t,x) for t,x in enumerate(xs)]
hmmProgram=(hmmModel,hmmObserves)

ar1Model = """
[ASSUME observation_noise (scope_include (quote hypers) 0 (gamma 1.0 1.0))]
[ASSUME rho (scope_include (quote hypers) 1 (beta 1 1))]
[ASSUME epsilon (scope_include (quote hypers) 2 (gamma 1 1) )]

[ASSUME get_state
  (mem (lambda (t)
	 (if (= t 0) 
	     (scope_include (quote state) 0 (normal 0 3.5))
	     (transition_fn (get_state (- t 1)) t))))]

[ASSUME get_observation
  (mem (lambda (t)
	 (observation_fn (get_state t))))]

[ASSUME transition_fn
  (lambda (state t)
    (scope_include (quote state) t (normal (* rho state) epsilon) ) ) ]

[ASSUME observation_fn
  (lambda (state)
    (normal (pow (- state 5) 2) observation_noise))]
"""

# note: we could follow robert. allow states 0 and 2 to be observed
# and then get the observation for state 1.the negative point is favored
# coz the other observed states are negative and noise terms (which are known)
# are very high

# 'default' is scope containing all random choices (latent is random choices
# hidden from venture.
# 'one' in the block position means choose a block uniformly at random
# 'all' means the union of all blocks is taken. 
# (pgibbs <scope> <block> <no_particles> <transitions>) if block is ordered
# all the blocks in scope are sorted, each dist in sequence of dists includes
# all random choices from next block.
# 'func-gibbs', while default mh and pgibbs use in-place mutation, this uses
# simultaneous particules to represent alternative possibilities. 
# composition: (cycle (<inf_exp> ... <#transitions>), (mixture ((<w1> <exp1>) ...)


ys = [5,5,5,.1,.1]
ar1Observes=[("(observation_fn (get_state %d))"%t, y) for t,y in enumerate(ys)]
ar1Program=(ar1Model,ar1Observes)


def loadTSModel(program=ar1Program, backend='puma', observesLength=3):
  assumes=program[0]
  observes=program[1][:observesLength]

  # load ripl with program
  v = mk_p_ripl() if backend=='puma' else mk_l_ripl()
  v.execute_program(assumes)
  [v.observe(exp,literal) for exp,literal in observes]
  queryExps=['(get_state %d)' % t for t,_ in enumerate(observes)]
  return Analytics(v,queryExps=queryExps)
  

inferProgs=["(mh default one 1)",
            "(cycle ((mh hypers one 3) (mh state one 10)) 1)",
            "(cycle ((mh hypers one 3) (pgibbs state ordered 6 1)) 1)",
            "(cycle ((mh hypers one 3) (func-pgibbs state ordered 12 1)) 1)"]

inferProgsParams = [ (200,1), (100,1), (20,1), (200,1) ]
inferProgsParams = zip([5]*4,[1]*4)
observesLength=3
model = loadTSModel( program=ar1Program, observesLength=observesLength)

def runInferProgs():
  histories = []
  for inferProg,inferParams in zip(inferProgs,inferProgsParams):
    print 'Infer Prog: %s'%inferProg
    sweeps,runs = inferParams
    history,ripl = model.runFromConditional(sweeps, runs=runs, infer=inferProg,
                                            name=inferProg)
    histories.append(history)
  return histories

histories = runInferProgs()
histories[3].quickHistogram('(get_state 0)')
[h.quickPlot('(get_state 0)') for h in histories]
[h.quickPlot('observation_noise') for h in histories]
[h.quickScatter('(get_state 0)','(get_state 1)') for h in histories]


historyOV=historyOverlay('MH vs. PGibbs',
                         (('MH',histories[0]),('PG',histories[3])))
historyOV.quickPlot('(get_state 0)')


def runFromJointDiagnostic(inferProgs,sweeps=100):
  hists = []
  for inferProg in inferProgs:
    hists.append(model.runFromJoint( sweeps, runs=1,
                                    infer=inferProg,name=inferProg))
  hists.append(model.sampleFromJoint(sweeps))
  return hists


fromJointHistories = runFromJointDiagnostic((inferProgs[0],inferProgs[3]), sweeps=200)
fromJointPairs = zip(['MH','PG','Fwd'],fromJointHistories)
fromJointOV=historyOverlay('MH vs. PGibbs vs. Forward',fromJointPairs)
fromJointOV.quickHistogram('(get_state 0)')
fromJointOV.quickPlot('(get_state 0)')

# Inference Diagnostic: testFromPrior
# The initial state *(get_state 0)* is drawn from N(0,3.5). We generate synthetic datasets from the prior and run inference. We take one sample from the approximate posterior (i.e. one sample per dataset). These samples would have distribution N(0,3.5) were the approximation perfect. We can diagnose problems with inference by comparing samples from the prior to these samples due to inference.  


def testFromPrior(noDatasets,sweeps=100,plotLimit=10,inferProg=None):
  if inferProg is None: 
    inferProg=inferProgs[0]
    
  datasetToHistory = []
  getStateList = []
  
  for dataset in range(noDatasets):
    history,_ = model.runConditionedFromPrior(sweeps,runs=1,infer=inferProg,
                                              name='Dataset %i'%dataset)

    getState = history.nameToSeries['(get_state 0)'][0].values[-1]
    print 'Dataset %i: last sample of (get_state 0) = %.2f'%(dataset,getState)
    getStateList.append(getState)

    if dataset < plotLimit: history.quickPlot('(get_state 0)')

    datasetToHistory.append( history )

  priorSamples = np.random.normal(0,3.5,noDatasets)
  fig,ax = plt.subplots(figsize=(10,4))
  ax.hist(getStateList, color='r', alpha=.6,
          label='Inferred (get_state 0) across datasets')
  ax.hist(priorSamples,color='y',alpha=.6,
          label='Samples from N(0,3.5) prior on (get_state 0)')
  ax.set_title('Inferred vs. Prior Samples for (get_state 0)')
  print 'getstate (across datasets): (mean, std) '
  print np.mean(getStateList),np.std(getStateList)
  print 'getstate (forward sample from prior): (mean, std) '
  print np.mean(priorSamples),np.std(priorSamples)
  
  return datasetToHistory,fig

noDatasets=5
datasetToHistory,fig = testFromPrior(noDatasets,sweeps=10)
testFromPairs = zip( map(str,range(noDatasets)), datasetToHistory)
testFromPriorOV = historyOverlay('testFromPrior',testFromPairs[:5])
                              

# want to actually see how good the second mode is? we can force and 
# look at logscore. but with enough particles, we should get it.
def sampStates(r):
  s=[r.sample('(get_state %d)'%t) for t in range(observesLength)]
  return r.get_global_logscore(),s

def forcer(r):
  print 'forcer'
  print 'samp:',sampStates(r)
  [r.force('(get_state %d)'%t,'7') for t in range(observesLength)]
  print 'samp:',sampStates(r)
  r.infer(1)
  print 'after infer'
  print 'samp:',sampStates(r)
  
# our problem
# here is autocorrelation, not convergence to staionarity/equilib.
# i expect that geweke will move around a fair amount. we 
# could also try running on multi datasets. what to see? we should 
# actually see success here. basically: you'll have cases where 
# groundTruth is like it is here, where it's a big x0 value and 
# the model won't find it. so it won't respect prior. 

# geweke


from ge import *

def test2(noDatasets=2):
  hists2=[]
  exps=['(get_state %d)'%t for t in range(observesLength)]
  gt={exp:[] for exp in exps}
  s=gt.copy()
  sweeps=200
  probes=range(int(.5*sweeps),sweeps,10)

  for i in range(noDatasets):
    print 'dataset: ',i
    h2,_=model.runConditionedFromPrior(sweeps,runs=1,infer=inf[0])
    hists2.append(h2)

    for exp in exps:
      gt[exp].append(  h2.groundTruth[exp]['value'] )
      s[exp].extend( [h2.nameToSeries[exp][0].values[p] for p in probes] )

  # sample from joint
  fwdh = model.sampleFromJoint(len(s[exps[0]]))

  fwd={exp:fwdh.nameToSeries[exp][0].values for exp in exps}

  stats_dict,fig = compareSampleDicts([s,fwd],['test2_inf','fwd'],plot=True)
  return hist2,s,fwd,stats_dict



#historyOverlay('Geweke vs. Forward', [(hisGeweke.label,hisGeweke),
        #                                        (hisForward.label,hisForward)])


# on the x^2 version with N(0,1) init
# ran h1 for 1500 steps, gets stuck in -2 local mode
# can make harder by making one mode more attractive: have a constant
# that biases the function
# h3 is able to move between modes much better (e.g. 200 sweeps)
