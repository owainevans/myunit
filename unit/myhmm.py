from venture import shortcuts
from venture.venturemagics.ip_parallel import *
from analytics import *

xs = [4.17811131241034, 3.8207451562269097, 3.8695630629179485,
      0.22006118284977338, 2.210199033799397, 3.783908156611711,
      2.837419867371207, 1.317835790137246, -0.16712980626716778,
      2.9172052420088024, 2.510820987230155, 3.8160095647125587,
      2.1845237960891737, 4.857767012696541, 3.575666111020788,
      0.5826540416187078, 4.911935337633685, 1.6865857289172699,
      2.096957795256201, 3.962559707705782, 2.0649737290837695,
      4.447773338208195, 3.0441473773992254, 1.9403530443371844,
      2.149892339815339, 1.8535027835799924, 1.3764327100611682,
      2.787737100652772, 4.605218953213757, 4.3600668534442955,
      4.479476152575004, 2.903384365135718, 3.228308841685054,
      2.768731834655059, 2.677169426912596, 4.548729323863021,
      4.45470931661095, 2.2756630109749754, 3.8043219817661464,
      4.041893001861111, 4.932539777501281, 3.392272043248744,
      3.5285486875160186, 1.7961542635140841, 2.9493126820691664,
      1.7582718429078779, 3.444330463983401, 2.031284816908312,
      1.6347773147087383, 4.423285189276542, 0.5149704854992727,
      4.470589149104097, 4.4519204418264575, 3.610788527431577,
      3.7908243330830036, 3.0038367596454187, 3.3486671878130028,
      4.474091346599369, 2.7734106792197633, 1.8127987198750086]

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
ys = [5,5,5,5,5,.1,.1]
# note: we could follow robert. allow states 0 and 2 to be observed
# and then get the observation for state 1.the negative point is favored
# coz the other observed states are negative and noise terms (which are known)
# are very high
ar1Observes=[("(observation_fn (get_state %d))"%t, y) for t,y in enumerate(ys)]
ar1Program=(ar1Model,ar1Observes)

def anaLoad(prog=ar1Program,ripl='p',length=5):
  v = mk_p_ripl() if ripl=='p' else mk_l_ripl()
  model=prog[0]
  observes=prog[1][:length]
  v.execute_program(model)
  [v.observe(exp,literal) for exp,literal in observes]
  queryExps=['(get_state %d)' % t for t,_ in enumerate(observes)]
  ana=Analytics(v,queryExps=queryExps)
  return ana
  
inf=["(mh default one 1)",
     "(cycle ((mh hypers one 3) (mh state one 10)) 1)",
     "(cycle ((mh hypers one 3) (pgibbs state ordered 6 1)) 1)",
     "(cycle ((mh hypers one 3) (func-pgibbs state ordered 12 1)) 1)"]

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

length=3
phmm = anaLoad(prog=hmmProgram,length=length)
par =  anaLoad(prog=ar1Program,length=length)
ana = par
h0,r0 = ana.runFromConditional(10,runs=1,infer=inf[0])
h1,r1 = ana.runFromConditional(10,runs=1,infer=inf[1])
h2,r2 = ana.runFromConditional(20,runs=1,infer=inf[2])
h3,r3 = ana.runFromConditional(10,runs=1,infer=inf[3])
hs=[h0,h1,h2,h3]


# want to actually see how good the second mode is? we can force and 
# look at logscore. but with enough particles, we should get it.
def sampStates(r):
  s=[r.sample('(get_state %d)'%t) for t in range(length)]
  return r.get_global_logscore(),s

def forcer(r):
  print 'forcer'
  print 'samp:',sampStates(r)
  [r.force('(get_state %d)'%t,'7') for t in range(length)]
  print 'samp:',sampStates(r)
  r.infer(1)
  print 'after infer'
  print 'samp:',sampStates(r)
  

# do diagnostics work? we can try running runFromJoint, and 
# see if that differentiates the two inference progs. our problem
# here is autocorrelation, not convergence to staionarity/equilib.
# i expect that geweke will move around a fair amount. we 
# could also try running on multi datasets. what to see? we should 
# actually see success here. basically: you'll have cases where 
# groundTruth is like it is here, where it's a big x0 value and 
# the model won't find it. so it won't respect prior. 

# geweke
#geh = ana.runFromJoint(100,runs=1,infer=inf[0])

noDatasets=2
hists2=[]
exps=['(get_state %d)'%t for t in range(length)]
gt={exp:[] for exp in exps}
s=gt.copy()
sweeps=100
probes=range(int(.5*sweeps),sweeps,10)

for i in range(noDatasets):
  print 'dataset: ',i
  h2,_=ana.runConditionedFromPrior(sweeps,runs=1,infer=inf[0])
  hists2.append(h2)
  
  for exp in exps:
    gt[exp].append(  h2.groundTruth[exp]['value'] )
    s[exp].extend( [h2.nameToSeries[exp][0].values[p] for p in probes] )

# sample from joint
fwdh = ana.sampleFromJoint(len(s[exps[0]]))

fwd={exp:fwdh.nameToSeries[exp][0].values for exp in exps}

from ge import *
stats_dict,fig = compareSampleDicts([s,fwd],['test2_inf','fwd'],plot=True)




#historyOverlay('Geweke vs. Forward', [(hisGeweke.label,hisGeweke),
        #                                        (hisForward.label,hisForward)])


# on the x^2 version with N(0,1) init
# ran h1 for 1500 steps, gets stuck in -2 local mode
# can make harder by making one mode more attractive: have a constant
# that biases the function
# h3 is able to move between modes much better (e.g. 200 sweeps)
