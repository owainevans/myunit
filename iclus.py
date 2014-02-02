from IPython.parallel import Client
from venture.shortcuts import *

def get_pid():
    import os
    return os.getpid()

def flip(p=.5):
    import numpy
    return numpy.random.binomial(1,p)



cli = Client()
dview = cli[:]
no_ripls = len(cli.ids)

def init_ripls(no_ripls):
    ripls = [make_church_prime_ripl() for i in range(no_ripls) ]
    for i,ripl in enumerate(ripls):
        ripl.set_seed(i)
    return ripls
v = init_ripls(no_ripls)


# def assume(sym,exp): return v.assume(sym,exp)
# def predict(exp): return v.predict(exp)
# def set_seed(seed): return v.set_seed(seed)

# dview.block = True
# v=make_church_prime_ripl()
# dview['v']=5

# dview.execute('from venture.shortcuts import make_church_prime_ripl')
# dview.execute('v=make_church_prime_ripl()')
# dview.map(set_seed,cli.ids)
# dview.apply(assume,'p','(beta 1 1)')
# dview.apply(assume,'x','(flip p)')
# dview.apply( lambda exp,val: v.observe(exp,val), 'x','true')
# dview.apply( lambda exp,val: v.observe(exp,val), '(flip p)','true')
# dview.apply( lambda n: v.infer(n),10 )
# r = dview.apply( lambda var: v.report(var), 1)



#blocking is optional pass through (default is async), but wait() method will block on last thing

# get access to master

# write nose testing: capture spec for new methods that aren't in methods

# test parallel machinery: compare to single ripl. fix seeds

# support discrete data, make sure tests cover discrete

# make sure 'procedure' data is handeled, throw an exception (Test should show exception)

# discrete/cts, discrete 'scatter, 2d heatmap.

# multiripl magic

# starcluster, venture installed, template

# clear shouldn't destroy the seed (delegate new seed after clear)

# continuous inference

# map somethign across all ripls

# for mripl, what's procesdure i used to display state (one magic)
# separate magic for running for the pgoram

1. cell for no_ripls. (doesnt need magic) 2. one magic for update display. 

display() :: ripl,fig  ->    string, fig 
mr.set_display(display)
mr.display(plot = true, model = random (vs. all) )
calls display attribute, displays results and collects them)

CRP: discrete values, plots.

Demos: IPython.parallle, skip bokeh, readme, intro to start walkers. 

Walk through, applying to crp mixture. 

ADD this spec to Asana. (Alexey a follower, + vikash)

baxter.

#Python Freenode:#Python (channel), minrk or min_rk

marshall on what econ can't do

can't do agent based model in econ coz can't identify anything computationally 
(or even in principle if your frequentists). so not clear what your sim shows.
hence go for regression instead where things are identifiable (but what things)

logic vs. progams: programs dont have uninterpreted constants that then 
can be talked of as satisying different values in different models. 
instead you have to get the thing from somewhere. (though type theory
inference is based on assuming its any object of a certain type, and 
so related to the logical case. just unclear what the types are in 
terms fo reference to the world). logical model doesn't involve a prior
over models. probalistic program doesn't either but we dont have way
of using it in this way -- we could try to do transformations on it
where we leave constants undefied, could use inference over proof
theory to help us. 

why CYC fails: scaling problems. if squishing all into limited logic
results in exp or high poly blow (as in chess example) then youre 
screwed. logic based approach involves treating facts as zero/1
known or not. (epistemic). and then using a particular form 
for representation (declarative). people tried to keep rich form
while having something like probabilities on sentences (christiano) but that's hard). what's wrong with logic: certainly zero/1 is very bad description
of our state of knolwefdge/understanding. but the representation may be bad
also, due to poor formalization, critical issues of modality, vagueness/
prototypicality, types, etc. dont assume that the representation is that
good in general. 

logic, like lisp, seems really powerful. but we still can't deal with 
many quantifiers! good things that come from logical more
 philosophical, about what things
to put in the model, what premises lead to plausible models; organizing thought
ruling out minimalist theories, etc. 

logic good for det domains of math and circuits. (what about 
logical planning?)

gerry sussman: need procedural thing in physics for non-linear
systems wher eyou can build good simulations but can't do analytical 
characterization from initial condition. so procedure is just
critical epistemically; declaarative analytical appproach fundamentally
limited (for all strengths). 

should the constants be in a linked list or hashmap or what. physicists 
says neither. aaronsonian. 




class MRipls():
    def __init__(self,cli):
        self.no_ripls = len(cli.ids)
        self.ids = cli.ids
        self.dview = cli[:]
        self.dview.block = True
        self.dview.execute('from venture.shortcuts import make_church_prime_ripl')
        self.dview.execute('v=make_church_prime_ripl()')
        self.dview.map(lambda seed: v.set_seed(seed), range(self.no_ripls) )
        
    def map_predict(self,exps):
        return self.dview.map(lambda exp: v.predict(exp), exps)
        
    def predict(self,exp):
        return self.dview.apply(lambda exp: v.predict(exp), exp)




v = MRipls(cli)
r = v.predict('(beta 1 1)')
rs = v.map_predict( [str(i**2) for i in range(10)] )


        



        
def mk2():
    myd['v'] = venture.shortcuts.make_church_prime_ripl()
    return myd['v']
    
def mk():
    global v
    v = venture.shortcuts.make_church_prime_ripl()
    return v

def pred(): return v.predict('(beta 1 1)')

dview.block = True
with dview.sync_imports():
    import venture.shortcuts


dview.push( { 'v':0 } )
dview.push( { 'myd':{} } )
#dview.apply(mk2)
