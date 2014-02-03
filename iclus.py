
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

# 1. cell for no_ripls. (doesnt need magic) 2. one magic for update display. 

# display() :: ripl,fig  ->    string, fig 
# mr.set_display(display)
# mr.display(plot = true, model = random (vs. all) )
# calls display attribute, displays results and collects them)

# CRP: discrete values, plots.

# Demos: IPython.parallle, skip bokeh, readme, intro to start walkers. 

# Walk through, applying to crp mixture. 

# ADD this spec to Asana. (Alexey a follower, + vikash)

# baxter.

# #Python Freenode:#Python (channel), minrk or min_rk


from IPython.parallel import Client
from venture.shortcuts import *
import numpy as np


copy_ripl_string="""
def build_exp(exp):
    'Take expression from directive_list and build the Lisp string'
    if type(exp)==str:
        return exp
    elif type(exp)==dict:
        return str(exp['value'])
    else:
        return '('+str(exp[0])+' ' + ' '.join(map(build_exp,exp[1:])) + ')'

def run_py_directive(ripl,d):
    'Removes labels'
    if d['instruction']=='assume':
        ripl.assume( d['symbol'], build_exp(d['expression']) )
    elif d['instruction']=='observe':
        ripl.observe( build_exp(d['expression']), d['value'] )
    elif d['instruction']=='predict':
        ripl.predict( build_exp(d['expression']) )
    
def copy_ripl(ripl,seed=None):
    '''copies ripl via di_list to fresh ripl, preserve directive_id
    by preserving order, optionally set_seed'''
    di_list = ripl.list_directives()
    new_ripl = make_church_prime_ripl()
    if seed: new_ripl.set_seed(seed)
    [run_py_directive(new_ripl,di) for di in di_list]
    return new_ripl
"""

def build_exp(exp):
    'Take expression from directive_list and build the Lisp string'
    if type(exp)==str:
        return exp
    elif type(exp)==dict:
        return str(exp['value'])
    else:
        return '('+str(exp[0])+' ' + ' '.join(map(build_exp,exp[1:])) + ')'

def run_py_directive(ripl,d):
    'Removes labels'
    if d['instruction']=='assume':
        ripl.assume( d['symbol'], build_exp(d['expression']) )
    elif d['instruction']=='observe':
        ripl.observe( build_exp(d['expression']), d['value'] )
    elif d['instruction']=='predict':
        ripl.predict( build_exp(d['expression']) )
    
def copy_ripl(ripl,seed=None):
    '''copies ripl via di_list to fresh ripl, preserve directive_id
    by preserving order, optionally set_seed'''
    di_list = ripl.list_directives()
    new_ripl = make_church_prime_ripl()
    if seed: new_ripl.set_seed(seed)
    [run_py_directive(new_ripl,di) for di in di_list]
    return new_ripl

# test for copy_ripl funtion
myv = make_church_prime_ripl()
myv.assume('x','(beta 1 1)'); myv.observe('(normal x 1)','5'); myv.predict('(flip)')
assert [build_exp(di['expression']) for di in myv.list_directives() ] ==  [build_exp(di['expression']) for di in copy_ripl(myv).list_directives() ]

# test for parallel use of copy_ripl_string

cli = Client(); dv = cli[:]; dv.block=True
dv.execute(copy_ripl_string)
dv.execute('from venture.shortcuts import make_church_prime_ripl')
dv.execute('v=make_church_prime_ripl()')
dv.execute('v.set_seed(1)')
dv.execute("v.assume('x','(beta 1 1)'); v.observe('(normal x 1)','5'); v.predict('(flip)')" )
dv.execute("v2 = copy_ripl(v,seed=1)" )
dv.execute("true_assert = [build_exp(di['expression']) for di in v.list_directives() ] ==  [build_exp(di['expression']) for di in copy_ripl(v).list_directives() ]")
assert all(dv['true_assert'])


## PLAN
# 1.local ripl with same seed as first remote ripl that gets same directives
# 2. check copying (1 ripl, 1 engine, add 1 extra with same seed)
# 3. note labeling problems and specify solution
# 4. how does copying interact with inference? (would need to copy values)

# 5. how to do display:
# cell magic where you write function def that takes variable ripl
# (or that has the name in header and just is a bunch of code on ripl)

# we have one multiripl in ipython global. given cell magic, we now 
# call the display method of multiripl. this takes the string from 
# cell and executes across all engines. (we also eval the code locally).
# we the have an appy:
#   dview.apply( (def f (return [user_func(ripl) for ripl in ripls]), no args) )

# simpler: just get people to use para magic %%px to define the function, using
# any variable name for the ripl. have them give the function name as a string. 
# then display(func_name) can be dview.execture('[ %s(r) for r in ripls]'%func_name)


# even simpler?:
#  user just defines function locally
#  they run mrip.display(function).
# display function says 
   
# def display(self,user_func):
#     def p_func():
#         return [user_func(ripl) for ripl in ripls]
#     dview.apply( p_func,no args)

# does this work? no, coz user_funct wouldnt be sent along. 



class MRipls():
    def __init__(self,no_ripls,block=False):
        self.local_ripl = make_church_prime_ripl()
        self.local_ripl.set_seed(0)   # same seed as first remote ripl
        self.no_ripls = no_ripls
        self.seeds = range(self.no_ripls)
        
        self.cli = Client()
        self.ids = cli.ids
        self.dview = cli[:]
        self.dview.block = block

        self.dview.execute('from venture.shortcuts import make_church_prime_ripl')
        self.dview.execute('ripls = []')
        self.dview.execute('seeds = []')
        self.dview.execute(copy_ripl_string) # defines copy_ripl and dependencies
        
        def mk_ripl(seed):
            ripls.append( make_church_prime_ripl() )
            ripls[-1].set_seed(seed)
            seeds.append(seed)
            import os
            pid = os.getpid()
            print 'Engine %i created ripl %i' % (pid,seed)
            return pid,seed # should we return a ripl for debugging?
            # should also return the position in the local ripls list
        
        
        self.id_seed_pairs = self.dview.map( mk_ripl, self.seeds )

        print self.id_seed_pairs.get()
        

    def assume(self,sym,exp,**kwargs):
        self.local_ripl.assume(sym,exp,**kwargs)
        def f(sym,exp,**kwargs):
            return [ripl.assume(sym,exp,**kwargs) for ripl in ripls]
        return self.dview.apply(f,sym,exp,**kwargs)       
        
    def observe(self,exp,val):
        self.local_ripl.observe(exp,val)
        def f(exp,val): return [ripl.observe(exp,val) for ripl in ripls]
        return self.dview.apply(f,exp,val)
    
    def predict(self,exp):
        self.local_ripl.predict(exp)
        def f(exp): return [ripl.predict(exp) for ripl in ripls]
        return self.dview.apply(f,exp)

    def infer(self,params):
        self.local_ripl.infer(params)
        def f(params): return [ripl.infer(params) for ripl in ripls]
        return self.dview.apply(f,params)

    def report(self,label_or_did,**kwargs):
        self.local_ripl.report(label_or_did,**kwargs)
        def f(label_or_did,**kwargs):
            return [ripl.report(label_or_did,**kwargs) for ripl in ripls]
        return self.dview.apply(f,label_or_did,**kwargs)
        
    def add_ripls(self,no_ripls,new_seeds=None):
        # could instead check this for each engine we map to
        # and just fail to copy a few
        if not all(self.dview['ripls']):
            print 'Error: some engines have no ripl, add_ripls failed'
            return None

        if not(new_seeds):
            last = self.seeds[-1]
            new_seeds = range( last+1, last+1+no_ripls)
        self.seeds += new_seeds


        def add_ripl_engine(seed):
            ripls.append( copy_ripl(ripls[0]) ) # ripls[0] must be present
            ripls[-1].set_seed(seed)
            seeds.append(seed)
            import os;   pid = os.getpid()
            print 'Engine %i created ripl %i' % (pid,seed)
            return pid,seed

        update = self.dview.map(add_ripl_engine,seeds)
        self.id_seed_pairs.append(update)
        
        return update
            

v = MRipls(4); 
test_v = make_church_prime_ripl(); test_v.set_seed(0)
ls_x = v.assume('x','(uniform_continuous 0 1000)').get()
test_x = test_v.assume('x','(uniform_continuous 0 1000)')
local_x = v.local_ripl.report(1)
assert( np.round(test_x) in np.round(ls_x) )
assert( np.round(local_x) in np.round(ls_x) )

# this fails with val = '-10.'
v.observe('(normal x 50)','-10')
test_v.observe('(normal x 50)','-10')
ls_obs = v.report(2).get()
test_obs = v.report(2)
local_obs = v.local_ripl.report(2)
assert( ( [ np.round(test_obs)]*v.no_ripls ) == np.round(ls_obs)  )
assert( ( [np.round(local_obs)]*v.no_ripls ) == np.round(ls_obs)  )

        
def mk2():
    myd['v'] = venture.shortcuts.make_church_prime_ripl()
    return myd['v']
    
def mk():
    global v
    v = venture.shortcuts.make_church_prime_ripl()
    return v

def pred(): return v.predict('(beta 1 1)')

# dview.block = True
# with dview.sync_imports():
#     import venture.shortcuts


# dview.push( { 'v':0 } )
# dview.push( { 'myd':{} } )
#dview.apply(mk2)
