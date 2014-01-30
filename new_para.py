from venture.shortcuts import *
import numpy as np
import matplotlib.pylab as plt
import multiprocessing,time
from multiprocessing import Pool
import time

ripl = make_church_prime_ripl()

worker_initialized = False
worker_state = {'init':False}


def worker(st_func_args):
    if not(worker_state['init']):
        p_name = multiprocessing.current_process().name
        name = 'w'+ p_name[p_name.rfind('-')+1:]
        print 'initializing ', name, '\n'
        worker_state['name'] = name
        
        worker_state['seed'] = int(name[1:])
        ripl.set_seed( worker_state['seed'] )
        
        worker_state['init'] = True

    name = worker_state['name']

    return name, eval(st_func_args)


    
class MultiRipl():
        
    def __init__(self,no_ripls):
        self.pool = Pool(processes = no_ripls)
        self.no_ripls = no_ripls

    def observe(self,exp,value):
        st =  'ripl.observe(" %s ", " %s " )' % (str(exp), str(value) )
        return self.delegate(st)
    
    def predict(self,args):
        st =  'ripl.predict(" %s " )' % str(args)
        return self.delegate(st)
    
    def assume(self,sym,exp,label=None):
        if label:
            st = 'ripl.assume(" %s ", " %s ",label="%s")' % ( str(sym),
                                                             str(exp),label)
        else:
            st = 'ripl.assume(" %s ", " %s " )' % (str(sym), str(exp) )
        return self.delegate(st)

    def infer(self,no_transitions):
        st = 'ripl.infer(%i)' % no_transitions
        return self.delegate(st)

    def report(self,inp):
        if type(inp)==str:
            st = 'ripl.report(" %s ")' % inp
        else:
            st = 'ripl.report( %i )' % inp 
        return self.delegate(st)

        
    def s(self,st):
        return self.delegate("ripl."+st)
        
    def delegate(self,st):
        args = [st] * self.no_ripls
        out = self.pool.map(worker, args )
        return out
        
    def snapshot(self,label):
        out = self.report(label)
        s_out = sorted(out,key=lambda pair: int(pair[0][1:]))
        vals =  [ pair[1] for pair in s_out]
        plt.hist(vals)
        plt.show()
        return vals

        
my = MultiRipl(10)

a=[0]*10
a[0] = my.assume("mu","(normal 0 20)",label='mu');
a[1] = my.observe('(normal mu 5.)', '7.' )
a[1] = my.observe('(normal mu 5.)', '6.5' )
a[3] = my.report('mu')
a[4] = my.infer(400)
a[5] = my.report('mu')


def snapshot(label,no_transitions,no_ripls):
    my = MultiRipl(no_ripls)
    
    my.assume("mu","(normal 0 20)",label='mu');
    my.observe('(normal mu 5.)', '7.' )
    my.observe('(normal mu 5.)', '6.5' )

    my.infer(no_transitions)

    out = my.report(label)
    s_vals = sorted(out,key=lambda pair: int(pair[0][1:]))
    return s_vals

# s_vals = snapshot('mu',100,10)
# se = [ pair[1] for pair in s_vals]
