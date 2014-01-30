from venture.shortcuts import *
import numpy as np
import matplotlib.pylab as plt
import multiprocessing,time
from multiprocessing import Pool

ripl = make_church_prime_ripl()
worker_state = {'init':False}


def worker(st_func_args):
    ## ripl.clear() resets seed and would set all workers to same seed
    # so we check for a reset with list_directives
    if not(worker_state['init']) or ripl.list_directives()==[]:
        p_name = multiprocessing.current_process().name
        name = 'w'+ p_name[p_name.rfind('-')+1:]
        print 'Init: ', name,'.'
        worker_state['name'] = name
        
        worker_state['seed'] = int(name[1:])
        ripl.set_seed( worker_state['seed'] )
        
        worker_state['init'] = True


    return worker_state['name'], eval(st_func_args)


    
class MultiRipl():
        
    def __init__(self,no_ripls):
        self.pool = Pool(processes = no_ripls)
        self.no_ripls = no_ripls
        self.total_transitions = 0

        ## FIXME maybe clear ripl on init
        print 'New MultiRipl with %i ripls' % self.no_ripls
    
    def clear(self):
        return self.delegate('ripl.clear()')

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
        self.total_transitions += no_transitions
        st = 'ripl.infer(%i)' % no_transitions
        return self.delegate(st)

    def report(self,inp):
        if type(inp)==str:
            st = 'ripl.report(" %s ")' % inp
        else:
            st = 'ripl.report( %i )' % inp 
        return self.delegate(st)

    def execute_program(self,prog):
        st = 'ripl.execute_program(" %s ")' % prog
        return self.delegate(st)

    def s(self,st): return self.delegate("ripl."+st)

    def delegate(self,st):
        args = [st] * self.no_ripls
        out = self.pool.map(worker, args )
        return out
        
    def snapshot(self,label,plot=False):
        out = self.report(label)
        s_out = sorted(out,key=lambda pair: int(pair[0][1:]))
        vals =  [ pair[1] for pair in s_out]
        if plot:
            plt.hist(vals,bins=20)
            plt.show()
        return vals

    def scatter(self,label1,label2):
        vals1 = self.snapshot(label1)
        vals2 = self.snapshot(label2)
        plt.scatter(vals1,vals2)
        plt.show()

    def probes(self,label,no_transitions,no_probes,plot=False):
        step = int(round (float(no_transitions+1) / no_probes) )
        probes = range(0,no_transitions+step,step)

        series = [];
        for probe in probes:
            self.infer(step)
            series.append( self.snapshot(label) )
            
        if plot=='hist':
            no_probes = len(probes)
            xmin = min([min(shot) for shot in series])
            xmax = max([max(shot) for shot in series])
            fig,ax = plt.subplots(ncols=no_probes,sharex=True,figsize=(12,5) )
            for i in range(no_probes):
                ax[i].hist(series[i],bins=20)
                ax[i].set_xlim([xmin,xmax])
                ax[i].set_title('Probe at %i out of %i' % (probes[i],no_transitions+step))
                
            fig.tight_layout()

        if plot=='series':
            fig,ax = plt.subplots()
            for ripl in range(self.no_ripls):
                vals = [shot[ripl] for shot in series]
                ax.plot(probes,vals,label='R'+str(ripl))
            ax.legend()

        return probes,series
        
    def terminate(self):
        self.pool.terminate()
        print 'terminated all ripl processes'
        
        
def wm(no_ripls):
    v = MultiRipl(no_ripls)
    v.clear()
    v.assume('r','(normal 0 30)')
    v.assume('s','(beta 1 1)')
    v.assume('w','(normal r 6.)')
    v.observe('w','50.')
    v.assume('w2','(normal r 6.)')
    v.observe('w2','50.')
   
    #v.infer(1)
    return v









# my = MultiRipl(10)
# a=[0]*10
# a[0] = my.assume("mu","(normal 0 20)",label='mu');
# a[1] = my.observe('(normal mu 5.)', '7.' )
# a[1] = my.observe('(normal mu 5.)', '6.5' )
# a[3] = my.report('mu')
# a[4] = my.infer(400)
# a[5] = my.report('mu')

# sprinkler = '''
# [clear]
# [assume rain (bernoulli .1) ]
# [assume spr (bernoulli .4) ]
# [assume wet (if (* rain spr)
#                  .9
#                  (if spr .7 (if rain .5 .05) ) ) ]'''

# spr = '''
# [clear]
# [assume rain (beta 1 1)]
# [assume spr (beta 1 1) ]
# [assume wet (lambda () (if (flip rain) true
#                             (if (flip spr) true (flip .1) ) ) )]
# '''
# spr = ' '.join(spr.split('\n'))

# spr = '''
# [clear]
# [assume rain (beta 1 1)]
# [assume spr (beta 1 1) ]
# [assume w1 (normal (+ rain spr) .1) ]
# [observe w1 1.3]

# '''



def snapshot(label,no_transitions,no_ripls):
    my = MultiRipl(no_ripls)
    
    my.assume("mu","(normal 0 20)",label='mu');
    my.observe('(normal mu 5.)', '7.' )
    my.observe('(normal mu 5.)', '6.5' )

    my.infer(no_transitions)

    out = my.report(label)
    s_vals = sorted(out,key=lambda pair: int(pair[0][1:]))
    return s_vals

