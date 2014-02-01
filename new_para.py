from venture.shortcuts import *
import numpy as np
import matplotlib.pylab as plt
import multiprocessing,time
from multiprocessing import Pool
from scipy.stats import kde
gkde = kde.gaussian_kde

from bokeh.plotting import *

ripl = make_church_prime_ripl()
worker_state = {'init':False}


def worker(st_func_args):
    ## ripl.clear() resets seed and would set all workers to same seed
    # so we check for a reset with list_directives
    if not(worker_state['init']) or ripl.list_directives()==[]:
        p_name = multiprocessing.current_process().name
        name = 'w'+ p_name[p_name.rfind('-')+1:]
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

    def snapshots(self,labels):
        return [ (label,snapshot(label)) for label in labels]

        
    def snapshot(self,label,plot=False):
        out = self.report(label)
        s_out = sorted(out,key=lambda pair: int(pair[0][1:]))
        vals =  [ pair[1] for pair in s_out]
        if plot:
            fig, ax  = plt.subplots(figsize=(10,6))
            ax.hist(vals)
            ax.set_xlabel(label)
            ax.set_title('%s at %i transitions' % (str(label),
                                                   self.total_transitions) )
            return vals,fig
        return vals


    def scatter(self,label1,label2):
        vals1 = self.snapshot(label1)
        vals2 = self.snapshot(label2)
        fig, ax  = plt.subplots(figsize=(10,6))
        ax.scatter(vals1,vals2)
        ax.set_xlabel(label1); ax.set_ylabel(label2)
        ax.set_title('%s vs. %s at %i transitions' % (str(label1),str(label2),
                                                      self.total_transitions) )
        return (vals1,vals2), fig
        
    

    def probes(self,label,no_transitions,no_probes,plot_hist=None,plot_series=None):
        start = self.total_transitions
        probes = map(int,np.round( np.linspace(0,no_transitions,no_probes) ) )

        series = [self.snapshot(label), ]
        for i in range(len(probes[:-1])):
            self.infer(probes[i+1]-probes[i])
            series.append( self.snapshot(label) )
            
        if plot_hist:
            xmin = min([min(shot) for shot in series])
            xmax = max([max(shot) for shot in series])
            xr = np.linspace(xmin,xmax,400)
            fig,ax = plt.subplots(ncols=no_probes,sharex=True,figsize=(12,6))
            kdfig,kdax = plt.subplots(ncols=no_probes,sharex=True,figsize=(12,6))
            for i in range(no_probes):
                ax[i].hist(series[i],bins=12)
                kdax[i].plot(xr,gkde(series[i])(xr))
                ax[i].set_xlim([xmin,xmax])
                t = '%s: start %i, probe at %i of %i' % (str(label),
                                                               start,probes[i],
                                                               no_transitions)
                ax[i].set_title(t); kdax[i].set_title(t)
            
            fig.tight_layout(); kdfig.tight_layout()

        if plot_series:
            fig,ax = plt.subplots()
            for ripl in range(self.no_ripls):
                vals = [shot[ripl] for shot in series]
                ax.plot(probes,vals,label='R'+str(ripl))

            t = '%s: start %i, probes at %s' % (str(label),start,str(probes))
            ax.set_title(t)
            #ax.legend()

        return probes,series

        
    def terminate(self):
        self.pool.terminate()
        print 'terminated all ripl processes'
        
        
def wm(no_ripls):
    v = MultiRipl(no_ripls)
    v.assume('r','(normal 0 30)')
    v.assume('s','(normal 0 30)')
    v.assume('w','(normal (+ r s) 5.)')
    v.observe('w','50.')
    v.assume('w2','(normal (+ r s) 6.)')
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



# def snapshot(label,no_transitions,no_ripls):
#     my = MultiRipl(no_ripls)
    
#     my.assume("mu","(normal 0 20)",label='mu');
#     my.observe('(normal mu 5.)', '7.' )
#     my.observe('(normal mu 5.)', '6.5' )

#     my.infer(no_transitions)

#     out = my.report(label)
#     s_vals = sorted(out,key=lambda pair: int(pair[0][1:]))
#     return s_vals

