
# #Python Freenode:#Python (channel), minrk or min_rk

from IPython.parallel import Client
from venture.shortcuts import make_church_prime_ripl
import numpy as np
import matplotlib.pylab as plt
from scipy.stats import kde
gaussian_kde = kde.gaussian_kde

### PLAN

#TODO 1
#- connect to EC2 and create some convenience features for doing so

# - do plotting and unit tests for map, write documentation and examples for it

# - do mapping example for CRP 

# - ask about the namespace issues for multiripls

# - too many methods, find some way to structure better.


# TODO 2
# 1. make display work and plot with discrete and cts inputs. 
#   so the crp example (maybe with ellipsoids). 
#     -- should add snapshot (with total transitions), as display will depend on it.

# 2. probes and scatter?

# 2.5. plotting for map

# - better type mathod

# - work out  

# 3. have the local ripl be optional

#4. map function should intercept infers to update total-transitions

# 5. add all directives

# 7. want the user to be able to specify a cluster (a Client() output
# object).
# 8. async should be an option for plotting as you might have lots
# of infers as part of your code, or you might easily (in interaction)
# write somethign that causes crashes. (blocking is enemy of 
# interactive development)

# 9. In Master: clear shouldn't destroy the seed (delegate new seed after clear)

# 10. continuous inference


# Notes on Parallel IPython

# Questions
# 1. How exactly to enable user to purge data from a process
# and to shutdown processes. Should we allow one engine
# to fail and still get results back from others (vs. just
# having to restart the whole thing again). 

# 2. Basic design idea: only wait on infer. Everything else
# is synchronous. Do we need to give user the option of 
# making everything asynchronous?

# If not: we could set blocking and then have everything 
# be blocking by default (map not map_sync). 

# For infer, we override by running apply_async. We get
# back an async object.  

# if infer is async, need to think about what if someone combines them
# someone does a %v cell magic and it has (infer 10) in it. 
# might need an infer to finish before the predict. so if 
# waiting for an infer to fnish, what will predict do?

# looks life v.predict(1000) waits for v.infer(1000) to finish. is this
# a general rule, that one command will wait for the other? presumably
# otherwise semantics wouldn't work out. 
#  a=v.infer(100000);b= v.predict('100000000000000')


# q: what to do when one or two engines segfault? engines die. how 
# can we recover. engines won't start again. how to properly kill the engines?

# cool stuff: interactive_wait. maybe what we want for stuff involving inference
# cool stuff:  %px from IPython.parallel import bind_kernel; bind_kernel()
#           %px %qtconsole



# seems we can use magic commands in a %px
# and so the engines are running ipython
# --though they don't get the ipy_ripl (why not?)

# question of what happens when you push a function to them
# functions can't be mutated, so a pointer to a function
# should be the same as copying the function, apart from 
# the issue of the enclosing env. so: the function you
# push is like a copy, it doesn't maintain the closure
# (makes sense, coz we can't send across functions with closures)

#e.g. s='local'; f=lambda:s; dv.push({'f':f}); %px f() = error (no s var)

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



make_mripl_string='''
try:
    mripls.append([]); no_mripls += 1; seeds_lists.append([])
except:
    mripls=[ [], ]; no_mripls=1; seeds_lists = [ [], ]'''


def make_mripl_string_function():
    try:
        mripls.append([]); no_mripls += 1; seeds_lists.append([])
    except:
        mripls=[ [], ]; no_mripls=1; seeds_lists = [ [], ]
    

def clear_all_engines():
    cli = Client()
    cli.clear(block=True)

def shutdown():
    cli = Client(); cli.shutdown()
    

class MRipl():
    
    def __init__(self,no_ripls,client=None,name=None):
        self.local_ripl = make_church_prime_ripl()
        self.local_ripl.set_seed(0)   # same seed as first remote ripl
        self.no_ripls = no_ripls
        self.seeds = range(self.no_ripls)
        self.total_transitions = 0
        
        self.cli = Client() if not(client) else client
        self.dview = self.cli[:]
        self.dview.block = True
        def p_getpids(): import os; return os.getpid()
        self.pids = self.dview.apply(p_getpids)
      
        self.dview.execute('from venture.shortcuts import make_church_prime_ripl')
        self.dview.execute(copy_ripl_string) # defines copy_ripl for self.add_ripl method
        
        self.dview.execute(make_mripl_string)
        self.mrid = self.dview.pull('no_mripls')[0] - 1  # all engines should return same number
        name = 'mripl' if not(name) else name
        self.name_rid = '%s_%i' % (name,self.mrid)


        def mk_ripl(seed,mrid):
            ripls = mripls[mrid]
            ripls.append( make_church_prime_ripl() )
            ripls[-1].set_seed(seed)
            
            seeds = seeds_lists[mrid]
            seeds.append(seed)
            
        self.dview.map( mk_ripl, self.seeds, [self.mrid]*self.no_ripls )
        self.update_ripls_info()
        print self.display_ripls()
        
    
    def lst_flatten(self,l): return [el for subl in l for el in subl]

    def clear(self):
        ## FIXME still has to reset seeds. note that resetting seeds means
        ## re-running code after a clear will give identical results (add a 
        # convenient way around this)
        self.total_transitions = 0
        self.local_ripl.clear()
        def f(mrid):
            ripls=mripls[mrid]; seeds=seeds_lists[mrid]
            [ripl.clear() for ripl in ripls]
            [ripls[i].set_seed(seeds[i]) for i in range(len(ripls))]
        return  self.dview.apply(f,self.mrid) 
    
    def assume(self,sym,exp,**kwargs):
        self.local_ripl.assume(sym,exp,**kwargs)
        def f(sym,exp,mrid,**kwargs):
            return [ripl.assume(sym,exp,**kwargs) for ripl in mripls[mrid]]
        return self.lst_flatten( self.dview.apply(f,sym,exp,self.mrid,**kwargs) )
        
    def observe(self,exp,val,label=None):
        self.local_ripl.observe(exp,val,label)
        def f(exp,val,label,mrid): return [ripl.observe(exp,val,label) for ripl in mripls[mrid]]
        return self.lst_flatten( self.dview.apply(f,exp,val,label,self.mrid) )
    
    def predict(self,exp,label=None,type=False):
        self.local_ripl.predict(exp,label,type)
        def f(exp,label,type,mrid): return [ripl.predict(exp,label,type) for ripl in mripls[mrid]]
        return self.lst_flatten( self.dview.apply(f,exp,label,type,self.mrid) )

    def infer(self,params,block=False):
        if isinstance(params,int):
            self.total_transitions += params
        else:
            self.total_transitions += params['transitions']
            ##FIXME: consider case of dict more carefully
        self.local_ripl.infer(params)
        
        def f(params,mrid): return [ripl.infer(params) for ripl in mripls[mrid]]

        if block:
            return self.lst_flatten( self.dview.apply_sync(f,params,self.mrid) )
        else:
            return self.lst_flatten( self.dview.apply_async(f,params,self.mrid) )

    def report(self,label_or_did,**kwargs):
        self.local_ripl.report(label_or_did,**kwargs)
        def f(label_or_did,mrid,**kwargs):
            return [ripl.report(label_or_did,**kwargs) for ripl in mripls[mrid]]
        return self.lst_flatten( self.dview.apply(f,label_or_did,self.mrid, **kwargs) )

    def forget(self,label_or_did):
        self.local_ripl.forget(label_or_did)
        def f(label_or_did,mrid):
            return [ripl.forget(label_or_did) for ripl in mripls[mrid]]
        return self.lst_flatten( self.dview.apply(f,label_or_did,self.mrid) )

    def execute_program(self,  program_string, params=None):
        self.local_ripl.execute_program( program_string, params )
        def f( program_string, params, mrid):
            return  [ripl.execute_program( program_string,params) for ripl in mripls[mrid]]
        return self.lst_flatten( self.dview.apply(f, program_string, params,self.mrid) )

    def get_global_logscore(self):
        self.local_ripl.get_global_logscore()
        def f(mrid):
            return [ripl.get_global_logscore() for ripl in mripls[mrid]]
        return self.lst_flatten( self.dview.apply(f,self.mrid) )

    def sample(self,exp,type=False):
        self.local_ripl.sample(exp,type)
        def f(exp,type,mrid):
               return [ripl.sample(exp,type) for ripl in mripls[mrid] ]
        return self.lst_flatten( self.dview.apply(f,exp,type,self.mrid) )

    def list_directives(self,type=False):
        self.local_ripl.list_directives(type)
        def f(type,mrid):
               return [ripl.list_directives(type) for ripl in mripls[mrid] ]
        return self.lst_flatten( self.dview.apply(f,type,self.mrid) )
               
    def add_ripls(self,no_new_ripls,new_seeds=None):
        'Add no_new_ripls ripls by mapping a copy_ripl function across engines'
        assert(type(no_new_ripls)==int and no_new_ripls>0)

        # could instead check this for each engine we map to
        pids_with_ripls = [ripl['pid'] for ripl in self.ripls_location]
        if any([pid not in pids_with_ripls for pid in self.pids]):
            print 'Error: some engines have no ripl, add_ripls failed'
            return None

        if not(new_seeds):
            # want to ensure that new seeds are different from all existing seeds
            next = max(self.seeds) + 1
            new_seeds = range( next, next+no_new_ripls )

        def add_ripl_engine(seed,mrid):
            # load the di_list from an existing ripl from ripls
            # we only set_seed after having loaded, so all ripls
            # created by a call to add ripls may have same starting values
            ripls=mripls[mrid]; seeds=seeds_lists[mrid]
            ripls.append( copy_ripl(ripls[0]) ) # ripls[0] must be present
            ripls[-1].set_seed(seed)
            seeds.append(seed)
            import os;   pid = os.getpid();
            print 'Engine %i created ripl %i' % (pid,seed)
            
        self.dview.map(add_ripl_engine,new_seeds,[self.mrid]*no_new_ripls)
        self.update_ripls_info()
        print self.display_ripls()
    
    def display_ripls(self):
        s= sorted(self.ripls_location,key=lambda x:x['pid'])
        ##FIXME improve output
        key = ['(pid, index, seed)'] 
        lst = key + [(d['pid'],d['index'],d['seed']) for d in s]
        #[ ('pid:',pid,' seeds:', d['seed']) for pid in self.pids for d
        return lst

    def update_ripls_info(self):
        'nb: reassigns attributes that store state of pool of ripls'
        def get_info(mrid):
            import os; pid=os.getpid()
            
            return [ {'pid':pid, 'index':i, 'seed':seeds_lists[mrid][i],
                      'ripl':str(ripl)  }  for i,ripl in enumerate( mripls[mrid] ) ]
        self.ripls_location = self.lst_flatten( self.dview.apply(get_info,self.mrid) )
        self.no_ripls = len(self.ripls_location)
        self.seeds = [ripl['seed'] for ripl in self.ripls_location]

        
    def remove_ripls(self,no_rm_ripls):
        'map over the engines to remove a ripl if they have >1'
        no_removed = 0
        def check_remove(x,mrid):
            ripls = mripls[mrid]
            if len(ripls) >= 2:
                ripls.pop()
                return 1
            else:
                return 0
       
        while no_removed < no_rm_ripls:
            result = self.dview.map(check_remove,[1]*no_rm_ripls, ([self.mrid]*no_rm_ripls))
            no_removed += len(result)

        ## FIXME should also remove seeds
        self.update_ripls_info()
        print self.display_ripls()

    
    def snapshot(self,labels_lst, plot=False, scatter=False, logscore=False):
        
        if not(isinstance(labels_lst,list)): labels_lst = [labels_lst] 
        values = { did_label: self.report(did_label) for did_label in labels_lst}
        
        if logscore: values['log']= self.get_global_logscore()
        
        out = {'values':values,
               'total_transitions':self.total_transitions,
               'ripls_info': self.ripls_location }

        if plot: out['figs'] = self.plot(out,scatter=scatter)

        return out


    def type_list(self,lst):
        if any([type(lst[0])!=type(i) for i in lst]):
            return 'mixed'
        elif isinstance(lst[0],float):
            return 'float'
        elif isinstance(lst[0],int):
            return 'int'
        elif isinstance(lst[0],bool):
            return 'bool'
        elif isinstance(lst[0],str):
            return 'string'
        else:
            return 'other'

        
    def plot(self,snapshot,scatter=False): #values,total_transitions=None,ripls_info=None,scatter_heat=False):
        'values={ did_label: values_list } '
        figs = []
        values = snapshot['values']
        no_trans = snapshot['total_transitions']
        no_ripls = self.no_ripls
        
        for label,vals in values.items():

            var_type = self.type_list(vals)
            
            if var_type =='float':
                fig,ax = plt.subplots(nrows=1,ncols=2,sharex=True,figsize=(9,4))
                xr = np.linspace(min(vals),max(vals),400)
                ax[0].plot(xr,gaussian_kde(vals)(xr))
                ax[0].set_xlim([min(vals),max(vals)])
                ax[0].set_title('Gaussian KDE: %s (transitions: %i, ripls: %i)' % (str(label), no_trans, no_ripls) )

                ax[1].hist(vals)
                ax[1].set_title('Hist: %s (transitions: %i, ripls: %i)' % (str(label), no_trans, no_ripls) )
                [a.set_xlabel('Variable %s' % str(label)) for a in ax]
            
            elif var_type =='int':
                fig,ax = plt.subplots()
                ax.hist(vals)
                ax.set_xlabel = 'Variable %s' % str(label)
                ax.set_title('Hist: %s (transitions: %i, ripls: %i)' % (str(label), no_trans, no_ripls) )
                
            elif var_type =='bool':
                ax.hist(vals)
            else:
                print 'couldnt plot' ##FIXME, shouldnt add fig to figs
            fig.tight_layout()
            figs.append(fig)

            
        if scatter:
            label0,vals0 = values.items()[0]
            label1,vals1 = values.items()[1]
            fig, ax  = plt.subplots(figsize=(6,4))
            ax.scatter(vals0,vals1)
            ax.set_xlabel(label0); ax.set_ylabel(label1)
            ax.set_title('%s vs. %s (transitions: %i, ripls: %i)' % (str(label0),str(label1),
                                                                    no_trans, no_ripls) )
            figs.append(fig)
        
        return figs


    def probes(self,did_label,no_transitions,no_probes,plot_hist=None,plot_series=None):
        label = did_label
        start = self.total_transitions
        probes = map(int,np.round( np.linspace(0,no_transitions,no_probes) ) )
        
        series = [self.snapshot(label)['values'][label], ]
        for i in range(len(probes[:-1])):
            self.infer(probes[i+1]-probes[i])
            series.append( self.snapshot(label)['values'][label] )

        if plot_hist:
            xmin = min([min(shot) for shot in series])
            xmax = max([max(shot) for shot in series])
            xr = np.linspace(xmin,xmax,400)
            fig,ax = plt.subplots(ncols=no_probes,sharex=True,figsize=(10,5))
            kdfig,kdax = plt.subplots(ncols=no_probes,sharex=True,figsize=(10,5))
            for i in range(no_probes):
                ax[i].hist(series[i],bins=12)
                kdax[i].plot(xr,gaussian_kde(series[i])(xr))
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
    


def mk_map_proc_string(mripl_name,mrid,proc_name):
    lhs = 'results[-1] = '
    subs = (mripl_name, mrid, proc_name, proc_name, mrid)
    rhs = '{"mripl_mrid_proc":("%s",%i,"%s"), "output": [%s(r) for r in mripls[%i]] }' % subs
    return lhs+rhs
    
add_results_list_string = '''
try: results.append([])
except: results=[ [], ]'''

def add_results_list():
    try: results.append([])
    except: results=[ [], ]

plotting_string = '''
import matplotlib.pylab as plt
%matplotlib inline'''

def mr_map(line, cell):
    '%mr_map proc_name mripl_name'
    ip = get_ipython()
    ip.run_cell(cell)  # run cell locally, for local ripl
    ip.run_cell_magic("px", '', cell)
    ip.run_cell_magic("px", '', plotting_string)  # import plt and select inline
    
    proc_name = str(line).split()[0]
    mripl_name =  str(line).split()[1]
    mripl = eval(mripl_name,globals(),ip.user_ns)
    ## FIXME: what should globals,locals be for this eval?
    mrid = mripl.mrid

    mripl.dview.execute(add_results_list_string)
    
    map_proc_string = mk_map_proc_string(mripl_name,mrid,proc_name)
    mripl.dview.execute(map_proc_string)

    result = mripl.dview.apply( lambda: results[-1])
    label = result[0]['mripl_mrid_proc']
    engs = [eng['output'] for eng in result]
    results_by_ripl = [ ripl for ripls in engs for ripl in ripls]
    return label,results_by_ripl


## Consider Version where we use execute instead of %px
    # one way: define function locally and sent it to all ripls
    #f_name = f_name_parens[:f_name_parens.find('(')]
    #res1 = v.dview.apply_sync(lambda:[func(r) for r in ripls])
    
    # second way: run all the code, which could include various 
    # defines and imports needed for the function, across all engines
    # then execute code that maps the defined function across all ripls
    # in an engine and pull out the resulting object (should be something
    # one can pull).

try:
    ip = get_ipython()
    ip.register_magic_function(mr_map, "cell")    
except:
    print 'no ipython'

def sp(no_ripls=2):    
    v = MRipl(no_ripls)
    v.assume('r','(normal 0 30)',label='r')
    v.assume('s','(normal 0 30)',label='s')
    v.assume('w1','(normal (+ r s) 5.)',label='w1')
    v.observe('w1','50.')
    v.assume('w2','(normal (+ r s) 6.)',label='w2')
    v.observe('w2','50.')
    return v

def crp(no_ripls=2):
    prog='''
    [assume alpha (uniform_continuous .9 1)]
    [assume crp (make_crp alpha) ]
    [assume z (mem (lambda (i) (crp) ) ) ]
    [assume mu (mem (lambda (z dim) (normal 0 30) ) ) ] 
    [assume sig (mem (lambda (z dim) (uniform_continuous .1 1) ) ) ]
    [assume x (mem (lambda (i dim) (normal (mu (z i) dim) (sig (z i) dim)))  ) ]'''
    v=make_church_prime_ripl()
    v.execute_program(prog)
    n=100
    xs = [v.predict('(x ' + str(i) + ' ' +str(j) +')') for i in range(n) for j in range(2) ]
    obs = [v.observe('(x ' + str(i) + ' ' +str(j) +')', str(xs[i*(j+1)]) ) for i in range(n) for j in range(2) ]
    x=np.array(xs)[range(0,len(xs),2)]
    y=np.array(xs)[range(1,len(xs),2)]
    
    
    
    return prog,v,xs,x,y

    
# prog,vsingle,xs,x,y = crp(2)
# v=MRipl(2)
# v.execute_program(prog)
# ns=100
# s0 = [v.predict('(x %i 0)' % i) for i in range(ns) ]
# s1 = [v.predict('(x %i 1)' % i) for i in range(ns) ]
# sz = [v.predict('(z %i)' % i) for i in range(ns) ]


# nor = np.random.normal
# xs = list(nor(0,1,20)) + list(nor(5,3,20))
# ys = list(nor(0,1,20)) + list(nor(5,3,20))
# [v.observe( '(x %i 0)' % i, str(x[i]), label=( 'x_%i0' % i) ) for i in range(len(xs))]
# [v.observe( '(x %i 1)' % i, str(x[i]), label=( 'x_%i1' % i) ) for i in range(len(ys))]

# v.infer(5)
# pz = [v.predict('(z %i)' % i) for i in range(ns) ]
# fig,ax = plt.subplots(2,1)
#ax[0].scatter(s0,s1,c=sz)
#ax[0].scatter(xs,ys,c=pz)

#plt.show()


# %%mr_map crp_plot v
# n=20
# def crp_plot(v):
#     x0 = [v.sample( '(x %i 0)' % i ) for i in range(n) ]
#     x1 = [v.sample( '(x %i 1' % i ) for i in range(n) ]
#     zs = [v.sample('(z %i)' % i) for i in range(n) ]
#     fig,ax = plt.subplots(2)
#     ax[0].scatter(x0,x1,c=zs)
    
#     px0 = [v.predict( '(x %i 0)' % i ) for i in range(n,2*n) ]
#     px1 = [v.predict( '(x %i 1)' % i ) for i in range(n,2*n) ]
#     pzs = [v.predict('(z %i)' % i) for i in range(n,2*n) ]
#     ax[1].scatter(px0,px1,c=pzs)
#     return zs,pzs



#clear_all_engines()
# v=sp(4)

# res=ip.run_cell_magic("mr_map",'foo v','def foo(r): return r.predict("r")')

    










        

