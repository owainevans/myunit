
# #Python Freenode:#Python (channel), minrk or min_rk

from IPython.parallel import Client
from venture.shortcuts import *
import numpy as np
import matplotlib.pylab as plt


### PLAN
# 1. make display work and plot with discrete and cts inputs. 
#   so the crp example (maybe with ellipsoids). 
#     -- should add snapshot (with total transitions), as display will depend on it.

# 3. have the local ripl be optional

#4. map function should intercept infers to update total-transitions

# 5. add all directives
# 6. record no_total_transitions (with snapshot)
# 7. want the user to be able to specify a cluster (a Client() output
# object).
# 8. async should be an option for plotting as you might have lots
# of infers as part of your code, or you might easily (in interaction)
# write somethign that causes crashes. (blocking is enemy of 
# interactive development)

# In Master: clear shouldn't destroy the seed (delegate new seed after clear)

# continuous inference


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
        self.dview.execute(copy_ripl_string) # defines copy_ripl for add_ripl method
        
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
        ## FIXME still has to reset seeds
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
        lst = str + [(d['pid'],d['index'],d['seed']) for d in s]
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

        ## FIXME gotta remove seeds also
        self.update_ripls_info()
        print self.display_ripls()

    
    def snapshot(self,labels_lst, plot=False, scatter_heat=False):
        
        if not(isinstance(labels_lst,list)): labels_lst = [labels_lst] 
        values = { did_label: self.report(did_label) for did_label in labels_lst}
        
        # make sure ripls_info matches up with values
        out = {'values':values,
               'total_transitions':self.total_transitions,
               'ripls_info': self.ripls_location }

        if plot: out['figs'] = self.plot(out,scatter_heat)

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

        
    def plot(self,out,scatter_heat): #values,total_transitions=None,ripls_info=None,scatter_heat=False):
        'values={ did_label: values_list } '
        figs = []
        
        for label,vals in out['values'].items():
            fig,ax = plt.subplots()
            ax.set_xlabel('Variable %s' % str(label))
            ax.set_title('Hist: %s (transitions: %i, ripls: %i' % (str(label), self.total_transitions, self.no_ripls) )
            var_type = self.type_list(vals)
            
            if var_type =='float':
                ax.hist(vals)
                # FIXME add KDE
            elif var_type =='int':
                ax.hist(vals)
            elif var_type =='bool':
                ax.hist(vals)
            else:
                print 'couldnt plot' ##FIXME, shouldnt add fig to figs
            figs.append(fig)
        
        
    
clear_all_engines()
v = MRipl(4)








## clean version pxlocal
def mr_map(line, cell):
    ip = get_ipython()
    ip.run_cell(cell)
    ip.run_cell_magic("px", line, cell)
    f_name_parens = cell.split()[1]
    f_name = f_name_parens[:f_name_parens.find('(')]
    
    res_id = np.random.randint(10**4) # FIXME
    code = 'res_%i = [ %s(ripl) for ripl in ripls]' % (res_id,f_name)
    v.dview.execute(code)
    res = v.dview['res_'+str(res_id)]
    print res
    return res

## Current best version
def pxlocal_line(line, cell):
    ip = get_ipython()
    ip.run_cell(cell)
    ip.run_cell_magic("px", '', cell)
    
    f_name = str(line).split()[0]
    mripl=eval( str(line).split()[1] ) 
    
    res_id = np.random.randint(10**4) # FIXME, should be appending results or adding to dict
    code = 'res_%i = [ %s(ripl) for ripl in ripls]' % (res_id,f_name)
    mripl.dview.execute(code)
    res = mripl.dview['res_'+str(res_id)]
    print 'f_name:',f_name,'m_ripl',line.split()[1],mripl
    return res



def pxlocal(line, cell):
    # one way: define function locally and sent it to all ripls
    #f_name = f_name_parens[:f_name_parens.find('(')]
    #res1 = v.dview.apply_sync(lambda:[func(r) for r in ripls])
    
    # second way: run all the code, which could include various 
    # defines and imports needed for the function, across all engines
    # then execute code that maps the defined function across all ripls
    # in an engine and pull out the resulting object (should be something
    # one can pull).
    ip = get_ipython()
    ip.run_cell(cell)
    ip.run_cell_magic("px", line='',cell=cell) # we remove line, which is normally here
    f_name_parens = cell.split()[1]
    f_name = f_name_parens[:f_name_parens.find('(')]
    if line: 
        f_name = str(line.strip())
    # NOTE that we put random code in if we find function name like this
    # better to use a line
    
    
    res_id = np.random.randint(10**4) # FIXME
    code = 'res_%i = [ %s(ripl) for ripl in ripls]' % (res_id,f_name)
    v.dview.execute(code)
    res = v.dview['res_'+str(res_id)]
    print res
    return res

try:
    ip = get_ipython()
    ip.register_magic_function(mr_map, "cell")   
except:
    print 'no ipython'




    










        

