from IPython.core.magic import (Magics,register_cell_magic, magics_class, line_magic,cell_magic, line_cell_magic)

import numpy as np  
from my_unit import *
import os,time,multiprocessing
from venture.shortcuts import *
ipy_ripl = make_church_prime_ripl()


############# Functions for multiprocess inference        


#FIXME need to work out how long to sleep    
def make_ripl():
    'sleep before making'
    time.sleep(1)
    return make_church_prime_ripl()


def build_class(py_lines,rand=False):
    'takes in lines of ripl.assume(...) code and builds class for model'
    
    class MyModelClass(VentureUnit):
        ra = lambda: np.random.randint(100); beta=lambda:np.random.beta(1,1)

        def makeAssumes(self):
            ## FIXME hack to get around fixed seed for cxx
            #self.assume('a','(beta 1 '+str(np.random.randint(100)) + ')')
            [eval(py_line) for py_line in py_lines if py_line[5]=='a']

        def makeObserves(self):
           # [self.observe("(normal a 1)","1") for i in range(5) ]
            [eval(py_line) for py_line in py_lines if py_line[5]=='o']


            # Example unit input from LDA
            # D = self.parameters['documents']
            # N = self.parameters['words_per_document']
        
            # for doc in xrange(D):
            #     for pos in xrange(N):
            #         self.observe("(get-word %d %d)" % (doc, pos), 0)

    return MyModelClass

# plan:
# you get a ripl from the entered program. from this you can get the dire
# need to be careful about the namespace issues.
# the workers get a ripl and a dict of directives. they run the directives
# (which just involves running methods on the ripl and using functions
# like build_py_dire that they gotta have in their namespace. 
#  then they run infer,logscore, report from their ripl.

# other crucial thing is the seediness issue. we can generate seeds for 
# each worker from the parent (so easy to lookup and change). workers
# then will have different seeds for each run (if they have multiple runs)
# which are their base seed + the run_id. 

## storage. worker is given an instruction for how much to store, e.g.
## 100 sweeps and 4 probes (evenly spaced). worker must store logscore
## and all the labeled variables (possibly ignoring observes for speed).
# e.g. 
##   could loop over the directive_list, then v.report(item['label'])
#   (possibly filtering out observes)
#   something like
history = { label:(vals_list,(probes,total samples) }
or      = { series={label:vals}, samples, probes, random_seed, worker_id }

# need to think about parallelizing to clusters and doing plots across
# these objects. the plots need to know the no-samples and probes, and 
# so some need to include those with every series (mb). 
# 


## load_program with %v onto ipy ripl
v=ipy_ripl()
no_samples = 100
no_probes = 5 # 0,25,50,75,100
# could let the input be a range, which we use for seeds
workers_range = (1,30)
nprocs = 30 # what's plausible? would like to be able to add
# more in an interactive way, if approx not good enough for us
# would want to add more with a different seed. so we might
# want to have an ordering for each worker, with a corresponding seed
# so that after doing 30, we could easily do another 20 new chains


# want to return snapshots from each worker at the probes
# so want a set of vals at each stage and then to plot posterior estimate

no_ripls = 4
ripls = [make_church_prime_ripl() for i in range(no_ripls)]

# note, want possibility of varying model for different workers
directives_list = [v.list_directives()] * no_ripls

np.random.seed(1000)
# would restrict rhs to workers_range above
seeds = np.random.randint(10**9,size = no_ripls)

worker_names = ['w'+str(i)+'_'+str(seeds[i]) for i in range(len(seeds) ) ]

# q = Queue(size)

for i in range(no_procs):
    # mk_worker( ripls[i],directives_list[i],seeds[i],worker_names[i],
    #            q, no_samples,no_probes,infer_msg,plot_msg )
    
    



def p_worker(ripl,di_list,seed,name,q,no_samples,no_probes,infer_msg,plot_msg):
    ripl.set_seed(seed)

    shots = {'worker_name'=name,
             'no_samples':no_samples,'no_probes':no_probes, #infer,plot,seed}
             'snapshots'=[] }
    snapshot = ( {labels:values}, no_samples, no_probes, sample_number )

    
        
    r.assume(blah)
    r.observe(blah)

    for i in no_samples/no_probes:
        infer(k)
        { label:v.report(label) for label in directives.keys() }
         + logscore

def build_py_directive(ripl,d):
    # add label if lacking one
    if not 'label' in d:
        if d['instruction']=='assume':
            d['label'] = d['symbol']
        elif d['instruction']=='observe':
            d['label'] = 'obs_'+d['directive_id']
        elif d['instruction']=='predict':
            d['label'] = 'pre_'+d['directive_id']

    if d['instruction']=='assume':
        ripl.assume( d['symbol'], build_exp(d['expression']), label=d['label'] )
    elif d['instruction']=='observe':
        ripl.observe( build_exp(d['expression']), label=d['label'] )
    elif d['instruction']=='predict':
        ripl.predict( build_exp(d['expression']), label=d['label'] )
    
    
def build_exp(exp):
    if type(exp)==str:
        return exp
    elif type(exp)==dict:
        return str(exp['value'])
    else:
        return '('+str(exp[0])+' ' + ' '.join(map(build_exp,exp[1:])) + ')'

## test
v = make_church_prime_ripl()
%v [assume xy ( beta 1   1)]
%v [observe (normal xy 1) 5 ]
%v [predict (beta xy (+ 1 5) ) ]
v2 = make_church_prime_ripl()
for d in v.list_directives():
    build_py_directive(v2,d)
print v.list_directives()
print v2.list_directives()
        



def worker(out_q,ModelClass,ripl,params,no_sweeps,infer_msg,plot_msg):
    print 'Starting:', multiprocessing.current_process().name
    print '--------------'
    unit_model = ModelClass(ripl,params)
    print unit_model.assumes
    if infer_msg=='runFromConditional':
        hist = unit_model.runFromConditional(sweeps=no_sweeps,runs=1)
        out_q.put(hist)
    print 'Exiting:', multiprocessing.current_process().name 



def multi_ripls(no_ripls,ModelClass_list,no_sweeps_list,infer_msg,plot_msg):

    # Parent code that runs the worker processes
    # do i need 'name'='main'?. we just run the script. worry that 
    # interactive will make it not work. look up again what interactive
    # entails for multiprocess. if we ran it, workers would have access
    # to a copy of the name space of their encloding env? key thing is 
    # we wouldn't want the workers to generate new child workers, which
    # we would get if we ran whole module. but here the code that generates
    # the workers is not in the global env of script. (we don't want the 
    # workers to run the ipython loading code, however, and so need to 
 owain   # watch out for that ... but their having multi_ripls in namespace
    # or the ipython magics, shouldn't itself be a problemo. 

    assert(len(no_sweeps_list)==no_ripls)
    assert(isinstance(infer_msg,str))
    # for mc in ModelClass_list:
    #     test_model = mc(make_ripl(),{})
    #     print 'assumes:',test_model.assumes,'\n','observes:',test_model.observes
    #     assert(test_model.assumes != [])
    
    nprocs= no_ripls; procs = []
    out_q = multiprocessing.Queue() # could add max-size
    hists = []
    ripls = [make_ripl() for i in range(no_ripls)]
    # make seeds for each ripl distinct
    np.random.seed(1000)
    seeds = map(int, np.random.randint(10**8,size=no_ripls))
    [ripl.set_seed(seed) for ripl,seed in zip(ripls,seeds) ]
    print seeds
    print [ripl.predict('(beta 1 1)') for ripl in ripls]
    params = [ {'venture_random_seed':seed} for seed in seeds ]
    print params
  

    for i in range(nprocs):
        mytarget = worker
        myargs=(out_q, ModelClass_list[i], ripls[i],params[i],
                no_sweeps_list[i],infer_msg,plot_msg)

        p = multiprocessing.Process(target=mytarget,args=myargs)
        
        procs.append(p)
        p.start()

    for p in procs:
        p.join()

    while not(out_q.empty()):
        hists.append(out_q.get())
                                        
    return hists


    
## Utility functions for pulling out tags                                          ## FIXME, all we need is tags and so this could be much shorter                                            
def remove_white(s):
    t=s.replace('  ',' ')
    return s if s==t else remove_white(t)

def cell_to_venture(s,terse=0):
    """Converts vchurch to python self.v.assume (OBSERVE is broken)"""
    s = str(s)
    s = s[:s.rfind(']')]
    ls = s.split(']')
    ls = [remove_white(line.replace('\n','')) for line in ls]

    # venture lines in python form, and dict of components of lines
    v_ls = []
    v_ls_d = {}

    for count,line in enumerate(ls):
        if terse==1: line = '[ASSUME '+line[1:]

        lparen = line.find('[')        
        tag = line[ lparen+1: ].split()[0].lower()

        if tag=='assume':
            var=line[1:].split()[1]
            exp = ' '.join( line[1:].split()[2:] )
            v_ls.append( "self.v.assume('%s', '%s')" % ( var, exp ) )
            v_ls_d[count] = (tag,var,exp)

        elif tag=='observe':
            var=line[1:].split()[1]
            exp = ' '.join( line[1:].split()[2:] )
            v_ls.append( "self.v.observe('%s', '%s')" % ( var, exp ) )
            v_ls_d[count] = (tag,var,exp)
        elif tag=='predict':
            exp = ' '.join( line[1:].split()[1:] )
            v_ls.append( "self.v.predict('%s')" % exp  )
            v_ls_d[count] = (tag,exp)
        elif tag=='infer':
            num = line[1:].split()[1]
            v_ls.append( "self.v.infer(%i)" % int(num) )
            v_ls_d[count] = (tag,num)
        elif tag=='clear':
            v_ls.append( "self.v.clear()" )
            v_ls_d[count] = (tag,'')   # comes with empty string for simplcity
        else:
            assert 0==1,"Did not recognize directive"

    #make tag upper
    for key in v_ls_d.keys():
        old = v_ls_d[key]
        v_ls_d[key] = (old[0].upper(),) + old[1:]

    return v_ls,v_ls_d


        
@magics_class
class ParaMagics(Magics):

    def __init__(self, shell):
        super(ParaMagics, self).__init__(shell)

    @line_cell_magic
    def vl2(self, line, cell=None):
        def format_parts(parts):
            'format the input string for pretty printing'
            return '[%s]' % ' '.join(parts)
        
        ## LINE MAGIC
        if cell is None:
            vouts = ipy_ripl.execute_instruction(str(line), params=None)

            py_lines,py_parts = cell_to_venture(line)
            
            for key in py_parts:
                print format_parts(py_parts[key])
                 
            if 'value' in vouts: print vouts['value'].get('value',None)

            return vouts
            
        ## CELL MAGIC    
        else:
            vouts = ipy_ripl.execute_program( str(cell), params=None )

            py_lines,py_parts = cell_to_venture(cell)
                              
            for count,v_line in enumerate(vouts):
                print format_parts(py_parts[count])
                if 'value' in v_line: print v_line['value'].get('value',None)

            return vouts
    

    @cell_magic
    def p(self, line, cell):
        'need to fix OBS issue'

        # Convert Venchurch to python directives
        py_lines,py_parts = cell_to_venture(cell)
        py_lines = [ py_line.replace('self.v.','self.') for py_line in py_lines]

        # Use line input to determine no_ripls
        try:
            no_ripls,no_sweeps = map(int,str(line).split(','))
        except: no_ripls, no_sweeps = 1,100
        print 'using %i explanations' % no_ripls

        # call multiprocess inference and plotting on the ripls
        no_sweeps_list = [no_sweeps]*no_ripls
        infer_msg = 'runFromConditional'                                        
        plot_msg = None
        ModelClass_list = [build_class(py_lines,rand=True) for i in range(no_ripls)]

        hists = multi_ripls(no_ripls, ModelClass_list,
                            no_sweeps_list,infer_msg,plot_msg)
        for h in hists:
            try: h.label = hists[0].label
            except: print 'Error calling hist.label'
        return hists
                            


    
    
    
    
    ## for ipythonNB, remove function defintion and uncomment following two lines
def load_ipython_extension(ip):
    """Load the extension in IPython."""
    ip.register_magics(ParaMagics)
    print 'loaded ParaMagics'
try:
    ip = get_ipython()
    load_ipython_extension(ip)
except:
    print 'failed to load'
    #     ip.register_magics(VentureMagics)
#     ip_register_success = 1

