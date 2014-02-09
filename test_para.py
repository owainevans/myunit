from venture.shortcuts import *
import numpy as np
import multiprocessing,time

## LATEST VERSION

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
#history = { label:(vals_list,(probes,total samples) }
#or      = { series={label:vals}, samples, probes, random_seed, worker_id }

# need to think about parallelizing to clusters and doing plots across
# these objects. the plots need to know the no-samples and probes, and 
# so some need to include those with every series (mb). 
# 


## load_program with %v onto ipy ripl

# could let the input be a range, which we use for seeds
workers_range = (1,30)
nprocs = 30 # what's plausible? would like to be able to add
# more in an interactive way, if approx not good enough for us
# would want to add more with a different seed. so we might
# want to have an ordering for each worker, with a corresponding seed
# so that after doing 30, we could easily do another 20 new chains


# want to return snapshots from each worker at the probes
# so want a set of vals at each stage and then to plot posterior estimate


def run_py_directive(ripl,d):
    '''Takes ripl and directive, labels directive (if no label),
    runs directive on ripl'''
    
    # add labels
    if not 'label' in d:
        if d['instruction']=='assume':
            d['label'] = d['symbol']
        elif d['instruction']=='observe':
            d['label'] = 'obs_'+str(d['directive_id'])
        elif d['instruction']=='predict':
            d['label'] = 'pre_'+str(d['directive_id'])

    if d['instruction']=='assume':
        ripl.assume( d['symbol'], build_exp(d['expression']), label=d['label'] )
    elif d['instruction']=='observe':
        ripl.observe( build_exp(d['expression']), d['value'], label=d['label'] )
    elif d['instruction']=='predict':
        ripl.predict( build_exp(d['expression']), label=d['label'] )
    
    
def build_exp(exp):
    'Take expression from directive_list and build the Lisp string'
    if type(exp)==str:
        return exp
    elif type(exp)==dict:
        return str(exp['value'])
    else:
        return '('+str(exp[0])+' ' + ' '.join(map(build_exp,exp[1:])) + ')'


class MultiRipls():

    def __init__(self):
        self.npseed = 1
        print 'MultiRipls Instance created'
        print 'Numpy seed: %i' % self.npseed

    def loadModel(self,directives_list,ripls_range,no_transitions,no_probes,infer_msg,plot_msg):
    
        self.no_ripls = ripls_range[1] - ripls_range[0]
        self.ripls = [make_church_prime_ripl() for i in range(self.no_ripls)]

        # could vary the directives, but for now they are same for all
        self.di_lists = [directives_list] * self.no_ripls

        # This seed should stay fixed in order to reproduce results
        seeds = np.random.randint(10**9,size = ripls_range[1])[ripls_range[0]:]
        self.seeds = map(int,seeds)
        assert len(self.seeds)==self.no_ripls

        self.worker_names = ['w'+str(i)+'_'+str(seeds[i]) for i in range(len(seeds) ) ]

        self.inq = multiprocessing.Queue()
        self.outq = multiprocessing.Queue()
        self.procs = []
        self.job_count = 0

        for i in range(self.no_ripls):
            myargs = (self.ripls[i], self.di_lists[i], self.seeds[i],
                      self.worker_names[i],
                      self.inq, self.outq,
                      infer_msg, plot_msg)
            p = multiprocessing.Process(group=None, target=p_worker,
                                        args=myargs, name=self.worker_names[i])
            self.procs.append(p)

            p.start()

    def addWorker(self,seed,name):

        self.ripls.append(make_church_prime_ripl())
        self.no_ripls += 1
        np.random.seed(npseed)
        self.seeds.append(np.random.randint(10**9,size=self.no_ripls)[-1]) 
        self.di_lists.append(self.di_lists[0])
        self.worker_names += 'w'+str(self.no_ripls)+'_'+str(self.seeds[-1])
        
        myargs =  (self.ripls[-1], self.di_lists[-1], self.seeds[-1],
                      self.worker_names[-1],
                      self.inq, self.outq,
                      infer_msg, plot_msg)
        multiprocessing.Process(group=None, target=p_worker, args=myargs)
                                


    def runJobs(self,no_transitions,no_probes,infer_msg,plot_msg):
   
        self.job_count += self.no_ripls

        for i in range(self.no_ripls):
            self.inq.put( (no_transitions, no_probes) )

        while int(outq.qsize()) != job_count:
            print "parent sleeps, waiting for outq to match job_count \n"
            time.sleep(1)

        print "All jobs complete, available at MyMulti.outq.get() \n"


    def terminateAll(self):
        print 'terminating procs'
        for p in procs: p.terminate()




def p_worker(ripl,di_list,seed,name,inq,outq,infer_msg,plot_msg):
    '''1. Set ripl seed with seed from parent process.
       2. Run labeled directives from di_list on ripl.
       3. Create output dict of parameters and snapshot for each labeled directive
          as well as logscore.
       4. Run ripl.infer() till no_transitions, storing values in snapshots
          at probe points (determine by no_probes).
       5. Could add to Queue either at every probe point, or (current setup)
          after finishing no_transitions.
       6. Infer_msg could be used to select specific variables to record or
          to use non-default infer instructions.'''
    print 'Process: ',multiprocessing.current_process().name    
    
    ripl.set_seed(seed)

    # run directives from di_list on ripl, add labels, pull new di_list
    for di in di_list:
        run_py_directive(ripl,di)
    di_list = ripl.list_directives()

    
    # create output dict
    out = {'worker_name':name,'total_transitions':0,
                'all_probes':[], 'infer_msg':infer_msg,
                'plot_msg':plot_msg, 'seed':seed,
                'labelToSnapshot':{} }

    for di in di_list:
        label = di['label']
        out['labelToSnapshot'][label] = { 'series': [] }
                                        
    # FIXME, ensure no conflict in labels with logscore    
    out['labelToSnapshot']['logscore']={'series': [] }
   
                                        

    while True:
        no_transitions,no_probes = inq.get()
        
        step = int(round (float(no_transitions+1) / no_probes) )
        start = out['total_transitions']
        probes = range(start,start+no_transitions+1,step)

        for probe in probes:
            ripl.infer(step)   # add infer instructions based on infer_msg

            # update each series (could filter out observes)
            for di in di_list:
                label = di['label']
                value = ripl.report(label)
                series = out['labelToSnapshot'][label]['series']
                series.append( value )
                if np.random.beta(1,1) < .3:
                    print '''name: %s, total_trans: %i, all_probes: %s,
                             cur_probe: %i, cur_di_label: %s, value: %f,
                             series: %s''' % (name,out['total_transitions'],
                                          str(out['all_probes']),
                                          probe,
                                              label,round(value,3),
                                              str(series[-1]) )

            # same for logscore
            value = ripl.get_global_logscore()
            out['labelToSnapshot']['logscore']['series'].append(value)


        # update stats
        out['total_transitions'] += no_transitions
        out['all_probes'] += probes

        outq.put(out)
        

        
    print 'Exiting (process %s)' % name
    return di_list,out



## Testing
v=make_church_prime_ripl()
v.clear(); v.assume("mu","(normal 0 20)");
v.assume('x','(normal mu .1)')
v.observe('x','8.')
v.observe("(normal mu 1)","5.")
v.observe("(normal mu 1)","5.")
v.predict("(normal mu .1)" )

# %v [clear]
# %v [assume mu (normal 0 20)]
# %v [observe (normal mu 1) 5 ]
# %v [observe (normal mu 0.6) 5 ]
# %v [predict (normal mu (+ .1 .2)) ]

di_list = v.list_directives()
ripls_range = (0,4); no_transitions = 30
no_probes = 3;     infer_msg = ''; plot_msg = '';


outq = multi_ripls('',di_list,ripls_range,no_transitions,no_probes,'','')
while not outq.empty():
    if int(outq.qsize())==1:
        out1 = outq.get()
    else:
        outq.get()
          
#assert len(out1['all_probes']) == no_probes + 1
assert out1['total_transitions'] == 2*no_transitions
assert out1['labelToSnapshot']['obs_2']['series'].index(5.)==0
logscore_series = out1['labelToSnapshot']['logscore']['series']
assert logscore_series[-1] > logscore_series[0]


# out2 = multi_ripls(di_list,(0,10),no_transitions,no_probes,'','')
# assert( len( np.unique( [len(item) for item in out] ) ) == 1)
# assert (out[2]==out2[2][:5])

# ripls,di_lists,seeds,worker_names = out
# wout = p_worker(ripls[0],di_lists[0],
#                 seeds[0],worker_names[0], multiprocessing.Queue(),
#                 no_transitions,no_probes,infer_msg,plot_msg )

# for di in wout[0]:
#     print build_exp(di['expression'])
# out_dict = wout[1]
# lab = out_dict['labelToSnapshot']
# print lab['mu']['series']


## test
# %v [clear]
# %v [assume xy ( beta 1   1)]
# %v [observe (normal xy 1) 5 ]
# %v [predict (beta xy (+ 1 5) ) ]
# v = ipy_ripl
# v2 = make_church_prime_ripl()
# for d in v.list_directives():
#     build_py_directive(v2,d)

# for i,d in enumerate(v.list_directives()):
#     print d
#     print v2.list_directives()[i]
#     print '-------------\n'



