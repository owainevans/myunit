from venture.venturemagics.ip_parallel import *
from venture.venturemagics.reg_demo_utils import simple_quadratic_model
from scipy.stats import probplot
from analytics import directive_split

def qq_plot(s1,s2,label1,label2):
    fig,ax = plt.subplots(figsize=(3,3))
    ax.scatter(sorted(s1),sorted(s2),s=4,lw=0)
    ax.set_xlabel(label1); ax.set_ylabel(label2)
    ax.set_title('PP Plot')

def qq_plot_all(dict1,dict2,label1,label2):
    ## http://people.reed.edu/~jones/Courses/P14.pdf
    # generally: need to do interpolation where samples mismatched
    assert len(dict1)==len(dict2)
    no_exps = len(dict1)
    subplot_rows = max(no_exps,2)
    fig,ax = plt.subplots(subplot_rows,2,figsize=(8.5,3*no_exps))
    
    for i,exp in enumerate(dict1.keys()):
        s1,s2 = (dict1[exp],dict2[exp])
        assert len(s1)==len(s2)

        ax[i,0].hist(s1,bins=20,alpha=0.7,color='b',label=label1)
        ax[i,0].hist(s2,bins=20,alpha=0.4,color='y',label=label2)
        ax[i,0].legend()
        ax[i,0].set_title('Hists: %s'%exp)

        ax[i,1].scatter(sorted(s1),sorted(s2),s=4,lw=0)
        ax[i,1].set_xlabel(label1)
        ax[i,1].set_ylabel(label2)
        ax[i,1].set_title('QQ Plot %s'%exp)

        xr = np.linspace(min(s1),max(s1),30)
        ax[i,1].plot(xr,xr)

    fig.tight_layout()
    return fig

def riplToAssumesObserves(ripl):
     assumes=[directive_split(di) for di in ripl.list_directives() if di['instruction']=='assume']
     observes=[directive_split(di) for di in ripl.list_directives() if di['instruction']=='observe']
     return assumes,observes

def forget_all_observes(ripl):
    for di in ripl.list_directives():
        if di['instruction']=='observe': ripl.forget(di['directive_id'])
    
def defaultSweep(assumes,infer_prog=None):
    return int(1+(1.5*len(assumes)))
    
def makeNameToSeries(assumes,queryExps,queryAssumes):
    if queryExps is None:
        queryExps = []
    nameToSeries = {exp:[] for exp in queryExps}
    if queryAssumes: nameToSeries.update( {exp:[] for exp,_ in assumes} )
    return nameToSeries

def makeHistoryForm(list_nameToSeries):
    historyForm = {}
    for name in list_nameToSeries[0].keys():
        historyForm[name] = [nTSeries[name] for nTSeries in list_nameToSeries]
    return historyForm

def makeSnapshots(historyForm):
    ':: {name0:[name0_series0,name0_series1,...,], name1: ,...}'
    s={name:[] for name in historyForm.keys()}
    
    for name,list_series in historyForm.items():
        len_series = len(list_series[0])
        for t in range(len_series):
            s[name].append([series[t] for series in list_series])
    return s
     
    
def geweke(ripl,assumes,observes,totalSamples,queryExps=None,
           stepSize=None,queryAssumes=True,observeToPredict=False):
    '''Geweke 2004 test. Sample values for *observes* from current
    ripl state, observe them, infer(*stepSize*), then forget observes.
    Repeat for totalSamples. Determine ripl seed and backend via *ripl*.'''
    
    if stepSize is None:
        stepSize = defaultSweep(assumes) # rough version of sweep

    [ripl.assume(sym,exp) for sym,exp in assumes]
    
    if len(observes[0])==2: # not observes=[exp1,...,]
        observes = [exp for exp,literal in observes]
    
    nameToSeries = makeNameToSeries(assumes,queryExps,queryAssumes)
    
    for sample in range(totalSamples):
        [series.append(ripl.sample(exp)) for exp,series in nameToSeries.items()]
        if not observeToPredict:
            [ripl.observe(exp, ripl.sample(exp)) for exp in observes]
        else:
            [ripl.predict(exp) for exp in observes]
        if sample<totalSamples-1:  ripl.infer(stepSize)
        forget_all_observes(ripl)

    return nameToSeries


def runFromConditional(ripl,assumes,observes,totalSamples,queryExps=None,
                       stepSize=None, queryAssumes=True):
    ## EDIT GLOBAL FUNCS
    def defaultSweep(assumes,infer_prog=None):
        return int(1+(1.5*len(assumes)))
    
    def makeNameToSeries(assumes,queryExps,queryAssumes):
        if queryExps is None:
            queryExps = []
        nameToSeries = {exp:[] for exp in queryExps}
        if queryAssumes: nameToSeries.update( {exp:[] for exp,_ in assumes} )
        return nameToSeries

    assert all([len(obs)==2 for obs in observes])
    
    if stepSize is None:
        stepSize = defaultSweep(assumes) # rough version of sweep

    [ripl.assume(sym,exp) for sym,exp in assumes]
    [ripl.observe(exp,literal) for exp,literal in observes]
    
    nameToSeries = makeNameToSeries(assumes,queryExps,queryAssumes)
    
    for sample in range(totalSamples):
        [series.append(ripl.sample(exp)) for exp,series in nameToSeries.items()]
        if sample<totalSamples-1:  ripl.infer(stepSize)

    return nameToSeries#,ripl FIXME can't output due to pickling


def mrRFC(mripl,assumes,observes,totalSamples,queryExps=None,
          stepSize=None,queryAssumes=True):
    'RunFromConditional mapped across mripl with same observes.'
    argsRFC = (assumes,observes,totalSamples)
    kwargsRFC = {'queryExps':queryExps , 'stepSize':stepSize, 'queryAssumes':True}
    list_nameToSeries = mr_map_proc(mripl,'all',runFromConditional,*argsRFC,**kwargsRFC)
    historyForm = makeHistoryForm(list_nameToSeries)
    return historyForm,mripl


def testFromPrior(mripl,no_datasets,assumes,observes,totalSamples,queryExps=None,
          stepSize=None,queryAssumes=True):
    seeds = [1]*no_datasets
    mripl.set_remote_seed(seeds)
    v=mripl.local_ripls[0]
    datasets=[]
    for i in range(no_datasets):
        datasets.append( [(exp,v.sample(exp)) for exp,_ in observes] )
        
    args=[(assumes,obs,totalSamples) for obs in datasets]
    kwargs=[{'queryExps':queryExps , 'stepSize':StepSize, 'queryAssumes':True}]*no_datasets
    argsList = zip(args,kwargs)

    list_nameToSeries = mr_map_array(mripl,runFromConditional,argsList,no_kwargs=False)
    historyForm = makeHistoryForm(list_nameToSeries)

    return historyForm,mripl


def mrForwardSample(mripl,assumes,observes,totalSamples,queryExps=None,
                    queryAssumes=True):
    
    mr = MRipl(totalSamples,backend=mripl.backend,local_mode=mripl.local_mode)
    nameToSeries = makeNameToSeries(assumes,queryExps,queryAssumes)
    [mr.assume(sym,exp) for sym,exp in assumes]
    return {name:mr.sample(name) for name in nameToSeries}


def compareSampleDicts(dicts,labels):
    for exp,_ in dicts[0].iteritems():
        for i,d in enumerate(dicts):
            stats = [np.mean(d[exp]), np.median(d[exp]), np.std(d[exp]) ]
            print 'Label %s, Exp %s, mean, median, std = %.3f %.3f %.3f'% [labels[i],
                                                                           exp] + stats
    fig = qq_plot_all(dicts[0],dicts[1],labels[0],labels[1])
    return fig

def testMrForwardSample(totalSamples=100):
    assumes=['mu','(normal 0 1)']
    observes = []
    forward_dict = mrForwardSample(MRipl(2),assumes,observes,
                                    totalSamples)
    RFC_dict = runFromConditional(mk_p_ripl,assumes,obseves,totalSamples)
    fig = compareSampleDicts([forward_dict,RFC_dict],['forwardSample','RFC'])
    return fig

def compare_stats(list_dicts,list_labels=None):
    if not list_labels:
        list_labels = ['Dict %i'%i for i in range(len(list_dicts))]
    print 'Stat, exp, ', ' '.join(list_labels)

    for exp in list_dicts[0].keys():
        lists = [d[exp] for d in list_dicts]
        print '%s means:'%exp, np.mean(lists,axis=1)
        print '%s stds:'%exp, np.std(lists,axis=1)
        print '----'


def testForwardSampleMRRFC(totalSamples=100):
    v=MRipl(10)
    
    


def testGeweke(totalSamples = 400, plot=False,
               observeToPredict=False, poisson=False):
    ripl=mk_p_ripl()
    assumes=[('mu','(normal 0 1)')]
    observes=[('(normal mu .1)','.5')]
    if poisson:
        assumes=[('mu','(poisson 30)')]
        observes=[('(normal mu 1)','6')]
    nameToSeries = geweke(ripl,assumes,observes,totalSamples,stepSize=5,
                       observeToPredict=observeToPredict)
    mean=np.mean(nameToSeries['mu'])
    std=abs(np.std(nameToSeries['mu']))
    print 'testGeweke: observeToPredict=%s'%observeToPredict
    print 'mu=(normal 0 1), infer mu: (mean,std)=', mean, std
    #assert .8 > abs(mean) and .8 > abs(std-1)
    if plot:
        dict2={'mu':np.random.normal(0,1,totalSamples)}
        label2='np.random.normal(0,1)'
        if poisson:
            dict2={'mu':np.random.poisson(30,totalSamples)}
            label2='np.random.poisson(30)'
        qq_plot_all(nameToSeries,dict2,'Geweke',label2)
        plt.figure()
        probplot(nameToSeries['mu'],dist='norm',plot=plt)
        plt.show()
    return nameToSeries


def testRFC(totalSamples,poisson=False,mr=False):
    ripl=mk_p_ripl() 
    assumes=[('mu','(normal 0 1)')]
    observes=[('(normal mu .1)','.5')]*5
    queryExps=['(normal mu .2)']
    if poisson:
        assumes=[('mu','(poisson 30)')]
        observes=[('(normal mu 1)','20')]*5
    nameToSeries = runFromConditional(ripl,assumes,observes,
                                        totalSamples,queryExps=queryExps,
                                        stepSize=5)
    mean=np.mean(nameToSeries['mu'])
    std=abs(np.std(nameToSeries['mu']))
    print 'testRFC: poisson=%s'%poisson
    print 'assumes=%s, observes=%s'%(str(assumes),str(observes))
    print 'infer mu (mean,var): ',mean,std

    # test with bigger model
    ripl.clear()
    ripl.execute_program(simple_quadratic_model)
    [ripl.observe('(y_x %i)'%i, str(i)) for i in range(-4,4) ]
    assumes,observes = riplToAssumesObserves(ripl)
    queryExps=['(f 0)','(f 1)']
    nameToSeries = runFromConditional(mk_p_ripl(),assumes,observes,
                                      totalSamples,queryExps=queryExps,stepSize=None)
    print '\n \n simple_quadratic: observes:',observes
    print 'means,std w0,w1,w2,(f 0):'
    print map(np.mean,[nameToSeries[name] for name in ['w0','w1','w2','(f 0)'] ])
    print map(np.std,[nameToSeries[name] for name in ['w0','w1','w2','(f 0)'] ])

    #test mripl RFC
    history,mripl = mrRFC(MRipl(4),assumes,observes,totalSamples,
                              queryExps=queryExps,stepSize=3)
    snapshots = makeSnapshots(history)
    print '\n \n MRipl RFC: simple_quadratic'
    print 'last snapshot means,std:'
    print map(np.mean,[snapshots[name][-1] for name in ['w0','w1','w2','(f 0)'] ])
    print map(np.std,[snapshots[name][-1] for name in ['w0','w1','w2','(f 0)'] ])

    return nameToSeries,history























quad = 0

# Simple quadratic model test
#if __name__ == "__main__":
if quad: 
    no_ripls=2; no_datasets=20
    size_data=5
    model=simple_quadratic_model #x_model_t+quad_fourier_model
    gen_data_exp = '(y_x (x_d))'
    observes = [gen_data_exp] * size_data
    queryExps = ['w0','w1','w2','noise']
    no_transitions=80
    mrkwargs={'backend':'puma', 'local_mode':True, 'no_local_ripls':no_ripls}

    # from inference
    store_exp_vals_inf = mr_condition_prior(no_ripls,no_datasets,model,
                                            observes,queryExps,no_transitions,
                                            **mrkwargs)
    # from prior
    no_samples = no_ripls * no_datasets
    mrkwargs={'backend':'puma', 'local_mode':True, 'no_local_ripls':no_samples}
    store_exp_vals_prior = mr_forward_sample(no_samples,model,queryExps,**mrkwargs)

    # from geweke
    total_samples = no_ripls * no_datasets
    step_size=1
    store_exp_vals_geweke = geweke(mk_p_ripl(),model,observes,
                                   step_size,total_samples,queryExps)

    assert len(store_exp_vals_inf['w0'])==len(store_exp_vals_prior['w0'])
    assert len(store_exp_vals_geweke['w0'])==len(store_exp_vals_prior['w0'])


    # analytic for quad model
    ana_store_exp_vals={'w0':np.random.normal(0,3,no_samples),
                        'w1':np.random.normal(0,1,no_samples),
                        'w2':np.random.normal(0,.3,no_samples),
                        'noise':np.random.gamma(2,1,no_samples)}

    compare_stats([ana_store_exp_vals,store_exp_vals_prior,
                   store_exp_vals_inf,store_exp_vals_geweke],
                  ['ana','prior','prior','ge'])

    plt.close('all')
    qq_plot_all(store_exp_vals_inf,store_exp_vals_prior,'inf','prior')
    qq_plot_all(ana_store_exp_vals,store_exp_vals_prior,'ana','prior')
    plt.show()














## OLD

def extract_directives(v_st):
    if isinstance(v_st,str):
        v_st = mk_p_ripl()
        v_st.execute_program(v_st)
    model = '\n'.join(display_directves(v,'assume'))
    observes = '\n'.join(display_directves(v,'observe'))
    return model,observes

def prepare_tests(directives_string):
    model,observes = extract_directives(directives_string)

def condition_prior(r,model,no_datasets,observes,no_transitions,queryExps):
    
    exp_vals = {exp:[] for exp in queryExps}
    r.execute_program(model)
    
    for dataset in range(no_datasets):
        data=[r.predict(obs,label='test'+str(i)) for i,obs in enumerate(observes)]
        [r.forget('test'+str(i)) for i,obs in enumerate(observes)]
        [r.observe(obs,data[i],label='test'+str(i)) for i,obs in enumerate(observes)]
    
        r.infer(no_transitions)
        [lst.append(r.sample(exp)) for exp,lst in exp_vals.items()]
        [r.forget('test'+str(i)) for i,obs in enumerate(observes)]
        
    return exp_vals


def mr_condition_prior(no_ripls, no_datasets, model,
                       observes, queryExps, no_transitions,**mrkwargs):
    v=MRipl(no_ripls,**mrkwargs)
    all_out = mr_map_proc(v,no_ripls,condition_prior,
                          model,no_datasets,observes,
                          no_transitions,queryExps)
 
    store_exp_vals={exp:[] for exp in queryExps}        
    [store_exp_vals[exp].extend(ripl_out[exp]) for exp in queryExps for ripl_out in all_out]
    return store_exp_vals


def geweke_predict(r,model,observes,step_size,total_samples,queryExps,use_sample=False):
    'observes=[exp,...]. we dont currently record all assumes'
    
    exp_vals = {exp:[] for exp in queryExps}
    r.execute_program(model)
    
    for sample in range(total_samples):
        data=[r.predict(obs,label='test'+str(i)) for i,obs in enumerate(observes)]
        [r.forget('test'+str(i)) for i,obs in enumerate(observes)]
        [r.observe(obs,data[i],label='test'+str(i)) for i,obs in enumerate(observes)]
        r.infer(step_size)
        [lst.append(r.sample(exp)) for exp,lst in exp_vals.items()]
        [r.forget('test'+str(i)) for i,obs in enumerate(observes)]

    if use_sample:
        for sample in range(total_samples):
            [r.observe(exp,r.sample(exp)) for exp in observes]
            r.infer(step_size)
            [lst.append(r.sample(exp)) for exp,lst in exp_vals.items()]
            forget_all_observes(r)
    
    return exp_vals



def geweke_sample(r,model,observes,step_size,total_samples,query_exps):
    exp_vals = {exp:[] for exp in query_exps}

    r.execute_program(model)
    obs_did_start = len(r.list_directives())+1 #did start at 1
    
    for sample in range(total_samples):
        data=[r.sample(obs) for i,obs in enumerate(observes)]
        [r.observe(obs,data[i]) for i,obs in enumerate(observes)]
        r.infer(step_size)
        [lst.append(r.sample(exp)) for exp,lst in exp_vals.items()]
        [r.forget(obs_did_start+i) for i,obs in enumerate(observes)]
    return exp_vals


def condition_prior_sample(r,model,no_datasets,list_observes,no_transitions,query_exps):

    exp_vals = {exp:[] for exp in query_exps}

    for dataset in range(no_datasets):
        r.clear()
        r.execute_program(model)
        
        [r.observe(exp,r.sample(exp)) for exp in observes]
        r.infer(no_transitions)
        [lst.append(r.sample(exp)) for exp,lst in exp_vals.items()]
        r.forget(all_observes(r))

    return exp_vals


#mr_map_proc(mripl,condition_prior_sample,model,no_datasets,observes,no_transitions,query_exps)


def condition_prior_fail(r,model,no_datasets,observes,no_transitions,
                         query_exps):
    'Fails because clearing resets the seeds. Every ripl in MR gives same values'
    exp_vals = {exp:[] for exp in query_exps}
    for dataset in range(no_datasets):
        r.clear()
        r.execute_program(model)
        obs_did_start = len(r.list_directives())+1 #dids start at 1
        data=[r.predict(obs) for i,obs in enumerate(observes)]
        [r.forget(obs_did_start+i) for i,obs in enumerate(observes)]
        [r.observe(obs,data[i]) for i,obs in enumerate(observes)]
        
        r.infer(no_transitions)
        [lst.append(r.sample(exp)) for exp,lst in exp_vals.items()]
        
    return exp_vals



def con_prior(r,no_datasets,gen_data_exp,size_data,no_transitions,vars):
    for i in no_datasets:
        data=[]
        for data_point in range(size_data):
            data.append(r.predict(gen_data_exp,label=str(datapoint)))

        for data_point in data:
            r.observe(gen_data_exp,label=str(datapoint))
            








# # simple version
# for i in no_datasets:
#     datasets = [r.predict(gen_data) for ripl in no_ripls for i in size_data]

#     mr_map_array(v,lambda r,exp,val:r.observe(exp,val),dataset[
