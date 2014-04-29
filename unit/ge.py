from venture.venturemagics.ip_parallel import *
from venture.venturemagics.reg_demo_utils import simple_quadratic_model
from scipy.stats import probplot
from analytics import directive_split
import time

############### TODO LISTE
# 0. Warn about problem with closures/namespace for inference programs

# 2. Install new version of IPython.

# 3. Do posterior checking. After inference, go through observes can 
#    generate N data points from posterior for each expression. Plot
#    the data points along with the observed value (maybe do t test also).


notes on tests:
ks test assumes iid. with geweke we dont get iid samples. 
so we would really need to correct for that. this is a 
reason to focus on qq_plot. we'll also neeeds lots of samples
to get a rejection of same distribution via the ks test. (get 
some flavor for this. maybe output that info). what should a 
bayesian do here?

if our worry is an error in the inference algorithm itself,
then we want to test whether the two dists are exactly the 
same and have a rigorous test with lots of samples. geweke
is then just a test for how much autocorrelation we'll have.
(since the autocorrelation is related to convergence in general
this could be very useful to have info about. but note that
this is not what KS is meant to test. with geweke, we know
the dists are the same (assuming no bugs) and what we really
want to measure is degree of independence in the geweke samples
so we want something that measures autocorrelation. why not 
just measure that? (idea that given one set of dep samples 
and one set of indie, there should be way of measuring of many
of the dep samples you need per indie, i.e. the effective 
sample size. we can measure the estimated kl between them
over time, but that doesn't seem likely to be the best test.)

with the testfromprior, we want to compare the distance of the 
distributions, but we need to take into account our limited 
number of samples from each. so KS is ok here. though again,
maybe once we get enough samples, it's clear dists are different
then question is how close are they. we might do KL here also,
as a useful estimator if we have loads of samples. (what is 
our estimator of KL like if variances are huge? we can just
try it with data from cauchy. massive numbers of samples.)

note that we could do a serious bayesian thing, with a dists
same CRP density estimate and a dists different model, with
bayes optimzation. we integrate out for predictions for each
one. not clear how useful this info is, vs. just getting QQ plot. 

other idea: you could take as input a function on the values of the 
observes. this could compute a summary statistic for the observes.
which we can then compute a posterior dist over. you can specify
a plot for the ripl and we can generate lots of sample plots. 
(nice general way of dealing. then user can customize.)

def qq_plot(s1,s2,label1,label2):
    fig,ax = plt.subplots(figsize=(3,3))
    ax.scatter(sorted(s1),sorted(s2),s=4,lw=0)
    ax.set_xlabel(label1); ax.set_ylabel(label2)
    ax.set_title('PP Plot')

def qq_plot_all(dict1,dict2,label1,label2):
    ## http://people.reed.edu/~jones/Courses/P14.pdf
    # FIXME do interpolation where samples mismatched
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

def forgetAllObserves(ripl):
    for di in ripl.list_directives():
        if di['instruction']=='observe': ripl.forget(di['directive_id'])

def progInfer(ripl,step,infer=None):
    if infer is None:
        ripl.infer(step)
    elif isinstance(infer, str):
        ripl.infer(infer)
    else:
        infer(ripl, step)
    
def defaultSweep(assumes): return int(1+(1.5*len(assumes)))
    
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
           stepSize=None,queryAssumes=True,observeToPredict=False,infer=None):
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
        if sample<totalSamples-1:
            progInfer(ripl,stepSize,infer=infer)
        forgetAllObserves(ripl)

    return nameToSeries



def runFromConditional(ripl,assumes,observes,totalSamples,queryExps=None,
                       stepSize=None, queryAssumes=True,infer=None):
    ## EDIT GLOBAL FUNCS
    def defaultSweep(assumes): return int(1+(1.5*len(assumes)))
    
    def makeNameToSeries(assumes,queryExps,queryAssumes):
        if queryExps is None:
            queryExps = []
        nameToSeries = {exp:[] for exp in queryExps}
        if queryAssumes: nameToSeries.update( {exp:[] for exp,_ in assumes} )
        return nameToSeries

    def progInfer(ripl,step,infer=None):
        if infer is None:
            ripl.infer(step)
        elif isinstance(infer, str):
            ripl.infer(infer)
        else:
            infer(ripl, step)

    assert all([len(obs)==2 for obs in observes])
    
    if stepSize is None:
        stepSize = defaultSweep(assumes) # rough version of sweep

    [ripl.assume(sym,exp) for sym,exp in assumes]
    [ripl.observe(exp,literal) for exp,literal in observes]
    
    nameToSeries = makeNameToSeries(assumes,queryExps,queryAssumes)
    
    for sample in range(totalSamples):
        [series.append(ripl.sample(exp)) for exp,series in nameToSeries.items()]
        if sample<totalSamples-1:
            progInfer(ripl,stepSize,infer=infer)

    return nameToSeries#,ripl FIXME can't output due to pickling


def mrRFC(mripl,assumes,observes,totalSamples,queryExps=None,
          stepSize=None,queryAssumes=True,infer=None):
    'RunFromConditional mapped across clear mripl with same observes.'
    argsRFC = (assumes,observes,totalSamples)
    kwargsRFC = {'queryExps':queryExps, 'stepSize':stepSize,
                 'queryAssumes':True, 'infer':infer}
    list_nameToSeries = mr_map_proc(mripl,'all',runFromConditional,*argsRFC,**kwargsRFC)
    historyForm = makeHistoryForm(list_nameToSeries)
    return historyForm,mripl


def multiConditionFromPrior(mripl,noDatasets,assumes,observes,totalSamples,
                            queryExps=None, stepSize=None, queryAssumes=True,
                            infer=None):
    assert mripl.no_ripls == noDatasets
    seeds = [1]*noDatasets
    mripl.mr_set_seeds(seeds=seeds)

    datasets=[]
    for i in range(noDatasets):
        v=mk_p_ripl()
        v.set_seed(i)
        [v.assume(sym,exp) for sym,exp in assumes]
        datasets.append( [(exp,v.sample(exp)) for exp,_ in observes] )
        
    args=[(assumes,obs,totalSamples) for obs in datasets]
    kwargs=[{'queryExps':queryExps,'stepSize':stepSize,
             'queryAssumes':True,'infer':infer}] * noDatasets
    argsList = zip(args,kwargs)


    list_nameToSeries = mr_map_array(mripl,runFromConditional,argsList,no_kwargs=False)
    historyForm = makeHistoryForm(list_nameToSeries)

    ## FIXME: add the comparison the samples from the prior (see test)
    # : forward sample with same mripl,assumes, etc.
    # the do compare dist and QQ plot and plot over time

    return historyForm,mripl



def mrForwardSample(mripl,assumes,observes,totalSamples,queryExps=None,
                    queryAssumes=True):
    totalSamples==mripl.no_ripls
    nameToSeries = makeNameToSeries(assumes,queryExps,queryAssumes)
    [mripl.assume(sym,exp) for sym,exp in assumes]
    return {name:mripl.sample(name) for name in nameToSeries}


def compareSampleDicts(dicts,labels,plot=False):
    stats = (np.mean,np.median,np.std,len)
    stats_dict = {}
    for exp,_ in dicts[0].iteritems():
        stats_dict[exp] = []
        for dict_i,label_i in zip(dicts,labels):
            samples=dict_i[exp]
            s_stats = tuple([s(samples) for s in stats])
            stats_dict[exp].append(s_stats)
            print 'Dict: %s. Exp: %s'%(label_i,exp)
            print 'Mean, median, std, N = %.3f  %.3f  %.3f  %i'%s_stats
    fig = qq_plot_all(dicts[0],dicts[1],labels[0],labels[1]) if plot else None
    return stats_dict,fig


def testMultiConditionFromPrior(totalSamples=100, noDatasets=10,
                                stepSize=40,plot=False,local_mode=False):
    assumes=[('mu','(normal 0 1)')]
    observes=[('(normal mu .05)','1.4')]*4
    mripl = MRipl(noDatasets,local_mode=local_mode)
    args = (mripl,noDatasets,assumes,observes,totalSamples)
    historyForm,mripl = multiConditionFromPrior(*args,stepSize=stepSize)

    # seeds all the same
    assert all([seed==mripl.seeds[0] for seed in mripl.seeds])

    # mu, with prior N(0,1), close to 0 after inference on one datapoint
    assert all([5 > abs(mu) for mu in mripl.sample('mu')])

    # find observes (can't use display_directives yet)
    di_list=mr_map_proc(mripl,'all',lambda r:r.list_directives())[0]
    di_obs = [di for di in di_list if di['instruction']=='observe']
    assert len(di_obs)==len(observes)
    
    # ripls start in same place for each exp
    snapshots = makeSnapshots(historyForm)
    last = {}
    for exp,snapshots in snapshots.iteritems():
        assert .01 > np.var(snapshots[0])
        last[exp] = snapshots[-1]

    # compare final snapshot on mu to N(0,1)
    normal_dict={'mu':np.random.normal(0,1,noDatasets)}
    dicts = [last,normal_dict]
    labels = ['Last_snapshot_multiCFP','True_N(0,1)']
    stats_dict,_ = compareSampleDicts(dicts,labels,plot=plot)
    if noDatasets>20 and totalSamples>10:
        stats_last,stats_N01 = np.array(stats_dict['mu'])
        assert all(np.abs(stats_last - stats_N01) < .6 )
    

    # show divergence of chains
    if plot:
        no_exps = len(historyForm.keys())
        fig,ax = plt.subplots(max(no_exps,2),1)
        for exp_ind,(exp,all_datasets) in enumerate(historyForm.items()):
            for dataset_ind,dataset in enumerate(all_datasets):
                ax[exp_ind].plot( dataset, label='Dataset %i' % dataset_ind)
            ax[exp_ind].set_title('Exp: %s, [multiConditionFromPrior]' % exp )
            ax[exp_ind].legend()
        fig.tight_layout()
    
    return historyForm,fig if plot else historyForm


def testMrForwardSample(totalSamples=100,plot=False):
    start = time.time()
    assumes=[('mu','(normal 0 1)')]
    observes = []
    forward_dict = mrForwardSample(MRipl(totalSamples),assumes,observes,totalSamples)
    print 'time after forwardsample:', time.time() - start
    
    RFC_dict = runFromConditional(mk_p_ripl(),assumes,observes,totalSamples)
    dicts =[forward_dict,RFC_dict]
    labels= ['forwardSample','RFC']
    stats_dict,fig = compareSampleDicts(dicts,labels,plot=plot)
    if totalSamples>50:
        stats_forward,stats_RFC = np.array(stats_dict['mu'])
        assert all(np.abs(stats_forward - stats_RFC) < .6 )
    return 


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


def testRFC(totalSamples,poisson=False):
    ripl=mk_p_ripl() 
    assumes=[('mu','(normal 0 1)')]
    observes=[('(normal mu .1)','.5')]*8
    queryExps=['(normal mu .2)']
    if poisson:
        assumes=[('mu','(poisson 30)')]
        observes=[('(normal mu 1)','20')]*8
    nameToSeries = runFromConditional(ripl,assumes,observes,
                                        totalSamples,queryExps=queryExps,
                                        stepSize=5)
    mean=np.mean(nameToSeries['mu'])
    std=abs(np.std(nameToSeries['mu']))
    if poisson:
        assert 2>abs(mean - 30)
    else:
        assert .2 > abs(mean - .5)
    print 'testRFC: poisson=%s'%poisson
    print 'assumes=%s, observes=%s'%(str(assumes),str(observes))
    print 'infer mu (mean,var): ',mean,std

    # test with bigger model
    ripl.clear()
    ripl.execute_program(simple_quadratic_model)
    [ripl.observe('(y_x %i)'%i, str(i)) for i in range(-5,5) ]
    assumes,observes = riplToAssumesObserves(ripl)
    queryExps=['(f 0)','(f 1)']
    nameToSeries = runFromConditional(mk_p_ripl(),assumes,observes,
                                      totalSamples,queryExps=queryExps,stepSize=None)
    print '\n \n simple_quadratic: observes: (i,i) for range(-5,5)'
    print 'means,std w0,w1,w2,(f 0):'
    means = map(np.mean,[nameToSeries[name] for name in ['w0','w1','w2','(f 0)'] ])
    stds = map(np.std,[nameToSeries[name] for name in ['w0','w1','w2','(f 0)'] ])
    assert .5>abs(means[0]) and .4>abs(means[1]-1) and .1>abs(means[2])
    print np.round(means,2),'\n',np.round(stds,2)
    
    #test mripl RFC
    for local_mode in [True,False]:
        start=time.time()
        history,mripl = mrRFC(MRipl(4,local_mode=local_mode),
                              assumes,observes,totalSamples,
                              queryExps=queryExps,stepSize=3)
        snapshots = makeSnapshots(history)
        print '\n \n MRipl RFC: simple_quadratic, local_model=%s'%str(local_mode)
        print 'last snapshot means,std:'
        means=map(np.mean,[snapshots[name][-1] for name in ['w0','w1','w2','(f 0)'] ])
        stds=map(np.std,[snapshots[name][-1] for name in ['w0','w1','w2','(f 0)'] ])
        assert .5>abs(means[0]) and .4>abs(means[1]-1) and .1>abs(means[2])
        print np.round(means,2),'\n',np.round(stds,2)
        print 'time: ',time.time() - start

    return nameToSeries,history























quad = 0

# Simple quadratic model test
#if __name__ == "__main__":
if quad: 
    no_ripls=2; noDatasets=20
    size_data=5
    model=simple_quadratic_model #x_model_t+quad_fourier_model
    gen_data_exp = '(y_x (x_d))'
    observes = [gen_data_exp] * size_data
    queryExps = ['w0','w1','w2','noise']
    no_transitions=80
    mrkwargs={'backend':'puma', 'local_mode':True, 'no_local_ripls':no_ripls}

    # from inference
    store_exp_vals_inf = mr_condition_prior(no_ripls,noDatasets,model,
                                            observes,queryExps,no_transitions,
                                            **mrkwargs)
    # from prior
    no_samples = no_ripls * noDatasets
    mrkwargs={'backend':'puma', 'local_mode':True, 'no_local_ripls':no_samples}
    store_exp_vals_prior = mr_forward_sample(no_samples,model,queryExps,**mrkwargs)

    # from geweke
    total_samples = no_ripls * noDatasets
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

def condition_prior(r,model,noDatasets,observes,no_transitions,queryExps):
    
    exp_vals = {exp:[] for exp in queryExps}
    r.execute_program(model)
    
    for dataset in range(noDatasets):
        data=[r.predict(obs,label='test'+str(i)) for i,obs in enumerate(observes)]
        [r.forget('test'+str(i)) for i,obs in enumerate(observes)]
        [r.observe(obs,data[i],label='test'+str(i)) for i,obs in enumerate(observes)]
    
        r.infer(no_transitions)
        [lst.append(r.sample(exp)) for exp,lst in exp_vals.items()]
        [r.forget('test'+str(i)) for i,obs in enumerate(observes)]
        
    return exp_vals


def mr_condition_prior(no_ripls, noDatasets, model,
                       observes, queryExps, no_transitions,**mrkwargs):
    v=MRipl(no_ripls,**mrkwargs)
    all_out = mr_map_proc(v,no_ripls,condition_prior,
                          model,noDatasets,observes,
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
            forgetAllObserves(r)
    
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


def condition_prior_sample(r,model,noDatasets,list_observes,no_transitions,query_exps):

    exp_vals = {exp:[] for exp in query_exps}

    for dataset in range(noDatasets):
        r.clear()
        r.execute_program(model)
        
        [r.observe(exp,r.sample(exp)) for exp in observes]
        r.infer(no_transitions)
        [lst.append(r.sample(exp)) for exp,lst in exp_vals.items()]
        r.forget(all_observes(r))

    return exp_vals


#mr_map_proc(mripl,condition_prior_sample,model,noDatasets,observes,no_transitions,query_exps)


def condition_prior_fail(r,model,noDatasets,observes,no_transitions,
                         query_exps):
    'Fails because clearing resets the seeds. Every ripl in MR gives same values'
    exp_vals = {exp:[] for exp in query_exps}
    for dataset in range(noDatasets):
        r.clear()
        r.execute_program(model)
        obs_did_start = len(r.list_directives())+1 #dids start at 1
        data=[r.predict(obs) for i,obs in enumerate(observes)]
        [r.forget(obs_did_start+i) for i,obs in enumerate(observes)]
        [r.observe(obs,data[i]) for i,obs in enumerate(observes)]
        
        r.infer(no_transitions)
        [lst.append(r.sample(exp)) for exp,lst in exp_vals.items()]
        
    return exp_vals



def con_prior(r,noDatasets,gen_data_exp,size_data,no_transitions,vars):
    for i in noDatasets:
        data=[]
        for data_point in range(size_data):
            data.append(r.predict(gen_data_exp,label=str(datapoint)))

        for data_point in data:
            r.observe(gen_data_exp,label=str(datapoint))
            








# # simple version
# for i in noDatasets:
#     datasets = [r.predict(gen_data) for ripl in no_ripls for i in size_data]

#     mr_map_array(v,lambda r,exp,val:r.observe(exp,val),dataset[
