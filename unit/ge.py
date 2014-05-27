from venture.venturemagics.ip_parallel import *
from venture.venturemagics.reg_demo_utils import simple_quadratic_model
from scipy.stats import probplot
from venture.test.stats import reportSameContinuous
from analytics import directive_split
import time
from history import *

############### TODO LISTE
# push the new incarnation of unit (analytics) and see if it passes
# write tests to compare geweke and inference in unit



# have functions work with ripls they output
# do we want to have a mode for continuing inference in the same harness? maybe we should 
# leave this for now? not crucial. could save a ripl after inference and ask for continuation
# of past chain. with functions below, should not be too hard to continue. just don't clear
# at the start. 

# snapshot should take history, avoid groundtruth series and plot groundtruth with annotation

# show normal snapshot with resultant mripl


# Add text output summary for Geweke (maybe grabbing text from comp dicts)

# Get testFromPrior working. Need plot with all runs, with groundTruth
# on plot with faint, dotted line (edit in history).

# write test for comparing Geweke in analytics to this Geweke. Same
# for conditioned from prior.

# write testFromPrior for Unit using Force (maybe). 

# Auto measure of cross vs. between chain variation? e.g. plotting 
# histograms or QQ plots? Need to add interpolation in general. 


# 0. Warn about problem with closures/namespace for inference programs

# 3. Finish testFromPrior, generalize plots from current tests. 

# 2. Install new version of IPython.

# n. Do posterior checking. After inference, go through observes can 
#    generate N data points from posterior for each expression. Plot
#    the data points along with the observed value (maybe do t test also).



def qqPlotAll(dicts,labels):
    
    # FIXME do interpolation where samples mismatched
    assert len(dicts[0])==len(dicts[1])
    numExps = len(dicts[0])
    fig,ax = plt.subplots(numExps,2,figsize=(12,4*numExps))
    
    for i,exp in enumerate(dicts[0].keys()):
        s1,s2 = (dicts[0][exp],dicts[1][exp])
        assert len(s1)==len(s2)

        def makeHists(ax):
            ax.hist(s1,bins=20,alpha=0.7,color='r',label=labels[0])
            ax.hist(s2,bins=20,alpha=0.4,color='y',label=labels[1])
            ax.legend()
            ax.set_title('Histogram: %s'%exp)

        def makeQQ(ax):
            ax.scatter(sorted(s1),sorted(s2),s=4,lw=0)
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax.set_title('QQ Plot %s'%exp)
            xr = np.linspace(min(s1),max(s1),30)
            ax.plot(xr,xr)
            
        if numExps==1:
            makeHists(ax[0])
            makeQQ(ax[1])
        else:
            makeHists(ax[i,0])
            makeQQ(ax[i,1])

    fig.tight_layout()
    return fig


def filterScalar(dct):
    'Remove non-scalars from {exp:values}'
    scalar=lambda x:isinstance(x,(float,int))
    scalarDct={}
    for exp,values in dct.items():
        if all(map(scalar,values)):
            scalarDct[exp]=values
    return scalarDct

def compareSampleDicts(dicts_hists,labels,plot=False):
    '''Input: dicts_hists :: ({exp:values}) | (History)
     where the first Series in History is used as values. History's are
     converted to dicts.''' 

    if not isinstance(dicts_hists[0],dict):
        dicts = [historyNameToValues(h,seriesInd=0) for h in dicts_hists]
    else:
        dicts = dicts_hists
        
    dicts = map(filterScalar,dicts) # could skip for Analytics
        
    stats = (np.mean,np.median,np.std,len) # FIXME stderr
    stats_dict = {}
    print 'compareSampleDicts: %s vs. %s \n'%(labels[0],labels[1])
    
    for exp in dicts[0].keys():
        stats_dict[exp] = []
        for dict_i,label_i in zip(dicts,labels):
            samples=dict_i[exp]
            s_stats = tuple([s(samples) for s in stats])
            stats_dict[exp].append(s_stats)
            print '\nDict: %s. Exp: %s'%(label_i,exp)
            print 'Mean, median, std, N = %.3f  %.3f  %.3f  %i'%s_stats

        testResult=reportSameContinuous(dicts[0][exp],dicts[1][exp])
        print 'KS SameContinuous:', '  '.join(testResult.report.split('\n')[-2:])
        stats_dict[exp].append( testResult )
        
    fig = qqPlotAll(dicts,labels) if plot else None
    
    return stats_dict,fig


##### HISTORY UTILS
def historyNameToValues(history,seriesInd=0,flatten=False):
    ''':: History -> {name:values}. Default is to take first series.
    If flatten then we combine all.'''
    nameToValues={}
    for name,listSeries in history.nameToSeries.items():
        if flatten:
            values = [el for series in listSeries for el in series.values]
        else:
            values = listSeries[seriesInd].values
        nameToValues[name]=values
    return nameToValues


def convertHistory(nameToSeries,label='convertHistory',data=None):
    'Takes {name:[series_vals0,series_vals1,...]} and creates history.History'
    assert isinstance(nameToSeries.values()[0],list)
    true_nameToSeries={name:[] for name in nameToSeries.keys()}
    for name,listSeries in nameToSeries.items():
        for series in listSeries:
            seriesLabel = '%s: %s'%(label,name)
            true_nameToSeries[name].append( Series(seriesLabel, series))
    history = History(label=label)
    history.nameToSeries = true_nameToSeries
    if data is not None: history.data = data
    return history

def makeNameToSeries(assumes,queryExps,queryAssumes):
    ##FIXME: not {name:[history.Series0]} but {name:values0}
    if queryExps is None:
        queryExps = []
    nameToSeries = {exp:[] for exp in queryExps}
    if queryAssumes: nameToSeries.update( {exp:[] for exp,_ in assumes} )
    return nameToSeries

def makeHistoryForm(nTSeries_or_lstNTSeries):
    if not isinstance(nTSeries_or_lstNTSeries,(tuple,list)):
        lstNTS=[nTSeries_or_lstNTSeries]
    else:
        lstNTS=nTSeries_or_lstNTSeries
    historyForm = {}
    for name in lstNTS[0].keys():
        historyForm[name] = [nTSeries[name] for nTSeries in lstNTS]
    return historyForm

def makeSnapshots(history):
    ':: {name0:[name0_series0,name0_series1,...,], name1: ,...}'
    if not isinstance(history,History):
        history = convertHistory(history)
    snapshots={}
    for name,listSeries in history.nameToSeries.items():
        arrayValues = np.array( [s.values for s in listSeries] )
        snapshots[name] = map(list,arrayValues.T) 
    return snapshots


## INFERENCE UTILS
def makeClearRipl(r_mr): 
    if isinstance(r_mr,MRipl):
        return MRipl(r_mr.no_ripls, backend=r_mr.backend,
                     local_mode=r_mr.local_mode)
    else:
        return mk_p_ripl() ## FIXME

        
def riplToArgs(ripl_mripl,totalSamples=100,stepSize=None,queryExps=None):
    assumes,observes = riplToAssumesObserves(ripl_mripl)
    args=(mk_p_ripl(),assumes,observes,totalSamples)
    kwargs=dict(queryExps=queryExps, queryAssumes=False, stepSize=stepSize,
                infer=None)
    return args,kwargs

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
    
##########


## INFERENCE AND DIAGNOSTICS

    
def geweke(ripl,assumes,observes,totalSamples,queryExps=None,
           stepSize=None,queryAssumes=True,infer=None):
    '''Geweke 2004 test. Sample values for *observes* from current
    ripl state, observe them, infer(*stepSize*), then forget observes.
    Repeat for totalSamples. Determine ripl seed and backend via *ripl*.'''
    
    if stepSize is None:
        stepSize = defaultSweep(assumes) # rough version of sweep
        
    ripl=makeClearRipl(ripl)
    [ripl.assume(sym,exp) for sym,exp in assumes]
    
    if len(observes[0])==2: # not observes=[exp1,...,]
        observes = [exp for exp,literal in observes]
    
    nameToSeries = makeNameToSeries(assumes,queryExps,queryAssumes)
    
    for sample in range(totalSamples):
        [series.append(ripl.sample(exp)) for exp,series in nameToSeries.items()]
        [ripl.observe(exp, ripl.sample(exp)) for exp in observes]
        
        if sample < totalSamples - 1:
            progInfer(ripl,stepSize,infer=infer)
        forgetAllObserves(ripl)

    return nameToSeries


def compareGeweke(plots,*args,**kwargs):
    '''Generate prior and Geweke samples and compare with QQ plot.
    Inputs: plots(bool),ripl,assumes,observes,totalSamples,
    queryExps=None,stepSize=None,queryAssumes=True,infer=None'''

    assumes,observes,totalSamples=args[1:4]
    mripl=MRipl(2,local_mode=True) ##FIXME localmode
    nameToSeriesGeweke=geweke(*args,**kwargs)
    nameToSeriesForward=mrForwardSample(mripl,*args,**kwargs)
    labels=['geweke','forward']
    print 'compareGeweke:'
    print 'assumes=%s \nobserves=%s \ntotalSamples=%i \n'%(assumes,observes,
                                                          totalSamples)
    stats_dict,fig=compareSampleDicts([nameToSeriesGeweke,
                                   nameToSeriesForward],labels,plot=True)
    observes = args[2]
    hisGeweke = convertHistory(makeHistoryForm(nameToSeriesGeweke),
                             label=labels[0],data=observes)
    hisForward = convertHistory(makeHistoryForm(nameToSeriesForward),
                                label=labels[1],data=None)
    histOver=historyOverlay('Geweke vs. Forward', [(hisGeweke.label,hisGeweke),
                                                (hisForward.label,hisForward)])
   
    if plots:
        for name in stats_dict.keys(): # filter non-scalars
            histOver.quickPlot(name)
    return fig,histOver



def runFromConditional(ripl,assumes,observes,totalSamples,queryExps=None,
                       stepSize=None, queryAssumes=True,infer=None,mripl=None):
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


def mrRFC(mripl,*argsRFC,**kwargsRFC):
    '''RunFromConditional mapped across mripl with same observes.
    *totalSamples* is total per ripl in mripl. Input mripl is not
    mutated. A new mripl with same attributes is created, mutated
    by inference and output.'''
    argsRFC=argsRFC[1:] # remove *ripl* for mr_map_proc
    newMripl=makeClearRipl(mripl)
    list_nameToSeries = mr_map_proc(newMripl,'all',runFromConditional,*argsRFC,**kwargsRFC)
    historyForm = makeHistoryForm(list_nameToSeries)
    return historyForm,newMripl 


def plotSnapshots(snapshots,indices=(0,-1),label='RFC',groundTruth=None):
    '''Inputs: snapshots={name:[snapshot0,...],..}. Indices are for
    snapshot lst'''
## FIXME add groundTruth support
    fig,ax = plt.subplots(len(snapshots),figsize=(3,3*len(snapshots)))
    for i,(exp,shots) in enumerate(snapshots.iteritems()):

        if indices==(0,-1):
            ax[i].hist(shots[0],bins=20,alpha=0.4,color='y',label='Prior')
            ax[i].hist(shots[-1], bins=20, alpha=0.7, color='b',
                       label='Last snapshot')
        else:    
            for ind in indices:
                ax[i].hist(shots[ind],
                           bins=20,alpha=0.5,label='Index %i'%ind)
        ax[i].legend()
        ax[i].set_title('%s snapshots: Exp=%s'%(label,exp))
    fig.tight_layout()
    return fig


def plotRFC(*args,**kwargs):
    '''Plot Markov chains (vs. groundTruth), prior and posterior snapshots.
    Inputs: historyRFCargs, historyRFCkwargs'''
    HistRFC,outMRipl=historyRFC(*args,**kwargs)
    snapshots = makeSnapshots(HistRFC)
    snapshotsFig = plotSnapshots(snapshots,indices=(0,-1),
                                 label=HistRFC.label)
    for name in HistRFC.nameToSeries.keys():
        HistRFC.quickPlot(name)

    # compare middle and last snapshots (convergence check)
    midPoint = int(.5 * len(snapshots.values()[0]) )
    middle = {exp:shots[midPoint] for exp,shots in snapshots.items()}
    last = {exp:shots[-1] for exp,shots in snapshots.items()}
    
    compareOuts=compareSampleDicts([middle,last],
                                   ['Middle Snapshot','Last Snapshot'],
                                   plot=True)
    return snapshots,snapshotsFig,outMRipl,compareOuts


def historyRFC(*args,**kwargs):
    '''RunFromConditional, with mripl or ripl, make History with
    groundTruth.
    Input: RFCargs,RFCkwargs,mripl=None,groundTruth=None,label=RFC'''
    mripl=kwargs.pop('mripl',None)
    groundTruth=kwargs.pop('groundTruth',None)
    label=kwargs.pop('label','RFC')
    observes=args[2]
    totalSamples=args[3]

    if mripl is None:
        historyForm=makeHistoryForm( runFromConditional(*args,**kwargs) )
    else:
        historyForm,outMRipl = mrRFC(mripl,*args,**kwargs)
        
    HistRFC = convertHistory(historyForm,label=label,data=observes)
    
    if groundTruth is not None:
        HistRFC.addGroundTruth(groundTruth,totalSamples)
    return HistRFC,outMRipl


def multiConditionFromPrior(mripl,noDatasets,assumes,observes,totalSamples,
                            **kwargsRFC):
    assert mripl.no_ripls == noDatasets
    seeds = [1]*noDatasets
    mripl.mr_set_seeds(seeds=seeds)

    datasets=[]
    for i in range(noDatasets):
        v=mk_p_ripl()
        v.set_seed(i)
        [v.assume(sym,exp) for sym,exp in assumes]
        datasets.append( [(exp,v.sample(exp)) for exp,_ in observes] )
        
    args = [(assumes,observes_i,totalSamples) for observes_i in datasets]
    kwargs = [kwargsRFC] * noDatasets
    argsList = zip(args,kwargs)

    list_nameToSeries = mr_map_array(mripl,runFromConditional,argsList,no_kwargs=False)
    historyForm = makeHistoryForm(list_nameToSeries)


    no_exps = len(historyForm.keys())
    fig,ax = plt.subplots(max(no_exps,2),1)
    for exp_ind,(exp,all_datasets) in enumerate(historyForm.items()):
        for dataset_ind,dataset in enumerate(all_datasets):
            ax[exp_ind].plot( dataset, label='Dataset %i' % dataset_ind)
        ax[exp_ind].set_title('Exp: %s, [multiConditionFromPrior]' % exp )
        ax[exp_ind].legend()
    fig.tight_layout()
    ## FIXME: add the comparison the samples from the prior (see test)
    # : forward sample with same mripl,assumes, etc.
    # the do compare dist and QQ plot and plot over time
    
    return historyForm,mripl


def mrForwardSample(mripl,ripl,assumes,observes,totalSamples,queryExps=None,
                    queryAssumes=True,**kwargs):
    'ripl arg present only so args are superset of RFC'
    mripl = MRipl(totalSamples,
                  backend=mripl.backend, local_mode=mripl.local_mode)
    nameToSeries = makeNameToSeries(assumes,queryExps,queryAssumes)
    [mripl.assume(sym,exp) for sym,exp in assumes]
    return {name:mripl.sample(name) for name in nameToSeries}




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
    nameToSeries = geweke(ripl,assumes,observes,totalSamples,stepSize=5)
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
        qqPlotAll((nameToSeries,dict2),('Geweke',label2))
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


from lda_files.lda_utils import *
def ulda(clear_ripl_mripl,totalSamples=100,stepSize=100,queryExps=None):

    gen_params={'true_no_topics':4, 'size_vocab':10,
                'no_docs':5, 'doc_length':20, 
                'alpha_t_prior':'(gamma .2 1)',
                'alpha_w_prior':'(gamma .2 1)'}
    generate_docs_out = generate_docs(gen_params)
    print_summary(generate_docs_out)

    # ripl with model and docs as observes
    v=mk_p_ripl()
    v.execute_program(mk_lda(gen_params,collapse=True))

    atomObserves=[] # problem of atoms being removed from python list_dir
    for doc_ind,doc in enumerate(generate_docs_out['data_docs']):
        for word_ind in range(len(doc)):
            exp='(word atom<%i> %i)'%(doc_ind,word_ind)
            val='atom<%i>'%doc[word_ind]
            v.observe(exp,val)
            atomObserves.append( (exp,val) )
                
    assumes,observes = riplToAssumesObserves(v)
    observes = atomObserves
    args=(clear_ripl_mripl,assumes,observes,totalSamples)
    if queryExps is None:
        dl = gen_params['doc_length']
        st_lst=['(word atom<0> %i)'%i for i in range(dl,dl+20)]
        exp = '(list %s )'% ' '.join(st_lst)
        queryExps = ['alpha_t','alpha_w', exp]
        
    kwargs=dict(queryExps=queryExps, queryAssumes=False, stepSize=stepSize,
                infer=None)

    return args,kwargs


def quadratic_linear_obs(clear_ripl_mripl,totalSamples=100,
                         stepSize=10,queryExps=None):
    ripl=mk_p_ripl() 
    ripl.execute_program(simple_quadratic_model)
    [ripl.observe('(y_x %i)'%i, str(i)) for i in range(-5,5) ]
    assumes,observes = riplToAssumesObserves(ripl)
    args=(clear_ripl_mripl,assumes,observes,totalSamples)
    if queryExps is None:
        queryExps = ['w0','w1','w2','(f 0)','noise']
    kwargs=dict(queryExps=queryExps, queryAssumes=False, stepSize=stepSize,
                infer=None)
    return args,kwargs


simple_quadratic_model='''
[assume w0 (normal 0 3) ]
[assume w1 (normal 0 1) ]
[assume w2 (normal 0 .3) ]
[assume x (mem (lambda (i) (x_d) ) )]
[assume x_d (lambda () (normal 0 5))]
[assume noise (gamma 2 1) ]
[assume f (lambda (x) (+ w0 (* w1 x) (* w2 (* x x)) ) ) ]
[assume y_x (lambda (x) (normal (f x) noise) ) ]
[assume y (mem (lambda (i) (y_x (x i)) ) )]
[assume model_name (quote simple_quadratic)]'''























## OLD

def addHistory(History1,History2):
    History1.label = '%s + %s'%(History1.label,History2.label)
    for (name, seriesList) in History2.nameToSeries.iteritems():
        for seriesObj in seriesList:
            History1.nameToSeries[name].append(seriesObj)


def longTest():
    args,kwargs = quadratic_linear_obs(mk_p_ripl(),
                                       stepSize=100, queryExps=['w0','w1','noise'],
                                       totalSamples=500)
    fig,hisGeweke,hisForward=compareGeweke(False,*args,**kwargs)
    return fig,hisGeweke,hisForward


def qq_plot(s1,s2,label1,label2):
    fig,ax = plt.subplots(figsize=(3,3))
    ax.scatter(sorted(s1),sorted(s2),s=4,lw=0)
    ax.set_xlabel(label1); ax.set_ylabel(label2)
    ax.set_title('PP Plot')



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
