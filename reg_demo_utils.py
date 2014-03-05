import numpy as np
from numpy.random import randn
import pandas as pd
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
#import seaborn as sns

def hexbin_plot(x,y):
    fig,ax = subplots()
    ax.hexbin(x, y, gridsize=40, cmap="BuGn", extent=(min(x),max(x), min(y),max(y)) )
    return fig,ax

def kde_plot(x,y):
    fig,ax = subplots()
    sns.kdeplot(x,y,shade=True,cmap=None,ax=ax)
    return fig,ax

from venture.venturemagics.ip_parallel import *; 
lite=False; clear_all_engines()
mk_l = make_lite_church_prime_ripl; mk_c = make_church_prime_ripl


x_model_crp='''
[assume alpha (uniform_continuous .01 1)]
[assume crp (make_crp alpha) ]
[assume z (mem (lambda (i) (crp) ) ) ]
[assume mu (mem (lambda (z) (normal 0 5) ) ) ] 
[assume sig (mem (lambda (z) (uniform_continuous .1 8) ) ) ]
[assume x_d (lambda () ( (lambda (z) (normal (mu z) (sig z) )) (crp) ) ) ]
[assume x (mem (lambda (i) (normal (mu (z i)) (sig (z i))))  ) ]
'''
x_model_t='''
[assume nu (gamma 10 1)]
[assume x_d (lambda () (student_t nu) ) ]
[assume x (mem (lambda (i) (x_d) ) )]
'''
pivot_model='''
[assume w0 (mem (lambda (p)(normal 0 3))) ]
[assume w1 (mem (lambda (p)(normal 0 3))) ]
[assume w2 (mem (lambda (p)(normal 0 1))) ]
[assume noise (mem (lambda (p) (gamma 2 1) )) ]
[assume pivot (normal 0 5)]
[assume p (lambda (x) (if (< x pivot) 0 1) ) ]

[assume f (lambda (x)
             ( (lambda (p) (+ (w0 p) (* (w1 p) x) (* (w2 p) (* x x)))  ) 
               (p x)  ) ) ]

[assume noise_p (lambda (fx x) (normal fx (noise (p x))) )] 

[assume y_x (lambda (x) (noise_p (f x) x) ) ]
                     
[assume y (mem (lambda (i) (y_x (x i))  ))] 
                     
[assume n (gamma 1 1) ]
[assume model_name (quote pivot)]
'''

pivot_check='''
[observe (x 0) 0.]
[observe pivot 10.]
[observe (w0 0) 0.]
[observe (w1 0) 1.]
[observe (w2 0) 0.]
[observe (noise 0) .01]'''

def test_pivot():
    v_crp=mk_c(); v_crp.execute_program(x_model_crp + pivot_model)
    v_t=mk_l(); v_t.execute_program(x_model_t+pivot_model)
    vs=[v_t,v_crp]
    for v in vs:
        v.execute_program(pivot_check)
        v.infer(1)
        assert v.predict('(= 0 (p (x 0)))')
        assert .1 > (0 - v.predict('(f (x 0))'))
        assert .5 > (0 - v.assume('y0','(y 0)') ) # y0 close to 0
        assert .5 > (0 - v.predict('(y_x (x 0))'))

        f= np.array( [v.predict('(f %i)' % i) for i in range(5)] )
        assert all( 0.1 > np.abs(f - np.arange(5)) )
        [v.observe('(y %i)' % i, str(i+.01) ) for i in range(20,25)]
        y_x20 = np.array( [v.predict('(y_x %i)' % i) for i in range(20,25)] )
        y20 = np.array( [v.predict('(y %i)' % i) for i in range(20,25)] )
    #assert all( [ 2 > (y_x20[i] - y20[i]) for y_x20,y20 ] ) 


quad_fourier_model='''
[assume w0 (normal 0 3) ]
[assume w1 (normal 0 3) ]
[assume w2 (normal 0 3) ]
[assume omega (normal 0 3) ]
[assume theta (normal 0 3) ]
[assume noise (gamma 2 1) ]

[assume model (if (flip) 1 0) ]
[assume quadratic (lambda (x) (+ w0 (* w1 x) (* w2 (* x x) ) ) ) ]
[assume fourier (lambda (x) (+ w0 (* w1 (sin (+ (* omega x) theta) ) ) ) ) ]
[assume f (if (= model 0) quadratic fourier) ]

[assume y_x (lambda (x) (normal (f x) noise) ) ]
[assume y (mem (lambda (i) (normal (f (x i) ) noise) ) )]
[assume n (gamma 1 1)]
[assume model_name (quote quad_fourier)]'''
quad_fourier_checks='''
[observe (x 0) 0.]
[observe w0 0.]
[observe w1 1.]
[observe w2 0.]
[observe model 0]
[observe noise .01]
'''

def test_quad_fourier(v):
    v=mk_c()
    v.execute_program(x_model_t + quad_fourier_model)
    v.execute_program(quad_fourier_checks)
    v.infer(1)
    assert .1 > abs( 0 - v.predict('(x 0)') )
    assert 0 == v.predict('model')
    assert 2 > abs( (0 - v.predict('(y 0)') ) )
    xfs = [v.predict('(list x (f (x %i) ))' % i) for i in range(10) ]
    xys = [v.predict('(list (x %i) (y %i))' % (i,i)) for i in range(10) ];
    assert all( [ .5 > (xy[0] - xy[1]) for xy in xys ] )
    assert all( [ xf[1] - xy[1] for (xf,xy) in zip(xfs,xys) ] )    
    assert .1 > ( v.predict('(y 0)') - v.predict('(y_x 0)') )


def generate_data(n,xparams=None,yparams=None,sin=True):
    'loc,scale = xparams, w0,w1,w2,omega,theta = yparams'
    if xparams:
        loc,scale = xparams; xs = np.random.normal(loc,scale,n)
    else:
        xs = np.random.normal(loc=0,scale=2.5,size=n)
    if yparams:
        w0,w1,w2,omega,theta = yparams
        ys = w0*(np.sin(omega*xs + theta))+w1 if sin else w0+(w1*xs)+(w2*(xs**2))
    else:
        ys = 3*np.sin(xs)
        
    xys = zip(xs,ys)
    fig,ax = plt.subplots(figsize=(6,2))
    ax.scatter(xs,ys);
    ax.set_title('Data from f w/ %s )' % str(yparams) ) if yparams else ax.set_title('Data from 3sin(x)')
    return xys


def observe_infer(vs,xys,no_transitions,with_index=False,withn=True):
    '''Input is list of ripls or mripls, xy pairs and no_transitions. Optionally
    observe the n variable to be the len(xys). We can either index the observations
    or we can treat them as drawn from x_d and y_x, which do not memoize but depend
    on the same hidden params. (Alternatively we could work out the last index and
    start from there).'''
    if with_index:
        for i,(x,y) in enumerate(xys):
            [v.observe('(x %i)' % i , '%f' % x, label='x%i' % i) for v in vs]
            [v.observe('(y %i)' % i , '%f' % y, label='y%i' % i ) for v in vs]
    else:        
        ## FIXME find some good labeling scheme
        for i,(x,y) in enumerate(xys):
            [v.observe('(x_d)', '%f' % x ) for v in vs]
            [v.observe('(y_x %f)' % x , '%f' % y ) for v in vs]
    if withn: [v.observe('n','%i' % len(xys)) for v in vs]

    [v.infer(no_transitions) for v in vs];


def logscores(mr,name='Model'):
    logscore = mr.get_global_logscore()
    try: name=mr.sample('model_name')
    except: pass
    print '%s logscore: (mean, max) ' % name, np.mean(logscore), np.max(logscore)
    return np.mean(logscore), np.max(logscore)


def plot_cond(ripl,no_reps=50):
    '''Plot f(x) with 1sd noise curves. Plot y_x with #(no_reps)
    y values for each x. Use xrange with limits based on posterior on P(x).'''
    
    # find x-range from min/max of observed points
    n = int( np.round( ripl.sample('n') ) )  #FIXME
    xs = [ripl.sample('(x %i)' % i) for i in range(n)]
    ys = [ripl.sample('(y %i)' % i) for i in range(n)]
    xr = np.linspace(1.5*min(xs),1.5*max(xs),20)
    f_xr = [ripl.sample('(f %f)' % x) for x in xr]
    
    # gaussian noise 1sd
    if 'model_name' in str(ripl.list_directives()):
        try: model_name = ripl.sample('model_name')
        except: pass
    else:
        model_name = 'anon model'
    noise=ripl.sample('(noise 0)') if model_name=='pivot' else ripl.sample('noise')
    f_a = [fx+noise for fx in f_xr]
    f_b = [fx-noise for fx in f_xr]

    # scatter for y conditional on x
    y_x = [  [ripl.sample('(y_x %f)' % x) for r in range(no_reps)] for x in xr]
    fig,ax = plt.subplots(1,2,figsize=(9,2),sharex=True,sharey=True)
    ax[0].scatter(xs,ys)
    ax[0].set_color_cycle(['m', 'gray','gray'])
    ax[0].plot(xr,f_xr,xr,f_a,xr,f_b)
    ax[0].set_title('Data and inferred f with 1sd noise (name= %s )' % model_name)
    
    [ ax[1].scatter(xr,[y[i] for y in y_x],s=6) for i in range(no_reps) ]
    ax[1].set_title('Single ripl: P(y/x) for uniform x-range (name= %s)' % model_name)
    
    return xs,ys


def plot_ygivenx(mr,x):
    return mr.snapshot(exp_list=['(y_x %f)' % x ],plot=True)


def plot_xgiveny(mr,y,no_transitions=100):
    '''P(x / Y=y), by combining ripls in mr. Works by finding next unused observation
    label, setting Y=y for that observation, running inference, sampling x and then
    forgetting the observation of y. NB: locally disruptive of inference.'''
    
    obs_label = [di for di in mr.list_directives()[0] if di['instruction']=='observe' and di.get('label')]
    # labels should have form 'y1','y2', etc.
    if obs_label:
        y_nums = [int(di['label'][1:]) for di in obs_label if di['label'].startswith('y')]
        next_label = max(y_nums)+1
    else:
        next_label = int(np.random.randint(1000,10**8))
    
    mr.observe('(y %i)' % next_label, str(y), label='y%i' % next_label )
    mr.infer(no_transitions)
    snapshot=mr.snapshot(exp_list=['(x %i)' % next_label],plot=True)
    mr.forget('y%i' % next_label)
    return snapshot


def plot_joint(ripl,no_reps=300):
    '''Sample from joint P(x,y) and plot as histogram and kde.'''
    xs = [ ripl.sample('(x_d)') for i in range(no_reps) ]
    ys = [ ripl.sample('(y_x %f)' % x) for x in xs]
    
    fig,ax = plt.subplots(1,2,figsize=(9,2),sharex=False,sharey=False)
    ax[0].scatter(xs,ys)
    ax[0].set_title('Single ripl: %i samples from P(y,x)' % no_reps)
    H, xedges, yedges = np.histogram2d(xs, ys, bins=(10, 10))
    extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]
    ax[1].imshow(H, extent=extent, interpolation='nearest')
    ax[1].set_title('Single ripl: hist of %i samples from P(y,x)' % no_reps)
    return xs,ys


def params_compare(mr,exp_pair,xys,no_transitions,plot=False):
    '''Look at dependency as data comes in'''
    
    # get prior values
    out_pr = mr.snapshot(exp_list=exp_pair,plot=plot,scatter=False)
    vals_pr = out_pr[-1]['values']
    out_list = []; vals_list=[] 
    
    # add observes
    for i,xy in enumerate(xys):
        observe_infer([mr],[xy],no_transitions,with_index=False,withn=False) # FIXME obs n somewhere?
        out_list.append( mr.snapshot(exp_list=exp_pair,plot=plot,scatter=False) )
        vals_list.append( out_list[-1]['values'] )
        
    # plot prior
    fig,ax = plt.subplots(size
    ax[0,0].scatter(vals_list[i][exp_pair[0]], vals_list[i][exp_pair[0]], s=6))
    ax[0,0].set_title('%s vs. %s (prior)' % (exp_pair[0],exp_pair[1]))
    ax[0,0].set_xlabel(exp_pair[0]); ax[i,0].set_ylabel(exp_pair[1])
    
    xys=np.array(xys); xs=xys[:,0]; ys=xys[:,1]

    fig,ax = plt.subplots(len(xys)+1,2,figsize=(13,4))
    for i,xy in enumerate(xys):
        ax[i,0].scatter(vals_list[i][exp_pair[0]], vals_list[i][exp_pair[0]], s=6)
        ax[i,0].set_title('%s vs. %s' % (exp_pair[0],exp_pair[1]))
        ax[i,0].set_xlabel(exp_pair[0]); ax[i,0].set_ylabel(exp_pair[1])
        ax[i,1].scatter(xs[:i], ys[:i], c='blue')
        ax[i,1].scatter(xs[i], ys[i], c='red')
        ax[i,1].set_title('Observed data with new point')
        
    fig.tight_layout()
    return fig,vals_list

def test_params_compare():
    xys = [(2,20),(0,0),(0.5,2),(-0.4,2)]#,(1,5)]#,(-1,5),(2.5,31)] # y=5x^2
    v_pivt = MRipl(20,lite=lite,verbose=False); v_pivcrp = MRipl(2,lite=lite,verbose=False)
    vs = [v_pivt,v_pivcrp]
    v_pivt.execute_program(x_model_t+quad_fourier_model)
    v_pivcrp.execute_program(x_model_crp+quad_fourier_model)
    
    no_transitions=50
    outs=[params_compare(v,['w2','noise'],xys,no_transitions) for v in vs]
    return None


def if_lst_flatten(l):
    if type(l[0])==list: return [el for subl in l for el in subl]
    return l

def test_funcs(mripl=False,n=14,fast=False):
    xys = generate_data(n,xparams=[0,3],yparams=[0,0,1,0,0],sin=False) # y=x^2
    if fast: xys = generate_data(8,xparams=[0,3],yparams=[0,0,1,0,0],sin=False) # y=x^2
    if mripl:
        v_piv = MRipl(2,lite=lite,verbose=False); v_fo = MRipl(2,lite=lite,verbose=False)
        vs = [v_piv,v_fo]
    else:
        v_piv = mk_c(); v_fo=mk_c(); vs = [v_piv,v_fo]
    v_piv.execute_program(x_model_t+pivot_model); v_fo.execute_program(x_model_t+quad_fourier_model)

    observe_infer(vs,xys,50,withn=True) if fast else observe_infer(vs,xys,300,withn=True)

    [logscores(v) for v in vs]

    if mripl:
        [mr_map_nomagic(v,plot_cond,limit=1) for v in vs]
    else:
        [plot_cond(v) for v in vs]

    if mripl: ## FIXME look over this test again
        outs = [mr_map_nomagic(v,plot_joint,limit=1)['out'][0] for v in vs]
        for out in outs:
            xs,ys = out
            print np.mean(xs), np.mean(ys)
            if not(fast):
                assert( 2 > np.abs( np.mean(xs) ) )
            else:
                assert( 4 > np.abs( np.mean(xs) ) )
    else:
        outs=[plot_joint(v,no_reps=200) for v in vs]
        for out in outs:
            xs,ys = out
            if not(fast):
                assert( 2 > np.abs( np.mean(xs) ) )
            else:
                assert( 4 > np.abs( np.mean(xs) ) )
        

    # in-sample guess should be close to true vals after enough inference
    [v.infer(300) for v in vs] if not(fast) else [v.infer(100) for v in vs]
    f0 = if_lst_flatten( [v.predict('(f 0)') for v in vs] )
    f1 = if_lst_flatten( [v.predict('(f 1)') for v in vs] )
    f1 = if_lst_flatten( [v.predict('(f -1)') for v in vs] )
    assert all( 10 > np.abs((0 - np.array(f0) ) )) and any( 5 > np.abs((0 - np.array(f0) ) ) )
    assert all( 10 > np.abs((1 - np.array(f1) ) )) and any( 5 > np.abs((1 - np.array(f0) ) ) )
    assert all( 10 > np.abs((1 - np.array(f1) ) )) and any( 5 > np.abs((1 - np.array(f0) ) ) )

    if mripl:  ##FIXME, look over xgiveny for how much inference we need
        snap_outs= [plot_ygivenx(v,0) for v in vs]
        ygiven0 = np.array( snap_outs[0]['values'].values()[0] )
        assert all( 6 > np.abs((0 - ygiven0) ) )

        snap_outs= [plot_xgiveny(v,0) for v in vs]
        xgiven0 = np.array( snap_outs[0]['values'].values()[0] )
        assert all( 6 > np.abs((0 - xgiven0) ) )
    



   


### PLAN: different plots/scores
#1. av logscore and best logscore.
# 2. plot the curve, adding noise error (easiest way is with y_x)
# 3. plot the joint (sample both x's and y's)
# 4. plot posterior on sets of params
# 5. plot posterior conditional
# 6. plot posterior joint (the posterior join density over x,y: get from running chain long time or combining chains)
# 7. plot p(x / y) for some particular y's 
