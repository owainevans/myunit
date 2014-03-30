import numpy as np
import matplotlib.pyplot as plt
from venture.venturemagics.ip_parallel import *; 
lite=False; 
mk_l_ripl = make_lite_church_prime_ripl; mk_p_ripl = make_puma_church_prime_ripl
vs = test_ripls()


simple_fourier_model='''
[assume w0 (normal 0 3) ]
[assume w1 (normal 0 3) ]
[assume omega (normal 0 3) ]
[assume theta (normal 0 3) ]
[assume x (mem (lambda (i) (x_d) ) )]
[assume x_d (lambda () (normal 0 5))]
[assume noise (gamma 2 1) ]
[assume f (lambda (x) (+ w0 (* w1 (sin (+ (* omega x) theta) ) ) ) ) ]
[assume y_x (lambda (x) (normal (f x) noise) ) ]
[assume y (mem (lambda (i) (y_x (x i))  ))] 
[assume n (gamma 1 1)]
[assume model_name (quote simple_fourier)]
'''
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
[assume model_name (quote simple_quadratic)]
'''
hi_quadratic_model='''
[assume mu_prior (mem (lambda (j) (normal 0 20))) ]
[assume sigma_prior (mem (lambda (j) (gamma 1 1)) ) ]
[assume w (mem (lambda (gp j) (normal (mu_prior j) (sigma_prior j) ) ) ) ] 
[assume x (mem (lambda (gp i) (x_d gp) ) )]
[assume x_d (lambda (gp) (normal 0 3))]
[assume noise (gamma 2 1) ]
[assume f (lambda (gp x) (+ (w gp 0) (* (w gp 1) x) (* (w gp 2) (* x x)) ) ) ]
[assume y_x (lambda (gp x) (normal (f gp x) noise) ) ]
[assume y (mem (lambda (gp i) (y_x gp (x gp i)) ) )]
[assume n (gamma 1 1)]
[assume model_name (quote hi_quadratic)]
'''
crp_model2='''
[assume alpha (uniform_continuous .01 1)]
[assume crp (make_crp alpha) ]
[assume gp (mem (lambda (i) (crp) ) ) ]
[assume mu (mem (lambda (gp) (normal 0 5) ) ) ] 
[assume sig (mem (lambda (gp) (uniform_continuous .1 8) ) ) ]
[assume x_d (lambda () ( (lambda (gp) (normal (mu gp) (sig gp) )) (crp) ) ) ]
[assume x (mem (lambda (i) (normal (mu (gp i)) (sig (gp i))))  ) ]
[assume w (mem (lambda (gp j) (normal 0 (pow 2 (* -1 j)) ) ) )] 
[assume noise (gamma 1 1) ]
[assume pick_f (lambda (gp) (lambda (x)
                     (+ (w gp 0) (* (w gp 1) x) (* (w gp 2) (* x x)) ) ) ) ]
[assume y_x (lambda (gp x) (normal ( (pick_f gp) x) noise) ) ]
[assume y (mem (lambda (i) (y_x (gp i) (x i)) ) )]
'''
crp_model='''
[assume alpha (uniform_continuous .01 1)]
[assume crp (make_crp alpha) ]
[assume gp (mem (lambda (i) (crp) ) ) ]
[assume mu (mem (lambda (gp) (normal 0 5) ) ) ] 
[assume sig (mem (lambda (gp) (uniform_continuous .1 8) ) ) ]
[assume x_d (lambda () ( (lambda (gp) (normal (mu gp) (sig gp) )) (crp) ) ) ]
[assume x (mem (lambda (i) (normal (mu (gp i)) (sig (gp i))))  ) ]
[assume w (mem (lambda (gp j) (normal 0 (pow 2 (* -1 j)) ) ) )] 
[assume noise (gamma 1 1) ]
[assume f (lambda (gp x) (+ (w gp 0) (* (w gp 1) x) (* (w gp 2) (* x x)) ) ) ]

[assume y (mem (lambda (i) (normal (f (gp i) (x i)) noise ) ))]
[assume model_name (quote crp)]
'''
#[assume y_x (lambda (gp x) (normal (f gp x) noise) ) ]

x_model_t='''
[assume nu (gamma 10 1)]
[assume x_d (lambda () (student_t nu) ) ]
[assume x (mem (lambda (i) (x_d) ) )]
'''
x_model_crp='''
[assume alpha (uniform_continuous .01 1)]
[assume crp (make_crp alpha) ]
[assume z (mem (lambda (i) (crp) ) ) ]
[assume mu (mem (lambda (z) (normal 0 5) ) ) ] 
[assume sig (mem (lambda (z) (uniform_continuous .1 8) ) ) ]
[assume x_d (lambda () ( (lambda (z) (normal (mu z) (sig z) )) (crp) ) ) ]
[assume x (mem (lambda (i) (normal (mu (z i)) (sig (z i))))  ) ]
'''
pivot_model='''
[assume w0 (mem (lambda (p)(normal 0 3))) ]
[assume w1 (mem (lambda (p)(normal 0 1))) ]
[assume w2 (mem (lambda (p)(normal 0 .3))) ]
[assume noise (mem (lambda (p) (gamma 2 1) )) ]
[assume pivot (normal 0 5)]
[assume p (lambda (x) (if (< x pivot) false true) ) ]

[assume f (lambda (x)
             ( (lambda (p) (+ (w0 p) (* (w1 p) x) (* (w2 p) (* x x)))  ) 
               (p x)  ) ) ]

[assume noise_p (lambda (fx x) (normal fx (noise (p x))) )] 

[assume y_x (lambda (x) (noise_p (f x) x) ) ]
              
[assume y (mem (lambda (i) (y_x (x i))  ))] 
                     
[assume n (gamma 1 100) ]
[assume model_name (quote pivot)]
'''
quad_fourier_model='''
[assume w0 (normal 0 3) ]
[assume w1 (normal 0 2.5) ]
[assume w2 (normal 0 .3) ]
[assume omega (normal 0 3) ]
[assume theta (normal 0 3) ]

[assume noise (gamma 2 1) ]

[assume model (if (flip) 1 0) ]
[assume quadratic (lambda (x) (+ w0 (* w1 x) (* w2 (* x x) ) ) ) ]
[assume fourier (lambda (x) (+ w0 (* w1 (sin (+ (* omega x) theta) ) ) ) ) ]
[assume f (if (= model 0) quadratic fourier) ]

[assume y_x (lambda (x) (normal (f x) noise) ) ]
[assume y (mem (lambda (i) (y_x (x i))  ))] 
[assume model_name (quote quad_fourier)]'''

logistic_model='''
[assume w0 (normal 0 3)]
[assume w1 (normal 0 3) ]
[assume log_mu (normal 0 3)]
[assume log_sig (normal 0 3) ]
[assume noise (gamma 2 1) ]

[assume sigmoid (lambda (x) (/ (- 1 (exp (* (- x log_mu) (* -1 log_sig) )) )
                               (+ 1 (exp (* (- x log_mu) (* -1 log_sig) )) ) ) )]
[assume f (lambda (x) (+ w0 (* w1 (sigmoid x) ) ) ) ]

[assume y_x (lambda (x) (normal (f x) noise) ) ]
[assume y (mem (lambda (i) (y_x (x i))  ))] 
[assume n (gamma 1 100)]
[assume model_name (quote logistic)]'''


def mk_piecewise(weight=.5,quad=True):
    s='''
    [assume myceil (lambda (x) (if (= x 0) 1
                                 (if (< 0 x)
                                   (if (< x 1) 1 (+ 1 (myceil (- x 1) ) ) )
                                   (* -1 (myceil (* -1 x) ) ) ) ) ) ]
    [assume w0 (mem (lambda (p)(normal 0 3))) ]
    [assume w1 (mem (lambda (p)(normal 0 3))) ]
    [assume w2 (mem (lambda (p)(normal 0 1))) ]
    [assume noise (mem (lambda (p) (gamma 5 1) )) ]
    [assume width <<width>>]

    [assume p_func (lambda (x) (1) )]
    [assume f (lambda (x)
                 ( (lambda (p) (+ (w0 p) (* (w1 p) x) (* (w2 p) (* x x)))  ) 
                   (p_func x)  ) ) ]
    [assume noise_p (lambda (x) 
                         (lambda (fx) (normal fx (noise (p_func x)) ) ) 
                            ) ]
    [assume y_x (lambda (x) ( (noise_p x) (f x) ) ) ]
    [assume y (mem (lambda (i) (y_x (x i))  ))] 
    [assume n (gamma 1 100) ]
    [assume model_name (quote piecewise)]
    '''
#    [assume p (lambda (x) (myceil (/ x width)))]
    if not(quad):
        s= s.replace('[assume w2 (mem (lambda (p)(normal 0 1))) ]',
                     '[assume w2 0]')
    return s.replace('<<width>>',str(weight))

def v_mk_piecewise(weight,quad):
    v=mk_l()
    v.execute_program(x_model_t_piece + mk_piecewise(weight=weight,quad=quad))
    return v


from scipy.stats import kde



def generate_data(n,xparams=None,yparams=None,sin_quad=True):
    'loc,scale = xparams, w0,w1,w2,omega,theta = yparams'
    if xparams:
        loc,scale = xparams; xs = np.random.normal(loc,scale,n)
    else:
        xs = np.random.normal(loc=0,scale=2.5,size=n)
    if yparams:
        w0,w1,w2,omega,theta = yparams
        params_d = {'w0':w0,'w1':w1,'w2':w2,'omega':omega,'theta':theta}
        ys = w0*(np.sin(omega*xs + theta))+w1 if sin_quad else w0+(w1*xs)+(w2*(xs**2))
    else:
        ys = 3*np.sin(xs)
        
    xys = zip(xs,ys)
    fig,ax = plt.subplots(figsize=(6,2)); ax.scatter(xs,ys)
    if yparams:
        if sin_quad:
            ax.set_title('Data from w0+w1*sin(omega(x-theta)) w/ %s )' % str(params_d) ) ## FIXME not whole dict
        else:
            ax.set_title('Data from w0+w1*x+w2*x^2 w/ %s )' % str(params_d) )
    else:
        ax.set_title('Data from 3sin(x)')
    return xys


def observe_xy(ripl_list,data,with_index=False):
    '''for each ripl in ripl_list, observe (x_d),(y_x x)
    if with_index=False, else observe (x i),(y i) starting from i=0'''
    if len(data[0])>2: data = zip(data[0],data[1])
    vs = ripl_list if isinstance(ripl_list,list) else [ripl_list]

    if with_index:
        for i,(x,y) in enumerate(data):
            [v.observe('(x %i)' % i , '%f' % x, label='x%i' % i) for v in vs]
            [v.observe('(y %i)' % i , '%f' % y, label='y%i' % i ) for v in vs]
    else:        
        for i,(x,y) in enumerate(data):
            [v.observe('(x_d)', '%f' % x ) for v in vs]
            [v.observe('(y_x %f)' % x , '%f' % y ) for v in vs]
    

def display_logscores(ripl_mripl):
    mr = ripl_mripl
    logscore = mr.get_global_logscore()
    name=get_name(mr) 
    print '%s logscore: (mean, max) ' % name, np.mean(logscore), np.max(logscore)
    return np.mean(logscore), np.max(logscore)



def posterior_conditional(mripl,data=[],x_range=(-3,3),no_xs=40,no_reps=40,figsize=(16,3.5),return_fig=False ):
    mr = mripl
    name=get_name(mr)
    
    if data:
        d_xs,d_ys = zip(*data)
        x_range = (min(d_xs)-1,max(d_xs)+1)
        if not x_range: x_range = (min(d_xs)-1,max(d_xs)+1)
    
    if not x_range: x_range = (-3,3)
        
    xr = np.linspace(x_range[0],x_range[1],no_xs)
    
    list_out=mr_plot_conditional(mr,plot=False,limit=6,data=data,x_range=x_range,no_xs=no_xs,no_reps=1)
    fs = [ ripl_out['f'] for ripl_out in list_out]
    

    ## get y_xs from ripls and compute 1sd intervals
    xys=[]; ymean=[]; ystd=[]
    for x in xr:
        # we get no_reps predicts from each ripl in mr
        x_y=if_lst_flatten([mr.sample('(y_x %f)' % x) for r in range(no_reps)])   
        ymean.append( np.mean(x_y) )
        ystd.append( np.abs( np.std(x_y) ) )
        xys.extend( [(x,y) for y in x_y] )
    
    xs,ys = zip(*xys)
    ymean = np.array(ymean); ystd = np.array(ystd)
    y_u = ymean+ystd; y_l = ymean-ystd
    
     # Plotting
    fig,ax = plt.subplots(1,3,figsize=figsize,sharex=True,sharey=True)

    if data: [ax[col].scatter(d_xs,d_ys,label='Data') for col in [0,1]]
    # sampled fs from mripl
    [ax[0].plot(xr,f_xr,alpha=.8,linewidth=.5) for xr,f_xr in fs]
    if data: ax[0].legend()
    ax[0].set_title('MR: Sampled fs w/ data [name: %s] ' % name )
    
    ax[1].scatter(xs,ys,alpha=0.5,s=5,facecolor='0.6', lw = 0)
    ax[1].plot(xr, ymean, 'k', alpha=.9,color='m',linewidth=1)
    ax[1].plot(xr, y_l, 'k', alpha=.8, color='m',linewidth=.5)
    ax[1].plot(xr, y_u, 'k', alpha=.8,color='m',linewidth=.5)
    ax[1].set_title('MR: Samples from P(y/X=x), w/ mean +- 1sd [name: %s] ' % name )
    if data: ax[1].legend()
        
    xi,yi,zi=heatplot(np.array(zip(xs,ys)),nbins=100)
    ax[2].pcolormesh(xi, yi, zi)
    ax[2].set_title('MR: GKDE P(y/X=x) [name: %s] ' % name )
    
    fig.tight_layout()
    
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


def params_compare(mr,exp_pair,xys,no_transitions,plot=False):
    '''Look at dependency between pair of expressions as data comes in'''
    name=get_name(mr)
    
    # get prior values
    out_pr = mr.snapshot(exp_list=exp_pair,plot=plot,scatter=False)
    vals_pr = out_pr['values']
    out_list = [out_pr]; vals_list=[vals_pr] 
    
    # add observes
    for i,xy in enumerate(xys):
        observe_infer([mr],[xy],no_transitions,with_index=False,withn=False) # FIXME obs n somewhere?
        out_list.append( mr.snapshot(exp_list=exp_pair,plot=plot,scatter=False) )
        vals_list.append( out_list[-1]['values'] )
        
    xys=np.array(xys); xs=[None] + list( xys[:,0] ); ys=[None] + list( xys[:,1] )

    fig,ax = plt.subplots(len(vals_list), 2, figsize=(12,len(vals_list)*3),
                          sharex='col',sharey='col')

    for i,vals in enumerate(vals_list):
        ax[i,0].scatter( vals[exp_pair[0]], vals[exp_pair[1]], c='.6',s=5,lw=0)
        ax[i,0].set_title('%s vs. %s (name=%s)' % (exp_pair[0],
                                                   exp_pair[1],name) )
        ax[i,0].set_xlabel(exp_pair[0]); ax[i,0].set_ylabel(exp_pair[1])
        if i>0:
            ax[i,1].scatter(xs[1:i], ys[1:i], c='blue',lw=0) ## FIXME start from 1 to ignore prior
            ax[i,1].scatter(xs[i], ys[i], c='red',lw=0)
            ax[i,1].set_title('Data with new point (%.2f,%.2f)'%(xs[i],ys[i]))
        
    fig.tight_layout()
    return fig,vals_list


def plot_posterior_joint(mr,no_reps=500,plot=True):
    name=get_name(mr); no_ripls=mr.no_ripls
    xy_st ='( (lambda (xval) (list xval (y_x xval)) ) (x_d) )'
    xys = if_lst_flatten( [mr.sample(xy_st) for i in range(no_reps) ] )
    
    xs= [xy[0] for xy in xys]; ys=[xy[1] for xy in xys]
    
    fig,ax = plt.subplots(1,2,figsize=(14,5),sharex=True,sharey=True)
    ax[0].scatter(xs,ys,s=5,c='m')
    ax[0].set_title('MRipl: Scatter P(x,y) (%i ripls, %i reps, name=%s)' % (no_ripls, no_reps,name) )
    xi,yi,zi=heatplot(np.array(zip(xs,ys)),nbins=100)
    ax[1].pcolormesh(xi, yi, zi)
    ax[1].set_title('MRipl: GKDE P(x,y) (%i ripls, %i reps, name=%s)' % (no_ripls,no_reps,name) )
    fig.tight_layout()
    return fig,xs,ys
    
    
def if_lst_flatten(l):
    if type(l[0])==list: return [el for subl in l for el in subl]
    return l
    
    
