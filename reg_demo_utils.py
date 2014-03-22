import numpy as np
import matplotlib.pyplot as plt
from venture.venturemagics.ip_parallel import *; 
lite=False; 
mk_l = make_lite_church_prime_ripl; mk_c = make_church_prime_ripl

vs = test_ripls()

## models to add
# zero biased weights on coefficient
# CRP regression
# hierarch regression 
# do model comparison
# do y given x and vice versa


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
'''
simple_quadratic_model='''
[assume w0 (normal 0 3) ]
[assume w1 (normal 0 3) ]
[assume w2 (normal 0 3) ]
[assume x (mem (lambda (i) (x_d) ) )]
[assume x_d (lambda () (normal 0 5))]
[assume noise (gamma 2 1) ]
[assume f (lambda (x) (+ w0 (* w1 x) (* w2 (* x x)) ) ) ]
[assume y_x (lambda (x) (normal (f x) noise) ) ]
[assume y (mem (lambda (i) (y_x (x i)) ) )]
[assume n (gamma 1 1)]
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
'''


def pred_xy(n):
    xys=[]
    for v in vs:
        xys.append( [v.predict('(list (x %i %i) (y %i %i))'%(gp,ind,gp,ind)) for gp in range(2) for ind in range(n) ] )
    return xys

def pred_cond(n=30):
    ys=[]
    xr=np.linspace(-3,3,n)
    for v in vs:
        ys.append([[v.predict('(f %i %f)'%(gp,x)) for x in xr] for gp in range(2)])
    ys=ys[0]
    return zip(xr,ys[0]),zip(xr,ys[1])

def diff_groups():
    xy0,xy1 = pred_cond()
    xy0=np.array(xy0); xy1=np.array(xy1);
    return  np.abs(xy0 - xy1)

def test_hi1():
    diff1 = diff_groups()
    # set all the priors on ws to have very low var, so all gps same
    [v.observe('(sigma_prior %i)'%j,'.1') for v in vs for j in range(3)]
    [v.observe('noise','.05') for v in vs]
    [v.infer(500) for v in vs]
    diff2 = diff_groups()
    assert 2 > np.mean(diff2[:,1])
    assert all(diff1[:,1] > diff2[:,1])

# if two groups are the same, we should learn a small val for sig prior
def test_hi2():
    ## fix the ws = 1 and then should find small shared sigma
    vs = [mk_l() for i in range(2)];
    [v.set_seed(np.random.randint(100)) for v in vs]
    [v.execute_program(hi_quadratic_model) for v in vs]
    sigmas1=[v.predict('(sigma_prior %i)'%j) for v in vs for j in range(3)]
    for gp in gps:
        [v.observe('(w %i %i)'%(gp,i),'1') for v in vs for i in range(10)]
    [v.infer(1000) for v in vs]
    sigmas2=[v.predict('(sigma_prior %i)'%j) for v in vs for j in range(3)]
    assert sigmas1>sigmas2

def test_hi3():
    ## learn same xys for each of 3 groups, so should be small shared sigma
    vs = [mk_c() for i in range(2)];
    [v.set_seed(np.random.randint(100)) for v in vs]
    [v.execute_program(hi_quadratic_model) for v in vs]
    sigmas1=[v.predict('(sigma_prior %i)'%j) for v in vs for j in range(3)]
    N=20
    xs=np.random.normal(0,3,N); ys = xs + np.random.normal(0,.01,N);
    xys=zip(list(xs),list(ys)); xys=lst_flatten([xys] * 3)
    xyg = [ [] ] * 3
    xyg[0] = xys[:20]; xyg[1] = xys[20:40]; xyg[2]=xys[40:]
    gps = [0,1,2]
    for gp in gps:
        xys = xyg[gp]
        for i,(x,y) in enumerate(xys):
            [v.observe('(x %i %i)' % (gp,i) , '%f' % x) for v in vs]
            [v.observe('(y %i %i)' % (gp,i), '%f' % y ) for v in vs]
    [v.infer(6000) for v in vs]
    sigmas2=[v.predict('(sigma_prior %i)'%j) for v in vs for j in range(3)]
    assert sigmas2 < sigmas1
    return sigmas1,sigmas2

def test_hi4():
    vs = [mk_c() for i in range(2)];
    [v.set_seed(np.random.randint(100)) for v in vs]
    [v.execute_program(hi_quadratic_model) for v in vs]

    plot_conditional_hi(vs[0],gps=6)
    sigmas1=[v.predict('(sigma_prior %i)'%j) for v in vs for j in range(3)]
    mus1=[v.predict('(mu_prior %i)'%j) for v in vs for j in range(3)]
    xs = np.linspace(-3,3,8)
    xyg=[zip(xs,list(4*i*xs)) for i in range(6) ]
    for gp in range(6):
        xys = xyg[gp]
        for i,(x,y) in enumerate(xys):
            [v.observe('(x %i %i)' % (gp,i) , '%f' % x) for v in vs]
            [v.observe('(y %i %i)' % (gp,i), '%f' % y ) for v in vs]
    [v.infer(3000) for v in vs]
    sigmas2=[v.predict('(sigma_prior %i)'%j) for v in vs for j in range(3)]
    mus2=[v.predict('(mu_prior %i)'%j) for v in vs for j in range(3)]
    assert sigmas2[1]>sigmas1[1] and sigmas2[4]>sigmas1[4]
    assert mus2[1]+3 > mus1[1] and mus2[4]+3 > mus2[4]

    plot_conditional_hi(vs[0],gps=6)
    return sigmas1,sigmas2,mus1,mus2

vs=[mk_c() for reps in range(2)]
[v.execute_program(hi_quadratic_model) for v in vs]
v=vs[0]


def plot_conditional_hi(ripl,gps,xr=(-3,3),data=[],no_reps=15,no_xs=40,return_fig=0):
    xr = np.linspace(xr[0],xr[1],no_xs)
    f_xr_g = [ 0 ] * gps; xys_g = [ 0 ] * gps
    fig,ax = plt.subplots(gps,2,figsize=(9,4*gps))

    for gp in range(gps):
        f_xr=[ripl.predict('(f %i %f)' % (gp,x)) for x in xr]
        xys=[[(x,ripl.predict('(y_x %i %f)'%(gp,x))) for r in range(no_reps)] for x in xr]
        f_xr_g.append(f_xr); xys_g.append(xys)
        xs=[xy[0] for xy in if_lst_flatten(xys)];
        ys=[xy[1] for xy in if_lst_flatten(xys)]

        if data: ax[gp,0].scatter(data[gp][0],data[gp][1])

        ax[gp,0].plot(xr,f_xr,color='m'); #ax[0].scatter(xr,fxp,xr,fxm)
        ax[gp,0].set_title('Ripl: f (+- 1sd) ' )
        ax[gp,1].scatter(xs,ys,s=5)
        ax[gp,1].set_title('Ripl: Scatter P(y/X=x,params) ' )
    





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
'''
#[assume y_x (lambda (gp x) (normal (f gp x) noise) ) ]
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
simp_mod='''
[assume alpha (uniform_continuous .01 .5)]
[assume crp (make_crp alpha) ]
[assume z (mem (lambda (i) (crp) ) ) ]
[assume mu (mem (lambda (z) (normal 0 5) ) ) ] 
[assume sig (mem (lambda (z) (uniform_continuous .1 8) ) ) ]
[assume x_d (lambda () ( (lambda (z) (normal (mu z) (sig z) )) (crp) ) ) ]
[assume x (mem (lambda (i) (normal (mu (z i)) (sig (z i))))  ) ]

[assume w (lambda (z) 1)]

[assume f (lambda (z x) (* (w z) x) ) ]

[assume y (mem (lambda (i) (normal (f (z i) (x i)) .1) ) ) ]
'''


def test_crp(highvar=True):
    vs = [mk_l() for i in range(3)];
    [v.set_seed(np.random.randint(100)) for v in vs]
    [v.execute_program(simp_mod) for v in vs]
    
    n=2*(15)
    #[plot_conditional_crp(v,N=1000) for v in vs]
    
    xs0 = np.random.normal(-2,1,int(.5*n)); xs1 = np.random.normal(2,1,int(.5*n))
    ys0 = (xs0+2)**2; ys1 = - ((xs0-2)**2)
    xys = zip(list(xs0)+list(xs1), list(ys0)+list(ys1) )
    
    if not highvar: xys = zip(np.random.normal(-2,1,n),(xs0+2)**2 )
                           
    for i,(x,y) in enumerate(xys):  
        [v.observe('(x %i)' %i, '%f' % x) for v in vs]
        [v.observe('(y %i)' %i, '%f' % y) for v in vs]
    
    [v.infer(100) for v in vs]
    [plot_conditional_crp(v,data=np.array(xys),N=n+1) for v in vs]
    
    return vs


def plot_conditional_crp(ripl,xr=(-4,4),data=[],N=1000,no_reps=10,no_xs=12,return_fig=0):
    ## N is where we draws x,ys from to get samples; should be bigger than observed points
    
    xr = np.linspace(xr[0],xr[1],no_xs)
    xys=[]

    from collections import Counter
    gps=Counter([ripl.sample('(gp %i)'%i) for i in range(N-1)])
    gps = gps.keys()[:3]
    
    fig,ax = plt.subplots(len(gps),1,figsize=(9,4*len(gps)))
    xyg=[ ] # len(gps)
    for count,gp in enumerate(gps):
        xys=[]
        for x in xr:
            xys.append( (x, ripl.predict('(f %i %f) % (gp,x)'))  )
                        
        if not(data==[]): ax[count,0].scatter(data[:,0],data[:,1],c='m')

        xs=[xy[0] for xy in if_lst_flatten(xys)]
        ys=[xy[1] for xy in if_lst_flatten(xys)]
        ax[count,0].scatter(xs,ys,c='blue',s=4)

        xyg.append(xys)
    
                        
    return xyg







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
[assume w1 (mem (lambda (p)(normal 0 3))) ]
[assume w2 (mem (lambda (p)(normal 0 1))) ]
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
[assume w1 (normal 0 3) ]
[assume w2 (normal 0 1) ]
[assume omega (normal 0 3) ]
[assume theta (normal 0 3) ]

[assume noise (gamma 2 1) ]

[assume model (if (flip) 1 0) ]
[assume quadratic (lambda (x) (+ w0 (* w1 x) (* w2 (* x x) ) ) ) ]
[assume fourier (lambda (x) (+ w0 (* w1 (sin (+ (* omega x) theta) ) ) ) ) ]
[assume f (if (= model 0) quadratic fourier) ]

[assume y_x (lambda (x) (normal (f x) noise) ) ]
[assume y (mem (lambda (i) (y_x (x i))  ))] 
[assume n (gamma 1 100)]
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

def heatplot(n2array,nbins=100):
    """Input is an nx2 array, plots graph and returns xi,yi,zi for colormesh""" 
    x, y = n2array.T
    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    k = kde.gaussian_kde(n2array.T)
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    #X,Y = np.meshgrid(x,y)     # Contour
    #Z = k(n2array.T)
    #axes[2].set_title('Contour Plot')
    #axes[2].contour(X,Y,n2array.T)
    
    # plot ax.pcolormesh(xi, yi, zi.reshape(xi.shape))
    return (xi, yi, zi.reshape(xi.shape))


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


def observe_infer(vs,xys,no_transitions,with_index=False,withn=False):
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
    name=get_name(mr) 
    print '%s logscore: (mean, max) ' % name, np.mean(logscore), np.max(logscore)
    return np.mean(logscore), np.max(logscore)


def get_name(r_mr):
    mr=1 if isinstance(r_mr,MRipl) else 0
    di_l = r_mr.list_directives()[0] if mr else r_mr.list_directives()
    if 'model_name' in str(di_l):
        try:
            n = r_mr.sample('model_name')[0] if mr else r_mr.sample('model_name')
            return n
        except: pass
    else:
        return 'anon model'


def plot_conditional(ripl,xr=(-3,3),data=[],no_reps=15, no_xs=80, hexbin=False, return_fig=False):
    
    try: n = int( np.round( ripl.sample('n') ) )  #FIXME
    except: n=0
    if n>0:
        d_xs = [ripl.sample('(x %i)' % i) for i in range(n)]
        d_ys = [ripl.sample('(y %i)' % i) for i in range(n)]
        xr = ( min(d_xs)-1,max(d_ys)+1 )
    
    xr = np.linspace(xr[0],xr[1],no_xs)
    
 
    f_xr = [ripl.predict('(f %f)' % x) for x in xr]
    name='mod'
    
    xys = [[(x,ripl.predict('(y_x %f)' % x)) for r in range(no_reps)] for x in xr]
    x_n = [ abs(np.std( [xy[1] for xy in xys[i] ]))  for i,x in enumerate(xr) ]
    xs=[xy[0] for xy in flatten(xys)]; ys=[xy[1] for xy in flatten(xys)]
    #fxp=[x+noi for x,noi in zip(xs,x_n)]; fxm=[x-noi for x,noi in zip(xs,x_n)]
    
    fig,ax = plt.subplots(1,3,figsize=(17,5),sharex=True,sharey=True)
    
    # plot data and f with noise
    if data: ax[0].scatter(data[0],data[1])
    if n>0: ax[0].scatter(d_xs,d_ys)
    ax[0].set_color_cycle(['m', 'gray','gray'])
    ax[0].plot(xr,f_xr,color='m'); #ax[0].scatter(xr,fxp,xr,fxm)
    ax[0].set_title('Ripl: f (+- 1sd) ' )
    
    ax[1].scatter(xs,ys,s=5); ax[1].set_title('Ripl: Scatter P(y/X=x,params) ' )
    
        
    xi,yi,zi=heatplot(np.array(zip(xs,ys)),nbins=100)
    ax[2].pcolormesh(xi, yi, zi)
    ax[2].set_title('Ripl: GKDE P(y/X=x,params) ' )
    
    if hexbin: 
        fig,ax=subplots(figsize=(8,5))
        ax.hexbin(xs, ys, gridsize=40, cmap="BuGn", extent=(min(xs),max(xs), min(ys),max(ys)) )   
        ax.set_title('Ripl: Hexbin P(y/X=x,params) ' )
    
    return xs,ys

def plot_cond(ripl,no_reps=20,return_fig=False,set_xr=None,plot=True):
    '''Plot f(x) with 1sd noise curves. Plot y_x with #(no_reps)
    y values for each x. Use xrange with limits based on posterior on P(x).'''
    
    if set_xr!=None:
        xr=set_xr; n=0
    else: # find x-range from min/max of observed points
        try: n = int( np.round( ripl.sample('n') ) )  #FIXME
        except: n=0
        if n==0:
            xr= np.linspace(-3,3,50);
        else:
            d_xs = [ripl.sample('(x %i)' % i) for i in range(n)]
            d_ys = [ripl.sample('(y %i)' % i) for i in range(n)]
            xr = np.linspace(1.5*min(d_xs),1.5*max(d_xs),30)
    
    f_xr = [ripl.sample('(f %f)' % x) for x in xr]
    
    # gaussian noise 1sd
    h_noise = ['pivot','piecewise']
    name=get_name(ripl)
    noise=ripl.sample('(noise 0)') if name in h_noise else ripl.sample('noise')
    f_a = [fx+noise for fx in f_xr]
    f_b = [fx-noise for fx in f_xr]

    # scatter for y conditional on x
    if plot:
        xys1 = [[(x,ripl.sample('(y_x %f)' % x)) for r in range(no_reps)] for x in xr]
        xys = if_lst_flatten(xys1)
        
        xs=[xy[0] for xy in xys]; ys=[xy[1] for xy in xys]
        
        #y_x = [  [ripl.sample('(y_x %f)' % x) for r in range(no_reps)] for x in xr]
        fig,ax = plt.subplots(1,3,figsize=(14,4),sharex=True,sharey=True)
        if n!=0: ax[0].scatter(d_xs,d_ys)
        ax[0].set_color_cycle(['m', 'gray','gray'])
        ax[0].plot(xr,f_xr,xr,f_a,xr,f_b)
        ax[0].set_title('Ripl: f (+- 1sd) (name= %s )' % name)
        ax[1].scatter(xs,ys,s=5,c='gray')
        #[ ax[1].scatter(xr,[y[i] for y in y_x],s=5,c='gray') for i in range(no_reps) ]
        ax[1].set_title('Ripl: Scatter P(y/X=x,params) (name= %s)' % name)
        
        xi,yi,zi=heatplot(np.array(zip(xs,ys)),nbins=100)
        ax[2].pcolormesh(xi, yi, zi)
        ax[2].set_title('Ripl: GKDE P(y/X=x,params) (name= %s)' % name )
        fig.tight_layout()
        if return_fig:
            return fig,xs,ys
        else:
            return xs,ys
    return xs,ys


def plot_joint(ripl,no_reps=500,return_fig=False):
    '''Sample from joint P(x,y), holding other params fixed '''
    name=get_name(ripl)
    
    xs = [ ripl.sample('(x_d)') for i in range(no_reps) ]
    ys = [ ripl.sample('(y_x %f)' % x) for x in xs]
    
    fig,ax = plt.subplots(1,2,figsize=(12,4),sharex=True,sharey=True)
    ax[0].scatter(xs,ys,s=5,c='gray')
    ax[0].set_title('Single ripl: %i samples from P(x,y / params) (name= %s)' % no_reps, name)

    xi,yi,zi=heatplot(np.array(zip(xs,ys)),nbins=100)
    ax[1].pcolormesh(xi, yi, zi)
    ax[1].set_title('Single ripl: GKDE of %i samples from P(x,y / params) (name= %s)' % (no_reps,name) )
    fig.tight_layout()
    
    if return_fig:
        return fig,xs,ys
    else:
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

    fig,ax = plt.subplots(len(vals_list), 2, figsize=(12,len(vals_list)*4))

    for i,vals in enumerate(vals_list):
        ax[i,0].scatter( vals[exp_pair[0]], vals[exp_pair[1]], s=6)
        ax[i,0].set_title('%s vs. %s (name=%s)' % (exp_pair[0],
                                                   exp_pair[1],name) )
        ax[i,0].set_xlabel(exp_pair[0]); ax[i,0].set_ylabel(exp_pair[1])
        if i>0:
            ax[i,1].scatter(xs[1:i], ys[1:i], c='blue') ## FIXME start from 1 to ignore prior
            ax[i,1].scatter(xs[i], ys[i], c='red')
            ax[i,1].set_title('Data with new point (%f,%f)'%(xs[i],ys[i]))
        
    fig.tight_layout()
    return fig,vals_list


def plot_posterior_conditional(mr,no_reps=20,set_xr=None,plot=True):
    name=get_name(mr)
    no_ripls = mr.no_ripls
    if set_xr!=None: ##FIXME should match single ripl version
        xr = set_xr
    else:
        # find x-range from min/max of observed points
        # only look at output of first ripl
        n = int( np.round( mr.sample('n')[0] ) )  #FIXME
        d_xs = [mr.sample('(x %i)' % i)[0] for i in range(n)]
        d_ys = [mr.sample('(y %i)' % i)[0] for i in range(n)]
        xr = np.linspace(1.5*min(d_xs),1.5*max(d_xs),20)

    if plot:
        #y_x = [if_lst_flatten([mr.sample('(y_x %f)' % x) for r in range(no_reps)] ) for x in xr]
        xys=[]
        for i in range(no_reps):
            xys.extend(if_lst_flatten([zip([x]*no_ripls,
                                           mr.sample('(y_x %f)' % x) ) for x in xr]))
        xs,ys=[xy[0] for xy in xys],[xy[1] for xy in xys]
        #assert len(xys)==no_reps*no_ripls*len(xr)
        #assert len(xys) == len(if_lst_flatten(y_x))
        
        fig,ax = plt.subplots(1,2,figsize=(14,5),sharex=True,sharey=True)
        if set_xr==None: ax[0].scatter(d_xs,d_ys,c='m')
        #[ ax[0].scatter(xr,[y[i] for y in y_x],s=6,c='gray') for i in range(no_reps) ]
        ax[0].scatter(xs,ys,s=6,c='gray')
        ax[0].set_title('Mripl: Scatter P(y/X=x) for uniform x-range (name= %s)' %name)
        xi,yi,zi=heatplot(np.array(zip(xs,ys)),nbins=100)
        ax[1].pcolormesh(xi, yi, zi)
        ax[1].set_title('Mripl: GKDE P(y/X=x) for uniform x-range (name= %s)' %name)
        fig.tight_layout()
        return fig,xs,ys
    return xs,ys


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
    
    



   


### PLAN: different plots/scores
#1. av logscore and best logscore.
# 2. plot the curve, adding noise error (easiest way is with y_x)
# 3. plot the joint (sample both x's and y's)
# 4. plot posterior on sets of params
# 5. plot posterior conditional
# 6. plot posterior joint (the posterior join density over x,y: get from running chain long time or combining chains)
# 7. plot p(x / y) for some particular y's 

