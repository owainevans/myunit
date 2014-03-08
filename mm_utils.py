import numpy as np
import matplotlib.pylab as plt
from venture.venturemagics.ip_parallel import *; 
from reg_demo_utils import *
lite=False; 
mk_l = make_lite_church_prime_ripl; mk_c = make_church_prime_ripl
normal = np.random.normal; multinomial = np.random.multinomial
uniform = np.random.uniform
dirichlet = np.random.dirichlet
multinoulli = lambda dist,size: np.argmax(multinomial(1,dist,size),axis=1)

def rep_with(text,dic):
    for i, j in dic.iteritems():
        text = text.replace(i, j)
    return text

mix_model='''



'''

crp_model='''
[assume alpha (uniform_continuous <<prior_alpha>>)]
[assume crp (make_crp alpha) ]
[assume z (mem (lambda (i) (crp) ) ) ]

[assume mu_loc <<prior_mu_loc>>]
[assume mu_scale <<prior_mu_scale>>]
[assume sigma_scale <<prior_sigma_scale>>]

[assume mu (mem (lambda (z dim) (normal mu_loc mu_scale) ) ) ] 
[assume sigma (mem (lambda (z dim) (gamma sigma_scale 1) ) )]
[assume x_d (lambda (dim)
             (  (lambda (z) (normal (mu z dim) (sigma z dim)))
                 (crp)  ) ) ]
[assume x (mem (lambda (i dim) (normal (mu (z i) dim) (sigma (z i) dim)))  ) ]

[assume n (gamma 1 1)]
[assume model_name (quote crp)]
'''
inf_vars={'<<prior_alpha>>':'.01 1',
          '<<prior_mu_loc>>':'(normal 0 20)',
          '<<prior_mu_scale>>':'(gamma 5 1)',
          '<<prior_sigma_scale>>':'(uniform_continuous .5 10)'}
zeros_vars={'<<prior_alpha>>':'.01 .05',
          '<<prior_mu_loc>>':'0',
          '<<prior_mu_scale>>':'1',
          '<<prior_sigma_scale>>':'1'}

inf_crp_model = rep_with(crp_model,inf_vars)
zeros_crp_model= rep_with(crp_model,zeros_vars)

def test_crp():
    #v=mk_c(); v.execute_program(inf_crp_model)
    #no_xs = 30
    ## FIXME: why doesn't this work?
    # v.observe('alpha','.001')
    # v.observe('(mu 0 0)','0')
    # v.observe('(mu 0 1)','0')
    # v.observe('(sig 0 0)','.01')
    # v.observe('(sig 0 1)','.01')
    # v.infer(1)
    # xys=[v.predict( '(list (x %i 0) (x %i 1))' %(i,i)) for i in range(no_xs) ]
    # print xys
    #assert .1 > abs( np.mean( if_lst_flatten(xys)) )
    no_xs = 30
    xys = np.random.normal(0,.01,size=(no_xs,2) )
    xys = np.random.normal(5,1,size=(no_xs,2) )
    print xys
    vinf=mk_c(); vinf.execute_program(inf_crp_model)
    vzeros=mk_c(); vzeros.execute_program(zeros_crp_model)
    vs=[vinf,vzeros]
    for v in vs:
        v.sample('(x 0 0)')
        print [v.predict( '(list (x %i 0) (x %i 1))'%(i,i) ) for i in range(20) ]
    for i,(x0,x1) in enumerate(xys):
        [v.observe('(x %i 0)'%i,'%f'%x0) for v in vs]
        [v.observe('(x %i 1)'%i,'%f'%x1) for v in vs]
        #v.observe('(z %i)'%i,'%i'%0)
    print [di for di in vinf.list_directives() if di['instruction']=='observe']
    [v.infer(10**4) for v in vs]    
    loops=3; mus=[[]]*loops; sigs=[[]]*loops
    for i,v in enumerate(vs):
        print 'i',i
        for r in range(loops):
            print 'in loop',i,'\n'
            v.infer(2500)
            zvals = np.unique([ v.sample('(z %i)' %ind) for ind in range(no_xs)] )
            print zvals
            mus[r]=[ (z,v.sample('(list (mu %i 0) (mu %i 1))'%(z,z)) ) for z in zvals]
            sigs[r]=[(z,v.sample('(list (sigma %i 0) (sigma %i 1))'%(z,z) )) for z in zvals]
        if i==0: print 'vinf'
        if i==1: print 'vzeros'
        print '-- mus:',mus, '\n sig:',sigs
    


def generate_data_mm(n,k=3,model='gauss'):
    dim=2
    
    if model=='gauss':
        thetas=[{'mu':normal(0,8,dim),'sig':uniform(.1,4,2)} for i in range(k)]
    else: # uniform
        thetas=[{'c':normal(0,8,dim),'w':uniform(.1,4,2)} for i in range(k)]
         
    alpha = [2]*(k-1) + [1] 
    dist = dirichlet(alpha) 
    zs=multinoulli(dist,n)
    x=np.zeros( (n,dim) )

    for i,z in enumerate(zs):
        th = thetas[z]
        if model=='gauss':
            x[i,0] = normal(th['mu'][0],th['sig'][0])
            x[i,1] = normal(th['mu'][1],th['sig'][1])
        else:
            x[i,:]=[ uniform(th['c'][j] - th['w'][j],
                             th['c'][j] + th['w'][j]) for j in range(dim) ]

    fig,ax = plt.subplots(figsize=(10,4))
    ax.scatter(x[:,0],x[:,1],c=zs,s=8,marker='+')
    ax.scatter([th['mu'][0] for th in thetas],[th['mu'][1] for th in thetas],
               s=12)
    
    params = {'x':x,'alpha':alpha,'dist':dist,
              'zs':zs,'model':model,'thetas':thetas}
    return x,params


nor = np.random.normal
n=10; 
x0 = list(nor(0,1,n/2)) + list(nor(5,1,n/2))
x1 = list(nor(0,1,n/2)) + list(nor(5,1,n/2))

#plt.scatter(x0,x1,c=([0]*(n/2) + [1]*(n/2)),s=25,marker='+')
