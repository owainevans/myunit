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

crp_model='''
[assume alpha (uniform_continuous .01 1)]
[assume crp (make_crp alpha) ]
[assume z (mem (lambda (i) (crp) ) ) ]
[assume mu (mem (lambda (z dim) (normal 0 5) ) ) ] 
[assume sig (mem (lambda (z dim) (uniform_continuous .1 8) ) ) ]
[assume x_d (lambda (dim)
             (  (lambda (z) (normal (mu z dim) (sig z dim)))
                 (crp)  ) ) ]
[assume x (mem (lambda (i dim) (normal (mu (z i) dim) (sig (z i) dim)))  ) ]

[assume n (gamma 1 1)]
[assume model_name (quote crp)]
'''


def test_crp():
    v=mk_l(); v.execute_program(crp_model)
    v.observe('alpha','.001')
    v.observe('(mu 0 0)','0')
    v.observe('(mu 0 1)','0')
    v.observe('(sig 0 0)','.01')
    v.observe('(sig 0 1)','.01')
    v.infer(500)
    xys=[v.predict( '(list (x %i 0) (x %i 1))' %(i,i)) for i in range(30) ]
    print xys
    #assert .1 > abs( np.mean( if_lst_flatten(xys)) )
    v=mk_c(); v.execute_program(crp_model)
    for i,(x0,x1) in enumerate(xys):
        #v.observe('(list (x %i 0) (x %i 1))'%(i,i),'(list %f %f)'%(x0,x1) )
        v.observe('(x %i 0)'%i,'%f'%x0);v.observe('(x %i 1)'%i,'%f'%x1)
        v.observe('(z %i)'%i,'%i'%0)
    mus=[],sigs=[]
    for r in range(5):
        v.infer(800)
        mus.append( v.sample('(list (mu 0 0) (mu 0 1))') )
        sigs.append( v.sample('(list (sig 0 0) (sig 0 1))') )
    print mus,sigs
    

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
