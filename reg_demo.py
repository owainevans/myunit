from venture.venturemagics.ip_parallel import *
from venture.shortcuts import make_lite_church_prime_ripl as mk_ripl
from venture.shortcuts import make_church_prime_ripl as c_mk_ripl
import numpy as np

model='''
[assume w0 (normal 0 1) ]
[assume w1 (normal 0 1) ]
[assume x (mem (lambda (i) (normal 0 1) ) )]
[assume noise (gamma 1 5) ]

[assume model (uniform_discrete 0 1) ]
[assume linear (lambda (x) (+ w0 (* w1 x) ) ) ]
[assume fourier (lambda (x) (+ w0 (* w1 (sin (+ (* omega x) theta) ) ) ) ) ]
[assume f (if (= model 0) linear fourier) ]

[assume y (mem (lambda (i) (normal (f (x i) ) noise) ) )]


[assume noise_f (mem (lambda (i) (gamma 1 5) ) ) ]
[assume noise (gamma 1 5) ]

[assume w0 (normal 0 1) ]
[assume w1 (normal 0 1) ]
[assume x (mem (lambda (i) (normal 0 1) ) )]

[assume y (mem (lambda (i) (normal (+ w0 (* w1 (x i) ) ) (noise_f i) )  ) ) ]
'''

mv=MRipl(2,lite=True)
vl = mk_ripl()
vc = mk_ripl()
vs = [mv,vl,vc]
[v.execute_program(model) for v in vs]



data=[ (0.1,0.), (2.,2.3), (-3.,-3.1) ] 

data_out = data[:] + [(4.,7)]


prior_w = zip([v.predict('w0') for v in vs],
              [v.predict('w1') for v in vs] )

for i,(x,y) in enumerate(data_out):
    [v.observe('(x %i)' % i , '%s' % str(x) ) for v in vs]
    [v.observe('(y %i)' % i , '%s' % str(y) ) for v in vs]
    
pr_noise = [v.predict('(noise %i)' % i) for v in vs for i in range(len(data_out)) ]
[v.infer(60) for v in vs]
post = zip([v.sample('w0') for v in vs],
           [v.sample('w1') for v in vs] )

post_noise = [v.predict('(noise %i)' % i) for v in vs for i in range(len(data_out)) ]
    
print 'prior:',prior_w
print 'posterior:',post
    
print 'noise'
print 'prior:',pr_noise
print 'prior:',post_noise

