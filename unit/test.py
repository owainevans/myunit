from analytics import Analytics
from venture.venturemagics.ip_parallel import *
v=mk_p_ripl()
v.assume('x','(beta 1 1)')
v.observe('(flip x)','true')
ana = Analytics(v)
h=ana.sampleFromJoint(10)
