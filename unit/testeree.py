from analytics import *
from venture.venturemagics import *

class Gauss(VentureUnit):
    def makeAssumes(self):
        self.assume('mu','(normal 0 10)')
        self.assume('la','(gamma 1 1)')
        return

    def makeObserves(self):
        self.observe('(normal mu .1)','10')
        return

from venture.unit import VentureUnit as oldUnit
class Gauss2(oldUnit):
    def makeAssumes(self):
        self.assume('mu','(normal 0 10)')
        self.assume('la','(gamma 1 1)')
        return

    def makeObserves(self):
        self.observe('(normal mu .1)','10')
        return


model = Gauss(mk_p_ripl())

sj = model.sampleFromJoint(40)
rj = model.runFromJoint(20)
rc = model.runFromConditional(20)


model2 = Gauss2(mk_p_ripl())

sj2 = model2.sampleFromJoint(40)
rj2 = model2.runFromJoint(20)
rc2 = model2.runFromConditional(20)


v=mk_p_ripl()
v.assume('mu','(normal 0 10)')
samples=[]
for i in range(200):
    v.infer(10)
    samples.append(v.report(1))
print 'samples mean std:', np.mean(samples),np.std(samples)
    
v.clear()
v.assume('mu','(normal 0 10)')
v.predict('(normal mu .1)')
samples2=[]
for i in range(200):
    v.infer(10)
    samples2.append(v.report(1))
print 'samples2 mean: ', np.mean(samples2),np.std(samples2)
