from venture.shortcuts import *
from my_unit import *


class Tricky(VentureUnit):
    def makeAssumes(self):
        self.assume("coin", "(beta 1 20)")
        [self.assume(var,"(flip coin)") for var in ["x"+str(i) for i in range(3)] ]
        
    def makeObserves(self):
        [self.observe(var,"true") for var in ["x"+str(i) for i in range(3)] ]

class Gauss(VentureUnit):
    def makeAssumes(self):
        self.assume("location",str(self.parameters['loc']) )
        self.assume("scale",str(self.parameters['scale']) )
        self.assume("mu1","(normal location scale)")
        self.assume("mu2","(normal location scale)")


        [self.assume(var,"(normal mu1 1.)") for var in ["x1_"+str(i) for i in range(3)] ]
        [self.assume(var,"(normal mu2 1.)") for var in ["x2_"+str(i) for i in range(3)] ]
        
    def makeObserves(self):
        [self.observe(var,"10.") for var in ["x1_"+str(i) for i in range(3)] ]
        [self.observe(var,"5.") for var in ["x2_"+str(i) for i in range(3)] ]


#parameters = {'topics' : 4, 'vocab' : 10, 'documents' : 8, 'words_per_document' : 12}
#history = model.runConditionedFromPrior(50)
#history = model.runFromJoint(50)
#history = model.sampleFromJoint(50)
#history = model.computeJointKL(200, 200, verbose=True)[2]
#history.plot(fmt='png')


params = {'loc':[-30,30], 'scale':[.1,1,100] }


make_ripl = make_church_prime_ripl
v = make_ripl()



runner = lambda params :Gauss(v, params).runConditionedFromPrior(sweeps=20, runs=1, verbose=False)
histories = produceHistories(params, runner)
plotAsymptotics(params, histories, 'sweep_time', aggregate=True)

#print 'variable "histories" was created from "histories = produceHistories(params, runner)"'


