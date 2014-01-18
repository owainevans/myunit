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
        self.assume("location","(normal 0 50)")
        self.assume("mu1","(normal location 5)")
        self.assume("mu2","(normal location 5)")


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


params = {}

make_ripl = make_church_prime_ripl

#tricky_instance = Tricky(vmodel,params)
print 'variable "tricky_instance" was created via "Tricky(vmodel,params)" '


#run_prior = tricky_instance.runConditionedFromPrior(50)


#v = make_ripl()
runner = lambda params : Tricky(v, params).runConditionedFromPrior(sweeps=20, runs=30, verbose=False)
#histories = produceHistories(params, runner)
#plotAsymptotics(params, histories, 'sweep_time', aggregate=True)

#print 'variable "histories" was created from "histories = produceHistories(params, runner)"'


