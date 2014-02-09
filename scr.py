from venture.shortcuts import *
import numpy as np
import matplotlib.pylab as plt

mkripl = make_church_prime_ripl

no_ripls = 4
no_draws = 100
ripls = [0] * no_ripls
data=np.zeros( (no_ripls,no_draws) )

for i in range(no_ripls):
    ripls[i] = make_church_prime_ripl()

    ripls[i].assume('bin','(lambda (bool) (if bool 1 0))')
    ripls[i].assume('nd','(lambda (a b) (= 1 (* (bin a) (bin b) )) )')
    ripls[i].assume('p1','(beta 1 1)');    ripls[i].assume('p2','(beta 1 1)')

    ripls[i].assume('get_x','(lambda () (if (flip) (flip p1) (flip p2) ))')

    for j in range(no_draws):
        data[i,j] = ripls[i].predict('(get_x)')

    for k in range(5):
        ripls[i].assume('x'+str(k),'(get_x)')

    ripls[i].assume('x6','(nd (flip p1) (flip p1))')


fig, ax = plt.subplots(ncols=no_ripls, nrows=2, sharex=True, sharey=True)

ax[0,0].hist( data[0,:] )
plt.show()

print np.random.randint(100)

    # axes[0].set_title('Scatterplot')
    # axes[0].plot(x, y, 'ko',ms=.5)
    
    # #axes[0, 1].set_title('Hexbin plot')
    # #axes[0, 1].hexbin(x, y, gridsize=nbins)
    # #
    # #axes[1, 0].set_title('2D Histogram')
    # #axes[1, 0].hist2d(x, y, bins=nbins)
    
    # # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    # k = kde.gaussian_kde(n2array.T)
