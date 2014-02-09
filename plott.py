import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde

#np.random.seed(1977)


def heatplot(n2array,mus4=None,nbins=100):
    """Input is an nx2 array, plots graph and returns xi,yi,zi for colormesh""" 
    x, y = n2array.T
    
    fig, axes = plt.subplots(ncols=2, nrows=1)
    
    axes[0].set_title('Scatterplot')
    axes[0].plot(x, y, 'ko',ms=.5)
    
    #axes[0, 1].set_title('Hexbin plot')
    #axes[0, 1].hexbin(x, y, gridsize=nbins)
    #
    #axes[1, 0].set_title('2D Histogram')
    #axes[1, 0].hist2d(x, y, bins=nbins)
    
    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    k = kde.gaussian_kde(n2array.T)
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
 			
    axes[1].set_title('Gaussian KDE')
    axes[1].pcolormesh(xi, yi, zi.reshape(xi.shape))
    
    
    # Contour
    #X,Y = np.meshgrid(x,y)
    #Z = k(n2array.T)
    #axes[2].set_title('Contour Plot')
    #axes[2].contour(X,Y,n2array.T)
    if mus4:
        axes[1].scatter(mus4[:,0],mus4[:,1],s=15,marker='o')
    fig.tight_layout()
    

    plt.show()
    
    return (zi, zi.reshape(xi.shape))
