import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import stats
from matplotlib.ticker import NullFormatter, NullLocator, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker


def get_label(weights):
        return '[%2.2f, %2.2f]' %(weights[0],weights[1])

def plot_data(x, t,label='Train-Data'):
    plt.scatter(x, t, marker='o', c='y', s=20,label=label + f' (N = {len(x):d})')


def plot_truth(x, y, label='Truth'):
    plt.plot(x, y, 'g--', label=label)


def plot_predictive(x, y, std, y_label='Prediction', std_label='1σ Uncertainty', plot_xy_labels=True, plot_uncertainty=True):
    y = y.ravel()
    uncertainty =np.sqrt(std.ravel())

    plt.plot(x, y, label=y_label)
    if plot_uncertainty:
        plt.fill_between(x.ravel(), y + uncertainty, y - uncertainty, alpha = 0.3, label=std_label)

    if plot_xy_labels:
        plt.xlabel('x')
        plt.ylabel('y')


def plot_posterior_samples(x, ys, plot_xy_labels=True):
    plt.plot(x, ys[:, 0], 'r-', alpha=0.5, label='Post. samples over w')
    for i in range(1, ys.shape[1]):
        plt.plot(x, ys[:, i], 'r-', alpha=0.5)

    if plot_xy_labels:
        plt.xlabel('x')
        plt.ylabel('y')

def plot_posterior_2D(mean, cov, w0, w1):
    #Parameters to set

    #Create grid and multivariate normal
    resolution = 100
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    X, Y = np.meshgrid(x,y)


    grid_flat = np.dstack(np.meshgrid(x, y)).reshape(-1, 2)
    Z = stats.multivariate_normal.pdf(grid_flat, mean=mean.ravel(), cov=cov).reshape(resolution, resolution)


    #Make a 3D plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z,cmap='viridis',linewidth=0)
    cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)
    ax.scatter(w0, w1, marker='x', c='r', s=20, label='Truth: '+get_label([w0,w1]))
    ax.scatter(mean[0],mean[1], marker='o', c='g', s=20, label='Posterior: '+get_label(mean))
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_zlim(-0.15,0.2)
    ax.set_zticks(np.linspace(0,0.2,5))
    ax.view_init(27, -21)

def plot_posterior_sns(mean,cov,w0,w1):

    data = np.random.multivariate_normal(mean.ravel(), cov, 200)
    df = pd.DataFrame(data, columns=["x", "y"])
    g = sns.jointplot(x="x", y="y", data=df, kind="kde", color="m")
    g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
    #g.plot_joint(plt.scatter, x=w0, y=w1, marker='x', c='r', s=20, label='Truth: '+get_label([w0,w1]))
    #g.plot_joint(plt.scatter,x=mean[0],y=mean[1], marker='o', c='g', s=20, label='Posterior: '+get_label(mean))
    g.ax_joint.collections[0].set_alpha(0)
    g.set_axis_labels("$X$", "$Y$")


# Gaussian Bell function (used for plot_posterior_joint_distribution)
def gaussian_bell(x,mu,variance):
    return (1.0/np.sqrt(2.0*np.pi*variance))*np.exp(-(x - mu)**2/(2.0*variance))


def plot_posterior_joint_distribution(mean, cov, w0, w1,xlim,ylim,ax_Pxy=None):
    """
    Draw a plot of two with histogramm and contour.
    
    Args:
        mean: Mean value (1x2)
        cov: Covariance (2x2).
        w0: Initial weight w_0.
        w1: Initial weight w_1.
        xlim: Limits of the x-axis [-1,1]
        ylim: Limits of the y-axis [-1,1]
        ax_Pxy: Subplot axes
    """


# https://matplotlib.org/2.0.1/mpl_toolkits/axes_grid/users/overview.html

#     ax_Pxy is the middle axes!!
    

    # Get current axix gca() if ax_Pxy is not set
    if ax_Pxy is None:
        ax_Pxy = plt.gca()
        
    '''1. Define the axes and set the labels and limits respectively''' 
    # Create new axes on the right and on the top of the current axes
    # The first argument of the new_vertical(new_horizontal) method is
    # the height (width) of the axes to be created in inches.
    # https://matplotlib.org/2.0.2/mpl_toolkits/axes_grid/api/axes_divider_api.html
    
    #ax_Pxy.figure.set_size_inches(25, 15)
    divider = make_axes_locatable(ax_Pxy)
    ax_Px = divider.append_axes("bottom", 1.0, pad=0.1)#, sharex=ax_Pxy)
    ax_Py = divider.append_axes("right", 1.0, pad=0.1)#, sharey=ax_Pxy)
    #ax_cb = divider.append_axes("right", size="7%", pad="1%")
    
    # set axis label formatters
    ax_Pxy.xaxis.set_major_formatter(NullFormatter())
    ax_Pxy.yaxis.set_major_formatter(NullFormatter())
    ax_Px.yaxis.set_major_formatter(NullFormatter())
    ax_Py.xaxis.set_major_formatter(NullFormatter())

    
    # define axis limits
    ax_Pxy.set_xlim(xlim[0], xlim[1])
    ax_Pxy.set_ylim(ylim[0], ylim[1])
    ax_Px.set_xlim(xlim[0], xlim[1])
    ax_Py.set_ylim(ylim[0], ylim[1])

    
    # label axes
    #ax_Pxy.set_xlabel('$x$')
    #ax_Pxy.set_ylabel('$y$')
    
    ax_Px.set_xlabel('$w0$')
    ax_Px.set_ylabel('$p(w1)$')
    #ax_Px.xaxis.set_label_position('bottom')
    ax_Px.yaxis.set_label_position('right')
    #ax_Px.set_xticks(np.linspace(-1, 1, bins.any()))
    ax_Px.set_xticks(np.arange(xlim[0], xlim[1], 0.2))
    #ax_Px.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))  

    ax_Py.set_ylabel('$w1$')
    ax_Py.set_xlabel('$p(w0)$')
    #ax_Py.xaxis.set_label_position('top')
    ax_Py.yaxis.set_label_position('right')
    #ax_Py.set_yticks(np.linspace(-1, 1, bins.any()))
    ax_Py.set_yticks(np.arange(ylim[0], ylim[1], 0.2))
    ax_Py.yaxis.set_ticks_position('right')
   

    '''2. Create multivariate normal random variable''' 
    resolution = 100
    #x,y= np.random.multivariate_normal(mean.ravel(), cov,resolution*resolution).T  
    grid_x = np.linspace(xlim[0],xlim[1], resolution)
    grid_y = np.linspace(ylim[0],ylim[1], resolution)
    X,Y = np.meshgrid(grid_x, grid_y)
    pos = np.dstack((X, Y))
    
    # Draw Probability density function.
    #rv = stats.multivariate_normal(mean=[w0,w1], cov=cov)
    rv = stats.multivariate_normal(mean=mean.ravel(), cov=cov)
    Z = rv.pdf(pos)
    
    # Draw random samples from a multivariate normal distribution.
    rvs = rv.rvs(size=resolution*resolution)
    x = rvs[:,0]
    y = rvs[:,1]
    
    '''3. Create the histogram, contour and density for side axes''' 
    
    '''3.1. Calculate the bins for the histogram''' 
    # determine nice limits and try to calculate good binwith
    # https://sciencing.com/determine-bin-width-histogram-8485512.html
    binwidth = 0.5*(((1/np.cbrt(len(x)))*np.std(x) + 
                     (1/np.cbrt(len(y)))*np.std(y))*3.49)
    xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(y))])
    lim = (int(xymax/binwidth) + 1)*binwidth
    bins = np.arange(-lim, lim + binwidth, binwidth)
    
#     H, xbins, ybins,im = ax_Pxy.hist2d(x,y,bins=bins,cmap=plt.cm.binary)#,norm=LogNorm())
#     H, xbins, ybins = np.histogram2d(x,y,bins=bins,normed=True)
    
    # some helper variables
    mu_x = mean[0]
    mu_y = mean[1]
    sigma_x = cov[0,0]
    sigma_y = cov[1,1]
    color='#AAAAFF'
    alpha=0.5
    
    
    '''3.2. Draw the density and histogramm for ax_Px'''
    # plot histogram
    _n, _bins, _patches = ax_Px.hist(x, bins=bins,density=True,color=color,alpha=alpha)
    # get pdf
    _y = gaussian_bell(_bins,mu_x,sigma_x)
    # plot density
    ax_Px.plot(_bins, _y, '-',color='b')
    
    '''3.3. Draw the density and histogramm for ax_Py'''
    # plot histogram
    _n, _bins, _patches = ax_Py.hist(y, bins=bins, density=True, color=color,alpha=alpha,orientation='horizontal')
    # get pdf
    _y = gaussian_bell(_bins,mu_y,sigma_y)
    # plot density
    ax_Py.plot(_y,_bins, 'r-',color='b')
    
       
    '''3.4. Create contour for middle axes and plot truth and mean'''
    # some helper variables
    color = cmap=plt.cm.binary   
    # set the Limits
    extent=(xlim[0],xlim[1], ylim[0],ylim[1])
    
#     ax_Pxy.contourf(xbins[1:],
#                     ybins[1:],
#                     H.T,
#                     origin='lower',extend='both',
#                     cmap=color,
#                     extent=extent)
    #cp = ax_Pxy.contour(X,Y,Z,extent=extent,extend='both')
    im = ax_Pxy.imshow(Z, origin='lower', extent=extent)
    
    
    # Scatter of truth and mean weights
    ax_Pxy.scatter(w0, w1, marker='x', c='r', s=20, label='Truth: '+get_label([w0,w1]))
    ax_Pxy.scatter(mean[0],mean[1], marker='o', c='g', s=20, label='Posterior: '+get_label(mean))
    
    
    
    '''4. Draw the horizontal and vertical lines for all axes''' 
    #ax_Px.plot(xbins[1:], H.sum(1), '-k', drawstyle='steps')
    ax_Pxy.axvline(x=w0, ls='-', c='r', lw=1)
    ax_Pxy.axhline(y=w1, ls='-', c='r', lw=1)
    ax_Pxy.axvline(x=mean[0], ls='-', c='g', lw=1)
    ax_Pxy.axhline(y=mean[1], ls='-', c='g', lw=1)

    ax_Px.axvline(x=w0, ls='-', c='r', lw=1)
    ax_Py.axhline(y=w1, ls='-', c='r', lw=1)
    ax_Px.axvline(x=mean[0], ls='-', c='g', lw=1)
    ax_Py.axhline(y=mean[1], ls='-', c='g', lw=1)
    

    # set the background color white
    ax_Pxy.set_facecolor('w') 
   




def plot_posterior(mean, cov, w0, w1):
    resolution = 100

    grid_x = grid_y = np.linspace(-1, 1, resolution)
    grid_flat = np.dstack(np.meshgrid(grid_x, grid_y)).reshape(-1, 2)

    densities = stats.multivariate_normal.pdf(grid_flat, mean=mean.ravel(), cov=cov).reshape(resolution, resolution)
    plt.contourf(grid_x,grid_y,densities)
    plt.imshow(densities, origin='lower', extent=(-1, 1, -1, 1))
    #plt.plot(w0,w1,'k*',markersize=10,label='Truth: '+label([w0,w1]))
    #plt.plot(mean[0],mean[1],'g*',markersize=10,label='Posterior: '+label(mean))

    plt.scatter(w0, w1, marker='x', c='r', s=20, label='Truth: '+get_label([w0,w1]))
    plt.scatter(mean[0],mean[1], marker='o', c='g', s=20, label='Posterior: '+get_label(mean))
    plt.xlabel('w0')
    plt.ylabel('w1')



def plot_precision(epoch,precision,label):
    plt.plot(range(0, epoch), precision, 'r--', label=label+f':{np.max(precision):.5f}')
    #plt.axvline(x=np.argmax(precision), ls='--', c='k', lw=1,label='epoch: %d'%epoch)
    plt.xlabel('Epochs')
    plt.ylabel('Precision')

def plot_ratio(epoch,ratio):
    plt.plot(range(0, epoch), ratio, 'r--', label=f'ratio min: {np.min(ratio):.5f} max: {np.max(ratio):.5f}')
    plt.xlabel('Epochs')
    plt.ylabel('Ratio (alpha/beta)')
    plt.ylim(0)


