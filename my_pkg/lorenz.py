'Пакет для построения графика аттрактора Лоренца'

def plotLorenzAttractor(xx, yy, zz,*,sigma = 10, rho = 28, beta = 2.667):
    
    '''Построение графика решения системы Лоренца'''
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(xx, yy, zz, lw=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Lorenz Attractor")
    ax.text(0,0,0, r'$\sigma={0}, \varrho={1}, \beta={2}$'.format(sigma,rho,beta),
            horizontalalignment='center',
            verticalalignment='center',
            transform = ax.transAxes)
    plt.show()
def lorenzAttractor1(sigma = 10, rho = 28, beta = 2.667, initialdata = (0.,1.,1.01), dt = 0.01, npoint = 10**4):
    
    '''Функция строит аттрактор Лоренца'''
    
    xx = [0]*(npoint); yy = [0]*(npoint); zz = [0]*(npoint)
    xx[0], yy[0], zz[0] = initialdata

    for i in range(npoint-1):
        dx = sigma*(yy[i] - xx[i])
        dy = xx[i]*(rho-zz[i])-yy[i]
        dz = xx[i]*yy[i] - beta*zz[i]
        xx[i + 1] = (dx * dt)+xx[i]
        yy[i + 1] = (dy * dt)+yy[i]
        zz[i + 1] = (dz * dt)+zz[i]

    plotLorenzAttractor(xx, yy, zz)
def lorenzAttractor2(sigma = 10, rho = 28, beta = 2.667, initialdata = (0.,1.,1.01), dt = 0.01, npoint = 10**4):
    
    '''Построение графика решения системы Лоренца c использованием numpy'''
    
    import numpy as np
    
    xx = [0]*(npoint); yy = [0]*(npoint); zz = [0]*(npoint)
    xx[0], yy[0], zz[0] = initialdata
    
    def lorenz(u,*,sigma = 10, beta = 2.667, rho = 28):
        x,y,z=u
        return [sigma*(y - x), x*(rho-z)-y, x*y - beta*z]

    for i in range(npoint-1):   
        dx,dy,dz = lorenz((xx[i],yy[i],zz[i]))
        xx[i + 1] = dx*dt+xx[i]
        yy[i + 1] = dy*dt+yy[i]
        zz[i + 1] = dz*dt+zz[i]
        
    plotLorenzAttractor(xx, yy, zz)
    
def lorenzAttractor3(sigma = 10, rho = 28, beta = 2.667, initialdata = (0.,1.,1.01), npoint = 10**4):
    '''Построение графика решения системы Лоренца c использованием numpy'''

    import numpy as np
    from scipy.integrate import odeint
    
    def lorenz(u,t,*,sigma = 10, beta = 2.667, rho = 28):
        x,y,z=u
        return [sigma*(y - x), x*(rho-z)-y, x*y - beta*z]
    
    t = np.linspace(0, 100, npoint)
    w = odeint(lorenz, initialdata, t)

    plotLorenzAttractor(w[:,0], w[:,1], w[:,2])
    