from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import geopy.distance

mpl.style.use('tableau-colorblind10')
mpl.rcParams['image.cmap'] = 'tab20'
mpl.rcParams['grid.color'] = 'k'
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['grid.linewidth'] = 0.5
# mpl.rcParams['axes.autolimit_mode'] = 'round_numbers'
# mpl.rcParams['axes.ypadding'] = 0
mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.ymargin'] = 0
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6

class GraphicBuilder:
    def __init__(self, inputDataFrame):
        self.prefijo = "PRED_"
        self.inputDataFrame = inputDataFrame

    def error_distance (self, columns = ["LATITUDE", "LONGITUDE", "FLOOR"]):
        altura = 2.5
        real_point = (self.inputDataFrame[columns[0]], self.inputDataFrame[columns[1]])
        pred_point= (self.inputDataFrame[self.prefijo + columns[0]], self.inputDataFrame[self.prefijo + columns[1]])
        #https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude
        #https://en.wikipedia.org/wiki/Vincenty%27s_formulae
        var_xy = geopy.distance.vincenty(real_point, pred_point).m
        var_z = np.abs(self.inputDataFrame[columns[2]] - self.inputDataFrame[self.prefijo + columns[2]]) * altura
        self.inputDataFrame['ERR_REG'] = var_xy
        self.inputDataFrame['ERR'] = np.sqrt(var_xy*var_xy + var_z*var_z)

    def graphicMap2D(self, filename="test", x = "LATITUDE", y="LONGITUDE", hue="BUILDINGID"):
        sns.lmplot(x=x, y=y, hue=hue, data=self.inputDataFrame, fit_reg=False, x_jitter=.1, markers='.')
        plt.show()
        plt.savefig("images/"+str(filename)+".svg")

    def graphicBuildings(self, columns = ["LATITUDE", "LONGITUDE", "FLOOR"], filename="test"):
        fig = plt.figure()
        fig.tight_layout()
        for index in range(3):
            ax = fig.add_subplot(1, 3, index+1, projection='3d')
            fig.tight_layout()
            dfGraphics = self.inputDataFrame[self.inputDataFrame.BUILDINGID==index]
            x = dfGraphics[columns[0]]
            y = dfGraphics[columns[1]]
            z = dfGraphics[columns[2]]
            N = 6
            xmin, xmax = min(x), max(x)
            ymin, ymax = min(y), max(y)
            c = 4*dfGraphics.BUILDINGID + dfGraphics.FLOOR # colorSet
            ax.set_xlabel('Latitude')
            ax.set_ylabel('Longitude')
            ax.set_zlabel('Floor')
            ax.set_title('Building '+str(index))
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_zlim(0, 4)
            ax.set_yticks(np.round(np.linspace(ymin, ymax, N), 3))
            ax.set_xticks(np.round(np.linspace(xmin, xmax, N), 3))
            ax.set_zticks(np.int32(np.linspace(0, 4, 5)))
            ax.set_xscale('linear')
            ax.set_yscale('linear')
            ax.view_init(10, 60)
            ax.scatter(x, y, z, c=c)
        plt.tight_layout()
        plt.show()
        plt.savefig("images/"+str(filename)+".svg")
        
    def graphicMap3D(self, columns = ["LATITUDE", "LONGITUDE", "FLOOR"], filename="test"):
        fig = plt.figure(figsize=(10,10))
        fig.tight_layout()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        fig.tight_layout()
        dfGraphics = self.inputDataFrame
        x = dfGraphics[columns[0]]
        y = dfGraphics[columns[1]]
        z = dfGraphics[columns[2]]
        N = 6
        xmin, xmax = min(x), max(x)
        ymin, ymax = min(y), max(y)
        c = 4 * dfGraphics.BUILDINGID + dfGraphics.FLOOR
        ax.set_xlabel('Latitude')
        ax.set_ylabel('Longitude')
        ax.set_zlabel('Floor')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(0, 4)
        ax.set_yticks(np.round(np.linspace(ymin, ymax, N), 3))
        ax.set_xticks(np.round(np.linspace(xmin, xmax, N), 3))
        ax.set_zticks(np.int32(np.linspace(0, 4, 5)))
        ax.set_xscale('linear')
        ax.set_yscale('linear')
        ax.view_init(20, 50)
        ax.scatter(x, y, z, c=c)
        plt.tight_layout()
        plt.show()
        plt.savefig("images/"+str(filename)+".svg", figsize=(10,10))



"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mticker
import matplotlib as mpl
%matplotlib notebook

mpl.style.use('tableau-colorblind10')
mpl.rcParams['image.cmap'] = 'tab20'
mpl.rcParams['grid.color'] = 'k'
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['grid.linewidth'] = 0.5
# mpl.rcParams['axes.autolimit_mode'] = 'round_numbers'
mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.ymargin'] = 0
mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.ymargin'] = 0
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6

import geopy.distance

def error_distance (inputDataFrame, x = ["LATITUDE", "PRED_LATITUDE"], y = ["LONGITUDE", "PRED_LONGITUDE"], z=["FLOOR", "PRED_FLOOR"]):
    altura = 2.5
    real_point = (inputDataFrame[x[0]], inputDataFrame[y[0]])
    pred_point= (inputDataFrame[x[1]], inputDataFrame[y[1]])
    #https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude
    #https://en.wikipedia.org/wiki/Vincenty%27s_formulae
    var_xy = geopy.distance.vincenty(real_point, pred_point).m
    var_z = np.abs(inputDataFrame[z[0]] - inputDataFrame[z[1]]) * altura
    inputDataFrame['ERR_REG'] = var_xy
    inputDataFrame['ERR'] = np.sqrt(var_xy*var_xy + var_z*var_z)
    

def graphicMap2D(inputDataFrame, filename="test", x = "LATITUDE", y="LONGITUDE", hue="BUILDINGID"):
    sns.lmplot(x=x, y=y, hue=hue, data=inputDataFrame, fit_reg=False, x_jitter=.1, markers='.')
    plt.show()
    plt.savefig("images/"+str(filename)+".svg")

def graphicBuildings(inputDataFrame, columns = ["LATITUDE", "LONGITUDE", "FLOOR"], filename="test"):
    fig = plt.figure(figsize=(15,8))
    fig.tight_layout()
    for index in range(3):
        ax = fig.add_subplot(1, 3, index+1, projection='3d')
        fig.tight_layout()
        dfGraphics = inputDataFrame[inputDataFrame.BUILDINGID==index]
        x = dfGraphics[columns[0]]
        y = dfGraphics[columns[1]]
        z = dfGraphics[columns[2]]
        N = 6
        xmin, xmax = min(x), max(x)
        ymin, ymax = min(y), max(y)
        c = 4*dfGraphics.BUILDINGID + dfGraphics.FLOOR # colorSet
        ax.set_xlabel('Latitude')
        ax.set_ylabel('Longitude')
        ax.set_zlabel('Floor')
        ax.set_title('Building '+str(index))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(0, 4)
        ax.set_yticks(np.round(np.linspace(ymin, ymax, N), 3))
        ax.set_xticks(np.round(np.linspace(xmin, xmax, N), 3))
        ax.set_zticks(np.int32(np.linspace(0, 4, 5)))
        ax.set_xscale('linear')
        ax.set_yscale('linear')
        ax.view_init(10, 60)
        ax.scatter(x, y, z, c=c)
    plt.tight_layout()
    plt.show()
    plt.savefig("images/"+str(filename)+".svg")
    
def graphicMap3D(inputDataFrame, columns = ["LATITUDE", "LONGITUDE", "FLOOR"], filename="test"):
    fig = plt.figure(figsize=(15,8))
    fig.tight_layout()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    fig.tight_layout()
    dfGraphics = inputDataFrame
    x = dfGraphics[columns[0]]
    y = dfGraphics[columns[1]]
    z = dfGraphics[columns[2]]
    N = 6
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)
    c = 4 * dfGraphics.BUILDINGID + dfGraphics.FLOOR
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    ax.set_zlabel('Floor')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(0, 4)
    ax.set_yticks(np.round(np.linspace(ymin, ymax, N), 3))
    ax.set_xticks(np.round(np.linspace(xmin, xmax, N), 3))
    ax.set_zticks(np.int32(np.linspace(0, 4, 5)))
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    ax.view_init(20, 50)
    ax.scatter(x, y, z, c=c)
    plt.tight_layout()
    plt.show()
    plt.savefig("images/"+str(filename)+".svg")
"""