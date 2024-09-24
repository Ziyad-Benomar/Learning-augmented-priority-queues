import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from experiments import uploadFromFile
import numpy as np





######################################################################
# Sorting figures
######################################################################

algoNames = {
    "SL": "LAPQ offline predictions",
    "OSL": "LAPQ online predictions",
    "DC": "LAPQ dirty comparisons",
    "FH": "Fibonacci heap",
    "BH": "Binary heap",
    "DS": "Displacement sort",
    "DHS": "Double-Hoover sort"
}

cols = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
dashes = [(1,0),(5,2),(6,2,1,2), (1,2), (5,5), (1,1), (1,4), (7,3,1,3,1,3,1,3), (10,3,1,3)]

def getLabelX(predGenID):
    if predGenID == "class":
        return r"(\#classes)/$n$"
    if predGenID == "decay":
        return r"(\#timesteps)/$(n\sqrt{n})$"
    if predGenID == "damage":
        return "Damage ratio"
    return "[????]"

def getXaxis(m=20):
    return np.linspace(0,1,m+1)

def plots(nvals, predGenID, figsize=(5,4), legendParams={}, labelsize=20, m=20, includeFH=True):
    algos = ["SL", "OSL", "DC", "BH"]
    if includeFH:
        algos.append("FH")
    # xlabel
    labelX = getLabelX(predGenID)
    numPlots = len(nvals)
    
    fg, ax = plt.subplots(1, numPlots)
    #plt.subplots_adjust(wspace=0.1)
    if numPlots == 1: ax = [ax]
    fg.set_size_inches(numPlots*figsize[0], figsize[1])
    for j in range(numPlots):
        n = nvals[j]
        xaxis = getXaxis(m)
        mean, std = uploadFromFile(f"data_{predGenID}_{n}.json")
        meanBC, stdBC = uploadFromFile(f"BC23_{predGenID}_{n}.json")
        if mean == None:
            raise ValueError(f"File 'data_{predGenID}_{n}.json' does not exist ==> No experiments made for n={n} in the {predGenID} setting.")
            return
        if meanBC == None:
            print(f"File 'BC23_{predGenID}_{n}.json' does not exist.")
        else:
            algos += ["DS", "DHS"]
            for algoID in ["DS", "DHS"]:
                expID = f"{algoID}_{predGenID}_{n}"
                mean[expID], std[expID] = meanBC[expID], stdBC[expID]
        for i in range(len(algos)):
            expID = f"{algos[i]}_{predGenID}_{n}"
            if expID in mean:
                #print(expID)
                ax[j].plot(xaxis, mean[expID], dashes=dashes[i], color=cols[i], label = algoNames[algos[i]])
                ax[j].fill_between(xaxis, mean[expID] -0.05-std[expID], mean[expID] +0.05+std[expID], color=cols[i], alpha=0.15)
                ax[j].set_xlabel(labelX, fontsize=labelsize)
                ax[j].set_title(f"n = {n}")
    ax[0].set_ylabel("(\#comparisons)/n", fontsize=labelsize)
    ax[-1].legend(**legendParams)
    figname = f"figures/{predGenID}"
    for n in nvals:
        figname += f"_{n}"
    fg.savefig(figname + ".pdf", bbox_inches='tight')



######################################################################
# Dijkstra's algorithm
######################################################################


pqNames = {
    "OSL": "LAPQ node rank predictions",
    "DC": "LAPQ node dirty comparisons",
    "BH": "Binary heap",
    "FH": "Fibonacci heap"
}

citySize = {
    "brussels": (1380, 2580),
    "paris": (9559, 18400),
    "budapest": (24069, 63165),
    "london": (128924, 300612),
    "new york": (55326, 139547)
}

def plotsDijkstra(cityNames, predGenID, sortedKeys={}, figsize=(5,4), legendParams={}, labelsize=20, m=20):
    pqIDs = ["OSL", "DC", "BH", "FH"]
    labelX = r"(\#classes)/$n$" if predGenID=="class" else r"(\#timesteps)/$n$"
    numPlots = len(cityNames)

    fg, ax = plt.subplots(1, numPlots)
    if numPlots == 1: ax = [ax]
    fg.set_size_inches(numPlots*figsize[0], figsize[1])
    for j in range(numPlots):
        cityName = cityNames[j]
        cityName = cityName.lower()
        xaxis = np.linspace(0,1,m+1) if predGenID=="class" else np.arange(m+1)
        mean, std = uploadFromFile(f"data_dijkstra_{predGenID}_{cityName}.json")
        for i in range(len(pqIDs)):
            expID = f"dijkstra_{pqIDs[i]}_{predGenID}_{cityName}"
            print(expID)
            if i==2 and cityName in sortedKeys:
                i2 = len(pqIDs)
                smean, sstd = sortedKeys[cityName]
                ax[j].plot(xaxis, smean*np.ones_like(xaxis), dashes=dashes[i2], color=cols[i2], label = "LAPQ key rank predctions")
                ax[j].fill_between(xaxis, (smean -0.05-sstd)*np.ones_like(xaxis), (smean+0.05+sstd)*np.ones_like(xaxis), color=cols[i2], alpha=0.15)
            if pqIDs[i] in ["FH", "BH"]:
                print(f"\nFOR {pqIDs[i]}: mean={mean[expID].mean()}, std={std[expID].mean()}\n")
                #mean[expID] = mean[expID].mean()*np.ones_like(mean[expID])
                #std[expID] = std[expID].mean()*np.ones_like(mean[expID])
            ax[j].plot(xaxis, mean[expID], dashes=dashes[i], color=cols[i], label = pqNames[pqIDs[i]])
            ax[j].fill_between(xaxis, mean[expID] -0.05-std[expID], mean[expID] +0.05+std[expID], color=cols[i], alpha=0.15)
            ax[j].set_xlabel(labelX, fontsize=labelsize)
            nodes, edges = citySize[cityName]
            ax[j].set_title(rf"{cityNames[j]}: $n={nodes}, m={edges}$", fontsize=17)
        figname = f"figures/{predGenID}"
    ax[0].set_ylabel("(\#comparisons)/n", fontsize=labelsize)
    ax[-1].legend(**legendParams)
    figname = f"figures/dijkstra_{predGenID}"
    for i in range(numPlots):
        cityName = cityNames[i]
        figname += f"_{cityName[:3]}"
    fg.savefig(figname + ".pdf", bbox_inches='tight')
    print("saved in ", figname+".pdf" )