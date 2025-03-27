#Alexander Poulin Jan 2025
import matplotlib.pyplot as plt
# import mplhep as hep 
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.colors as mcolors

"""
This file contains functions to create plots for the tracking project.
"""

def percent_difference(a, b):
    return abs(a - b) / ((a + b) / 2) * 100

def percent_difference_error(a, b, sigma_a, sigma_b):
    """Computes the propagated error in percent difference."""
    pd = percent_difference(a, b)
    factor = 200 / (a + b)**2
    sigma_pd = factor * (abs(a - b) * sigma_a + abs(b - a) * sigma_b)
    return pd, sigma_pd

def calcEfficiency(typeFile, hist):
    # if typeFile == "Bkg": #make efficiency same between signal and bkg
    if typeFile == "":
        efficiency = round(hist["no_neighbors_removed"] / (hist["neighbors_remained"] + hist["no_neighbors_removed"]), 2)
    # elif typeFile == "Signal":
    if typeFile == "Signal" or typeFile == "Bkg":
        efficiency = round(hist["neighbors_remained"] / (hist["neighbors_remained"] + hist["no_neighbors_removed"]), 2)
    else:
        efficiency = 0 #placeholder for combined
    return efficiency

def calcBinomError(subsetHist, hist, subsetHistError, histError, inPercent=True):
    #calculate binomial error based off TH1:Divide
    resultEfficiency = []
    resultEfficiencyError = []
    c1 = 1
    c2 = 1 #placeholder for weights
    if len(subsetHist) != len(hist):
        print("Error: subsetHist and hist must be the same length")
        return None
    for i in range(len(subsetHist)):
        b1 = subsetHist[i]
        b2 = hist[i]
        if b2: #if b2 is not zero
            resultEfficiency.append(b1 * c1 / b2 * c2) #use if weights
        else:
            resultEfficiency.append(0)
        
        #get error
        if b2 == 0:
            resultEfficiencyError.append(0)
            continue
        b1sq = b1 * b1
        b2sq = b2 * b2
        c1sq = c1 * c1
        c2sq = c2 * c2
        e1sq = subsetHistError[i] * subsetHistError[i]
        e2sq = histError[i] * histError[i]
        if b1 != b2:
            resultEfficiencyError.append(abs((1 - 2 * b1 / b2) * e1sq + b1sq * e2sq / b2sq) / b2sq)
        else:
            resultEfficiencyError.append(0)
    return resultEfficiency, resultEfficiencyError
    
    
def hist_plot(h, outname, title, xMin=-1, xMax=-1, yMin=-1, yMax=-1, 
              xLabel="", yLabel="Events", logY=False, logX=False, 
              label="*MC Particle", autoBin=False,
              binLow = 0.00001, binHigh=2, binSteps=0.3, binType="exp", 
              figure = plt.figure(), axe = "", save=True, barType="bar",
              includeLegend = True, lineStyle = "", alpha=1,
              density=False, legendOutside=False, includeGrid=True, ncols=1, pdf=False):
    """
    Create a histogram plot with the given parameters.
    
    Inputs:
        h: histogram data
        outname: output file name
        title: plot title
        xMin: minimum x value
        xMax: maximum x value
        yMin: minimum y value
        yMax: maximum y value
        xLabel: x-axis label
        yLabel: y-axis label
        logY: log scale for y-axis
        logX: log scale for x-axis
        label: label for the histogram
        autoBin: automatically determine binning
        binLow: low end of binning
        binHigh: high end of binning
        binSteps: binning steps
        binType: binning type
        figure: figure object
        axe: axis object
        save: save the plot
        barType: type of bar
        includeLegend: include legend
        lineStyle: line style
        alpha: transparency
    Return: None, save the plot
    """
    
    #recreate plot_hist but using hist instead of hep
    if axe == "":
        figure.clf()
        axe = figure.subplots()
    
    if autoBin or binType not in ["exp", "lin"]:
        if binType not in ["exp", "lin"]:
            print("Invalid binType, using auto binning")
        axe.hist(h, histtype=barType, linewidth=2, label=label, alpha=alpha, density=density) #autoBin lets hist decide the binning
    else:
        if binType not in ["exp", "lin"]:
            print("Invalid binType, using auto binning")
            axe.hist(h, histtype=barType, linewidth=2, label=label, alpha=alpha, density=density)
        elif binType == "exp":
            binn = np.exp(np.arange(np.log(binLow), np.log(binHigh), binSteps))
        elif binType == "lin":
            binn = np.arange(binLow, binHigh, binSteps)
        if lineStyle != "":
            axe.hist(h, bins=binn, histtype=barType, linewidth=2, label=label, linestyle=lineStyle, alpha=alpha, density=density)
        else:
            axe.hist(h, bins=binn, histtype=barType, linewidth=2, label=label, alpha=alpha, density=density)
    
    if save:
        axe.set_title(title)
        axe.set_xlabel(xLabel)
        axe.set_ylabel(yLabel)
        if includeGrid:
            axe.grid(which='both')  # Enable both major and minor grid lines
            axe.minorticks_on()  # Enable minor ticks
            axe.grid(which='major', linestyle='--', alpha=0.6)  # Dashed grid with transparency
            axe.grid(which='minor', linestyle=':', alpha=0.4)  # Minor grid (dotted)
        
        if includeLegend:
            axe.legend(fontsize='x-small')
            if legendOutside:
                axe.legend(loc='upper left', ncols=ncols, bbox_to_anchor=(1.05, 1))
        if logY:
            axe.set_yscale("log")
        if logX:
            axe.set_xscale("log")
        
        if xMin != -1 and xMax != -1:
            axe.set_xlim([xMin, xMax])
        if yMin != -1 and yMax != -1:
            axe.set_ylim([yMin, yMax])
        
        figure.savefig(outname, dpi=300, bbox_inches="tight")
        if pdf:
            figure.savefig(outname.replace(".png", ".pdf"), bbox_inches="tight")
    
def multi_hist_plot(h, outname, title, xMin=-1, xMax=-1, yMin=-1, yMax=-1,
                    xLabel="", yLabel="Events", 
                    logY=False, logX=False, autoBin=False, barType="step",
                    binLow = 0.00001, binHigh=2, binSteps=0.3, binType="exp", 
                    label="*MC Particle", figure = plt.figure(), axe = "",
                    contrast=False, density=False, pdf=False):
    """
    Create a histogram plot (which will be multi-stacked) with the given parameters.
    
    Inputs:
        h: dictionary of histogram data
        outname: output file name
        title: plot title
        xMin: minimum x value
        xMax: maximum x value
        yMin: minimum y value
        yMax: maximum y value
        xLabel: x-axis label
        yLabel: y-axis label
        logY: log scale for y-axis
        logX: log scale for x-axis
        autoBin: automatically determine binning
        binLow: low end of binning
        binHigh: high end of binning
        binSteps: binning steps
        binType: binning type
        figure: figure object
        axe: axis object
        contrast: create more difference between the lines
        density: density plot
        
    Return: None, save the plot
    """
    
    if axe == "":
        figure.clf()
        axe = figure.subplots()
    #h is a dictionary, iterate through all keys and plot them
    
    #if a key has no values, remove:
    keys = list(h.keys())
    for key in keys:
        if len(h[key]) == 0:
            h.pop(key)
            
    ncol=1
    legendOutside=False
    if len(h.keys()) > 10:
        ncol=3
    elif len(h.keys()) > 5:
        ncol=2
    if len(h.keys()) > 3:
        legendOutside=True
    
    i = 0
    for key in h.keys():
        if key == list(h.keys())[-1]:
            # print("plotting last key")
            hist_plot(h[key], outname, title, xLabel=xLabel, yLabel=yLabel, 
                        logY=logY, logX=logX, autoBin=autoBin, 
                        binLow=binLow, binHigh=binHigh,binSteps=binSteps,
                        binType=binType, barType=barType,
                        save=True, label=key, xMax=xMax, xMin=xMin, yMax=yMax, yMin=yMin,
                        figure=figure, axe=axe, density=density, ncols=ncol, legendOutside=legendOutside, pdf=pdf)
        else:
            #alternate linestyles:
            lineStyle = ""
            alpha = 1
            print(key)
            if contrast: #create more difference between the lines
                colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 'black']
                lineStyle = ["-","--",":","-."]
                alpha = 0.7
                hist_plot(h[key], outname, title, xLabel=xLabel, yLabel=yLabel, 
                            logY=logY, logX=logX, autoBin=autoBin, 
                            binLow=binLow, binHigh=binHigh,binSteps=binSteps,
                            binType=binType, barType=barType,
                            save=False, label=key, 
                            figure=figure, axe=axe, xMax=xMax, xMin=xMin, yMax=yMax, yMin=yMin,
                            lineStyle=lineStyle[i % len(lineStyle)], lineColor=colors[i % len(colors)],
                            alpha=alpha, density=density, legendOutside=legendOutside, ncols=ncol, pdf=pdf)
            else:
                hist_plot(h[key], outname, title, xLabel=xLabel, yLabel=yLabel, 
                                logY=logY, logX=logX, autoBin=autoBin, 
                                binLow=binLow, binHigh=binHigh,binSteps=binSteps,
                                binType=binType, barType=barType,
                                save=False, label=key, xMax=xMax, xMin=xMin, yMax=yMax, yMin=yMin,
                                figure=figure, axe=axe, density=density, legendOutside=legendOutside, 
                                ncols=ncol, pdf=pdf)
        i += 1
    
def bar_step_multi_hist_plot(h1, hr, outname, title, xMin=-1, xMax=-1, yMin=-1, yMax=-1, 
                             xLabel="", yLabel="Events", logY=False, logX=False, autoBin=False, 
                             binLow = 0.00001, binHigh=2, binSteps=0.3, binType="exp", 
                             label="*MC Particle", figure = plt.figure(), axe = "", pdf=False):
    """
    Create a histogram plot (plot first hist with filled, each subseequent will be lines) with the given parameters.
    
    Inputs:
        h1: histogram data
        hr: dictionary of histogram data
        outname: output file name
        title: plot title
        xMin: minimum x value
        xMax: maximum x value
        yMin: minimum y value
        yMax: maximum y value
        xLabel: x-axis label
        yLabel: y-axis label
        logY: log scale for y-axis
        logX: log scale for x-axis
        autoBin: automatically determine binning
        binLow: low end of binning
        binHigh: high end of binning
        binSteps: binning steps
        binType: binning type
        figure: figure object
        axe: axis object
    Return: None, save the plot
    """
    
    if axe == "":
        figure.clf()
        axe = figure.subplots()

    #plot first one filled
    hist_plot(h1, outname, title, xMin, xMax, yMin, yMax, xLabel, yLabel, \
        logY, logX, label, autoBin, barType="bar", binLow=binLow, \
            binHigh=binHigh, binSteps=binSteps, binType=binType, \
                figure=figure, axe=axe, save=False, pdf=pdf)
    
    #plot the rest as lines
    multi_hist_plot(hr, outname, title, xMin, xMax, yMin, yMax, xLabel, yLabel, \
        logY, logX, autoBin, "step", binLow, binHigh, binSteps, binType=binType, \
            label=label, figure=figure, axe=axe, pdf=pdf)
    
def bar_plot(hkeys, hvalues, outname, title, xLabel, yLabel, 
             width=0.8, logY=True, save=True, rotation=0, 
             label="*MC Particle", statusUpdate=False, additionalText="", 
             includeLegend = False, tickrange=[], sort=True,
             figure = plt.figure(), axe = "", fontSize=20, pdf=False):
    """
    Create a bar plot with the given parameters.
    
    Inputs:
        hkeys: keys for the histogram
        hvalues: values for the histogram
        outname: output file name
        title: plot title
        xLabel: x-axis label
        yLabel: y-axis label
        width: width of the bars
        logY: log scale for y-axis
        save: save the plot
        rotation: rotation of the x-axis labels
        label: label for the histogram
        statusUpdate: print status updates
        additionalText: additional text to add to the plot
        includeLegend: include legend
        figure: figure object
        axe: axis object
        fontSize: font size for the x-axis labels
    Return: None, save the plot
    """
    
    if axe == "":
        figure.clf()
        axe = figure.subplots()
    
    if statusUpdate:
        print("Beginning to bar plot...")
        
    if len(tickrange) == 0:
        tickrange = range(len(hkeys))
        #increase distance between them by 2:
        if rotation > 60:
            tickrange = [x*1.5 for x in tickrange]
    
    if sort:
        sortedValues = sorted(hvalues, reverse=True)
        sortedKeys = [x for _, x in sorted(zip(hvalues, hkeys), reverse=True)]
    else:
        sortedValues = hvalues
        sortedKeys = hkeys
    # print(f"hvalues: {hvalues}")
    axe.bar(tickrange, sortedValues, width=width, label=label)
    
    if statusUpdate:
        print("Finished plotting, updating plot parameters...")
        
    if save:
        #fix when h.keys() is long and overlaps with neighboring labels:
        axe.set_xticks(tickrange)
        axe.set_xticklabels(hkeys, rotation = rotation, fontsize=fontSize)
        
        axe.set_title(title)
        axe.set_xlabel(xLabel)
        axe.set_ylabel(yLabel)
        
        if logY:
            axe.set_yscale("log")
        
        # Legend to the side
        if includeLegend:
            axe.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

        if additionalText != "":
            axe.text(1.05, 0.7, additionalText, transform=axe.transAxes, fontsize=13, va='top')
        
        if statusUpdate:
            print("Saving plot...")
        figure.savefig(outname, bbox_inches="tight")
        if pdf:
            figure.savefig(outname.replace(".png", ".pdf"), bbox_inches="tight")
        
def multi_bar_plot(d_dict, outname, title, xLabel, yLabel, 
             width=0.8, logY=False, rotation=0, includeLegend=False, filled=True,
             label=[], statusUpdate=False, additionalText="", sort=True, pdf=False):
    """
    Create a multi-bar plot with d_dict, where it is a dictionary of dictionaries.
    Each parent dictionary will be a different label. The child dictionary will be the keys and values.
    
    Inputs:
        d_dict: dictionary of dictionaries
        outname: output file name
        title: plot title
        xLabel: x-axis label
        yLabel: y-axis label
        width: width of the bars
        logY: log scale for y-axis
        rotation: rotation of the x-axis labels
        label: label for the histogram
        statusUpdate: print status updates
        additionalText: additional text to add to the plot
        sort: sort the keys by value
    Return: None, save the plot
    """
    
    #maybe add remove any empty lists from d_dict
    
    figure = plt.figure()
    axe = figure.subplots()
    bars = []
    
    if statusUpdate:
        print("Beginning to multi-bar plot...")
    
    for i, key in enumerate(d_dict.keys()):
        if sort:
            sortedValues = sorted(d_dict[key].values(), reverse=True)
            sortedKeys = sorted(d_dict[key], key=d_dict[key].get, reverse=True)
        else:
            sortedValues = d_dict[key].values()
            sortedKeys = d_dict[key].keys()
        tickrange = range(len(sortedKeys))
        #increase distance between them by 2:
        if rotation > 60:
            tickrange = [x*1.5 for x in tickrange]
        
        # offsets = np.linspace(-width, width, sortedKeys)
        
        #include offeset
        
        if filled:
            axe.bar(tickrange, sortedValues, width=width, label=key, alpha=0.8)
        else:
            #plot as bar plot that is just the outline of bar with edgecolor as bar color
            bar = axe.bar(tickrange, sortedValues, width=width, label=key)
            bars.append(bar)
        
    if not filled:
        for i,barCont in enumerate(bars):
            for bar in barCont:
                color = bar.get_facecolor()  # Get assigned color
                bar.set_edgecolor(color)  # Set edge color
                bar.set_facecolor('none')  # Now remove fill
            
    if statusUpdate:
        print("Finished plotting, updating plot parameters...")
        
    #fix when h.keys() is long and overlaps with neighboring labels:
    axe.set_xticks(tickrange)
    axe.set_xticklabels(sortedKeys, rotation = rotation)
    
    axe.set_title(title)
    axe.set_xlabel(xLabel)
    axe.set_ylabel(yLabel)
    
    if logY:
        axe.set_yscale("log")
        
    # print(axe.get_legend_handles_labels()[1])
        
    # Legend to the side if includeLegend
    if includeLegend:
        if len(axe.get_legend_handles_labels()[1]) > 6:
            axe.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        else:
            axe.legend()
    
    if additionalText != "":
        axe.text(1.05, 0.7, additionalText, transform=axe.transAxes, fontsize=13, va='top')
        
    if statusUpdate:
        print("Saving plot...")
    figure.savefig(outname, bbox_inches="tight")
    if pdf:
        figure.savefig(outname.replace(".png", ".pdf"), bbox_inches="tight")
    
    
def xy_plot(x, y, outname, title, xLabel, yLabel, logY=False, logX=False, 
            label="*MC Particle", statusUpdate=False, additionalText="", 
            figure = plt.figure(), axe = "", includeLegend = True, scatter=True, 
            errorBars=False, yerr=[], includeGrid=True, save=True, color="", pdf=False,
            weights=False, cmap='viridis', mark='.'):
    """
    Create a xy plot with the given parameters.
    
    Inputs:
        x: x values
        y: y values
        outname: output file name
        title: plot title
        xLabel: x-axis label
        yLabel: y-axis label
        logY: log scale for y-axis
        logX: log scale for x-axis
        label: label for the histogram
        statusUpdate: print status updates
        additionalText: additional text to add to the plot
        figure: figure object
        axe: axis object
        includeLegend: include legend
        scatter: scatter plot
        errorBars: include error bars
        yerr: y error values
        includeGrid: include grid
    Return: None, save the plot
    """
    if axe == "":
        figure.clf()
        axe = figure.subplots()
    if statusUpdate:
        print("Beginning to xy plot...")
    if scatter:
        if errorBars:
            axe.errorbar(x, y, yerr, label=label, linestyle='none', marker=mark, markersize=4)
        elif weights:
            axe.scatter(x, y, label=label, c=cmap, cmap=cmap)
        else:
            if color == "":
                axe.scatter(x, y, label=label)
            else:
                axe.scatter(x, y, label=label, color=color)
    else:
        if color == "":
            axe.step(x, y, label=label)
        else:
            axe.step(x, y, label=label, color=color)
        
        
    if save:
        if statusUpdate:
            print("Finished plotting, updating plot parameters...")
            
        #fix when h.keys() is long and overlaps with neighboring labels:
        axe.set_title(title)
        axe.set_xlabel(xLabel)
        axe.set_ylabel(yLabel)
        
        if includeGrid:
            # print("including grid")
            axe.grid(which='both')  # Enable both major and minor grid lines
            axe.minorticks_on()  # Enable minor ticks
            axe.grid(which='major', linestyle='--', alpha=0.6)  # Dashed grid with transparency
            axe.grid(which='minor', linestyle=':', alpha=0.4)  # Minor grid (dotted)
                
        if logY:
            axe.set_yscale("log")
        if logX:
            axe.set_xscale("log")
        if includeLegend:
                # axe.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
            plt.legend(fontsize='x-small')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        if additionalText != "":
            # axe.text(1.05, 0.7, additionalText, transform=axe.transAxes, fontsize=13, va='top')
            axe.text(0.5, -0.2,  additionalText, ha="center", va="bottom", fontsize=9, transform=axe.transAxes)
        
        print(f"outname: {outname}")
        figure.savefig(outname, bbox_inches="tight")
        if pdf:
            figure.savefig(outname.replace(".png", ".pdf"), bbox_inches="tight")
    
    
def hist2d(x, y, outname, title, xLabel, yLabel, logScale=False, 
            label="*MC Particle", statusUpdate=False, additionalText="", cmap="Blues", colorbarLabel="",
            figure = plt.figure(), axe = "", includeLegend = False, includeColorbar=True,
            save=True, binSize=100, binSizeX=-1, binSizeY=-1, weights=None, pdf=False):
    """
    Create a 2d histogram plot with the given parameters.
    
    Inputs:
        x: x values
        y: y values
        outname: output file name
        title: plot title
        xLabel: x-axis label
        yLabel: y-axis label
        logY: log scale for y-axis
        logX: log scale for x-axis
        label: label for the histogram
        statusUpdate: print status updates
        additionalText: additional text to add to the plot
        figure: figure object
        axe: axis object
        includeLegend: include legend
    Return: None, save the plot
    """
    if axe == "":
        figure.clf()
        axe = figure.subplots()
    if statusUpdate:
        print("Beginning to 2dhist plot...")
    if binSizeX == -1 or binSizeY == -1:
        binSizeX = binSize
        binSizeY = binSize
        
    if weights is not None:
        if logScale:
            hist, xedges, yedges, im = axe.hist2d(x, y, bins=(binSizeX, binSizeY), cmap=cmap, weights=weights, norm=mcolors.LogNorm())
        else:
            hist, xedges, yedges, im = axe.hist2d(x, y, bins=(binSizeX, binSizeY), cmap=cmap, weights=weights)
    else:
        if logScale:
            hist, xedges, yedges, im = axe.hist2d(x, y, bins=(binSizeX, binSizeY), cmap=cmap, norm=mcolors.LogNorm())
        else:
            hist, xedges, yedges, im = axe.hist2d(x, y, bins=(binSizeX, binSizeY), cmap=cmap)
        
    if save:
        if statusUpdate:
            print("Finished plotting, updating plot parameters...")
            
        #fix when h.keys() is long and overlaps with neighboring labels:
        axe.set_title(title)
        axe.set_xlabel(xLabel)
        axe.set_ylabel(yLabel)
        
        
        
        if includeLegend:
                # axe.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
            plt.legend(fontsize='x-small')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        if additionalText != "":
            axe.text(1.05, 0.7, additionalText, transform=axe.transAxes, fontsize=13, va='top')
        
        #include grid
        axe.grid(True, linestyle='--', alpha=0.6)
        axe.grid(True, which='minor', linestyle=':', alpha=0.1)
        
        #include colorbar
        if includeColorbar:
            cbar = figure.colorbar(im, ax=axe)
            cbar.set_label(colorbarLabel, rotation=-90, labelpad=15)
        
        print(f"outname: {outname}")
        figure.savefig(outname, bbox_inches="tight")
        #increase resolution
        if pdf:
            figure.savefig(outname.replace(".png", ".pdf"), bbox_inches="tight")
        
def heatmap(z, outname, title, xLabel, yLabel, logScale=False,
            label="*MC Particle", statusUpdate=False, additionalText="", colorbarLabel="",
            figure = plt.figure(figsize=(10, 10)), axe = "", save=True, cmap="Blues", pdf=False):
    """
    Create a imshow plot with the given parameters.
    
    Inputs:
        z: matrix in shape of 2d, with each point being the z value
        outname: output file name
        title: plot title
        xLabel: x-axis label
        yLabel: y-axis label
        logY: log scale for y-axis
        logX: log scale for x-axis
        label: label for the histogram
        statusUpdate: print status updates
        additionalText: additional text to add to the plot
        figure: figure object
        axe: axis object
    Return: None, save the plot
    """
    if axe == "":
        figure.clf()
        axe = figure.subplots()
    if statusUpdate:
        print("Beginning to heatmap plot...")
        
    if logScale:
        img = axe.imshow(z, cmap=cmap, norm=LogNorm(), interpolation="nearest", aspect='5', origin='lower')
    else:
        img = axe.imshow(z, cmap=cmap, interpolation="nearest", aspect='5', origin='lower')
    
    if save:
        if statusUpdate:
            print("Finished plotting, updating plot parameters...")
            
        #fix when h.keys() is long and overlaps with neighboring labels:
        axe.set_title(title)
        axe.set_xlabel(xLabel)
        axe.set_ylabel(yLabel)
        
        # #set x and y rangegs
        # axe.set_aspect('equal', adjustable='box')
        # axe.invert_yaxis()
        
        # #colorbar
        cbar = figure.colorbar(img, ax=axe, shrink=0.6)
        cbar.set_label(colorbarLabel, rotation=-90, labelpad=15)
        
        # xw
        # if includeLegend:
        #         # axe.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        #     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        if additionalText != "":
            axe.text(1.05, 0.7, additionalText, transform=axe.transAxes, fontsize=13, va='top')
        
        print(f"outname: {outname}")
        figure.savefig(outname, bbox_inches="tight")
        if pdf:
            figure.savefig(outname.replace(".png", ".pdf"), bbox_inches="tight")


if __name__ == "__main__":
    #dont do anything
    pass