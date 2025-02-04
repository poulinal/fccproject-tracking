#Alexander Poulin Jan 2025
import matplotlib.pyplot as plt
import mplhep as hep 
import numpy as np

"""
This file contains functions to create plots for the tracking project.
"""



hep.style.use(hep.style.CMS)

def plot_2dhist(h, outname, title, xMin=-1, xMax=-1, yMin=-1, yMax=-1, xLabel="", yLabel="Events", logY=False):
    fig = plt.figure()
    ax = fig.subplots()

    hep.hist2dplot(h, label="", ax=ax)

    ax.set_title(title)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    #ax.legend(fontsize='x-small')
    if logY:
        ax.set_yscale("log")

    if xMin != -1 and xMax != -1:
        ax.set_xlim([xMin, xMax])
    if yMin != -1 and yMax != -1:
        ax.set_ylim([yMin, yMax])


    fig.savefig(outname, bbox_inches="tight")
    fig.savefig(outname.replace(".png", ".pdf"), bbox_inches="tight")

    
def hist_plot(h, outname, title, xMin=-1, xMax=-1, yMin=-1, yMax=-1, 
              xLabel="", yLabel="Events", logY=False, logX=False, 
              label="*MC Particle", autoBin=False,
              binLow = 0.00001, binHigh=2, binSteps=0.3, binType="exp", 
              figure = plt.figure(), axe = "", save=True, barType="bar",
              includeLegend = True, lineStyle = "", alpha=1,
              density=False, legendOutside=False, includeGrid=True, ncols=1):
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
        axe = figure.subplots()
    
    if autoBin: #autoBin lets hist decide the binning
        axe.hist(h, histtype=barType, linewidth=2, label=label, alpha=alpha, density=density)
    else:
        if binType == "exp":
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
            axe.grid(True, linestyle='--', alpha=0.6)  # Dashed grid with transparency
            axe.grid(True, which='minor', linestyle=':', alpha=0.4)  # Minor grid (dotted)
        
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
        #fig.savefig(outname.replace(".png", ".pdf"), bbox_inches="tight")
    
def multi_hist_plot(h, outname, title, xMin=-1, xMax=-1, yMin=-1, yMax=-1,
                    xLabel="", yLabel="Events", 
                    logY=False, logX=False, autoBin=False, barType="step",
                    binLow = 0.00001, binHigh=2, binSteps=0.3, binType="exp", 
                    label="*MC Particle", figure = plt.figure(), axe = "",
                    contrast=False, density=False):
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
                        figure=figure, axe=axe, density=density, ncols=ncol, legendOutside=legendOutside)
        else:
            #alternate linestyles:
            lineStyle = ""
            alpha = 1
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
                            alpha=alpha, density=density, legendOutside=legendOutside, ncols=ncol)
            else:
                hist_plot(h[key], outname, title, xLabel=xLabel, yLabel=yLabel, 
                                logY=logY, logX=logX, autoBin=autoBin, 
                                binLow=binLow, binHigh=binHigh,binSteps=binSteps,
                                binType=binType, barType=barType,
                                save=False, label=key, xMax=xMax, xMin=xMin, yMax=yMax, yMin=yMin,
                                figure=figure, axe=axe, density=density, legendOutside=legendOutside, ncols=ncol)
        i += 1
    
def bar_step_multi_hist_plot(h1, hr, outname, title, xMin=-1, xMax=-1, yMin=-1, yMax=-1, 
                             xLabel="", yLabel="Events", logY=False, logX=False, autoBin=False, 
                             binLow = 0.00001, binHigh=2, binSteps=0.3, binType="exp", 
                             label="*MC Particle", figure = plt.figure(), axe = ""):
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
        axe = figure.subplots()

    #plot first one filled
    hist_plot(h1, outname, title, xMin, xMax, yMin, yMax, xLabel, yLabel, \
        logY, logX, label, autoBin, barType="bar", binLow=binLow, \
            binHigh=binHigh, binSteps=binSteps, binType=binType, \
                figure=figure, axe=axe, save=False)
    
    #plot the rest as lines
    multi_hist_plot(hr, outname, title, xMin, xMax, yMin, yMax, xLabel, yLabel, \
        logY, logX, autoBin, "step", binLow, binHigh, binSteps, binType=binType, \
            label=label, figure=figure, axe=axe)
    
def bar_plot(hkeys, hvalues, outname, title, xLabel, yLabel, 
             width=0.8, logY=False, save=True, rotation=0, 
             label="*MC Particle", statusUpdate=False, additionalText="", 
             includeLegend = True,
             figure = plt.figure(), axe = "", fontSize=20):
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
        axe = figure.subplots()
    
    if statusUpdate:
        print("Beginning to bar plot...")
        
    tickrange = range(len(hkeys))
    #increase distance between them by 2:
    if rotation > 60:
        tickrange = [x*1.5 for x in tickrange]
        
    axe.bar(tickrange, sorted(hvalues), width=width, label=label)
    
    if statusUpdate:
        print("Finished plotting, updating plot parameters...")
        
    if save:
        #fix when h.keys() is long and overlaps with neighboring labels:
        axe.set_xticks(tickrange)
        axe.set_xticklabels(hkeys, rotation = rotation, fontsize=fontSize)
        
        axe.set_title(title)
        axe.set_xlabel(xLabel)
        axe.set_ylabel(yLabel)
        axe.set_yscale("log")
        
        # Legend to the side
        if includeLegend:
            axe.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

        if additionalText != "":
            axe.text(1.05, 0.7, additionalText, transform=axe.transAxes, fontsize=13, va='top')
        
        if statusUpdate:
            print("Saving plot...")
        figure.savefig(outname, bbox_inches="tight")
        #fig.savefig(outname.replace(".png", ".pdf"), bbox_inches="tight")
        
def multi_bar_plot(hStacked, h, outname, title, xLabel, yLabel, 
             width=0.8, logY=False, rotation=0, 
             label=[], statusUpdate=False, additionalText="", sort=True):
    """
    Create a multi-bar plot with the given parameters.
    
    Inputs:
        hStacked: keys for the histogram
        h: values for the histogram
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
    
    
    figure = plt.figure()
    axe = figure.subplots()
    
    #make sure to plot the keys with higher values first
    sortedKeys = h.keys()
    if sort:
        values = h.values()
        sortedValues = sorted(values, reverse=True)
        sortedKeys = []
        for val in sortedValues:
            for key in h.keys():
                if h[key] == val:
                    sortedKeys.append(key)
        print(sortedKeys)
        print(sortedValues)
    for key in sortedKeys:
        #if hStakced is a string
        if type(hStacked) == str:
            if key == list(sortedKeys)[-1]:
                bar_plot(hStacked, h[key], outname, title, xLabel, yLabel, width, 
                         logY, save=True, rotation=rotation, label=key, 
                         statusUpdate=statusUpdate, additionalText=additionalText, 
                         figure=figure, axe=axe)
            else:
                bar_plot(hStacked, h[key], outname, title, xLabel, yLabel, width, 
                         logY, save=False, rotation=rotation, label=key, 
                         statusUpdate=statusUpdate, additionalText=additionalText, 
                         figure=figure, axe=axe)
    
def xy_plot(x, y, outname, title, xLabel, yLabel, logY=False, logX=False, 
            label="*MC Particle", statusUpdate=False, additionalText="", 
            figure = plt.figure(), axe = "", includeLegend = True, scatter=True, 
            errorBars=False, yerr=[], includeGrid=True):
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
        axe = figure.subplots()
    if statusUpdate:
        print("Beginning to xy plot...")
    if scatter:
        if errorBars:
            axe.errorbar(x, y, yerr, label=label, linestyle='none', marker='_')
        else:
            axe.scatter(x, y, label=label)
    else:
        axe.step(x, y, label=label)
    
    if statusUpdate:
        print("Finished plotting, updating plot parameters...")
        
    #fix when h.keys() is long and overlaps with neighboring labels:
    axe.set_title(title)
    axe.set_xlabel(xLabel)
    axe.set_ylabel(yLabel)
    
    if includeGrid:
            axe.grid(True, linestyle='--', alpha=0.6)  # Dashed grid with transparency
            axe.grid(True, which='minor', linestyle=':', alpha=0.4)  # Minor grid (dotted)
            
    if logY:
        axe.set_yscale("log")
    if logX:
        axe.set_xscale("log")
    plt.legend(fontsize='x-small')
    if includeLegend:
            # axe.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    if additionalText != "":
        axe.text(1.05, 0.7, additionalText, transform=axe.transAxes, fontsize=13, va='top')
    
    print(f"outname: {outname}")
    figure.savefig(outname, bbox_inches="tight")
    