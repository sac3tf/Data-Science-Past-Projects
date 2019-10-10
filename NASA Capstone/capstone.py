# import relevant libraries
import numpy as np
import matplotlib.pyplot as plt
from astroquery.splatalogue import Splatalogue  # We must acknowledge authors for use of this library
from astropy import units as u

# read in the data
filenames = ["./Data/Win0.clean1.contsub_Jy.rest.scom.c.txt",
             "./Data/Win1.clean1.contsub_Jy.rest.scom.c.txt",
             "./Data/Win2.clean1.contsub_Jy.rest.scom.c.txt",
             "./Data/Win3.clean1.contsub_Jy.rest.scom.c.txt"]
# 'data' will be a 4 element list, with each element representing the data from 1 text file
data = [np.loadtxt(f) for f in filenames]

show_plot = False

def create_plot():
    """ Recreate the plot from Cordiner et. al. 2015, Figure 1 """
    # Defining the figure might need more finess if there are more than 4 datasets
    # We need to consult a domain expert to learn if ALMA data always comes back with 4 datasets
    fig, axs = plt.subplots(2, 2, figsize = (10, 6))
    for index, ax in enumerate(axs.flat):
        # Plot each data set
        ax.plot(data[index][:,0],
                data[index][:,1],
                linewidth = 0.25)
        ax.set(ylabel = "Flux (Jy)")
        ax.set(xlabel = "Frequency (GHz)")
        # Clean up the xticks
        ax.set_xticks(np.arange(round(data[index][0,0], 1), round(data[index][-1,0], 1), step = 0.5))
        # Remove the space from the borders of the plot along the X axis
        ax.autoscale(enable = True, axis = 'x', tight = True)
    # Add some space between the plots
    plt.subplots_adjust(hspace = .3, wspace = .3)
    return fig, axs

# Add lines to the plot where molecules were found
def add_lines(id, molecules):
    """This function will add dashed lines to the plot where molecules were detected"""
    for freq in molecules.keys():
        axs[id].axvline(x = float(freq),
                        ymin = 0,
                        ymax = 1,
                        dashes = [6, 2],
                        linewidth = 0.25,
                        color = "gray",
                        alpha = 0.5)

# Find the frequencies at which molecule flux is significant
def find_molecules():
    """ Classify the molecules from their frequency for each dataset """
    for id, dataset in enumerate(data):
        # Locate the indices where the flux is greater than 3 standard deviations
        # There are 4 datasets.  Column 0 is the frequency, column 1 is the flux
        # Splatalogue appears to be accurate up to 5 decimal places
        molecules = {}  # An empty dictionary to store the results of each detected molecule and rest frequency
        delta = 0.00005 # +/- when searching frequencies
        frequencies = np.round(dataset[np.where( dataset[:, 1] >= 3 * np.std(dataset[:, 1])), 0], 5)[0]
        for freq in frequencies:
            results = Splatalogue.query_lines( (freq - delta)*u.GHz, (freq + delta)*u.GHz)
            # Append the chemical names corresponding to the searched frequency.
            molecules[freq] = results["Chemical Name"].tolist() if len(results) > 0 else "Unknown"
            # Append the chemical name and frequency to the dictionary of all molecules found
            if len(results) > 0:
                for molecule in results["Chemical Name"].tolist():
                    if molecule in all_molecules.keys():
                        all_molecules[molecule].append(freq)
                    else:
                        all_molecules[molecule] = [freq]
            else:
                if "Unknown" in all_molecules.keys():
                    all_molecules["Unknown"].append(freq)
                else:
                    all_molecules["Unknown"] = [freq]
        add_lines(id, molecules)

# Run it all
fig, axs = create_plot()
axs = axs.flat
all_molecules = {}  # This will store the molecule name and every frequency it is found at
find_molecules()
for molecule in all_molecules.keys():
    print(molecule, ": ", all_molecules[molecule])
plt.show()
