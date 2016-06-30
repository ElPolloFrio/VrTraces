# Written in 2012 by L. M. Kanofsky, WFO LSX
# Used through 2014
# Updated 6/2016
#
# This script was written to support local research efforts at WFO LSX.
#
# Input: a specially-formatted CSV file containing data points
# for mesovortices.
# Output: a set of graphics including Vr traces

# See the README file for details.

import numpy as np
import string, logging, os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.font_manager
import matplotlib.tri as tri

def setUpTheLogger():
    # Someone set up us the logger? Ha ha ha.

    # Create the logger object.
    logger = logging.getLogger('vrtraces')
    logger.setLevel(logging.INFO)

    # Create a console handler.
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Create a formatter and add it to the handler.
    logformatter = logging.Formatter('%(asctime)s - %(name)-10s - %(levelname)s - %(message)s')
    ch.setFormatter(logformatter)
    
    # Add the handler to the logger.
    logger.addHandler(ch)

    # Log some headers.
    logger.info('---------------------------------')
    logger.info('------ Creating Vr Traces -------')
    logger.info('---------------------------------')
    
    return logger


def load_file(filename, lumberjack):
    # Load in the file contents as a numpy array.
    # Skip headers.
    data = np.loadtxt(filename, dtype = 'string', delimiter = ',', skiprows = 1)
    lumberjack.info('Loaded %s', filename)

    return data


def setVars(configfile):
    # Define some variables in an external configuration file to make this
    # script easier to use for people who don't program...which is,
    # unfortunately, the expected user base.

    # filename: (string) Filename containing mesovortex data points. Either
    #   specify the full path (e.g.,
    #   C:\\Users\\LMK\\Documents\\Python code\\sampleVR.csv on Windows) or
    #   the filename alone if the CSV file is in the same directory as this
    #   script.
    #
    # mv_name: (string) Name or designation of the mesovortex. This is used
    #   in the figure titles. 
    #
    # vr_ylim: (integer or decimal) Upper limit of the y-axis for the Vr
    #   plots. Pick a value that's near or slightly higher than the highest
    #   height in the spreadsheet for Vr values.
    #
    # tz_units: (string) Time zone designation to use on the x-axis for all
    #   time series plots. This should be the same time zone used in the
    #   spreadsheet. Typically, that would be UTC since radar data sets are
    #   in UTC. However, some forecasters might prefer to keep their
    #   spreadsheets in local time instead.
    #
    # corediam_units: (string) Core diameter units to use on the y-axis for
    #   core diameter plots. 
    #
    # hght_units: (string) Height units to use on the y-axis for the Vr plots.
    #
    # corediam_colmap: (string, must be a valid matplotlib colormap name)
    #   A matplotlib colormap from which to select colors to use for the core
    #   diameter plot. Good options include 'hsv', 'gist_rainbow', 'Spectral',
    #   and 'Paired'. Note that these names are case-sensitive.
    #
    # filled_contour_colmap: (string, must be a valid matplotlib colormap name)
    #   A matplotlib colormap to use on the filled contour plots. Good options
    #   include 'hot_r'.
    #
    # modulo_for_timeticks: (integer) Increase this value to 2 or 4 if too
    #   many time ticks are shown on the x-axis. To show all time ticks, set
    #   this value to 1.
    #
    # VrPointsBins: (comma-separated list of integers or decimal numbers) Bin values
    #   for the Vr plot with color-coded Vr values.
    #
    # VrPointsCols: (comma-separated list of strings which must be valid matplotlib
    #   color designations) Colors associated with the bin values defined
    #   above. This is used on the Vr plot with color-voded Vr values.
    #
    # VrPointsBins_algfail_adj: (decimal) The contoured plot with
    #   raw numbers always has problems because the data set is too sparse.
    #   The contours for the gridded interpolation sometimes look incorrect
    #   because of the interpolation. Sometimes, the contours look better
    #   by contouring at 14.9 instead of 15.0, for example. Set this value
    #   to -0.1 (use other values at your own risk) to produce a plot
    #   with contours at VrPointsBins + VrPointsBins_algfail_adj. It
    #   may or may not improve the appearance of the contours with the
    #   gridded interpolation.
    #
    # tri_subdivisions: (integer) Another interpolation method is the
    #   triangular interpolation. This plot is produced by default. Depending
    #   on the data set, the contours might look better after using a
    #   triangular refiner. Set this to the number of subdivisions for each
    #   triangle. Typical values are 1-3. This script will disallow anything
    #   above 4.
    # 
    # VrContoursBins: (comma-separated list of integers or decimal numbers) Bin
    #   values for the gridded and triangulated contoured Vr plots. Note that the
    #   plotting fails if the lowest specified contour is not found in the data.
    #
    # VrContoursCols: (comma-separated list of strings which must be valid matplotlib
    #   color designations) Colors associated with the bin values defined above. This
    #   is used on the gridded and triangulated contoured Vr plots.
    #
    # ContourIntervals: (comma-separated list of integers or decimal numbers) Bin
    #   values for the gridded and triangulated contoured Vr plots.
    #
    # ContourCols: (comma-separated list of strings which must be valid matplotlib
    #   color designations) Colors associated with the bin values defined above. This
    #   is used on the gridded and triangulated contoured Vr plots.
    #
    # ContourTickLabels: (comma-separated list of integers or decimals) This is for
    #   convenience when using the gridded contoured plots with a contouring algorithm
    #   failure. Plotting at 14.9 but labeling it as 15, for example, because the contouring
    #   algorithm failed when plotting at 15. Make sure that each defined interval has an
    #   associated label.
    #
    # rcParamsFontSize: (integer) Ron always wanted bigger numbers. Change this to change
    #   the font size, as in plt.rcParams['font.size'] = 18.
    
    dictPlotParms = {}

    cfile = open(configfile, 'r')
    clines = cfile.readlines()
    cfile.close()

    for line in clines:
        # Drop newlines
        line = line.strip()
        # Ignore comment lines
        if line[0] is not '#':
            # Chose to use "=" as a separator b/c of colons in file paths
            assert line.count('=') is 1, 'Oops. Only one "=" is allowed per line. Go fix this line in the config file:\n%s' % (line)
            parts = line.split('=')
            key = parts[0].strip()
            val = parts[1].strip()
            # Convert from strings to other data types
            if key in ['vr_ylim', 'VrPointsBins_algfail_adj']:
                val = float(val)
            if key in ['modulo_for_timeticks', 'tri_subdivisions']:
                val = int(val)
            if key in ['VrPointsBins', 'VrPointsCols', 'VrContoursBins', 'VrContoursCols', 'ContourIntervals', 'ContourCols', 'ContourTickLabels']:
                # Turn a comma-separated text list into a Python list
                val = val.split(',')
                val = [x.strip() for x in val]
                if key in ['VrPointsBins', 'VrContoursBins', 'ContourIntervals']:
                    # Convert bin values from strings to floats
                    val = [float(x) for x in val]
                if key in ['ContourIntervals', 'ContourCols']:
                    # Convert to a Python tuple
                    val = tuple(val)
            # Assign configuration items to the dictionary
            dictPlotParms[key] = val

    # Apply algorithm failure adjustment value
    temp = np.array(dictPlotParms['VrPointsBins']) + dictPlotParms['VrPointsBins_algfail_adj']
    dictPlotParms['VrPointsBins'] = temp.tolist()

    # QC user input from the config file.

    # "corediam_colmap" and "filled_contour_colmap" must be valid maplotlib colormap
    # names.
    valid_colmaps = sorted(x for x in plt.cm.datad)
    for x in ['corediam_colmap', 'filled_contour_colmap']:
        assert dictPlotParms[x] in valid_colmaps, 'Oops. The config file has an invalid entry for %s. "%s" is not a valid matplotlib colormap.' % (x, dictPlotParms[x])

    # The entries in "VrPointsCols", "VrContoursCols", and "ContourCols" must each be
    # valid matplotlib color designations.
    valid_colors = matplotlib.colors.cnames.keys()
    # Add single-letter names to the list of valid colors
    for k in matplotlib.colors.ColorConverter.colors.keys():
        valid_colors.append(k)
    for x in ['VrPointsCols', 'VrContoursCols', 'ContourCols']:
        for c in dictPlotParms[x]:
            assert c in valid_colors, 'Oops. The config file has an invalid entry for %s. "%s" is not a valid color designation.' % (x, c)

    # Enforce "tri_subdivisions" LTE 4
    assert dictPlotParms['tri_subdivisions'] <= 4, 'Oops. The value for tri_subdivisions is too high. Typical values are 1-3.'

    # Make sure each specified bin has an associated color.
    thing1 = ['VrPointsBins', 'VrContoursBins', 'ContourIntervals']
    thing2 = ['VrPointsCols', 'VrContoursCols', 'ContourTickLabels']
    for a, b in zip(thing1, thing2):
        assert len(dictPlotParms[a]) is len(dictPlotParms[b]), 'Oops. %s has %s entries but %s has %s. They should be equal.' % (a, len(dictPlotParms[a]), b, len(dictPlotParms[b]))

    # The contour intervals for filled plots are slightly different. Proper plotting requires
    # one more interval than there are colors.
    assert len(dictPlotParms['ContourIntervals']) is len(dictPlotParms['ContourCols']) + 1, 'Oops. VrContoursBins has %s entries but VrContoursCols has %s. They should be equal.' % (len(dictPlotParms['ContourIntervals']), len(dictPlotParms['ContourCols']))

    return dictPlotParms


logObj= setUpTheLogger()
dVars = setVars('config.txt')
rawdata = load_file(dVars['filename'], logObj)
