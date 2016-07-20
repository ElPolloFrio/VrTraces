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
import scipy.interpolate

def setUpTheLogger():
    # Someone set up us the logger? Ha ha ha.

    # Create the logger object.
    logger = logging.getLogger('vrtraces')
    logger.setLevel(logging.INFO)

    # Create a console handler.
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Create a formatter and add it to the handler.
    logformatter = logging.Formatter('%(asctime)s - %(name)-8s - %(levelname)s - %(message)s')
    ch.setFormatter(logformatter)
    
    # Add the handler to the logger.
    logger.addHandler(ch)

    # Log some headers.
    logger.info('----------------------------------')
    logger.info('------- Creating Vr Traces -------')
    logger.info('----------------------------------')
    
    return logger


def load_file(filename, lumberjack):
    # Load in the file contents as a numpy array.
    # Skip headers.
    data = np.loadtxt(filename, dtype = 'string', delimiter = ',', skiprows = 1)
    lumberjack.info('Loaded %s', filename)

    return data


def set_vars(configfile):
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
    # tz_units: (string) Time zone designation to use on the x-axis for all
    #   time series plots. This should be the same time zone used in the
    #   spreadsheet. Typically, that would be UTC since radar data sets are
    #   in UTC. However, some forecasters might prefer to keep their
    #   spreadsheets in local time instead.
    #
    # corediam_units: (string) Core diameter units to use on the y-axis for
    #   core diameter plots. 
    #
    # height_units: (string) Height units to use on the y-axis for the Vr plots.
    #
    # corediam_colmap: (string, must be a valid matplotlib colormap name)
    #   A matplotlib colormap from which to select colors to use for the core
    #   diameter plot. Good options include 'hsv', 'gist_rainbow', 'Spectral',
    #   and 'Paired'. Note that these names are case-sensitive.
    #
    # shear_colmap: (string, must be a valid matplotlib colormap name)
    #   A matplotlib colormap from which to select colors to use for the shear
    #   plot.
    #
    # filled_contour_colmap: (string, must be a valid matplotlib colormap name)
    #   A matplotlib colormap to use on the filled contour plots. Good options
    #   include 'hot_r'.
    #
    # RayWolf_colmap: (string, must be a valid matplotlib colormap name)
    #   A matplotlib colormap from which to select colors to use for the Ray
    #   Wolfe plot. 
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
    # Vr_algfail_adj: (decimal) The contoured plot with raw numbers nearly
    #   always has problems because the data set is too sparse. The contours
    #   for the gridded interpolation sometimes look incorrect because
    #   of the interpolation. Sometimes, the contours look better by contouring
    #   at 14.9 instead of 15.0, for example. Set this value to -0.1 (use other
    #   values at your own risk) to produce a plot with contours at 
    #   VrContoursBins + Vr_algfail_adj. It may or may not improve the
    #   appearance of the contours with the gridded interpolation. Set this value
    #   to 0 to plot contours exactly at the values in VrContoursBins.
    #
    # tri_subdivisions: (integer) Another interpolation method is the
    #   triangular interpolation. This plot is produced by default. Depending
    #   on the data set, the contours might look better after using a
    #   triangular refiner. Set this to the number of subdivisions for each
    #   triangle. Typical values are 1-3. This script will disallow anything
    #   above 4.
    #
    # ContourIntervals: (comma-separated list of integers or decimal numbers) Bin
    #   values for the gridded and triangulated contoured Vr plots. Note that the
    #   plotting may fail if the lowest specified contour is not found in the data.
    #
    # ContourCols: (comma-separated list of strings which must be valid matplotlib
    #   color designations) Colors associated with the bin values defined above. This
    #   is used on the gridded and triangulated contoured Vr plots.
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
            if key in ['Vr_algfail_adj']:
                val = float(val)
            if key in ['modulo_for_timeticks', 'tri_subdivisions']:
                val = int(val)
            if key in ['VrPointsBins', 'VrPointsCols', 'ContourIntervals', 'ContourCols']:
                # Turn a comma-separated text list into a Python list
                val = val.split(',')
                val = [x.strip() for x in val]
                if key == 'ContourIntervals':
                    # Generate tick labels in case they are needed for alg_adj
                    dictPlotParms['ContourTickLabels'] = val
                if key in ['VrPointsBins', 'ContourIntervals']:
                    # Convert bin values from strings to floats
                    val = [float(x) for x in val]
                if key in ['ContourIntervals', 'ContourCols']:
                    # Convert to a Python tuple
                    val = tuple(val)
            # Assign configuration items to the dictionary
            dictPlotParms[key] = val

    # QC user input from the config file.

    # 'corediam_colmap' and 'filled_contour_colmap' must be valid maplotlib colormap
    # names.
    valid_colmaps = sorted(x for x in plt.cm.datad)
    for x in ['corediam_colmap', 'filled_contour_colmap', 'RayWolf_colmap', 'shear_colmap']:
        assert dictPlotParms[x] in valid_colmaps, 'Oops. The config file has an invalid entry for %s. "%s" is not a valid matplotlib colormap.' % (x, dictPlotParms[x])

    # The entries in 'VrPointsCols', 'VrContoursCols', and 'ContourCols' must each be
    # valid matplotlib color designations.
    valid_colors = matplotlib.colors.cnames.keys()
    # Add single-letter names to the list of valid colors
    for k in matplotlib.colors.ColorConverter.colors.keys():
        valid_colors.append(k)
    for x in ['VrPointsCols', 'ContourCols']:
        for c in dictPlotParms[x]:
            if '#' in c:
                # Confirm that it's a valid hex color code
                assert c[0] is '#', 'Oops. A hex color code must start with "#".'
                assert len(c) is 7, 'Oops. A hex color code is usually 7 characters long including the "#". This one has %s characters.' % (len(c))
                # FutureDev: check that only alphanumeric chars are present.
            else:
                # Check the color name
                assert c in valid_colors, 'Oops. The config file has an invalid entry for %s. "%s" is not a valid color designation.' % (x, c)

    # Enforce 'tri_subdivisions' <= 4
    assert dictPlotParms['tri_subdivisions'] <= 4, 'Oops. The value for tri_subdivisions is too high. Typical values are 1-3.'

    # Make sure each specified bin has an associated color.
    thing1 = ['VrPointsBins']
    thing2 = ['VrPointsCols']
    for a, b in zip(thing1, thing2):
        assert len(dictPlotParms[a]) is len(dictPlotParms[b]), 'Oops. %s has %s entries but %s has %s. They should be equal.' % (a, len(dictPlotParms[a]), b, len(dictPlotParms[b]))

    # The contour intervals for filled plots are slightly different. Proper plotting requires
    # one more interval than there are colors.
    assert len(dictPlotParms['ContourIntervals']) is len(dictPlotParms['ContourCols']) + 1, 'Oops. ContourIntervals has %s entries but ContourCols has %s. There should be one more interval specified than there are colors specified. See the sample config files.' % (len(dictPlotParms['ContourIntervals']), len(dictPlotParms['ContourCols']))

    return dictPlotParms


def process_data(data, dictPlotParms, lumberjack):
    # This function processes a raw CSV data set in an expected format and creates
    # Python objects suitable for plotting. 
    #
    # Input:
    # data: raw CSV data set as returned by load_file
    # dictPlotParms: dictionary of plotting parameters as returned by set_vars
    # lumberjack: the logger
    #
    # Output:
    # dictPlotThis: a dictionary containing data sets suitable for plotting

    dictPlotThis = {}
    
    # Figure out how many elevation angles are present.
    elevstr = np.unique(data[:,0]).tolist()[:]
    numElevAngles = len(elevstr)
    lumberjack.info('There are %s elevation angles', numElevAngles)

    # Values in the time column ought to be zero-padded to a width of 4
    # characters. Sometimes, working with CSV files in Excel causes the
    # leading zeros to be silently dropped. See the README file for
    # instructions on how to preserve leading zeros in Excel.
    rawtime = data[:,1].tolist()
    rawtime = ['%04i' % (int(x)) for x in rawtime]

    # Figure out how many time steps are present.
    #timestr = np.unique(data[:,1]).tolist()[:]
    timestr = np.unique(rawtime)
    numTimeTicks = len(timestr)
    lumberjack.info('There are %s time steps', numTimeTicks)
    # Construct array 't' to plot time, but use 'timestr' for labels
    t = np.arange(1, numTimeTicks + 1)
    # Don't store timestr yet because there may be adjustments needed below.
    dictPlotThis['t'] = t

    # Convert data from strings to floats and turn any missing
    # values ('M') into np.nan
    elev = np.array([float(a) for a in elevstr])
    # 'elevstr' was sorted by string values, so 10.0 comes before
    # 2.4 deg, for example. Sort it again now that it's numerical.
    elev.sort()
    dictPlotThis['elev'] = elev

    # Build an array of data values, converting from string to float and
    # turning any missing values ('M') into np.nan.
    #
    # Note: if load_file is changed to not skip the header row, then
    # this line must be changed to data[1:,]. 
    data_arr = data[0:,].copy()
    data = np.empty(data_arr.shape)
    
    for a in np.arange(0, len(data)):
        for b in np.arange(0, len(data_arr[a])):
            if (data_arr[a][b] == 'M'):
                data[a][b] = np.nan
            else:
                data[a][b] = float(data_arr[a][b])

    # Vr traces data preparation.
    # Every data point has a time, height, and value.
    lumberjack.info('Starting Vr traces data preparation')
    # x: time ticks for each elevation angle
    x = np.nan * np.zeros((numElevAngles, numTimeTicks))
    # y: height for each elevation angle at each time
    y = np.nan * np.zeros((numElevAngles, numTimeTicks))
    # z: Vr value for each elevation angle at each time
    z = np.nan * np.zeros((numElevAngles, numTimeTicks))
    
    for a in np.arange(0, len(elev)):
        indices = np.where(data[:,0] == elev[a])
        temp_vr = data[indices][:,1:4]
        if len(temp_vr) < numTimeTicks:
            shorttime = data_arr[indices][:,1]
            # Since every elevation angle isn't necessarily present at every
            # time step, start with the array of nans and fill it in with known data.
            newvr = np.nan * np.zeros((numTimeTicks, 3))
            # Set the time ticks, otherwise x[a] = temp_vr[:,0] below might have
            # nans for the time.
            newvr[:,0] = [float(d) for d in timestr]
            for b in np.arange(0, len(timestr)):
                for c in np.arange(0, len(shorttime)):
                    if timestr[b] == shorttime[c]:
                        newvr[b] = temp_vr[c]
                        
            temp_vr = newvr
            
        x[a] = temp_vr[:,0]
        y[a] = temp_vr[:,1]
        z[a] = temp_vr[:,2]

    dictPlotThis['x'] = x
    dictPlotThis['y'] = y
    dictPlotThis['z'] = z        

    # Determine vr_ylim from the data set. Round to the nearest increment ('val')
    # that's greater than ymax.
    val = 0.5
    ymax = np.nanmax(y)
    if ymax % val == 0:
        ymax = ymax + val
    vr_ylim = val * np.ceil(ymax/val)
    dictPlotThis['vr_ylim'] = vr_ylim
    lumberjack.info('Upper bound for Vr plots shall be %s' % (vr_ylim))
    
    lumberjack.info('Finished Vr traces data preparation')

    # Some plotting functions require versions of y and z without nans
    y2 = np.nan_to_num(y)
    z2 = np.nan_to_num(z)

    dictPlotThis['y2'] = y2
    dictPlotThis['z2'] = z2
    
    # Make a 2D time array for plotting
    t2D = np.array([])
    for a in np.arange(0, numElevAngles):
        t2D = np.append(t2D, t)
    t2D.shape = (len(x), numTimeTicks)
    dictPlotThis['t2D'] = t2D

    # Interpolate Vr values in time (x) and height (y) to improve the
    # appearance of contours.

    # Define the interpolation grid.
    # gridx: 0 to numTimeTicks by half-tick increments, plus a margin
    # gridy: y-min to y-max by 0.25 increments, plus a margin

    # Modify these parameters to change the amount of interpolation.
    xspan = 0.5
    yspan = 0.25
    xmin = 0.5
    xmax = numTimeTicks + 1
    ymin = y2.min()
    ymax = y2.max() + 0.5

    gridx, gridy = np.mgrid[xmin:xmax:xspan, ymin:ymax:yspan]
    lumberjack.info('Defined the Vr interpolation grid')
    
    # Vr is a function of x and y. Only the values of Vr(t, hght) are known at
    # the data points provided in the spreadsheet. Need to get those data points
    # in the following form:
    # [x-coord, y-coord]
    # The values similarly must be in a flattened form to match the x,y values
    points = np.vstack([t2D.transpose().flatten(), y2.transpose().flatten()]).transpose()
    values = z2.transpose().flatten()
    # For some unknown reason, 'cubic' interpolation hangs after upgrading to Python 2.7,
    # scipy 0.12, and numpy 1.7. Problem fixed after upgrading to scipy 0.16.1 and
    # numpy 1.10.2.
    np_ver = np.__version__.split('.')
    sp_ver = scipy.__version__.split('.')
    if ((np_ver[0] is '1' and np_ver[1] is '7') and (sp_ver[0] is '0' and sp_ver[1] is '12')):
        method = 'linear'
        lumberjack.warning('Using linear interpolation due to versions of numpy and scipy')
    else:
        method = 'cubic'
    gridz = scipy.interpolate.griddata(points, values, (gridx, gridy), method = method)
    lumberjack.info('Calculated Vr grid-z')


    def ymxb(xpt1, ypt1, xpt2, ypt2):
        # A helper function. Given points x1,y1 and x2,y2 assumed to be along the
        # same line, this function returns m and b from y = mx + b.
        #
        # m = (y2 - y1)/(x2 - x1)
        # b = y1 - mx1

        m = (ypt2 - ypt1)/(xpt2 - xpt1)
        b = ypt1 - m*xpt1

        return m, b

    # Mask Vr gridz values where they fall outside the bounds of the input data set. This
    # is to mitigate an interpolation artifact where tight gradients are shown between
    # y = 0 and the lowest height for which data is present. It also addresses the issue
    # of incorrect interpolation above the max y-values.
    # min_y and max_y are 1D vectors containing the min/max values
    # of y at each time step.
    ymask = np.ma.make_mask_none(gridz.shape)
    min_y = np.nanmin(y, axis = 0)
    max_y = np.nanmax(y, axis = 0)
    xstep = t2D[0]
    epsilon = 0.01
    min_y = min_y - epsilon
    max_y = max_y + epsilon
    # The interpolation adds intermediate points between time steps. In order to mask
    # intermediate values which fall outside the bounds of the input data set, one must
    # first calculate appropriate threshold values at the intermediate points.
    xstep_interp = []
    min_y_interp = []
    max_y_interp = []
    for i in range(len(xstep)-1):
        diff = xstep[i + 1] - xstep[i]
        # 1/xspan yields the number of intervals between xstep[i+1] and xstep[i].
        # The number of intermediate points is one less than the number of intervals.
        num_intermed_pts = int(1.0/xspan) - 1
        xstep_interp.append(xstep[i])
        # Rely on good ol' y = mx + b. Calculate the slope and intercept for the
        # line formed by the max/min_y value at this time step and the max/min_y
        # value at the next time step. Then use these values to calculate the
        # interpolated values of max/min_y along the line at the interpolated
        # x-values.
        min_m, min_b = ymxb(xstep[i], min_y[i], xstep[i+1], min_y[i+1])
        max_m, max_b = ymxb(xstep[i], max_y[i], xstep[i+1], max_y[i+1])

        min_y_interp.append(min_y[i])
        max_y_interp.append(max_y[i])

        # Recall that range(a,b) loops up to but not including b.
        for j in range(1, num_intermed_pts + 1):
            xstep_intermed = xstep[i] + j*xspan
            xstep_interp.append(xstep_intermed)
            min_y_interp.append((min_m*xstep_intermed) + min_b)
            max_y_interp.append((max_m*xstep_intermed) + max_b)

    xstep_interp.append(xstep[-1])
    min_y_interp.append(min_y[-1])
    max_y_interp.append(max_y[-1])

    # At each time step, determine which points fall outside the bounds of the
    # input data set and mask them.
    for ind, val in enumerate(xstep_interp):
        #xind = np.where(gridx == val)
        r, c = np.where(gridx == val)
        if np.isnan(min_y_interp[ind]):
            # Mask all values if all data is missing at this time step
            #ymask[xind] = True
            ymask[r, c] = True
        else:
            # Mask values which fall outside the bounds of the input data set.
            #lo_ind = np.where(gridy[xind] <= min_y_interp[ind])
            #hi_ind = np.where(gridy[xind] >= max_y_interp[ind])

            lo_ind = np.where(gridy[r, c] <= min_y_interp[ind])
            hi_ind = np.where(gridy[r, c] >= max_y_interp[ind])
            
            # Fails because the value is not stored into the array. Suspect this
            # has something to do with advanced indexing and the difference
            # between a copy and a view. 
            #ymask[xind][lo_ind] = True
            #ymask[xind][hi_ind] = True

            #print min_y_interp[ind], gridy[r, c][lo_ind]

    gridz = np.ma.masked_where(ymask, gridz)
    #gridz = np.ma.masked_array(gridz, mask = ymask)

##    # Mask Vr gridz values where they are less than the input data. This is to
##    # mitigate an interpolation artifact where tight gradients are shown between
##    # y = 0 and the lowest height for which there is data present. 
##    thresh = 0.5 * np.nanmin(z)
##    gridz = np.ma.masked_where(gridz < thresh, gridz)

    dictPlotThis['gridx'] = gridx
    dictPlotThis['gridy'] = gridy
    dictPlotThis['gridz'] = gridz

    # The gridded interpolation sometimes doesn't do a good enough job, mostly because
    # these data sets tend to be too sparse. Try triangular interpolation instead.
    triang = tri.Triangulation(t2D.flatten(), y2.flatten())
    lumberjack.info('Calculated Vr triangular interpolation')
    
    # Mask triangles from an interpolation artifact. If the y-midpoint of any given
    # triangle is less than the lowest y-value in the data set, mask it.
    #
    # The lengths of ymid and xmid are the number of triangles. numtri = triang.triangles.shape[0].
    ymid = y2.flatten()[triang.triangles].mean(axis = 1)
    xmid = t2D.flatten()[triang.triangles].mean(axis = 1)
    ymask = np.where(ymid < np.nanmin(y), True, False)

    # Mask triangles on a timestep-by-timestep basis using max/min y-values
    # at each timestep.
    #
    # The problem with this approach is that trifinder(x, y) only returns the triangle
    # indices which contain that x,y point. It does not return the triangle indices for
    # all triangles which are LTE/GTE that x,y point, which is what is needed.
    #min_y = np.nanmin(y, axis = 0)
    #max_y = np.nanmax(y, axis = 0)
    #xstep = t2D[0]
    #epsilon = 0.01
    #min_y = min_y - epsilon
    #max_y = max_y + epsilon
    ## Get the triangle indices for these points
    #trifinder = triang.get_trifinder()
    #toolo_ind = trifinder(xstep, min_y)
    #toohi_ind = trifinder(xstep, max_y)
    ## Mark the invalid triangles
    #minmask[toolo_ind] = True
    #minmask[toohi_ind] = True

    # Surely there are more elegant ways to do this. Hey future me, did you find any?
    #
    # min_y and max_y are 1D vectors containing the min/max values
    # of y at each time step.
    min_y = np.nanmin(y, axis = 0)
    max_y = np.nanmax(y, axis = 0)
    xstep = t2D[0]
    epsilon = 0.01
    min_y = min_y - epsilon
    max_y = max_y + epsilon
    # The interpolation adds intermediate points between time steps. In fact, the x-midpoint
    # values resemble the following: x, x + 1/3, x + 2/3, x + 1 and so on. In order to compare
    # triangle midpoints to max/min y-values, one must first generate the interpolated values
    # for x at time steps x + 1/3 and x + 2/3. One must also generate interpolated values for
    # y-min and y-max at those interpolated x-values. 
    xstep_interp = []
    min_y_interp = []
    max_y_interp = []
    for i in range(len(xstep)-1):
        diff = xstep[i + 1] - xstep[i]
        xstep_interp.append(xstep[i])
        xstep1 = xstep[i] + (1/3.0)*diff
        xstep2 = xstep[i] + (2/3.0)*diff
        xstep_interp.append(xstep1)
        xstep_interp.append(xstep2)
        # Rely on good 'ol y = mx + b.
        # Calculate the slope and intercept parameters for the line formed
        # by the max/min_y value at this time step and the max/min_y value
        # at the next time step. Then use these values to calculate the
        # interpolated values of max/min_y along the line at the two
        # interpolated x-values.
        min_m, min_b = ymxb(xstep[i], min_y[i], xstep[i+1], min_y[i+1])
        max_m, max_b = ymxb(xstep[i], max_y[i], xstep[i+1], max_y[i+1])
        min_y_interp.append(min_y[i])
        min_y_interp.append((min_m*xstep1) + min_b)
        min_y_interp.append((min_m*xstep2) + min_b)
        max_y_interp.append(max_y[i])
        max_y_interp.append((max_m*xstep1) + max_b)
        max_y_interp.append((max_m*xstep2) + max_b)

    xstep_interp.append(xstep[-1])
    min_y_interp.append(min_y[-1])
    max_y_interp.append(max_y[-1])
    
    # The trifinder returns triangle indices for the triangle containing
    # a given x,y point.
    trifinder = triang.get_trifinder()
    # Determine which triangles have midpoints that fall below/above
    # the valid y-vals at each time step. 
    for ind, val in enumerate(xstep_interp):
        # Get the triangles centered at this time step.
        xind = np.where(np.round(xmid, 3) == round(val, 3))
        xtri = trifinder(xmid[xind], ymid[xind])
        # Determine which triangles at this time step are centered too low
        # or too high and then mask them.
        if np.isnan(min_y_interp[ind]):
            # Mask all triangles if all data is missing at this time step.
            ymask[xind] = True
        else:
            lo_ind = np.where(ymid[xind] <= min_y_interp[ind])
            hi_ind = np.where(ymid[xind] >= max_y_interp[ind])
            invalid_ind = xtri[lo_ind]
            ymask[invalid_ind] = True
            invalid_ind = xtri[hi_ind]
            ymask[invalid_ind] = True

    # Mask triangles if they are too flat (edge artifacts)
    min_circle_ratio = .01
    flatmask = tri.TriAnalyzer(triang).get_flat_tri_mask(min_circle_ratio)

    mask = np.logical_or(ymask, flatmask)
    triang.set_mask(mask)

    # Refine the triangular grids for higher-res contouring
    subdiv = dictPlotParms['tri_subdivisions']
    refiner = tri.UniformTriRefiner(triang)
    tri_refi, z_refi = refiner.refine_field(z2.flatten(), subdiv = subdiv)

    num_tri_refi = tri_refi.triangles.shape[0]
    minmask2 = np.zeros(num_tri_refi, dtype = np.bool)

    # Mask refined triangles on a timestep-by-timestep basis using max/min y-values
    # at each timestep.
    min_y = np.nanmin(y, axis = 0)
    max_y = np.nanmax(y, axis = 0)
    xstep = t2D[0]
    epsilon = 0.01
    min_y = min_y - epsilon
    max_y = max_y + epsilon
    # Get the triangle indices for these points
    trifinder = tri_refi.get_trifinder()
    toolo_ind2 = trifinder(xstep, min_y)
    toohi_ind2 = trifinder(xstep, max_y)
    # Mark the invalid triangles
    minmask2[toolo_ind2] = True
    minmask2[toohi_ind2] = True

    # Mask refined triangles if they are too flat (edge artifacts)
    flatmask2 = tri.TriAnalyzer(tri_refi).get_flat_tri_mask(min_circle_ratio)
    
    mask2 = np.logical_or(minmask2, flatmask2)
    tri_refi.set_mask(mask2)

    dictPlotThis['triang'] = triang
    dictPlotThis['tri_refi'] = tri_refi
    dictPlotThis['z_refi'] = z_refi

    # The plots for core diameter and shear are essentially time series plots.
    #
    # Construct an array full of np.nans based on the number of elevation
    # angles (rows) and the number of time ticks (columns).
    #
    # Since every elevation angle isn't necessarily present at every time step,
    # start with the array of nans and fill it in with known data. 
    ts_vars = {
        'core diameter':
            {'col_ind': 4,
            'varname': 'corediam'},
        'shear':
            {'col_ind': 5,
            'varname': 'shear'}
    }

    for k in ts_vars.keys():
        lumberjack.info('Starting %s preparation' % (k))
        vals = np.nan * np.zeros((numElevAngles, numTimeTicks))
        
        for a in np.arange(0, len(elev)):
            indices = np.where(data[:,0] == elev[a])
            temp_vals = data[indices][:,ts_vars[k]['col_ind']]
            if len(temp_vals) < numTimeTicks:
                shorttime = data_arr[indices][:,1]
                newvals = np.nan * np.zeros((1, numTimeTicks))
                for b in np.arange(0, len(timestr)):
                    for c in np.arange(0, len(shorttime)):
                        if timestr[b] == shorttime[c]:
                            newvals[0][b] = temp_vals[c]
                temp_vals = newvals
            vals[a] = temp_vals
        
        dictPlotThis[ts_vars[k]['varname']] = vals
        lumberjack.info('Finished %s preparation' % (k))

    # If there are too many time ticks displayed, adjust the amount with
    # the modulo criteria.
    modulo_for_timeticks = dictPlotParms['modulo_for_timeticks']
    newlist = []
    for index, item in enumerate(timestr):
        if index % modulo_for_timeticks == 0:
            newlist.append(item)
        else:
            newlist.append('')
    timestr = newlist
    dictPlotThis['timestr'] = timestr

    return dictPlotThis


def make_plots(dictUserParms, dictPlotThis, lumberjack):
    # This function creates and saves a set of plots based on data from
    # a spreadsheet.
    #
    # Input:
    # dictUserParms: dictionary of plotting parameters as returned by set_vars
    # dictPlotThis: dictionary of Python objects suitable for plotting
    # lumberjack: the logger
    #
    # Output:
    # plots saved to disk

    # Generate plot labels
    mv_name = dictUserParms['mv_name']
    tz_units = dictUserParms['tz_units']
    corediam_units = dictUserParms['corediam_units']
    height_units = dictUserParms['height_units']
    
    dictPlotLabels = {
        'corediam': {
            'title': 'Core diameter for %s' % (mv_name),
            'xlabel': 'Time (%s)' % (tz_units),
            'ylabel': 'Core diameter (%s)' % (corediam_units)
        },
        'shear': {
            'title': 'Shear for %s' % (mv_name),
            'xlabel': 'Time (%s)' % (tz_units),
            'ylabel': 'Shear ' + r'$\left(V_r / d * 10^{-3} s^{-1}\right)$'
        },
        'Vr': {
            'title': r'$V_r$ trace for %s' % (mv_name),
            'xlabel': 'Time (%s)' % (tz_units),
            'ylabel': 'Height (%s)' % (height_units)
        }
    }

    lumberjack.info('Generated plot labels')

    # Shorter variable names are easier to read
    t = dictPlotThis['t']
    timestr = dictPlotThis['timestr']
    elev = dictPlotThis['elev']
    corediam = dictPlotThis['corediam']
    corediam_labels = dictPlotLabels['corediam']
    fontsize = dictUserParms['rcParamsFontSize']
    base_fontsize = 12
    
    shear = dictPlotThis['shear']
    shear_labels = dictPlotLabels['shear']
    
    t2D = dictPlotThis['t2D']
    x = dictPlotThis['x']
    y = dictPlotThis['y']
    z = dictPlotThis['z']
    y2 = dictPlotThis['y2']
    z2 = dictPlotThis['z2']
    vr_ylim = dictPlotThis['vr_ylim']
    vr_labels = dictPlotLabels['Vr']

    points_bins = dictUserParms['VrPointsBins']
    points_cols = dictUserParms['VrPointsCols']
    # Need a later version of numpy for this.
    #binIndices = np.digitize(z2.flatten(), bins, right = True)
    points_binIndices = np.digitize(z2.flatten(), points_bins)
    points_binIndices = np.reshape(points_binIndices, z2.shape)

    gridx = dictPlotThis['gridx']
    gridy = dictPlotThis['gridy']
    gridz = dictPlotThis['gridz']

    ContourIntervals = dictUserParms['ContourIntervals']
    ContourCols = dictUserParms['ContourCols']

    # Handle the algorithm failure parameter, if specified
    ContourAlgFailure = False
    adj_factor = dictUserParms['Vr_algfail_adj']
    if adj_factor != 0:
        ContourAlgFailure = True
        temp = np.array(ContourIntervals) + adj_factor
        ContourIntervals_AlgFail = tuple(temp.tolist())
        ContourTickLabels = dictUserParms['ContourTickLabels']

    triang = dictPlotThis['triang']
    tri_refi = dictPlotThis['tri_refi']
    z_refi = dictPlotThis['z_refi']
    
    # Marker set for the Vr plots
    markers = ['+', 'o', '^', 's', 'x', 'd', '>', 'p', '1', '8', '<', 'D', 'v', '*', 'H', 'h', '4']

    # Construct the base filename for each saved plot.
    base_fname = os.path.splitext(os.path.basename(dPlotVars['filename']))[0]

    figsize = (10, 7.5)

    # Offset value for plotting Vr values
    y_offset = 0.025

    # Helper dictionary for consistent logging messages
    dictLogMsg = {
        'plot_corediam': 'core diameter',
        'plot_shear': 'shear',
        'plot_VrPoints': 'Vr data points alone',
        'plot_VrPoints_NumOnly': 'Vr data points as numbers without markers',
        'plot_VrContours_Raw': 'Vr data points with contours based on raw input values',
        'plot_VrContours_GridInterp': 'Vr data points with contours based on gridded interpolation',
        'plot_VrContours_TriangInterp': {
            'unrefined': 'Vr data points with contours based on triangular interpolation',
            'refined': 'Vr data points with contours based on triangular interpolation, using a refiner'
        },
        'plot_VrContours_Filled_GridInterp': 'Vr filled contours with gridded interpolation',
        'plot_VrContours_Filled_TriangInterp': {
            'unrefined': 'Vr filled contours with triangular interpolation',
            'refined': 'Vr filled contours with triangular interpolation, using a refiner'
        },
        'plot_Vr_RayWolf': 'Ray Wolf\'s scatter plot',
        'plot_Vr_Corediam': 'Vr plot with core diameter'
    }

    def is_all_missing(data, name):
        # Helper function. Returns 'True' if all data is missing.
        # data: (np array) data to check
        # name: (string) plot text to use in the logger message

        nancheck = False

        nanind = np.where(np.isnan(data))[0]
        if len(nanind) is data.size:
            nancheck = True
            lumberjack.warning('No non-missing data points found for %s. Skipping this plot.' % (name))

        return nancheck
    

    def plot_corediam():
        # Rely on variable scope

        # Don't bother plotting if all data is missing.
        if is_all_missing(corediam, dictLogMsg['plot_corediam']):
            return None
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        
        # Specify the line colors from a colormap.
        #
        # Number of curves to plot.
        ncurves = len(elev)
        # 0-indexed list of line numbers used to take samples from the colormap.
        values = range(ncurves)
        colmap = plt.get_cmap(dictUserParms['corediam_colmap'])
        cNorm = colors.Normalize(vmin = 0, vmax = values[-1])
        scalarMap = cmx.ScalarMappable(norm = cNorm, cmap = colmap)
        #print scalarMap.get_clim()

        for a in np.arange(0, len(elev)):
            colorVal = scalarMap.to_rgba(values[a])
            if elev[a] <= 1.4:
                ax.plot(t, corediam[a], '-', linewidth = 2, color = colorVal, label = '%.1f deg' % elev[a])
            elif elev[a] <= 5.0:
                ax.plot(t, corediam[a], '--', linewidth = 2, color = colorVal, label = '%.1f deg' % elev[a])
            else:
                ax.plot(t, corediam[a], ':', linewidth = 2, color = colorVal, label = '%.1f deg' % elev[a])

        ## Alternate method to automatically pick line colors, but it is not always very readable
        ## and duplicates sometimes occur. 
        #for a in np.arange(0, len(elev)):
        #    if elev[a] <= 1.4:
        #        ax.plot(t, corediam[a], '-', linewidth = 2, label = '%.1f deg' % elev[a])
        #    else:
        #        ax.plot(t, corediam[a], '--', linewidth = 2, label = '%.1f deg' % elev[a])

        ax.yaxis.grid(True)
        # LFP stands for "legend font properties"
        LFP = matplotlib.font_manager.FontProperties(size = base_fontsize)
        ax.legend(loc = 2, scatterpoints = 1, prop = LFP)    
        plt.title(corediam_labels['title'])
        plt.xlabel(corediam_labels['xlabel'])
        plt.ylabel(corediam_labels['ylabel'])
        plt.xticks(t, timestr, rotation = '60')
        plt.rcParams['font.size'] = fontsize
        plt.tight_layout()

        fname = '{}_{}.png'.format(base_fname, 'corediam')
        fig.set_size_inches(figsize)
        plt.savefig(fname)

        return None


    def plot_shear():
        # Rely on variable scope

        # Don't bother plotting if all data is missing.
        if is_all_missing(shear, dictLogMsg['plot_shear']):
            return None

        # Specify the line colors from a colormap.
        #
        # Number of curves to plot.
        ncurves = len(elev)
        # 0-indexed list of line numbers used to take samples from the colormap.
        values = range(ncurves) 
        colmap = plt.get_cmap(dictUserParms['shear_colmap'])
        cNorm = colors.Normalize(vmin = 0, vmax = values[-1])
        scalarMap = cmx.ScalarMappable(norm = cNorm, cmap = colmap)
        #print scalarMap.get_clim()

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        for a in np.arange(0, len(elev)):
            colorVal = scalarMap.to_rgba(values[a])
            ax.plot(t, shear[a], '-', linewidth = 2, color = colorVal, label = '%.1f deg' % elev[a])
        ax.yaxis.grid(True)
        # LFP stands for "legend font properties"
        LFP = matplotlib.font_manager.FontProperties(size = base_fontsize)
        ax.legend(loc = 0, scatterpoints = 1, prop = LFP)
        plt.title(shear_labels['title'])
        plt.xlabel(shear_labels['xlabel'])
        plt.ylabel(shear_labels['ylabel'])
        plt.xticks(t, timestr, rotation = '60')
        plt.tight_layout()

        fname = '{}_{}.png'.format(base_fname, 'shear')
        fig.set_size_inches(figsize)
        plt.savefig(fname)

        return None


    def plot_VrPoints():
        # Vr data points alone
        # Rely on variable scope

        # Don't bother plotting if all data is missing.
        if is_all_missing(z, dictLogMsg['plot_VrPoints']):
            return None
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        
        for a in np.arange(0, len(x)):
            plt.scatter(t2D[a,:], y[a,:], c = 'k', marker = markers[a], label = '%.1f deg' % elev[a])
        plt.title(vr_labels['title'])
        plt.xlabel(vr_labels['xlabel'])
        plt.ylabel(vr_labels['ylabel'])
        plt.xticks(t, timestr, rotation = '60')
        ax.set_ylim(0, vr_ylim)
        ax.set_xlim(-0.5, len(t)+3)
        # LFP stands for "legend font properties"
        LFP = matplotlib.font_manager.FontProperties(size = base_fontsize - 2)
        ax.legend(loc = 0, scatterpoints = 1, prop = LFP)

        # Plot the numbers
        for looper in np.arange(0,len(x)):
            for count in range(len(x[looper])):
                if z2[looper][count] != 0:
                    # Format the number to get rid of trailing zeros
                    plt.text(t2D[looper][count], y2[looper][count] + y_offset, '%.0f' % z2[looper][count], fontsize = base_fontsize + 6)
        plt.rcParams['font.size'] = fontsize
        plt.tight_layout()

        fname = '{}_{}.png'.format(base_fname, 'VrPoints')
        fig.set_size_inches(figsize)
        plt.savefig(fname)

        return None


    def plot_VrPoints_NumOnly():
        # Vr data points alone as color-coded numbers without markers
        # Rely on variable scope

        # Don't bother plotting if all data is missing.
        if is_all_missing(z, dictLogMsg['plot_VrPoints_NumOnly']):
            return None
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        
        plt.title(vr_labels['title'])
        plt.xlabel(vr_labels['xlabel'])
        plt.ylabel(vr_labels['ylabel'])
        plt.xticks(t, timestr, rotation = '60')
        ax.set_ylim(0, vr_ylim)
        ax.set_xlim(-0.5, len(t)+3)
        # LFP stands for "legend font properties"
        LFP = matplotlib.font_manager.FontProperties(size = base_fontsize + 2)
        ax.legend(loc = 0, scatterpoints = 1, prop = LFP)

        # Plot the numbers
        for looper in np.arange(0,len(x)):
            for count in range(len(x[looper])):
                if z2[looper][count] != 0:
                    # format the number to get rid of trailing zeros
                    plt.text(t2D[looper][count], y2[looper][count], '%.0f' % z2[looper][count], color = points_cols[points_binIndices[looper][count]], fontsize = base_fontsize + 6)
        plt.rcParams['font.size'] = fontsize
        plt.tight_layout()

        fname = '{}_{}.png'.format(base_fname, 'VrPoints_ColorNum')
        fig.set_size_inches(figsize)
        plt.savefig(fname)

        return None


    def plot_VrContours_Raw():
        # Vr data points with contours based on raw input values
        # Rely on variable scope
        
        # NEVER USE THIS PLOT. The data set is nearly always too sparse for the
        # contouring algorithm to handle properly.

        # Don't bother plotting if all data is missing.
        if is_all_missing(z, dictLogMsg['plot_VrContours_Raw']):
            return None
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        
        # The line below fails if the lowest contour specified isn't found in the data set.
        # FutureDev: check for this case and adjust the bin values and colors as needed
        #CS = ax.contour(t2D, y2, z2, VrContoursBins, linewidths = 3, colors = VrContoursCols, corner_mask = True)
        CS = ax.contour(t2D, y2, z2, list(ContourIntervals)[:-1], linewidths = 3, colors = ContourCols, corner_mask = True)
        plt.clabel(CS, inline = 1, fontsize = base_fontsize + 3, fmt = '%.0f')

        for a in np.arange(0, len(x)):
            plt.scatter(t2D[a,:], y[a,:], c = 'k', marker = markers[a], label = '%.1f deg' % elev[a])
        plt.title(vr_labels['title'])
        plt.xlabel(vr_labels['xlabel'])
        plt.ylabel(vr_labels['ylabel'])
        plt.xticks(t, timestr, rotation = '60')
        ax.set_ylim(0, vr_ylim)
        ax.set_xlim(-0.5, len(t)+3)
        # LFP stands for "legend font properties"
        LFP = matplotlib.font_manager.FontProperties(size = base_fontsize)
        ax.legend(loc = 0, scatterpoints = 1, prop = LFP)
        # Code to plot the numbers as a troubleshooting aid. Turns out that Ron likes
        # this, so include it in the final image. 
        for looper in np.arange(0,len(x)):
            for count in range(len(x[looper])):
                if z2[looper][count] != 0:
                    # Format the number to get rid of trailing zeros.
                    plt.text(t2D[looper][count], y2[looper][count] + y_offset, '%.0f' % z2[looper][count], fontsize = base_fontsize + 6)
        plt.rcParams['font.size'] = fontsize
        plt.tight_layout()

        fname = '{}_{}.png'.format(base_fname, 'VrContours_Raw')
        fig.set_size_inches(figsize)
        plt.savefig(fname)

        return None


    def plot_VrContours_GridInterp():
        # Vr points with contours based on gridded interpolation.
        # Rely on variable scope

        # Don't bother plotting if all data is missing.
        if is_all_missing(z, dictLogMsg['plot_VrContours_GridInterp']):
            return None
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        if ContourAlgFailure:
            ConInt = ContourIntervals_AlgFail
            suffix = '_AlgFail'
            lumberjack.info('Applied algorithm failure adjustment factor')
        else:
            suffix = ''
            ConInt = ContourIntervals
        
        CS = plt.contour(gridx, gridy, gridz, ConInt, linewidths = 3, colors = ContourCols, corner_mask = True)

        if ContourAlgFailure:
            # Plot at val+adj but label it as val because of problems with the
            # contouring algorithm. 
            fmt = {}
            lbls = ContourTickLabels
            for l, s in zip(CS.levels, lbls):
                fmt[l] = s
            plt.clabel(CS, CS.levels, inline = True, fmt = fmt, fontsize = base_fontsize + 3)
            ## Plot the grid used for interpolation as a troubleshooting guide.
            #plt.scatter(gridx, gridy)
        else:
            plt.clabel(CS, inline = 1, fontsize = base_fontsize + 3, fmt = '%.0f')

        # Scatterplot of the data points only, not the interpolation points.
        for a in np.arange(0, len(x)):
            plt.scatter(t2D[a,:], y[a,:], c = 'k', marker = markers[a], label = '%.1f deg' % elev[a])
        # Code to plot the numbers as a troubleshooting aid. Turns out Ron likes
        # this, so include it in the final image.
        for looper in np.arange(0,len(x)):
            for count in range(len(x[looper])):
                if z2[looper][count] != 0:
                    # Format the number to get rid of trailing zeros
                    plt.text(t2D[looper][count], y2[looper][count] + y_offset, '%.0f' % z2[looper][count], fontsize = base_fontsize + 6)
        plt.ylim(0,vr_ylim)
        ax.set_xlim(-0.5, len(t)+3)
        plt.title(vr_labels['title'])
        plt.xlabel(vr_labels['xlabel'])
        plt.ylabel(vr_labels['ylabel'])
        plt.xticks(t, timestr, rotation = '60')
        ax.set_ylim(0, vr_ylim)
        # LFP stands for "legend font properties"
        LFP = matplotlib.font_manager.FontProperties(size = base_fontsize - 2)
        ax.legend(loc = 0, scatterpoints = 1, prop = LFP)
        plt.rcParams['font.size'] = fontsize
        plt.tight_layout()

        fname = '{}_{}{}.png'.format(base_fname, 'VrContours_InterpGrid', suffix)
        fig.set_size_inches(figsize)
        plt.savefig(fname)

        return None


    def plot_VrContours_TriangInterp(refine = False):
        # Vr points with contours based on triangular interpolation
        # Rely on variable scope

        if refine:
            triangles = tri_refi
            zvals = z_refi
            suffix = '_TriRefi'
            key = 'refined'
        else:
            triangles = triang
            zvals = z2.flatten()
            suffix = ''
            key = 'unrefined'

        if ContourAlgFailure:
            ConInt = ContourIntervals_AlgFail
            suffix_alg = '_AlgFail'
            lumberjack.info('Applied algorithm failure adjustment factor')
        else:
            suffix_alg = ''
            ConInt = ContourIntervals

        # Don't bother plotting if all data is missing.
        if is_all_missing(z, dictLogMsg['plot_VrContours_TriangInterp'][key]):
            return None

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        #CS = ax.tricontour(triangles, zvals, VrContoursBins, linewidths = 3, colors = VrContoursCols, corner_mask = True)
        CS = ax.tricontour(triangles, zvals, ConInt, linewidths = 3, colors = ContourCols, corner_mask = True)

        ## Plot triangles as a troubleshooting aid
        #plt.triplot(triangles, 'k-')

        ## Plot triangle midpoints as a troubleshooting aid
        #ymid = y2.flatten()[triang.triangles].mean(axis = 1)
        #xmid = t2D.flatten()[triang.triangles].mean(axis = 1)
        #plt.scatter(xmid, ymid, marker = 'o', color = 'red', s = 5)

        if ContourAlgFailure:
            # Plot at val+adj but label it as val because of problems with the
            # contouring algorithm. 
            fmt = {}
            lbls = ContourTickLabels
            for l, s in zip(CS.levels, lbls):
                fmt[l] = s
            plt.clabel(CS, CS.levels, inline = True, fmt = fmt, fontsize = base_fontsize + 3)
            ## Plot the grid used for interpolation as a troubleshooting guide.
            #plt.triplot(triangles, 'k-')
        else:
            plt.clabel(CS, inline = 1, fontsize = base_fontsize + 3, fmt = '%.0f')

        for a in np.arange(0, len(x)):
            plt.scatter(t2D[a,:], y[a,:], c = 'k', marker = markers[a], label = '%.1f deg' % elev[a])
        plt.title(vr_labels['title'])
        plt.xlabel(vr_labels['xlabel'])
        plt.ylabel(vr_labels['ylabel'])
        plt.xticks(t, timestr, rotation = '60')
        ax.set_ylim(0, vr_ylim)
        ax.set_xlim(-0.5, len(t)+3)
        # LFP stands for "legend font properties"
        LFP = matplotlib.font_manager.FontProperties(size = base_fontsize - 2)
        ax.legend(loc = 0, scatterpoints = 1, prop = LFP)
        # Code to plot the numbers as a troubleshooting aid. Turns out that Ron likes
        # this, so include it in the final image. 
        for looper in np.arange(0,len(x)):
            for count in range(len(x[looper])):
                if z2[looper][count] != 0:
                    # Format the number to get rid of trailing zeros.
                    plt.text(t2D[looper][count], y2[looper][count] + y_offset, '%.0f' % z2[looper][count], fontsize = base_fontsize + 6)
        plt.rcParams['font.size'] = fontsize
        plt.tight_layout()

        fname = '{}_{}{}{}.png'.format(base_fname, 'VrContours_InterpTriang', suffix, suffix_alg)
        fig.set_size_inches(figsize)
        plt.savefig(fname)

        return None


    def plot_VrContours_Filled_GridInterp():
        # Vr points with filled contours based on gridded interpolation
        # Rely on variable scope

        # Don't bother plotting if all data is missing.
        if is_all_missing(z, dictLogMsg['plot_VrContours_Filled_GridInterp']):
            return None

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        if ContourAlgFailure:
            ConInt = ContourIntervals_AlgFail
            suffix = '_AlgFail'
            lumberjack.info('Applied algorithm failure adjustment factor to Vr contours plot with gridded interpolation')            
        else:
            suffix = ''
            ConInt = ContourIntervals
        
        CSF = plt.contourf(gridx, gridy, gridz, ConInt, cmap = dictUserParms['filled_contour_colmap'], corner_mask = True)

        if ContourAlgFailure:
            # Plot at val+adj but label it as val because of problems with the
            # contouring algorithm. 
            cbar = fig.colorbar(CSF, ticks = ConInt)
            cbar.ax.set_yticklabels(ContourTickLabels)
        else:
            plt.colorbar(ax = ax)

        # Scatterplot of the data points only, not the interpolation points.
        for a in np.arange(0, len(x)):
            plt.scatter(t2D[a,:], y[a,:], c = 'k', marker = markers[a], label = '%.1f deg' % elev[a])
        ### code to plot the numbers as a troubleshooting aid.
        ##for looper in np.arange(0,len(x)):
        ##    for count in range(len(x[looper])):
        ##        if z2[looper][count] != 0:
        ##            # format the number to get rid of trailing zeros
        ##            plt.text(t2D[looper][count], y2[looper][count] + y_offset, '%.0f' % z2[looper][count])
        plt.ylim(0, vr_ylim)
        ax.set_xlim(-0.5, len(t) + 3)
        plt.title(vr_labels['title'])
        plt.xlabel(vr_labels['xlabel'])
        plt.ylabel(vr_labels['ylabel'])
        plt.xticks(t, timestr, rotation = '60')
        # LFP stands for "legend font properties"
        LFP = matplotlib.font_manager.FontProperties(size = base_fontsize - 2)
        ax.legend(loc = 0, scatterpoints = 1, prop = LFP)
        plt.rcParams['font.size'] = fontsize
        plt.tight_layout()

        fname = '{}_{}{}.png'.format(base_fname, 'VrFilled_InterpGrid', suffix)
        fig.set_size_inches(figsize)
        plt.savefig(fname)

        return None


    def plot_VrContours_Filled_TriangInterp(refine = False):
        # Vr points with filled contours based on triangular interpolation
        # Rely on variable scope

        if refine:
            triangles = tri_refi
            zvals = z_refi
            suffix = '_TriRefi'
            key = 'refined'
        else:
            triangles = triang
            zvals = z2.flatten()
            suffix = ''
            key = 'unrefined'

        if ContourAlgFailure:
            ConInt = ContourIntervals_AlgFail
            suffix_alg = '_AlgFail'
            lumberjack.info('Applied algorithm failure adjustment factor')
        else:
            suffix_alg = ''
            ConInt = ContourIntervals

        # Don't bother plotting if all data is missing.
        if is_all_missing(z, dictLogMsg['plot_VrContours_Filled_TriangInterp'][key]):
            return None

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        
        #CSF = ax.tricontourf(triangles, zvals, ContourIntervals, cmap = dictUserParms['filled_contour_colmap'], corner_mask = True)
        CSF = ax.tricontourf(triangles, zvals, ConInt, cmap = dictUserParms['filled_contour_colmap'], corner_mask = True)

        if ContourAlgFailure:
            # Plot at val+adj but label it as val because of problems with the
            # contouring algorithm. 
            cbar = fig.colorbar(CSF, ticks = ConInt)
            cbar.ax.set_yticklabels(ContourTickLabels)
        else:
            cbar = fig.colorbar(CSF)

        # Scatterplot of the data points only, not the interpolation points.
        for a in np.arange(0, len(x)):
            plt.scatter(t2D[a,:], y[a,:], c = 'k', marker = markers[a], label = '%.1f deg' % elev[a])

        ### code to plot the numbers as a troubleshooting aid.
        ##for looper in np.arange(0,len(x)):
        ##    for count in range(len(x[looper])):
        ##        if z2[looper][count] != 0:
        ##            # format the number to get rid of trailing zeros
        ##            plt.text(t2D[looper][count], y2[looper][count] + 0.025, '%.0f' % z2[looper][count])
            
        plt.ylim(0, vr_ylim)
        ax.set_xlim(-0.5, len(t) + 3)
        plt.title(vr_labels['title'])
        plt.xlabel(vr_labels['xlabel'])
        plt.ylabel(vr_labels['ylabel'])
        plt.xticks(t, timestr, rotation = '60')
        # LFP stands for "legend font properties"
        LFP = matplotlib.font_manager.FontProperties(size = base_fontsize - 2)
        ax.legend(loc = 0, scatterpoints = 1, prop = LFP)
        plt.rcParams['font.size'] = fontsize
        plt.tight_layout()

        fname = '{}_{}{}{}.png'.format(base_fname, 'VrFilled_InterpTriang', suffix, suffix_alg)
        fig.set_size_inches(figsize)
        plt.savefig(fname)
        
        return None


    def plot_Vr_RayWolf():
        # Vr points styled according to a request from Ray Wolf (SOO at WFO ???).
        # Scatter plot with the size of the marker indicating the Vr value.
        # Rely on variable scope

        # Don't bother plotting if all data is missing.
        if is_all_missing(z, dictLogMsg['plot_Vr_RayWolf']):
            return None

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        # Take a matplotlib colormap and normalize it to the data set values.
        colmap = plt.get_cmap(dictUserParms['RayWolf_colmap'])
        vmin = np.nanmin(z)
        vmax = np.nanmax(z)
        # Be clever about picking the upper and lower bounds.
        if (abs(vmax - vmin) >= 20):
            cbarstep = 5
        else:
            cbarstep = 2
        vmin = int(vmin - (vmin % cbarstep))
        if vmin < 0:
            vmin = 0
        vmax = int(vmax + cbarstep - (vmax % cbarstep))
        cbarticks = np.arange(vmin, vmax + cbarstep, cbarstep)
        # Normalize the colormap to the chosen data range.
        cNorm = colors.Normalize(vmin = vmin, vmax = vmax)
        scalarMap = cmx.ScalarMappable(norm = cNorm, cmap = colmap)

        # Must specify the radius of the marker size for each point, but must first
        # calculate those values based on the aspect of the data set with which it
        # should vary.
        # Experiment to find an expression which brings out the values and looks good.
        for a in np.arange(0, len(x)):
            colorVal = scalarMap.to_rgba(z[a,:])
            # Scatterplot fails if z[a,:] has no non-nan entries
            nanind = np.where(np.isnan(z[a,:]))[0]
            if len(nanind) is z[a,:].size:
                pass
            else:
                # Varying radius, constant color
                #plt.scatter(t2D[a,:], y[a,:], s = (z[a,:]/2)**2, c = 'k', marker = 'o', label = '%.1f deg' % elev[a])
                # Constant radius, varying color
                #plt.scatter(t2D[a,:], y[a,:], s = 35, c = colorVal, marker = 'o', label = '%.1f deg' % elev[a])
                # Varying radius, varying color
                plt.scatter(t2D[a,:], y[a,:], s = (z[a,:]/2)**2, c = colorVal, marker = 'o', label = '%.1f deg' % elev[a])

        # Since neither imshow nor contourf was used, plt.colorbar() won't work. To get a colorbar,
        # one must first create a dummy scalar mappable from which to create the colorbar.
        sm = plt.cm.ScalarMappable(cmap = colmap)
        sm.set_array([vmin, vmax])
        cbar = plt.colorbar(sm, ticks = cbarticks)
            
        plt.ylim(0, vr_ylim)
        ax.set_xlim(-0.5, len(t)+3)
        plt.title(vr_labels['title'])
        plt.xlabel(vr_labels['xlabel'])
        plt.ylabel(vr_labels['ylabel'])
        plt.xticks(t, timestr, rotation = '60')
        plt.tight_layout()

        # FutureDev: figure out how to get a legend to show the meaning of marker size. Also add transparency?

        fname = '{}_{}_RayWolf.png'.format(base_fname, 'VrPoints')
        fig.set_size_inches(figsize)
        plt.savefig(fname)

        return None


    def plot_Vr_Corediam():
        # Vr points styled in two ways: by core diameter and by Vr value. Consider it to be
        # a modification of the Ray Wolf plot above. The marker size indicates core
        # diameter and the marker color indicates Vr value.
        #
        # Rely on variable scope.

        # Don't bother plotting if all data is missing. This plot requires both corediam and z.
        if is_all_missing(corediam, dictLogMsg['plot_Vr_Corediam']):
            return None

        if is_all_missing(z, dictLogMsg['plot_Vr_Corediam']):
            return None

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        # Take a matplotlib colormap and normalize it to the data set values.
        colmap = plt.get_cmap(dictUserParms['RayWolf_colmap'])
        vmin = np.nanmin(z)
        vmax = np.nanmax(z)
        # Be clever about picking the upper and lower bounds.
        if (abs(vmax - vmin) >= 20):
            cbarstep = 5
        else:
            cbarstep = 2
        vmin = int(vmin - (vmin % cbarstep))
        if vmin < 0:
            vmin = 0
        vmax = int(vmax + cbarstep - (vmax % cbarstep))
        cbarticks = np.arange(vmin, vmax + cbarstep, cbarstep)
        # Normalize the colormap to the chosen data range.
        cNorm = colors.Normalize(vmin = vmin, vmax = vmax)
        scalarMap = cmx.ScalarMappable(norm = cNorm, cmap = colmap)

        # Marker size is determined by the value of core diameter for each point, arbitrarily
        # binned and then mapped to one of 5 marker sizes to improve readability.
        numbins = 5
        core_min = np.nanmin(corediam)
        core_max = np.nanmax(corediam)
        if core_max - core_min < numbins:
            binmin = 0
        else:
            binmin = core_min
        bins = np.linspace(binmin, core_max, num = numbins, endpoint = True)
        inds = np.digitize(corediam, bins)
        # The bin index array is pulling double duty as both the bin index and the base
        # from which to calculate the marker size. Take 2*inds as the radius and calculate
        # the area of a circle, which shall be the marker size.
        area = np.pi * (inds * 2)**2
        
        for a in np.arange(0, len(x)):
            colorVal = scalarMap.to_rgba(z[a,:])
            # Scatterplot fails if z[a,:] has no non-nan entries
            nanind = np.where(np.isnan(z[a,:]))[0]
            if len(nanind) is z[a,:].size:
                pass
            else:
                # Varying radius, varying color
                #plt.scatter(t2D[a,:], y[a,:], s = area[a,:], c = colorVal, marker = 'o', label = '%.1f deg' % elev[a])
                plt.scatter(t2D[a,:], y[a,:], s = area[a,:], c = colorVal, marker = 'o')

        # Since neither imshow nor contourf was used, plt.colorbar() won't work. To get a colorbar,
        # one must first create a dummy scalar mappable from which to create the colorbar.
        sm = plt.cm.ScalarMappable(cmap = colmap)
        sm.set_array([vmin, vmax])
        cbar = plt.colorbar(sm, ticks = cbarticks)

        # Adding a legend for the marker size is tricky because of how easy matplotlib makes it
        # to add legends most of the time. This is not "most of the time". Get around this by 
        # plotting the 5 markers somewhere off-screen, labeling them with the bin labels, and
        # then adding a legend as usual.
        # From the numpy documentation for np.digitize:
        # bins[i-1] <= x < bins[i] if 'bins' is monotonically increasing and 'right' is False.
        lbls = ['%s <= diam < %s' % (bins[i-1], bins[i]) for i, v in enumerate(bins)]
        # The first entry is invalid. bins[i-1] when i = 0 yields the last value in bins.
        lbls = lbls[1:]
        # The last entry requires special formatting.
        lbls.append('%s <= diam' % bins[-1])
        leg_x = range(0, len(bins))
        leg_y = [2*vr_ylim for i in leg_x]
        leg_inds = np.digitize(bins, bins)
        leg_area = np.pi * (leg_inds * 2)**2
        for i in leg_x:
            plt.scatter(leg_x[i], leg_y[i], s = leg_area[i], marker = 'o', label = lbls[i])
        plt.legend(title = 'Core diameter')
        
        
        plt.ylim(0, vr_ylim)
        ax.set_xlim(-0.5, len(t)+3)
        plt.title(vr_labels['title'])
        plt.xlabel(vr_labels['xlabel'])
        plt.ylabel(vr_labels['ylabel'])
        plt.xticks(t, timestr, rotation = '60')
        plt.tight_layout()

        fname = '{}_{}_VrCorediam.png'.format(base_fname, 'VrPoints')
        fig.set_size_inches(figsize)
        plt.savefig(fname)

        return None
    

    # Testing
    
    # Core diameter
    lumberjack.info('Plotting %s' % (dictLogMsg['plot_corediam']))
    plot_corediam()

    # Shear
    lumberjack.info('Plotting %s' % (dictLogMsg['plot_shear']))
    plot_shear()

    # Vr data points
    lumberjack.info('Plotting %s' % (dictLogMsg['plot_VrPoints']))
    plot_VrPoints()

    # Vr values color-coded by binned values
    lumberjack.info('Plotting %s' % (dictLogMsg['plot_VrPoints_NumOnly']))
    plot_VrPoints_NumOnly()

    ## Vr contours based on raw input values. NEVER USE THIS PLOT.
    ##lumberjack.info('Plotting %s' % (dictLogMsg['plot_VrContours_Raw']))
    ##plot_VrContours_Raw()
    
    # Vr contours based on gridded interpolation.
    lumberjack.info('Plotting %s' % (dictLogMsg['plot_VrContours_GridInterp']))
    plot_VrContours_GridInterp()

    # Vr contours based on triangular interpolation.
    lumberjack.info('Plotting %s' % (dictLogMsg['plot_VrContours_TriangInterp']['unrefined']))
    plot_VrContours_TriangInterp(refine = False)

    # Vr contours based on triangular interpolation with a refiner.
    lumberjack.info('Plotting %s' % (dictLogMsg['plot_VrContours_TriangInterp']['refined']))
    plot_VrContours_TriangInterp(refine = True)

    # Vr filled contours based on gridded interpolation.
    lumberjack.info('Plotting %s' % (dictLogMsg['plot_VrContours_Filled_GridInterp']))
    plot_VrContours_Filled_GridInterp()

    # Vr filled contours based on triangular interpolation.
    lumberjack.info('Plotting %s' % (dictLogMsg['plot_VrContours_Filled_TriangInterp']['unrefined']))
    plot_VrContours_Filled_TriangInterp(refine = False)

    # Vr filled contours based on triangular interpolation with a refiner.
    lumberjack.info('Plotting %s' % (dictLogMsg['plot_VrContours_Filled_TriangInterp']['refined']))
    plot_VrContours_Filled_TriangInterp(refine = True)

    # Vr plot style requested by Ray Wolf
    lumberjack.info('Plotting %s' % (dictLogMsg['plot_Vr_RayWolf']))
    plot_Vr_RayWolf()

    # Vr plot where the marker size indicates core diameter value (if this data set was provided) and
    # the marker color indicates the Vr value.
    lumberjack.info('Plotting %s' % (dictLogMsg['plot_Vr_Corediam']))
    plot_Vr_Corediam()

    return None


# Load a data set, process it, and plot it.
logObj = setUpTheLogger()
dPlotVars = set_vars('config.txt')
rawdata = load_file(dPlotVars['filename'], logObj)
dPlotData = process_data(rawdata, dPlotVars, logObj)
status = make_plots(dPlotVars, dPlotData, logObj)

# Perform an orderly shutdown of the logger (flush and close all handlers).
logging.shutdown()
