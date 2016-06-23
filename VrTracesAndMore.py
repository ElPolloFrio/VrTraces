# Written in 2012 by L. M. Kanofsky, WFO LSX
# Used through 2014
# Updated 6/2016
#
# This script was written to support local research efforts at WFO LSX.
#
# Input: a specially-formatted CSV file containing data points
# for mesovortices.
# Output: a set of graphics including Vr traces

# The CSV file should resemble the following example, with headers:
# elevation angle, time, height, vr value, core diameter, shear
# 0.5, 1307, 0.7, 16, 2.2, 0.0111
# 0.9, 1307, 0.81, 17, 3.1, 0.0119
# 1.3, 1307, 1.12, 18, 2.7, 0.0168
# 1.8, 1307, 1.57, 18, 3.1, 0.0011
# 2.4, 1307, 2.1, 19, 3.1, 0.0084
# 0.5, 1311, 0.61, 17, 2.6, 0.0103
# 0.9, 1311, 0.9, 20, 1.8, 0.01403
# 3.1, 1311, 2.6, 23, 3.3, 0.0128
#
# Use 'M' for any missing value. Exception: don't use 'M' for time because
# it will cause an error. 
#
# The values for an elevation angle or at a given time step might be missing
# for any number of reasons (e.g., range folding). It's important to include
# all time steps but mark missing records as "missing", otherwise it will
# look as if a volume scan had been inadvertently skipped. 
#
# Every time step must be present at least once even if no data is available
# for any elevation angle at that time step. If a time step is skipped, then
# the plots will be incorrect in the x-axis. To handle a missing time step,
# enter one line with the correct time stamp and put 'M' for all other
# values in that row.
#
# It's OK to skip elevation angles irregularly (e.g., the 1307 timestep
# in the short example above doesn't have an entry for 3.1 degrees, but the
# 1311 timestep does).
#
# If values for shear or core diameter are not provided, then use 'M' for the
# values in the entire column (except for the header).
#
# Remember to format the time column in Excel as '0000' to preserve leading
# zeros. Do this every time the file is opened in Excel, even if no changes
# were made: right-click the column letter for the time column, choose "format
# cells", select "custom format", type 0000 in the box, click "apply" or "OK"
# on all dialog boxes, then save and close the file. 

import numpy as np
import string, logging
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.font_manager
import matplotlib.tri as tri

def setUpTheLogger():
    # Create the logger object
    logger = logging.getLogger('vrtraces')
    logger.setLevel(logging.INFO)

    # Create a console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Create a formatter and add it to the handler
    logformatter = logging.Formatter('%(asctime)s - %(name)-10s - %(levelname)s - %(message)s')
    ch.setFormatter(logformatter)
    
    # Add the handler to the logger
    logger.addHandler(ch)

    # Log some headers
    logger.info('---------------------------------')
    logger.info('----- Creating Vr Traces --------')
    logger.info('---------------------------------')
    
    return logger


def load_file(filename):
    # Load in the file contents as a numpy array
    # Skip headers
    data = np.loadtxt(filename, dtype = 'string', delimiter = ',', skiprows = 1)
    logger.info('Loaded %s', filename)

    return data


