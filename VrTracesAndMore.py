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
import string, logging
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.font_manager
import matplotlib.tri as tri

def setUpTheLogger():
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


def load_file(filename):
    # Load in the file contents as a numpy array.
    # Skip headers.
    data = np.loadtxt(filename, dtype = 'string', delimiter = ',', skiprows = 1)
    logger.info('Loaded %s', filename)

    return data


