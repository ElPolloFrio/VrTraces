This script was written to support local research efforts at WFO LSX. 

The script was written by L. M. Kanofsky in 2012 and it was used through 2014. It was refactored and updated in June 2016.

It requires the following:
Python 2.7 and compatible versions of numpy, scipy, and matplotlib.

Input: a CSV file containing data points for mesovortices
Output: a set of graphics including Vr traces

Data points for mesovortices are typically obtained by examining archived radar data from a WSR-88D or TDWR. Archived radar data is freely available from NCEI (formerly NCDC) and can be viewed with the free NCDC Weather and Climate Toolkit or with a specialized program such as GR2Analyst.

Sample CSV files are provided, along with the output graphics produced by this script.

The CSV file should resemble the following example, including headers:
elevation angle, time, height, vr value, core diameter, shear
0.5, 1307, 0.7, 16, 2.2, 0.0111
0.9, 1307, 0.81, 17, 3.1, 0.0119
1.3, 1307, 1.12, 18, 2.7, 0.0168
1.8, 1307, 1.57, 18, 3.1, 0.0011
2.4, 1307, 2.1, 19, 3.1, 0.0084
0.5, 1311, 0.61, 17, 2.6, 0.0103
0.9, 1311, 0.9, 20, 1.8, 0.01403
3.1, 1311, 2.6, 23, 3.3, 0.0128

Use '"M" for any missing value. Exception: don't use "M" in the "time" column because it will cause an error. 

Every time step between the first and last volume scan of interest MUST be present at least once in the file, even if no data is available for any elevation angle at that time step. If a time step is skipped, then the plots will be incorrect in the time axis and could be seen as misleading. To handle a missing time step, enter one line with the correct time stamp and put "M" for all other values in that row. See the sample file.

The Vr, core diameter, or shear values for an elevation angle at a given time step might be missing for any number of reasons (e.g., range folding). It's important to mark missing values as "missing", otherwise it will look as if a volume scan had been inadvertently skipped. 

Every elevation angle DOES NOT need to be present for every time step. It's OK if some elevation angles are given sporadically. For example, in the short CSV example above, the 1307 time step does not have an entry for 3.1 degrees, but the 1311 time step does. 

Sometimes, a forecaster may not be interested in all three quantities (Vr, core diameter, or shear), or he/she may not have collected data points for all three quantities. If no data points are available for Vr, core diameter, or shear, then use "M" for the values in that entire column (except for the header). If no data is provided, that plot will be skipped automatically. 

Some people might use Excel to edit CSV files. Excel will sometimes drop the leading zeros in the "time" column, which causes problems for this script. Therefore, this script will attempt to automatically restore leading zeros. However, there might be cases where the user must manually format the "time" column in Excel to preserve leading zeros (e.g., "0026") every time the file is opened in Excel, even if no changes were made. To manually preserve leading zeros in the "time" column in Excel, do the following:
0. Open the file in Excel.
1. Right-click on the column letter for the "time" column.
2. Choose "format cells".
3. Select "custom format" from the dialog box.
4. Type "0000" (without quotes) in the entry box.
5. Click "apply" or "OK" on all dialog boxes.
6. Save the file.
7. Close the file.