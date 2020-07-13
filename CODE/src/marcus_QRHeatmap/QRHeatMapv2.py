import pandas as pd
import numpy as np
import statistics as stats
import string

# Global Variables
metrageInc = 5 # 5 metre increments
#numTRCRuns = 4 # Number of runs of TRC
startMetrage = 0.0025 # Starting metrage of TRC measurement

###### File Structure Parameters
#### TRC
TRCSkipRows = 4 # Number of rows to skip at top of TRC CSV
TRCTopL = 'TOP L' # Name of Top Left column
TRCTopR = 'TOP R' # Name of Top Right column
TRCTwist = 'TW 3' # Name of Twist column

# Alert user to case sensitive input
print('Obtaining user input NB CASE SENTITIVE')
# Read the number of TRC files
numTRCRuns = int(input('How many TRC files are to be processed (in /data): '))
TRCFilenames = []
for j in range(numTRCRuns):
    TRCFilenames.append('TRC'+str(j+1)+'.csv')

#TRCFilenames = ['531401 - DM A 0-9.69 NORTHGATE-SANDGATE-201701311116.csv',
#                '531401 - DM A 0-9.69 NORTHGATE-SANDGATE-201707041300.csv',
#                '531401 - DM A 0-9.69 NORTHGATE-SANDGATE-201710171136.csv',
#                '531401 - DM A 0-9.69 NORTHGATE-SANDGATE-201801311009.csv']

dataDir = './data/' # Directory in which TCR and GPR files are held

#### GPR
GPRSkipRows = 15 # Number of rows to skip at top of GPR XLS
GPRFileName = 'GPR1.xlsx' # USER TO INPUT
GPRSheetName = 'Brisbane 2015 Trackbed Metrics' # USER TO INPUT

division = input('GPR: enter division (eg "Brisbane"): ')
subdivision = input('GPR: enter subdivision (eg "SF"): ')
track = int(input('GPR: enter track (eg "401"): '))

#division = 'Brisbane' # USER TO INPUT
divisionCol = 'Division' # column name of division
#subdivision = 'SF' # Shorncliffe line
subdivisionCol = 'Sub-division' # column name of subdivision
#lineSegment = '531'
#lineSegmentCol = 2 # column name of line segment
#track = 401 # Up or down line
trackCol = 'Track ID' # column name of track direction
PVCLeft = 'PVC Value' # Name of left PVC column
PVCCentre = 'PVC Value.1' # Name of centre PVC column
PVCRight = 'PVC Value.2' # Name of right PVC column
StartKM = "Start KM" # Name of Start KM column
EndKM = "End KM" # Name of Start KM column

outFile = subdivision + str(track) + 'OUT.xlsx' # Output File

###### Read TRC files
TRCList = [] # List to store TRC files as dataframes

for j in range(numTRCRuns):
    print('Reading TRC file:', TRCFilenames[j])
    TRCTempDF = pd.read_csv(dataDir+TRCFilenames[j], skiprows=TRCSkipRows)
    TRCTempDF.rename(columns=lambda x: x.strip(), inplace=True)
    TRCList.append(TRCTempDF)

###### Read GPR
print('Reading GPR file:', GPRFileName)
GPRDF = pd.read_excel(dataDir+GPRFileName, sheet_name = GPRSheetName, skiprows=GPRSkipRows)

# Filter GPR DF by segment etc
print('Segmenting GPR: division=', division, '; subdivision=', subdivision, '; track=', track)
GPRSegment = GPRDF.loc[ (GPRDF[divisionCol]==division) & (GPRDF[subdivisionCol]==subdivision ) &
                       (GPRDF[trackCol]==track )]

# Establish dataframe with heat map
calcDF = pd.DataFrame(columns=['METRAGE', 'Start Km', 'End Km','GPR Left', 'GPR Centre', 'GPR Right',
                               'SDTopLeft1', 'SDTopLeft2','SDTopLeft3','SDTopLeft4',
                               'SDTopRight1','SDTopRight2','SDTopRight3','SDTopRight4',
                              'SDTwist1','SDTwist2','SDTwist3','SDTwist4',
                              'Combined1','Combined2','Combined3','Combined4'])

# Find largest distance measured by any TRC run
maxDist = np.min( [float(TRCList[0].iloc[-1:,]['METRAGE']), float(TRCList[1].iloc[-1:,]['METRAGE']),
    float(TRCList[2].iloc[-1:,]['METRAGE']), float(TRCList[3].iloc[-1:,]['METRAGE']) ])

# Update Start and End Km
for i in range( int(maxDist*1000/metrageInc) ):
    calcDF.loc[i, 'METRAGE'] = startMetrage + i*metrageInc/1000
    calcDF.loc[i, 'Start Km'] = calcDF.loc[i, 'METRAGE'] - metrageInc/1000/2
    calcDF.loc[i, 'End Km'] = calcDF.loc[i, 'METRAGE'] + metrageInc/1000/2

# Calculate standard deviations: Top Left, Top Right and Twist for 4 TRC Runs
print('Calculating std dev: left, right, twist, combined')
colLeft = 'SDTopLeft'
colRight = 'SDTopRight'
colTwist = 'SDTwist'
colCombined = 'Combined'

# Repeat for each TRC File
for j in range(0, numTRCRuns):
    colNameLeft = colLeft + str(j + 1)
    colNameRight = colRight + str(j + 1)
    colNameTwist = colTwist + str(j + 1)
    colNameCombined = colCombined + str(j + 1)

    # Repeat for each metrage range
    for i in range(2, len(calcDF) - 2):
        inRange = np.where((TRCList[j].loc[:, 'METRAGE'] > calcDF.loc[i - 2, 'METRAGE']) &
                           (TRCList[j].loc[:, 'METRAGE'] <= calcDF.loc[i + 2, 'METRAGE']))
        calcDF.loc[i, colNameLeft] = stats.stdev(TRCList[j].loc[inRange][TRCTopL])
        calcDF.loc[i, colNameRight] = stats.stdev(TRCList[j].loc[inRange][TRCTopR])
        calcDF.loc[i, colNameTwist] = stats.stdev(TRCList[j].loc[inRange][TRCTwist])
        calcDF.loc[i, colNameCombined] = (calcDF.loc[i, colNameLeft] + calcDF.loc[i, colNameRight]) / 2 + calcDF.loc[
            i, colNameTwist]

    # Copy 2nd row to first row
    calcDF.loc[1, colNameLeft] = calcDF.loc[2, colNameLeft]
    calcDF.loc[1, colNameRight] = calcDF.loc[2, colNameRight]
    calcDF.loc[1, colNameTwist] = calcDF.loc[2, colNameTwist]
    calcDF.loc[1, colNameCombined] = calcDF.loc[2, colNameCombined]

# Insert GPR data
print('Inserting GPR data')
for i in range(1, len(calcDF) - 2):
    inRange = np.where(round(GPRSegment.loc[:, StartKM], 3) == round((calcDF.loc[i, 'Start Km']), 3))

    # Only process if found unique corresponding range in GPR
    if (len(inRange[0]) == 1):
        calcDF.loc[i, 'GPR Left'] = GPRSegment.iloc[inRange[0][0]][PVCLeft]
        calcDF.loc[i, 'GPR Centre'] = GPRSegment.iloc[inRange[0][0]][PVCCentre]
        calcDF.loc[i, 'GPR Right'] = GPRSegment.iloc[inRange[0][0]][PVCRight]

# Display result
print(calcDF)

# Write result to Excel file
print('Writing result to file')

###### Save dataframe to EXcel with conditional formatting

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter(outFile, engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
calcDF.to_excel(writer, sheet_name='Heat Map')

# Get the xlsxwriter workbook and worksheet objects.
workbook  = writer.book
worksheet = writer.sheets['Heat Map']

# Apply a conditional format to the cell range.
#worksheet.conditional_format('T3:W1942', {'type': '3_color_scale',
#                                       'min_value': 1,
#                                       'min_color': 'green',
#                                       'mid_value': 4.9,
#                                       'mid_color': 'yellow',
#                                       'max_value': 7,
#                                       'max_color': 'red'})

from openpyxl.styles import PatternFill, colors

# Cell range of GPR data should be fixed
cellRange1 = 'E3:G'+str(len(calcDF)-1)

# Cell range of TCR data depends on number of TCR files
alphabet = list(string.ascii_uppercase)
startColLetter = 7+3*numTRCRuns
endColLetter = 6+4*numTRCRuns
print(alphabet[startColLetter], alphabet[endColLetter])
cellRange2 = alphabet[startColLetter]+str('3:')+alphabet[endColLetter]+str(len(calcDF)-1)

# Establish colour scheme
grey_format = workbook.add_format({'bg_color': '#808080'})
green_format = workbook.add_format({'bg_color': '#00FF00'})
yellow_format = workbook.add_format({'bg_color': '#FFFF00'})
orange_format = workbook.add_format({'bg_color': '#FF9933'})
red_format = workbook.add_format({'bg_color': '#FF0000'})
pink_format = workbook.add_format({'bg_color': '#FF66FF'})
purple_format = workbook.add_format({'bg_color': '#660066'})

worksheet.conditional_format(cellRange1, {'type': 'cell',
                                          'criteria': '<',
                                          'value':  0,
                                          'format':   grey_format})

worksheet.conditional_format(cellRange1, {'type': 'cell',
                                          'criteria': 'between',
                                          'minimum':  0,
                                          'maximum':  10,
                                          'format':   green_format})

worksheet.conditional_format(cellRange1, {'type': 'cell',
                                          'criteria': 'between',
                                          'minimum':  10,
                                          'maximum':  20,
                                          'format':   yellow_format})

worksheet.conditional_format(cellRange1, {'type': 'cell',
                                          'criteria': 'between',
                                          'minimum':  20,
                                          'maximum':  30,
                                          'format':   orange_format})

worksheet.conditional_format(cellRange1, {'type': 'cell',
                                          'criteria': 'between',
                                          'minimum':  30,
                                          'maximum':  40,
                                          'format':   red_format})

worksheet.conditional_format(cellRange1, {'type': 'cell',
                                          'criteria': 'between',
                                          'minimum':  40,
                                          'maximum':  59.9,
                                          'format':   pink_format})

worksheet.conditional_format(cellRange1, {'type': 'cell',
                                          'criteria': '>=',
                                          'value':  60,
                                          'format':   purple_format})


worksheet.conditional_format(cellRange2, {'type': '3_color_scale',
                                       'min_color': 'green',
                                       'max_color': 'red'})

# Close the Pandas Excel writer and output the Excel file.
writer.save()
