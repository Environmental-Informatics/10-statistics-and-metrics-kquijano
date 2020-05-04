#!/bin/env python
# Created on March 25, 2020
#  by Keith Cherkauer
#
# This script servesa as the solution set for assignment-10 on descriptive
# statistics and environmental informatics.  See the assignment documention 
# and repository at:
# https://github.com/Environmental-Informatics/assignment-10.git for more
# details about the assignment.
#

"""
Karoll Quijano - kquijano

ABE 651: Environmental Informatics

Assignment 10
Statistics and Metrics
"""

import pandas as pd
import scipy.stats as stats
import numpy as np

def ReadData( fileName ):
    """This function takes a filename as input, and returns a dataframe with
    raw data read from that file in a Pandas DataFrame.  The DataFrame index
    should be the year, month and day of the observation.  DataFrame headers
    should be "agency_cd", "site_no", "Date", "Discharge", "Quality". The 
    "Date" column should be used as the DataFrame index. The pandas read_csv
    function will automatically replace missing values with np.NaN, but needs
    help identifying other flags used by the USGS to indicate no data is 
    availabiel.  Function returns the completed DataFrame, and a dictionary 
    designed to contain all missing value counts that is initialized with
    days missing between the first and last date of the file."""
    
    # define column names
    colNames = ['agency_cd', 'site_no', 'Date', 'Discharge', 'Quality']

    # open and read the file
    DataDF = pd.read_csv(fileName, header=1, names=colNames,  
                         delimiter=r"\s+",parse_dates=[2], comment='#',
                         na_values=['Eqp'])
    DataDF = DataDF.set_index('Date')
    
    #  remove negative streamflow values as a gross error check
    DataDF.loc[(DataDF['Discharge']<0)]=np.nan 
            
    # Number of missing values
    MissingValues = DataDF["Discharge"].isna().sum()
    
    return( DataDF, MissingValues )

# Load data     
#DataDFTR, MissingValuesTR = ReadData('TippecanoeRiver_Discharge_03331500_19431001-20200315.txt')    
#DataDFWC, MissingValuesWC = ReadData('WildcatCreek_Discharge_03335000_19540601-20200315.txt')    


def ClipData( DataDF, startDate, endDate ):
    """This function clips the given time series dataframe to a given range 
    of dates. Function returns the clipped dataframe and and the number of 
    missing values."""

    # Clip data 
    DataDF = DataDF[startDate:endDate]
    # Count missing values 
    MissingValues = DataDF["Discharge"].isna().sum()    
        
    return( DataDF, MissingValues )
    
#DataDFTR, MissingValuesTR = ClipData( DataDFTR, '1969-10-01', '2019-09-30' )    
#DataDFWC, MissingValuesWC = ClipData( DataDFWC, '1969-10-01', '2019-09-30' )    
        

def CalcTqmean(Qvalues):
    """This function computes the Tqmean of a series of data, typically
       a 1 year time series of streamflow, after filtering out NoData
       values.  Tqmean is the fraction of time that daily streamflow
       exceeds mean streamflow for each year. Tqmean is based on the
       duration rather than the volume of streamflow. The routine returns
       the Tqmean value for the given data array."""
    
    # Remove NA
    Qvalues = Qvalues.dropna()
   
    # Mean of streamflow for the serie of data
    Qmean = Qvalues.mean()

    # Index when daily streamflow exceeds mean value
    index = (Qvalues > Qmean)
    Tqmean = (index.sum()/len(Qvalues))
    
    return ( Tqmean )

#meanDFTR = CalcTqmean(DataDFTR.Discharge)       
#meanDFWC = CalcTqmean(DataDFWC.Discharge)    


def CalcRBindex(Qvalues):
    """This function computes the Richards-Baker Flashiness Index
       (R-B Index) of an array of values, typically a 1 year time
       series of streamflow, after filtering out the NoData values.
       The index is calculated by dividing the sum of the absolute
       values of day-to-day changes in daily discharge volumes
       (pathlength) by total discharge volumes for each year. The
       routine returns the RBindex value for the given data array."""
    
    # Remove NA
    Qvalues = Qvalues.dropna()
    
    #Sum of Abs values
    tmpSum = np.abs( Qvalues[:-1].values - Qvalues[1:].values ).sum()
    
    # Divides the sum 
    RBindex = ( tmpSum / Qvalues[1:].sum() )    
        
    return ( RBindex )

#CalcRBindexDFTR = CalcRBindex(DataDFTR.Discharge)
#CalcRBindexDFWC = CalcRBindex(DataDFWC.Discharge)


def Calc7Q(Qvalues):
    """This function computes the seven day low flow of an array of 
       values, typically a 1 year time series of streamflow, after 
       filtering out the NoData values. The index is calculated by 
       computing a 7-day moving average for the annual dataset, and 
       picking the lowest average flow in any 7-day period during
       that year.  The routine returns the 7Q (7-day low flow) value
       for the given data array."""

    # Remove NA
    Qvalues = Qvalues.dropna()    
    
    # Rolling window for 7 days
    val7Q=Qvalues.rolling(window=7).mean().min()
        
    return ( val7Q )

#Calc7QDFTR = Calc7Q(DataDFTR.Discharge)
#Calc7QDFWC = Calc7Q(DataDFWC.Discharge)


def CalcExceed3TimesMedian(Qvalues):
    """This function computes the number of days with flows greater 
       than 3 times the annual median flow. The index is calculated by 
       computing the median flow from the given dataset (or using the value
       provided) and then counting the number of days with flow greater than 
       3 times that value.   The routine returns the count of events greater 
       than 3 times the median annual flow value for the given data array."""

    # Remove NA
    Qvalues = Qvalues.dropna()
    
    # Numb of discharges greater than 3 times the annual median
    median3x = (Qvalues>3*Qvalues.median()).sum()
    
    return (median3x)

#CalcExceed3TimesMedianTR = CalcExceed3TimesMedian(DataDFTR.Discharge)
#CalcExceed3TimesMedianWC = CalcExceed3TimesMedian(DataDFWC.Discharge)


def GetAnnualStatistics(DataDF):
    """This function calculates annual descriptive statistcs and metrics for 
    the given streamflow time series.  Values are retuned as a dataframe of
    annual values for each water year.  Water year, as defined by the USGS,
    starts on October 1."""
    
    # Column names 
    ColNames = ['site_no', 'Mean Flow', 'Peak Flow', 'Median  Flow', 'Coeff Var','Skew','Tqmean','R-B Index','7Q','3xMedian']  
   
    # Resample data annually from Oct
    WYDataDF = DataDF.resample('AS-OCT').mean() 

    # New dataframe to store annual metric values
    WYDataDF = pd.DataFrame(0, index=WYDataDF.index,columns=ColNames) 

    WYDataDF['site_no'] =  DataDF['site_no'][0]
    WYDataDF["Mean Flow"] = DataDF["Discharge"].resample('AS-OCT').mean()
    WYDataDF["Peak Flow"] = DataDF["Discharge"].resample('AS-OCT').max()
    WYDataDF["Median Flow"] = DataDF["Discharge"].resample('AS-OCT').median()
    WYDataDF["Coeff Var"] = DataDF["Discharge"].resample('AS-OCT').std() / WYDataDF["Mean Flow"] * 100.
    WYDataDF["Skew"] = DataDF["Discharge"].resample('AS-OCT').apply(stats.skew)
    
    WYDataDF["Tqmean"] = DataDF["Discharge"].resample('AS-OCT').apply(CalcTqmean)
    WYDataDF["R-B Index"] = DataDF["Discharge"].resample('AS-OCT').apply(CalcRBindex)
    WYDataDF["7Q"] = DataDF["Discharge"].resample('AS-OCT').apply(Calc7Q)
    WYDataDF["3xMedian"] = DataDF["Discharge"].resample('AS-OCT').apply(CalcExceed3TimesMedian)
    
    return ( WYDataDF )

#GetAnnualStatisticsTR = GetAnnualStatistics(DataDFTR)


def GetMonthlyStatistics(DataDF):
    """This function calculates monthly descriptive statistics and metrics 
    for the given streamflow time series.  Values are returned as a dataframe
    of monthly values for each year."""
 
    # Resample data  
    MoDataDF = DataDF.resample('MS').mean()
    
    MoDataDF['site_no'] =  DataDF['site_no'][0]
    MoDataDF["Mean Flow"] = DataDF["Discharge"].resample('MS').mean()
    MoDataDF["Coeff Var"] = DataDF["Discharge"].resample('MS').std() / MoDataDF["Mean Flow"] * 100.
    MoDataDF["Tqmean"] = DataDF["Discharge"].resample('MS').apply(CalcTqmean)
    MoDataDF["R-B Index"] = DataDF["Discharge"].resample('MS').apply(CalcRBindex)

    return ( MoDataDF )

#GetMonthlyStatisticsTR = GetMonthlyStatistics(DataDFTR)


def GetAnnualAverages(WYDataDF):
    """This function calculates annual average values for all statistics and
    metrics.  The routine returns an array of mean values for each metric
    in the original dataframe."""
    
    # Compute averages
    AnnualAverages=WYDataDF.mean(axis=0)
        
    return( AnnualAverages )

#GetAnnualAveragesTR = GetAnnualAverages (GetAnnualStatisticsTR)


def GetMonthlyAverages(MoDataDF):
    """This function calculates annual average monthly values for all 
    statistics and metrics.  The routine returns an array of mean values 
    for each metric in the original dataframe."""
    
    # Select months
    Months = MoDataDF.index.month
    
    # New DF for means 
    MonthlyAverages = MoDataDF.groupby(Months).mean()
    
    return( MonthlyAverages )

#GetMonthlyAveragesTR = GetMonthlyAverages(GetMonthlyStatisticsTR)


# the following condition checks whether we are running as a script, in which 
# case run the test code, otherwise functions are being imported so do not.
# put the main routines from your code after this conditional check.

if __name__ == '__main__':

    # define filenames as a dictionary
    # NOTE - you could include more than jsut the filename in a dictionary, 
    #  such as full name of the river or gaging site, units, etc. that would
    #  be used later in the program, like when plotting the data.
    fileName = { "Wildcat": "WildcatCreek_Discharge_03335000_19540601-20200315.txt",
                 "Tippe": "TippecanoeRiver_Discharge_03331500_19431001-20200315.txt" }
    
    # define blank dictionaries (these will use the same keys as fileName)
    DataDF = {}
    MissingValues = {}
    WYDataDF = {}
    MoDataDF = {}
    AnnualAverages = {}
    MonthlyAverages = {}
    
    # process input datasets
    for file in fileName.keys():
        
        print( "\n", "="*50, "\n  Working on {} \n".format(file), "="*50, "\n" )
        
        DataDF[file], MissingValues[file] = ReadData(fileName[file])
        print( "-"*50, "\n\nRaw data for {}...\n\n".format(file), DataDF[file].describe(), "\n\nMissing values: {}\n\n".format(MissingValues[file]))
        
        # clip to consistent period
        DataDF[file], MissingValues[file] = ClipData( DataDF[file], '1969-10-01', '2019-09-30' )
        print( "-"*50, "\n\nSelected period data for {}...\n\n".format(file), DataDF[file].describe(), "\n\nMissing values: {}\n\n".format(MissingValues[file]))
        
        # calculate descriptive statistics for each water year
        WYDataDF[file] = GetAnnualStatistics(DataDF[file])
        
        # calcualte the annual average for each stistic or metric
        AnnualAverages[file] = GetAnnualAverages(WYDataDF[file])
        
        print("-"*50, "\n\nSummary of water year metrics...\n\n", WYDataDF[file].describe(), "\n\nAnnual water year averages...\n\n", AnnualAverages[file])

        # calculate descriptive statistics for each month
        MoDataDF[file] = GetMonthlyStatistics(DataDF[file])

        # calculate the annual averages for each statistics on a monthly basis
        MonthlyAverages[file] = GetMonthlyAverages(MoDataDF[file])
        
        print("-"*50, "\n\nSummary of monthly metrics...\n\n", MoDataDF[file].describe(), "\n\nAnnual Monthly Averages...\n\n", MonthlyAverages[file])
        
        
        
    ############ Output Files ##############
    # Annual
    WY_wild = WYDataDF['Wildcat']
    WY_wild['Station'] = 'Wildcat'
    WY_tippe = WYDataDF['Tippe']
    WY_tippe['Station'] = 'Tippe'
    WY_wild = WY_wild.append(WY_tippe)
    WY_wild.to_csv('Annual_Metrics.csv',sep=',',index=True) 

    AA_wild=AnnualAverages['Wildcat']
    AA_wild['Station']='Wildcat'
    AA_tippe=AnnualAverages['Tippe']
    AA_tippe['Station']='Tippe'
    AA_wild=AA_wild.append(AA_tippe)  
    AA_wild.to_csv('Average_Annual_Metrics.txt',sep='\t',index=True)
    
    # Monthly 
    Mo_wild = MoDataDF['Wildcat']
    Mo_wild['Station'] = 'Wildcat'
    Mo_tippe = MoDataDF['Tippe']
    Mo_tippe['Station'] = 'Tippe'
    Mo_wild = Mo_wild.append(Mo_tippe) 
    Mo_wild.to_csv('Monthly_Metrics.csv',sep=',',index=True)  
    
    MA_wild=MonthlyAverages['Wildcat']
    MA_wild['Station']='Wildcat'
    MA_tippe=MonthlyAverages['Tippe']
    MA_tippe['Station']='Tippe'
    MA_wild=MA_wild.append(MA_tippe) 
    MA_wild.to_csv('Average_Monthly_Metrics.txt',sep='\t',index=True)        