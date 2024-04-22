#%% Imports
# File locations, os
import os
import re
import winshell

# Pandas
import pandas as pd
from pandas.tseries.frequencies import to_offset

# Numpy
import numpy as np

# Scipy
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis

# Statsmodels
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose, STL, MSTL

# Plotting
from matplotlib import pyplot as plt
import seaborn as sns

# UI
from shiny import reactive
from shiny.express import render, ui, input
from shiny.types import FileInfo
from shiny.ui import page_navbar
from shinyswatch import theme

# Misc
from adtk.detector import InterQuartileRangeAD, ThresholdAD
import asyncio
from datetime import timedelta, date
from functools import partial
import itertools
from itertools import zip_longest, cycle, islice
import math
from operator import itemgetter
import ruptures as rpt
from sklearn.linear_model import LinearRegression
from time import perf_counter
#%% Global Variables
# With 10k samples in a PACF, I get 40 lags
SampleRateList = ['1us','5us','25us','100us','500us','1ms','1ms','25ms','100ms','500ms',
                  '1s','5s','15s','1min','5min','30min','1h','3h','6h','12h','1d','1w','1M','6M','1Y']
        
# Same list of common resample frequencies, except month and year are handled
SampleRateListCompare = ['1us','5us','25us','100us','500us','1ms','1ms','25ms','100ms','500ms',
                         '1s','5s','15s','1min','5min','30min','1h','3h','6h','12h','1d','1w','4w','24w','48w']
#%% Data Input - Functions
def ClusteringSingleDimension(points):
# =============================================================================
#     Clusters a list into 10% intervals, returns the means of these clusters
# =============================================================================
    clusters = []
    eps = 0.1
    points_sorted = sorted(points)
    curr_point = points_sorted[0]
    curr_cluster = [curr_point]
    curr_mean = curr_point

    for point in points_sorted[1:]:
        if point <= curr_point + eps * curr_mean:
            curr_cluster.append(point)
            curr_mean = np.mean(curr_cluster)
        else:
            clusters.append(curr_cluster)
            curr_cluster = [point]
            curr_mean = point
        curr_point = point
    clusters.append(curr_cluster)
    
    Means = list(map(lambda Cluster: np.mean(Cluster),clusters))
    
    return Means

def RelevantPeriodicities(Dataset,TimeColumn,RelevantColumn, MaxSamples, StrengthCutoff):
# =============================================================================
#     Returns the list of statistically relevant periodicities, and statistically irrelevant periodicities
# =============================================================================
    SampleRateOriginal = Dataset[TimeColumn][1] - Dataset[TimeColumn][0]
    TimespanOriginal = Dataset[TimeColumn].iloc[-1] - Dataset[TimeColumn].iloc[0]
    
    # Get first index where the resample rate is less frequent than the original
    Index = next(idx for idx, value in enumerate(SampleRateListCompare) if pd.to_timedelta(value) > SampleRateOriginal)
    try:
        IndexEnd = next(idx for idx, value in enumerate(SampleRateListCompare) if pd.to_timedelta(value) > TimespanOriginal)
    except:
        IndexEnd = len(SampleRateList)
    ResampleRates = [to_offset(SampleRateOriginal).freqstr]
    # Output is a list of downsample rates starting with original sample rate
    ResampleRates.extend(SampleRateList[Index: IndexEnd])
    
    # MaxSamples = 10000
    
    RelevantPeriodNums, StrongPeriodNums = [], []
    
    for ResampleIndex, ResampleRate in enumerate(ResampleRates):
        ResampledData = Dataset.resample(ResampleRate,on=TimeColumn).mean()
        SampleRateCurrent = ResampledData.index[1] - ResampledData.index[0]
        
        DownsampleRate = SampleRateCurrent / SampleRateOriginal
        
        if len(ResampledData) > MaxSamples:
            Length = MaxSamples
        else:
            Length = len(ResampledData[RelevantColumn])
        
        # PACF requires all missing values to be filled
        for Run in np.arange(len(ResampledData)//Length):
            if len(ResampledData) > MaxSamples:
                ResampledSeries = ResampledData[RelevantColumn][Run*MaxSamples:(Run+1)*MaxSamples].interpolate()
            else:
                ResampledSeries = ResampledData[RelevantColumn].interpolate()
                
            PACFValues, PACFConfInt = pacf(ResampledSeries,alpha=0.05)
            PACFConfUpper = [1.0]
            PACFConfUpper.extend(PACFConfInt[1:, 1] - PACFValues[1:])
            
            ACFValues, ACFConfInt = acf(ResampledSeries,alpha=0.05)
            ACFConfUpper = [1.0]
            ACFConfUpper.extend(ACFConfInt[1:, 1] - ACFValues[1:])
            
        # =============================================================================
        #     Statistically Relevant Period Indices
        # =============================================================================
            # Statistically relevant periods have atleast a PACF value of StrengthCutoff and are above the conf int
            PACFPeakIndices, _ = find_peaks(PACFValues,height=np.maximum(StrengthCutoff,PACFConfUpper))
            ACFPeakIndices, _ = find_peaks(ACFValues,height=np.maximum(StrengthCutoff,ACFConfUpper))
            # Relevant indices appear in the ACF as prominent, and index is greater than 3
            RelevantIndices = [Index for Index in PACFPeakIndices if (Index in ACFPeakIndices) and (Index >= 3)]
            RelevantPeriodNums.extend(list(map(lambda Index: Index * DownsampleRate, RelevantIndices)))
        # =============================================================================
        #     Strong Period Indices (not necessarily statistically relevant)
        # =============================================================================
            PACFPeakIndices, _ = find_peaks(PACFValues,height=StrengthCutoff)
            ACFPeakIndices, _ = find_peaks(ACFValues)
            # Relevant indices appear in the ACF as prominent, and index is greater than 3
            StrongIndices = [Index for Index in PACFPeakIndices if (Index in ACFPeakIndices) and (Index >= 3)]
            StrongPeriodNums.extend(list(map(lambda Index: Index * DownsampleRate, StrongIndices)))
            
    RelevantPeriodNums = list(set(RelevantPeriodNums))
    StrongPeriodNums = list(set(StrongPeriodNums) - set(RelevantPeriodNums))
    
    if len(RelevantPeriodNums) > 0:
        RelevantPeriods = [int(Mean) for Mean in ClusteringSingleDimension(RelevantPeriodNums)]
    else:
        RelevantPeriods = []
        
    if len(StrongPeriodNums) > 0:
        StrongPeriods = []
        StrongPeriodsAll = [int(Mean) for Mean in ClusteringSingleDimension(StrongPeriodNums)]
        for StrongPeriod in StrongPeriodsAll:
            # I also need to take out anything in a +-10% interval of an existing relevant mean interval
            if len(RelevantPeriods) > 0:
                InRelevant = False
                for Mean in RelevantPeriods:
                    if (StrongPeriod > Mean * 0.9) or (StrongPeriod < Mean * 1.1):
                        InRelevant = True
                if InRelevant is False:
                    StrongPeriods.append(StrongPeriod)
            else:
                StrongPeriods = StrongPeriodsAll
    else:
        StrongPeriods = []
        
    return RelevantPeriods, StrongPeriods
#%% Missing Value Handling - Functions
def MissingValueEvaluator(Data):
# =============================================================================
# Find missing values and evaluate them
# Evaluate based on:
    # Start point for MV sequence
    # MV type (point or continuous)
    # MV length in sequence
    # MV % in sequence for each of the variables
# =============================================================================
    # Get sub-dataframe with missing values and their indices
    MissingValueData = Data[Data.isnull().any(axis=1)]
    MissingValueIndices = MissingValueData.index.values.tolist()
    
    Columns = list(Data.columns)
    Columns.remove('Timestamp')
    
    # Create a dataframe where we evaluate the missing value sequences
    MissingValueDataframe = pd.DataFrame()
    MissingValueSequences = []
    for k, g in itertools.groupby(enumerate(MissingValueIndices), lambda x: x[0]-x[1]):
        CurrentSequence = list(map(itemgetter(1), g))
        
        if len(CurrentSequence) == 1:
            Classification = 'Point Error'
        if len(CurrentSequence) > 1:
            Classification = 'Continuous Error'
        
        SequenceData = Data.iloc[CurrentSequence]
        
        MissingValueDictionary = {'Start Point': [CurrentSequence[0]],'Length': [len(CurrentSequence)],
                                  'Missing Value Type': [Classification]}
        
        for Column in Columns:
            Name = 'Missing Value in ' + Column
            MissingValuePercentage = np.round(SequenceData[Column].isnull().sum() / len(CurrentSequence) * 100,2)
            MissingValueDictionary[Name] = [MissingValuePercentage]
        
        MissingValueDataframe = pd.concat([MissingValueDataframe,pd.DataFrame.from_dict(MissingValueDictionary)],
                                          ignore_index = True)
        
        MissingValueSequences.append(CurrentSequence)

# =============================================================================
# Missing Value Evaluation Dataframe Creation
# =============================================================================
    DF = MissingValueDataframe.copy()
    # Evaluate the missing value dataframe
    if len(DF) != 0:
        MissingValueEvaluation = pd.DataFrame()
        
        TotalPointNum = len(Data)
        UniqueMVNum = DF['Length'].sum()
        UniquePENum = len(DF.loc[DF['Missing Value Type']=='Point Error'])
        UniqueCELengthTotal = UniqueMVNum - UniquePENum
        UniqueCEAverageLength = UniqueCELengthTotal / DF['Length'].loc[DF['Missing Value Type']=='Continuous Error'].count()
        CEInstances = len(DF) - UniquePENum
        
        PEPercTotalGlobal = np.round(100*UniquePENum/TotalPointNum,4)
        PEPercMVGlobal = np.round(100*UniquePENum/UniqueMVNum,4)
        if UniquePENum > 0:
            PEPercPEGlobal = 100
        else:
            PEPercPEGlobal = 'No Missing Points'
        CEPercTotalGlobal = np.round(100*UniqueCELengthTotal/TotalPointNum,4)
        CEPercMVGlobal = np.round(100*UniqueCELengthTotal/UniqueMVNum,4)
        if UniqueCELengthTotal > 0:
            CEPercCEGlobal = 100
            CEDelta = 0
        else:
            CEPercCEGlobal = 'No Missing Continuous Values'
            CEDelta = 'No Missing Continuous Values'
        
        PETotals, PEMVs, PEPEs, CEInsts = [PEPercTotalGlobal], [PEPercMVGlobal], [PEPercPEGlobal], [CEInstances]
        CETotals, CEMVs, CECEs, CEDeltas = [CEPercTotalGlobal], [CEPercMVGlobal], [CEPercCEGlobal], [CEDelta]
        PEInsts = [UniquePENum]
        
        for Column in Columns:
            PEInCol = DF.loc[(DF['Missing Value Type']=='Point Error') &
                                      (DF[f'Missing Value in {Column}']==100)].count().loc[f'Missing Value in {Column}']
            
            DF['CELengthInCol'] = DF['Length'].mul(DF[f'Missing Value in {Column}']/100).astype(int)
            CEInCol = DF['CELengthInCol'].loc[DF['Missing Value Type']=='Continuous Error'].sum()
            CEInstancesInCol = len(DF.loc[(DF['Missing Value Type']=='Continuous Error') &
                                          (DF[f'Missing Value in {Column}']>0)])
            if CEInstancesInCol > 0:
                CEAverageLengthInCol = CEInCol / CEInstancesInCol
            else:
                CEAverageLengthInCol = 0
            
            PEPercData = np.round(100*PEInCol/TotalPointNum,4)
            PEPercMV = np.round(100*PEInCol/UniqueMVNum,4)
            if UniquePENum > 0:
                PEPercPE = np.round(100*PEInCol/UniquePENum,4)
            else:
                PEPercPE = 'No Point Errors in Data'
            CEPercData = np.round(100*CEInCol/TotalPointNum,4)
            CEPercMV = np.round(100*CEInCol/UniqueMVNum,4)
            if UniqueCELengthTotal > 0:
                CEPercCE = np.round(100*CEInCol/UniqueCELengthTotal,4)
                CEAvgLenDifference = np.round(100*(CEAverageLengthInCol-UniqueCEAverageLength)/UniqueCEAverageLength,4)
            else:
                CEPercCE = 'No Continuous Errors In Data'
                CEAvgLenDifference = 'No Continuous Errors In Data'
                
            PEInsts.append(PEInCol)
            PETotals.append(PEPercData)
            PEMVs.append(PEPercMV)
            PEPEs.append(PEPercPE)
            CEInsts.append(CEInstancesInCol)
            CETotals.append(CEPercData)
            CEMVs.append(CEPercMV)
            CECEs.append(CEPercCE)
            CEDeltas.append(CEAvgLenDifference)
        
        MissingValueEvaluationDictionary = {'Point Error Instances in Feature':PEInsts,
                                            'Percent of All Data as Point Errors in Feature':PETotals,
                                            'Percent of All Missing Values as Point Errors in Feature':PEMVs,
                                            'Percent of All Point Errors as Point Errors in Feature':PEPEs,
                                            'Continuous Error Instances in Feature':CEInsts,
                                            'Percent of All Data as Continuous Errors in Feature':CETotals,
                                            'Percent of All Missing Values as Continuous Errors in Feature':CEMVs,
                                            'Percent of All Continuous Errors as Continuous Errors in Feature':CECEs,
                                            'Percentage Increase in Average Continuous Error Length From Global Value':CEDeltas}
        
        Index = ['Global']
        Index.extend(Columns)
        
        MissingValueEvaluation = pd.concat([MissingValueEvaluation,
                                            pd.DataFrame.from_dict(MissingValueEvaluationDictionary)],
                                            ignore_index = True)
        MissingValueEvaluation.index = Index
    else:
        MissingValueEvaluation = 'No Missing Values'

# =============================================================================
# Columnwise Missing Value Evaluation
# =============================================================================
    MVCols = []
    for Column in Columns:
        MissingValueData = Data[Column][Data[Column].isnull()]
        MissingValueIndices = MissingValueData.index.values.tolist()
        
        # Create a dataframe where we evaluate the missing value sequences
        MVColsTemp = pd.DataFrame()
        for k, g in itertools.groupby(enumerate(MissingValueIndices), lambda x: x[0]-x[1]):
            CurrentSequence = list(map(itemgetter(1), g))
            
            MissingValueDictionary = {'Start Point': [CurrentSequence[0]],'Length': [len(CurrentSequence)]}
            
            MVColsTemp = pd.concat([MVColsTemp,pd.DataFrame.from_dict(MissingValueDictionary)],ignore_index = True)
            
        MVCols.append(MVColsTemp)
        
    MVCols = (dict(zip_longest(Columns, MVCols)))

    return {'MVDataFrame': MissingValueDataframe, 'MVEvaluation': MissingValueEvaluation, 'MVCols': MVCols}

def SCImputation(Data, Column, SCMVs, SCCalcLength):
    SCStartEnds = list(map(lambda i: (SCMVs.iloc[i]['Start Point'],
                                      SCMVs.iloc[i]['Start Point'] + SCMVs.iloc[i]['Length'] - 1),
                           np.arange(len(SCMVs))))
    
    for Index, StartEnd in enumerate(SCStartEnds):
        # Calculate how much space there is before and after to run the ewm
        Start, End = StartEnd[0], StartEnd[1]
        PriorTemp = Data[Column][: Start]
        if len(list(PriorTemp[PriorTemp.isnull()].index)) > 0:
            FirstMVBefore = max(list(PriorTemp[PriorTemp.isnull()].index))
            SpaceBefore = Start - FirstMVBefore
        else:
            FirstMVBefore = -1
            SpaceBefore = Start
        
        PostTemp = Data[Column][End + 1:]
        if len(list(PostTemp[PostTemp.isnull()].index)) > 0:
            FirstMVAfter = min(list(PostTemp[PostTemp.isnull()].index))
            SpaceAfter = FirstMVAfter - End
        else:
            FirstMVAfter = len(Data)
            SpaceAfter = len(Data) - End
        
        # Get prior and post data, all without missing values
        if SpaceBefore > SCCalcLength:
            PriorData = Data[Column][Start - SCCalcLength: Start + 1]
        else:
            PriorData = Data[Column][FirstMVBefore + 1: Start + 1]
            
        if SpaceAfter > SCCalcLength:
            PostData = Data[Column][End + 1: End + SCCalcLength + 1]
        else:
            PostData = Data[Column][End + 1: FirstMVAfter]
        
        # Get prior and post data both x and y for linear interpolation
        PriorXY = (max(list(PriorData.index)), PriorData.ewm(alpha=0.2).mean()[max(list(PriorData.index))])
        PostXY = (min(list(PostData.index)), PostData.ewm(alpha=0.2).mean()[min(list(PostData.index))])
        
        TempDF = Data.loc[PriorXY[0]:PostXY[0],Column].copy()
        TempDF[PriorXY[0]] = PriorXY[1]
        TempDF[PostXY[0]] = PostXY[1]
        
        Data.loc[PriorXY[0]:PostXY[0],Column] = TempDF.loc[PriorXY[0]:PostXY[0]].interpolate()
    return Data

# If you are maintaining this function, you are in for some shit boiii
def LCImputation(Data, Column, LCMVs, CutoffMultiplier, Periods):
    # =============================================================================
    #     Start LCMV
    # =============================================================================    
        LCStartEnds = list(map(lambda i: (LCMVs.iloc[i]['Start Point'],
                                          LCMVs.iloc[i]['Start Point'] + LCMVs.iloc[i]['Length'] - 1),
                                np.arange(len(LCMVs))))
        
        MaximumPeriods = [min(Periods)] * len(LCMVs)
        for Period in Periods:
            # Get indices of LCMVs where length of CMV greater than cutoff * period
            CurrentIndices = list(filter(lambda i: list(LCMVs['Length'])[i] > CutoffMultiplier * Period, np.arange(len(LCMVs))))
            # Get maximum periods where the cutoff*period is still reached
            for Index in CurrentIndices:
                MaximumPeriods[Index] = Period
          
    # =============================================================================
    #     Find the starting LC to fill
    # =============================================================================
        # How many times max period is available before and after my CMV
        BeforeAftersSpace, BeforeAfters = [], []
        for Index, StartEnd in enumerate(LCStartEnds):
            # Calculate how much space there is before and after to run the ewm
            Start, End = StartEnd[0], StartEnd[1]
            PriorTemp = Data[Column][: Start]
            if len(list(PriorTemp[PriorTemp.isnull()].index)) > 0:
                FirstMVBefore = max(list(PriorTemp[PriorTemp.isnull()].index))
                SpaceBefore = Start - FirstMVBefore
            else:
                FirstMVBefore = -1
                SpaceBefore = Start
            
            PostTemp = Data[Column][End + 1:]
            if len(list(PostTemp[PostTemp.isnull()].index)) > 0:
                FirstMVAfter = min(list(PostTemp[PostTemp.isnull()].index))
                SpaceAfter = FirstMVAfter - End
            else:
                FirstMVAfter = len(Data)
                SpaceAfter = len(Data) - End - 1
                
            BeforeAftersSpace.append((SpaceBefore, SpaceAfter))
            BeforeAfters.append((SpaceBefore/MaximumPeriods[Index], SpaceAfter/MaximumPeriods[Index]))
        # Smallest available space before or after the CMV
        SmallestBeforeAfters = list(map(lambda List: min(List), BeforeAfters))
        
        LCData = pd.DataFrame({'LCStartEnds':LCStartEnds,'MaximumPeriods':MaximumPeriods,
                               'BeforeAftersSpace':BeforeAftersSpace, 'BeforeAfters':BeforeAfters,
                               'SmallestBeforeAfters': SmallestBeforeAfters})
        
        # Indices from largest available space to smallest
        IndexOrder = list(np.argsort(SmallestBeforeAfters))
        IndexOrder.reverse()
        
        LeftoverSCMVs = pd.DataFrame(columns = LCMVs.columns)
        
        AllPeriodsUsed = []
        
        for Iteration in range(len(LCMVs)):
            # Always fill the CMV with the most available space to fill
            LCIndex = np.argmax(LCData['SmallestBeforeAfters'])
            
            StartIndex = LCData['LCStartEnds'][LCIndex][0]
            EndIndex = LCData['LCStartEnds'][LCIndex][1]
            
            MVLength = EndIndex - StartIndex + 1
            
            SidesToUse = []
            PeriodsUsed = []
            PeriodStartIndices = []

            LeftPeriodsUsed = []
            RightPeriodsUsed = []
            
    # =============================================================================
    #         Seasonal Decomposition
    # =============================================================================
            for PeriodIndex, Period in enumerate(Periods):
                # Keep to the maximum period determined earlier
                if Period <= LCData['MaximumPeriods'][LCIndex]:
                    BeforePeriods = LCData['BeforeAftersSpace'][LCIndex][0] / Period
                    if LCData['BeforeAftersSpace'][LCIndex][0] % Period == 0:
                        BeforePeriods -= 1
                    NotEnoughBefore = BeforePeriods < 3
                    
                    AfterPeriods = LCData['BeforeAftersSpace'][LCIndex][1] / Period
                    if LCData['BeforeAftersSpace'][LCIndex][1] % Period == 0:
                        AfterPeriods -= 1
                    NotEnoughAfter = AfterPeriods < 3
                    
                    # Stop SD if I can no longer decompose because of a lack of data
                    if NotEnoughBefore and NotEnoughAfter:
                        break
                    else:
                        if LCData['BeforeAftersSpace'][LCIndex][0] > 0:
                            InterpolYBefore = Data[Column][StartIndex - int(np.min((1000, LCData['BeforeAftersSpace'][LCIndex][0]))): StartIndex].ewm(alpha=0.2).mean().iloc[-1]
                        if LCData['BeforeAftersSpace'][LCIndex][1] > 0:
                            InterpolYAfter = Data[Column][EndIndex + 1: EndIndex + 1 + int(np.min((1000, LCData['BeforeAftersSpace'][LCIndex][1])))].ewm(alpha=0.2).mean().iloc[0]

                        # print(f'\n\n\n\nIntial Interpol Values: {InterpolYBefore}, {InterpolYAfter}\n\n\n\n')

                        BeforePeriods = int(min((5, BeforePeriods)))
                        AfterPeriods = int(min((5, AfterPeriods)))
                        
                        CurrentSidesToUse = []
                        
                        if not NotEnoughBefore:
                            BeforeData = Data[Column][StartIndex - int(BeforePeriods) * Period: StartIndex]
                            
                            if PeriodIndex == 0:
                                BeforeSeasonals = []
                            
                            # Use weighing method to get seasonality, nearer periods having greater weight
                            for CurrentPeriods in np.arange(2, BeforePeriods + 1):
                                CurrentSlice = BeforeData[-(CurrentPeriods * Period):]

                                # I go through all previously used periods and deseasonalize my slice of data
                                if len(LeftPeriodsUsed) > 0:
                                    for PriorPeriod in LeftPeriodsUsed:
                                        Model = seasonal_decompose(CurrentSlice, period = PriorPeriod)
                                        CurrentSlice -= Model.seasonal

                                WeightMultiplier = BeforePeriods * 2 - 1 - 2 * (CurrentPeriods - 1)
                                BeforeDecomposition = seasonal_decompose(CurrentSlice, period = Period)

                                if CurrentPeriods == 2:
                                    SeasonalComponent = np.array(list(islice(cycle(BeforeDecomposition.seasonal), BeforePeriods * Period))) * WeightMultiplier / ((BeforePeriods -  1) ** 2)
                                else:
                                    SeasonalComponent += np.array(list(islice(cycle(BeforeDecomposition.seasonal), BeforePeriods * Period))) * WeightMultiplier / ((BeforePeriods -  1) ** 2)

                            BeforeSeasonals.append(SeasonalComponent)
                            
                            CurrentSidesToUse.append('Before')
                            UsedPeriod = Period
                            LeftPeriodsUsed.append(Period)
                            PeriodStartIndices.append(BeforeDecomposition.seasonal.index[0])
                            
                            # Forwardfill trend and use last value
                            TrendData = BeforeDecomposition.trend.ffill().bfill()
                            InterpolYBefore = TrendData[TrendData.index[-1]]
                        
                        if not NotEnoughAfter:
                            AfterData = Data[Column][EndIndex + 1: EndIndex + 1 + int(AfterPeriods) * Period]
                            # print(f'Column: {Column}, End Index: {EndIndex}, After Periods: {AfterPeriods}, Period: {Period}')
                            # print(f'Missing Values: {AfterData.isnull().sum()}')
                            
                            if PeriodIndex == 0:
                                AfterSeasonals = []

                            # Use weighing method to get seasonality, nearer periods having greater weight
                            for CurrentPeriods in np.arange(2, AfterPeriods + 1):
                                CurrentSlice = AfterData[: CurrentPeriods * Period + 1]

                                if len(LeftPeriodsUsed) > 0:
                                    for PostPeriod in RightPeriodsUsed:
                                        Model = seasonal_decompose(CurrentSlice, period = PostPeriod)
                                        CurrentSlice -= Model.seasonal

                                WeightMultiplier = AfterPeriods * 2 - 1 - 2 * (CurrentPeriods - 1)
                                AfterDecomposition = seasonal_decompose(CurrentSlice, period = Period)

                                if CurrentPeriods == 2:
                                    SeasonalComponent = np.array(list(islice(cycle(AfterDecomposition.seasonal), AfterPeriods * Period))) * WeightMultiplier / ((AfterPeriods -  1) ** 2)
                                else:
                                    SeasonalComponent += np.array(list(islice(cycle(AfterDecomposition.seasonal), AfterPeriods * Period))) * WeightMultiplier / ((AfterPeriods -  1) ** 2)

                            AfterSeasonals.append(SeasonalComponent)
                            
                            CurrentSidesToUse.append('After')
                            UsedPeriod = Period
                            RightPeriodsUsed.append(Period)
                            
                            # Backfill trend and use first value
                            TrendData = AfterDecomposition.trend.ffill().bfill()
                            InterpolYAfter = TrendData[TrendData.index[0]]
                            
                        SidesToUse.append(CurrentSidesToUse)
                        PeriodsUsed.append(UsedPeriod)
                        # The periods loop either reaches its end by having no more eligible periods or by going through all
                    
    # =============================================================================
    #       Impute the LC
    # =============================================================================
            if len(SidesToUse) > 0:
                # Seasonal Imputation
                for SideIndex, Sides in enumerate(SidesToUse):
                    ToImputeBeforeMultiplier = [0] * MVLength
                    ToImputeBefore = [0] * MVLength
                    ToImputeAfterMultiplier = [0] * MVLength
                    ToImputeAfter = [0] * MVLength
                    
                    if 'Before' in Sides:
                        # The before period starts shifted, I can not just repeat the data and sum it up
                        StartShift = StartIndex - PeriodStartIndices[SideIndex]
                        
                        ToImputeBefore = list(islice(cycle(BeforeSeasonals[SideIndex]),
                                                     MVLength + StartShift))
                        ToImputeBefore = ToImputeBefore[StartShift:]
                        
                        ToImputeBeforeMultiplier = [1] * MVLength
                        
                    if 'After' in Sides:
                        # For after I create a list of int * period length, and start from ith value
                            # This is because in the after seasonality the start point 
                        ImputationLength = math.ceil(MVLength / PeriodsUsed[SideIndex]) * PeriodsUsed[SideIndex]
                        IndexDifference = ImputationLength - MVLength
                        
                        ToImputeAfter = list(islice(cycle(AfterSeasonals[SideIndex]),ImputationLength))
                        ToImputeAfter = ToImputeAfter[IndexDifference:]
                        
                        ToImputeAfterMultiplier = [1] * MVLength
                        
                    if len(Sides) == 2:
                        ToImputeBeforeMultiplier = np.linspace(1,0,MVLength)
                        ToImputeAfterMultiplier = np.linspace(0,1,MVLength)
                        
                    if SideIndex == 0:
                        Imputation = pd.Series(ToImputeBeforeMultiplier) * pd.Series(ToImputeBefore)
                    else:
                        Imputation += pd.Series(ToImputeBeforeMultiplier) * pd.Series(ToImputeBefore)
                    Imputation += pd.Series(ToImputeAfterMultiplier) * pd.Series(ToImputeAfter)
                    
                SeasonalStartY = Imputation.iloc[0]
                SeasonalEndY = Imputation.iloc[-1]
                
                # Linear Interpolation
                if LCData['BeforeAftersSpace'][LCIndex][0] == 0:
                    InterpolYBefore = InterpolYAfter
                if LCData['BeforeAftersSpace'][LCIndex][1] == 0:
                    InterpolYAfter = InterpolYBefore

                LinearInterpolation = [np.nan] * (MVLength + 2)
                LinearInterpolation[0] = InterpolYBefore
                LinearInterpolation[-1] = InterpolYAfter
                LinearInterpolation = pd.Series(LinearInterpolation).interpolate()[1:-1]
                LinearInterpolation = LinearInterpolation.reset_index(drop=True)
                
                Imputation += LinearInterpolation
                
                Imputation = list(Imputation)
                
                Data.loc[StartIndex:EndIndex,Column] = Imputation
                
                AllPeriodsUsed.append(PeriodsUsed)
                
            else:
                # If no seasonal components were created, I need to do SC
                TempLeftover = pd.DataFrame({'Start Point': [StartIndex], 'Length': [MVLength]})
                LeftoverSCMVs = pd.concat((LeftoverSCMVs, TempLeftover), ignore_index = True)
                    
    # =============================================================================
    #       Recalculate next best LCMV to fill
    # =============================================================================
            LCData = LCData.drop(LCIndex).reset_index(drop=True)
              
        # =============================================================================
        #     Find the starting LC to fill
        # =============================================================================
            # How many times max period is available before and after my CMV
            BeforeAftersSpace, BeforeAfters = [], []
            for Index, StartEnd in enumerate(LCData['LCStartEnds']):
                # Calculate how much space there is before and after to run the ewm
                Start, End = StartEnd[0], StartEnd[1]
                PriorTemp = Data[Column][: Start]
                if len(list(PriorTemp[PriorTemp.isnull()].index)) > 0:
                    FirstMVBefore = max(list(PriorTemp[PriorTemp.isnull()].index))
                    SpaceBefore = Start - FirstMVBefore
                else:
                    FirstMVBefore = -1
                    SpaceBefore = Start
                
                PostTemp = Data[Column][End + 1:]
                if len(list(PostTemp[PostTemp.isnull()].index)) > 0:
                    FirstMVAfter = min(list(PostTemp[PostTemp.isnull()].index))
                    SpaceAfter = FirstMVAfter - End
                else:
                    FirstMVAfter = len(Data)
                    SpaceAfter = len(Data) - End
                    
                BeforeAftersSpace.append((SpaceBefore, SpaceAfter))
                BeforeAfters.append((SpaceBefore/MaximumPeriods[Index], SpaceAfter/MaximumPeriods[Index]))
            # Smallest available space before or after the CMV
            SmallestBeforeAfters = list(map(lambda List: min(List), BeforeAfters))
            
            LCData['BeforeAftersSpace'] = BeforeAftersSpace
            LCData['BeforeAfters'] = BeforeAfters
            LCData['SmallestBeforeAfters'] = SmallestBeforeAfters
            
        return {'Data': Data, 'Leftover SCMVs': LeftoverSCMVs, 'Periods Used': AllPeriodsUsed}
#%% Anomaly Detection and Rectification - Functions
def PELTPenalty(Samples, Sensitivity):
    # Sensitivity between 0 and 1
    # Sensitivity 0.5 corresponds to AIC
    return np.power(Samples/4, 1-Sensitivity) * 2 * np.log(Samples)

def SlopePenalty(Sensitivity):
    # Sensitivity between 0 and 1
    # Sensitivity 0.5 corresponds to AIC
    return np.power(6, 2-2*Sensitivity)

# Deseasonalizes a series within a dataframe
def EmergencyDeseasonalize(Dataframe, Columns, MaxSamples):
    for Column in Columns:
        RelevantPeriods, StrongPeriods = RelevantPeriodicities(Dataframe, 'Timestamp', Column, MaxSamples, 0.3)

        Periods = RelevantPeriods
        Periods.extend(StrongPeriods)
        
        if len(Periods) == 1:
            Period = Periods[0]
            Model = STL(Dataframe[Column],period=Period).fit()
            Deseasonalized = Model.trend + Model.resid
            Dataframe[Column] = Deseasonalized
        
        elif len(Periods) > 1:
            Model = MSTL(Dataframe[Column],periods=Periods).fit()
            Deseasonalized = Model.trend + Model.resid
            Dataframe[Column] = Deseasonalized
            
    return Dataframe[Columns]

def WindowSlidingSegmentation(Series, Window, MinSize, Jump, UsedStrictnesses, **kwargs):
    if 'Task' not in kwargs:
        kwargs['Task'] = 'Mean Shift'
        
    if kwargs['Task'] == 'Mean Shift':
        Model = 'l2'
    if kwargs['Task'] == 'Volatility Shift':
        Model = 'normal'
    
    ChangepointList = []
    StrictnessNames = []

    # BIC Strictness
    if 'BIC' in UsedStrictnesses:
        BIC = Series.var() * np.log10(len(Series))
        
        PreviousChangepoints = []

        for Multiplier in np.logspace(-4,2,19):
            ChangepointAlgorithm = rpt.Window(model=Model,width=Window,min_size=MinSize,jump=Jump).fit(np.array(Series))
            CurrentChangepoints = ChangepointAlgorithm.predict(pen=BIC*Multiplier)
            
            if set(CurrentChangepoints) != set(PreviousChangepoints):
                ChangepointList.append(CurrentChangepoints)
                PreviousChangepoints = CurrentChangepoints

                StrictnessNames.append('BIC '+'{:.2e}'.format(Multiplier))
                
            if len(CurrentChangepoints) == 0:
                break

    # LogSamples Strictness
    if 'Log of Samples' in UsedStrictnesses:
        Penalty = np.log10(len(Series))
        
        PreviousChangepoints = []

        for Multiplier in np.logspace(-4,2,19):
            ChangepointAlgorithm = rpt.Window(model=Model,width=Window,min_size=MinSize,jump=Jump).fit(np.array(Series))
            CurrentChangepoints = ChangepointAlgorithm.predict(pen=Penalty*Multiplier)
            
            if set(CurrentChangepoints) != set(PreviousChangepoints):
                ChangepointList.append(CurrentChangepoints)
                PreviousChangepoints = CurrentChangepoints

                StrictnessNames.append('Log of Samples '+'{:.2e}'.format(Multiplier))
                
            if len(CurrentChangepoints) == 0:
                break

    # Variance Strictness
    if 'Variance' in UsedStrictnesses:
        Penalty = Series.var()
        
        PreviousChangepoints = []

        for Multiplier in np.logspace(-4,2,19):
            ChangepointAlgorithm = rpt.Window(model=Model,width=Window,min_size=MinSize,jump=Jump).fit(np.array(Series))
            CurrentChangepoints = ChangepointAlgorithm.predict(pen=Penalty*Multiplier)
            
            if set(CurrentChangepoints) != set(PreviousChangepoints):
                ChangepointList.append(CurrentChangepoints)
                PreviousChangepoints = CurrentChangepoints

                StrictnessNames.append('Variance '+'{:.2e}'.format(Multiplier))
                
            if len(CurrentChangepoints) == 0:
                break

    # PELT Strictness
    if 'PELT Penalty' in UsedStrictnesses:
        PreviousChangepoints = []

        for Sensitivity in np.linspace(0,1,num=11):
            ChangepointAlgorithm = rpt.Window(model=Model,width=Window,min_size=MinSize,jump=Jump).fit(np.array(Series))
            CurrentChangepoints = ChangepointAlgorithm.predict(pen=PELTPenalty(len(Series), Sensitivity))
            
            if set(CurrentChangepoints) != set(PreviousChangepoints):
                ChangepointList.append(CurrentChangepoints)
                PreviousChangepoints = CurrentChangepoints

                StrictnessNames.append('PELT Penalty '+'{:.2e}'.format(Sensitivity))
                
            if len(CurrentChangepoints) == 0:
                break

    # Slope Strictness
    if 'Slope Penalty' in UsedStrictnesses:
        PreviousChangepoints = []

        for Sensitivity in np.linspace(0,1,num=11):
            ChangepointAlgorithm = rpt.Window(model=Model,width=Window,min_size=MinSize,jump=Jump).fit(np.array(Series))
            CurrentChangepoints = ChangepointAlgorithm.predict(pen=SlopePenalty(Sensitivity))
            
            if set(CurrentChangepoints) != set(PreviousChangepoints):
                ChangepointList.append(CurrentChangepoints)
                PreviousChangepoints = CurrentChangepoints

                StrictnessNames.append('Slope Penalty '+'{:.2e}'.format(Sensitivity))
                
            if len(CurrentChangepoints) == 0:
                break
        
    return ChangepointList, StrictnessNames

def ChangepointDirections(Changepoints, Series, Method, **kwargs):
# =============================================================================
#     See what direction the change happens when a changehold threshold is crossed
# =============================================================================
    # If std is within +-5% of series std, it is termed average, otherwise it is increased/decreased
    # If mean is with series mean +-1 std of series std it is termed average
    ValueCutoff = 5

    if Method == 'Level Shift':
        SeriesValue = np.mean(Series)
        SeriesSTD = np.std(Series)

        LowerLimit = SeriesValue - 1 * SeriesSTD
        UpperLimit = SeriesValue + 1 * SeriesSTD

    elif Method == 'Volatility Shift':
        SeriesValue = np.std(Series)

        LowerLimit = SeriesValue * (100 - ValueCutoff) / 100
        UpperLimit = SeriesValue * (100 + ValueCutoff) / 100

    for Index, EndIndex in enumerate(Changepoints):
        if Index == 0:
            StartIndex = 0
            
            Slice = Series[StartIndex: EndIndex]
            
            if Method == 'Level Shift':
                SliceValue = np.mean(Slice)
            elif Method == 'Volatility Shift':
                SliceValue = np.std(Slice)

            if SliceValue >= LowerLimit and SliceValue <= UpperLimit:
                FeatureIsAverage = np.repeat(True,len(Slice))
                FeatureIsIncreased = np.repeat(False,len(Slice))
            elif SliceValue > UpperLimit:
                FeatureIsAverage = np.repeat(False,len(Slice))
                FeatureIsIncreased = np.repeat(True,len(Slice))
            elif SliceValue < LowerLimit:
                FeatureIsAverage = np.repeat(False,len(Slice))
                FeatureIsIncreased = np.repeat(False,len(Slice))

            if SeriesValue != 0:
                FeatureRelative = np.repeat(SliceValue / SeriesValue, len(Slice))
            else:
                FeatureRelative = np.repeat((SliceValue + 1) / (SeriesValue + 1), len(Slice))

        else:
            StartIndex = Changepoints[Index - 1]
            
            Slice = Series[StartIndex: EndIndex]
            
            if Method == 'Level Shift':
                SliceValue = np.mean(Slice)
            elif Method == 'Volatility Shift':
                SliceValue = np.std(Slice)
            
            if SliceValue >= LowerLimit and SliceValue <= UpperLimit:
                FeatureIsAverage = np.append(FeatureIsAverage, np.repeat(True,len(Slice)))
                FeatureIsIncreased = np.append(FeatureIsIncreased, np.repeat(False,len(Slice)))
            elif SliceValue > UpperLimit:
                FeatureIsAverage = np.append(FeatureIsAverage, np.repeat(False,len(Slice)))
                FeatureIsIncreased = np.append(FeatureIsIncreased, np.repeat(True,len(Slice)))
            elif SliceValue < LowerLimit:
                FeatureIsAverage = np.append(FeatureIsAverage, np.repeat(False,len(Slice)))
                FeatureIsIncreased = np.append(FeatureIsIncreased, np.repeat(False,len(Slice)))

            if SeriesValue != 0:
                FeatureRelative = np.append(FeatureRelative, np.repeat(SliceValue / SeriesValue, len(Slice)))
            else:
                FeatureRelative = np.append(FeatureRelative, np.repeat((SliceValue + 1) / (SeriesValue + 1), len(Slice)))
            
    if 'Periodwise' in kwargs:
        if kwargs['Periodwise'] is True:
            FeatureIsAverage = np.repeat(FeatureIsAverage, kwargs['Period'])
            FeatureIsAverage = np.append(FeatureIsAverage,np.repeat(FeatureIsAverage[-1],
                                                                    len(kwargs['OriginalSeries'])-len(FeatureIsAverage)))
        
            FeatureIsIncreased = np.repeat(FeatureIsIncreased, kwargs['Period'])
            FeatureIsIncreased = np.append(FeatureIsIncreased,np.repeat(FeatureIsIncreased[-1],
                                                                        len(kwargs['OriginalSeries'])-len(FeatureIsIncreased)))
            
            FeatureRelative = np.repeat(FeatureRelative, kwargs['Period'])
            FeatureRelative = np.append(FeatureRelative,np.repeat(FeatureRelative[-1],len(kwargs['OriginalSeries'])-len(FeatureRelative)))
            
    FeatureIsDecreased = np.array(list(map(lambda x, y: not(x or y), FeatureIsAverage, FeatureIsIncreased)))

    return FeatureIsAverage, FeatureIsIncreased, FeatureIsDecreased, FeatureRelative

def PointAnomaliesFromChangepoints(Data,Changepoints,Detrend):
    PointAnomalies = []
    
    for Iteration, EndIndex in enumerate(Changepoints):
        if Iteration == 0:
            StartIndex = 0
        else:
            StartIndex = Changepoints[Iteration-1] + 1
        
        DataSlice = Data[StartIndex:EndIndex + 1]
        Indices = DataSlice.index
        DataSlice = np.array(DataSlice)
# =============================================================================
#         Each slice of changepoints is assumed to be linear, so subtract lin regression from value to get
#         point anomalies
# =============================================================================
        if Detrend is True:        
            DataIndices = np.arange(len(DataSlice)).reshape((-1, 1))
            
            LinearModel = LinearRegression()
            LinearModel.fit(DataIndices, DataSlice)
            ModelValues = LinearModel.predict(DataIndices)
            
            Detrended = pd.Series(DataSlice - ModelValues,index=Indices)
        else:        
            Detrended = pd.Series(DataSlice,index=Indices)
        
        IQRAnomalyModel = InterQuartileRangeAD(c=1.5)
        
        # Sometimes detrended can have NaNs, not sure what to do with this however
        if Detrended.isnull().sum() == 0:
            PointAnomalies.extend(list(IQRAnomalyModel.fit_detect(Detrended)))
        
    return PointAnomalies

def GetSeasonalData(TransformData, SeasonalCol, Period):
    # Prepare periodwise seasonal data
    Times, AbsSum, Skewness, Kurtosis = [], [], [], []
    i = 0
    
    while (i+1)*Period < len(TransformData):
        Times.append(TransformData.index[i*Period])
        AbsSum.append(np.sum(np.abs(TransformData[SeasonalCol][i*Period:(i+1)*Period])))
        Skewness.append(skew(TransformData[SeasonalCol][i*Period:(i+1)*Period]))
        Kurtosis.append(kurtosis(TransformData[SeasonalCol][i*Period:(i+1)*Period]))
        
        i += 1
        
    Skewness = np.array(Skewness)
    Kurtosis = np.array(Kurtosis)
    
    Shape = np.power(Skewness, 2) + np.power(Kurtosis, 2)
    
    SeasonalData = pd.DataFrame({'Timestamp':Times,'Absolute Sum':AbsSum,'Shape':Shape})
    
    return SeasonalData

def AmplitudeOutliers(SeasonalData, Period, Seasonal, SeasonalChangepoints):
    FullPower = np.array(list(np.repeat(SeasonalData['Absolute Sum'],Period)))
    NormalizedValues = np.array(Seasonal.iloc[:len(FullPower)]) / FullPower

    Indices = []

    for Index, SegmentEnd in enumerate(SeasonalChangepoints):
        if Index == 0:
            SegmentStart = 0
        else:
            SegmentStart = SeasonalChangepoints[Index - 1]
            
        DataSlice = NormalizedValues[SegmentStart * Period: SegmentEnd * Period]
        DataSliceReshaped = DataSlice.reshape(SegmentEnd-SegmentStart,Period)
        
        for PeriodElement in np.arange(Period):
            CurrentSlice = DataSliceReshaped[:,PeriodElement]
            
            Q1 = np.percentile(CurrentSlice, 25)
            Q3 = np.percentile(CurrentSlice, 75)
            IQR = Q3 - Q1
            Threshold = 1.5 * IQR
            TemporaryIndices = np.where((CurrentSlice < Q1 - Threshold) | (CurrentSlice > Q3 + Threshold))[0]
            CurrentIndices = (SegmentStart + TemporaryIndices) * Period + PeriodElement
            
            Indices.extend(CurrentIndices)
            
    SeasonalPointAnomalies = np.repeat(False, len(Seasonal))
    SeasonalPointAnomalies.put(Indices, True)
    
    return SeasonalPointAnomalies

def ShapeAnomalies(SeasonalData, SeasonalChangepoints, Seasonal, Period):
    SeasonalTimeIndexed = SeasonalData.copy()
    SeasonalTimeIndexed.set_index('Timestamp',inplace=True)

    # Use of shape 2, square sum, so that 0 values do not have a strong effect
    ShapeAnomalies = PointAnomaliesFromChangepoints(SeasonalTimeIndexed['Shape'], SeasonalChangepoints, False)

    ShapeAnomaliesFull = np.repeat(ShapeAnomalies,Period)
    ShapeAnomaliesFull = np.append(ShapeAnomaliesFull,np.repeat(False,len(Seasonal)-len(ShapeAnomaliesFull)))
    
    return ShapeAnomaliesFull

def NormalityAnomalies(Series, Window):
    # Normality Deviations
    ResidualRollingSkew = Series.rolling(window=Window).skew()
    SkewThreshold = ThresholdAD(high=2, low=-2)
    SkewAnomalies = list(SkewThreshold.detect(ResidualRollingSkew).fillna(False))

    ResidualRollingKurt = Series.rolling(window=Window).kurt()
    KurtThreshold = ThresholdAD(high=7, low=-7)
    KurtAnomalies = list(KurtThreshold.detect(ResidualRollingKurt).fillna(False))

    # Where either has an error is an error
    Anomalies = SkewAnomalies or KurtAnomalies
    
    Indices = [i for i, val in enumerate(Anomalies) if val]
    # Get normality changepoint indices
    Changepoints = []
    for k, g in itertools.groupby(enumerate(Indices), lambda x: x[0]-x[1]):
        CurrentSequence = list(map(itemgetter(1), g))
        Start = CurrentSequence[0]
        End = min(CurrentSequence[-1] + 1, len(Anomalies)-1)
        Changepoints.extend([Start,End])
    
    if (len(Anomalies) - 1) not in Changepoints:
        Changepoints.append(len(Anomalies)-1)
    
    IsError = np.repeat(False, len(Series))
    IsError.put(Changepoints, True)
        
    return IsError

# Get indices to remove under anomaly rectification mode and also what amount of data is removed
def GetDeletionIndices(ErrorIndicesInit, ErrorFeaturesInit, ADResult, Features):
    RequiresCorrection = pd.DataFrame({'Location': ErrorIndicesInit, 'Feature': ErrorFeaturesInit})

    ErrorLocations = sorted(list(set(ErrorIndicesInit)))
    AllFeatures, NFeatures = [], []

    # Taking care of repeat indices in my detected errors
    for ErrorLocation in ErrorLocations:    
        Indices = list(RequiresCorrection[RequiresCorrection['Location'] == ErrorLocation].index)
        
        IncludedFeatures = [RequiresCorrection['Feature'].iloc[i] for i in Indices]
        IncludedFeatures = sorted(list(set(IncludedFeatures)))
        
        AllFeatures.append(IncludedFeatures)
        NFeatures.append(len(IncludedFeatures))
        
    RequiresCorrection = pd.DataFrame({'Location': ErrorLocations, 'Feature': AllFeatures, 'Number of Features': NFeatures})

    RemovalIndices = []
    RemovedAmounts = []

    for Feature in Features:
        ToRemove = [RequiresCorrection['Location'][i] for i, val in enumerate(RequiresCorrection['Feature']) if Feature in val]
        
        ToRemoveBool = np.repeat(False, len(ADResult))
        ToRemoveBool[ToRemove] = True

        RemovalIndices.append(list(ToRemoveBool))
        RemovedAmounts.append(np.round(sum(ToRemoveBool) / len(ADResult)*100, 2))

    return {'Remove Index': RemovalIndices, 'Removed Percentages': RemovedAmounts}

def GetErrorDeletionMVEvalDF(Data):
    MVDF = MissingValueEvaluator(Data)['MVDataFrame']
    
    MVDFToVis = pd.DataFrame({'Start Point': MVDF['Start Point'], 'Length': MVDF['Length']})

    MVEval = MissingValueEvaluator(Data)['MVEvaluation']
    MVEvalToVis = pd.DataFrame({'Feature': list(MVEval.index), 'Point MV Instances in Feature': MVEval['Point Error Instances in Feature'],
                                'Continuous MV Instances in Feature': MVEval['Continuous Error Instances in Feature'],
                                'Percent of Data as Continuous MV in Feature': MVEval['Percent of All Data as Continuous Errors in Feature']})
    
    MVCols = MissingValueEvaluator(Data)['MVCols']

    return {'MVDataFrame': MVDF, 'MVEvaluation': MVEval, 'MVDFToVis': MVDFToVis, 'MVEvalToVis': MVEvalToVis, 'MVCols': MVCols}
#%% Data Input - Reactive Values, Effects, and Calculations
FilePath = reactive.Value()
FileName = reactive.Value()

CurrentCSV = reactive.Value(pd.DataFrame())
CurrentCSVRendered = reactive.Value(pd.DataFrame())
CurrentDTypes = reactive.Value(pd.DataFrame())

PeriodicityDataFrame = reactive.Value(pd.DataFrame())
PeriodicityDataFrameReadable = reactive.Value(pd.DataFrame())

IVSliderCVSingle = reactive.Value(0)
IVSliderCVPair = reactive.Value(0)

PeriodicityRuntime = reactive.Value()

# Initialize CSV
@reactive.Effect
@reactive.event(input.CSVLocation, input.SepText, input.ResetCSV, input.CSVHeader)
def InitializeCSV():
    if input.CSVLocation:
        file: list[FileInfo] | None = input.CSVLocation()

        if file is not None:
            FilePath.set(file[0]["datapath"])
            FileName.set(file[0]["name"])
    
    if len(input.SepText()) == 0:
        Separator = ","
    else:
        Separator = input.SepText()
    
    Data = pd.read_csv(FilePath.get(), sep = Separator, low_memory=False, header=input.CSVHeader() - 1)
    Columns = list(Data.columns)
    DTypes = list(Data.dtypes.astype(str))
    DTypeData = pd.DataFrame({'Column': Columns,'Data Type': DTypes})
    
    CurrentCSV.set(Data)
    CurrentCSVRendered.set(Data)
    CurrentDTypes.set(DTypeData)

# Create headers
@reactive.Effect
@reactive.event(input.CreateHeaders)
def DeleteCSVCols():
    Data = CurrentCSV.get().copy()
    Columns = list(Data.columns)

    for ColIndex, OldName in enumerate(Columns):
        Data = Data.rename(columns = {OldName: f'Column {ColIndex + 1}'})
    
    Columns = list(Data.columns)
    DTypes = list(Data.dtypes.astype(str))
    DTypeData = pd.DataFrame({'Column': Columns,'Data Type': DTypes})

    CurrentCSV.set(Data)
    CurrentCSVRendered.set(Data)
    CurrentDTypes.set(DTypeData)

# Delete selected columns
@reactive.Effect
@reactive.event(input.DeleteCols)
def DeleteCSVCols():
    Data = CurrentCSV.get().copy()
    Columns = list(Data.columns)

    if len(input.RenderCSVDtypes_selected_rows()) > 0:
        ToDelete = input.RenderCSVDtypes_selected_rows()

        for ColNum in ToDelete:
            Data = Data.drop([Columns[ColNum]], axis=1)
    
    Columns = list(Data.columns)
    DTypes = list(Data.dtypes.astype(str))
    DTypeData = pd.DataFrame({'Column': Columns,'Data Type': DTypes})

    CurrentCSV.set(Data)
    CurrentCSVRendered.set(Data)
    CurrentDTypes.set(DTypeData)

# Combine timestamp columns
@reactive.Effect
@reactive.event(input.CombineCols)
def CombineDatetimeCols():
    Data = CurrentCSV.get().copy()
    Columns = list(Data.columns)

    if len(input.RenderCSVDtypes_selected_rows()) > 0:
        ToCombine = input.RenderCSVDtypes_selected_rows()
        for Index, ColNum in enumerate(ToCombine):
            if Index == 0:
                Data['Timestamp'] = list(map(lambda x: str(x),
                                                np.array(Data[Columns[ColNum]])))
            else: 
                Data['Timestamp'] = list(map(lambda x, y: x + '-' + str(y),
                                                Data['Timestamp'],np.array(Data[Columns[ColNum]])))
            if Columns[ColNum] != 'Timestamp':
                Data = Data.drop([Columns[ColNum]],axis=1)

    Timestamp = Data.pop('Timestamp') 
    Data.insert(0, 'Timestamp', Timestamp)
    
    Columns = list(Data.columns)
    DTypes = list(Data.dtypes.astype(str))
    DTypeData = pd.DataFrame({'Column': Columns,'Data Type': DTypes})

    CurrentCSV.set(Data)
    CurrentCSVRendered.set(Data)
    CurrentDTypes.set(DTypeData)

# Resample csv dataframe
@reactive.Effect
@reactive.event(input.DoResampleDataframe)
def ResampleInputCSV():
    Data = CurrentCSV.get().copy()

    if 'Timestamp' in Data.columns:
        ResUnit = input.ResFreqUnit()
        ResValue = input.ResFreqValue()

        ResFrequency = f'{ResValue}{ResUnit}'

        Data = Data.resample(ResFrequency, on='Timestamp').mean().reset_index()
        
        Columns = list(Data.columns)
        DTypes = list(Data.dtypes.astype(str))
        DTypeData = pd.DataFrame({'Column': Columns,'Data Type': DTypes})

        CurrentCSV.set(Data)

        RenderData = Data.copy()
        RenderData['Timestamp'] = [str(x) for x in list(RenderData['Timestamp'])]

        MVImputedCSV.set(Data.copy())

        CurrentCSVRendered.set(RenderData)
        CurrentDTypes.set(DTypeData)

@reactive.Effect
@reactive.event(input.InitVisMaxSamplesSingle)
def GetIVSliderValueSingle():
    CV = input.InitVisStartSingle()

    if CV > len(GetVisResampledDataSingle())-input.InitVisMaxSamplesSingle():
        CV = len(GetVisResampledDataSingle())-input.InitVisMaxSamplesSingle()
    else:
        CV = input.InitVisStartSingle()

    IVSliderCVSingle.set(CV)

@reactive.Effect
@reactive.event(input.InitVisMaxSamplesPair)
def GetIVSliderValuePair():
    CV = input.InitVisStartPair()

    if CV > len(GetVisResampledDataPair())-input.InitVisMaxSamplesPair():
        CV = len(GetVisResampledDataPair())-input.InitVisMaxSamplesPair()
    else:
        CV = input.InitVisStartPair()

    IVSliderCVPair.set(CV)

@reactive.Calc
def GetVisColumns():
    Columns = list(CurrentCSV.get().columns)
    Columns.remove('Timestamp')
    return Columns

@reactive.Calc
def GetVisResampleRates():
    # Get readable and computer useable resample rates for plotting, very noobish 50 line code
    Data = CurrentCSV.get().copy()
    
    SampleRateOriginal = Data['Timestamp'][1] - Data['Timestamp'][0]
    
    # Get first index where the resample rate is less frequent than the original
    Index = next(idx for idx, value in enumerate(SampleRateListCompare) if pd.to_timedelta(value) > SampleRateOriginal)
    
    SampleRateOriginal = to_offset(SampleRateOriginal).freqstr.replace('T','min')
    if SampleRateOriginal[0].isalpha():
        SampleRateOriginal = '1' + SampleRateOriginal

    ResampleRates = [SampleRateOriginal]
    # Output is a list of downsample rates starting with original sample rate
    ResampleRates.extend(SampleRateList[Index:])
    ResampleRatesReadable = []

    for value in ResampleRates:
        Current = str(re.sub('(\d+(\.\d+)?)', r' \1 ', value))
        Current = Current[1:]

        if Current[-3:] == ' ms':
            Current = Current.replace('ms','Milliseconds')
        if Current[-2:] == ' s':
            Current = Current.replace('s','Seconds')
        if Current[-4:] == ' min':
            Current = Current.replace('min','Minutes')
        if Current[-2:] == ' h':
            Current = Current.replace('h','Hours')
        if Current[-2:] == ' d':
            Current = Current.replace('d','Days')
        if Current[-2:] == ' w':
            Current = Current.replace('w','Weeks')
        if Current[-2:] == ' M':
            Current = Current.replace('M','Months')
        if Current[-2:] == ' Y':
            Current = Current.replace('Y','Years')
        
        if Current[:2] == '1 ':
            Current = Current[:-1]

        ResampleRatesReadable.append(Current)
    return {'Readable':ResampleRatesReadable, 'Computable':ResampleRates}

@reactive.Calc
def GetVisResampledDataPair():
    Data = CurrentCSV.get().copy()
    ResampleRate = GetVisResampleRates()['Computable'][GetVisResampleRates()['Readable'].index(input.InitVisResRatePair())]
    
    Data = Data.resample(ResampleRate,on='Timestamp').mean()
    return Data

@reactive.Calc
def GetVisResampledDataSingle():
    Data = CurrentCSV.get().copy()
    ResampleRate = GetVisResampleRates()['Computable'][GetVisResampleRates()['Readable'].index(input.InitVisResRateSingle())]
    
    Data = Data.resample(ResampleRate,on='Timestamp').mean()
    return Data
#%% Missing Value Handling - Reactive Values, Effects, and Calculations
MVImputedCSV = reactive.Value(pd.DataFrame())

# Get Missing Value Dataframe
@reactive.Calc
def GetMVEvalDF():
    MVDF = MissingValueEvaluator(CurrentCSV.get())['MVDataFrame']
    
    MVDFToVis = pd.DataFrame({'Start Point': MVDF['Start Point'], 'Length': MVDF['Length']})

    MVEval = MissingValueEvaluator(CurrentCSV.get())['MVEvaluation']
    MVEvalToVis = pd.DataFrame({'Feature': list(MVEval.index), 'Point MV Instances in Feature': MVEval['Point Error Instances in Feature'],
                                'Continuous MV Instances in Feature': MVEval['Continuous Error Instances in Feature'],
                                'Percent of Data as Continuous MV in Feature': MVEval['Percent of All Data as Continuous Errors in Feature']})
    
    MVCols = MissingValueEvaluator(CurrentCSV.get())['MVCols']

    return {'MVDataFrame': MVDF, 'MVEvaluation': MVEval, 'MVDFToVis': MVDFToVis, 'MVEvalToVis': MVEvalToVis, 'MVCols': MVCols}
#%% Data Transformation - Reactive Values, Effects, and Calculations
# Create resampling dataframe, with runtime estimates and used periods
@reactive.Calc
def CalculateResamplingDataframe():
    OriginalLength = len(CurrentCSV.get())
    STLSlope = 1.39e-6
    MSTLSlope = 2.46e-6

    Data = PeriodicityDataFrame.get().copy()

    if 'Statistically Relevant Strong Periodicities' in Data.columns:
        AllPeriodicities = []

        for Periodicities in Data['Statistically Relevant Strong Periodicities'].tolist():
            AllPeriodicities.extend(Periodicities)
                    
        UniquePeriodicities = sorted(list(set(AllPeriodicities)))

        Divisors = []
        Runtimes = []
        LostPeriodicityList = []
        NewSampleRates = []
        NewSampleRatesTimedelta = []
        UsedPeriods = []
        NewDataAmounts = []
        RuntimeComponentList = []

        SampleRateOriginal = CurrentCSV.get()['Timestamp'][1] - CurrentCSV.get()['Timestamp'][0]

        for Divisor in np.arange(1, max(UniquePeriodicities) + 1):
            # I may lose a periodicity if I resample, this may be up to the user to see if they want to do it
            PeriodsSubset = sorted(i for i in UniquePeriodicities if i > Divisor)
            
            IsDivisorList = []
            for Period in PeriodsSubset:
                IsDivisorList.append(Period % Divisor == 0)
            
            if (False not in IsDivisorList) and (len(IsDivisorList) > 0):
                Divisors.append(Divisor)
                
                PeriodsLost = sorted(i for i in UniquePeriodicities if i <= Divisor)
                PeriodsLost = list(map(lambda x: x * SampleRateOriginal, PeriodsLost))

                NewSampleRatesTimedelta.append(Divisor * SampleRateOriginal)
                NewSampleRates.append(str(Divisor * SampleRateOriginal))
                NewDataAmounts.append(OriginalLength // Divisor)
                
                if len(PeriodsLost) > 0:
                    PeriodsLostStr = str(PeriodsLost[0])
                    if len(PeriodsLost) > 1:
                        for i in np.arange(1, len(PeriodsLost)):
                            PeriodsLostStr += ', ' + str(PeriodsLost[i])
                else:
                    PeriodsLostStr = 'No Periods Lost'
                    
                LostPeriodicityList.append(PeriodsLostStr)
                
                Runtime = 0
                PeriodsPerColumn = []
                RuntimeComponents = []
                
                for ColumnPeriodicities in Data['Statistically Relevant Strong Periodicities'].tolist():
                    ColPerSubset = sorted(i for i in ColumnPeriodicities if i > Divisor)
                    PeriodsPerColumn.append(list(np.array(ColPerSubset) // Divisor))
                    
                    if len(ColPerSubset) == 0:
                        RuntimeComponents.append(0)

                    if len(ColPerSubset) == 1:
                        Runtime += STLSlope * OriginalLength * sum(ColPerSubset) / (Divisor ** 2)
                        RuntimeComponents.append(STLSlope * OriginalLength * sum(ColPerSubset) / (Divisor ** 2))
                        
                    elif len(ColPerSubset) > 1:
                        Runtime += MSTLSlope * OriginalLength * sum(ColPerSubset) / (Divisor ** 2)
                        RuntimeComponents.append(MSTLSlope * OriginalLength * sum(ColPerSubset) / (Divisor ** 2))
                        
                UsedPeriods.append(PeriodsPerColumn)

                RuntimeComponentList.append(RuntimeComponents)
                
                Runtime = timedelta(minutes = Runtime // 60)
                if Runtime < timedelta(minutes = 0):
                    Runtime = timedelta(minutes = 0)
                Runtimes.append(str(Runtime))
                
        ResamplingDataframe = pd.DataFrame({'Downsampling': Divisors, 'New Sampling Rate': NewSampleRates, 'Samples in Resampled Dataset': NewDataAmounts,
                                            'New Timedelta Sampling Rate': NewSampleRatesTimedelta, 'Runtime Estimate': Runtimes,
                                            'Runtime Components': RuntimeComponentList, 'Periods Lost': LostPeriodicityList, 'Used Periods': UsedPeriods})
        
        return ResamplingDataframe
    else:
        return pd.DataFrame()
    
@reactive.Calc
def GetDataTransformData():
    ResampleData = CalculateResamplingDataframe().iloc[input.RenderDataTransformResampleDataframe_selected_rows()[0]]
    ResampleRate = ResampleData['New Timedelta Sampling Rate']
    RuntimeComponents = ResampleData['Runtime Components']
    RuntimeSum = sum(RuntimeComponents)
    if RuntimeSum == 0:
        RuntimeSum = 1
    UsedPeriods = ResampleData['Used Periods']
    
    ResampleData = pd.Series({'Resample Rate': ResampleRate, 'Runtime Components': RuntimeComponents, 'Runtime Sum': RuntimeSum, 'Used Periods': UsedPeriods})
    
    return ResampleData
#%% Anomaly Detection and Rectification - Reactive Values, Effects, and Calculations
DataTransforms = reactive.Value([])
DataTransformColumns = reactive.Value([])
DataAnomalies = reactive.Value([])

ADFeaturePlotData = reactive.Value(pd.DataFrame())
DataAnomaliesFiltered = reactive.Value(pd.DataFrame())
AnomalyRemovalIndexDataframe = reactive.Value(pd.DataFrame())

OriginalFeatureData = reactive.Value(pd.DataFrame())
ErrorUncorrectedData = reactive.Value(pd.DataFrame())
ErrorCorrectedData = reactive.Value(pd.DataFrame())

ADInitVisSelectedComp = reactive.Value('None')
ADInitVisSelectedPenType = reactive.Value('None')
ADInitVisSelectedPenValue = reactive.Value('None')
ADInitVisSelectedErrors = reactive.Value('None')

# Read Transformed CSV's
@reactive.Effect
@reactive.event(input.DataTransformFiles)
def ReadDataTransformData():
    Files: list[FileInfo] | None = input.DataTransformFiles()

    Columns = []
    AllTransforms = []

    if Files is not None:
        for File in Files:
            Data = pd.read_csv(File['datapath'])
            Column = File['name'].split('.')[0]

            Data['Timestamp'] = pd.to_datetime(Data['Timestamp'])

            AllTransforms.append(Data)
            Columns.append(Column.split(' - ')[0])

    DataTransforms.set(AllTransforms)
    DataTransformColumns.set(Columns)

@reactive.Effect
@reactive.event(input.AnomalyDetectFiles)
def ReadAnomalyDetectDataUnfiltered():
    DataFiles: list[FileInfo] | None = input.DataTransformFiles()
    AnomalyFiles: list[FileInfo] | None = input.AnomalyDetectFiles()

    Columns = []

    AllTransforms = []
    AllAnomalies = []

    if DataFiles is not None:
        if AnomalyFiles is not None:
            for DataFile in DataFiles:
                DataFileName = DataFile['name'].split('.')[0].split(' - ')[0]
                for AnomalyFile in AnomalyFiles:
                    AnomalyFileName = AnomalyFile['name'].split('.')[0].split(' - ')[0]

                    if DataFileName == AnomalyFileName:
                        Columns.append(DataFileName)

                        Data = pd.read_csv(DataFile['datapath'])
                        Data['Timestamp'] = pd.to_datetime(Data['Timestamp'])
                        AllTransforms.append(Data)

                        AnomalyData = pd.read_csv(AnomalyFile['datapath'])
                        AnomalyData['Timestamp'] = pd.to_datetime(AnomalyData['Timestamp'])
                        AllAnomalies.append(AnomalyData)

    DataTransformColumns.set(Columns)
    DataTransforms.set(AllTransforms)
    DataAnomalies.set(AllAnomalies)

@reactive.Effect
@reactive.event(input.FinalizedAnomalyDetectionData)
def ReadAnomalyDetectDataFiltered():
    DetectionResultFiles: list[FileInfo] | None = input.FinalizedAnomalyDetectionData()

    if DetectionResultFiles is not None:
        for DetectionResultFile in DetectionResultFiles:
            DetectionResult = pd.read_csv(DetectionResultFile['datapath'])
            DetectionResult['Timestamp'] = pd.to_datetime(DetectionResult['Timestamp'])

            DataAnomaliesFiltered.set(DetectionResult)

# Generate to anomaly feature plot data based on settings
@reactive.Effect
@reactive.event(input.ADFeatVisColumnAllOrSelect, input.ADFeatVisSelectedColumns,
                input.ADFeatVisErrorAllOrSelect, input.ADFeatVisErrorCategory, input.ADFeatVisSelectedErrors,
                input.ADFeatVisTrendPenType, input.ADFeatVisTrendPenMin, input.ADFeatVisTrendPenMax,
                input.ADFeatVisSeasonalPenType, input.ADFeatVisSeasonalPenMin, input.ADFeatVisSeasonalPenMax,
                input.ADFeatVisResidualPenType, input.ADFeatVisResidualPenMin, input.ADFeatVisResidualPenMax,
                input.ADFeatVisPlotType)
def GetADFeaturePlotData():
    # Select Plot Data
    if 'Boolean' in input.ADFeatVisPlotType():
        if input.ADFeatVisErrorAllOrSelect() == 'All':
            if input.ADFeatVisErrorCategory() == 'Point':
                ErrorCategories = ['Trend Point Anomalies', 'Amplitude Point Anomalies', 'Residual Point Anomalies', 'Residual Normality Anomalies']
            elif input.ADFeatVisErrorCategory() == 'Continuous':
                ErrorCategories = ['Trend Level Average', 'Trend Level Increased', 'Trend Level Decreased', 'Amplitude Level Average',
                                    'Amplitude Level Increased', 'Amplitude Level Decreased', 'Shape Anomalies', 'Residual Volatility Average',
                                    'Residual Volatility Increased', 'Residual Volatility Decreased']
        
        elif input.ADFeatVisErrorAllOrSelect() == 'Selected':
            ErrorCategories = input.ADFeatVisSelectedErrors()

    elif 'Relative' in input.ADFeatVisPlotType():
        ErrorCategories = ['Trend Level Relative','Amplitude Level Relative','Residual Volatility Relative']

    if input.ADFeatVisColumnAllOrSelect() == 'All':
        RelevantFeatures = DataTransformColumns.get()
    else:
        RelevantFeatures = input.ADFeatVisSelectedColumns()

    FeatureErrorDictionary = {}

    for FeatureIndex, Feature in enumerate(RelevantFeatures):
        ColumnIndex = DataTransformColumns.get().index(Feature)

        FeatureAnomalies = DataAnomalies.get()[ColumnIndex].copy()

        if FeatureIndex == 0:
            FeatureErrorDictionary['Timestamp'] = FeatureAnomalies['Timestamp']

        Components = list(FeatureAnomalies.columns)
        Components.remove('Timestamp')

        Components = list(map(lambda x: x.split(' ')[0], Components))
        Components = list(set(Components))

        HasSeasonalComponent = False

        SeasonalCols = [x for x in Components if x.startswith('Seasonal')]
        if len(SeasonalCols) > 0:
            Periods = list(map(lambda x: int(x.split('_')[-1]), SeasonalCols))
            Periods = sorted(Periods)

            Components = ['Trend']
            for Period in Periods:
                Components.append(f'Seasonal_{Period}')
            Components.append('Residual')

            HasSeasonalComponent = True

        else:
            Components = ['Trend', 'Residual']

        # Trend Error Components
        Component = 'Trend'
        PenaltyType = input.ADFeatVisTrendPenType()
        PenaltyMin = float(input.ADFeatVisTrendPenMin())
        PenaltyMax = float(input.ADFeatVisTrendPenMax())

        # BIC, Variance, and Log of Samples has the strictest penalty be the one of the smallest value, while with the other penalty types it's the highest
        if PenaltyType == 'BIC' or PenaltyType == 'Log of Samples' or PenaltyType == 'Variance':
            UseMin = True
        else:
            UseMin = False

        AllColumns = list(FeatureAnomalies.columns)
        RelevantColumns = [x for x in AllColumns if Component in x and PenaltyType in x]
        PenaltyValues = [float(x.split(' - ')[-1].split(' ')[-2]) for x in RelevantColumns]
        PenaltyValues = list(set(PenaltyValues))

        # If my set penalty value is in the list of all available ones, use that penalty, otherwise use the closest one
        if UseMin:
            if PenaltyMin in PenaltyValues:
                StrictestIndex = PenaltyValues.index(PenaltyMin)
            else:
                # With use minimum penalty value, I use the penalty that is smaller than the cutoff if available
                StrictestIndex = [i for i, val in enumerate(PenaltyValues) if val < PenaltyMin]
                
                if len(StrictestIndex) > 0:
                    StrictestIndex = StrictestIndex[-1]
                else:
                    StrictestIndex = 0
        else:
            if PenaltyMax in PenaltyValues:
                StrictestIndex = PenaltyValues.index(PenaltyMax)
            else:
                # With use maximum penalty value, I use the penalty that is larger than the cutoff if available
                StrictestIndex = [i for i, val in enumerate(PenaltyValues) if val > PenaltyMax]
                
                if len(StrictestIndex) > 0:
                    StrictestIndex = StrictestIndex[0]
                else:
                    StrictestIndex = len(PenaltyValues) - 1

        if len(PenaltyValues) > 0:
            PenaltyValue = '{:.2e}'.format(PenaltyValues[StrictestIndex])

            for ErrorCategory in ErrorCategories:
                ColumnText = f'{ErrorCategory} - {PenaltyType} {PenaltyValue} Strictness'
                
                if ColumnText in RelevantColumns:
                    FeatureErrorDictionary[f'{Feature} - {ErrorCategory}'] = FeatureAnomalies[ColumnText]

        # Seasonal Error Components
        if HasSeasonalComponent:
            PenaltyType = input.ADFeatVisSeasonalPenType()
            PenaltyMin = float(input.ADFeatVisSeasonalPenMin())
            PenaltyMax = float(input.ADFeatVisSeasonalPenMax())

            for Period in Periods:
                Component = f'Seasonal_{Period}'

                # BIC, Variance, and Log of Samples has the strictest penalty be the one of the smallest value, while with the other penalty types it's the highest
                if PenaltyType == 'BIC' or PenaltyType == 'Log of Samples' or PenaltyType == 'Variance':
                    UseMin = True
                else:
                    UseMin = False

                AllColumns = list(FeatureAnomalies.columns)
                RelevantColumns = [x for x in AllColumns if Component in x and PenaltyType in x]
                PenaltyValues = [float(x.split(' - ')[-1].split(' ')[-2]) for x in RelevantColumns]
                PenaltyValues = list(set(PenaltyValues))

                # If my set penalty value is in the list of all available ones, use that penalty, otherwise use the closest one
                if UseMin:
                    if PenaltyMin in PenaltyValues:
                        StrictestIndex = PenaltyValues.index(PenaltyMin)
                    else:
                        # With use minimum penalty value, I use the penalty that is smaller than the cutoff if available
                        StrictestIndex = [i for i, val in enumerate(PenaltyValues) if val < PenaltyMin]
                        
                        if len(StrictestIndex) > 0:
                            StrictestIndex = StrictestIndex[-1]
                        else:
                            StrictestIndex = 0
                else:
                    if PenaltyMax in PenaltyValues:
                        StrictestIndex = PenaltyValues.index(PenaltyMax)
                    else:
                        # With use maximum penalty value, I use the penalty that is larger than the cutoff if available
                        StrictestIndex = [i for i, val in enumerate(PenaltyValues) if val > PenaltyMax]
                        
                        if len(StrictestIndex) > 0:
                            StrictestIndex = StrictestIndex[0]
                        else:
                            StrictestIndex = len(PenaltyValues) - 1

                if len(PenaltyValues) > 0:
                    PenaltyValue = '{:.2e}'.format(PenaltyValues[StrictestIndex])

                    for ErrorCategory in ErrorCategories:
                        ColumnText = f'{Component} {ErrorCategory} - {PenaltyType} {PenaltyValue} Strictness'
                        
                        if ColumnText in RelevantColumns:
                            FeatureErrorDictionary[f'{Feature} - {Component} {ErrorCategory}'] = FeatureAnomalies[ColumnText]

        # Residual Error Components
        Component = 'Residual'
        PenaltyType = input.ADFeatVisResidualPenType()
        PenaltyMin = float(input.ADFeatVisResidualPenMin())
        PenaltyMax = float(input.ADFeatVisResidualPenMax())

        # BIC, Variance, and Log of Samples has the strictest penalty be the one of the smallest value, while with the other penalty types it's the highest
        if PenaltyType == 'BIC' or PenaltyType == 'Log of Samples' or PenaltyType == 'Variance':
            UseMin = True
        else:
            UseMin = False

        AllColumns = list(FeatureAnomalies.columns)
        RelevantColumns = [x for x in AllColumns if Component in x and PenaltyType in x]
        PenaltyValues = [float(x.split(' - ')[-1].split(' ')[-2]) for x in RelevantColumns]
        PenaltyValues = list(set(PenaltyValues))
        
        # If my set penalty value is in the list of all available ones, use that penalty, otherwise use the closest one
        if UseMin:
            if PenaltyMin in PenaltyValues:
                StrictestIndex = PenaltyValues.index(PenaltyMin)
            else:
                # With use minimum penalty value, I use the penalty that is smaller than the cutoff if available
                StrictestIndex = [i for i, val in enumerate(PenaltyValues) if val < PenaltyMin]
                
                if len(StrictestIndex) > 0:
                    StrictestIndex = StrictestIndex[-1]
                else:
                    StrictestIndex = 0
        else:
            if PenaltyMax in PenaltyValues:
                StrictestIndex = PenaltyValues.index(PenaltyMax)
            else:
                # With use maximum penalty value, I use the penalty that is larger than the cutoff if available
                StrictestIndex = [i for i, val in enumerate(PenaltyValues) if val > PenaltyMax]
                
                if len(StrictestIndex) > 0:
                    StrictestIndex = StrictestIndex[0]
                else:
                    StrictestIndex = len(PenaltyValues) - 1

        if len(PenaltyValues) > 0:
            PenaltyValue = '{:.2e}'.format(PenaltyValues[StrictestIndex])

            for ErrorCategory in ErrorCategories:
                ColumnText = f'{ErrorCategory} - {PenaltyType} {PenaltyValue} Strictness'
                
                if ColumnText in RelevantColumns:
                    FeatureErrorDictionary[f'{Feature} - {ErrorCategory}'] = FeatureAnomalies[ColumnText]

    ADFeaturePlotData.set(pd.DataFrame(FeatureErrorDictionary))

# Set initialized value of UI elements
@reactive.Effect
@reactive.event(input.ADColToVis)
def SetADInitVisComp():
    ADInitVisSelectedComp.set(input.ADCompToVis())

@reactive.Effect
@reactive.event(input.ADColToVis, input.ADCompToVis)
def SetADInitVisPenType():
    ADInitVisSelectedPenType.set(input.ADVisStrictnessType())

@reactive.Effect
@reactive.event(input.ADColToVis, input.ADCompToVis, input.ADVisStrictnessType)
def SetADInitVisPenValue():
    ADInitVisSelectedPenValue.set(input.ADVisStrictnessValue())

@reactive.Effect
@reactive.event(input.ADColToVis, input.ADCompToVis, input.ADVisStrictnessType, input.ADVisStrictnessValue)
def SetADInitVisError():
    ADInitVisSelectedErrors.set(input.ADVisErrors())

# Get Transformed Data Resample Rates
@reactive.Calc
def GetVisResampleRatesTransform():
    if len(DataTransforms.get()) > 0:
        ColNum = DataTransformColumns.get().index(input.TransformColToVis())
        Data = DataTransforms.get()[ColNum].copy()
        
        SampleRateOriginal = Data['Timestamp'][1] - Data['Timestamp'][0]
        
        # Get first index where the resample rate is less frequent than the original
        Index = next(idx for idx, value in enumerate(SampleRateListCompare) if pd.to_timedelta(value) > SampleRateOriginal)
        
        SampleRateOriginal = to_offset(SampleRateOriginal).freqstr.replace('T','min')
        if SampleRateOriginal[0].isalpha():
            SampleRateOriginal = '1' + SampleRateOriginal

        ResampleRates = [SampleRateOriginal]
        # Output is a list of downsample rates starting with original sample rate
        ResampleRates.extend(SampleRateList[Index:])
        ResampleRatesReadable = []

        for value in ResampleRates:
            Current = str(re.sub('(\d+(\.\d+)?)', r' \1 ', value))
            Current = Current[1:]

            if Current[-3:] == ' ms':
                Current = Current.replace('ms','Milliseconds')
            if Current[-2:] == ' s':
                Current = Current.replace('s','Seconds')
            if Current[-4:] == ' min':
                Current = Current.replace('min','Minutes')
            if Current[-2:] == ' h':
                Current = Current.replace('h','Hours')
            if Current[-2:] == ' d':
                Current = Current.replace('d','Days')
            if Current[-2:] == ' w':
                Current = Current.replace('w','Weeks')
            if Current[-2:] == ' M':
                Current = Current.replace('M','Months')
            if Current[-2:] == ' Y':
                Current = Current.replace('Y','Years')
            
            if Current[:2] == '1 ':
                Current = Current[:-1]

            ResampleRatesReadable.append(Current)
        return {'Readable':ResampleRatesReadable, 'Computable':ResampleRates}
    else:
        return {'Readable':[], 'Computable':[]}

# Generate Resampled Transform Data for Visualization
@reactive.Calc
def GetTransformVisResampleData():
    if len(DataTransforms.get()) > 0:
        ColNum = DataTransformColumns.get().index(input.TransformColToVis())
        Data = DataTransforms.get()[ColNum].copy()

        ResampleRate = GetVisResampleRatesTransform()['Computable'][GetVisResampleRatesTransform()['Readable'].index(input.TransformVisResRate())]

        Data = Data.resample(ResampleRate,on='Timestamp').mean()
        return Data

# Get error correction visualization resample rates   
@reactive.Calc
def GetECVisResampleRates():
    # Get readable and computer useable resample rates for plotting, very noobish 50 line code
    Data = OriginalFeatureData.get()

    if 'Timestamp' in Data.columns:
        SampleRateOriginal = Data['Timestamp'][1] - Data['Timestamp'][0]
        
        # Get first index where the resample rate is less frequent than the original
        Index = next(idx for idx, value in enumerate(SampleRateListCompare) if pd.to_timedelta(value) > SampleRateOriginal)
        
        SampleRateOriginal = to_offset(SampleRateOriginal).freqstr.replace('T','min')
        if SampleRateOriginal[0].isalpha():
            SampleRateOriginal = '1' + SampleRateOriginal

        ResampleRates = [SampleRateOriginal]
        # Output is a list of downsample rates starting with original sample rate
        ResampleRates.extend(SampleRateList[Index:])
        ResampleRatesReadable = []

        for value in ResampleRates:
            Current = str(re.sub('(\d+(\.\d+)?)', r' \1 ', value))
            Current = Current[1:]

            if Current[-3:] == ' ms':
                Current = Current.replace('ms','Milliseconds')
            if Current[-2:] == ' s':
                Current = Current.replace('s','Seconds')
            if Current[-4:] == ' min':
                Current = Current.replace('min','Minutes')
            if Current[-2:] == ' h':
                Current = Current.replace('h','Hours')
            if Current[-2:] == ' d':
                Current = Current.replace('d','Days')
            if Current[-2:] == ' w':
                Current = Current.replace('w','Weeks')
            if Current[-2:] == ' M':
                Current = Current.replace('M','Months')
            if Current[-2:] == ' Y':
                Current = Current.replace('Y','Years')
            
            if Current[:2] == '1 ':
                Current = Current[:-1]

            ResampleRatesReadable.append(Current)
        return {'Readable':ResampleRatesReadable, 'Computable':ResampleRates}
    else:
        return {'Readable':[], 'Computable':[]}
#%% UI Start
ui.page_opts(
    title="Time Series Data Cleaning User Interface",  
    page_fn=partial(page_navbar, id="page", selected='Data Input'),
)

theme.journal()

plt.style.use('ggplot')
#%% UI Data Input Panel
with ui.nav_panel("Data Input"):
    with ui.navset_card_underline():
        # Read file and set separator
        with ui.nav_panel("Read File"):
            with ui.layout_column_wrap(width = 1,heights_equal='row'):
                with ui.layout_column_wrap(width = 1/4):
                    with ui.card():
                        ui.input_file("CSVLocation","Select a File with Comma Separated Values to Read", width = '100%')

                    with ui.card():
                        ui.input_numeric("CSVHeader", "Set Header Row", 1, min=1,width = '100%')

                    with ui.card():
                        ui.input_text("SepText","CSV Separator",",",width='100%')
                        # Reset default as ','

                    with ui.card():
                        @render.ui
                        def SetCSVStart():
                            if len(CurrentCSV.get()) > 50:
                                return ui.input_slider("CSVStart","Render CSV From This Row",
                                                    0,len(CurrentCSV.get())-50,min,
                                                    width='100%')
        
                with ui.card():
                    "Current Dataset"
                    @render.data_frame
                    def RenderUneditedCSV():
                        ToRender = CurrentCSVRendered.get().copy()
                        
                        return render.DataGrid(ToRender.iloc[input.CSVStart():input.CSVStart()+50],width='100%')
        
        # Combine columns if necessary and set datatypes
        with ui.nav_panel("Set Column Datatypes"):
            with ui.layout_columns():
                # DType Dataframe
                with ui.card():
                    "Current Data Types"
                    # @render.ui()
                    # def SelectedRows():
                    #     rows = input.RenderCSVDtypes_selected_rows()  
                    #     selected = ", ".join(str(i) for i in sorted(rows)) if rows else "None"
                    #     return f"Rows selected: {selected}"

                    @render.data_frame
                    def RenderCSVDtypes():
                        return render.DataGrid(CurrentDTypes.get(),
                                            row_selection_mode="multiple",
                                            width='100%')
                    
                # DType Settings
                with ui.accordion(id="DTypeAccordion", open=["General Formatting", 'Timestamp Column from Existing Column']):
                    with ui.accordion_panel('General Formatting'):
                        with ui.layout_column_wrap(width = 1, heights_equal='row'):
                            with ui.tooltip(id="tt1_1", placement="top"):
                                ui.input_action_button("ResetCSV", "Reset",width='100%')
                                "Reset to original unedited CSV."

                            ui.input_action_button("CreateHeaders", "Generate Headers",width='100%')
                            
                            ui.input_action_button("DeleteCols", "Delete Selected Columns",width='100%')

                    with ui.accordion_panel('Timestamp Column from Existing Column'):
                        with ui.layout_column_wrap(width = 1, heights_equal='row'):
                            with ui.tooltip(id="tt1_4", placement="top"):
                                ui.input_action_button("CombineCols", "Combine Date Columns or Select Timestamp Column",width='100%')
                                "Combine selected columns into one or select timestamp column."
                            
                            @render.text()
                            def RenderFirstNonNullTimestamp():
                                Data = CurrentCSV.get().copy()
                                if 'Timestamp' in Data.columns:
                                    Timestamp = Data['Timestamp']
                                    ToWrite = str(Timestamp.loc[~Timestamp.isnull()].iloc[0])
                                    return f'First available timestamp value: {ToWrite}'
                                else:
                                    return 'Timestamp column not yet set'

                            with ui.tooltip(id="tt1_5", placement="top"):
                                ui.input_text("DatetimeRule","Datetime Format","%Y-%m-%d %H:%M:%S",width='100%')
                                # Delete default setting
                                ui.HTML("Format of the datetime data in the relevant column<br>")
                                ui.HTML("<br>Check the text above to see what the first non-null timestamp is as an aid<br>")
                                ui.HTML("<br>Example: %Y/%M/%D can represent 2020/12/30<br>")
                                ui.HTML("<br>Representations:<br>")
                                ui.HTML("4-Digit Year: %Y<br>")
                                ui.HTML("2-Digit Year: %y<br>")
                                ui.HTML("Month: %m<br>")
                                ui.HTML("Day: %d<br>")
                                ui.HTML("Hour: %H<br>")
                                ui.HTML("Minute: %M<br>")
                                ui.HTML("Second: %S<br>")
                                ui.HTML("Microsecond: %f (anything between deciseconds and microseconds is recognized)")

                            with ui.tooltip(id="tt1_6", placement="top"):
                                ui.input_action_button("ConvertToDatetime", "Convert to Relevant Data Types",width='100%')
                                "Convert timestamp column to datetime based on the given format, also converting all other columns to numeric."

                            @render.ui
                            @reactive.event(input.ConvertToDatetime)
                            async def CreateDtypeConversionProgressBar():
                                with ui.Progress(min=1, max = len(GetVisColumns()) + 1) as Progress:
                                    Progress.set(message="Calculating Datatype Conversion")

                                    # Convert timestamp to datetime and all other to numeric
                                    Data = CurrentCSV.get()
                                    Columns = list(Data.columns)

                                    StartTime = perf_counter()

                                    Data['Timestamp'] = pd.to_datetime(Data['Timestamp'], format=input.DatetimeRule(), errors = 'coerce')

                                    Progress.set(1)

                                    Columns.remove('Timestamp')

                                    for ColIndex, Column in enumerate(Columns):
                                        Data[Column] = pd.to_numeric(Data[Column], errors = 'coerce')

                                        CurrentTime = perf_counter()
                                        ElapsedTime = CurrentTime - StartTime
                                        RemainingTime = timedelta(seconds=int(ElapsedTime * ((len(Columns) + 1) - (ColIndex + 1)) / (ColIndex + 2)))
                                        
                                        Progress.set(ColIndex + 2, detail=f"Estimated Remaining Time: {RemainingTime}")

                                    Columns = list(Data.columns)
                                    DTypes = list(Data.dtypes.astype(str))
                                    DTypeData = pd.DataFrame({'Column': Columns,'Data Type': DTypes})

                                    RenderedData = Data.copy()
                                    RenderedData['Timestamp'] = RenderedData['Timestamp'].astype("string")

                                    CurrentCSV.set(Data)
                                    CurrentCSVRendered.set(RenderedData)
                                    CurrentDTypes.set(DTypeData)

                                    MVImputedCSV.set(Data.copy())

                                    Datetimes = DTypes.count('datetime64[ns]')
                                    Objects = DTypes.count('object')

                                    if Datetimes == 1 and Objects == 0:
                                        ConversionResult = 'All columns successfully converted.'
                                    else:
                                        ConversionResult = 'Problem in converting datatypes.'

                                    UniqueTimedeltas = len(Data['Timestamp'].diff().value_counts())
                                    if UniqueTimedeltas == 1:
                                        ResampleResult = 'Sampling rate is constant, dataframe resampling not necessary.'
                                    else:
                                        ResRate = str(Data['Timestamp'].diff().mode()[0])
                                        ResampleResult = f'Dataframe resampling recommended'

                                return ui.HTML(f"{ConversionResult}<br>{ResampleResult}")
                        
                    with ui.accordion_panel('Create New Timestamp Column'):
                        with ui.layout_column_wrap(width = 1, heights_equal='row'):
                            ui.input_date("TimestampEndDate", "Set Timestamp Column End Date", width = '100%')

                            "Timestamp Sampling Rate"

                            with ui.layout_column_wrap(width = 1/2):
                                ui.input_numeric("TimestampFreqValue", "", 1, width='100%')
                                
                                ui.input_select("TimestampFreqUnit", "", {'us': 'Microseconds', 'ms': 'Milliseconds', 's': 'Seconds',
                                                                          'min': 'Minutes', 'h': 'Hours', 'D': 'Days',
                                                                          'W': 'Weeks', 'MS': 'Months', 'YS': 'Years'},
                                                selected='s', width = '100%')
                                
                            ui.input_action_button('TimestampFromScratch', 'Create Timestamp Column and Convert All Others To Numeric', width='100%')

                            # Create timestamp column and convert all else to numeric
                            @render.ui
                            @reactive.event(input.TimestampFromScratch)
                            async def CreateTimestampColProgressBar():
                                NCols = len(CurrentCSV.get().copy().columns)

                                if 'Timestamp' in CurrentCSV.get().copy().columns:
                                    NCols -= 1

                                with ui.Progress(min=1, max = NCols) as Progress:
                                    Progress.set(message='Creating Timestamp Column')

                                    Data = CurrentCSV.get().copy()

                                    EndDate = input.TimestampEndDate()
                                    ResUnit = input.TimestampFreqUnit()
                                    ResValue = input.TimestampFreqValue()

                                    ResFrequency = f'{ResValue}{ResUnit}'

                                    Timestamp = pd.date_range(end=EndDate, freq=ResFrequency, periods=len(Data))

                                    if 'Timestamp' in Data.columns:
                                        Data = Data.drop(['Timestamp'], axis=1)

                                    Data.insert(0, 'Timestamp', Timestamp)

                                    Columns = list(Data.columns)
                                    Columns.remove('Timestamp')
                                    
                                    Progress.set(1, detail = 'Converting Remaining Columns to Numeric')

                                    StartTime = perf_counter()

                                    for ColIndex, Column in enumerate(Columns):
                                        Data[Column] = pd.to_numeric(Data[Column], errors = 'coerce')

                                        CurrentTime = perf_counter()
                                        ElapsedTime = CurrentTime - StartTime
                                        RemainingTime = timedelta(seconds=int(ElapsedTime * ((len(Columns) + 1) - (ColIndex + 1)) / (ColIndex + 2)))
                                        
                                        Progress.set(ColIndex + 2, detail=f"Estimated Remaining Time: {RemainingTime}")
                                    
                                    Columns = list(Data.columns)
                                    DTypes = list(Data.dtypes.astype(str))
                                    DTypeData = pd.DataFrame({'Column': Columns,'Data Type': DTypes})

                                    CurrentCSV.set(Data)
                                    MVImputedCSV.set(Data.copy())

                                    RenderData = Data.copy()
                                    RenderData['Timestamp'] = [str(x) for x in list(RenderData['Timestamp'])]

                                    CurrentCSVRendered.set(RenderData)
                                    CurrentDTypes.set(DTypeData)

                                    Datetimes = DTypes.count('datetime64[ns]')
                                    Objects = DTypes.count('object')

                                    if Datetimes == 1 and Objects == 0:
                                        ConversionResult = 'All columns successfully converted.'
                                    else:
                                        ConversionResult = 'Problem in converting datatypes.'

                                return ui.HTML(f"{ConversionResult}")

                    with ui.accordion_panel('Resample Dataframe'):
                        with ui.layout_column_wrap(width = 1, heights_equal='row'):
                            @render.ui
                            @reactive.event(input.ConvertToDatetime, input.TimestampFromScratch, input.DoResampleDataframe)
                            def RenderOriginalResRate():
                                Data = CurrentCSV.get().copy()

                                if 'Timestamp' in Data.columns:
                                    DType = str(Data['Timestamp'].dtype)

                                    if 'datetime' in DType:

                                        UniqueTimedeltas = len(Data['Timestamp'].diff().value_counts())
                                        if UniqueTimedeltas == 1:
                                            ResRate = str(Data['Timestamp'].diff().mode()[0])
                                            ResampleResult = f'Constant sampling rate of {ResRate}'
                                        else:
                                            ResRate = str(Data['Timestamp'].diff().mode()[0])
                                            ResampleResult = f'Non-constant sampling rate with most frequent sample frequency of {ResRate}'
                                        
                                        return ui.HTML(f'Current Sampling Rate:<br>{ResampleResult}')

                            "New Resampling Rate"

                            with ui.layout_column_wrap(width = 1/2):
                                ui.input_numeric("ResFreqValue", "", 1, width='100%')
                                
                                ui.input_select("ResFreqUnit", "", {'us': 'Microseconds', 'ms': 'Milliseconds', 's': 'Seconds',
                                                                            'min': 'Minutes', 'h': 'Hours', 'd': 'Days',
                                                                            'w': 'Weeks', 'M': 'Months', 'Y': 'Years'},
                                                selected='s', width = '100%')
                                
                            ui.input_action_button('DoResampleDataframe', 'Resample Dataframe', width='100%')
        
        # Detect Prominent Periodicities
        with ui.nav_panel("Prominent Periodicities"):
            with ui.layout_column_wrap(width=1, heights_equal='row'):
                with ui.layout_column_wrap(width = 1/2):
                    with ui.layout_column_wrap(width=1, heights_equal='row'):
                        ui.input_numeric('ACFSamples', 'Set Samples in (P)ACF Calculation', 20000, min = 10, width = '100%')

                        @render.text
                        def RenderPeriodicityRuntime():
                            if len(PeriodicityDataFrame.get()) > 0: 
                                return f'Samples in (P)ACF: {input.ACFSamples()}, Runtime: {PeriodicityRuntime.get()}'
                            
                    ui.input_slider('ACFStrengthCutoff','Set (P)ACF Autocorrelation Strength Cutoff', 0.1, 0.9, 0.3, step=0.025)

                with ui.card():
                    with ui.tooltip(id="tt2_1", placement="right"):
                        ui.input_action_button("GetPeriodicities", "Calculate Prominent Periodicities",width='100%')
                        "Calculate relevant periodicities for each column."

                    @render.ui
                    @reactive.event(input.GetPeriodicities)
                    async def CreatePeriodicityProgressBar():
                        with ui.Progress(min=1, max = len(GetVisColumns())) as Progress:
                            Progress.set(message="Calculating Prominent Periodicities")

                            PeriodicityDataFrame.set(pd.DataFrame())
                            PeriodicityDataFrameReadable.set(pd.DataFrame())

                            Data = CurrentCSV.get()
                            Columns = list(Data.columns)
                            Columns.remove('Timestamp')

                            StartTime = perf_counter()

                            # Convert from pd default ns to ms
                            SampleRateOriginal = Data['Timestamp'][1] - Data['Timestamp'][0]

                            RPList, SPList = [], []
                            RPListReadable, SPListReadable = [], []

                            for ColIndex, Column in enumerate(Columns):
                                print(f'Relevant Periodicities, Column {ColIndex + 1} of {len(Columns)}')
                                CurrentTime = perf_counter()
                                ElapsedTime = CurrentTime - StartTime
                                RemainingTime = timedelta(seconds=int(ElapsedTime * (len(Columns) - (ColIndex)) / (ColIndex + 1)))
                                
                                if ColIndex == 0:
                                        Progress.set(ColIndex + 1)
                                else:
                                    Progress.set(ColIndex + 1, detail=f"Estimated Remaining Time: {RemainingTime}")

                                RelevantPeriods, StrongPeriods = RelevantPeriodicities(Data, 'Timestamp', Column, input.ACFSamples(), input.ACFStrengthCutoff())
                                
                                # Get numeric dataframe for computations
                                RPList.append(RelevantPeriods)
                                SPList.append(StrongPeriods)

                                # Get readable dataframe for the user in shiny
                                if len(RelevantPeriods) > 0:
                                    for index, x in enumerate(RelevantPeriods):
                                        if index == 0:
                                            RPReadable = str(x*SampleRateOriginal)
                                        else:
                                            RPReadable += "; " + str(x*SampleRateOriginal)
                                else:
                                    RPReadable = ""

                                if len(StrongPeriods) > 0:
                                    for index, x in enumerate(StrongPeriods):
                                        if index == 0:
                                            SPReadable = str(x*SampleRateOriginal)
                                        else:
                                            SPReadable += "\n" + str(x*SampleRateOriginal)
                                else:
                                    SPReadable = ""

                                RPListReadable.append(RPReadable)
                                SPListReadable.append(SPReadable)

                            PeriodicityData = pd.DataFrame({'Statistically Relevant Strong Periodicities': RPList, 'Statistically Irrelevant Strong Periodicities': SPList}, index = Columns)
                            PeriodicityDataReadable = pd.DataFrame({'Column': Columns, 'Statistically Relevant Strong Periodicities': RPListReadable, 'Statistically Irrelevant Strong Periodicities': SPListReadable})

                            PeriodicityDataFrame.set(PeriodicityData)
                            PeriodicityDataFrameReadable.set(PeriodicityDataReadable)

                            Runtime = timedelta(seconds=int(perf_counter() - StartTime))
                            PeriodicityRuntime.set(str(Runtime))

                        return "Done computing!"
                    
                with ui.card():
                    "Prominent Periodicities"
                    @render.data_frame
                    def RenderPeriodicities():
                        return render.DataGrid(PeriodicityDataFrameReadable.get(),width='100%')
                    
        # Initial visualization
        with ui.nav_panel("Initial Visualization"):
            with ui.accordion(id="InitialVisualizationAcc", open="Single Plot", multiple=False):
                with ui.accordion_panel("Single Plot"):
                    with ui.layout_column_wrap(width=1, heights_equal='row'):
                        with ui.layout_column_wrap(width=1/3):
                            @render.ui()
                            def InitVisResampleDropdownSingle():
                                return ui.input_select("InitVisResRateSingle","Resampling Rate",GetVisResampleRates()['Readable'],width='100%')
                        
                            ui.input_numeric("InitVisMaxSamplesSingle","Maximum Samples on Plot",300,min=10,width='100%')

                            @render.ui()
                            def InitVisColDropdownSingle():
                                return ui.input_select("InitVisColSingle","Column to Visualize on Plot",GetVisColumns(),width='100%')
                        
                        @render.ui()
                        def InitVisStartTimeSliderSingle():
                            if len(GetVisResampledDataSingle()) > input.InitVisMaxSamplesSingle():
                                return ui.input_slider("InitVisStartSingle","Starting Index of Plot",0,len(GetVisResampledDataSingle())-input.InitVisMaxSamplesSingle(),IVSliderCVSingle.get(),
                                                       step=input.InitVisMaxSamplesSingle()//6,animate=True, width='100%')
                        
                        @render.plot(alt="No Column Selected")
                        def InitPlotSingle():
                            if len(GetVisResampledDataSingle()) > input.InitVisMaxSamplesSingle():
                                Data = GetVisResampledDataSingle()

                                Data = Data[Data.index[input.InitVisStartSingle()]: Data.index[input.InitVisStartSingle() + input.InitVisMaxSamplesSingle() - 1]][input.InitVisColSingle()]
                            else:
                                Data = GetVisResampledDataSingle()[input.InitVisColSingle()]

                            fig, ax = plt.subplots()
                            ax.plot(Data)
                            ax.set_title(input.InitVisColSingle())
                            ax.set_xlabel("Timestamp")
                            ax.set_ylabel("Value")

                            return fig
                
                with ui.accordion_panel("Paired Plots"):
                    with ui.layout_column_wrap(width=1, heights_equal='row'):
                        with ui.layout_column_wrap(width=1/4):
                            @render.ui()
                            def InitVisResampleDropdownPair():
                                return ui.input_select("InitVisResRatePair","Resampling Rate",GetVisResampleRates()['Readable'],width='100%')
                        
                            ui.input_numeric("InitVisMaxSamplesPair","Maximum Samples on a Given Plot",300,min=10,width='100%')

                            @render.ui()
                            def InitVisCol1Dropdown():
                                return ui.input_select("InitVisCol1","Column to Visualize on Plot 1",GetVisColumns(),width='100%')
                            
                            @render.ui()
                            def InitVisCol2Dropdown():
                                return ui.input_select("InitVisCol2","Column to Visualize on Plot 2",GetVisColumns(),width='100%')
                        
                        @render.ui()
                        def InitVisStartTimeSliderPair():
                            if len(GetVisResampledDataPair()) > input.InitVisMaxSamplesPair():
                                return ui.input_slider("InitVisStartPair","Starting Index of Plot",0,len(GetVisResampledDataPair())-input.InitVisMaxSamplesPair(),IVSliderCVPair.get(),
                                                       step=input.InitVisMaxSamplesPair()//6,animate=True, width='100%')
                        
                        with ui.layout_column_wrap(width=1/2):
                            @render.plot(alt="No Column Selected")
                            def InitPlotA():
                                if len(GetVisResampledDataPair()) > input.InitVisMaxSamplesPair():
                                    Data = GetVisResampledDataPair()

                                    Data = Data[Data.index[input.InitVisStartPair()]: Data.index[input.InitVisStartPair() + input.InitVisMaxSamplesPair() - 1]][input.InitVisCol1()]
                                else:
                                    Data = GetVisResampledDataPair()[input.InitVisCol1()]

                                fig, ax = plt.subplots()
                                ax.plot(Data)
                                ax.set_title(input.InitVisCol1())
                                ax.set_xlabel("Timestamp")
                                ax.set_ylabel("Value")

                                return fig
                            
                            @render.plot(alt="No Column Selected")
                            def InitPlotB():
                                if len(GetVisResampledDataPair()) > input.InitVisMaxSamplesPair():
                                    Data = GetVisResampledDataPair()

                                    Data = Data[Data.index[input.InitVisStartPair()]: Data.index[input.InitVisStartPair() + input.InitVisMaxSamplesPair() - 1]][input.InitVisCol2()]
                                else:
                                    Data = GetVisResampledDataPair()[input.InitVisCol2()]

                                fig, ax = plt.subplots()
                                ax.plot(Data)
                                ax.set_title(input.InitVisCol2())
                                ax.set_xlabel("Timestamp")
                                ax.set_ylabel("Value")

                                return fig
#%% UI Missing Value Handling Panel
with ui.nav_panel("Missing Value Handling"):
    with ui.navset_card_underline():
        with ui.nav_panel("Descriptive Dataframes"):
            with ui.layout_column_wrap(width = 1, heights_equal='row'):
                @render.text
                def NoMVsText():
                    MVs = CurrentCSV.get().copy().isnull().sum().sum()
                    if MVs == 0:
                        return 'No Missing Values - You May Move on To Data Transformation'

                with ui.layout_columns(col_widths=(4,8)):
                    with ui.card():
                        "Missing Value Occurrences"

                        @render.ui
                        def RenderMVDFStartSlider():
                            MVs = CurrentCSV.get().copy().isnull().sum().sum()
                            if MVs > 0:
                                MVDF = GetMVEvalDF()['MVDFToVis']
                                Occurrences = len(MVDF)
                                if Occurrences > 100:
                                    return ui.input_slider('MVOccDFStart', 'Set Starting Occurrence On Dataframe', 1, Occurrences - 100, 1, width = '100%')

                        @render.data_frame
                        def RenderMVDF():
                            MVs = CurrentCSV.get().copy().isnull().sum().sum()
                            if MVs > 0:
                                ToRender = GetMVEvalDF()['MVDFToVis']
                                ToRender = pd.DataFrame({'Occurrence':np.arange(1, len(ToRender)+1), 'Start Point': ToRender['Start Point'], 'Length': ToRender['Length']})

                                if len(ToRender) > 100:
                                    ToRender = ToRender.iloc[input.MVOccDFStart() - 1: input.MVOccDFStart() + 100]
                                    
                                return render.DataGrid(ToRender,width='100%')
                        
                    with ui.card():
                        "Missing Value Characterization Across Features"
                        @render.data_frame
                        def RenderMVEvaluation():
                            MVs = CurrentCSV.get().copy().isnull().sum().sum()
                            if MVs > 0:
                                return render.DataGrid(GetMVEvalDF()['MVEvalToVis'],width='100%')
                            
        with ui.nav_panel("Visualization"):
            with ui.accordion(id="MVVisAcc", open="Cumulative Sum", multiple=False):
                with ui.accordion_panel("Cumulative Sum"):
                    with ui.layout_column_wrap(width=1/2):
                        ui.input_radio_buttons("MVVisAllOrSelectCumSum","Columns To Visualize",['All','Selected'],width='100%')

                        @render.ui()
                        def MVVisSelectedColumnsCumSum():
                            return ui.input_select("MVVisColumnsCumSum","Selected Columns",GetVisColumns(),width='100%', multiple=True)
                        
                    @render.plot(alt="No Column Selected")
                    def MVCumSumPlot():
                        if input.MVVisAllOrSelectCumSum() == 'All':
                            Columns = list(CurrentCSV.get().columns)
                            Columns.remove('Timestamp')
                        elif input.MVVisAllOrSelectCumSum() == 'Selected':
                            Columns = input.MVVisColumnsCumSum()

                        NullMatrix = CurrentCSV.get().isnull()
                        UniqueMVs = NullMatrix.any(axis=1).sum()

                        fig, ax = plt.subplots()
                        for Column in Columns:
                            CumSum = NullMatrix[Column].cumsum()
                            ax.plot(CurrentCSV.get()['Timestamp'], CumSum, label = Column)
                        ax.set_xlabel('Timestamp')
                        ax.set_ylim(0,UniqueMVs * 1.2)
                        ax.set_title('Cumulative Sum of Missing Values')
                        ax.legend(loc='upper left')

                        return fig

                with ui.accordion_panel("Spanwise Proportions"):
                    with ui.layout_column_wrap(width=1/3):
                        ui.input_radio_buttons("MVVisAllOrSelectSpan","Columns To Visualize",['All','Selected'],width='100%')

                        @render.ui()
                        def MVVisSelectedColumnsSpan():
                            return ui.input_select("MVVisColumnsSpan","",GetVisColumns(),width='100%', multiple=True)
                        
                        with ui.input_numeric("MVVisSpans","Number of Spans",10,width='100%'):
                            @render.text
                            def SpanText():
                                SpanLength = len(CurrentCSV.get())//input.MVVisSpans()
                                SpanTime = str(SpanLength * (CurrentCSV.get()['Timestamp'][1] - CurrentCSV.get()['Timestamp'][0]))
                                return f"Span length: {SpanTime}"
                        
                    @render.plot(alt="No Column Selected")
                    def MVSpanPlot():
                        if input.MVVisAllOrSelectSpan() == 'All':
                            Columns = list(CurrentCSV.get().columns)
                            Columns.remove('Timestamp')
                        elif input.MVVisAllOrSelectSpan() == 'Selected':
                            Columns = input.MVVisColumnsSpan()

                        NullMatrix = CurrentCSV.get().isnull()
                        UniqueMVs = NullMatrix.any(axis=1).sum()
                        Spans = input.MVVisSpans()

                        SplitMatrices = np.array_split(NullMatrix, Spans)

                        fig, ax = plt.subplots()
                        for Column in Columns:
                            InSpan = list(map(lambda Span: SplitMatrices[Span][Column].sum() / UniqueMVs, np.arange(Spans)))
                            ax.bar(np.arange(1,Spans+1), InSpan, label = Column, alpha = 0.6)
                        ax.set_xlabel('Span')
                        ax.set_xticks(np.arange(1,Spans+1))
                        ax.set_ylim(0,1)
                        ax.set_title('Proportion of All Missing Values Within Span')
                        ax.legend(loc='upper left')

                        return fig
                    
                with ui.accordion_panel("Nullity Correlation Heatmap"):
                    with ui.layout_column_wrap(width=1/2):
                        ui.input_radio_buttons("MVVisAllOrSelectNCH","Columns To Visualize",['All','Selected'],width='100%')

                        @render.ui()
                        def MVVisSelectedColumnsNCH():
                            return ui.input_select("MVVisColumnsNCH","Selected Columns",GetVisColumns(),width='100%', multiple=True)
                        
                    @render.plot(alt="No Column Selected")
                    def MVNCHPlot():
                        if input.MVVisAllOrSelectNCH() == 'All':
                            Columns = list(CurrentCSV.get().columns)
                            Columns.remove('Timestamp')
                        elif input.MVVisAllOrSelectNCH() == 'Selected':
                            Columns = list(input.MVVisColumnsNCH())

                        Data = CurrentCSV.get()[Columns]

                        fig, ax = plt.subplots()
                        ax = sns.heatmap(Data.loc[:, Data.isnull().any()].isnull().corr(), annot=True, fmt='.2f', vmin = 0,
                                       vmax = 1,linewidths = 0.5, linecolor = 'black')
                        plt.xticks(rotation=45)
                        plt.yticks(rotation=0)
                        ax.set_title('Nullity Correlation Heatmap')

                        return fig
                            
        with ui.nav_panel("Imputation"):
            with ui.layout_column_wrap(width=1/2):
                with ui.card():
                    ui.h6("Calculation")
                    with ui.tooltip(id="ttMV_1", placement="top"):
                        @render.ui()
                        def LCMVCutoffInput():
                            return ui.input_numeric('LCMVCutoff','Set Cutoff Permillage for Long Continuous Missing Values', 10, min=1, max=1000, width='100%')
                        ui.HTML("The maximum period length to take into account in the long continuous imputation is one onethousands of this value.")
                
                    with ui.layout_column_wrap(width=1,heights_equal='row'):
                        with ui.tooltip(id="ttMV_2", placement="top"):
                            ui.input_action_button('DoMVImputation','Calculate Missing Value Imputation', width='100%')
                            ui.HTML("Requires the calculation of prominent periodicities (under data input panel).")

                        @render.ui
                        @reactive.event(input.DoMVImputation)
                        async def CreateMVImputeProgressBar():
                            with ui.Progress(min=1, max = len(GetVisColumns())) as Progress:
                                Progress.set(message="Calculating Missing Value Imputation")

                                # MV Imputation for All Columns and All MV Types
                                Data = CurrentCSV.get().copy()
                                PeriodsSeries = PeriodicityDataFrame.get()['Statistically Relevant Strong Periodicities']

                                Columns = list(Data.columns)
                                Columns.remove('Timestamp')

                                CutoffMultiplier = input.LCMVCutoff() / 1000

                                StartTime = perf_counter()

                                for ColIndex, Column in enumerate(Columns):
                                    print(f'Missing Value Imputation, Column {ColIndex + 1} of {len(Columns)}')
                                    CurrentTime = perf_counter()
                                    ElapsedTime = CurrentTime - StartTime
                                    RemainingTime = timedelta(seconds=int(ElapsedTime * (len(Columns) - (ColIndex)) / (ColIndex + 1)))

                                    if ColIndex == 0:
                                        Progress.set(ColIndex + 1)
                                    else:
                                        Progress.set(ColIndex + 1, detail=f"Estimated Remaining Time: {RemainingTime}")

                                    Periods = PeriodsSeries[Column]

                                    if len(Periods) > 0:
                                        # I use 1 as a cutoff minimum, meaning that a missing point is always linear interpolated rather than seasonal decomposed
                                        SCCutoff = max(1, int(CutoffMultiplier * min(Periods)))
                                        Periodic = True
                                    else:
                                        Periodic = False

                                    MVDFCol = GetMVEvalDF()['MVCols'][Column]
                                    if len(MVDFCol) > 0:
                                        
                                        if Periodic == True:
                                            SCMVs = MVDFCol.loc[np.where(MVDFCol['Length'] <= SCCutoff)]
                                            LCMVs = MVDFCol.loc[np.where(MVDFCol['Length'] > SCCutoff)]
                                        elif Periodic == False:
                                            SCMVs = MVDFCol.copy()
                                            LCMVs = []
                                        
                                        if len(SCMVs) > 0:
                                            SCMVs = SCMVs.reset_index(drop=True)

                                            Data = SCImputation(Data, Column, SCMVs, 1000)
                                            
                                        if len(LCMVs) > 0:
                                            LCMVs = LCMVs.reset_index(drop=True)

                                            LCResult = LCImputation(Data, Column, LCMVs, CutoffMultiplier, Periods)
                                            Data = LCResult['Data']
                                            LeftoverSCMVs = LCResult['Leftover SCMVs']
                                            
                                            if len(LeftoverSCMVs) > 0:
                                                Data = SCImputation(Data, Column, LeftoverSCMVs, 1000)

                                MVImputedCSV.set(Data)

                            return f"Done computing! Remaining Missing Values: {Data.isnull().sum().sum()}"
                
                with ui.card():
                    ui.h6("Plotting")
                    @render.ui()
                    def MVImputeColInput():
                        return ui.input_select("MVImputeCol","Selected Column",GetVisColumns(),width='100%')
                    
                    with ui.layout_column_wrap(width=1/2):
                        @render.ui()
                        def MVOccurrenceSelect():
                            MVsForCol = GetMVEvalDF()['MVCols'][input.MVImputeCol()]
                            Occurrences = len(MVsForCol)

                            if Occurrences > 0:
                                return ui.input_numeric("MVImputePlotOccurrence", "Missing Occurrence To Plot", 1, min=1, max=Occurrences, width='100%')
                            else:
                                return "No Missing Values in Selected Column"

                        with ui.tooltip(id="ttMV_4", placement="top"):
                            ui.input_numeric("MVPlotSideLength", "Plot Size", 1, min=1, max=100, width='100%')
                            ui.HTML("How many times the missing value length should be plotted before and after the missing value (if the missing value length is greater than 50).")  
                            ui.HTML("<br><br>(The continuous missing value imputation uses at maximum +-5 periods before and after the missing values)")

            @render.plot(alt="Nothing To Plot")
            def MVOccurrencePlot():
                Column = input.MVImputeCol()
                Occurrence = input.MVImputePlotOccurrence()

                MVColData = GetMVEvalDF()['MVCols'][Column]

                MVStart = MVColData['Start Point'][Occurrence - 1]
                MVLength = MVColData['Length'][Occurrence - 1]
                MVEnd = MVStart + MVLength - 1

                if MVLength < 50:
                    LengthOnSide = 50 * input.MVPlotSideLength()
                elif MVLength >= 50:
                    LengthOnSide = MVLength * input.MVPlotSideLength()

                Unimputed = CurrentCSV.get()
                Imputed = MVImputedCSV.get()

                PlotStart = max(0, MVStart - LengthOnSide)
                PlotEnd = min(len(Imputed) - 1, MVEnd + LengthOnSide)

                fig, ax = plt.subplots()
                ax.plot(Unimputed['Timestamp'][PlotStart: PlotEnd + 1], Unimputed[Column][PlotStart: PlotEnd + 1], label = 'Original Data', color = 'C0')
                ax.plot(Imputed['Timestamp'][MVStart - 1: MVEnd + 2], Imputed[Column][MVStart - 1: MVEnd + 2], label = 'Imputed Data', color = 'C1')
                if MVLength < 30:
                    ax.scatter(Unimputed['Timestamp'][PlotStart: PlotEnd + 1], Unimputed[Column][PlotStart: PlotEnd + 1], marker = '.', color = 'C0')
                    ax.scatter(Imputed['Timestamp'][MVStart: MVEnd + 1], Imputed[Column][MVStart: MVEnd + 1], marker = '.', color = 'C1')
                ax.set_xlabel('Timestamp')
                ax.set_ylabel('Value')
                ax.set_title(f'Imputation of {Column} at Missing Value Occurrence {Occurrence}')
                ax.legend(loc='upper right')

                return fig
#%% UI Data Transformation Panel
with ui.nav_panel("Data Transformation"):
    with ui.card():
        ui.input_text('FolderName', 'Name of the Folder to Save Files on Desktop', str(date.today()), width='100%')

    with ui.card():
        with ui.layout_column_wrap(width=1/2):
            with ui.layout_column_wrap(width=1):
                with ui.tooltip(id="ttMV_3", placement="top"):
                    ui.input_action_button('SaveMVImputation','Save Imputation Results (Optional)', width='100%')
                    ui.HTML("Optional, as this data will be available if the user moves forward with anomaly detection.")
                    ui.HTML("<br><br>Saves to desktop.")

                # Save MV Dataframe
                @render.ui
                @reactive.event(input.SaveMVImputation)
                async def SaveMVDataframe():
                    with ui.Progress(min=1, max = len(GetVisColumns())) as Progress:
                        Progress.set(message="Saving Missing Value Imputation")

                        Data = MVImputedCSV.get().copy()

                        if len(Data) == 0:
                            Data = CurrentCSV.get().copy()

                        MVLocations = MissingValueEvaluator(CurrentCSV.get())['MVCols']
                        Columns = list(Data.columns)
                        Columns.remove('Timestamp')

                        Desktop = winshell.desktop()

                        Folder = Desktop + '\\Data Cleaning - ' + input.FolderName()

                        if not os.path.exists(Folder):
                            os.makedirs(Folder)

                        NewPath = Folder + '\\Missing Value Imputed.csv'

                        for Column in Columns:
                            ColName = Column + ' Is Imputed'

                            Indices = []
                            StartLengths = MVLocations[Column]
                            for i in range(len(StartLengths)):
                                Start = StartLengths['Start Point'][i]
                                Length = StartLengths['Length'][i]
                                
                                CurrentSequence = np.arange(Start, Start + Length)
                                Indices.extend(CurrentSequence)
                            
                            IsMV = np.repeat(False, len(Data))
                            IsMV[Indices] = True
                            Data[ColName] = IsMV

                        Data.to_csv(NewPath, index=False)
                    
                    return f"File saved!"

            with ui.layout_column_wrap(width=1):
                with ui.tooltip(id="ttDT_1", placement="top"):
                    ui.input_action_button("DoDataTransformation", "Transform Data", width = '100%')
                    ui.HTML("Decompose data into trend and residual, and seasonal where there is a prominent periodicity.")
                    ui.HTML("<br><br>Requires:")
                    ui.HTML("<br>Prominent periodicities (data input panel)")
                    ui.HTML("<br>Missing value imputation (Unless there are no missing values in the data to begin with)")
                    ui.HTML("<br><br>Saves to desktop.")

                @render.ui
                @reactive.event(input.DoDataTransformation)
                async def TransformData():
                    with ui.Progress(min=0, max = GetDataTransformData()['Runtime Sum']) as Progress:
                        TotalTimeEstimate = str(timedelta(seconds=int(GetDataTransformData()['Runtime Sum'])))
                        Progress.set(message="Transforming Data", detail=f'Runtime Estimate: {TotalTimeEstimate}')

                        if CurrentCSV.get().copy().isnull().sum().sum() == 0:
                            Data = CurrentCSV.get().copy()
                        else:
                            Data = MVImputedCSV.get().copy()

                        ResampledData = Data.resample(on='Timestamp', rule=GetDataTransformData()['Resample Rate']).mean()

                        Timestamp = np.array(ResampledData.index)

                        STLSlope = 1.39e-6
                        MSTLSlope = 2.46e-6
                        
                        STLMultiplier, OddMSTLMultiplier, EvenMSTLMultiplier = 1, 1, 1

                        RuntimeEstimates = GetDataTransformData()['Runtime Components']
                        UsedPeriods = GetDataTransformData()['Used Periods']
                        ProgressValue = 0

                        Desktop = winshell.desktop()
                        Folder = Desktop + '\\Data Cleaning - ' + input.FolderName() + '\\Data Transformations'
                        if not os.path.exists(Folder):
                            os.makedirs(Folder)

                        StartTimeAll = perf_counter()

                        for ColIndex, Column in enumerate(list(ResampledData.columns)):
                            StartTimeCol = perf_counter()

                            NewPath = Folder + f'\\{Column} - Data Transform.csv'

                            RuntimeEstimate = RuntimeEstimates[ColIndex]
                            ProgressValue += RuntimeEstimate

                            Periods = UsedPeriods[ColIndex]

                            Series = np.array(ResampledData[Column])

                            # EWMA Smoothing
                            if len(Periods) == 0:
                                Series = pd.Series(Series)

                                Trend = Series.ewm(alpha=0.2).mean()
                                Residual = Series - Trend

                                TransformResult = pd.DataFrame({'Timestamp': Timestamp, 'Trend': Trend, 'Residual': Residual})

                            # STL Decomposition
                            if len(Periods) == 1:
                                Model = STL(Series, period=Periods[0]).fit()
                                CalculationTime = perf_counter() - StartTimeCol

                                TransformResult = pd.DataFrame({'Timestamp': Timestamp, 'Trend': Model.trend, f'Seasonal_{Periods[0]}': Model.seasonal,
                                                                'Residual': Model.resid})

                                if ColIndex == 0:
                                    STLMultiplier = CalculationTime / RuntimeEstimate
                                    OddMSTLMultiplier = CalculationTime / RuntimeEstimate
                                    EvenMSTLMultiplier = CalculationTime / RuntimeEstimate
                                else:
                                    STLMultiplier = CalculationTime / RuntimeEstimate

                            # MSTL Decomposition
                            if len(Periods) > 1:
                                Model = MSTL(Series, periods=Periods).fit()
                                CalculationTime = perf_counter() - StartTimeCol

                                TransformDict = {'Timestamp': Timestamp, 'Trend': Model.trend}
                                for PeriodIndex, Period in enumerate(Periods):
                                    TransformDict[f'Seasonal_{Period}'] = Model.seasonal[:, PeriodIndex]
                                TransformDict['Residual'] = Model.resid

                                TransformResult = pd.DataFrame(TransformDict)

                                if ColIndex == 0:
                                    STLMultiplier = CalculationTime / RuntimeEstimate
                                    OddMSTLMultiplier = CalculationTime / RuntimeEstimate
                                    EvenMSTLMultiplier = CalculationTime / RuntimeEstimate
                                elif len(Periods) % 2 == 0:
                                    EvenMSTLMultiplier = CalculationTime / RuntimeEstimate
                                elif len(Periods) % 2 == 1:
                                    OddMSTLMultiplier = CalculationTime / RuntimeEstimate

                            TransformResult.to_csv(NewPath, index=False)

                            # Remaining runtime estimate
                            RemainingEstimate = 0
                            if ColIndex < len(UsedPeriods) - 1:
                                for Periods in UsedPeriods[ColIndex + 1:]:
                                    if len(Periods) == 1:
                                        RemainingEstimate += STLMultiplier * STLSlope * len(ResampledData) * Periods[0]
                                    elif len(Periods) > 1:
                                        if len(Periods) % 2 == 0:
                                            RemainingEstimate += EvenMSTLMultiplier * MSTLSlope * len(ResampledData) * sum(Periods)
                                        elif len(Periods) % 2 == 1:
                                            RemainingEstimate += OddMSTLMultiplier * MSTLSlope * len(ResampledData) * sum(Periods)

                            RemainingEstimate = str(timedelta(seconds=int(RemainingEstimate)))

                            Progress.set(value = ProgressValue, detail=f'Runtime Estimate: {RemainingEstimate}')

                    return f"Data Transformed!"

    with ui.card():
        "The Effect of Downsampling on Runtime Estimate - Select a Row Here Before Transforming Data"
        @render.data_frame
        def RenderDataTransformResampleDataframe():
            ToRender = CalculateResamplingDataframe()
            if len(ToRender) > 0:
                ToRender = ToRender[['Downsampling','Runtime Estimate','Periods Lost','New Sampling Rate','Samples in Resampled Dataset']]

            return render.DataGrid(ToRender, row_selection_mode = 'single', width = '100%')
#%% UI Anomaly Detection Panel
with ui.nav_panel("Anomaly Detection"):
    with ui.navset_card_underline():
        with ui.nav_panel("Transformation Results"):
            with ui.layout_column_wrap(width=1, heights_equal='row'):
                with ui.layout_column_wrap(width = 1/3):
                    ui.input_file("DataTransformFiles","Select All Transformed Data", multiple=True, width = '100%')
                
                    @render.ui
                    def RenderTransformVisColSelect():
                        if len(DataTransforms.get()) > 0:
                            return ui.input_select('TransformColToVis','Select a Column to Visualize',DataTransformColumns.get(), width = '100%')
                    
                    @render.ui
                    def RenderTransformVisComponentSelect():
                        if len(DataTransforms.get()) > 0:
                            ColNum = DataTransformColumns.get().index(input.TransformColToVis())
                            Data = DataTransforms.get()[ColNum].copy()

                            Columns = list(Data.columns)
                            Columns.remove('Timestamp')
                        
                            return ui.input_select('TransformCompToVis','Select Components to Visualize',Columns, selected=Columns[0], multiple=True, size=3,
                                                    width = '100%')

                    @render.ui()
                    def RenderTransformVisResampleSelect():
                        if len(DataTransforms.get()) > 0:
                            return ui.input_select("TransformVisResRate","Resampling Rate",GetVisResampleRatesTransform()['Readable'],width='100%')
                    
                    @render.ui()
                    def RenderTransformVisMaxSize():
                        if len(DataTransforms.get()) > 0:
                            return ui.input_numeric('TransformVisMaxSize','Maximum Samples on Plot',300000,min=1, width='100%')
                        
                    @render.ui
                    def RenderTransformStartSlider():
                        if len(DataTransforms.get()) > 0:
                            if len(GetTransformVisResampleData()) > input.TransformVisMaxSize():
                                Maximum = len(GetTransformVisResampleData()) - input.TransformVisMaxSize()
                                return ui.input_slider('TransformVisStartPoint', 'Starting Index of Plot', min = 0, max = Maximum, value=0, width='100%',
                                                        step=input.TransformVisMaxSize()//6, animate=True)

                @render.plot
                def RenderTransformResultPlot():
                    if len(DataTransforms.get()) > 0:
                        Data = GetTransformVisResampleData()

                        if len(GetTransformVisResampleData()) > input.TransformVisMaxSize():
                            Data = Data.iloc[input.TransformVisStartPoint(): input.TransformVisStartPoint() + input.TransformVisMaxSize() - 1]

                        if len(input.TransformCompToVis()) > 0:
                            for Index, Component in enumerate(input.TransformCompToVis()):
                                if Index == 0:
                                    PlotY = Data[Component]
                                else:
                                    PlotY += Data[Component]

                            fig, ax = plt.subplots()

                            ax.plot(PlotY)
                            ax.set_xlabel('Timestamp')
                            ax.set_ylabel('Value')

                            fig.suptitle(f'Plot of Selected Components of {input.TransformColToVis()}')
                            
                            return fig
                        
        with ui.nav_panel("Detect Anomalies"):
            with ui.accordion(id="ADVisAccordion", multiple=False):
                with ui.accordion_panel("Calculation"):
                    with ui.layout_column_wrap(width = 1/3):
                        @render.ui
                        def RenderADCalcWindowSelection():
                            if len(DataTransformColumns.get()) > 0:
                                return ui.input_numeric('ADCalcWindow','Sliding Window Size',500,min=10,width='100%')
                            
                        @render.ui
                        def RenderADCalcMinsizeSelection():
                            if len(DataTransformColumns.get()) > 0:
                                return ui.input_numeric('ADCalcMinsize','Changepoint Minimum Length',5,min=5,width='100%')
                            
                        @render.ui
                        def RenderADCalcJumpSelection():
                            if len(DataTransformColumns.get()) > 0:
                                return ui.input_numeric('ADCalcJump','Sliding Window Jump',5,min=1,width='100%')
                            
                    ui.input_checkbox_group('ADCalcStrictnesses', 'Strictnesses Used in Calculations',
                                            ['BIC', 'Log of Samples', 'PELT Penalty', 'Slope Penalty', 'Variance'],
                                            inline=True, width='100%', selected='BIC')

                    with ui.layout_column_wrap(width = 1/2):
                        # Calculate Anomaly Detection
                        with ui.layout_column_wrap(width = 1):
                            @render.ui
                            def RenderDoAnomalyDetectionButton():
                                if len(DataTransformColumns.get()) > 0:
                                    return ui.input_action_button('DoAnomalyDetection','Compute Anomaly Detection',width='100%')

                            # Generate Anomaly Detection with Multiple Possible Strictnesses
                            @render.ui
                            @reactive.event(input.DoAnomalyDetection)
                            async def GenerateAnomalyDetectionResults():
                                with ui.Progress(min=1, max = len(DataTransformColumns.get())) as Progress:
                                    Progress.set(message="Detecting Anomalies")

                                    StartTime = perf_counter()

                                    AllTransforms = DataTransforms.get()
                                    AllColumns = DataTransformColumns.get()
                                    AllAnomalies = []

                                    Desktop = winshell.desktop()
                                    Folder = Desktop + '\\Data Cleaning - ' + input.FolderName() + '\\Anomaly Detection Results'

                                    if not os.path.exists(Folder):
                                        os.makedirs(Folder)

                                    for ColIndex, Column in enumerate(AllColumns):
                                        CurrentTime = perf_counter()
                                        if ColIndex > 0:
                                            RuntimeEstimate = timedelta(seconds=int((CurrentTime - StartTime)*(len(AllColumns)-(ColIndex))/(ColIndex)))
                                            Progress.set(value = ColIndex + 1, detail=f'Estimated Remaining Runtime: {str(RuntimeEstimate)}')
                                        else:
                                            Progress.set(value = ColIndex + 1)

                                        NewPath = Folder + f'\\{Column} - Anomaly Detection.csv'
                                        
                                        ColumnTransforms = AllTransforms[ColIndex].copy()
                                        Timestamp = ColumnTransforms['Timestamp']

                                        ColumnAnomalies = {'Timestamp': Timestamp}

                                        # Deseasonalize Trend and Residual
                                        ColumnTransforms[['Trend','Residual']] = EmergencyDeseasonalize(ColumnTransforms, ['Trend','Residual'], 20000)

                                        ColumnTransforms = ColumnTransforms.set_index('Timestamp')

                                        # Trend Anomalies
                                        Trend = ColumnTransforms['Trend']

                                        TrendChangepointList, StrictnessNames = WindowSlidingSegmentation(Trend, input.ADCalcWindow(), input.ADCalcMinsize(),
                                                                                                          input.ADCalcJump(), input.ADCalcStrictnesses())

                                        for TCPIndex, TrendChangepoints in enumerate(TrendChangepointList):
                                            Strictness = StrictnessNames[TCPIndex]

                                            IsTCP = np.repeat(False, len(Trend))
                                            IsTCP[TrendChangepoints[:-1]] = True

                                            TCPInit, TCPInc, TCPDec, TCPRel = ChangepointDirections(TrendChangepoints, Trend, 'Level Shift')
                                            TrendPointAnomalies = PointAnomaliesFromChangepoints(Trend, TrendChangepoints, True)

                                            ColumnAnomalies[f'Trend Changepoint - {Strictness} Strictness'] = IsTCP
                                            ColumnAnomalies[f'Trend Level Relative - {Strictness} Strictness'] = TCPRel
                                            ColumnAnomalies[f'Trend Level Average - {Strictness} Strictness'] = TCPInit
                                            ColumnAnomalies[f'Trend Level Increased - {Strictness} Strictness'] = TCPInc
                                            ColumnAnomalies[f'Trend Level Decreased - {Strictness} Strictness'] = TCPDec
                                            ColumnAnomalies[f'Trend Point Anomalies - {Strictness} Strictness'] = TrendPointAnomalies

                                        # Seasonal Anomalies
                                        SeasonalCols = [x for x in list(ColumnTransforms.columns) if x.startswith('Seasonal_')]
                                        if len(SeasonalCols) > 0:
                                            for SeasonalCol in SeasonalCols:
                                                Period = int(SeasonalCol[9:])

                                                SeasonalData = GetSeasonalData(ColumnTransforms, SeasonalCol, Period)
                                                
                                                if len(SeasonalData) > 10:
                                                    SeasonalData[['Absolute Sum','Shape']] = EmergencyDeseasonalize(SeasonalData, ['Absolute Sum','Shape'], 20000)

                                                    Window = max(input.ADCalcWindow()//Period, 5)
                                                    Minsize = max(input.ADCalcMinsize()//Period, 2)
                                                    Jump = max(input.ADCalcJump()//Period, 1)

                                                    SeasonalChangepointList, StrictnessNames = WindowSlidingSegmentation(SeasonalData['Absolute Sum'], Window, Minsize,
                                                                                                                         Jump, input.ADCalcStrictnesses())

                                                    for SCPIndex, SeasonalChangepoints in enumerate(SeasonalChangepointList):
                                                        Strictness = StrictnessNames[SCPIndex]

                                                        ActualSCPs = list(map(lambda x: int(x * Period), SeasonalChangepoints))

                                                        IsSCP = np.repeat(False, len(Trend))
                                                        IsSCP[ActualSCPs[:-1]] = True

                                                        SCPInit, SCPInc, SCPDec, SCPRel = ChangepointDirections(SeasonalChangepoints,SeasonalData['Absolute Sum'],
                                                                                                                'Level Shift', Periodwise = True, Period = Period,
                                                                                                                OriginalSeries = ColumnTransforms[SeasonalCol])
                                                        
                                                        # Point Outliers in Amplitude
                                                        AmplitudeErrors = AmplitudeOutliers(SeasonalData, Period, ColumnTransforms[SeasonalCol], SeasonalChangepoints)
                                                        # Shape Deviations
                                                        ShapeErrors = ShapeAnomalies(SeasonalData, SeasonalChangepoints, ColumnTransforms[SeasonalCol], Period)

                                                        ColumnAnomalies[f'{SeasonalCol} Changepoint - {Strictness} Strictness'] = IsSCP
                                                        ColumnAnomalies[f'{SeasonalCol} Amplitude Level Relative - {Strictness} Strictness'] = SCPRel
                                                        ColumnAnomalies[f'{SeasonalCol} Amplitude Level Average - {Strictness} Strictness'] = SCPInit
                                                        ColumnAnomalies[f'{SeasonalCol} Amplitude Level Increased - {Strictness} Strictness'] = SCPInc
                                                        ColumnAnomalies[f'{SeasonalCol} Amplitude Level Decreased - {Strictness} Strictness'] = SCPDec
                                                        ColumnAnomalies[f'{SeasonalCol} Amplitude Point Anomalies - {Strictness} Strictness'] = AmplitudeErrors
                                                        ColumnAnomalies[f'{SeasonalCol} Shape Anomalies - {Strictness} Strictness'] = ShapeErrors

                                        # Residual Anomalies
                                        Residual = ColumnTransforms['Residual']

                                        ResidualChangepointList, StrictnessNames = WindowSlidingSegmentation(Residual, input.ADCalcWindow(), input.ADCalcMinsize(),
                                                                                                             input.ADCalcJump(), input.ADCalcStrictnesses(),
                                                                                                             Task='Volatility Shift')

                                        for RCPIndex, ResidualChangepoints in enumerate(ResidualChangepointList):
                                            Strictness = StrictnessNames[RCPIndex]

                                            IsRCP = np.repeat(False, len(Trend))
                                            IsRCP[ResidualChangepoints[:-1]] = True

                                            RCPInit, RCPInc, RCPDec, RCPRel = ChangepointDirections(ResidualChangepoints,Residual,'Volatility Shift')
                                            # Point Outliers
                                            ResidualPointAnomalies = PointAnomaliesFromChangepoints(Residual, ResidualChangepoints, False)
                                            # Normality Deviations
                                            NormalityErrors = NormalityAnomalies(Residual, 50)

                                            ColumnAnomalies[f'Residual Changepoint - {Strictness} Strictness'] = IsRCP
                                            ColumnAnomalies[f'Residual Volatility Relative - {Strictness} Strictness'] = TCPRel
                                            ColumnAnomalies[f'Residual Volatility Average - {Strictness} Strictness'] = RCPInit
                                            ColumnAnomalies[f'Residual Volatility Increased - {Strictness} Strictness'] = RCPInc
                                            ColumnAnomalies[f'Residual Volatility Decreased - {Strictness} Strictness'] = RCPDec
                                            ColumnAnomalies[f'Residual Point Anomalies - {Strictness} Strictness'] = ResidualPointAnomalies
                                            ColumnAnomalies[f'Residual Normality Anomalies - {Strictness} Strictness'] = NormalityErrors

                                        ColumnAnomalyDF = pd.DataFrame(ColumnAnomalies)
                                        AllAnomalies.append(ColumnAnomalyDF)
                                        ColumnAnomalyDF.to_csv(NewPath, index=False)

                                    DataAnomalies.set(AllAnomalies)

                                return "Anomaly Detection Completed and Saved!"
                            
                        with ui.tooltip(id="ttAD_1", placement="top"):
                            ui.input_file('AnomalyDetectFiles',
                                          'Select Previously Calculated Anomaly Detection Data',
                                          multiple=True, width='100%')
                            ui.HTML('Requires first reading the data transformations')
                        
                with ui.accordion_panel("Visualization"):
                    with ui.layout_column_wrap(width=1, heights_equal='row'):
                        with ui.layout_column_wrap(width = 1/4):
                            @render.ui
                            def RenderADVisColSelect():
                                if len(DataAnomalies.get()) > 0:
                                    return ui.input_select('ADColToVis','Select a Column to Visualize',DataTransformColumns.get(), width = '100%')
                            
                            @render.ui
                            def RenderADVisComponentSelect():
                                if len(DataAnomalies.get()) > 0:
                                    ColNum = DataTransformColumns.get().index(input.ADColToVis())
                                    AnomalyData = DataAnomalies.get()[ColNum].copy()

                                    Columns = list(AnomalyData.columns)
                                    Columns.remove('Timestamp')

                                    Columns = list(map(lambda x: x.split(' ')[0], Columns))
                                    Columns = list(set(Columns))

                                    SeasonalCols = [x for x in Columns if x.startswith('Seasonal')]
                                    if len(SeasonalCols) > 0:
                                        Periods = list(map(lambda x: int(x.split('_')[-1]), SeasonalCols))
                                        Periods = sorted(Periods)

                                        Columns = ['Trend']
                                        for Period in Periods:
                                            Columns.append(f'Seasonal_{Period}')
                                        Columns.append('Residual')

                                    else:
                                        Columns = ['Trend', 'Residual']

                                    InitialComponent = ADInitVisSelectedComp.get()
                                    if InitialComponent not in Columns:
                                        SelectedComp = 0
                                    else:
                                        SelectedComp = Columns.index(InitialComponent)
                                
                                    return ui.input_select('ADCompToVis','Select Component to Visualize',Columns, selected=Columns[SelectedComp], width = '100%')

                            @render.ui()
                            def RenderADVisStrictnessTypeSelect():
                                if len(DataAnomalies.get()) > 0:
                                    ColNum = DataTransformColumns.get().index(input.ADColToVis())
                                    AnomalyData = DataAnomalies.get()[ColNum].copy()

                                    Component = input.ADCompToVis()

                                    RelevantHeaders = [x for x in list(AnomalyData.columns) if x.startswith(Component)]
                                    
                                    StrictnessTexts = list(map(lambda x: x.split(' - ')[-1], RelevantHeaders))
                                    StrictnessTypes = list(map(lambda x: x.split(' ')[:-2], StrictnessTexts))
                                    StrictnessTypeTexts = []
                                    for TypeText in StrictnessTypes:
                                        Separator = ' '
                                        StrictnessTypeTexts.append(Separator.join(TypeText))
                                        
                                    UniqueTypes = sorted(list(set(StrictnessTypeTexts)))

                                    InitialStrictness = ADInitVisSelectedPenType.get()
                                    if InitialStrictness == 'None':
                                        SelectedPen = 0
                                    else:
                                        SelectedPen = UniqueTypes.index(InitialStrictness)

                                    return ui.input_select("ADVisStrictnessType", "Penalty Calculation Method", UniqueTypes, selected=UniqueTypes[SelectedPen],
                                                           width='100%')
                                
                            @render.ui()
                            def RenderADVisStrictnessValueSelect():
                                if len(DataAnomalies.get()) > 0:
                                    ColNum = DataTransformColumns.get().index(input.ADColToVis())
                                    AnomalyData = DataAnomalies.get()[ColNum].copy()

                                    Component = input.ADCompToVis()

                                    RelevantHeaders = [x for x in list(AnomalyData.columns) if x.startswith(Component)]
                                    
                                    StrictnessTexts = list(map(lambda x: x.split(' - ')[-1], RelevantHeaders))

                                    Type = input.ADVisStrictnessType()

                                    StrictnessTextsInType = [x for x in StrictnessTexts if Type in x]
                                    StrictnessValues = list(map(lambda x: float(x.split(' ')[-2]), StrictnessTextsInType))
                                    StrictnessValues = sorted(list(set(StrictnessValues)))
                                    StrictnessValues = list(map(lambda x: '{:.2e}'.format(x), StrictnessValues))

                                    InitialStrictness = ADInitVisSelectedPenValue.get()
                                    if InitialStrictness not in StrictnessValues:
                                        SelectedVal = 0
                                    else:
                                        SelectedVal = StrictnessValues.index(InitialStrictness)

                                    return ui.input_select("ADVisStrictnessValue", "Penalty Calculation Strictness", StrictnessValues,
                                                           selected=StrictnessValues[SelectedVal], width='100%')

                        with ui.layout_column_wrap(width = 1/3): 
                            @render.ui()
                            def RenderADVisErrorSelect():
                                if len(DataAnomalies.get()) > 0:
                                    ColNum = DataTransformColumns.get().index(input.ADColToVis())
                                    AnomalyData = DataAnomalies.get()[ColNum].copy()

                                    Errors = [x for x in list(AnomalyData.columns) if input.ADCompToVis() in x]
                                    Errors = [x for x in Errors if input.ADVisStrictnessType() + ' ' + input.ADVisStrictnessValue() + ' Strictness' in x]
                                    Errors = [x for x in Errors if 'Changepoint' not in x]
                                    Errors = [x for x in Errors if not x.split(' - ')[0].endswith(' Relative')]

                                    Errors = list(map(lambda x: x.split(' - ')[0], Errors))
                                    Errors = list(map(lambda x: x[len(input.ADCompToVis()):], Errors))

                                    SelectedErrors = ADInitVisSelectedErrors.get()
                                    print(f'\n\nSelected Errors: {SelectedErrors}\n\n')
                                    if len(SelectedErrors) > 0:
                                        if SelectedErrors[0] not in Errors:
                                            InitialErrors = Errors[:3]
                                        else:
                                            IndexList = []
                                            for SelectedError in SelectedErrors:
                                                IndexList.append(Errors.index(SelectedError))
                                            InitialErrors = list(map(lambda i: Errors[i], IndexList))
                                    else:
                                        InitialErrors = Errors[:3]

                                    return ui.input_select("ADVisErrors", "Error Types to Visualize", Errors, selected=InitialErrors, multiple=True,
                                                        size=3, width='100%')

                            @render.ui()
                            def RenderADVisMaxSize():
                                if len(DataAnomalies.get()) > 0:
                                    InitialSize = len(DataAnomalies.get()[0])

                                    return ui.input_numeric('ADVisMaxSize','Maximum Samples on Plot',InitialSize, min=1, width='100%')
                                
                            @render.ui
                            def RenderADStartSlider():
                                if len(DataAnomalies.get()) > 0:
                                    ColNum = DataTransformColumns.get().index(input.ADColToVis())
                                    Data = DataTransforms.get()[ColNum].copy()

                                    if len(Data) > input.ADVisMaxSize():
                                        Maximum = len(Data) - input.ADVisMaxSize()
                                        return ui.input_slider('ADVisStartPoint', 'Starting Index of Plot', min = 0, max = Maximum, value=0, width='100%',
                                                                step=input.ADVisMaxSize()//6, animate=True)

                        @render.plot
                        def RenderADResultPlot():
                            if len(DataAnomalies.get()) > 0:
                                ColNum = DataTransformColumns.get().index(input.ADColToVis())
                                Data = DataTransforms.get()[ColNum].copy()
                                AnomalyData = DataAnomalies.get()[ColNum].copy()

                                if len(Data) > input.ADVisMaxSize():
                                    Data = Data.iloc[input.ADVisStartPoint(): input.ADVisStartPoint() + input.ADVisMaxSize() - 1]
                                    AnomalyData = AnomalyData.iloc[input.ADVisStartPoint(): input.ADVisStartPoint() + input.ADVisMaxSize() - 1]

                                Strictness = input.ADVisStrictnessType() + ' ' + input.ADVisStrictnessValue()
                                Errors = input.ADVisErrors()

                                RelevantColumns = [x for x in list(AnomalyData.columns) if Strictness + ' Strictness' in x]
                                RelevantColumns = [x for x in RelevantColumns if input.ADCompToVis() in x]

                                Subset = []
                                for Error in Errors:
                                    Subset.extend([x for x in RelevantColumns if Error in x])
                                RelevantColumns = Subset

                                X = list(Data['Timestamp'])
                                Y = list(Data[input.ADCompToVis()])

                                fig, ax = plt.subplots()

                                ax.plot(X, Y, alpha = 0.6)

                                for ColIndex, Col in enumerate(RelevantColumns):
                                    
                                    ColName = Col.split(input.ADCompToVis())[1][1:]
                                    ColName = ColName.split(Strictness + ' Strictness')[0][:-3]
                                    
                                    if 'Average' in ColName or 'Increased' in ColName or 'Decreased' in ColName or 'Shape' in ColName:
                                        Continuous = True

                                        if 'Shape' not in ColName:
                                            PlotChangepoints = True
                                        else:
                                            PlotChangepoints = False

                                    else:
                                        Continuous = False
                                        PlotChangepoints = False
                                    
                                    PlotColor = f'C{ColIndex+1}'
                                    
                                    Indices = [index for index, value in enumerate(list(AnomalyData[Col])) if value]
                                    
                                    Sequences = []
                                    
                                    if Continuous:
                                        for k, g in itertools.groupby(enumerate(Indices), lambda x: x[0]-x[1]):
                                            CurrentSequence = list(map(itemgetter(1), g))
                                            
                                            print(k, g)
                                            
                                            XStart = CurrentSequence[0]
                                            XEnd = CurrentSequence[-1] + 1
                                            
                                            Sequences.append([XStart, XEnd])
                                        
                                        for Index, Values in enumerate(Sequences):
                                            if Index == 0:
                                                ax.fill_betweenx((np.min(Y),np.max(Y)),(X[Values[0]]),(X[np.min((len(X) - 1, Values[1]))]),
                                                                color=PlotColor, alpha=0.4, label=ColName)
                                            else:
                                                ax.fill_betweenx((np.min(Y),np.max(Y)),(X[Values[0]]),(X[np.min((len(X) - 1, Values[1]))]),
                                                                color=PlotColor, alpha=0.4)
                                    else:
                                        ax.scatter(pd.Series(X).iloc[Indices], pd.Series(Y).iloc[Indices], color = PlotColor, marker = 'x', label = ColName)

                                    if PlotChangepoints:
                                        Changepoints = AnomalyData[f'{input.ADCompToVis()} Changepoint - {Strictness} Strictness']
                                        ChangepointIndices = Changepoints.index[Changepoints].tolist()

                                        for ChangepointIndex in ChangepointIndices:
                                            ax.axvline(X[ChangepointIndex], color = 'dimgrey', linestyle = '-.')
                                        
                                ax.legend(loc='upper right')
                                fig.suptitle(f'Selected Errors in the {input.ADCompToVis()} Component of {input.ADColToVis()}')
                            
                                return fig
                            
        with ui.nav_panel("Anomalies Between Features"):
            with ui.accordion(id='ADFeatVisAccordion', open='Plot Settings', height='200%'):
                with ui.accordion_panel('Plot Settings'):
                    with ui.layout_column_wrap(width=1, heights_equal='row'):
                        # Plot Columns and Plot Type
                        with ui.layout_columns(col_widths=(5,7)):
                            with ui.card():
                                ui.h6('Features to Plot')

                                with ui.layout_column_wrap(width=1/2):
                                    ui.input_radio_buttons("ADFeatVisColumnAllOrSelect","Columns To Visualize",['All','Selected'],width='100%')

                                    @render.ui()
                                    def RenderFeatVisColumnDropdown():
                                        return ui.input_select("ADFeatVisSelectedColumns","Selected Columns",DataTransformColumns.get(),width='100%', multiple=True,
                                                               size=3)

                            with ui.card():
                                ui.h6('Errors To Plot (On Boolean Plots)')
                                
                                with ui.layout_column_wrap(width = 1/3):
                                    ui.input_radio_buttons('ADFeatVisErrorCategory', 'Category of Errors to Plot', ['Point','Continuous'], selected='Point',
                                                           width = '100%')
                                    
                                    ui.input_radio_buttons('ADFeatVisErrorAllOrSelect', 'Error Types to Plot', ['All','Selected'], selected='All', width = '100%')

                                    @render.ui
                                    def RenderFeatVisErrorTypeDropdown():
                                        if input.ADFeatVisErrorCategory() == 'Point':
                                            ErrorTypeList = ['Trend Point Anomalies', 'Amplitude Point Anomalies', 'Residual Point Anomalies',
                                                             'Residual Normality Anomalies']
                                            
                                        elif input.ADFeatVisErrorCategory() == 'Continuous':
                                            ErrorTypeList = ['Trend Level Average', 'Trend Level Increased', 'Trend Level Decreased',
                                                            'Amplitude Level Average', 'Amplitude Level Increased', 'Amplitude Level Decreased', 'Shape Anomalies',
                                                            'Residual Volatility Average', 'Residual Volatility Increased', 'Residual Volatility Decreased']
                                            
                                        return ui.input_select('ADFeatVisSelectedErrors', 'Selected Error Types to Plot', ErrorTypeList, multiple=True,
                                                               width = '100%', size=3)
                                    
                        # Plot Component Penalty Settings           
                        with ui.layout_column_wrap(width=1/3):
                            with ui.card():
                                # Trend Settings
                                with ui.layout_column_wrap(width=1, heights_equal='row'):
                                    ui.h6('Trend')

                                    with ui.layout_column_wrap(width=1/3):
                                        @render.ui
                                        def RenderADFeatVisTrendPenSelect():
                                            if len(DataAnomalies.get()) > 0:
                                                Component = 'Trend'

                                                for ColNum, ColumnName in enumerate(DataTransformColumns.get()):
                                                    AnomalyData = DataAnomalies.get()[ColNum].copy()

                                                    RelevantHeaders = [x for x in list(AnomalyData.columns) if x.startswith(Component)]
                                                    
                                                    StrictnessTexts = list(map(lambda x: x.split(' - ')[-1], RelevantHeaders))
                                                    StrictnessTypes = list(map(lambda x: x.split(' ')[:-2], StrictnessTexts))
                                                    StrictnessTypeTexts = []

                                                    for TypeText in StrictnessTypes:
                                                        Separator = ' '
                                                        StrictnessTypeTexts.append(Separator.join(TypeText))
                                                        
                                                    UniqueTypes = sorted(list(set(StrictnessTypeTexts)))

                                                    if ColNum == 0:
                                                        PenaltyTypes = UniqueTypes
                                                        PreviousTypes = UniqueTypes
                                                    else:
                                                        PenaltyTypes = [value for value in PreviousTypes if value in UniqueTypes]
                                                        PreviousTypes = UniqueTypes
                                                    
                                                return ui.input_select('ADFeatVisTrendPenType', 'Penalty Type', PenaltyTypes, selected=PenaltyTypes[0],
                                                                        width='100%')

                                        @render.ui
                                        def RenderADFeatVisTrendPenMin():
                                            if len(DataAnomalies.get()) > 0:
                                                PenaltyType = input.ADFeatVisTrendPenType()
                                                Component = 'Trend'

                                                for ColNum, ColumnName in enumerate(DataTransformColumns.get()):
                                                    AnomalyData = DataAnomalies.get()[ColNum].copy()

                                                    RelevantHeaders = [x for x in list(AnomalyData.columns) if x.startswith(Component)]
                                                    
                                                    StrictnessTexts = list(map(lambda x: x.split(' - ')[-1], RelevantHeaders))

                                                    StrictnessTextsInType = [x for x in StrictnessTexts if PenaltyType in x]
                                                    StrictnessValues = list(map(lambda x: float(x.split(' ')[-2]), StrictnessTextsInType))
                                                    StrictnessValues = sorted(list(set(StrictnessValues)))

                                                    if ColNum == 0:
                                                        UniqueStrictnesses = StrictnessValues
                                                    else:
                                                        UniqueStrictnesses.extend(StrictnessValues)

                                                UniqueStrictnesses = sorted(list(set(UniqueStrictnesses)))

                                                return ui.input_select('ADFeatVisTrendPenMin', 'Minimum Value', UniqueStrictnesses, selected=UniqueStrictnesses[0],
                                                                       width='100%')

                                        @render.ui
                                        def RenderADFeatVisTrendPenMax():
                                            if len(DataAnomalies.get()) > 0:
                                                PenaltyType = input.ADFeatVisTrendPenType()
                                                Component = 'Trend'
                                                SelectedMin = float(input.ADFeatVisTrendPenMin())

                                                for ColNum, ColumnName in enumerate(DataTransformColumns.get()):
                                                    AnomalyData = DataAnomalies.get()[ColNum].copy()

                                                    RelevantHeaders = [x for x in list(AnomalyData.columns) if x.startswith(Component)]
                                                    
                                                    StrictnessTexts = list(map(lambda x: x.split(' - ')[-1], RelevantHeaders))

                                                    StrictnessTextsInType = [x for x in StrictnessTexts if PenaltyType in x]
                                                    StrictnessValues = list(map(lambda x: float(x.split(' ')[-2]), StrictnessTextsInType))
                                                    StrictnessValues = sorted(list(set(StrictnessValues)))

                                                    if ColNum == 0:
                                                        UniqueStrictnesses = StrictnessValues
                                                    else:
                                                        UniqueStrictnesses.extend(StrictnessValues)

                                                UniqueStrictnesses = [x for x in UniqueStrictnesses if x > SelectedMin]
                                                UniqueStrictnesses = sorted(list(set(UniqueStrictnesses)))

                                                return ui.input_select('ADFeatVisTrendPenMax', 'Maximum Value', UniqueStrictnesses, selected=UniqueStrictnesses[-1],
                                                                       width='100%')

                            with ui.card():
                                # Seasonal Settings
                                with ui.layout_column_wrap(width=1, heights_equal='row'):
                                    ui.h6('Seasonal')

                                    with ui.layout_column_wrap(width=1/3):
                                        @render.ui
                                        def RenderADFeatVisSeasonalPenSelect():
                                            if len(DataAnomalies.get()) > 0:
                                                Component = 'Seasonal'

                                                FirstDone = False

                                                for ColNum, ColumnName in enumerate(DataTransformColumns.get()):
                                                    AnomalyData = DataAnomalies.get()[ColNum].copy()

                                                    RelevantHeaders = [x for x in list(AnomalyData.columns) if x.startswith(Component)]

                                                    if len(RelevantHeaders) > 0:
                                                        StrictnessTexts = list(map(lambda x: x.split(' - ')[-1], RelevantHeaders))
                                                        StrictnessTypes = list(map(lambda x: x.split(' ')[:-2], StrictnessTexts))
                                                        StrictnessTypeTexts = []

                                                        for TypeText in StrictnessTypes:
                                                            Separator = ' '
                                                            StrictnessTypeTexts.append(Separator.join(TypeText))
                                                            
                                                        UniqueTypes = sorted(list(set(StrictnessTypeTexts)))

                                                        if not FirstDone:
                                                            PenaltyTypes = UniqueTypes
                                                            PreviousTypes = UniqueTypes

                                                            FirstDone = True
                                                        else:
                                                            PenaltyTypes = [value for value in PreviousTypes if value in UniqueTypes]
                                                            PreviousTypes = UniqueTypes
                                                    
                                                return ui.input_select('ADFeatVisSeasonalPenType', 'Penalty Type', PenaltyTypes, selected=PenaltyTypes[0],
                                                                        width='100%')

                                        @render.ui
                                        def RenderADFeatVisSeasonalPenMin():
                                            if len(DataAnomalies.get()) > 0:
                                                PenaltyType = input.ADFeatVisSeasonalPenType()
                                                Component = 'Seasonal'

                                                FirstDone = False

                                                for ColNum, ColumnName in enumerate(DataTransformColumns.get()):
                                                    AnomalyData = DataAnomalies.get()[ColNum].copy()

                                                    RelevantHeaders = [x for x in list(AnomalyData.columns) if x.startswith(Component)]
                                                    
                                                    StrictnessTexts = list(map(lambda x: x.split(' - ')[-1], RelevantHeaders))

                                                    StrictnessTextsInType = [x for x in StrictnessTexts if PenaltyType in x]
                                                    StrictnessValues = list(map(lambda x: float(x.split(' ')[-2]), StrictnessTextsInType))
                                                    StrictnessValues = sorted(list(set(StrictnessValues)))

                                                    if ColNum == 0:
                                                        UniqueStrictnesses = StrictnessValues
                                                    else:
                                                        UniqueStrictnesses.extend(StrictnessValues)
                                                        
                                                UniqueStrictnesses = sorted(list(set(UniqueStrictnesses)))

                                                return ui.input_select('ADFeatVisSeasonalPenMin', 'Minimum Value', UniqueStrictnesses, selected=UniqueStrictnesses[0],
                                                                       width='100%')

                                        @render.ui
                                        def RenderADFeatVisSeasonalPenMax():
                                            if len(DataAnomalies.get()) > 0:
                                                PenaltyType = input.ADFeatVisSeasonalPenType()
                                                Component = 'Seasonal'
                                                SelectedMin = float(input.ADFeatVisSeasonalPenMin())

                                                FirstDone = False

                                                for ColNum, ColumnName in enumerate(DataTransformColumns.get()):
                                                    AnomalyData = DataAnomalies.get()[ColNum].copy()

                                                    RelevantHeaders = [x for x in list(AnomalyData.columns) if x.startswith(Component)]

                                                    if len(RelevantHeaders) > 0:
                                                        StrictnessTexts = list(map(lambda x: x.split(' - ')[-1], RelevantHeaders))

                                                        StrictnessTextsInType = [x for x in StrictnessTexts if PenaltyType in x]
                                                        StrictnessValues = list(map(lambda x: float(x.split(' ')[-2]), StrictnessTextsInType))
                                                        StrictnessValues = sorted(list(set(StrictnessValues)))

                                                        if not FirstDone:
                                                            UniqueStrictnesses = StrictnessValues

                                                            FirstDone = True
                                                        else:
                                                            UniqueStrictnesses.extend(StrictnessValues)

                                                UniqueStrictnesses = [x for x in UniqueStrictnesses if x > SelectedMin]
                                                UniqueStrictnesses = sorted(list(set(UniqueStrictnesses)))

                                                return ui.input_select('ADFeatVisSeasonalPenMax', 'Maximum Value', UniqueStrictnesses, selected=UniqueStrictnesses[-1],
                                                                       width='100%')

                            with ui.card():
                                # Residual Settings
                                with ui.layout_column_wrap(width=1, heights_equal='row'):
                                    ui.h6('Residual')

                                    with ui.layout_column_wrap(width=1/3):
                                        @render.ui
                                        def RenderADFeatVisResidualPenSelect():
                                            if len(DataAnomalies.get()) > 0:
                                                Component = 'Residual'

                                                for ColNum, ColumnName in enumerate(DataTransformColumns.get()):
                                                    AnomalyData = DataAnomalies.get()[ColNum].copy()

                                                    RelevantHeaders = [x for x in list(AnomalyData.columns) if x.startswith(Component)]
                                                    
                                                    StrictnessTexts = list(map(lambda x: x.split(' - ')[-1], RelevantHeaders))
                                                    StrictnessTypes = list(map(lambda x: x.split(' ')[:-2], StrictnessTexts))
                                                    StrictnessTypeTexts = []

                                                    for TypeText in StrictnessTypes:
                                                        Separator = ' '
                                                        StrictnessTypeTexts.append(Separator.join(TypeText))
                                                        
                                                    UniqueTypes = sorted(list(set(StrictnessTypeTexts)))

                                                    if ColNum == 0:
                                                        PenaltyTypes = UniqueTypes
                                                        PreviousTypes = UniqueTypes
                                                    else:
                                                        PenaltyTypes = [value for value in PreviousTypes if value in UniqueTypes]
                                                        PreviousTypes = UniqueTypes
                                                    
                                                return ui.input_select('ADFeatVisResidualPenType', 'Penalty Type', PenaltyTypes, selected=PenaltyTypes[0],
                                                                        width='100%')

                                        @render.ui
                                        def RenderADFeatVisResidualPenMin():
                                            if len(DataAnomalies.get()) > 0:
                                                PenaltyType = input.ADFeatVisResidualPenType()
                                                Component = 'Residual'

                                                for ColNum, ColumnName in enumerate(DataTransformColumns.get()):
                                                    AnomalyData = DataAnomalies.get()[ColNum].copy()

                                                    RelevantHeaders = [x for x in list(AnomalyData.columns) if x.startswith(Component)]
                                                    
                                                    StrictnessTexts = list(map(lambda x: x.split(' - ')[-1], RelevantHeaders))

                                                    StrictnessTextsInType = [x for x in StrictnessTexts if PenaltyType in x]
                                                    StrictnessValues = list(map(lambda x: float(x.split(' ')[-2]), StrictnessTextsInType))
                                                    StrictnessValues = sorted(list(set(StrictnessValues)))

                                                    if ColNum == 0:
                                                        UniqueStrictnesses = StrictnessValues
                                                    else:
                                                        UniqueStrictnesses.extend(StrictnessValues)
                                                        
                                                UniqueStrictnesses = sorted(list(set(UniqueStrictnesses)))

                                                return ui.input_select('ADFeatVisResidualPenMin', 'Minimum Value', UniqueStrictnesses, selected=UniqueStrictnesses[0],
                                                                       width='100%')

                                        @render.ui
                                        def RenderADFeatVisResidualPenMax():
                                            if len(DataAnomalies.get()) > 0:
                                                PenaltyType = input.ADFeatVisResidualPenType()
                                                Component = 'Residual'
                                                SelectedMin = float(input.ADFeatVisResidualPenMin())

                                                for ColNum, ColumnName in enumerate(DataTransformColumns.get()):
                                                    AnomalyData = DataAnomalies.get()[ColNum].copy()

                                                    RelevantHeaders = [x for x in list(AnomalyData.columns) if x.startswith(Component)]
                                                    
                                                    StrictnessTexts = list(map(lambda x: x.split(' - ')[-1], RelevantHeaders))

                                                    StrictnessTextsInType = [x for x in StrictnessTexts if PenaltyType in x]
                                                    StrictnessValues = list(map(lambda x: float(x.split(' ')[-2]), StrictnessTextsInType))
                                                    StrictnessValues = sorted(list(set(StrictnessValues)))

                                                    if ColNum == 0:
                                                        UniqueStrictnesses = StrictnessValues
                                                    else:
                                                        UniqueStrictnesses.extend(StrictnessValues)

                                                UniqueStrictnesses = [x for x in UniqueStrictnesses if x > SelectedMin]
                                                UniqueStrictnesses = sorted(list(set(UniqueStrictnesses)))

                                                return ui.input_select('ADFeatVisResidualPenMax', 'Maximum Value', UniqueStrictnesses, selected=UniqueStrictnesses[-1],
                                                                       width='100%')
                                                     
                        ui.input_action_button('ADFeatVisSaveSettings', 'Move Forward with Selected Subset of Anomaly Data', width='100%')

                        @render.ui
                        @reactive.event(input.ADFeatVisSaveSettings)
                        async def CreateAnomalyDetectionSaveProgressBar():
                            with ui.Progress() as Progress:
                                Progress.set(message="Saving Anomaly Detection Results")

                                ErrorCategories = ['Trend Level Average', 'Trend Level Increased', 'Trend Level Decreased', 'Trend Point Anomalies',
                                                   'Amplitude Level Average', 'Amplitude Level Increased', 'Amplitude Level Decreased',
                                                   'Shape Anomalies', 'Amplitude Point Anomalies',
                                                   'Residual Volatility Average', 'Residual Volatility Increased', 'Residual Volatility Decreased',
                                                   'Residual Point Anomalies', 'Residual Normality Anomalies']
                                
                                ErrorCategoriesRelative = ['Trend Level Relative', 'Trend Point Anomalies',
                                                           'Amplitude Level Relative', 'Shape Anomalies', 'Amplitude Point Anomalies',
                                                           'Residual Volatility Relative', 'Residual Point Anomalies', 'Residual Normality Anomalies']

                                FeatureErrorDictionary = {}
                                FeatureErrorDictionaryRelative = {}

                                for FeatureIndex, Feature in enumerate(DataTransformColumns.get()):
                                    ColumnIndex = DataTransformColumns.get().index(Feature)
                                    FeatureAnomalies = DataAnomalies.get()[ColumnIndex].copy()

                                    if FeatureIndex == 0:
                                        FeatureErrorDictionary['Timestamp'] = FeatureAnomalies['Timestamp']
                                        FeatureErrorDictionaryRelative['Timestamp'] = FeatureAnomalies['Timestamp']

                                    ValueData = DataTransforms.get()[ColumnIndex].copy()
                                    ValueComponents = list(ValueData.columns)
                                    ValueComponents.remove('Timestamp')
                                    for Index, ValueComp in enumerate(ValueComponents):
                                        if Index == 0:
                                            Value = ValueData[ValueComp]
                                        else:
                                            Value += ValueData[ValueComp]

                                    FeatureErrorDictionary[Feature] = Value
                                    FeatureErrorDictionaryRelative[Feature] = Value

                                    Components = list(FeatureAnomalies.columns)
                                    Components.remove('Timestamp')

                                    Components = list(map(lambda x: x.split(' ')[0], Components))
                                    Components = list(set(Components))

                                    HasSeasonalComponent = False

                                    SeasonalCols = [x for x in Components if x.startswith('Seasonal')]
                                    if len(SeasonalCols) > 0:
                                        Periods = list(map(lambda x: int(x.split('_')[-1]), SeasonalCols))
                                        Periods = sorted(Periods)

                                        Components = ['Trend']
                                        for Period in Periods:
                                            Components.append(f'Seasonal_{Period}')
                                        Components.append('Residual')

                                        HasSeasonalComponent = True

                                    else:
                                        Components = ['Trend', 'Residual']

                                    # Trend Error Components
                                    Component = 'Trend'
                                    PenaltyType = input.ADFeatVisTrendPenType()
                                    PenaltyMin = float(input.ADFeatVisTrendPenMin())
                                    PenaltyMax = float(input.ADFeatVisTrendPenMax())

                                    # BIC, Variance, and Log of Samples has the strictest penalty be the one of the smallest value, while with the other penalty types it's the highest
                                    if PenaltyType == 'BIC' or PenaltyType == 'Log of Samples' or PenaltyType == 'Variance':
                                        UseMin = True
                                    else:
                                        UseMin = False

                                    AllColumns = list(FeatureAnomalies.columns)
                                    RelevantColumns = [x for x in AllColumns if Component in x and PenaltyType in x]
                                    PenaltyValues = [float(x.split(' - ')[-1].split(' ')[-2]) for x in RelevantColumns]
                                    PenaltyValues = list(set(PenaltyValues))

                                    # If my set penalty value is in the list of all available ones, use that penalty, otherwise use the closest one
                                    if UseMin:
                                        if PenaltyMin in PenaltyValues:
                                            StrictestIndex = PenaltyValues.index(PenaltyMin)
                                        else:
                                            # With use minimum penalty value, I use the penalty that is smaller than the cutoff if available
                                            StrictestIndex = [i for i, val in enumerate(PenaltyValues) if val < PenaltyMin]
                                            
                                            if len(StrictestIndex) > 0:
                                                StrictestIndex = StrictestIndex[-1]
                                            else:
                                                StrictestIndex = 0
                                    else:
                                        if PenaltyMax in PenaltyValues:
                                            StrictestIndex = PenaltyValues.index(PenaltyMax)
                                        else:
                                            # With use maximum penalty value, I use the penalty that is larger than the cutoff if available
                                            StrictestIndex = [i for i, val in enumerate(PenaltyValues) if val > PenaltyMax]
                                            
                                            if len(StrictestIndex) > 0:
                                                StrictestIndex = StrictestIndex[0]
                                            else:
                                                StrictestIndex = len(PenaltyValues) - 1

                                    if len(PenaltyValues) > 0:
                                        PenaltyValue = '{:.2e}'.format(PenaltyValues[StrictestIndex])

                                        for ErrorCategory in ErrorCategories:
                                            ColumnText = f'{ErrorCategory} - {PenaltyType} {PenaltyValue} Strictness'
                                            
                                            if ColumnText in RelevantColumns:
                                                FeatureErrorDictionary[f'{Feature} - {ErrorCategory}'] = FeatureAnomalies[ColumnText]

                                        for ErrorCategory in ErrorCategoriesRelative:
                                            ColumnText = f'{ErrorCategory} - {PenaltyType} {PenaltyValue} Strictness'
                                            
                                            if ColumnText in RelevantColumns:
                                                FeatureErrorDictionaryRelative[f'{Feature} - {ErrorCategory}'] = FeatureAnomalies[ColumnText]

                                    # Seasonal Error Components
                                    if HasSeasonalComponent:
                                        PenaltyType = input.ADFeatVisSeasonalPenType()
                                        PenaltyMin = float(input.ADFeatVisSeasonalPenMin())
                                        PenaltyMax = float(input.ADFeatVisSeasonalPenMax())

                                        for Period in Periods:
                                            Component = f'Seasonal_{Period}'

                                            # BIC, Variance, and Log of Samples has the strictest penalty be the one of the smallest value, while with the other penalty types it's the highest
                                            if PenaltyType == 'BIC' or PenaltyType == 'Log of Samples' or PenaltyType == 'Variance':
                                                UseMin = True
                                            else:
                                                UseMin = False

                                            AllColumns = list(FeatureAnomalies.columns)
                                            RelevantColumns = [x for x in AllColumns if Component in x and PenaltyType in x]
                                            PenaltyValues = [float(x.split(' - ')[-1].split(' ')[-2]) for x in RelevantColumns]
                                            PenaltyValues = list(set(PenaltyValues))

                                            # If my set penalty value is in the list of all available ones, use that penalty, otherwise use the closest one
                                            if UseMin:
                                                if PenaltyMin in PenaltyValues:
                                                    StrictestIndex = PenaltyValues.index(PenaltyMin)
                                                else:
                                                    # With use minimum penalty value, I use the penalty that is smaller than the cutoff if available
                                                    StrictestIndex = [i for i, val in enumerate(PenaltyValues) if val < PenaltyMin]
                                                    
                                                    if len(StrictestIndex) > 0:
                                                        StrictestIndex = StrictestIndex[-1]
                                                    else:
                                                        StrictestIndex = 0
                                            else:
                                                if PenaltyMax in PenaltyValues:
                                                    StrictestIndex = PenaltyValues.index(PenaltyMax)
                                                else:
                                                    # With use maximum penalty value, I use the penalty that is larger than the cutoff if available
                                                    StrictestIndex = [i for i, val in enumerate(PenaltyValues) if val > PenaltyMax]
                                                    
                                                    if len(StrictestIndex) > 0:
                                                        StrictestIndex = StrictestIndex[0]
                                                    else:
                                                        StrictestIndex = len(PenaltyValues) - 1

                                            if len(PenaltyValues) > 0:
                                                PenaltyValue = '{:.2e}'.format(PenaltyValues[StrictestIndex])

                                                for ErrorCategory in ErrorCategories:
                                                    ColumnText = f'{Component} {ErrorCategory} - {PenaltyType} {PenaltyValue} Strictness'
                                                    
                                                    if ColumnText in RelevantColumns:
                                                        FeatureErrorDictionary[f'{Feature} - {Component} {ErrorCategory}'] = FeatureAnomalies[ColumnText]

                                                for ErrorCategory in ErrorCategoriesRelative:
                                                    ColumnText = f'{Component} {ErrorCategory} - {PenaltyType} {PenaltyValue} Strictness'
                                                    
                                                    if ColumnText in RelevantColumns:
                                                        FeatureErrorDictionaryRelative[f'{Feature} - {Component} {ErrorCategory}'] = FeatureAnomalies[ColumnText]

                                    # Residual Error Components
                                    Component = 'Residual'
                                    PenaltyType = input.ADFeatVisResidualPenType()
                                    PenaltyMin = float(input.ADFeatVisResidualPenMin())
                                    PenaltyMax = float(input.ADFeatVisResidualPenMax())

                                    # BIC, Variance, and Log of Samples has the strictest penalty be the one of the smallest value, while with the other penalty types it's the highest
                                    if PenaltyType == 'BIC' or PenaltyType == 'Log of Samples' or PenaltyType == 'Variance':
                                        UseMin = True
                                    else:
                                        UseMin = False

                                    AllColumns = list(FeatureAnomalies.columns)
                                    RelevantColumns = [x for x in AllColumns if Component in x and PenaltyType in x]
                                    PenaltyValues = [float(x.split(' - ')[-1].split(' ')[-2]) for x in RelevantColumns]
                                    PenaltyValues = list(set(PenaltyValues))
                                    
                                    # If my set penalty value is in the list of all available ones, use that penalty, otherwise use the closest one
                                    if UseMin:
                                        if PenaltyMin in PenaltyValues:
                                            StrictestIndex = PenaltyValues.index(PenaltyMin)
                                        else:
                                            # With use minimum penalty value, I use the penalty that is smaller than the cutoff if available
                                            StrictestIndex = [i for i, val in enumerate(PenaltyValues) if val < PenaltyMin]
                                            
                                            if len(StrictestIndex) > 0:
                                                StrictestIndex = StrictestIndex[-1]
                                            else:
                                                StrictestIndex = 0
                                    else:
                                        if PenaltyMax in PenaltyValues:
                                            StrictestIndex = PenaltyValues.index(PenaltyMax)
                                        else:
                                            # With use maximum penalty value, I use the penalty that is larger than the cutoff if available
                                            StrictestIndex = [i for i, val in enumerate(PenaltyValues) if val > PenaltyMax]
                                            
                                            if len(StrictestIndex) > 0:
                                                StrictestIndex = StrictestIndex[0]
                                            else:
                                                StrictestIndex = len(PenaltyValues) - 1

                                    if len(PenaltyValues) > 0:
                                        PenaltyValue = '{:.2e}'.format(PenaltyValues[StrictestIndex])

                                        for ErrorCategory in ErrorCategories:
                                            ColumnText = f'{ErrorCategory} - {PenaltyType} {PenaltyValue} Strictness'
                                            
                                            if ColumnText in RelevantColumns:
                                                FeatureErrorDictionary[f'{Feature} - {ErrorCategory}'] = FeatureAnomalies[ColumnText]

                                        for ErrorCategory in ErrorCategoriesRelative:
                                            ColumnText = f'{ErrorCategory} - {PenaltyType} {PenaltyValue} Strictness'
                                            
                                            if ColumnText in RelevantColumns:
                                                FeatureErrorDictionaryRelative[f'{Feature} - {ErrorCategory}'] = FeatureAnomalies[ColumnText]

                                ErrorDataframe = pd.DataFrame(FeatureErrorDictionary)
                                ErrorDataframeRelative = pd.DataFrame(FeatureErrorDictionaryRelative)
                                
                                Desktop = winshell.desktop()

                                Folder = Desktop + '\\Data Cleaning - ' + input.FolderName()

                                if not os.path.exists(Folder):
                                    os.makedirs(Folder)

                                NewPath = Folder + '\\Boolean Anomaly Detection Results - Set Strictnesses.csv'
                                
                                DataAnomaliesFiltered.set(ErrorDataframe)
                                ErrorDataframe.to_csv(NewPath, index=False)

                                NewPathRel = Folder + '\\Relative Anomaly Detection Results - Set Strictnesses.csv'
                                ErrorDataframeRelative.to_csv(NewPathRel, index=False)

                            return "Anomaly Detection Results Saved!"


                with ui.accordion_panel('Plot with the Strictest Available Settings'):
                    with ui.layout_column_wrap(width = 1, heights_equal='row'):
                        with ui.layout_columns(col_widths=(6,3,3)):
                            ui.input_radio_buttons('ADFeatVisPlotType', 'Select Plot Type', ['Boolean Cumulative Sum Plot','Boolean Spanwise Plot',
                                                                                             'Boolean Correlation Heatmap', 'Relative Correlation Heatmap'],
                                                    selected = 'Boolean Cumulative Sum Plot', width='100%', inline=True)

                            @render.ui
                            def ADFeatVisPlotHideOrShowLegend():
                                if input.ADFeatVisPlotType() == 'Boolean Cumulative Sum Plot' or input.ADFeatVisPlotType() == 'Boolean Spanwise Plot':
                                    return ui.input_radio_buttons('ADFeatVisShowLegend', 'Legend on Plot', ['Show','Hide'],
                                                                  selected = 'Show', width='100%', inline=True)
                            
                            with ui.layout_column_wrap(width=1):
                                @render.ui
                                def ADFeatVisPlotNumberOfSpans():
                                    if input.ADFeatVisPlotType() == 'Boolean Spanwise Plot':
                                        return ui.input_numeric('ADFeatVisSpans', 'Number of Spans to Plot', 10, min=2, width='100%')

                                @render.text
                                def SpanTextADFeatPlot():
                                    if input.ADFeatVisPlotType() == 'Boolean Spanwise Plot':
                                        Data = ADFeaturePlotData.get().copy()

                                        SpanLength = len(Data)//input.ADFeatVisSpans()
                                        SpanTime = str(SpanLength * (Data['Timestamp'][1] - Data['Timestamp'][0]))
                                        return f"Span length: {SpanTime}"

                        @render.plot(alt='No Data Available')
                        def ADFeatVisPlot():
                            if len(DataAnomalies.get()) > 0:
                                PlotData = ADFeaturePlotData.get().copy()

                                if len(PlotData) > 0:
                                    PlotType = input.ADFeatVisPlotType()

                                    BooleanColumns = list(PlotData.columns)
                                    BooleanColumns.remove('Timestamp')

                                    NewColumns = list(np.arange(1, len(BooleanColumns)+1))
                                    NewColumns = [f'Col {x}' for x in NewColumns]

                                    for BooleanIndex, BooleanColumn in enumerate(BooleanColumns):
                                        PlotData = PlotData.rename(columns={BooleanColumn: NewColumns[BooleanIndex]})

                                    if PlotType == 'Boolean Cumulative Sum Plot':
                                        fig, ax = plt.subplots()

                                        CurrentMax = 0

                                        for Column in NewColumns:
                                            CumSum = PlotData[Column].cumsum()
                                            ax.plot(PlotData['Timestamp'], CumSum, label = Column)
                                            CurrentMax = max(max(CumSum), CurrentMax)

                                        ax.set_xlabel('Timestamp')
                                        ax.set_ylim(0,CurrentMax * 1.2)
                                        ax.set_title('Cumulative Sum of Detected Anomalies')

                                        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
                                        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
                                        LegendCols = max(len(NewColumns) // 10, 1)
                                        if input.ADFeatVisShowLegend() == 'Show':
                                            fig.legend(lines, labels, loc='upper left', ncols=LegendCols)

                                        return fig

                                    if PlotType == 'Boolean Spanwise Plot':
                                        Spans = input.ADFeatVisSpans()

                                        SplitMatrices = np.array_split(PlotData, Spans)

                                        fig, ax = plt.subplots()

                                        for Column in NewColumns:
                                            InSpan = list(map(lambda Span: SplitMatrices[Span][Column].sum(), np.arange(Spans)))
                                            ax.bar(np.arange(1,Spans+1), InSpan, label = Column, alpha=0.6)
                                        ax.set_xlabel('Span')
                                        ax.set_xticks(np.arange(1,Spans+1))
                                        ax.set_title('Anomalies Detected Within Span')
                                        
                                        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
                                        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
                                        LegendCols = max(len(NewColumns) // 10, 1)
                                        if input.ADFeatVisShowLegend() == 'Show':
                                            fig.legend(lines, labels, loc='upper left', ncols=LegendCols)

                                        return fig

                                    if PlotType == 'Boolean Correlation Heatmap':
                                        Data = PlotData.copy()[NewColumns]

                                        fig, ax = plt.subplots()
                                        if len(NewColumns) > 20:
                                            ax = sns.heatmap(Data.loc[:, Data.any()].corr(), vmin = 0, vmax = 1,linewidths = 0.5,
                                                             linecolor = 'black')
                                        else:
                                            ax = sns.heatmap(Data.loc[:, Data.any()].corr(), annot=True, fmt='.2f', vmin = 0, vmax = 1,linewidths = 0.5,
                                                             linecolor = 'black')
                                        plt.xticks(rotation=45)
                                        plt.yticks(rotation=0)
                                        ax.set_title('Anomaly Correlation Heatmap')
                                    
                                        return fig
                                    
                                    if PlotType == 'Relative Correlation Heatmap':
                                        Data = PlotData.copy()[NewColumns]

                                        fig, ax = plt.subplots()
                                        if len(NewColumns) > 20:
                                            ax = sns.heatmap(Data.corr(), vmin = 0, vmax = 1,linewidths = 0.5,
                                                             linecolor = 'black')
                                        else:
                                            ax = sns.heatmap(Data.corr(), annot=True, fmt='.2f', vmin = 0, vmax = 1,linewidths = 0.5,
                                                             linecolor = 'black')
                                        plt.xticks(rotation=45)
                                        plt.yticks(rotation=0)
                                        ax.set_title('Anomaly Correlation Heatmap Between The Relative Values Within Components')
                                    
                                        return fig
                                
                        @render.data_frame
                        def RenderADFeatVisLegend():
                            PlotData = ADFeaturePlotData.get().copy()
                            BooleanColumns = list(PlotData.columns)
                            
                            if len(BooleanColumns) > 0:
                                BooleanColumns.remove('Timestamp')

                                ToRender = pd.DataFrame({'Column Number': np.arange(1, len(BooleanColumns)+1), 'Error Type': BooleanColumns})

                                return render.DataGrid(ToRender, row_selection_mode = 'multiple', width = '100%')
#%% UI Anomaly Rectification Panel      
with ui.nav_panel("Anomaly Rectification"):
    with ui.navset_card_underline():
        with ui.nav_panel("Calculation"):
            with ui.layout_column_wrap(width = 1, heights_equal='row'):
                with ui.layout_column_wrap(width=1/2):
                    ui.input_file('FinalizedAnomalyDetectionData', 'Select Anomaly Detection Result with Boolean Set Strictnesses', width = '100%')

                    ui.input_slider('AnomalyCorrectionMinimumCorrelation','Set Correlation Cutoff Value', 0.1, 1, 0.8, step=0.05, width = '100%')

                "Amount of Data Being Corrected (Please wait for the dataframe to appear before calculating anomaly rectification)"
                @render.data_frame
                def RenderDataCorrectionAmountTable():
                    ADResult = DataAnomaliesFiltered.get().copy()
                    if len(ADResult) > 0:
                        AllColumns = list(ADResult.columns)
                        AllColumns.remove('Timestamp')

                        # Get unique feature names
                        Features = []
                        FeatureIndices = []
                        for Col in AllColumns:
                            FeatureName = Col.split(' - ')[0]
                            Features.append(FeatureName)
                            
                        Features = list(set(Features))
                        AllColumnsNoFeatures = [x for x in AllColumns if x not in Features]

                        FeaturePeriodicities = []
                        # Get periods for each feature
                        for Feature in Features:
                            FeatureErrorColumns = [x.split(' - ')[1] for x in AllColumnsNoFeatures if x.startswith(Feature)]
                            Periodicities = [int(x.split(' ')[0].split('_')[1]) for x in FeatureErrorColumns if x.startswith('Seasonal')]
                            FeaturePeriodicities.append(sorted(list(set(Periodicities))))

                        # Get indices where the feature value is
                        FeatureIndices = []
                        for Feature in Features:
                            FeatureIndex = [i for i in list(np.arange(len(AllColumns))) if AllColumns[i] == Feature][0]
                            FeatureIndices.append(FeatureIndex)

                        ErrorIndices = list(np.arange(len(AllColumns)))
                        ErrorIndices = [index for index in ErrorIndices if index not in FeatureIndices]

                        # Get separate point and continuous anomalies for each feature
                        PointErrorsByFeature = []
                        ContinuousErrorsByFeature = []
                        AllContinuousErrors = []

                        for Feature in Features:
                            PointErrors = []
                            ContinuousErrors = []
                            
                            FeatureErrorIndices = [i for i in ErrorIndices if Feature in AllColumns[i]]
                            
                            for ErrorIndex in FeatureErrorIndices:
                                ErrorType = AllColumns[ErrorIndex].split(' - ')[-1]
                            
                                if 'Point' in ErrorType or 'Normality' in ErrorType:
                                    PointErrors.append(ErrorIndex)
                                else:
                                    ContinuousErrors.append(ErrorIndex)
                                    
                            PointErrorsByFeature.append(PointErrors)
                            ContinuousErrorsByFeature.append(ContinuousErrors)
                            AllContinuousErrors.extend(ContinuousErrors)

                        # =============================================================================
                        # I look for a point anomaly in each feature
                        # If only one feature has a point anomaly, it is an outlier
                        # =============================================================================
                        HasPointAnomaly = {}
                        PointErrorIndices, PointErrorFeatures = [], []

                        for FeatureIndex, Feature in enumerate(Features):
                            FeaturePointErrors = PointErrorsByFeature[FeatureIndex]
                            FeaturePointErrorCols = [AllColumns[i] for i in FeaturePointErrors]
                            
                            SubData = ADResult[FeaturePointErrorCols]
                            # Get true or false, feature has at least 1 point anomaly
                            HasPointAnomaly[Feature] = [i >= 1 for i in list(SubData.sum(axis=1))]

                        HasPointAnomaly = pd.DataFrame(HasPointAnomaly)
                        PointEvent = HasPointAnomaly.sum(axis=1)
                        # 1 true
                        Indices = list(PointEvent[PointEvent == 1].index)
                        if len(Indices) > 0:
                            OutOfPlaceFeatures = list(HasPointAnomaly.iloc[Indices].idxmax(axis=1))
                            
                            PointErrorIndices.extend(Indices)
                            PointErrorFeatures.extend(OutOfPlaceFeatures)

                        # =============================================================================
                        # Continuous anomalies: first determine what components are strongly correlated between features
                        # e.g.: if voltage trend is elevated, current seasonality is also elevated
                        # Get correlation matrix, look at all correlations outside the own feature
                        #     Corr matrix
                        #     Go by column by column
                        #     Eliminate rows from own feature
                        #     Look for elements greater than cutoff
                        # =============================================================================
                        CorrelationCutoff = input.AnomalyCorrectionMinimumCorrelation()

                        ContinuousErrorColumns = [AllColumns[i] for i in AllContinuousErrors]
                        ContinuousErrorCorrelationData = ADResult[ContinuousErrorColumns].corr()
                        CECorrIndex = ContinuousErrorCorrelationData.index

                        AllCorrelatedRows = []

                        for ContinuousError in CECorrIndex:
                            # I am looking for inter-series correlation, so I weed out anything from the same feature
                            Feature = ContinuousError.split(' - ')[0]
                            
                            RowData = ContinuousErrorCorrelationData.loc[ContinuousError]
                            CorrelatedRows = list(RowData[RowData > CorrelationCutoff].index)
                            # CorrelatedRows = [x for x in CorrelatedRows if Feature not in x]
                            CorrelatedRows.append(ContinuousError)
                            
                            UniqueFeatures = len(set([x.split(' - ')[0] for x in CorrelatedRows]))
                            
                            CorrelatedRows = sorted(CorrelatedRows)
                            CorrelatedRows = ' AND '.join(CorrelatedRows)
                            
                            if UniqueFeatures > 1:
                                AllCorrelatedRows.append(CorrelatedRows)

                        # All correlated rows now in text format, weed out the repeats
                        AllCorrelatedRows = sorted(list(set(AllCorrelatedRows)))
                        AllCorrelatedRows = [x.split(' AND ') for x in AllCorrelatedRows]
                        # Correlations between at least 3 features (with 2 features, I do not know what is relevant)
                        AllCorrelatedRows = [x for x in AllCorrelatedRows if len(x) > 2]

                        ContinuousErrorIndices, ContinuousErrorFeatures = [], []

                        for CorrelatedRows in AllCorrelatedRows:
                            SubData = ADResult[CorrelatedRows]
                            ContinuousEvent = SubData.sum(axis=1)
                            
                            # I look for only 1 out of place, meaning that the sum is either 1 (1 true, all else false) or NCol - 1 (1 false, all else true)
                            NumberOfColumns = len(CorrelatedRows)
                            
                            # 1 true
                            Indices = list(ContinuousEvent[ContinuousEvent == 1].index)
                            if len(Indices) > 0:
                                OutOfPlaceFeatures = list(SubData.iloc[Indices].idxmax(axis=1))
                                
                                ContinuousErrorIndices.extend(Indices)
                                ContinuousErrorFeatures.extend([x.split(' - ')[0] for x in OutOfPlaceFeatures])
                            
                            # 1 false
                            Indices = list(ContinuousEvent[ContinuousEvent == NumberOfColumns - 1].index)
                            if len(Indices) > 0:
                                OutOfPlaceFeatures = list(SubData.iloc[Indices].idxmin(axis=1))
                                
                                ContinuousErrorIndices.extend(Indices)
                                ContinuousErrorFeatures.extend([x.split(' - ')[0] for x in OutOfPlaceFeatures])

                        OnlyPointRemoval = GetDeletionIndices(PointErrorIndices, PointErrorFeatures, ADResult, Features)
                        OnlyContinuousRemoval = GetDeletionIndices(ContinuousErrorIndices, ContinuousErrorFeatures, ADResult, Features)

                        ErrorIndicesInit = PointErrorIndices
                        ErrorIndicesInit.extend(ContinuousErrorIndices)

                        ErrorFeaturesInit = PointErrorFeatures
                        ErrorFeaturesInit.extend(ContinuousErrorFeatures)

                        AllErrorRemoval = GetDeletionIndices(ErrorIndicesInit, ErrorFeaturesInit, ADResult, Features)

                        ResultDictionary = {}
                        ResultDictionary['Feature'] = Features
                        ResultDictionary['Only Point Error Removal'] = OnlyPointRemoval['Removed Percentages']
                        ResultDictionary['Only Continuous Error Removal'] = OnlyContinuousRemoval['Removed Percentages']
                        ResultDictionary['Removal of All Detected Errors'] = AllErrorRemoval['Removed Percentages']

                        ToRender = pd.DataFrame(ResultDictionary)
                        ToRender = ToRender.sort_values(by=['Feature'])

                        ResultDictionary['PERIndices'] = OnlyPointRemoval['Remove Index']
                        ResultDictionary['CERIndices'] = OnlyContinuousRemoval['Remove Index']
                        ResultDictionary['AERIndices'] = AllErrorRemoval['Remove Index']
                        ResultDictionary['Periods'] = FeaturePeriodicities

                        AnomalyRemovalIndexDataframe.set(pd.DataFrame(ResultDictionary))

                        return render.DataGrid(ToRender, width = '100%')

                with ui.layout_column_wrap(width=1/2):
                    ui.input_numeric('MaximumToCorrect', 'Maximum Percentage of Data to Correct Per Feature', 10, min=0, max=100, width = '100%')

                    with ui.tooltip(id="ttAnomalyCorrection_1", placement="top"):
                        @render.ui()
                        def LCMVCutoffInputAnomalyCorrection():
                            return ui.input_numeric('AnomalyCorrectionPeriodicityPermillage','Set Cutoff Permillage for Long Continuous Missing Values',
                                                    10, min=1, max=1000, width = '100%')
                        ui.HTML("The maximum period length to take into account in the long continuous imputation is one onethousands of this value.")

                ui.input_action_button('DeleteErrorsAndImpute', 'Calculate Anomaly Rectification and Save Results', width='100%')

                @render.ui
                @reactive.event(input.DeleteErrorsAndImpute)
                async def CreateErrorRectificationProgressBar():
                    FeatureData = DataAnomaliesFiltered.get().copy()
                    AnomalyIndexData = AnomalyRemovalIndexDataframe.get().copy()

                    with ui.Progress(min=1, max = len(AnomalyIndexData) + 1) as Progress:
                        Progress.set(message="Rectifying Detected Errors")

                        Features = list(AnomalyIndexData['Feature'])
                        DataColumns = Features.copy()
                        DataColumns.insert(0, 'Timestamp')
                        FeatureData = FeatureData[DataColumns]

                        OriginalFeatureData.set(FeatureData.copy())

                        MaxCorrectionAmount = input.MaximumToCorrect()

                        # Error Deletion
                        for FeatureIndex, Feature in enumerate(Features):
                            PEPerc = AnomalyIndexData['Only Point Error Removal'][FeatureIndex]
                            CEPerc = AnomalyIndexData['Only Continuous Error Removal'][FeatureIndex]
                            AllEPerc = AnomalyIndexData['Removal of All Detected Errors'][FeatureIndex]

                            if PEPerc > MaxCorrectionAmount:
                                PEPerc = 0
                            if CEPerc > MaxCorrectionAmount:
                                CEPerc = 0
                            if AllEPerc > MaxCorrectionAmount:
                                AllEPerc = 0

                            CorrectionAmounts = [PEPerc, CEPerc, AllEPerc]

                            if max(CorrectionAmounts) > 0:
                                CorrectionMethod = np.argmax(CorrectionAmounts)

                                if CorrectionMethod == 0:
                                    Indices = AnomalyIndexData['PERIndices'][FeatureIndex]
                                elif CorrectionMethod == 1:
                                    Indices = AnomalyIndexData['CERIndices'][FeatureIndex]
                                elif CorrectionMethod == 2:
                                    Indices = AnomalyIndexData['AERIndices'][FeatureIndex]
                            
                                FeatureData[Feature][Indices] = np.nan

                        FeaturesWithErrorDeletion = FeatureData.copy()
                        ErrorUncorrectedData.set(FeaturesWithErrorDeletion.copy())

                        Progress.set(detail="Error Deletion Complete")

                        MVDictionary = GetErrorDeletionMVEvalDF(FeaturesWithErrorDeletion)
                            
                        # MV Imputation for All Columns and All MV Types
                        Data = FeaturesWithErrorDeletion.copy()
                        PeriodsSeries = AnomalyIndexData['Periods']

                        Columns = list(Data.columns)
                        Columns.remove('Timestamp')

                        CutoffMultiplier = input.AnomalyCorrectionPeriodicityPermillage() / 1000

                        for ColIndex, Column in enumerate(Columns):
                            print(f'Error Rectification, Column {ColIndex + 1} of {len(Columns)}')

                            Progress.set(ColIndex + 1, detail = '')

                            Periods = PeriodsSeries[ColIndex]

                            if len(Periods) > 0:
                                # I use 1 as a cutoff minimum, meaning that a missing point is always linear interpolated rather than seasonal decomposed
                                SCCutoff = max(1, int(CutoffMultiplier * min(Periods)))
                                Periodic = True
                            else:
                                Periodic = False

                            MVDFCol = MVDictionary['MVCols'][Column]
                            if len(MVDFCol) > 0:
                                
                                if Periodic == True:
                                    SCMVs = MVDFCol.loc[np.where(MVDFCol['Length'] <= SCCutoff)]
                                    LCMVs = MVDFCol.loc[np.where(MVDFCol['Length'] > SCCutoff)]
                                elif Periodic == False:
                                    SCMVs = MVDFCol.copy()
                                    LCMVs = []
                                
                                if len(SCMVs) > 0:
                                    SCMVs = SCMVs.reset_index(drop=True)

                                    Data = SCImputation(Data, Column, SCMVs, 1000)
                                    
                                if len(LCMVs) > 0:
                                    LCMVs = LCMVs.reset_index(drop=True)

                                    LCResult = LCImputation(Data, Column, LCMVs, CutoffMultiplier, Periods)
                                    Data = LCResult['Data']
                                    LeftoverSCMVs = LCResult['Leftover SCMVs']
                                    
                                    if len(LeftoverSCMVs) > 0:
                                        Data = SCImputation(Data, Column, LeftoverSCMVs, 1000)

                                    # print(f'\n\n{Column} SCMVs: {SCMVs}\nLCMVs: {LCMVs}\n\n')

                        Desktop = winshell.desktop()

                        Folder = Desktop + '\\Data Cleaning - ' + input.FolderName()

                        if not os.path.exists(Folder):
                            os.makedirs(Folder)

                        NewPath = Folder + '\\Detected Errors Corrected.csv'
                        Data.to_csv(NewPath)

                        ErrorCorrectedData.set(Data.copy())

                    return f"Detected Errors Rectified! Remaining Missing Values: {Data.isnull().sum().sum()}"

        with ui.nav_panel("Visualization"):
            with ui.layout_column_wrap(width=1, heights_equal='row'):
                with ui.layout_column_wrap(width=1/3):
                    @render.ui()
                    def ECVisResampleDropdownSingle():
                        return ui.input_select("ECVisResRateSingle","Resampling Rate",GetECVisResampleRates()['Readable'],width='100%')

                    @render.ui()
                    def ECVisMaxSamplesSetting():
                        Data = OriginalFeatureData.get().copy()

                        if len(Data) > 0:
                            return ui.input_numeric("ECVisMaxSamplesSingle","Maximum Samples on Plot",len(Data),min=10,width='100%')

                    @render.ui()
                    def ECVisColDropdownSingle():
                        Data = OriginalFeatureData.get().copy()
                        Columns = sorted(list(Data.columns))
                        if 'Timestamp' in Columns:
                            Columns.remove('Timestamp')

                        return ui.input_select("ECVisColSingle","Column to Visualize on Plot",Columns,width='100%')
                    
                with ui.layout_column_wrap(width=1/2):
                    ui.input_switch('ECPlotOriginal','Plot Original Data', True, width='100%')

                    ui.input_switch('ECPlotCorrected','Plot Corrected Data', True, width='100%')
                
                @render.ui()
                def ECVisStartTimeSliderSingle():
                    Data = OriginalFeatureData.get().copy()
                    if 'Timestamp' in Data.columns:
                        ResRate = GetECVisResampleRates()['Computable'][GetECVisResampleRates()['Readable'].index(input.ECVisResRateSingle())]

                        Resampled = OriginalFeatureData.get().copy().resample(ResRate,on='Timestamp').mean()

                        if len(Resampled) > input.ECVisMaxSamplesSingle():
                            return ui.input_slider("ECVisStartSingle","Starting Index of Plot",0,len(Resampled)-input.ECVisMaxSamplesSingle(),0,
                                                step=input.ECVisMaxSamplesSingle()//6,animate=True, width='100%')
                
                @render.plot(alt="No Column Selected")
                def ECPlotSingle():
                    if input.ECPlotOriginal() or input.ECPlotCorrected():
                        Original = OriginalFeatureData.get().copy()
                        Corrected = ErrorCorrectedData.get().copy()

                        if len(Original) > 0 and len(Corrected) > 0:
                            ResRate = GetECVisResampleRates()['Computable'][GetECVisResampleRates()['Readable'].index(input.ECVisResRateSingle())]

                            OriginalResampled = Original.copy().resample(ResRate,on='Timestamp').mean()
                            CorrectedResampled = Corrected.copy().resample(ResRate,on='Timestamp').mean()

                            if len(OriginalResampled) > input.ECVisMaxSamplesSingle() and len(CorrectedResampled) > input.ECVisMaxSamplesSingle():
                                OriginalResampled = OriginalResampled[OriginalResampled.index[input.ECVisStartSingle()]: OriginalResampled.index[input.ECVisStartSingle() + input.ECVisMaxSamplesSingle() - 1]][input.ECVisColSingle()]
                                CorrectedResampled = CorrectedResampled[CorrectedResampled.index[input.ECVisStartSingle()]: CorrectedResampled.index[input.ECVisStartSingle() + input.ECVisMaxSamplesSingle() - 1]][input.ECVisColSingle()]
                            else:
                                OriginalResampled = OriginalResampled[input.ECVisColSingle()]
                                CorrectedResampled = CorrectedResampled[input.ECVisColSingle()]

                            fig, ax = plt.subplots()
                            if input.ECPlotOriginal():
                                ax.plot(OriginalResampled, label='Observed Data', color='C0', alpha=0.7)
                                if input.ECVisMaxSamplesSingle() < 300:
                                    ax.scatter(OriginalResampled.index, OriginalResampled, color='C0', alpha=0.7)
                            if input.ECPlotCorrected():
                                ax.plot(CorrectedResampled, label='Corrected Data', color='C1', alpha=0.7)
                                if input.ECVisMaxSamplesSingle() < 300:
                                    ax.scatter(CorrectedResampled.index, CorrectedResampled, color='C1', alpha=0.7)
                            ax.set_title(input.ECVisColSingle())
                            ax.set_xlabel("Timestamp")
                            ax.set_ylabel("Value")
                            ax.legend(loc='upper right')

                            return fig