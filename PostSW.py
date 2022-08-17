# -*- coding: utf-8 -*- 
""" 
Created on Tue Mar  1 16:46:11 2022 
 
@author: Lyle 
""" 
 
import pandas as pd 
import numpy as np 
from tqdm.autonotebook import tqdm, trange
 
 
def extract_data_batch(files, allindices, batchnames, interval): 
#Extract the data from a collection of simulation outputs 
     
    #Loop over all files in the batch 
    for i in tqdm(range(len(files)), desc='Extracting data from files'): 
        if allindices[0]!=None:
            #Set correct indices for this file 
            if (i+1)*interval<=len(allindices): 
                indices=list(allindices[range(interval*i, interval*(i+1))]) 
            else: 
                indices=list(allindices[range(interval*i, len(allindices))]) 
            #Extract data, feeding in file and indices to label the dataframe rows 
            data=extract_data(files[i], indices) 
        else:
            data=extract_data(files[i], [None]) 
        #Correct radians vs degrees 
        data=correctRads(data) 
         
        #Set batch 
        data.loc[:, "batch"]=batchnames[i] 
         
        #Concatenate to current collection of all data 
        if i==0: 
            alldata=data 
        else: 
            alldata=pd.concat([alldata, data])  
    return alldata 
 
def extract_data(filepath, indices, filtered=True, matrow=27): 
#Extract the data from a single simulation output 
    # print(filepath)
    data=pd.read_csv(filepath, skiprows=[0,1,2,matrow], index_col=0) 
     
    paramlen=len(data.index) 
    data.drop(data.columns[0:2], axis=1, inplace=True) 
     
    #Remove extra rows caused by excel saving csv files with trailing commas 
    for i in range(len(data.columns)): 
        if not str(data.columns[i]).startswith("Unnamed"): 
            max_nonempty_col=i 
    data=data[data.columns[:max_nonempty_col+1]] 
    #If we are indices is none, just set indices in order
    if indices[0]==None:
        indices=list(range(1, len(data.columns)))
    data.columns=["drop"]+indices 
    data.drop(["drop"], axis=1, inplace=True) 
     
    skips=list(range(paramlen+6)) 
    skips.remove(matrow) 
    materials=pd.read_csv(filepath, skiprows=skips, index_col=0) 
    materials=list(materials.columns)[2:] 
    materials = [x for x in materials if not x.startswith(' annealed') or x.startswith(' normalized') ] 
    materials=materials[1:] 
    data=data.transpose() 
    data.loc[:, "Material"]=materials 
    data.columns.name = None 
    rearranged= [data.columns[-1]] + list(data.columns[:-1]) 
    data=data[rearranged] 
    if filtered==True: #Remove non-floats
        data=data[data.iloc[:,-12:].applymap(lambda x: type(x)==float).all(1)] 
    colorder=list(data.columns[:-12])+[ 
        "Dropout X Displacement", 
        "Dropout Y Displacement", 
        "BB X Displacement", 
        "BB Y Displacement", 
        "BB Z Displacement", 
        "BBR Y Displacement", 
        "BBL Y Displacement", 
        "Min FS 1", 
        "Max Stress 1", 
        "Min FS 3", 
        "Max Stress 3", 
        "Mass" 
        ] 
    data=data[colorder] 
    return data 
 
 
def calcQOA(alldata): 
#Calculate the specific quantities of interest from the simulated quantities 
 
    params=alldata.iloc[:,:-13] 
    batch=alldata.iloc[:,-1]
    data=alldata.iloc[:,-13:-1] 
    # print(params)
    # print(data)
    # print(batch)
    names=["Sim 1 Dropout X Disp.", "Sim 1 Dropout Y Disp.", "Sim 1 Bottom Bracket X Disp.", 
       "Sim 1 Bottom Bracket Y Disp.", "Sim 2 Bottom Bracket Z Disp.", "Sim 3 Bottom Bracket R Disp.", 
       "Sim 3 Bottom Bracket L Disp.", "Sim 1 Safety Factor", "Sim 1 Maximum Stress", "Sim 3 Safety Factor", 
       "Sim 3 Maximum Stress", "Model Mass"] 
    data.columns=names 
#     print(data) 
#     print(data["Sim 3 Bottom Bracket L Disp."]) 
    data.loc[:, "Sim 3 Bottom Bracket Y Disp."]=data["Sim 3 Bottom Bracket L Disp."] #Just copy this now for simplicity 
    data.loc[:, "Sim 3 Bottom Bracket X Rot."]=data["Sim 3 Bottom Bracket L Disp."] 
    for i in tqdm(range(len(data.index)), desc='Calculating Quanitities of Interest from Sim Output'): 
        if type(data["Sim 3 Bottom Bracket Y Disp."].iloc[i])==float or type(data["Sim 3 Bottom Bracket Y Disp."].iloc[i])==np.float64: 
            data.loc[:, "Sim 3 Bottom Bracket Y Disp."].iloc[i]=(data.loc[:,"Sim 3 Bottom Bracket L Disp."].iloc[i]+data.loc[:,"Sim 3 Bottom Bracket R Disp."].iloc[i])/2 
        if type(data["Sim 3 Bottom Bracket X Rot."].iloc[i])==float or type(data["Sim 3 Bottom Bracket X Rot."].iloc[i])==np.float64: 
            data.loc[:, "Sim 3 Bottom Bracket X Rot."].iloc[i]=(data.loc[:,"Sim 3 Bottom Bracket L Disp."].iloc[i]-data.loc[:,"Sim 3 Bottom Bracket R Disp."].iloc[i])/.14 
    data.drop(["Sim 3 Bottom Bracket R Disp.", "Sim 3 Bottom Bracket L Disp.", "Sim 1 Maximum Stress","Sim 3 Maximum Stress"], axis=1, inplace=True) 
    order=["Sim 1 Dropout X Disp.", "Sim 1 Dropout Y Disp.", "Sim 1 Bottom Bracket X Disp.", 
       "Sim 1 Bottom Bracket Y Disp.", "Sim 2 Bottom Bracket Z Disp.", "Sim 3 Bottom Bracket Y Disp.",  
        "Sim 3 Bottom Bracket X Rot.", "Sim 1 Safety Factor", "Sim 3 Safety Factor", "Model Mass"] 
    data=pd.concat([params, data[order], batch], axis=1) 
    return data 
 
def renameMats(data): 
    #Rename materials from SW-readable materials to simple names 
    names=[] 
    for value in list(data["Material"]): 
        if value.startswith("AISI") or value.startswith(" AISI") : 
            names.append("Steel") 
        elif value.startswith("Ti") or value.startswith(" Ti"): 
            names.append("Titanium") 
        elif value.startswith("6061") or value.startswith(" 6061"): 
            names.append("Aluminum") 
#         elif value.startswith(" AISI"): 
#             print("check") 
#             names.append("Steel") 
        else: 
            raise Exception("Unknown Material!") 
    data.loc[:, "Material"]=names 
    return data 
 
def correctRads(data): 
    #If we determine angle data was reported as radians, change to degrees: 
    #(SW sometimes saves as rad and sometimes as deg) 
    if np.mean(data["ST Angle"])<10: 
        data.loc[:, "ST Angle"]=data["ST Angle"]*180/np.pi 
    if np.mean(data["HT Angle"])<10: 
        data.loc[:, "HT Angle"]=data["ST Angle"]*180/np.pi 
    return data 
 
def setBridges(data): 
    data=data.copy()
    columns=["SSB_Include", "CSB_Include"]+ list(data.columns) 
    #Multiply by 1 to convert to numerical instead of bool 
    data["SSB_Include"]=(data.loc[:, "SSB OD"]!=0)*1 
    data["CSB_Include"]=(data.loc[:, "CSB OD"]!=0)*1 
    data = data[columns] #Reorder columns
    #Get average nonzero value of dataframe 
#     print(data.loc[data.index[data.loc[:, "SSB OD"]!=0], "SSB OD"]) 
    ssbmean = np.mean(data.loc[data.index[data.loc[:, "SSB OD"]!=0], "SSB OD"]) 
    csbmean = np.mean(data.loc[data.index[data.loc[:, "CSB OD"]!=0], "CSB OD"]) 
    data.loc[:, "SSB OD"]=ssbmean*(1-data.loc[:, "SSB_Include"])+data.loc[:, "SSB OD"]*data.loc[:, "SSB_Include"] 
    data.loc[:, "CSB OD"]=csbmean*(1-data.loc[:, "CSB_Include"])+data.loc[:, "CSB OD"]*data.loc[:, "CSB_Include"] 
    return data 
 
 