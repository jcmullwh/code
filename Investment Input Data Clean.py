# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 20:56:31 2021

@author: Jason Mullins
"""

import re
import pandas as pd



def readFile():
    
    global location
    
    f = open(location, "r")
    
    content = f.read()
    
    f.close()
    
    
    return content


location = 'File Location'

entire = readFile()

entire_list = entire.split()

# Function to make list of lists starting with each date
def findDates(entire_list):

    pattern = "\d{2}[/]\d{2}[/]\d{2}"

    dateStartingList = []
    workingInfo = []
    
    for x in entire_list:
        dateCheck = re.match(pattern, x)
        if dateCheck == None:
            workingInfo.append(x)
        else:
            dateStartingList.append(workingInfo)
            workingInfo = [x]
    return dateStartingList

def delBlanks(dateStartingList):
    
    tracker = 0 #For some reason I had to do this repeatedly; and now for some reason I don't have any blanks at all
    repeater = 0
    workCheck = 1
    
    while workCheck:
        for x in dateStartingList:
            if dateStartingList[tracker] == []:
                repeater += 1
                del dateStartingList[tracker]
            tracker += 1
            if repeater == 0:
                workCheck = 0
    return(dateStartingList)

# Function to delete non-transaction lists within list. Will have to customize for each run

def delNonTrans(dateStartingList):

    workCheck = 1
    while workCheck:
        counter = 0
        doneCheck = 0
        for x in dateStartingList:
            if x[0] == 'ENV#' or x[0] == '12/31/15' or x[0] == '12/31/15)' or x[0] == '01/01/15' or x[0] == '01/01/15)':
                del dateStartingList[counter]
                doneCheck = 1
            counter += 1
        if doneCheck == 0:
            workCheck = 0
    return dateStartingList

# Function to clean non-date characters and turn into date time data type

def cleanDate(dateStartingList):

    pattern = "\d{2}[/]\d{2}[/]\d{2}"
    
    for x in dateStartingList:
        dateCheck = re.match(pattern, x[0])
        print(x[0])
        if dateCheck != None:
            date = str(x[0])
            date = date.strip(")")
            date = [pd.to_datetime(date, infer_datetime_format=True)]
            
    return dateStartingList

# Get rid of useless transactions. Will need to customize for each run

def removeHoldings(dateStartingList):
    workCheck = 1
    repeater = 0
    while workCheck:
        tracker = 0
        repeater = 0
        for x in dateStartingList:
            if len(x) > 1:
                print(x[1])
                if x[1] == 'CO':
                    repeater += 1
                    del dateStartingList[tracker]
            else:
                del dateStartingList[tracker]
                repeater += 1
            tracker += 1
            print(tracker)
            if repeater != 0:
                workCheck = 1
            else:
                workCheck = 0
            print(repeater)
            print(workCheck)
    return dateStartingList

# Function to get rid of redundant transactions and deposits

def delUselessTrans(dateStartingList):

    workCheck = 1
    while workCheck:
        tracker = 0
        repeater = 0
        for x in dateStartingList:
            for i in x:
                if i == "ELECTRONIC":
                    del dateStartingList[tracker]
                    repeater += 1
                if i == "MMKT":
                    del dateStartingList[tracker]
                    repeater += 1
            tracker += 1
            if repeater != 0:
                workCheck = 1
            else:
                workCheck = 0
    return(dateStartingList)

#Function to consolidate sections of each transaction

def consolSec(dateStartingList):

    stepper = 0
    workingList = []
    formattedList = []
    pattern = "\d{2}[/]\d{2}[/]\d{2}"

    for i in dateStartingList:
        for x in i:
            dateCheck = re.match(pattern, x)
            if dateCheck != None:
                workingList = [x,0,0]
                notBought = 0
                bought = 0
                stepper = 1
                afterDesc = 0
            if x == 'BOUGHT' or x == 'REINVESTMENT' or x == 'SOLD':
                workingList[1] = str(x)
                print(workingList[1])
                beginMarker = stepper
            if x == '@':
                if workingList[1] == 'REINVESTMENT' or workingList[1] == 'SOLD':
                    endMarker = stepper + 4
                    notBought = endMarker
                    workingList[2] = ' '.join(i[beginMarker:endMarker])
                    print(workingList[2])
                else:    
                    endMarker = stepper + 1
                    workingList[2] = ' '.join(i[beginMarker:endMarker])
                    bought = endMarker
            if workingList[2] != 0:
                afterDesc += 1
                if workingList[1] == 'REINVESTMENT' or workingList[1] == 'SOLD':
                    if afterDesc > 5:
                        workingList.append(x)
                else:
                    if afterDesc > 2:
                        workingList.append(x)
            stepper += 1
        formattedList.append(workingList)
        
    return formattedList


datesList = findDates(entire_list)

transactions = delNonTrans(datesList)

transactions = removeHoldings(transactions)

#[n for n in transactions2014 if not 'ELECTRONIC' in n and not 'MMKT' in n]

equityOnly = delUselessTrans(transactions)

formattedList = consolSec(equityOnly)

formattedList_7 = [n[0:7] for n in formattedList]

headers = ['date', 'transaction', 'description', 'quantity', 'amount', 'cost basis', 'realized']

df = pd.DataFrame(formattedList_7,columns = headers)

df['short description'] = df['description'].str.strip('Securities Purchased=Securities Purchased ')

df['description'] = df['short description']

df = df[['date', 'transaction', 'description', 'quantity', 'amount', 'cost basis', 'realized']]

df.drop(range(21,25))

