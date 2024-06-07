# Author: Aidan Martas

import os
import mysql.connector
from mysql.connector import connect, Error
import time
import CAIRD
import json
import requests
import CAIRDExceptions

outputdir = "/dark/amart04/CAIRD/CAIRDWorkingFiles/"

# testbot
webhook = "https://hooks.slack.com/services/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"
ClassificationNames = {
                        1: "Eyeball",
                        2: "Bad subtraction",
                        3: "Bad subtraction on bright star" ,
                        4: "Dipole",
                        5: "Real transient",
                        6: "Moving object",
                        7: "Edge of chip",
                        8: "Artifact",
                        9: "Active galactic nucleus",
                        10: "Artifical star",
                        11: "SN",
                        12: "Cataclysmic variable",
                        13: "Rings",
                        14: "Variable star",
                        15: "Limit",
                        16: "Kilonova",
                        17: "LBV",
                        18: "Nova",
                        19: "Satellite",
                        20: "Pointing jump"
                        }


def SlackMessage(payload, webhook):
    """Send a Slack message to a channel via a webhook. 
    
    Args:
        payload (dict): Dictionary containing Slack message, i.e. {"text": "This is a test"}
        webhook (str): Full Slack webhook URL for your chosen channel. 
    
    Returns:
        HTTP response code, i.e. <Response [503]>
    """
    return requests.post(webhook, json.dumps(payload))

def MessageFormatter(text):
    payload = {
        "blocks": [
                {
                "type": "section",
                "text":{
                    "type":"mrkdwn",
                    "text": text
                }
            }
        ]
    }
    return payload

def QueryData():
    try:
        with connect(
                host = "localhost",
                user = "dlt40",
                password = "dlt40",
                database="dlt40") as connection:
            print(connection, "QueryData")
            GetData = "SELECT id, classificationid, targetid, filepath, filename, xpos, ypos, ra0, dec0, fwhm, ellipticity, bkg, fluxmax, fluxrad FROM candidates ORDER BY id DESC LIMIT 1"
            with connection.cursor() as cursor:
                cursor.execute(GetData)
                DataRow = cursor.fetchall()
                return DataRow
    except Error as e:
        print(e)

def QueryRow(CandID):
    try:
        with connect(
                host = "localhost",
                user = "dlt40",
                password = "dlt40",
                database="dlt40") as connection:
            print(connection, "QueryRow")
            GetData = "SELECT id, classificationid, targetid, filepath, filename, xpos, ypos, ra0, dec0, fwhm, ellipticity, bkg, fluxmax, fluxrad FROM candidates WHERE id=" + str(CandID)
            with connection.cursor() as cursor:
                cursor.execute(GetData)
                DataRow = cursor.fetchall()
                return DataRow
    except Error as e:
        print(e)

def QueryLastCandID():
    try:
        with connect(
                host = "localhost",
                user = "dlt40",
                password = "dlt40",
                database="dlt40") as connection:
            print(connection, "QueryLastCandID")
            GetData = "SELECT id FROM candidates ORDER BY id DESC LIMIT 1"
            with connection.cursor() as cursor:
                cursor.execute(GetData)
                DataRow = cursor.fetchall()
                return DataRow
    except Error as e:
        print(e)

def UpdateTable(CAIRDClassification, CandID, CID, TID, filedir, filename, xpos, ypos, RA, DEC, fwhm, ellipticity, fluxmax, fluxrad, reviewedby):
    try:
        with connect(
                host = "localhost",
                user = "dlt40",
                password = "dlt40",
                database="dlt40") as connection:
            print(connection)
            WriteData = """
                        INSERT INTO CAIRDCandidates (candidateid, targetid, directory, filename, xpos, ypos, rfclass, cairdclass, ra0 ,dec0, fluxrad, ellipticity, fwhm, fluxmax, reviewedby)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """
            
            RowInfo = (CandID, TID, filedir, filename, xpos, ypos, ClassificationNames[CID], CAIRDClassification, RA, DEC, fluxrad, ellipticity, fwhm, fluxmax, reviewedby)

            with connection.cursor() as cursor:
                cursor.execute(WriteData, RowInfo)
                connection.commit()
                print("Added candidate " + str(CID) + " to the classification table as a: " + ClassificationNames[CID])
    except Error as e:
        print("ERROR OCCURRED")
        print(e)

def EvalImg(DataArr):
    
    CandID, CID, TID, filedir, filename, xpos, ypos, RA, DEC, fwhm, ellipticity, bkg, fluxmax, fluxrad = DataArr[0]
    print(DataArr[0])
    print(DataArr[0][0], "CandID")




    if CID != 1:
        print("Skipping classification - image is a: " + ClassificationNames[CID])
        return
    filepath = os.path.join(filedir, filename)
    
    filepath = filepath[:-9]
    
    scipath = filepath + "clean.fits"
    refpath = filepath + "ref.fits"
    diffpath = filepath + "diff.fits"

    print("Classifying...")
    try:
        return CAIRD.ClassifyImage(scipath, refpath, diffpath, xpos, ypos, CID, TID, RA, DEC, fluxrad, ellipticity, fwhm, bkg, fluxmax), CandID, TID
    except CAIRDExceptions.CorruptImage:
        print("Invalid image - skipping")
        return None



x = True
UserID = "U06B6N2LGQ6"

PrevCand = None
CandID = 0
while True == True: # RUNS FOREVER
    MaxCandID = QueryLastCandID()[0][0]
    print(MaxCandID, "MaxCandID")
    if CandID == 0:
        CandID = MaxCandID
        print("Initialized CandID", CandID)

    if MaxCandID < CandID:
        print("Out of images to classify")
        time.sleep(1)
        continue
    print(CandID, "CandID EARLY")
    RowQuery = QueryRow(CandID)
    print(RowQuery, "ROW QUERY ROW QUERY ROW QUERY")
    if RowQuery == None:
        print("Image not marked for classification - skipping")
        CandID += 1
        time.sleep(1)
        continue
    elif EvalImg(RowQuery) == None:
        print("Image not marked for classification - skipping")
        CandID += 1
        time.sleep(1)
        continue
    else:
        DataArr = QueryRow(CandID)
        Classification, CandID, TID = EvalImg(DataArr)
        print(CandID, "Post-CandID")
        CandID, CID, TID, filedir, filename, xpos, ypos, RA, DEC, fwhm, ellipticity, bkg, fluxmax, fluxrad = DataArr[0]
        print(CandID, "PostCandID2")
        print(Classification)
        print(type(Classification))

        if type(Classification) != dict:
            message = f"<@{UserID}> " + "Classification failed on candidate.\n Target Link: https://dark.physics.ucdavis.edu/dlt40/view?object=" + str(TID)
            payload = MessageFormatter(message)
            #SlackMessage(payload, webhook)
            CandID += 1
            continue
        else:
            key = max(Classification, key=Classification.get)
            if key == "SN":
                message = f"<@{UserID}> " + key + " detected with " + str(Classification[key]*100)[:-12] + "%" + " confidence.\n Target Link: https://dark.physics.ucdavis.edu/dlt40/view?object=" + str(TID)
            else:
                message = key + " detected with " + str(Classification[key]*100)[:-12] + "%" + " confidence.\n Target Link: https://dark.physics.ucdavis.edu/dlt40/view?object=" + str(TID)
            payload = MessageFormatter(message)
            #SlackMessage(payload, webhook)
            CID = DataArr[0][1]
            UpdateTable(key, CandID, CID, TID, filedir, filename, xpos, ypos, RA, DEC, fwhm, ellipticity, fluxmax, fluxrad, "CAIRD")
            CandID += 1
            pass
