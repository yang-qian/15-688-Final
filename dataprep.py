import pandas as pd
import numpy as np
import csv
import functools
import regex as re

# combine device, patient, and followup csvs into one
# and then separate pre- and post-implant variables
# @input:
# - patient_INTERMACS_Data_Dictionary.csv
# - device_INTERMACS_Data_Dictionary.csv
# - followup_INTERMACS_Data_Dictionary.csv
# @output
# - all (pickled df): three csvs joined
# - pre(pickled df): pre-implant-related columns, match with ouput on 'OPER_ID'
# - post (pickled df): post-implant-related columns, match with input on 'OPER_ID'

blood_type_dict = {
    1: 'O',
    2: 'A',
    3: 'B',
    4: 'AB',
    998: '-',
}

nyha_dict = {
    'Class I: No Limitation or Symptoms': 'ClassI',
    'Class II: Slight Limitation': 'ClassII',
    'Class III: Marked Limitation': 'ClassIII',
    'Class IV: Unable to Carry on Minimal Physical Activity': 'ClassIV',
    'Unknown': '-',
}

num_card_hosp_dict = {'0-1': 0, '2-3': 2, '4 or more': 4, '-':  None }

time_card_dgn_dict = {
    '< 1 month': 0,
    '1 month - 1 year': 1,
    '1-2 years': 12,
    '> 2 years': 24,
    '-': None,
}


def feature_dict():
    """
    imports INTERMACS Data Dictionary with Format Value tables into one dictionary
    :returns:
    - a dictionary that combines all three data dictinoaries
    """

    # import data dictionary csv as dict
    # :return: {'VARIABLE': {'TABLE': csv_name, 'TYPE': ..., 'FORMAT_VALUE', ....}}

    p = pd.read_csv('data/patient_INTERMACS_Data_Dictionary.csv',
                    index_col='VARIABLE', encoding="ISO-8859-1", low_memory=False).to_dict(orient='index')
    d = pd.read_csv('data/device_INTERMACS_Data_Dictionary.csv',
                    index_col='VARIABLE', encoding="ISO-8859-1", low_memory=False).to_dict(orient='index')
    f = pd.read_csv('data/followup_INTERMACS_Data_Dictionary.csv',
                    index_col='VARIABLE', encoding="ISO-8859-1", low_memory=False).to_dict(orient='index')

    # print (len(d.keys() | p.keys() | e.keys() | f.keys()))

    _dict = {**d, **p, **f}  # all columns 1080 columns in total

    # parse FORMAT_VALUE into dictionary
    # i.e. transform string ".=Missing; .U=Unknown; 0=No; 1=Yes; 9=Unknown"
    # to dict {'.': 'Missing'; '.U'= 'Unknown',  0=No; 1=Yes; 9 = 'Unknown'}

    for k, v in _dict.items():

        # if FORMAT_VALUE has value
        if isinstance(v['FORMAT_VALUE'], str):

            # transform string to list of tuples
            _v = [tuple(item.split('=')) for item in v['FORMAT_VALUE'].split('; ')]
            
            # print(dict(re.findall(r'([^ ]*?)=([a-zA-Z\s]+)[^;]*?', v['FORMAT_VALUE'])))
            
            # re.findall 
            v['FORMAT_VALUE'] = dict(_v)

            # print(k)
            # print(_dict[k]['FORMAT_VALUE'])


    return _dict


def import_and_join(feature_dict):
    """
    import three datasets (device, patient, events) and join on "OPER_ID"
    """
    # load csvs
    # leave only pre-implant rows in follow-up data
    pdf = pd.read_csv('data/patientnewdata.csv',
    na_values = ['nan'],
    low_memory = False,
    encoding="ISO-8859-1")  # 456 cols 17075 rows (patients)
    ddf = pd.read_csv('data/devicenewdata.csv',
    na_values = ['nan'],
    low_memory = False,
    encoding = "ISO-8859-1")  # 19207 rows (implants)
    fdf = pd.read_csv('data/followupnewdata.csv', 
                      na_values=['nan'],
                      low_memory=False, encoding="ISO-8859-1")
    fdf = fdf.loc[fdf['FORM_ID'] == 'Pre-Implant']  # 19207 rows (implants)

    # merge device and followup tables
    df = ddf.merge(fdf[list(fdf.columns.difference(ddf.columns)) +
                       ['OPER_ID']], on='OPER_ID', how='left', suffixes=['', ''])

    # append patient table to the joined table
    df = df.merge(pdf[list(pdf.columns.difference(df.columns)) + ['OPER_ID']],
                  on='OPER_ID', how='left', suffixes=['', ''])  # 1447 cols 19207rows (implants)

    """
    filter out patient_IDs without death records
    """
    # 4989 rows are death records
    # == 4989 patients have death records in the df
    dead_pts = df[df['DEAD'] == 1.0]['PATIENT_ID'].unique()
    df = df[df['PATIENT_ID'].isin(dead_pts)]  # 5820 rows left

    # convert nominal data type/ coding
    df['BLOOD_TYPE'].replace(blood_type_dict, inplace=True)
    df['TIME_CARD_DGN'].replace(time_card_dgn_dict, inplace=True)
    df['NUM_CARD_HOSP'].replace(num_card_hosp_dict, inplace=True)
    df['NYHA'].replace(nyha_dict, inplace=True)

    """
    drop columns with no value
    """
    # print("before dropna", len(df.columns)) #806
    df = df.dropna(axis=1, how='all')
    # print("after dropna", len(df.columns)) #653

    """
    seperate columns about pre-implant conditions and post-implant outcomes
    """

    idx_col = ['OPER_ID', 'PATIENT_ID', 'FORM_ID']

    # elicit outcome cols from each table 
    # starting with device table
    # - VERSION: intermacs version
    # - DEAD/DEATH_ are death related outcome, EXPL_ explant-related, 
    # - INT_ interval btw implant and result time
    # - LOS: length of stay from implant to discharge
    # - DIS_INT are intervention since implant
    # - QRTR: quarter of implant
    outcome_col = ['VERSION',
                    'DEAD',
                    'DEATH_DEVICE_EXPLANT',
                    'DEATH_LOCATION',
                    'TXPL',
                    'SURGICAL_APPROACH',
                    'TIMING_DEATH',
                    'INT_DEAD',
                    'INT_EXPL',
                    'INT_TXPL',
                    'PRIMARY_COD',
                    'PRIMARY_COD_CANCER',
                    'EXPL',
                    'EXPLANT_DEVICE_TY',
                    'EXPLANT_REASON',
                    'EXPL_THROM',
                    'LOS',
                    'DISCHARGE_TO',
                    'DISCHARGE_STATUS',
                    'DIS_INT_AVS_REPAIR_NC',
                    'DIS_INT_AVS_REPAIR_WC',
                    'DIS_INT_AVS_REPLACE_BIO',
                    'DIS_INT_AVS_REPLACE_MECH',
                    'DIS_INT_BLEED_GT_48',
                    'DIS_INT_BLEED_LE_48',
                    'DIS_INT_BRONCHOSCOPY',
                    'DIS_INT_CARD_OTHER',
                    'DIS_INT_CARD_UNKNOWN',
                    'DIS_INT_DIALYSIS',
                    'DIS_INT_DRAINAGE',
                    'DIS_INT_INV_CARD_PROC',
                    'DIS_INT_MVS_REPAIR',
                    'DIS_INT_MVS_REPLACE_BIO',
                    'DIS_INT_MVS_REPLACE_MECH',
                    'DIS_INT_NONE',
                    'DIS_INT_OTHER',
                    'DIS_INT_PVS_REPAIR',
                    'DIS_INT_PVS_REPLACE_BIO',
                    'DIS_INT_PVS_REPLACE_MECH',
                    'DIS_INT_REINTUBATION',
                    'DIS_INT_SURG_PROC_DEV',
                    'DIS_INT_SURG_PROC_NC',
                    'DIS_INT_SURG_PROC_OTHER',
                    'DIS_INT_SURG_PROC_UNKNOWN',
                    'DIS_INT_TRANSPLANT',
                    'DIS_INT_TVS_REPAIR_DEVEGA',
                    'DIS_INT_TVS_REPAIR_OTHER',
                    'DIS_INT_TVS_REPAIR_RING',
                    'DIS_INT_TVS_REPLACE_BIO',
                    'DIS_INT_TVS_REPLACE_MECH',
                    'DIS_INT_UNKNOWN'] 

    # outcome cols from events in addition to the DEATH, EXPLANT, DIS_INT related one
    # - OP outcomes of each operation
    # - PC_PUMP_EXCHANGE pump exchange
    outcome_col += ['OP',
                    'OP1COD',
                    'OP1CONT',
                    'OP1DEAD',
                    'OP1DEV_TY',
                    'OP1EVTID',
                    'OP1EXPDEV',
                    'OP1EXPL',
                    'OP1EXPREA',
                    'OP1INTD',
                    'OP1INTR',
                    'OP1INTT',
                    'OP1REC',
                    'OP1TXPL',
                    'OP2COD',
                    'OP2CONT',
                    'OP2DEAD',
                    'OP2DEV_TY',
                    'OP2EVTID',
                    'OP2EXPDEV',
                    'OP2EXPL',
                    'OP2EXPREA',
                    'OP2INTD',
                    'OP2INTR',
                    'OP2INTT',
                    'OP2REC',
                    'OP2TXPL',
                    'OP3COD',
                    'OP3CONT',
                    'OP3DEAD',
                    'OP3DEV_TY',
                    'OP3EVTID',
                    'OP3EXPDEV',
                    'OP3EXPL',
                    'OP3EXPREA',
                    'OP3INTD',
                    'OP3INTR',
                    'OP3INTT',
                    'OP3REC',
                    'OP3TXPL',
                    'OP4COD',
                    'OP4CONT',
                    'OP4DEAD',
                    'OP4DEV_TY',
                    'OP4EVTID',
                    'OP4EXPDEV',
                    'OP4EXPL',
                    'OP4EXPREA',
                    'OP4INTD',
                    'OP4INTR',
                    'OP4INTT',
                    'OP4REC',
                    'OP4TXPL',
                    'OP5COD',
                    'OP5CONT',
                    'OP5DEAD',
                    'OP5DEV_TY',
                    'OP5EVTID',
                    'OP5EXPDEV',
                    'OP5EXPL',
                    'OP5EXPREA',
                    'OP5INTD',
                    'OP5INTR',
                    'OP5INTT',
                    'OP5REC',
                    'OP5TXPL',
                    'OP6COD',
                    'OP6CONT',
                    'OP6DEAD',
                    'OP6DEV_TY',
                    'OP6EVTID',
                    'OP6EXPDEV',
                    'OP6EXPL',
                    'OP6EXPREA',
                    'OP6INTD',
                    'OP6INTR',
                    'OP6INTT',
                    'OP6REC',
                    'OP6TXPL',
                    'OP7COD',
                    'OP7CONT',
                    'OP7DEAD',
                    'OP7DEV_TY',
                    'OP7EVTID',
                    'OP7EXPDEV',
                    'OP7EXPL',
                    'OP7EXPREA',
                    'OP7INTD',
                    'OP7INTR',
                    'OP7INTT',
                    'OP7REC',
                    'OP7TXPL',
                    'OP8COD',
                    'OP8CONT',
                    'OP8DEAD',
                    'OP8DEV_TY',
                    'OP8EVTID',
                    'OP8EXPDEV',
                    'OP8EXPL',
                    'OP8EXPREA',
                    'OP8INTD',
                    'OP8INTR',
                    'OP8INTT',
                    'OP8REC',
                    'OP8TXPL']

    # patient csv
    # - KCCQ, LIFE, LIFESTYLE: quality of life measures
    # - OUTCOME
    # - EXCH_NEW_STUDY: excharged
    # - PC_PUMP pump change
    outcome_col += ['INT_AFOL',
                    'INT_FOL',
                    'KCCQ12',
                    'KCCQ12PL',
                    'KCCQ12QL',
                    'KCCQ12SF',
                    'KCCQ12SL',
                    'KCCQ_PARENT_QUESTION',
                    'LIFE',
                    'LIFESTYLE_CHORES',
                    'LIFESTYLE_HOBBIES',
                    'LIFESTYLE_VISITING',
                    'OUTCOME',
                    'OUTCOME_I',
                    'PC_PUMP_EXCHANGE',
                    'PC_PUMP_EXCHANGE_REASON'
                    ]

    # among these outcome cols, exclude the columns have been removed when dropna
    # this reduced outcomes from 172 cols >> 118cols
    outcome_col = [col for col in outcome_col if col in df.columns.tolist()]

    # the rest of the columns are pre-implant variables eligible to be features
    input_col = list(set(df.columns.get_values()) -
                     set(outcome_col) - set(idx_col))

    """
    export the whole df, preimplant and outcome dfs to respective csvs
    """
    df.to_csv('data/03-14-2018-all.csv', encoding='utf-8')
    # excludes PATIENT_ID and FORM_ID
    df[outcome_col + ['OPER_ID']].to_csv('data/03-14-2018-post.csv', encoding='utf-8')
    # excludes PATIENT_ID and FORM_ID
    df[input_col + ['OPER_ID']].to_csv('data/03-14-2018-pre.csv', encoding='utf-8')

    """
    prepare data for modeling by combine input cols and label 'INT_DEAD'
    """
    traintest = df[input_col + ['INT_DEAD']]
    traintest.to_csv('data/03-14-2018-traintest.csv', encoding='utf-8')
    print('Integrated and imported %d input columns, %d output columns and %d observations.' %(len(input_col),len(outcome_col),len(df))) #532 col 5821 rows

    # take all columns of df as object (str) columns at the moment
    df[df.columns] = df[df.columns].astype(str) 

    return df, input_col, outcome_col, idx_col

# feature_dict = feature_dict()
#df = import_and_join(feature_dict)


