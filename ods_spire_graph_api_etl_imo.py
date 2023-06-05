import multiprocessing
import numpy as np
from connect_db import connect_db
import sys

from loguru import logger
from notifier_messenger import logger_samo
from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter, Retry
from multiprocessing import Pool, Process
import time
import pandas as pd
from datetime import timedelta, time as dttime, datetime
from sqlalchemy import create_engine
import sqlalchemy
import warnings
warnings.filterwarnings('ignore')
# https://connect.ihsmarkit.com/gta/standard-reports
# sharedapi W5xk8953
# run every hour, so, use time to seperate your tasks !
def query_():
  return """
  query(
    $imos: [IMO!]
  ){
  vessels(
    first: 1000
    imo: $imos
  ) {
    pageInfo {
      hasNextPage
      endCursor
    }
    nodes {
      id
      updateTimestamp
      staticData {
        aisClass
        flag
        name
        callsign
        timestamp
        updateTimestamp
        shipType
        shipSubType
        mmsi
        imo
        callsign
        dimensions {
          a
          b
          c
          d
          width
          length
        }
      }
      lastPositionUpdate {
        accuracy
        collectionType
        course
        heading
        latitude
        longitude
        maneuver
        navigationalStatus
        rot
        speed
        timestamp
        updateTimestamp
      }
      currentVoyage {
        destination
        draught
        eta
        timestamp
        updateTimestamp
        matchedPort {
          matchScore
          port {
            name
            unlocode
            centerPoint {
              latitude
              longitude
            }
          }
        }
      }
    }
  }
}
"""



@logger.catch
def run_query(ls):
    URL = "https://api.spire.com/graphql"
    print("query with ls of {len}".format(len = ls[0]))

    response = requests.post(
        url=URL,
        json={'query': query_(),  'variables': {'imos': ls}},
        headers={
            "Authorization": "Bearer 2QbTEVJmv2X0iO01NT0UupGdAgnH33qb",
            "Content-Type": "application/json"
        })

    if response.status_code == 200:
        print(200, "\n")
        result = response.json()
        a = pd.json_normalize(result["data"]["vessels"]["nodes"])
        return a
        # result_queue.put(a)
        # return 1
    else:
        raise Exception("Query failed to run by returning code of {}. {}".format(
            response.status_code, response.text))

# @logger.catch


# def mul_(x, result_queue = ''):
#     x[:] = x[::-1]
#     logger_samo('graphql_spire_for_vts').add_log_info(
#             'timesleep 1 logaaaa') 
#     time.sleep(3)
#     # shared_list = shared_list.append(x)
#     # result_queue.put(x)
#     return x


def split_list(ls, sizes):
  # check if the sum of sizes matches the length of ls
  assert sum(sizes) == len(ls), "The sum of sizes must equal the length of ls"
  # initialize an empty list to store the result
  result = []
  # initialize an index to track the position in ls
  index = 0
  # loop through each size in sizes
  for size in sizes:
    # append a sublist of ls from index to index + size to result
    result.append(ls[index:index + size])
    # update index by adding size
    index += size
  # return result
  return result
@logger.catch
def mutation_data(AllPositions): 
    AllPositions = AllPositions[["staticData.imo",
                                            "lastPositionUpdate.collectionType",
                                            "currentVoyage.draught",
                                            "lastPositionUpdate.latitude",
                                            "lastPositionUpdate.longitude",
                                            "lastPositionUpdate.heading",
                                            "lastPositionUpdate.speed",
                                            'lastPositionUpdate.timestamp',
                                            "updateTimestamp",
                                            "staticData.dimensions.length", "staticData.dimensions.width", "staticData.mmsi",
                                            "currentVoyage.destination",
                                            "currentVoyage.eta",
                                            "currentVoyage.matchedPort.matchScore",
                                            "currentVoyage.matchedPort.port.name",
                                            "currentVoyage.matchedPort.port.unlocode",
                                            "staticData.name", "lastPositionUpdate.navigationalStatus",
                                            "staticData.shipType", "currentVoyage.updateTimestamp"]]
    AllPositions.rename(columns={"staticData.imo": "ShipID", "lastPositionUpdate.collectionType": "AIS_Type",
                                    "currentVoyage.draught": "Draught", "lastPositionUpdate.latitude": "Latitude",
                                    "lastPositionUpdate.longitude": "Longitude",
                                    "currentVoyage.matchedPort.matchScore": "Match_Score",
                                    "lastPositionUpdate.heading": "Heading", "lastPositionUpdate.speed": "Speed",
                                    "lastPositionUpdate.timestamp": "MovementDatetime", "currentVoyage.destination": "Destination",
                                    "currentVoyage.eta": "ETA",
                                    "currentVoyage.matchedPort.port.name": "Destination_Clean",
                                    "currentVoyage.matchedPort.port.unlocode": "Destination_Locode",
                                    "staticData.name": "ShipName",
                                    "lastPositionUpdate.navigationalStatus": "MoveStatus",
                                    "staticData.shipType": "ship_type",
                                    "currentVoyage.updateTimestamp": "static_updated_at", "staticData.mmsi": "MMSI",
                                    "staticData.dimensions.length": "Length", "staticData.dimensions.width": "Width"},
                        inplace=True)

    AllPositions["MovementDatetime"] = pd.to_datetime(AllPositions["MovementDatetime"], format='%Y-%m-%dT%H:%M:%S',
                                                        errors='coerce')
    AllPositions["ETA"] = pd.to_datetime(AllPositions["ETA"], format='%Y-%m-%dT%H:%M:%S', errors='coerce')
    AllPositions["static_updated_at"] = pd.to_datetime(AllPositions["static_updated_at"],
                                                        format='%Y-%m-%dT%H:%M:%S', errors='coerce')
    
    AllPositions = AllPositions.dropna(subset=['ShipID'])
    AllPositions["ShipID"] = AllPositions["ShipID"].astype(int).astype(str)
    AllPositions = AllPositions[[
        'ShipID',
        'Latitude',
        'Longitude',
        'Speed',
        'MovementDatetime',
        'Heading' ,
        'Draught',
        'Destination',
        'Destination_Clean',
        'ETA',
        'MoveStatus',]]
    AllPositions.rename({"Destination_Clean":"DestinationTidied"}, axis=1, inplace=1)
    AllPositions['InsertDate']=datetime.today().strftime('%Y%m%d%H%M')
    AllPositions = AllPositions[AllPositions['ShipID'].astype(str).apply( len ) == 7]
    AllPositions = AllPositions[~AllPositions['Draught'].isna()]
    AllPositions = AllPositions.sort_values(['MovementDatetime'], ascending = False)
    AllPositions = AllPositions.groupby('ShipID', as_index = False).first()
    AllPositions['InsertDate'] = datetime.now()
    return AllPositions.drop_duplicates()
    

def get_imo_ls(path):
    """
    #* get imo list from tbl bugne with clarkson data
    #* join previous project ls inlocal
    concat and boom.
    """
    conn = create_engine("mssql+pyodbc://research:research@GEN-NT-SQL11\MATLAB:56094/{DB}?driver=SQL+Server+Native+Client+10.0".format(DB = 'FairplayFleetDB'))
    imo_sql = """ SELECT distinct X03_IMO_NUMBER
                FROM [FairplayFleetDB].[dbo].[Tbl_BUNGE]
                where 
                (Z11_MAIN_STATUS = 'F'
                or REMOVAL_DATE >= DATEADD(month, -2, GETDATE()))
                and X03_IMO_NUMBER is not null
                ;"""
                
    imo_ls = pd.read_sql(imo_sql, conn)
    imo_ls.columns = ['imo']
    imo_sql_mysql = """
        select imo from ods_imo_from_sqlserver_df
        """
    conn = create_engine("mysql+pymysql://root:BungeFreight@173.194.246.105/FairplayFleetDB")
    imo_ls_mysql = pd.read_sql(imo_sql_mysql, conn, )
    df = pd.concat([imo_ls_mysql, imo_ls], axis = 0).drop_duplicates()
    df.to_sql('ods_imo_from_sqlserver_df', con=conn, if_exists='replace', dtype={
            "imo": sqlalchemy.types.BigInteger(), 
        })
    ls = df['imo'].tolist()
    logger.info(f'len of imo ls {len(ls)} \n\n')
    return ls

@logger.catch
def insert_to_db(df):
    mode = 'prod'
    daily_history_mark = False

    if eleven_clock == 23:
        logger.info("\n its time hour 23. insert history table. \n")
        daily_history_mark = True


    if (mode == 'test') and (len(df) > 0):
        ods_table_name  = "ods_vesselposition_spire_shipid_dtl_hf"  # should be hf, hourly full table.
        dwd_table_name  = "dwd_vesselposition_spire_shipid_dtl_hf" 
        dwd_table_name2 = "dwd_vesselposition_spire_shipid_dtl_hist_di"

        """
        * layer: ods *
        * subject: vessel position  *
        * source sys: spire *
        * prime key: shipid *
        * table level: dtl *
        * granularity: hourly  *
        * if test: add _test to tail. *
        """
    elif (mode == 'prod') and (len(df) > 0):
        ods_table_name  = "ods_vesselposition_spire_shipid_dtl_hf"  # should be hf, hourly full table.
        dwd_table_name  = "vesselposition_last" 
        dwd_table_name2 = "vesselposition_dailyhistory"
    
    # raw write
    else:
        raise Exception("df length to 0 from graphql.")
    conn = create_engine("mysql+pymysql://root:BungeFreight@173.194.246.105/IHSVesselPositionDB")
    # k = conn.execute("""
    #     select count(0),day(movementdatetime)   from  IHSVesselPositionDB.vesselposition_dailyhistory 
    #             PARTITION (p7)
    #             where year(MovementDateTime) = 2023
    #             and month (movementdatetime) = 2
    #             and day(movementdatetime) >= 21
    #             group by day(movementdatetime)
    #             limit 20
    #     """)
     
    # pd.DataFrame(k)#.to_csv('./datadown/tmp.csv')

    # conn.execute("""
    #     delete from IHSVesselPositionDB.vesselposition_dailyhistory 
    #     partition(p7)
    #         where year(MovementDateTime) = 2023
    #             and month (movementdatetime) = 2
    #             and day(movementdatetime) in ( 23,24)
    #     """)

    # conn.execute("truncate table IHSVesselPositionDB.{a}".format(a = table_name))

    # df = pd.read_csv(path + 'dataup\\ods_vesselposition_spire_shipid_dtl_hf.csv',)

    #! ods
    df.to_sql(
        "{table_name}".format(table_name = ods_table_name)
        , conn
        , if_exists ="replace"
        , index = False
        , dtype = {
            "ShipID": sqlalchemy.types.BigInteger(),   #. bigint(20)
            "Latitude": sqlalchemy.types.Float(precision=5,asdecimal=True),   #. double DEFAULT
            "Longitude": sqlalchemy.types.Float(precision=5,asdecimal=True),   #. double DEFAULT
            "Speed": sqlalchemy.types.Float(precision=3,asdecimal=True),   #. double DEFAULT
            "MovementDatetime": sqlalchemy.types.DateTime(),   #. timestamp NULL
            "Heading": sqlalchemy.types.Float(precision=3,asdecimal=True),   #. double DEFAULT
            "Draught": sqlalchemy.types.Float(precision=3,asdecimal=True),   #. double DEFAULT
            "Destination": sqlalchemy.types.String(25),   #. text,
            "DestinationTidied": sqlalchemy.types.String(25),   #. text,
            "ETA": sqlalchemy.types.DateTime(),
            "MoveStatus": sqlalchemy.types.String(25),   #. text,
            "InsertDate": sqlalchemy.types.DateTime()  #. text,
        }
        )

    #! dwd
    #* RETRIEVE AND REPLACE
    df_online = pd.read_sql(
        "{table_name}".format(table_name = dwd_table_name)
        , conn
    )
    df_online['InsertDate'] = datetime.now()
    df.columns = ['ShipID', 'Latitude', 'Longitude', 'Speed', 'MovementDateTime',
       'Heading', 'Draught', 'Destination', 'DestinationTidied', 'ETA',
       'MoveStatus', 'InsertDate']
    df_online[['MovementDateTime', 'ETA']] = df_online[['MovementDateTime', 'ETA']].astype('datetime64[ns]')
    df[['MovementDateTime', 'ETA']] = df[['MovementDateTime', 'ETA']].astype('datetime64[ns]')
    df['ShipID'] = df['ShipID'].astype(int).astype(str)
    df_online['ShipID'] = df_online['ShipID'].astype(int).astype(str)
    df = pd.concat(
        [df, df_online], axis = 0, ignore_index=True
        ).sort_values(
            'MovementDateTime', 
            ascending=False)
    df = df.drop_duplicates(subset=['ShipID'],keep="first")
    df['ShipID'] = df['ShipID'].astype(int)

    df.to_sql(
        dwd_table_name
        , con=conn
        , if_exists='replace'
        , index=False
        , dtype={
            "ShipID": sqlalchemy.types.BigInteger(),   #. bigint(20)
            "Latitude": sqlalchemy.types.Float(precision=5,asdecimal=True),   #. double DEFAULT
            "Longitude": sqlalchemy.types.Float(precision=5,asdecimal=True),   #. double DEFAULT
            "Speed": sqlalchemy.types.Float(precision=3,asdecimal=True),   #. double DEFAULT
            "MovementDatetime": sqlalchemy.types.DateTime(),   #. timestamp NULL
            "Heading": sqlalchemy.types.Float(precision=3,asdecimal=True),   #. double DEFAULT
            "Draught": sqlalchemy.types.Float(precision=3,asdecimal=True),   #. double DEFAULT
            "Destination": sqlalchemy.types.String(25),   #. text,
            "DestinationTidied": sqlalchemy.types.String(25),   #. text,
            "ETA": sqlalchemy.types.DateTime(),
            "MoveStatus": sqlalchemy.types.String(25),   #. text,
            "InsertDate": sqlalchemy.types.DateTime()  #. text,
        }
         
         )
    count_dwd = len(df)
    logger.info(f'{dwd_table_name} mark, count dwd table is {count_dwd}')
    
    #! dwd2
    if daily_history_mark == True:
        logger.info('history mark')
        rfc3339_format = '%Y-%m-%dT%H:%M:%S.%fZ'
        now = datetime.today()
        beginning_of_day = datetime.combine(now, dttime.min)
        df = df[
            pd.to_datetime(df['MovementDateTime']).astype('datetime64[ns]') >= beginning_of_day
            ]
        # df['insert_datetime'] = now.strftime('%Y-%m-%d')
        
        df = df [["ShipID",
                "Latitude",
                "Longitude",
                "Speed",
                "MovementDateTime",
                "Heading",
                "Draught",
                "InsertDate",]]
        df.to_sql(
            "{table_name}".format(table_name = dwd_table_name2)
            , conn
            , if_exists ="append"
            , index = False
            # , dtype={"Insert"}
        )
        conn.execute("""
                update IHSVesselPositionDB.vesselposition_dailyhistory 
                partition(p7)
                    set insertdate = now()
                    where  (insertdate) = '0000-00-00 00:00:00'
                """)

if __name__ == "__main__":
    try:
        eleven_clock = datetime.now().hour
        path = "D:\\samo\\bunge_freight_etl\\" 
        logpaht = "D:\\samo\\bunge_freight_etl\\logs\\"
        logger.add(
            logpaht + "ods_vesselposition_spire_shipid_dtl" + "_{timestamp}.log".format(
                timestamp = datetime.today().strftime('%Y%m%d')), backtrace=True, diagnose=True)  # Caution, may leak sensitive data in prod
        # ---------------------------------------------------------------------------- #
        #                              create nested list                              #
        # ---------------------------------------------------------------------------- #

        ls = get_imo_ls(path)
        sizes = [1000] * int(np.floor(len(ls)/1000)) + [np.mod(len(ls),1000)]# create a list with desired sizes (45 lists with 1000 elements and one with 600)
        result = split_list(ls, sizes) # call the function
        
        URL = "https://api.spire.com/graphql"
        df = pd.DataFrame()
        q = query_()
        start_time = time.time()
        logger.info("start")
        # ---------------------------------------------------------------------------- #
        #                                      run                                     #
        # ---------------------------------------------------------------------------- #
        for i in tqdm(range(1, int(np.floor(len(result)/10)) + 2)): 
            pool = multiprocessing.Pool(10)
            k = result[(i-1)*10 : i*10]
            results = pool.map(run_query, k) 
            pool.close() 
            df = pd.concat([df, pd.concat(results, axis=0)], ignore_index=True)  
            # df = df.append(pd.concat(results, axis=0))

            time.sleep(1)
            # break
        df.to_csv(path + 'dataup\\ods_vesselposition_spire_shipid_dtl_hf.csv', index = None)
        df = mutation_data(df)
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"len(df) == {len(df)} \n Script ran in {total_time:.2f} seconds.")
        

        # ---------------------------------------------------------------------------- #
        #                                     write                                    #
        # ---------------------------------------------------------------------------- #
        insert_to_db(df)
    except Exception as e:
        logpaht = "D:\\samo\\bunge_freight_etl\\logs\\"
        with open(logpaht + 'error_log_spire_graph_imo.txt', 'w') as f:
            f.write(str(e))

    
    

    