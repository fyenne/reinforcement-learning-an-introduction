import ceODBC
from sqlalchemy import create_engine
class connect_db():
    """
    self.conn return connection.
    connect_close: shut down connection
    """
    def __init__(self, db='TestData'):
        
        self.conn   = ceODBC.connect('Driver={SQL Server};'
                'Server=GEN-NT-SQL11\MATLAB;'
                'uid=research;pwd=research')
        self.curser = self.conn.cursor()
        self.conn_alchemy = create_engine("mssql+pyodbc://research:research@GEN-NT-SQL11\MATLAB:56094/{DB}?driver=SQL+Server+Native+Client+10.0".format(DB = db))

    
    def connect_close(self):
        try:
            self.conn.close()
        except:
            self.conn_alchemy.close()
