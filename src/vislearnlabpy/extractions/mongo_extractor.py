import datetime
import pymongo as pm
class MongoExtractor():
    def __init__(self, conn_str, database_name, collection_name, output_dir="mongo_output", date=None):
        conn = pm.MongoClient(conn_str)
        self.database = conn[database_name]
        self.collection = self.database[collection_name]
        self.output_dir = output_dir
        if date is not None:
            try:
                self.date = datetime.strptime(date,  "%Y%m%d").date()
            except ValueError:
                raise ValueError("Invalid date format. Please use YYYYMMDD.")
        else:
            self.date = None
