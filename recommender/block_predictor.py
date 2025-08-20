import pandas as pd
import pyodbc
from dotenv import dotenv_values
import os

config_db = dotenv_values(".env")

# DB environment variables
AZ_SQL_DRIVER = os.getenv("AZ_SQL_DRIVER")
DATA_READ_DB_USER = os.getenv("DATA_READ_DB_USER")
DATA_READ_DB_PASS = os.getenv("DATA_READ_DB_PASS")
DATA_DB_PORT = os.getenv("DATA_DB_PORT")
DATA_DB_SERV = os.getenv("DATA_DB_SERV")
RECOMMENDERS_DB = os.getenv("RECOMMENDERS_DB")

if None in [AZ_SQL_DRIVER, DATA_READ_DB_USER, DATA_READ_DB_PASS, DATA_DB_PORT, DATA_DB_SERV, RECOMMENDERS_DB]:
    raise Exception(
        "Missing one of these environment variables: "
        "'AZ_SQL_DRIVER', 'DATA_READ_DB_USER', 'DATA_READ_DB_PASS', 'DATA_DB_PORT', 'DATA_DB_SERV', 'RECOMMENDERS_DB'")
#read table from database
def read_table(table_name):
    serv_conn_string = f"DRIVER={AZ_SQL_DRIVER};SERVER={DATA_DB_SERV};PORT={DATA_DB_PORT};DATABASE={RECOMMENDERS_DB};UID={DATA_READ_DB_USER};PWD={DATA_READ_DB_PASS};String Types=Unicode"

    try:
        conn = pyodbc.connect(serv_conn_string)
        pdf = pd.read_sql_query("SELECT * FROM " + table_name, con=conn)
        conn.commit()
        conn.close()
        return pdf
    except Exception as e:
        print("Can't connect to the database.")
        return False
    return;

scoring_table = read_table("scoring_table")