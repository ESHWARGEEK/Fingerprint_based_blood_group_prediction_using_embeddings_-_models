import oracledb

try:
    conn = oracledb.connect(user="vec_user", password="mypassword", dsn="localhost:1521/FREEPDB1")
    print("Connection successful!")
    conn.close()
except Exception as e:
    print("Connection failed:", e)
