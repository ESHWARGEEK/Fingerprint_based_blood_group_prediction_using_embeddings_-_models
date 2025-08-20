import pandas as pd
import oracledb
import ast
import array

# 1) Load CSV and parse embedding string
df = pd.read_csv("../csv/fingerprint_embeddings.csv") # Use the correct path
df["embedding"] = df["embedding"].apply(lambda x: ast.literal_eval(x))

# 2) Connect to the database
conn = oracledb.connect(user="vec_user", password="mypassword", dsn="localhost:1521/FREEPDB1")
cur = conn.cursor()

# 3) Prepare rows for insertion
rows = [
    (row["label"], array.array('f', row["embedding"]))
    for _, row in df.iterrows()
]

# 4) Insert rows one by one
try:
    for row in rows:
        cur.execute(
            "INSERT INTO vec_user.fingerprints (label, embedding) VALUES (:1, :2)",
            row
        )
    conn.commit()
    print(f"✅ Successfully inserted {len(rows)} rows.")

except oracledb.DatabaseError as e:
    print(f"❌ A database error occurred: {e}")
    conn.rollback() # Rollback changes on error

finally:
    # 5) Always close the cursor and connection
    if cur:
        cur.close()
    if conn:
        conn.close()
