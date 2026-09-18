import urllib.parse
import pandas as pd
from sqlalchemy import create_engine

# 1. Load your enriched dataset CSV
csv_file_path = 'Final_Master_dataset.csv'
df = pd.read_csv(csv_file_path)
df['Week_Start'] = pd.to_datetime(df['Week_Start'])

# 2. Define your MySQL credentials (especially useful if your password contains special characters like '@', '#', etc.)
db_user = 'root'
db_password = '!@Mstrong@1234'
db_host = 'localhost'
db_port = '3306'
db_name = 'gdelttrends'

# 3. URL-encode the password to safely handle special symbols in connection strings
safe_password = urllib.parse.quote_plus(db_password)

# 4. Construct the connection string using the encoded password
connection_string = (
    f'mysql+pymysql://{db_user}:{safe_password}@{db_host}:{db_port}/{db_name}'
)
engine = create_engine(connection_string)

# 5. Write the dataframe into the table 'final_master_dataset'
table_name = 'final_master_dataset'
print(
    f'⏳ Uploading {len(df)} rows into table \'{table_name}\' in MySQL database'
    f" '{db_name}'..."
)

df.to_sql(
    name=table_name,
    con=engine,
    if_exists='replace',
    index=False,
    chunksize=1000,
)

print(f"✅ Successfully loaded data into MySQL table '{table_name}'!")