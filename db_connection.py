import sqlite3
import pandas as pd

# Read the Excel file
excel_file = r'C:\Users\saran\OneDrive\Desktop\TSoM\Python\Python Final Project\Data\Books_rating.csv'
df = pd.read_csv(excel_file)

# Connect to the SQLite database
conn = sqlite3.connect('books_rating.db')

# Write the DataFrame to the SQLite database
df.to_sql('books_rating_tbl', con=conn, if_exists='replace', index=False)

print("Successfully established connection to database")

# Commit the transaction and close the connection
conn.commit()
conn.close()
