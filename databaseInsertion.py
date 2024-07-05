import mysql.connector
from datetime import datetime

# Database configuration
db_config = {
    'user': 'root',
    'password': '1234',
    'host': 'localhost',
    'database': 'face recognition'
}

db = mysql.connector.connect(**db_config)
cursor = db.cursor()

def insert_to_db(name, accuracy):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    query = "INSERT INTO students (name, accuracy, timestamp) VALUES (%s, %s, %s)"
    values = (name, accuracy, timestamp)
    cursor.execute(query, values)
    db.commit()

# Testing database insertion
if __name__ == "__main__":
    insert_to_db("Person_1", 95.6)
    cursor.close()
    db.close()
