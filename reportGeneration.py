import pandas as pd
import mysql.connector
# Database configuration
db_config = {
    'user': 'root',
    'password': '1234',
    'host': 'localhost',
    'database': 'face recognition'
}

db = mysql.connector.connect(**db_config)
cursor = db.cursor()

def generate_report():
    query = """
    SELECT name, MAX(timestamp) as last_seen
    FROM students
    GROUP BY name
    """
    cursor.execute(query)
    results = cursor.fetchall()

    # Create a DataFrame
    df = pd.DataFrame(results, columns=['Name', 'Last Seen'])

    # Save to Excel file
    df.to_excel('recognition_report.xlsx', index=False)

# Testing report generation
if __name__ == "__main__":
    generate_report()
    cursor.close()
    db.close()
