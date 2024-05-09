import streamlit as st



import mysql.connector
from sqlalchemy import create_engine


def delete_data():
    try:
        # Connect to MySQL database
        connection = mysql.connector.connect(
            host="localhost",
            port="3306",
            user="root",
            password="new_password",
            database="caption_database"  # Name of your database
        )

        # Create a cursor object to execute SQL queries
        cursor = connection.cursor()

        # SQL query to insert data into the "caption" table
        sql_DELETE_query = "DELETE FROM caption"
        

        # Execute the SQL query
        cursor.execute(sql_DELETE_query)

        # Commit the transaction
        connection.commit()

        # Close cursor and connection
        cursor.close()
        connection.close()

        st.success("Data deleted successfully from the 'caption' table!")
    except mysql.connector.Error as error:
        st.error(f"Error deleting data from 'caption' table: {error}")


def main():
    st.title("Delete captions from database")
    st.write("click the delete check box to clear the database")

   
    # Add checkbox to trigger deletion
    delete_checkbox = st.checkbox("Delete Data ")

    
   

    # If checkbox is checked, delete data
    if delete_checkbox:
        delete_data()  
        
    

if __name__ == "__main__":
    main()
