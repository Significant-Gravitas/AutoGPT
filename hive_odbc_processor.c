#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sql.h>
#include <sqlext.h>

// DSN for connecting to Hive. This should be configured by the user.
// For example, in /etc/odbc.ini or ~/.odbc.ini
#define HIVE_DSN "HiveDSN" 
// Default path for the temporary CSV file.
// IMPORTANT: HiveServer2 needs access to this path for LOAD DATA LOCAL INPATH.
// It's best to use an absolute path that is accessible by the user running HS2.
#define TEMP_CSV_PATH "temp_aggregated_data.csv"
#define CATEGORY_MAX_LEN 256
#define INITIAL_AGGREGATES_CAPACITY 10

// Structure to hold aggregated data
struct CategoryAggregate {
    char category[CATEGORY_MAX_LEN];
    double sum_value;
    // int count; // Optional: for AVG etc. Not used in current sum-only logic
};

// Global array for aggregates (for simplicity in this example)
// A more robust solution might use a dynamically growing linked list or array
struct CategoryAggregate *aggregates = NULL;
int aggregates_count = 0;
int aggregates_capacity = 0;

// Function to print ODBC error details
void odbc_error(SQLHANDLE handle, SQLSMALLINT handle_type, SQLRETURN retcode, const char *fn_name) {
    SQLCHAR sqlstate[6];
    SQLINTEGER native_error;
    SQLCHAR message_text[SQL_MAX_MESSAGE_LENGTH];
    SQLSMALLINT text_length;
    SQLSMALLINT rec_num = 1;

    if (retcode == SQL_SUCCESS || retcode == SQL_SUCCESS_WITH_INFO) {
        if (retcode == SQL_SUCCESS_WITH_INFO) {
            fprintf(stderr, "ODBC INFO from %s:\n", fn_name);
        } else {
            // If SQL_SUCCESS, no need to print further, unless for verbose logging
            return;
        }
    } else if (retcode == SQL_INVALID_HANDLE) {
        fprintf(stderr, "ODBC ERROR from %s: Invalid handle!\n", fn_name);
        return;
    } else if (retcode == SQL_ERROR) {
        fprintf(stderr, "ODBC ERROR from %s:\n", fn_name);
    } else if (retcode == SQL_NO_DATA) {
        fprintf(stderr, "ODBC INFO from %s: No data found.\n", fn_name);
        return;
    } else {
        fprintf(stderr, "ODBC Call from %s returned: %d\n", fn_name, retcode);
    }

    while (SQLGetDiagRec(handle_type, handle, rec_num, sqlstate, &native_error,
                         message_text, sizeof(message_text), &text_length) == SQL_SUCCESS) {
        fprintf(stderr, "  SQLState: %s, Native Error: %ld\n", sqlstate, native_error);
        fprintf(stderr, "  Message: %s\n", message_text);
        rec_num++;
    }
    if (rec_num == 1 && retcode != SQL_SUCCESS_WITH_INFO) { // SQL_SUCCESS_WITH_INFO might not have a specific record if it's a driver informational message
         fprintf(stderr, "  SQLGetDiagRec failed to retrieve error details.\n");
    }
}


// Function to initialize and connect to the database
SQLHDBC connect_db(SQLHENV henv, const char *dsn_connection_string) {
    SQLHDBC hdbc = SQL_NULL_HDBC;
    SQLRETURN ret;

    // Allocate Connection Handle
    ret = SQLAllocHandle(SQL_HANDLE_DBC, henv, &hdbc);
    if (ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO) {
        odbc_error(henv, SQL_HANDLE_ENV, ret, "SQLAllocHandle (DBC)");
        return SQL_NULL_HDBC;
    }

    // Connect to Hive
    // The dsn_connection_string can be a simple DSN name like "HiveDSN"
    // or a full DSN-less connection string like:
    // "DRIVER={Actual Hive ODBC Driver Name};Host=your_host;Port=10000;Database=default;AuthMech=X;UID=user;PWD=pass;"
    printf("Attempting to connect using connection string: %s\n", dsn_connection_string);
    
    ret = SQLDriverConnect(hdbc, NULL, (SQLCHAR *)dsn_connection_string, SQL_NTS,
                           NULL, 0, NULL, SQL_DRIVER_NOPROMPT); // Use SQL_DRIVER_NOPROMPT or SQL_DRIVER_COMPLETE_REQUIRED

    if (ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO) {
        odbc_error(hdbc, SQL_HANDLE_DBC, ret, "SQLDriverConnect");
        SQLFreeHandle(SQL_HANDLE_DBC, hdbc);
        return SQL_NULL_HDBC;
    }
    printf("Successfully connected to Hive.\n");
    return hdbc;
}

// Function to add or update an aggregate
void add_or_update_aggregate(const char* category, double value) {
    for (int i = 0; i < aggregates_count; i++) {
        if (strncmp(aggregates[i].category, category, CATEGORY_MAX_LEN) == 0) {
            aggregates[i].sum_value += value;
            return;
        }
    }

    // Category not found, add new one
    if (aggregates_count >= aggregates_capacity) {
        aggregates_capacity = (aggregates_capacity == 0) ? INITIAL_AGGREGATES_CAPACITY : aggregates_capacity * 2;
        struct CategoryAggregate *temp = realloc(aggregates, aggregates_capacity * sizeof(struct CategoryAggregate));
        if (temp == NULL) {
            perror("Failed to reallocate memory for aggregates");
            // In a real app, handle this more gracefully
            exit(EXIT_FAILURE);
        }
        aggregates = temp;
    }

    strncpy(aggregates[aggregates_count].category, category, CATEGORY_MAX_LEN - 1);
    aggregates[aggregates_count].category[CATEGORY_MAX_LEN - 1] = '\0'; // Ensure null termination
    aggregates[aggregates_count].sum_value = value;
    aggregates_count++;
}

// Function to read data from source_data and process it
int read_and_process_data(SQLHDBC hdbc) {
    SQLHSTMT hstmt = SQL_NULL_HSTMT;
    SQLRETURN ret;
    // Buffers for data fetched from Hive
    SQLCHAR category_buf[CATEGORY_MAX_LEN];
    SQLDOUBLE value_buf;
    // Indicators for null values
    SQLLEN category_ind, value_ind;

    // Allocate Statement Handle
    ret = SQLAllocHandle(SQL_HANDLE_STMT, hdbc, &hstmt);
    if (ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO) {
        odbc_error(hdbc, SQL_HANDLE_DBC, ret, "SQLAllocHandle (STMT for read)");
        return 0; // Indicate failure
    }

    // Execute Query
    SQLCHAR *query = (SQLCHAR *)"SELECT category, value FROM source_data;"; // Removed 'id' as it's not used in processing
    printf("Executing query: %s\n", query);
    ret = SQLExecDirect(hstmt, query, SQL_NTS);
    if (ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO) {
        odbc_error(hstmt, SQL_HANDLE_STMT, ret, "SQLExecDirect (SELECT)");
        SQLFreeHandle(SQL_HANDLE_STMT, hstmt);
        return 0; // Indicate failure
    }

    // Bind columns for fetched data. This links the C variables to the result set columns.
    // Column 1: category (SQL_C_CHAR)
    // SQLBindCol(hstmt, 1, SQL_C_CHAR, category_buf, sizeof(category_buf), &category_ind);
    // Column 2: value (SQL_C_DOUBLE)
    // SQLBindCol(hstmt, 2, SQL_C_DOUBLE, &value_buf, sizeof(value_buf), &value_ind);
    // Using SQLGetData instead of SQLBindCol as per design doc for flexibility

    printf("Fetching and processing data...\n");
    // Fetch rows
    while (SQL_SUCCEEDED(ret = SQLFetch(hstmt))) {
        // Retrieve data for each column using SQLGetData
        // Column 1: category
        ret = SQLGetData(hstmt, 1, SQL_C_CHAR, category_buf, sizeof(category_buf), &category_ind);
        if (!SQL_SUCCEEDED(ret) && ret != SQL_NO_DATA) { // Allow SQL_NO_DATA for last partial chunk
            odbc_error(hstmt, SQL_HANDLE_STMT, ret, "SQLGetData (category)");
            continue; 
        }
        // Column 2: value
        ret = SQLGetData(hstmt, 2, SQL_C_DOUBLE, &value_buf, sizeof(value_buf), &value_ind);
        if (!SQL_SUCCEEDED(ret) && ret != SQL_NO_DATA) {
            odbc_error(hstmt, SQL_HANDLE_STMT, ret, "SQLGetData (value)");
            continue;
        }
        
        // Check if data is not NULL
        if (category_ind != SQL_NULL_DATA && value_ind != SQL_NULL_DATA) {
            add_or_update_aggregate((char*)category_buf, value_buf);
        } else {
            if (category_ind == SQL_NULL_DATA) printf("Skipping row due to NULL category.\n");
            if (value_ind == SQL_NULL_DATA) printf("Skipping row due to NULL value for category %s.\n", (char*)category_buf);
        }
    }

    // Check final state of SQLFetch
    if (ret != SQL_NO_DATA && ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO) {
         odbc_error(hstmt, SQL_HANDLE_STMT, ret, "SQLFetch (final state)");
    }
    
    printf("Finished processing data. Aggregated %d categories.\n", aggregates_count);

    // Cleanup statement handle
    SQLFreeStmt(hstmt, SQL_CLOSE); // Close cursor before freeing handle
    SQLFreeHandle(SQL_HANDLE_STMT, hstmt);
    return 1; // Indicate success
}

// Function to write aggregated data to a temporary CSV file
int write_aggregates_to_csv(const char* filename) {
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        perror("Failed to open temporary CSV file");
        return 0; // Failure
    }

    printf("Writing aggregates to CSV: %s\n", filename);
    // No header for simpler LOAD DATA
    // fprintf(fp, "category,sum_value\n"); 

    for (int i = 0; i < aggregates_count; i++) {
        fprintf(fp, "%s,%.2f\n", aggregates[i].category, aggregates[i].sum_value);
    }

    fclose(fp);
    printf("Successfully wrote %d records to %s.\n", aggregates_count, filename);
    return 1; // Success
}

// Function to load data from CSV into processed_data Hive table
int load_csv_to_hive(SQLHDBC hdbc, const char* csv_filepath_for_hql) {
    SQLHSTMT hstmt = SQL_NULL_HSTMT;
    SQLRETURN ret;
    char load_query[512];

    // Construct the LOAD DATA query.
    // The path specified in 'LOAD DATA LOCAL INPATH' is from the perspective of the client
    // where this C program is running. HiveServer2 will pull the file from this client.
    // Ensure the path is absolute or correctly relative for the C program's execution environment.
    sprintf(load_query, "LOAD DATA LOCAL INPATH '%s' OVERWRITE INTO TABLE processed_data", csv_filepath_for_hql);

    // Allocate Statement Handle
    ret = SQLAllocHandle(SQL_HANDLE_STMT, hdbc, &hstmt);
    if (ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO) {
        odbc_error(hdbc, SQL_HANDLE_DBC, ret, "SQLAllocHandle (STMT for LOAD)");
        return 0; // Failure
    }

    printf("Executing HQL: %s\n", load_query);
    ret = SQLExecDirect(hstmt, (SQLCHAR*)load_query, SQL_NTS);
    if (ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO) {
        odbc_error(hstmt, SQL_HANDLE_STMT, ret, "SQLExecDirect (LOAD DATA)");
        SQLFreeHandle(SQL_HANDLE_STMT, hstmt);
        return 0; // Failure
    }

    printf("Successfully loaded data from %s into processed_data table.\n", csv_filepath_for_hql);
    SQLFreeStmt(hstmt, SQL_CLOSE); // Close cursor (if any)
    SQLFreeHandle(SQL_HANDLE_STMT, hstmt);
    return 1; // Success
}


int main(int argc, char *argv[]) {
    SQLHENV henv = SQL_NULL_HENV;
    SQLHDBC hdbc = SQL_NULL_HDBC;
    SQLRETURN ret;
    
    // Default DSN, can be overridden by command line argument
    char *connection_string = HIVE_DSN; 

    if (argc > 1) {
        connection_string = argv[1]; 
        printf("Using connection string from command line: %s\n", connection_string);
    } else {
        printf("Using default DSN: %s. You can also pass a full connection string as a command line argument.\n", HIVE_DSN);
        printf("Example full string: \"DRIVER={Hive Driver Name};Host=hs2host;Port=10000;...\"\n");
    }
    printf("Temporary CSV will be created at: %s\n", TEMP_CSV_PATH);
    printf("The LOAD DATA LOCAL INPATH query will use this path. Ensure HiveServer2 can access this file from the client machine or the path is correctly configured.\n");


    // Allocate Environment Handle
    ret = SQLAllocHandle(SQL_HANDLE_ENV, SQL_NULL_HANDLE, &henv);
    if (ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO) {
        fprintf(stderr, "Failed to allocate environment handle. Retcode: %d\n", ret);
        // Cannot use odbc_error as handle might be invalid
        return EXIT_FAILURE;
    }

    // Set ODBC Version
    ret = SQLSetEnvAttr(henv, SQL_ATTR_ODBC_VERSION, (SQLPOINTER)SQL_OV_ODBC3, 0);
    if (ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO) {
        odbc_error(henv, SQL_HANDLE_ENV, ret, "SQLSetEnvAttr (ODBC_VERSION)");
        SQLFreeHandle(SQL_HANDLE_ENV, henv);
        return EXIT_FAILURE;
    }

    // Connect to Database
    hdbc = connect_db(henv, connection_string);
    if (hdbc == SQL_NULL_HDBC) {
        SQLFreeHandle(SQL_HANDLE_ENV, henv);
        return EXIT_FAILURE;
    }

    // Read and Process Data
    if (read_and_process_data(hdbc)) {
        // Write Aggregates to CSV
        if (aggregates_count > 0) {
            if (write_aggregates_to_csv(TEMP_CSV_PATH)) {
                // Load CSV to Hive
                if (!load_csv_to_hive(hdbc, TEMP_CSV_PATH)) {
                    fprintf(stderr, "Failed to load CSV data into Hive.\n");
                }
            } else {
                fprintf(stderr, "Failed to write aggregates to CSV.\n");
            }
        } else {
            printf("No data aggregated, skipping CSV write and Hive load.\n");
        }
    } else {
        fprintf(stderr, "Failed to read and process data from Hive.\n");
    }

    // Cleanup
    printf("Disconnecting from Hive.\n");
    SQLDisconnect(hdbc);
    SQLFreeHandle(SQL_HANDLE_DBC, hdbc);
    SQLFreeHandle(SQL_HANDLE_ENV, henv);

    if (aggregates != NULL) {
        free(aggregates);
    }
    
    // Optional: remove temporary file after processing
    // remove(TEMP_CSV_PATH);

    printf("Processing finished.\n");
    return EXIT_SUCCESS;
}

/*
--- Compilation (Linux with unixODBC) ---
gcc hive_odbc_processor.c -o hive_odbc_processor -lodbc

--- Setup ---
1.  Install unixODBC and unixODBC development libraries:
    On Debian/Ubuntu: sudo apt-get install unixodbc unixodbc-dev
    On RHEL/CentOS: sudo yum install unixODBC unixODBC-devel

2.  Install your Hive ODBC Driver. This is vendor-specific (e.g., Cloudera, Hortonworks, Microsoft, Simba).
    Follow the driver documentation to install it. It usually involves placing the .so file in a specific directory.

3.  Configure your DSN in an odbc.ini file.
    System-wide: /etc/odbc.ini
    User-specific: ~/.odbc.ini

    Example odbc.ini entry (replace with your actual driver details):
    [HiveDSN] ; This is the DSN name used by the program by default
    Description=Hive DSN Example
    Driver=<Full path to your Hive ODBC driver .so file, e.g., /usr/lib/hive/lib/native/libclouderahiveodbc.so>
    Host=<Your HiveServer2 Hostname or IP>
    Port=10000 ; Or your HiveServer2 port
    Database=default ; Or the database you want to connect to
    AuthMech=3 ; Authentication mechanism. Common values:
               ; 0 = No Authentication (No SASL) - if HS2 is configured for this
               ; 1 = Kerberos
               ; 2 = User Name
               ; 3 = User Name and Password
               ; Consult your Hive driver documentation for specific values.
    ; UID=<Your Hive Username> ; Required if AuthMech needs username
    ; PWD=<Your Hive Password> ; Required if AuthMech needs password
    ; SSL settings, Kerberos realm, Service Principal, etc., may also be needed depending on your Hive security configuration.

--- Running ---
1.  Ensure Hive tables `source_data` and `processed_data` are created as per `table_definitions_and_analysis.md`.
    Example HQL (run in Hive CLI or Beeline):
    CREATE TABLE IF NOT EXISTS source_data (id INT, category STRING, value DOUBLE) ROW FORMAT DELIMITED FIELDS TERMINATED BY ',';
    CREATE TABLE IF NOT EXISTS processed_data (category STRING, aggregated_value DOUBLE) ROW FORMAT DELIMITED FIELDS TERMINATED BY ',';

    -- Populate source_data with some sample data:
    -- INSERT INTO source_data VALUES (1, 'Electronics', 150.0), (2, 'Books', 25.5), (3, 'Electronics', 200.0), (4, 'Home', 75.0), (5, 'Books', 15.0);
    -- Note: Direct INSERT INTO might not work for TEXTFILE tables in older Hive. Use LOAD DATA or create as ORC with transactional=true.
    -- For simplicity, you can create a sample CSV file (e.g., sample_source_data.csv):
    -- 1,Electronics,150.0
    -- 2,Books,25.5
    -- 3,Electronics,200.0
    -- 4,Home,75.0
    -- 5,Books,15.0
    -- And then load it:
    -- LOAD DATA LOCAL INPATH './sample_source_data.csv' OVERWRITE INTO TABLE source_data;


2.  Compile the C program:
    gcc hive_odbc_processor.c -o hive_odbc_processor -lodbc

3.  Run the compiled program:
    ./hive_odbc_processor
    Or, to use a specific connection string (e.g., if DSN is not set up or you want to override):
    ./hive_odbc_processor "DRIVER=/path/to/your/libxxxx.so;Host=your_hs2_host;Port=10000;Database=default;AuthMech=X;UID=user;PWD=pass"

--- Important Considerations for `LOAD DATA LOCAL INPATH` ---
*   The `LOCAL` keyword means the file path is relative to the client machine (where this C program runs).
*   HiveServer2 must be configured to allow `LOAD DATA LOCAL INPATH`.
    Check `hive-site.xml` for `hive.server2.enable.doAs` (often needs to be `false` for simple local loads unless user impersonation is fully set up) and potentially `hive.security.authorization.enabled`.
*   The user running the C program needs read access to `TEMP_CSV_PATH`.
*   The user that HiveServer2 runs as (or impersonates) needs to be able to access the file from the client. Network configuration might play a role if client and server are on different machines.
*   Using an absolute path for `TEMP_CSV_PATH` is generally safer for `LOAD DATA LOCAL INPATH`.
*/
