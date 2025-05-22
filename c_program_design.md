# C Program Design: Hive ODBC Interaction

## 1. Program Objective

The C program will connect to a Hive database using ODBC. It will read data from the `source_data` table, perform an aggregation (sum of `value` by `category`) within the C program itself, write these aggregated results to a temporary local CSV file, and then load this CSV file into the `processed_data` Hive table.

## 2. Core Libraries

The following standard C and ODBC libraries will be necessary:

*   `stdio.h` (for standard I/O operations like `printf`, file operations)
*   `stdlib.h` (for memory allocation `malloc`, `free`, `exit`)
*   `string.h` (for string manipulation like `strcpy`, `strcmp`)
*   `sql.h` (core ODBC definitions and types like `SQLHENV`, `SQLHDBC`, `SQLHSTMT`)
*   `sqlext.h` (extended ODBC definitions and function prototypes like `SQLDriverConnect`, `SQLGetDiagRec`)

## 3. Key Data Structures (in C)

To hold the aggregated data in memory before writing to a temporary file:

*   **Structure Definition:**
    ```c
    struct CategoryAggregate {
        char category[256]; // Assuming category names are within 255 chars
        double sum_value;
        struct CategoryAggregate *next; // For linked list implementation
    };
    ```

*   **Storage Mechanism:**
    A **singly linked list** of `CategoryAggregate` structures is a suitable choice. This allows for dynamic addition of new categories as they are encountered without needing to pre-define the number of categories or manage array resizing.
    *   A pointer to the head of the list (`struct CategoryAggregate *head = NULL;`) will be maintained.

## 4. Program Flow

### a. Initialization

1.  **Allocate Environment Handle:**
    *   Use `SQLAllocHandle(SQL_HANDLE_ENV, SQL_NULL_HANDLE, &henv)` to get an environment handle (`SQLHENV henv`).
2.  **Set ODBC Version:**
    *   Call `SQLSetEnvAttr(henv, SQL_ATTR_ODBC_VERSION, (SQLPOINTER)SQL_OV_ODBC3, 0)` to specify ODBC version 3.x.
3.  **Allocate Connection Handle:**
    *   Use `SQLAllocHandle(SQL_HANDLE_DBC, henv, &hdbc)` to get a connection handle (`SQLHDBC hdbc`).

### b. Connection

1.  **Connect to Hive:**
    *   Use `SQLDriverConnect(hdbc, NULL, (SQLCHAR*)"DSN=MyHiveDSN;UID=user;PWD=password;", SQL_NTS, NULL, 0, NULL, SQL_DRIVER_NOPROMPT)`.
    *   **Placeholder for Connection String:** `"DSN=YourHiveDSN;Host=your_hiveserver2_host;Port=10000;HiveServerType=2;AuthMech=YourAuthMechanism;UID=your_username;PWD=your_password;Database=your_database;"`
        *   The exact string will depend on the Hive ODBC driver and HiveServer2 configuration (DSN-based or DSN-less).

### c. Error Handling

*   Crucially, the return code of every ODBC function (e.g., `SQLAllocHandle`, `SQLDriverConnect`, `SQLExecDirect`, `SQLFetch`) **must be checked** (e.g., against `SQL_SUCCESS` or `SQL_SUCCESS_WITH_INFO`).
*   A utility function, say `void extract_error(char *fn, SQLHANDLE handle, SQLSMALLINT type)`, should be implemented. This function will use `SQLGetDiagRec` in a loop to retrieve and display detailed error information (SQLState, Native Error Code, Message Text) when an ODBC call fails or returns warnings.

### d. Data Reading

1.  **Allocate Statement Handle:**
    *   Use `SQLAllocHandle(SQL_HANDLE_STMT, hdbc, &hstmt)` to get a statement handle (`SQLHSTMT hstmt`).
2.  **Prepare and Execute Query:**
    *   Define the HQL query: `SQLCHAR *query = (SQLCHAR *)"SELECT id, category, value FROM source_data;"`.
    *   Execute using `SQLExecDirect(hstmt, query, SQL_NTS)`.
3.  **Bind Columns (Optional but Recommended for Fixed Data Types):**
    *   Alternatively, data can be fetched with `SQLGetData` without prior binding. For this design, we'll use `SQLGetData` as it's more flexible if column sizes are unknown. If binding:
        ```c
        // SQLINTEGER id;
        // SQLCHAR category_buffer[256];
        // SQLDOUBLE value;
        // SQLLEN id_indicator, category_indicator, value_indicator;
        // SQLBindCol(hstmt, 1, SQL_C_SLONG, &id, sizeof(id), &id_indicator);
        // SQLBindCol(hstmt, 2, SQL_C_CHAR, category_buffer, sizeof(category_buffer), &category_indicator);
        // SQLBindCol(hstmt, 3, SQL_C_DOUBLE, &value, sizeof(value), &value_indicator);
        ```
4.  **Fetch Rows:**
    *   Loop using `while (SQLFetch(hstmt) == SQL_SUCCESS)`.
5.  **Retrieve Data (inside fetch loop):**
    *   Use `SQLGetData` for each column:
        ```c
        // SQLINTEGER id; // If not binding
        // SQLCHAR category_buffer[256]; // If not binding
        // SQLDOUBLE value; // If not binding
        // SQLLEN id_indicator, category_indicator, value_indicator;

        // SQLGetData(hstmt, 1, SQL_C_SLONG, &id, sizeof(id), &id_indicator);
        // SQLGetData(hstmt, 2, SQL_C_CHAR, category_buffer, sizeof(category_buffer), &category_indicator);
        // SQLGetData(hstmt, 3, SQL_C_DOUBLE, &value, sizeof(value), &value_indicator);
        ```

### e. Data Processing (in C)

*   Inside the `SQLFetch` loop, after retrieving `category_buffer` and `value`:
    1.  Search the linked list of `CategoryAggregate` structures for an entry with the current `category_buffer`.
    2.  **If found:** Add the `value` to its `sum_value`.
    3.  **If not found:**
        *   Allocate a new `CategoryAggregate` node.
        *   Copy `category_buffer` into its `category` field.
        *   Set its `sum_value` to the current `value`.
        *   Add this new node to the linked list.

### f. Data Writing (to Temp File)

1.  **Open Temporary CSV File:**
    *   `FILE *temp_csv = fopen("temp_aggregated_data.csv", "w");`
    *   Check if `temp_csv` is NULL (error opening file).
2.  **Iterate and Write:**
    *   Traverse the linked list of `CategoryAggregate` structures.
    *   For each node, write a line to the CSV: `fprintf(temp_csv, "%s,%.2f\n", node->category, node->sum_value);`
3.  **Close File:**
    *   `fclose(temp_csv);`

### g. Data Loading (Temp File to Hive)

1.  **Prepare and Execute HQL:**
    *   Use the same statement handle `hstmt` or allocate a new one if the previous one was not freed. If reusing, call `SQLCloseCursor(hstmt)` or `SQLFreeStmt(hstmt, SQL_CLOSE)` if it was from a `SELECT`.
    *   Define the HQL query: `SQLCHAR *load_query = (SQLCHAR *)"LOAD DATA LOCAL INPATH 'temp_aggregated_data.csv' OVERWRITE INTO TABLE processed_data;";`
        *   **Note:** The path `'temp_aggregated_data.csv'` is relative to where the C program is executed. HiveServer2 must be configured with `hive.server2.enable.doAs=false` (or appropriate proxy user setup) and potentially `hive.server2.allow.local.infile.loading=true` (though this property is not standard and behavior can depend on Hive version/distribution) for `LOAD DATA LOCAL INPATH` to work securely and correctly. The user running HiveServer2 also needs read access to the path where the file is temporarily placed if not truly "local" to the client process's perspective in all Hive setups.
    *   Execute using `SQLExecDirect(hstmt, load_query, SQL_NTS)`. Check the return code carefully.

### h. Cleanup

1.  **Free Statement Handle:**
    *   `SQLFreeHandle(SQL_HANDLE_STMT, hstmt);`
2.  **Disconnect:**
    *   `SQLDisconnect(hdbc);`
3.  **Free Connection Handle:**
    *   `SQLFreeHandle(SQL_HANDLE_DBC, hdbc);`
4.  **Free Environment Handle:**
    *   `SQLFreeHandle(SQL_HANDLE_ENV, henv);`
5.  **Free Linked List:**
    *   Iterate through the `CategoryAggregate` linked list and `free()` each node.
6.  **Delete Temporary File:**
    *   `remove("temp_aggregated_data.csv");` (optional, good for cleanup)

## 5. Main Functions Overview

*   `int main()`:
    *   Calls initialization functions.
    *   Calls `connect_db()`.
    *   Calls `read_and_process_data()`.
    *   Calls `write_aggregates_to_csv()`.
    *   Calls `load_csv_to_hive()`.
    *   Handles overall error checking and calls cleanup routines.
*   `SQLRETURN connect_db(SQLHENV henv, SQLHDBC *hdbc_ptr)`:
    *   Allocates `hdbc`.
    *   Sets connection attributes if necessary.
    *   Calls `SQLDriverConnect`.
    *   Returns status.
*   `SQLRETURN read_and_process_data(SQLHDBC hdbc, struct CategoryAggregate **head_ptr)`:
    *   Allocates `hstmt`.
    *   Executes `SELECT` query.
    *   Fetches rows and uses `SQLGetData`.
    *   Builds/updates the linked list of aggregates.
    *   Frees `hstmt`.
    *   Returns status.
*   `int write_aggregates_to_csv(const char *filename, struct CategoryAggregate *head)`:
    *   Opens, writes to, and closes the CSV file.
    *   Returns 0 on success, -1 on failure.
*   `SQLRETURN load_csv_to_hive(SQLHDBC hdbc, const char *csv_filepath)`:
    *   Allocates `hstmt`.
    *   Executes `LOAD DATA LOCAL INPATH` query.
    *   Frees `hstmt`.
    *   Returns status.
*   `void odbc_error(const char *function_name, SQLHANDLE handle, SQLSMALLINT handle_type)`:
    *   Uses `SQLGetDiagRec` to print detailed ODBC error messages.
    *   Includes SQLState, native error code, and message.

## 6. Assumptions/Prerequisites for C code execution

*   **Hive ODBC Driver:** A compatible Hive ODBC driver is installed on the machine where the C program will be compiled and run. The driver must be correctly registered with the system's ODBC Driver Manager (e.g., unixODBC).
*   **DSN or Connection String:** Either a Data Source Name (DSN) for Hive is pre-configured in `odbc.ini`, or all necessary parameters for a DSN-less connection string are known.
*   **HiveServer2 Configuration:**
    *   HiveServer2 is running and accessible from the client machine.
    *   HiveServer2 must be configured to allow `LOAD DATA LOCAL INPATH`. This often involves security considerations and settings like `hive.server2.enable.doAs=false` or proper proxy user configurations. Specific settings can vary by Hive distribution (e.g., Cloudera, Hortonworks, Apache Hive).
*   **Table Existence:**
    *   The input table `source_data` must exist in Hive and be populated with data in the expected format (`id INT, category STRING, value DOUBLE`).
    *   The output table `processed_data` must exist in Hive with the correct schema (`category STRING, aggregated_value DOUBLE`) to receive the loaded data.
*   **Permissions:** The Hive user utilized by the C program must have:
    *   `SELECT` permissions on `source_data`.
    *   `INSERT` (or `WRITE`) permissions on `processed_data`.
    *   Permissions to execute `LOAD DATA LOCAL INPATH`.
*   **Compilation Environment:** A C compiler (like GCC) and the ODBC development libraries (e.g., `unixODBC-dev` or equivalent) must be installed to compile the C program.
