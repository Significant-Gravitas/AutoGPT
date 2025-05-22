# Table Definitions and Analysis Logic

This document outlines the Hive table structures for source and processed data, along with the HQL queries for their creation and the analysis logic.

## 1. Input Table Definition

*   **Name:** `source_data`
*   **Description:** This table holds the raw data that needs to be processed.
*   **Columns:**
    *   `id INT`: Unique identifier for each record.
    *   `category STRING`: The category to which the record belongs.
    *   `value DOUBLE`: A numerical value associated with the record, used for aggregation.
*   **Example HQL to create this table:**

```hql
CREATE TABLE IF NOT EXISTS source_data (
    id INT,
    category STRING,
    value DOUBLE
)
COMMENT 'Table to store raw source data for analysis'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE;
```

## 2. Output Table Definition

*   **Name:** `processed_data`
*   **Description:** This table stores the results of the aggregation performed on the `source_data` table.
*   **Columns:**
    *   `category STRING`: The distinct category from the source data.
    *   `aggregated_value DOUBLE`: The sum of `value` for each `category` from the `source_data` table.
*   **Example HQL to create this table:**

```hql
CREATE TABLE IF NOT EXISTS processed_data (
    category STRING,
    aggregated_value DOUBLE
)
COMMENT 'Table to store aggregated results by category'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE;
```

## 3. Analysis Logic Description

*   **Analysis to be performed:** Calculate the sum of `value` from the `source_data` table for each distinct `category`, and store the results (category and its sum) into the `processed_data` table.
*   **HQL query for analysis and insertion:**

```hql
INSERT OVERWRITE TABLE processed_data
SELECT
    category,
    SUM(value) AS aggregated_value
FROM
    source_data
GROUP BY
    category;
```
