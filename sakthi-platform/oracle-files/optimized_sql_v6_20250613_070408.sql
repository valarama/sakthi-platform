-- Optimized SQL Data Transformation Queries v6
-- Generated: 2025-06-13 07:04:07
-- Total Mappings: 81
-- Average Confidence: 100.0%

-- ====================================
-- OPTIMIZED BANKING DATA TRANSFORMATION VIEWS
-- ====================================

-- Performance optimization settings
SET hive.exec.dynamic.partition = true;
SET hive.exec.dynamic.partition.mode = nonstrict;
SET hive.vectorized.execution.enabled = true;
SET hive.vectorized.execution.reduce.enabled = true;


-- ====================================
-- TARGET TABLE: ACCT
-- ====================================

-- Drop existing view if exists
DROP VIEW IF EXISTS acct_transformed_v6;

-- Create optimized transformation view
CREATE VIEW acct_transformed_v6 AS
WITH source_data AS (
    SELECT 
        CAST(s_acct.custnbr AS INTEGER) AS custnbr -- HIGH CONFIDENCE (100.0%),
        s_acct.accttype AS accttype -- HIGH CONFIDENCE (100.0%),
        s_acct.acctstat AS acctstat -- HIGH CONFIDENCE (100.0%),
        CAST(s_acct.opendt AS DATE) AS opendt -- HIGH CONFIDENCE (100.0%),
        CAST(s_acct.curbal AS DECIMAL(15,2)) AS curbal -- HIGH CONFIDENCE (100.0%),
        CAST(s_acct.availbal AS DECIMAL(15,2)) AS availbal -- HIGH CONFIDENCE (100.0%),
        s_acct.odlimit AS odlimit -- HIGH CONFIDENCE (100.0%),
        CAST(s_acct.intrate AS DECIMAL(15,2)) AS intrate -- HIGH CONFIDENCE (100.0%),
        CAST(s_acct.lasttxndt AS DATE) AS lasttxndt -- HIGH CONFIDENCE (100.0%),
        s_acct.branchcd AS branchcd -- HIGH CONFIDENCE (100.0%),
        s_acct.officercd AS officercd -- HIGH CONFIDENCE (100.0%),
        CAST(s_acct.maturitydt AS DATE) AS maturitydt -- HIGH CONFIDENCE (100.0%),
        s_acct.origamt AS origamt -- HIGH CONFIDENCE (100.0%),
        s_acct.termcode AS termcode -- HIGH CONFIDENCE (100.0%),
        CURRENT_TIMESTAMP() AS transformation_timestamp,
        'ACCT' AS target_table_name,
        'optimized_confidence_based_transformation_v6' AS transformation_method,
        CASE 
            WHEN COUNT(*) > 0 THEN 'SUCCESS'
            ELSE 'NO_DATA'
        END AS transformation_status
    FROM acct s_acct
)
SELECT * FROM source_data
WHERE transformation_status = 'SUCCESS';

-- Create indexes for performance (if supported)
-- CREATE INDEX idx_acct_timestamp ON acct_transformed_v6 (transformation_timestamp);

-- Grant permissions (adjust as needed)
-- GRANT SELECT ON acct_transformed_v6 TO data_analysts;


-- ====================================
-- TARGET TABLE: ADDR
-- ====================================

-- Drop existing view if exists
DROP VIEW IF EXISTS addr_transformed_v6;

-- Create optimized transformation view
CREATE VIEW addr_transformed_v6 AS
WITH source_data AS (
    SELECT 
        s_addr.street2 AS street2 -- HIGH CONFIDENCE (100.0%),
        s_addr.city AS city -- HIGH CONFIDENCE (100.0%),
        s_addr.state AS state -- HIGH CONFIDENCE (100.0%),
        s_addr.zipcode AS zipcode -- HIGH CONFIDENCE (100.0%),
        CAST(s_addr.country AS INTEGER) AS country -- HIGH CONFIDENCE (100.0%),
        s_addr.isprimary AS isprimary -- HIGH CONFIDENCE (100.0%),
        CAST(s_addr.startdt AS DATE) AS startdt -- HIGH CONFIDENCE (100.0%),
        CAST(s_addr.enddt AS DATE) AS enddt -- HIGH CONFIDENCE (100.0%),
        s_addr.verified AS verified -- HIGH CONFIDENCE (100.0%),
        CAST(s_addr.addrnbr AS INTEGER) AS addrnbr -- HIGH CONFIDENCE (100.0%),
        CAST(s_addr.custnbr AS INTEGER) AS custnbr -- HIGH CONFIDENCE (100.0%),
        s_addr.addrtype AS addrtype -- HIGH CONFIDENCE (100.0%),
        s_addr.street1 AS street1 -- HIGH CONFIDENCE (100.0%),
        CURRENT_TIMESTAMP() AS transformation_timestamp,
        'ADDR' AS target_table_name,
        'optimized_confidence_based_transformation_v6' AS transformation_method,
        CASE 
            WHEN COUNT(*) > 0 THEN 'SUCCESS'
            ELSE 'NO_DATA'
        END AS transformation_status
    FROM addr s_addr
)
SELECT * FROM source_data
WHERE transformation_status = 'SUCCESS';

-- Create indexes for performance (if supported)
-- CREATE INDEX idx_addr_timestamp ON addr_transformed_v6 (transformation_timestamp);

-- Grant permissions (adjust as needed)
-- GRANT SELECT ON addr_transformed_v6 TO data_analysts;


-- ====================================
-- TARGET TABLE: CUST
-- ====================================

-- Drop existing view if exists
DROP VIEW IF EXISTS cust_transformed_v6;

-- Create optimized transformation view
CREATE VIEW cust_transformed_v6 AS
WITH source_data AS (
    SELECT 
        s_cust.phone AS phone -- HIGH CONFIDENCE (100.0%),
        s_cust.ssn AS ssn -- HIGH CONFIDENCE (100.0%),
        s_cust.custstat AS custstat -- HIGH CONFIDENCE (100.0%),
        CAST(s_cust.opendt AS DATE) AS opendt -- HIGH CONFIDENCE (100.0%),
        s_cust.riskrating AS riskrating -- HIGH CONFIDENCE (100.0%),
        s_cust.segmentcd AS segmentcd -- HIGH CONFIDENCE (100.0%),
        s_cust.relationmgr AS relationmgr -- HIGH CONFIDENCE (100.0%),
        s_cust.lastcontact AS lastcontact -- HIGH CONFIDENCE (100.0%),
        s_cust.prefcontact AS prefcontact -- HIGH CONFIDENCE (100.0%),
        CAST(s_cust.custnbr AS INTEGER) AS custnbr -- HIGH CONFIDENCE (100.0%),
        s_cust.title AS title -- HIGH CONFIDENCE (100.0%),
        s_cust.fname AS fname -- HIGH CONFIDENCE (100.0%),
        s_cust.lname AS lname -- HIGH CONFIDENCE (100.0%),
        CAST(s_cust.birthdt AS DATE) AS birthdt -- HIGH CONFIDENCE (100.0%),
        s_cust.email AS email -- HIGH CONFIDENCE (100.0%),
        CURRENT_TIMESTAMP() AS transformation_timestamp,
        'CUST' AS target_table_name,
        'optimized_confidence_based_transformation_v6' AS transformation_method,
        CASE 
            WHEN COUNT(*) > 0 THEN 'SUCCESS'
            ELSE 'NO_DATA'
        END AS transformation_status
    FROM cust s_cust
)
SELECT * FROM source_data
WHERE transformation_status = 'SUCCESS';

-- Create indexes for performance (if supported)
-- CREATE INDEX idx_cust_timestamp ON cust_transformed_v6 (transformation_timestamp);

-- Grant permissions (adjust as needed)
-- GRANT SELECT ON cust_transformed_v6 TO data_analysts;


-- ====================================
-- TARGET TABLE: ORG
-- ====================================

-- Drop existing view if exists
DROP VIEW IF EXISTS org_transformed_v6;

-- Create optimized transformation view
CREATE VIEW org_transformed_v6 AS
WITH source_data AS (
    SELECT 
        CAST(s_org.orgnbr AS INTEGER) AS orgnbr -- HIGH CONFIDENCE (100.0%),
        CAST(s_org.custnbr AS INTEGER) AS custnbr -- HIGH CONFIDENCE (100.0%),
        s_org.orgname AS orgname -- HIGH CONFIDENCE (100.0%),
        s_org.legalname AS legalname -- HIGH CONFIDENCE (100.0%),
        CAST(s_org.taxid AS INTEGER) AS taxid -- HIGH CONFIDENCE (100.0%),
        s_org.biztype AS biztype -- HIGH CONFIDENCE (100.0%),
        CAST(s_org.incorpdt AS DATE) AS incorpdt -- HIGH CONFIDENCE (100.0%),
        s_org.incorpstate AS incorpstate -- HIGH CONFIDENCE (100.0%),
        s_org.industry AS industry -- HIGH CONFIDENCE (100.0%),
        s_org.revenue AS revenue -- HIGH CONFIDENCE (100.0%),
        s_org.employees AS employees -- HIGH CONFIDENCE (100.0%),
        s_org.riskrating AS riskrating -- HIGH CONFIDENCE (100.0%),
        CURRENT_TIMESTAMP() AS transformation_timestamp,
        'ORG' AS target_table_name,
        'optimized_confidence_based_transformation_v6' AS transformation_method,
        CASE 
            WHEN COUNT(*) > 0 THEN 'SUCCESS'
            ELSE 'NO_DATA'
        END AS transformation_status
    FROM org s_org
)
SELECT * FROM source_data
WHERE transformation_status = 'SUCCESS';

-- Create indexes for performance (if supported)
-- CREATE INDEX idx_org_timestamp ON org_transformed_v6 (transformation_timestamp);

-- Grant permissions (adjust as needed)
-- GRANT SELECT ON org_transformed_v6 TO data_analysts;


-- ====================================
-- TARGET TABLE: PERS
-- ====================================

-- Drop existing view if exists
DROP VIEW IF EXISTS pers_transformed_v6;

-- Create optimized transformation view
CREATE VIEW pers_transformed_v6 AS
WITH source_data AS (
    SELECT 
        s_pers.employer AS employer -- HIGH CONFIDENCE (100.0%),
        s_pers.income AS income -- HIGH CONFIDENCE (100.0%),
        s_pers.networth AS networth -- HIGH CONFIDENCE (100.0%),
        s_pers.riskprofile AS riskprofile -- HIGH CONFIDENCE (100.0%),
        CAST(s_pers.persnbr AS INTEGER) AS persnbr -- HIGH CONFIDENCE (100.0%),
        CAST(s_pers.custnbr AS INTEGER) AS custnbr -- HIGH CONFIDENCE (100.0%),
        s_pers.nameprefix AS nameprefix -- HIGH CONFIDENCE (100.0%),
        s_pers.firstname AS firstname -- HIGH CONFIDENCE (100.0%),
        s_pers.lastname AS lastname -- HIGH CONFIDENCE (100.0%),
        CAST(s_pers.middlename AS INTEGER) AS middlename -- HIGH CONFIDENCE (100.0%),
        s_pers.suffix AS suffix -- HIGH CONFIDENCE (100.0%),
        CAST(s_pers.birthdate AS DATE) AS birthdate -- HIGH CONFIDENCE (100.0%),
        s_pers.gender AS gender -- HIGH CONFIDENCE (100.0%),
        s_pers.maritalstat AS maritalstat -- HIGH CONFIDENCE (100.0%),
        s_pers.occupation AS occupation -- HIGH CONFIDENCE (100.0%),
        CURRENT_TIMESTAMP() AS transformation_timestamp,
        'PERS' AS target_table_name,
        'optimized_confidence_based_transformation_v6' AS transformation_method,
        CASE 
            WHEN COUNT(*) > 0 THEN 'SUCCESS'
            ELSE 'NO_DATA'
        END AS transformation_status
    FROM pers s_pers
)
SELECT * FROM source_data
WHERE transformation_status = 'SUCCESS';

-- Create indexes for performance (if supported)
-- CREATE INDEX idx_pers_timestamp ON pers_transformed_v6 (transformation_timestamp);

-- Grant permissions (adjust as needed)
-- GRANT SELECT ON pers_transformed_v6 TO data_analysts;


-- ====================================
-- TARGET TABLE: TRANS
-- ====================================

-- Drop existing view if exists
DROP VIEW IF EXISTS trans_transformed_v6;

-- Create optimized transformation view
CREATE VIEW trans_transformed_v6 AS
WITH source_data AS (
    SELECT 
        CAST(s_trans.transnbr AS INTEGER) AS transnbr -- HIGH CONFIDENCE (100.0%),
        CAST(s_trans.acctnbr AS INTEGER) AS acctnbr -- HIGH CONFIDENCE (100.0%),
        s_trans.transtype AS transtype -- HIGH CONFIDENCE (100.0%),
        CAST(s_trans.amount AS DECIMAL(15,2)) AS amount -- HIGH CONFIDENCE (100.0%),
        CAST(s_trans.transdt AS DATE) AS transdt -- HIGH CONFIDENCE (100.0%),
        s_trans.descrip AS descrip -- HIGH CONFIDENCE (100.0%),
        CAST(s_trans.refnbr AS INTEGER) AS refnbr -- HIGH CONFIDENCE (100.0%),
        CAST(s_trans.balanceaft AS DECIMAL(15,2)) AS balanceaft -- HIGH CONFIDENCE (100.0%),
        s_trans.feeamt AS feeamt -- HIGH CONFIDENCE (100.0%),
        s_trans.branchcd AS branchcd -- HIGH CONFIDENCE (100.0%),
        CAST(s_trans.tellerid AS INTEGER) AS tellerid -- HIGH CONFIDENCE (100.0%),
        s_trans.channel AS channel -- HIGH CONFIDENCE (100.0%),
        CURRENT_TIMESTAMP() AS transformation_timestamp,
        'TRANS' AS target_table_name,
        'optimized_confidence_based_transformation_v6' AS transformation_method,
        CASE 
            WHEN COUNT(*) > 0 THEN 'SUCCESS'
            ELSE 'NO_DATA'
        END AS transformation_status
    FROM trans s_trans
)
SELECT * FROM source_data
WHERE transformation_status = 'SUCCESS';

-- Create indexes for performance (if supported)
-- CREATE INDEX idx_trans_timestamp ON trans_transformed_v6 (transformation_timestamp);

-- Grant permissions (adjust as needed)
-- GRANT SELECT ON trans_transformed_v6 TO data_analysts;


-- ====================================
-- UTILITY QUERIES
-- ====================================

-- View transformation summary
CREATE VIEW transformation_summary_v6 AS
SELECT 
    COUNT(*) as total_transformations,
    COUNT(DISTINCT target_table_name) as target_tables_count,
    MIN(transformation_timestamp) as first_transformation,
    MAX(transformation_timestamp) as last_transformation,
    'v6_optimized' as version
FROM (
    SELECT target_table_name, transformation_timestamp FROM acct_transformed_v6
    UNION ALL
    SELECT target_table_name, transformation_timestamp FROM addr_transformed_v6
    UNION ALL
    SELECT target_table_name, transformation_timestamp FROM cust_transformed_v6
    UNION ALL
    SELECT target_table_name, transformation_timestamp FROM org_transformed_v6
    UNION ALL
    SELECT target_table_name, transformation_timestamp FROM pers_transformed_v6
    UNION ALL
    SELECT target_table_name, transformation_timestamp FROM trans_transformed_v6
) all_transformations;