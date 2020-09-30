/* Make a table */
use hw4;

/* Compare the running time between index scan and full table scan at selectivity 50% */
SELECT CURRENT_TIMESTAMP() as start_time_indext_scan_1;
SELECT SUM(a)
FROM TEST
WHERE a <= 200000;
SELECT CURRENT_TIMESTAMP() as finish_time_indext_scan_1;

SELECT CURRENT_TIMESTAMP() as start_time_indext_scan_10;
SELECT SUM(a)
FROM TEST
WHERE a <= 2000000;
SELECT CURRENT_TIMESTAMP() as finish_time_indext_scan_10;

SELECT CURRENT_TIMESTAMP() as start_time_indext_scan_20;
SELECT SUM(a)
FROM TEST
WHERE a <= 4000000;
SELECT CURRENT_TIMESTAMP() as finish_time_indext_scan_20;

SELECT CURRENT_TIMESTAMP() as start_time_indext_scan_30;
SELECT SUM(a)
FROM TEST
WHERE a <= 6000000;
SELECT CURRENT_TIMESTAMP() as finish_time_indext_scan_30;

SELECT CURRENT_TIMESTAMP() as start_time_indext_scan_40;
SELECT SUM(a)
FROM TEST
WHERE a <= 8000000;
SELECT CURRENT_TIMESTAMP() as finish_time_indext_scan_40;

SELECT CURRENT_TIMESTAMP() as start_time_indext_scan_50;
SELECT SUM(a)
FROM TEST
WHERE a <= 10000000;
SELECT CURRENT_TIMESTAMP() as finish_time_indext_scan_50;

SELECT CURRENT_TIMESTAMP() as start_time_indext_scan_60;
SELECT SUM(a)
FROM TEST
WHERE a <= 12000000;
SELECT CURRENT_TIMESTAMP() as finish_time_indext_scan_60;

SELECT CURRENT_TIMESTAMP() as start_time_indext_scan_70;
SELECT SUM(a)
FROM TEST
WHERE a <= 14000000;
SELECT CURRENT_TIMESTAMP() as finish_time_indext_scan_70;

SELECT CURRENT_TIMESTAMP() as start_time_indext_scan_80;
SELECT SUM(a)
FROM TEST
WHERE a <= 16000000;
SELECT CURRENT_TIMESTAMP() as finish_time_indext_scan_80;

SELECT CURRENT_TIMESTAMP() as start_time_indext_scan_90;
SELECT SUM(a)
FROM TEST
WHERE a <= 18000000;
SELECT CURRENT_TIMESTAMP() as finish_time_indext_scan_90;

SELECT CURRENT_TIMESTAMP() as start_time_indext_scan_100;
SELECT SUM(a)
FROM TEST
WHERE a <= 20000000;
SELECT CURRENT_TIMESTAMP() as finish_time_indext_scan_100;


SELECT CURRENT_TIMESTAMP() as start_time_indext_scan_1;
SELECT SUM(b)
FROM TEST
WHERE b <= 200000;
SELECT CURRENT_TIMESTAMP() as finish_time_indext_scan_1;

SELECT CURRENT_TIMESTAMP() as start_time_indext_scan_10;
SELECT SUM(b)
FROM TEST
WHERE b <= 2000000;
SELECT CURRENT_TIMESTAMP() as finish_time_indext_scan_10;

SELECT CURRENT_TIMESTAMP() as start_time_indext_scan_20;
SELECT SUM(b)
FROM TEST
WHERE b <= 4000000;
SELECT CURRENT_TIMESTAMP() as finish_time_indext_scan_20;

SELECT CURRENT_TIMESTAMP() as start_time_indext_scan_30;
SELECT SUM(b)
FROM TEST
WHERE b <= 6000000;
SELECT CURRENT_TIMESTAMP() as finish_time_indext_scan_30;

SELECT CURRENT_TIMESTAMP() as start_time_indext_scan_40;
SELECT SUM(b)
FROM TEST
WHERE b <= 8000000;
SELECT CURRENT_TIMESTAMP() as finish_time_indext_scan_40;

SELECT CURRENT_TIMESTAMP() as start_time_indext_scan_50;
SELECT SUM(b)
FROM TEST
WHERE b <= 10000000;
SELECT CURRENT_TIMESTAMP() as finish_time_indext_scan_50;

SELECT CURRENT_TIMESTAMP() as start_time_indext_scan_60;
SELECT SUM(b)
FROM TEST
WHERE b <= 12000000;
SELECT CURRENT_TIMESTAMP() as finish_time_indext_scan_60;

SELECT CURRENT_TIMESTAMP() as start_time_indext_scan_70;
SELECT SUM(b)
FROM TEST
WHERE b <= 14000000;
SELECT CURRENT_TIMESTAMP() as finish_time_indext_scan_70;

SELECT CURRENT_TIMESTAMP() as start_time_indext_scan_80;
SELECT SUM(b)
FROM TEST
WHERE b <= 16000000;
SELECT CURRENT_TIMESTAMP() as finish_time_indext_scan_80;

SELECT CURRENT_TIMESTAMP() as start_time_indext_scan_90;
SELECT SUM(b)
FROM TEST
WHERE b <= 18000000;
SELECT CURRENT_TIMESTAMP() as finish_time_indext_scan_90;

SELECT CURRENT_TIMESTAMP() as start_time_indext_scan_100;
SELECT SUM(b)
FROM TEST
WHERE b <= 20000000;
SELECT CURRENT_TIMESTAMP() as finish_time_indext_scan_100;


