/* ----------------------------------- Q1 ----------------------------------- */
SELECT lob.lob, sum(claim_cost) AS sum_claim_costs 
-- use LEFT JOIN to make sure we show the sum claim costs FOR lob's that don't exist in the table line_of_business
FROM claim c LEFT JOIN line_of_business lob ON c.lob_id = lob.id  
GROUP BY lob.lob;
--lob     |sum_claim_costs|
----------+---------------+
--Auto    |         293195|
--Property|        5117489|
/* -------------------------------------------------------------------------- */


/* ----------------------------------- Q2 ----------------------------------- */
WITH claim_cost_rank AS (
SELECT *, rank() OVER (PARTITION BY lob.lob ORDER BY claim_cost DESC) AS rank_claim
FROM claim c JOIN line_of_business lob ON c.lob_id = lob.id 
)
SELECT name, lob, claim_cost, update_timestamp FROM claim_cost_rank WHERE rank_claim <= 2;

-- name   |lob     |claim_cost|update_timestamp       |
-- -------+--------+----------+-----------------------+
-- Ernie  |Auto    |    100000|2020-12-20 16:02:34.000|
-- Ernest |Auto    |    100000|2020-11-19 12:04:54.000|
-- Carol  |Property|   4560000|2020-06-01 02:44:13.000|
-- Harriet|Property|    500000|2020-11-12 04:38:56.000|


-- this shows the output of the above common table expression (CTE)
--SELECT *, rank() OVER (PARTITION BY lob.lob ORDER BY claim_cost DESC) AS rank_claim
--FROM claim c JOIN line_of_business lob ON c.lob_id = lob.id
--id|name   |claim_cost|lob_id|update_timestamp       |id|lob     |rank_claim|
----+-------+----------+------+-----------------------+--+--------+----------+
-- 5|Ernie  |    100000|     1|2020-12-20 16:02:34.000| 1|Auto    |         1|
-- 5|Ernest |    100000|     1|2020-11-19 12:04:54.000| 1|Auto    |         1|
-- 5|Ernest |     90555|     1|2020-07-31 09:23:09.000| 1|Auto    |         3|
-- 6|Fred   |      2340|     1|2020-09-22 14:02:10.000| 1|Auto    |         4|
-- 2|Bob    |       200|     1|2020-05-22 10:23:21.000| 1|Auto    |         5|
-- 6|Fred   |       100|     1|2020-12-22 12:34:59.000| 1|Auto    |         6|
-- 7|Gary   |         0|     1|2020-10-11 05:48:23.000| 1|Auto    |         7|
-- 3|Carol  |   4560000|     2|2020-06-01 02:44:13.000| 2|Property|         1|
-- 8|Harriet|    500000|     2|2020-11-12 04:38:56.000| 2|Property|         2|
-- 1|Alisha |     20000|     2|2020-01-17 15:19:20.000| 2|Property|         3|
-- 1|Alisha |     16000|     2|2020-01-15 05:13:28.000| 2|Property|         4|
-- 1|Alisha |     10000|     2|2020-01-13 15:13:28.000| 2|Property|         5|
-- 9|Ian    |     10000|     2|2020-12-05 10:05:46.000| 2|Property|         5|
-- 4|Donald |      1289|     2|2020-07-13 12:01:50.000| 2|Property|         7|
-- 2|Bob    |       200|     2|2020-07-19 00:20:16.000| 2|Property|         8|

/* -------------------------------------------------------------------------- */