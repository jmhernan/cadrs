/* create student grade history records using enrollment cohorts */
--.headers on
--.mode csv
--.output output/ghf_2017_cohort.csv

DROP VIEW IF EXISTS ghf_cohort_17;

CREATE VIEW ghf_cohort_17
AS
SELECT *
FROM hsCourses 
WHERE ResearchID IN (
SELECT DISTINCT ResearchID
FROM enr_2017cohort);

select count(*)
from (
select DISTINCT ResearchID
from ghf_cohort_17);

-- 2018 view
DROP VIEW IF EXISTS ghf_cohort_18;

CREATE VIEW ghf_cohort_18
AS
SELECT *
FROM hsCourses 
WHERE ResearchID IN (
SELECT DISTINCT ResearchID
FROM enr_2018cohort);

select count(*)
from (
select DISTINCT ResearchID
from ghf_cohort_18);

-- Tukwila HS Cohort View 
DROP VIEW IF EXISTS ghf_tukwila17;
CREATE VIEW ghf_tukwila17
AS
SELECT *
FROM hsCourses 
WHERE ResearchID IN (
SELECT DISTINCT ResearchID
FROM enr_2017cohort_tukwila);

DROP VIEW IF EXISTS ghf_tukwila18;
CREATE VIEW ghf_tukwila18
AS
SELECT *
FROM hsCourses 
WHERE ResearchID IN (
SELECT DISTINCT ResearchID
FROM enr_2018cohort_tukwila);