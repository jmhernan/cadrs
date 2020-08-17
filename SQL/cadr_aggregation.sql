/* 
Logic to Aggregate CADR Eligibility
-----------------------------------
Uses algorithm output flag, in addition it utilizes several rule based conditions 
to create VIEWS that can be used to create aggregate values

Rule Based Conditions:
English
Math
Science
Folreign Language
*/
DROP TABLE IF EXISTS cadr_pred;
.mode csv
.import /home/joseh/source/cadrs/data/svm_cadr_student_predictions_20200815.csv cadr_pred
.schema cadr_pred

DROP TABLE IF EXISTS cadr_pred;
.mode csv
.import /Users/josehernandez/Documents/eScience/projects/cadrs/data/svm_cadr_student_predictions_20200815.csv cadr_pred
.schema cadr_pred

select count(distinct ResearchID)
from cadr_pred;

select count(*)
from cadr_pred;

select ResearchID, p_CADRS, count(*) as num
from cadr_pred
group by ResearchID, p_CADRS
limit 10;

--COMBINED ENGLISH 
DROP VIEW IF EXISTS english_cadr;
CREATE VIEW english_cadr
AS
select distinct a.ResearchID, b.eng_total_creds, c.ell_cred, d.elec_cred, (b.eng_total_creds + c.ell_cred) as ell_sum_creds,
(b.eng_total_creds + d.elec_cred) as elc_sum_creds, e.ap_cred
from cadr_pred a 
left join(
    select distinct ResearchID, sum(CreditsEarned) as eng_total_creds
    from cadr_pred
    where p_CADRS like '%el_cadr%'
    group by ResearchID
) b on a.ResearchID = b.ResearchID
left join(
    select distinct ResearchID, 1 as ell_cred, sum(CreditsEarned) as total_creds
    from cadr_pred
    where p_CADRS in ('non_cadr') and CourseTitle LIKE '%ell%'
    group by ResearchID
    having total_creds >= 1
) c on a.ResearchID = c.ResearchID
left join(
    select distinct ResearchID, 1 as elec_cred, sum(CreditsEarned) as total_creds
    from cadr_pred
    where CourseTitle like '%journal%' or CourseTitle like '%debate%' or
    CourseTitle like '%public speaking%'
    group by ResearchID
) d  on a.ResearchID = d.ResearchID
left join(
    select distinct ResearchID, 1 as ap_cred
    from cadr_pred
    where p_CADRS in ('el_cadr') and  AdvancedPlacementFlag like '1' -- and GradeLevelWhenCourseTaken like '12'
) e on a.ResearchID = e.ResearchID;


--MATH 
DROP VIEW IF EXISTS math_cadr;
CREATE VIEW math_cadr
AS 
select distinct a.ResearchID, b.math_total, c.ap_cred
from cadr_pred a
left join(
    select distinct ResearchID, sum(CreditsEarned) as math_total
    from cadr_pred
    where p_CADRS like '%math_cadr%'
    group by ResearchID    
)b on a.ResearchID = b.ResearchID
left join(
    select distinct ResearchID, 1 as ap_cred, sum(CreditsEarned) as ap_total
    from cadr_pred
    where p_CADRS in ('math_cadr') and AdvancedPlacementFlag like '1' and 
    GradeLevelWhenCourseTaken in ('11','12')
    group by ResearchID
    having ap_total >= 1
) c on a.ResearchID = c.ResearchID;

--SCIENCE
DROP VIEW IF EXISTS sci_cadr;
CREATE VIEW sci_cadr
AS 
select distinct a.ResearchID, b.sci_total
from cadr_pred a
left join(
    select distinct ResearchID, sum(CreditsEarned) as sci_total
    from cadr_pred
    where p_CADRS like '%sci_cadr%'
    group by ResearchID
) b on a.ResearchID = b.ResearchID;

--SOCIAL SCIENCE
DROP VIEW IF EXISTS soc_cadr;
CREATE VIEW soc_cadr
AS 
select distinct a.ResearchID, b.soc_total
from cadr_pred a
left join(
    select distinct ResearchID, sum(CreditsEarned) as soc_total
    from cadr_pred
    where p_CADRS like '%ssh_cadr%'
    group by ResearchID
) b on a.ResearchID = b.ResearchID;

-- FOREIGN LANGUAGE (Rule based needs to capture those second level courses + AP)
DROP VIEW IF EXISTS flang_cadr;
CREATE VIEW flang_cadr
AS 
select distinct a.ResearchID, b.flang_total, c.ap_cred
from cadr_pred a 
left join(
    select distinct ResearchID, sum(CreditsEarned) as flang_total
    from cadr_pred
    where p_CADRS like '%fll%'
    group by ResearchID
) b on a.ResearchID = b.ResearchID
left join(
    select distinct ResearchID, 1 as ap_cred, sum(CreditsEarned) as ap_total
    from cadr_pred
    where p_CADRS in ('fll_cadr') and AdvancedPlacementFlag like '1'
    group by ResearchID
    having ap_total >= 1
) c on a.ResearchID = c.ResearchID;
 
-- ART
DROP VIEW IF EXISTS art_cadr;

CREATE VIEW art_cadr
AS 
select distinct a.ResearchID, b.art_total, c.math_over, d.sci_over, e.soc_over, g.flang_over
from cadr_pred a
left join(
    select distinct ResearchID, sum(CreditsEarned) as art_total
    from cadr_pred
    where p_CADRS like '%fpa_cadr%'
    group by ResearchID
) b on a.ResearchID = b.ResearchID
left join(
    select distinct ResearchID, sum(CreditsEarned)-3 as math_over
    from cadr_pred
    where p_CADRS like '%math_cadr%'
    group by ResearchID
    having math_over > 0
) c on a.ResearchID = c.ResearchID
left join(
    select distinct ResearchID, sum(CreditsEarned)-2 as sci_over
    from cadr_pred
    where p_CADRS like '%sci_cadr%' 
    group by ResearchID
    having sci_over > 0
) d on a.ResearchID = d.ResearchID
left join(
    select distinct ResearchID, sum(CreditsEarned)-3 as soc_over
    from cadr_pred
    where p_CADRS like '%ssh_cadr%'
    group by ResearchID
    having soc_over > 0
) e on a.ResearchID = e.ResearchID
left join(
    select distinct ResearchID, sum(CreditsEarned)-2 as flang_over
    from cadr_pred
    where p_CADRS like '%fll_cadr%'
    group by ResearchID
    having flang_over > 0
) g on a.ResearchID = g.ResearchID;

-- probably do the cadr flag on the views
select count(*)
from english_cadr
where eng_total_creds >= 4 or ell_sum_creds >= 4 or elc_sum_creds >= 4 or ap_cred = 1;

select count(*)
from math_cadr
where math_total >= 3 or ap_cred = 1;

select count(*)
from sci_cadr
where sci_total >=2;

select count(*)
from soc_cadr
where soc_total >=3;

select count(*)
from flang_cadr
where flang_total >= 2 or ap_cred = 1;

select * from art_cadr limit 10;

-- CADR AGGREGATOR
DROP VIEW IF EXISTS agg_cadr;
CREATE VIEW agg_cadr
AS
select distinct a.ResearchID, b.art_cadr_v, c.math_cadr_v, 
d.eng_cadr_v, e.sci_cadr_v, f.soc_cadr_v, g.flang_cadr_v
from cadr_pred a
left join(
select ResearchID, 
COALESCE(art_total,0), COALESCE(math_over,0), COALESCE(soc_over,0), COALESCE(flang_over,0), COALESCE(sci_over,0), 
(COALESCE(art_total,0) + COALESCE(math_over,0)) as art_m,
(COALESCE(art_total,0) + COALESCE(soc_over,0)) as art_ss, 
(COALESCE(art_total,0) + COALESCE(flang_over,0)) as art_fl,
(COALESCE(art_total,0) + COALESCE(sci_over,0)) as art_sci,
1 as art_cadr_v
from art_cadr
where art_total >=1 or art_m >= 1 or art_ss >= 1 or art_fl >= 1 or art_sci >= 1) b
on a.ResearchID = b.ResearchID
left join(
select ResearchID, 1 as math_cadr_v
from math_cadr
where math_total >= 3 or ap_cred = 1) c
on a.ResearchID = c.ResearchID
left join(
select ResearchID, 1 as eng_cadr_v
from english_cadr
where eng_total_creds >= 4 or ell_sum_creds >= 4 or elc_sum_creds >= 4 or ap_cred = 1) d
on a.ResearchID = d.ResearchID
left join(
select ResearchID, 1 as sci_cadr_v
from sci_cadr
where sci_total >=2) e 
on a.ResearchID = e.ResearchID
left join(
select ResearchID, 1 as soc_cadr_v
from soc_cadr
where soc_total >=3) f 
on a.ResearchID = f.ResearchID
left join(
select ResearchID, 1 as flang_cadr_v
from flang_cadr
where flang_total >= 2 or ap_cred = 1) g
on a.ResearchID=g.ResearchID;

select count(*)
from agg_cadr
where eng_cadr_v = 1 and math_cadr_v = 1 and 
sci_cadr_v = 1 and flang_cadr_v = 1 and art_cadr_v = 1 and soc_cadr_v = 1; 

-- ADD NSC FLAG
DROP VIEW IF EXISTS cadr_val;
CREATE VIEW cadr_val
AS
select a.*, b.nsc_4yr, c.CADR_eligible
from agg_cadr a
left join(
    Select ResearchID, 1 as nsc_4yr
    from nsc_cohort_17
) b on a.ResearchID = b.ResearchID
left join (
    select ResearchID, 1 as CADR_eligible
    from agg_cadr
    where eng_cadr_v = 1 and math_cadr_v = 1 and 
    sci_cadr_v = 1 and flang_cadr_v = 1 and art_cadr_v = 1 and soc_cadr_v = 1
) c on a.ResearchID = c.ResearchID;


select count(*)
from cadr_val;
where nsc_4yr = 1;

-- Test Aggregator with credits needed
-- We MINGHT USE THIS LATER ON

DROP VIEW IF EXISTS agg_cadr_test;
CREATE VIEW agg_cadr_test
AS
select
CASE 
    WHEN a.eng_cadr_v ISNULL THEN b.eng_total_creds ELSE a.eng_cadr_v
END
AS eng_val
from agg_cadr a
left join (
    select ResearchID, eng_total_creds
    from english_cadr
    where eng_total_creds < 4 
) b 
on a.ResearchID = b.ResearchID;

-- Check folks that have courses all throughout their HS years 
DROP VIEW IF EXISTS  cadr_test_agg;
CREATE VIEW cadr_test_agg
AS
SELECT a.*, 
        CASE WHEN b.DistinctGradeLevelCount = 4 THEN 1 ELSE 0 END AS CompleteHSRecords 
FROM cadr_val a
LEFT JOIN(
    SELECT
		ResearchID,
		COUNT(DISTINCT GradeLevelWhenCourseTaken) AS DistinctGradeLevelCount
	FROM cadr_pred
	WHERE 
		GradeLevelWhenCourseTaken IN (9, 10, 11, 12)
	GROUP BY ResearchID
) b
on a.ResearchID = b.ResearchID;

-- out of 170 students that have any HS grades in SGH, 137 have complete transcripts
SELECT
	SUM(CompleteHSRecords) AS CompleteHSRecords,
	COUNT(*) 
FROM cadr_test_agg;

-- More robust
DROP VIEW IF EXISTS  min_credits_all;
CREATE VIEW min_credits_all
AS
SELECT  ResearchID,
		GradeLevelWhenCourseTaken,
		SUM(CreditsAttempted) AS CreditsAttemptedTotal,
		CASE WHEN SUM(CreditsAttempted) >= 6.0 THEN 1 ELSE 0 END AS HasMinCredits
	FROM cadr_pred
	WHERE 
		GradeLevelWhenCourseTaken IN (9, 10, 11, 12)
	GROUP BY
		ResearchID,
		GradeLevelWhenCourseTaken;

DROP VIEW IF EXISTS  cadr_test_agg_robust;
CREATE VIEW cadr_test_agg_robust
AS
SELECT  a.*,
        b.CompleteHSRecordsRobust
FROM cadr_test_agg a
LEFT JOIN(
    SELECT  ResearchID,
            CASE WHEN(COUNT(DISTINCT GradeLevelWhenCourseTaken) = 4 AND 
            COUNT(DISTINCT GradeLevelWhenCourseTaken) = SUM(HasMinCredits)) THEN 1 ELSE 0 END AS CompleteHSRecordsRobust
    FROM min_credits_all
    GROUP BY ResearchID
) b
on a.ResearchID = b.ResearchID;

-- out of 170 students that have any HS grades in SGH, 63 have complete transcripts when counting minimum credits
SELECT
	SUM(CompleteHSRecordsRobust) AS CompleteHSRecordsRobust,
	COUNT(*) 
FROM cadr_test_agg_robust;

SELECT 
    SUM(CADR_eligible) AS CADR_eligibility,
    SUM(nsc_4yr) AS nsc_eligibility,
    COUNT(*)
FROM cadr_test_agg_robust
WHERE CompleteHSRecords = 1;

SELECT 
    SUM(CADR_eligible) AS CADR_eligibility,
    SUM(nsc_4yr) AS nsc_eligibility,
    COUNT(*)
FROM cadr_test_agg_robust
WHERE CompleteHSRecordsRobust = 1;

SELECT 
    SUM(CADR_eligible) AS CADR_eligibility,
    SUM(nsc_4yr) AS nsc_eligibility,
    COUNT(*)
FROM cadr_test_agg_robust
WHERE CompleteHSRecordsRobust = 1 and nsc_4yr = 1;

/* Aggregation by district */

SELECT a.DistrictCode, sum(num), sum(total)
FROM (
    SELECT COUNT(distinct en.ResearchID) num, DistrictCode
    FROM enr_2017cohort en
    INNER JOIN (
    SELECT *
    FROM cadr_test_agg_robust
    WHERE CADR_eligible = 1
    ) c on en.ResearchID = c.ResearchID
GROUP BY DistrictCode) a
INNER JOIN(
    SELECT COUNT(distinct en.ResearchID) total, DistrictCode
    FROM enr_2017cohort en
    GROUP BY DistrictCode
) b on a.DistrictCode = b.DistrictCode
GROUP BY a.DistrictCode;

-- Using the new table 
SELECT count(*) 
FROM cadr_test_agg_robust;

SELECT count(*)
FROM enr_2017cohort;

SELECT * 
FROM agg_cadr
LIMIT 10;

DROP VIEW IF EXISTS cadr_val;
CREATE VIEW cadr_val
AS
select a.*, c.CADR_eligible
from agg_cadr a
left join (
    select ResearchID, 1 as CADR_eligible
    from agg_cadr
    where eng_cadr_v = 1 and math_cadr_v = 1 and 
    sci_cadr_v = 1 and flang_cadr_v = 1 and art_cadr_v = 1 and soc_cadr_v = 1
) c on a.ResearchID = c.ResearchID;

SELECT COUNT(*)
FROM cadr_val;

DROP VIEW IF EXISTS  cadr_crs_agg;
CREATE VIEW cadr_crs_agg
AS
SELECT a.*, 
        CASE WHEN b.DistinctGradeLevelCount = 4 THEN 1 ELSE 0 END AS CompleteHSRecords 
FROM cadr_val a
LEFT JOIN(
    SELECT
		ResearchID,
		COUNT(DISTINCT GradeLevelWhenCourseTaken) AS DistinctGradeLevelCount
	FROM cadr_pred
	WHERE 
		GradeLevelWhenCourseTaken IN (9, 10, 11, 12)
	GROUP BY ResearchID
) b
on a.ResearchID = b.ResearchID;

SELECT *
FROM cadr_crs_agg
LIMIT 10;

DROP VIEW IF EXISTS  cadr_district_table;
CREATE VIEW cadr_district_table
AS
SELECT b.DistrictCode, a.*
FROM cadr_crs_agg a
LEFT JOIN(
SELECT distinct ResearchID, DistrictCode
FROM enr_2017cohort
) b ON a.ResearchID = b.ResearchID;

SELECT *
FROM cadr_district_table
WHERE DistrictCode = 17403
LIMIT 10;

SELECT DistrictCode, COUNT(distinct ResearchID) num
FROM cadr_district_table
WHERE CADR_eligible = 1
GROUP BY DistrictCode;

SELECT DistrictCode, COUNT(distinct ResearchID) total 
FROM cadr_district_table
GROUP BY DistrictCode;
