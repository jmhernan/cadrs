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
DROP TABLE IF EXISTS cadr_pred_tuk;
.mode csv
.import /home/joseh/source/cadrs/data/svm_cadr_student_predictions_tukwila_202008.csv cadr_pred_tuk
.schema cadr_pred_tuk

DROP TABLE IF EXISTS cadr_pred_tuk;
.mode csv
.import /Users/josehernandez/Documents/eScience/projects/cadrs/data/svm_cadr_student_predictions_renton_20200817.csv cadr_pred_tuk
.schema cadr_pred_tuk

select count(distinct ResearchID)
from cadr_pred_tuk;

select count(*)
from cadr_pred_tuk;

select ResearchID, p_CADRS, count(*) as num
from cadr_pred_tuk
group by ResearchID, p_CADRS
limit 10;

--COMBINED ENGLISH 
DROP VIEW IF EXISTS english_cadr_tuk;
CREATE VIEW english_cadr_tuk
AS
select distinct a.ResearchID, b.eng_total_creds, c.ell_cred, d.elec_cred, (b.eng_total_creds + c.ell_cred) as ell_sum_creds,
(b.eng_total_creds + d.elec_cred) as elc_sum_creds, e.ap_cred
from cadr_pred_tuk a 
left join(
    select distinct ResearchID, sum(CreditsEarned) as eng_total_creds
    from cadr_pred_tuk
    where p_CADRS like '%el_cadr%'
    group by ResearchID
) b on a.ResearchID = b.ResearchID
left join(
    select distinct ResearchID, 1 as ell_cred, sum(CreditsEarned) as total_creds
    from cadr_pred_tuk
    where p_CADRS in ('non_cadr') and CourseTitle LIKE '%ell%'
    group by ResearchID
    having total_creds >= 1
) c on a.ResearchID = c.ResearchID
left join(
    select distinct ResearchID, 1 as elec_cred, sum(CreditsEarned) as total_creds
    from cadr_pred_tuk
    where CourseTitle like '%journal%' or CourseTitle like '%debate%' or
    CourseTitle like '%public speaking%'
    group by ResearchID
) d  on a.ResearchID = d.ResearchID
left join(
    select distinct ResearchID, 1 as ap_cred
    from cadr_pred_tuk
    where p_CADRS in ('el_cadr') and  AdvancedPlacementFlag like '1' -- and GradeLevelWhenCourseTaken like '12'
) e on a.ResearchID = e.ResearchID;


--MATH 
DROP VIEW IF EXISTS math_cadr_tuk;
CREATE VIEW math_cadr_tuk
AS 
select distinct a.ResearchID, b.math_total, c.ap_cred
from cadr_pred_tuk a
left join(
    select distinct ResearchID, sum(CreditsEarned) as math_total
    from cadr_pred_tuk
    where p_CADRS like '%math_cadr%'
    group by ResearchID    
)b on a.ResearchID = b.ResearchID
left join(
    select distinct ResearchID, 1 as ap_cred, sum(CreditsEarned) as ap_total
    from cadr_pred_tuk
    where p_CADRS in ('math_cadr') and AdvancedPlacementFlag like '1' and 
    GradeLevelWhenCourseTaken in ('11','12')
    group by ResearchID
    having ap_total >= 1
) c on a.ResearchID = c.ResearchID;

--SCIENCE
DROP VIEW IF EXISTS sci_cadr_tuk;
CREATE VIEW sci_cadr_tuk
AS 
select distinct a.ResearchID, b.sci_total
from cadr_pred_tuk a
left join(
    select distinct ResearchID, sum(CreditsEarned) as sci_total
    from cadr_pred_tuk
    where p_CADRS like '%sci_cadr%'
    group by ResearchID
) b on a.ResearchID = b.ResearchID;

--SOCIAL SCIENCE
DROP VIEW IF EXISTS soc_cadr_tuk;
CREATE VIEW soc_cadr_tuk
AS 
select distinct a.ResearchID, b.soc_total
from cadr_pred_tuk a
left join(
    select distinct ResearchID, sum(CreditsEarned) as soc_total
    from cadr_pred_tuk
    where p_CADRS like '%ssh_cadr%'
    group by ResearchID
) b on a.ResearchID = b.ResearchID;

-- FOREIGN LANGUAGE (Rule based needs to capture those second level courses + AP)
DROP VIEW IF EXISTS flang_cadr_tuk;
CREATE VIEW flang_cadr_tuk
AS 
select distinct a.ResearchID, b.flang_total, c.ap_cred
from cadr_pred_tuk a 
left join(
    select distinct ResearchID, sum(CreditsEarned) as flang_total
    from cadr_pred_tuk
    where p_CADRS like '%fll%'
    group by ResearchID
) b on a.ResearchID = b.ResearchID
left join(
    select distinct ResearchID, 1 as ap_cred, sum(CreditsEarned) as ap_total
    from cadr_pred_tuk
    where p_CADRS in ('fll_cadr') and AdvancedPlacementFlag like '1'
    group by ResearchID
    having ap_total >= 1
) c on a.ResearchID = c.ResearchID;
 
-- ART
DROP VIEW IF EXISTS art_cadr_tuk;

CREATE VIEW art_cadr_tuk
AS 
select distinct a.ResearchID, b.art_total, c.math_over, d.sci_over, e.soc_over, g.flang_over
from cadr_pred_tuk a
left join(
    select distinct ResearchID, sum(CreditsEarned) as art_total
    from cadr_pred_tuk
    where p_CADRS like '%fpa_cadr%'
    group by ResearchID
) b on a.ResearchID = b.ResearchID
left join(
    select distinct ResearchID, sum(CreditsEarned)-3 as math_over
    from cadr_pred_tuk
    where p_CADRS like '%math_cadr%'
    group by ResearchID
    having math_over > 0
) c on a.ResearchID = c.ResearchID
left join(
    select distinct ResearchID, sum(CreditsEarned)-2 as sci_over
    from cadr_pred_tuk
    where p_CADRS like '%sci_cadr%' 
    group by ResearchID
    having sci_over > 0
) d on a.ResearchID = d.ResearchID
left join(
    select distinct ResearchID, sum(CreditsEarned)-3 as soc_over
    from cadr_pred_tuk
    where p_CADRS like '%ssh_cadr%'
    group by ResearchID
    having soc_over > 0
) e on a.ResearchID = e.ResearchID
left join(
    select distinct ResearchID, sum(CreditsEarned)-2 as flang_over
    from cadr_pred_tuk
    where p_CADRS like '%fll_cadr%'
    group by ResearchID
    having flang_over > 0
) g on a.ResearchID = g.ResearchID;

-- probably do the cadr flag on the views
select count(*)
from english_cadr_tuk
where eng_total_creds >= 4 or ell_sum_creds >= 4 or elc_sum_creds >= 4 or ap_cred = 1;

select count(*)
from math_cadr_tuk
where math_total >= 3 or ap_cred = 1;

select count(*)
from sci_cadr_tuk
where sci_total >=2;

select count(*)
from soc_cadr_tuk
where soc_total >=3;

select count(*)
from flang_cadr_tuk
where flang_total >= 2 or ap_cred = 1;

select * from art_cadr_tuk limit 10;

-- CADR AGGREGATOR
DROP VIEW IF EXISTS agg_cadr_tuk;
CREATE VIEW agg_cadr_tuk
AS
select distinct a.ResearchID, b.art_cadr_v, c.math_cadr_v, 
d.eng_cadr_v, e.sci_cadr_v, f.soc_cadr_v, g.flang_cadr_v
from cadr_pred_tuk a
left join(
select ResearchID, 
COALESCE(art_total,0), COALESCE(math_over,0), COALESCE(soc_over,0), COALESCE(flang_over,0), COALESCE(sci_over,0), 
(COALESCE(art_total,0) + COALESCE(math_over,0)) as art_m,
(COALESCE(art_total,0) + COALESCE(soc_over,0)) as art_ss, 
(COALESCE(art_total,0) + COALESCE(flang_over,0)) as art_fl,
(COALESCE(art_total,0) + COALESCE(sci_over,0)) as art_sci,
1 as art_cadr_v
from art_cadr_tuk
where art_total >=1 or art_m >= 1 or art_ss >= 1 or art_fl >= 1 or art_sci >= 1) b
on a.ResearchID = b.ResearchID
left join(
select ResearchID, 1 as math_cadr_v
from math_cadr_tuk
where math_total >= 3 or ap_cred = 1) c
on a.ResearchID = c.ResearchID
left join(
select ResearchID, 1 as eng_cadr_v
from english_cadr_tuk
where eng_total_creds >= 4 or ell_sum_creds >= 4 or elc_sum_creds >= 4 or ap_cred = 1) d
on a.ResearchID = d.ResearchID
left join(
select ResearchID, 1 as sci_cadr_v
from sci_cadr_tuk
where sci_total >=2) e 
on a.ResearchID = e.ResearchID
left join(
select ResearchID, 1 as soc_cadr_v
from soc_cadr_tuk
where soc_total >=3) f 
on a.ResearchID = f.ResearchID
left join(
select ResearchID, 1 as flang_cadr_v
from flang_cadr_tuk
where flang_total >= 2 or ap_cred = 1) g
on a.ResearchID=g.ResearchID;

select count(*)
from agg_cadr_tuk
where eng_cadr_v = 1 and math_cadr_v = 1 and 
sci_cadr_v = 1 and flang_cadr_v = 1 and art_cadr_v = 1 and soc_cadr_v = 1; 

-- ADD NSC FLAG
DROP VIEW IF EXISTS cadr_tuk_val;
CREATE VIEW cadr_tuk_val
AS
select a.*, b.nsc_4yr, c.CADR_eligible
from agg_cadr_tuk a
left join(
    Select ResearchID, 1 as nsc_4yr
    from nsc_tuk_cohort_17
) b on a.ResearchID = b.ResearchID
left join (
    select ResearchID, 1 as CADR_eligible
    from agg_cadr_tuk
    where eng_cadr_v = 1 and math_cadr_v = 1 and 
    sci_cadr_v = 1 and flang_cadr_v = 1 and art_cadr_v = 1 and soc_cadr_v = 1
) c on a.ResearchID = c.ResearchID;


select count(*)
from cadr_tuk_val
where nsc_4yr = 1;
-- Test Aggregator with credits needed
DROP VIEW IF EXISTS agg_cadr_tuk_test;
CREATE VIEW agg_cadr_tuk_test
AS
select
CASE 
    WHEN a.eng_cadr_v ISNULL THEN b.eng_total_creds ELSE a.eng_cadr_v
END
AS eng_val
from agg_cadr_tuk a
left join (
    select ResearchID, eng_total_creds
    from english_cadr_tuk
    where eng_total_creds < 4 
) b 
on a.ResearchID = b.ResearchID;

-- Check folks that have courses all throughout their HS years 
DROP VIEW IF EXISTS  Tukwila_test_agg;
CREATE VIEW Tukwila_test_agg
AS
SELECT a.*, 
        CASE WHEN b.DistinctGradeLevelCount = 4 THEN 1 ELSE 0 END AS CompleteHSRecords 
FROM cadr_tuk_val a
LEFT JOIN(
    SELECT
		ResearchID,
		COUNT(DISTINCT GradeLevelWhenCourseTaken) AS DistinctGradeLevelCount
	FROM cadr_pred_tuk
	WHERE 
		GradeLevelWhenCourseTaken IN (9, 10, 11, 12)
	GROUP BY ResearchID
) b
on a.ResearchID = b.ResearchID;

-- out of 170 students that have any HS grades in SGH, 137 have complete transcripts
SELECT
	SUM(CompleteHSRecords) AS CompleteHSRecords,
	COUNT(*) 
FROM Tukwila_test_agg;

-- More robust
DROP VIEW IF EXISTS  min_credits;
CREATE VIEW min_credits
AS
SELECT  ResearchID,
		GradeLevelWhenCourseTaken,
		SUM(CreditsAttempted) AS CreditsAttemptedTotal,
		CASE WHEN SUM(CreditsAttempted) >= 6.0 THEN 1 ELSE 0 END AS HasMinCredits
	FROM cadr_pred_tuk
	WHERE 
		GradeLevelWhenCourseTaken IN (9, 10, 11, 12)
	GROUP BY
		ResearchID,
		GradeLevelWhenCourseTaken;

DROP VIEW IF EXISTS  Tukwila_test_agg_robust;
CREATE VIEW Tukwila_test_agg_robust
AS
SELECT  a.*,
        b.CompleteHSRecordsRobust
FROM Tukwila_test_agg a
LEFT JOIN(
    SELECT  ResearchID,
            CASE WHEN(COUNT(DISTINCT GradeLevelWhenCourseTaken) = 4 AND 
            COUNT(DISTINCT GradeLevelWhenCourseTaken) = SUM(HasMinCredits)) THEN 1 ELSE 0 END AS CompleteHSRecordsRobust
    FROM min_credits
    GROUP BY ResearchID
) b
on a.ResearchID = b.ResearchID;

-- out of 170 students that have any HS grades in SGH, 63 have complete transcripts when counting minimum credits
SELECT
	SUM(CompleteHSRecordsRobust) AS CompleteHSRecordsRobust,
	COUNT(*) 
FROM Tukwila_test_agg_robust;

SELECT 
    SUM(CADR_eligible) AS CADR_eligibility,
    SUM(nsc_4yr) AS nsc_eligibility,
    COUNT(*)
FROM Tukwila_test_agg_robust
WHERE CompleteHSRecords = 1;

SELECT 
    SUM(CADR_eligible) AS CADR_eligibility,
    SUM(nsc_4yr) AS nsc_eligibility,
    COUNT(*)
FROM Tukwila_test_agg_robust
WHERE CompleteHSRecordsRobust = 1;

SELECT 
    SUM(CADR_eligible) AS CADR_eligibility,
    SUM(nsc_4yr) AS nsc_eligibility,
    COUNT(*)
FROM Tukwila_test_agg_robust
WHERE CompleteHSRecordsRobust = 1 and nsc_4yr = 1;