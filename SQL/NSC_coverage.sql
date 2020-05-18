/* Create NSC cohorts for validation */

DROP VIEW IF EXISTS nsc_cohort_17;
CREATE VIEW nsc_cohort_17
AS
select *
from postSecDems
where EnrollmentBegin >= '2017-06-01' and EnrollmentBegin <= '2017-12-31' and 
v2year4year = 4 and ResearchID in (
    SELECT DISTINCT ResearchID
    FROM enr_2017cohort 
);


DROP VIEW IF EXISTS nsc_tuk_cohort_17;
CREATE VIEW nsc_tuk_cohort_17
AS
select *
from postSecDems
where EnrollmentBegin >= '2017-06-01' and EnrollmentBegin <= '2017-12-31' and 
v2year4year = 4 and DistrictCode = 17406 and ResearchID in (
    SELECT DISTINCT ResearchID
    FROM enr_2017cohort 
);

/* NSC COVERAGE FOR 2017 IS PROMISING! 
    NSC COVERAGE FOR 2018 GRAD CLASS IS QUESTIONABLE
    2018 NSC ENROLLMENTS BETWEEN 06-2018 AND 12-2018 = 28 */

/*
DROP VIEW IF EXISTS nsc_cohort_18;
CREATE VIEW nsc_cohort_18
AS
select *
from postSecDems
where EnrollmentBegin >= '2018-06-01' and EnrollmentBegin <= '2018-12-31' and 
v2year4year = 4 and ResearchID in (
    SELECT DISTINCT ResearchID
    FROM enr_2018cohort 
);


DROP VIEW IF EXISTS nsc_tuk_cohort_18;
CREATE VIEW nsc_tuk_cohort_18
AS
select *
from postSecDems
where EnrollmentBegin >= '2018-06-01' and EnrollmentBegin <= '2018-12-31' and 
v2year4year = 4 and DistrictCode = 17406 and ResearchID in (
    SELECT DISTINCT ResearchID
    FROM enr_2018cohort 
);
*/

