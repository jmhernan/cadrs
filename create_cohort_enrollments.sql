/* Create cohort views */

DROP VIEW IF EXISTS enr_2017cohort;

CREATE VIEW enr_2017cohort
AS
SELECT *
FROM enrollment enr
JOIN Dim_School sch
    ON enr.SchoolCode = sch.SchoolCode
    AND enr.ReportSchoolYear = sch.AcademicYear
WHERE enr.GradeLevelSortOrder = 15 AND enr.GradReqYear = 2017 AND enr.dGraduate = 1 AND sch.dRoadMapRegionFlag = 1;

select count(*) from enr_2017cohort;
-- 7,542 rows

DROP VIEW IF EXISTS enr_2018cohort;

CREATE VIEW enr_2018cohort
AS
SELECT *
FROM enrollment enr
JOIN Dim_School sch
    ON enr.SchoolCode = sch.SchoolCode
    AND enr.ReportSchoolYear = sch.AcademicYear
WHERE  enr.GradeLevelSortOrder = 15 AND enr.GradReqYear = 2018 AND enr.dGraduate = 1 AND sch.dRoadMapRegionFlag = 1;

select count(*) from enr_2018cohort;
-- 7,542 rows

/* To use with Tukwila testing given their manageable enrollment size and CADR completion */
DROP VIEW IF EXISTS enr_2017cohort_tukwila;

CREATE VIEW enr_2017cohort_tukwila
AS
SELECT *
FROM enrollment enr
JOIN Dim_School sch
    ON enr.SchoolCode = sch.SchoolCode
    AND enr.ReportSchoolYear = sch.AcademicYear
WHERE  enr.GradeLevelSortOrder = 15 AND enr.GradReqYear = 2017 AND enr.dGraduate = 1 AND sch.dRoadMapRegionFlag = 1
    AND enr.DistrictCode = 17406;

-- select count(*) from enr_2017cohort_tukwila;
-- 170

DROP VIEW IF EXISTS enr_2018cohort_tukwila;

CREATE VIEW enr_2018cohort_tukwila
AS
SELECT *
FROM enrollment enr
JOIN Dim_School sch
    ON enr.SchoolCode = sch.SchoolCode
    AND enr.ReportSchoolYear = sch.AcademicYear
WHERE  enr.GradeLevelSortOrder = 15 AND enr.GradReqYear = 2018 AND enr.dGraduate = 1 AND sch.dRoadMapRegionFlag = 1
    AND enr.DistrictCode = 17406;

-- select count(*) from enr_2018cohort_tukwila;
-- 171
