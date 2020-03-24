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

--COMBINED ENGLISH 
CREATE VIEW english_cadr
AS
select distinct a.ResearchID, b.eng_total_creds, c.ell_cred, d.elec_cred, (b.eng_total_creds + c.ell_cred) as ell_sum_creds,
(b.eng_total_creds + d.elec_cred) as elc_sum_creds, e.ap_cred 
from cadr_pred a 
left join(
    select distinct ResearchID, sum(CreditsEarned) as eng_total_creds
    from cadr_pred
    where p_CADRS like '%1%' and content_area in ('English Language and Literature')
    group by ResearchID
) b on a.ResearchID = b.ResearchID
left join(
    select distinct ResearchID, 1 as ell_cred, sum(CreditsEarned) as total_creds
    from cadr_pred
    where content_area in ('English Language and Literature') and CourseTitle LIKE '%ell%'
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
    where content_area in ('English Language and Literature') and  AdvancedPlacementFlag like '1' -- and GradeLevelWhenCourseTaken like '12'
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
    where p_CADRS like '%1%' and content_area in ('Computer and Information Sciences', 'Mathematics')
    group by ResearchID    
)b on a.ResearchID = b.ResearchID
left join(
    select distinct ResearchID, 1 as ap_cred, sum(CreditsEarned) as ap_total
    from cadr_pred
    where content_area in ('Mathematics') and AdvancedPlacementFlag like '1' and 
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
    where p_CADRS like '%1%' and content_area in ('Life and Physical Sciences')
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
    where p_CADRS like '%1%' and content_area in ('Social Sciences and History')
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
    where p_CADRS like '%1%' and content_area in ('Foreign Language and Literature')
    group by ResearchID
) b on a.ResearchID = b.ResearchID
left join(
    select distinct ResearchID, 1 as ap_cred, sum(CreditsEarned) as ap_total
    from cadr_pred
    where content_area in ('Foreign Language and Literature') and AdvancedPlacementFlag like '1'
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
    where p_CADRS like '%1%' and content_area in ('Fine and Performing Arts')
    group by ResearchID
) b on a.ResearchID = b.ResearchID
left join(
    select distinct ResearchID, sum(CreditsEarned)-3 as math_over
    from cadr_pred
    where p_CADRS like '%1%' and content_area in ('Computer and Information Sciences', 'Mathematics')
    group by ResearchID
    having math_over > 0
) c on a.ResearchID = c.ResearchID
left join(
    select distinct ResearchID, sum(CreditsEarned)-2 as sci_over
    from cadr_pred
    where p_CADRS like '%1%' and content_area in ('Life and Physical Sciences')
    group by ResearchID
    having sci_over > 0
) d on a.ResearchID = d.ResearchID
left join(
    select distinct ResearchID, sum(CreditsEarned)-3 as soc_over
    from cadr_pred
    where p_CADRS like '%1%' and content_area in ('Social Sciences and History')
    group by ResearchID
    having soc_over > 0
) e on a.ResearchID = e.ResearchID
left join(
    select distinct ResearchID, sum(CreditsEarned)-2 as flang_over
    from cadr_pred
    where p_CADRS like '%1%' and content_area in ('Foreign Language and Literature')
    group by ResearchID
    having flang_over > 0
) g on a.ResearchID = g.ResearchID;

-- probably do the cadr flag on the views