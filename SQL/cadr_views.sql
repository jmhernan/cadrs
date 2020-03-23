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
    select distinct ResearchID, 1 as ap_cred
    from cadr_pred
    where content_area in ('Mathematics') and AdvancedPlacementFlag like '1' and GradeLevelWhenCourseTaken in ('11','12')
) c on a.ResearchID = c.ResearchID;

--SCIENCE