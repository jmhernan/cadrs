/* Adding the NSC validation to the aggregated CADR subject table */

-- ADD NSC FLAG
DROP VIEW IF EXISTS cadr_tuk_val;
CREATE VIEW cadr_tuk_val
AS
select a.*, b.nsc_4yr
from agg_cadr_tuk a
left join(
    Select ResearchID, 1 as nsc_4yr
    from nsc_tuk_cohort_17
) b on a.ResearchID = b.ResearchID;