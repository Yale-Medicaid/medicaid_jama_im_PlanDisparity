local fname `1' // Name of input data to use
local sname `2' // Name of output save file for fstats

set matsize 11000
set linesize 250

import delimited `fname'

local outcomes all_cost_wins50 ///
               pc_numdays ///
               capch_all ///
               ednoadmit_numdays ///
               pharm_numclms ///
               asthma_numclms ///
               diabetes_numclms ///
               statins_numclms ///
               antihypertensive_numclms ///
               ed_avoid_numdays ///
               amr ///
               ha1c ///
               wcv_3_ch ///
               wcv_12_ch ///
               wcv_18_ch ///
               aap_ad ///
               chl_all ///
               add

quietly file open myfile using `sname', write replace
file write myfile "regression" _tab "outcome" _tab "variable" _tab "coeff" _tab "se" ///
                               _tab "pval" _tab "tstat" _tab "lb" _tab "ub" _n

// No Controls
foreach outcome of local outcomes{
  reghdfe `outcome' io(10 20 30 40 50).plan_race_asgn i.plan_id_asgn, ///
             absorb(uor) cluster(uor)
                     
  local names : colfullnames e(b)
    foreach var of local names{
      local t = _b[`var']/_se[`var']
      local p = 2*ttail(e(df_r),abs(`t'))
      local lb = (_b[`var'] - invttail(e(df_r), 0.025) * _se[`var'])
      local ub = (_b[`var'] + invttail(e(df_r), 0.025) * _se[`var'])

      file write myfile "raw" _tab "`outcome'" _tab ("`var'") ///
                                   _tab (_b[`var']) _tab (_se[`var']) ///
                                   _tab (`p') _tab (`t') _tab (`lb') _tab (`ub') _n
    }            
  test 11.plan_race_asgn = 21.plan_race_asgn = 31.plan_race_asgn = 41.plan_race_asgn = 51.plan_race_asgn
  file write myfile "raw" _tab "`outcome'" _tab ("plan)diff") ///
                               _tab ("None") _tab ("None") ///
                               _tab (r(p)) _tab (r(F)) _tab ("None") _tab ("None") _n  

}


// AGE SEX
foreach outcome of local outcomes{
  reghdfe `outcome' io(10 20 30 40 50).plan_race_asgn i.plan_id_asgn,  ///
       absorb(uor recip_age_g recip_gender_g) cluster(uor)

                     
  local names : colfullnames e(b)
    foreach var of local names{
      local t = _b[`var']/_se[`var']
      local p = 2*ttail(e(df_r),abs(`t'))
      local lb = (_b[`var'] - invttail(e(df_r), 0.025) * _se[`var'])
      local ub = (_b[`var'] + invttail(e(df_r), 0.025) * _se[`var'])

      file write myfile "age_sex" _tab "`outcome'" _tab ("`var'") ///
                               _tab (_b[`var']) _tab (_se[`var']) ///
                               _tab (`p') _tab (`t') _tab (`lb') _tab (`ub') _n
    }            
  test 11.plan_race_asgn = 21.plan_race_asgn = 31.plan_race_asgn = 41.plan_race_asgn = 51.plan_race_asgn
  file write myfile "age_sex" _tab "`outcome'" _tab ("plan)diff") ///
                           _tab ("None") _tab ("None") ///
                           _tab (r(p)) _tab (r(F)) _tab ("None") _tab ("None") _n  

}


// age_sex_elig
foreach outcome of local outcomes{
  reghdfe `outcome' io(10 20 30 40 50).plan_race_asgn i.plan_id_asgn,  ///
       absorb(uor recip_age_g recip_gender_g elig_cat) cluster(uor)

                     
  local names : colfullnames e(b)
    foreach var of local names{
      local t = _b[`var']/_se[`var']
      local p = 2*ttail(e(df_r),abs(`t'))
      local lb = (_b[`var'] - invttail(e(df_r), 0.025) * _se[`var'])
      local ub = (_b[`var'] + invttail(e(df_r), 0.025) * _se[`var'])

      file write myfile "age_sex_elig" _tab "`outcome'" _tab ("`var'") ///
                               _tab (_b[`var']) _tab (_se[`var']) ///
                               _tab (`p') _tab (`t') _tab (`lb') _tab (`ub') _n
    }            
  test 11.plan_race_asgn = 21.plan_race_asgn = 31.plan_race_asgn = 41.plan_race_asgn = 51.plan_race_asgn
  file write myfile "age_sex_elig" _tab "`outcome'" _tab ("plan)diff") ///
                           _tab ("None") _tab ("None") ///
                           _tab (r(p)) _tab (r(F)) _tab ("None") _tab ("None") _n  

}


// demo
foreach outcome of local outcomes{
  reghdfe `outcome' io(10 20 30 40 50).plan_race_asgn i.plan_id_asgn,  ///
       absorb(uor recip_age_g recip_gender_g recip_zip elig_cat) cluster(uor)

                     
  local names : colfullnames e(b)
    foreach var of local names{
      local t = _b[`var']/_se[`var']
      local p = 2*ttail(e(df_r),abs(`t'))
      local lb = (_b[`var'] - invttail(e(df_r), 0.025) * _se[`var'])
      local ub = (_b[`var'] + invttail(e(df_r), 0.025) * _se[`var'])

      file write myfile "demo" _tab "`outcome'" _tab ("`var'") ///
                               _tab (_b[`var']) _tab (_se[`var']) ///
                               _tab (`p') _tab (`t') _tab (`lb') _tab (`ub') _n
    }            
  test 11.plan_race_asgn = 21.plan_race_asgn = 31.plan_race_asgn = 41.plan_race_asgn = 51.plan_race_asgn
  file write myfile "demo" _tab "`outcome'" _tab ("plan)diff") ///
                           _tab ("None") _tab ("None") ///
                           _tab (r(p)) _tab (r(F)) _tab ("None") _tab ("None") _n  

}


// Demo + HCCs
foreach outcome of local outcomes{
  reghdfe `outcome' io(10 20 30 40 50).plan_race_asgn i.plan_id_asgn hcc_*,  ///
       absorb(uor recip_age_g recip_gender_g recip_zip elig_cat) cluster(uor)

                     
  local names : colfullnames e(b)
    foreach var of local names{
      local t = _b[`var']/_se[`var']
      local p = 2*ttail(e(df_r),abs(`t'))
      local lb = (_b[`var'] - invttail(e(df_r), 0.025) * _se[`var'])
      local ub = (_b[`var'] + invttail(e(df_r), 0.025) * _se[`var'])

      file write myfile "demo_health" _tab "`outcome'" _tab ("`var'") ///
                                      _tab (_b[`var']) _tab (_se[`var']) ///
                                      _tab (`p') _tab (`t') _tab (`lb') _tab (`ub') _n
    }            
  test 11.plan_race_asgn = 21.plan_race_asgn = 31.plan_race_asgn = 41.plan_race_asgn = 51.plan_race_asgn
  file write myfile "demo_health" _tab "`outcome'" _tab ("plan)diff") ///
                                  _tab ("None") _tab ("None") ///
                                  _tab (r(p)) _tab (r(F)) _tab ("None") _tab ("None") _n  

}



clear
exit
