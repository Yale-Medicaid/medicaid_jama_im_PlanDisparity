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
                               _tab "pval" _tab "lb" _tab "ub" _n

// No Controls
foreach outcome of local outcomes{
  ivreghdfe `outcome' (i.plan_id i.plan_id#i.recip_race_g = ///
                       i.plan_id_asgn i.plan_id_asgn#i.recip_race_g), ///
          absorb(uor) cluster(uor)
                     
  local names : colfullnames e(b)
    foreach var of local names{
      local t = _b[`var']/_se[`var']
      local p = 2*ttail(e(df_r),abs(`t'))
      local lb = (_b[`var'] - invttail(e(df_r), 0.025) * _se[`var'])
      local ub = (_b[`var'] + invttail(e(df_r), 0.025) * _se[`var'])

      file write myfile "raw" _tab "`outcome'" _tab ("`var'") ///
                                   _tab (_b[`var']) _tab (_se[`var']) ///
                                   _tab (`p') _tab (`lb') _tab (`ub') _n
    }            
  test 1.plan_id#1.recip_race_g =  2.plan_id#1.recip_race_g =  ///
       3.plan_id#1.recip_race_g =  4.plan_id#1.recip_race_g = ///
       5.plan_id#1.recip_race_g 

  file write myfile "raw" _tab "`outcome'" _tab ("plan)diff") ///
                               _tab ("None") _tab ("None") ///
                               _tab (r(p)) _tab ("None") _tab ("None") _n  

}

// Age + Sex
foreach outcome of local outcomes{
  ivreghdfe `outcome' (i.plan_id i.plan_id#i.recip_race_g = ///
                       i.plan_id_asgn i.plan_id_asgn#i.recip_race_g), ///
       absorb(uor recip_age_g recip_gender_g) cluster(uor)

                     
  local names : colfullnames e(b)
    foreach var of local names{
      local t = _b[`var']/_se[`var']
      local p = 2*ttail(e(df_r),abs(`t'))
      local lb = (_b[`var'] - invttail(e(df_r), 0.025) * _se[`var'])
      local ub = (_b[`var'] + invttail(e(df_r), 0.025) * _se[`var'])

      file write myfile "age_sex" _tab "`outcome'" _tab ("`var'") ///
                               _tab (_b[`var']) _tab (_se[`var']) ///
                               _tab (`p') _tab (`lb') _tab (`ub') _n
    }            
  test 1.plan_id#1.recip_race_g =  2.plan_id#1.recip_race_g =  ///
       3.plan_id#1.recip_race_g =  4.plan_id#1.recip_race_g = ///
       5.plan_id#1.recip_race_g 

  file write myfile "age_sex" _tab "`outcome'" _tab ("plan)diff") ///
                           _tab ("None") _tab ("None") ///
                           _tab (r(p)) _tab ("None") _tab ("None") _n  

}


// demo
foreach outcome of local outcomes{
  ivreghdfe `outcome' (i.plan_id i.plan_id#i.recip_race_g = ///
                       i.plan_id_asgn i.plan_id_asgn#i.recip_race_g), ///
       absorb(uor recip_age_g recip_gender_g elig_cat) cluster(uor)

                     
  local names : colfullnames e(b)
    foreach var of local names{
      local t = _b[`var']/_se[`var']
      local p = 2*ttail(e(df_r),abs(`t'))
      local lb = (_b[`var'] - invttail(e(df_r), 0.025) * _se[`var'])
      local ub = (_b[`var'] + invttail(e(df_r), 0.025) * _se[`var'])

      file write myfile "age_sex_elig" _tab "`outcome'" _tab ("`var'") ///
                               _tab (_b[`var']) _tab (_se[`var']) ///
                               _tab (`p') _tab (`lb') _tab (`ub') _n
    }            
  test 1.plan_id#1.recip_race_g =  2.plan_id#1.recip_race_g =  ///
       3.plan_id#1.recip_race_g =  4.plan_id#1.recip_race_g = ///
       5.plan_id#1.recip_race_g 

  file write myfile "age_sex_elig" _tab "`outcome'" _tab ("plan)diff") ///
                           _tab ("None") _tab ("None") ///
                           _tab (r(p)) _tab ("None") _tab ("None") _n  

}


// demo
foreach outcome of local outcomes{
  ivreghdfe `outcome' (i.plan_id i.plan_id#i.recip_race_g = ///
                       i.plan_id_asgn i.plan_id_asgn#i.recip_race_g), ///
       absorb(uor recip_age_g recip_gender_g recip_zip elig_cat) cluster(uor)

                     
  local names : colfullnames e(b)
    foreach var of local names{
      local t = _b[`var']/_se[`var']
      local p = 2*ttail(e(df_r),abs(`t'))
      local lb = (_b[`var'] - invttail(e(df_r), 0.025) * _se[`var'])
      local ub = (_b[`var'] + invttail(e(df_r), 0.025) * _se[`var'])

      file write myfile "demo" _tab "`outcome'" _tab ("`var'") ///
                               _tab (_b[`var']) _tab (_se[`var']) ///
                               _tab (`p') _tab (`lb') _tab (`ub') _n
    }            
  test 1.plan_id#1.recip_race_g =  2.plan_id#1.recip_race_g =  ///
       3.plan_id#1.recip_race_g =  4.plan_id#1.recip_race_g = ///
       5.plan_id#1.recip_race_g 

  file write myfile "demo" _tab "`outcome'" _tab ("plan)diff") ///
                           _tab ("None") _tab ("None") ///
                           _tab (r(p)) _tab ("None") _tab ("None") _n  

}


// Demo + HCCs
foreach outcome of local outcomes{
  ivreghdfe `outcome' (i.plan_id i.plan_id#i.recip_race_g = ///
                       i.plan_id_asgn i.plan_id_asgn#i.recip_race_g) hcc_*, ///
       absorb(uor recip_age_g recip_gender_g recip_zip elig_cat) cluster(uor)

                     
  local names : colfullnames e(b)
    foreach var of local names{
      local t = _b[`var']/_se[`var']
      local p = 2*ttail(e(df_r),abs(`t'))
      local lb = (_b[`var'] - invttail(e(df_r), 0.025) * _se[`var'])
      local ub = (_b[`var'] + invttail(e(df_r), 0.025) * _se[`var'])

      file write myfile "demo_health" _tab "`outcome'" _tab ("`var'") ///
                                      _tab (_b[`var']) _tab (_se[`var']) ///
                                      _tab (`p') _tab (`lb') _tab (`ub') _n
    }            
  test 1.plan_id#1.recip_race_g =  2.plan_id#1.recip_race_g =  ///
       3.plan_id#1.recip_race_g =  4.plan_id#1.recip_race_g = ///
       5.plan_id#1.recip_race_g 

  file write myfile "demo_health" _tab "`outcome'" _tab ("plan)diff") ///
                                  _tab ("None") _tab ("None") ///
                                  _tab (r(p)) _tab ("None") _tab ("None") _n  

}



clear
exit
