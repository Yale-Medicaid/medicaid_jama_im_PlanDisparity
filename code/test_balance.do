local fname `1' // Name of input data to use
local sname `2' // Name of output save file for fstats

set matsize 11000
set linesize 250

import delimited `fname'

local regressands age_le5 ///
				  age_6to17 ///
				  age_18to64 ///
				  predicted_risk ///
				  is_black ///
				  is_white ///
				  is_female ///
				  is_male ///
				  cond_asthma ///
				  cond_dep_bip_psych ///
				  cond_diabetes ///
				  cond_pregnancy ///
				  cond_cardiov_cond


				  
quietly file open myfile using `sname', write replace
file write myfile "Baseline" _tab "coeff" _tab "se" _tab ///
				  "Fstat" _tab "pval" _tab "Nobs" _n

foreach regvar of local regressands{
	reghdfe `regvar' i.plan_id_test, absorb(uor) vce(cluster uor)

	capture noisily testparm i.plan_id_test
					
    file write myfile "`regvar'" _tab "NaN" _tab "NaN" _tab ///
					   (e(F)) _tab (r(p)) _tab (e(N)) _n
					   
	local names : colfullnames e(b)
	foreach var of local names{
		file write myfile ("`regvar'_`var'") _tab (_b[`var']) _tab ///
						  (_se[`var']) _tab "NaN" _tab "NaN" _tab "NaN" _n
	}
					   
}

file close myfile
clear
exit
