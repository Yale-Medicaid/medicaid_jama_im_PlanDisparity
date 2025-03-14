# medicaid_jama_im_PlanDisparity
Data replication ReadME for
*Attributing Racial Differences in Care to Plan Performance or Selection: Evidence from Randomization in Medicaid*

Jacob Wallace, Chima D. Ndumele, Anthony Lollo, Danil Agafiev-Macambira, Matthew Lavallee, Beniamino Green, Kate A. Duchowny, and J. Michael McWilliams.

## Replication Code
The programs for the main paper analyses are provide as `.py` files that need to be run in a python environment. We also provide the `.do` files which run the main regressions (these files are called via `subprocess` within the `.py` scripts). 

#### Required Packages and Versions
The analysis was performed with the following libraries and versions, so to ensure reproducibility packages and versions should match.

```
python 3.9.1

pandas 1.3.1
numpy 1.19.5
scipy 1.6.0
statsmodels 0.12.2
matplotlib 3.3.3
```

#### Data Sources
*Primary data source*

The primary data (Louisiana Department of Health) for this project are confidential and contain all individual-level fully adjudicated Medicaid health care claims, eligibility and demographic data from 2010 through 2016. These data may be obtained via Data Use Agreements (DUAs) with the Louisiana Department of Health. A data dictionary cannot be supplied unless a DUA is in place. It can take months to secure a DUA and gain access to the data. The authors will assist with any reasonable replication attempts for two years following publication. 


The primary data, outlined in the table below, consist of a single enrollment file and a single demographic file which were extracted on August 2018. Claims files were split into quarterly extracts between 2010-2016. There was a single network file. 

*Secondary data sources*

This project uses several secondary data sources to construct outcomes. Some of these data are not able to be provided here, due to the licensing use agreements, but can be freely requested or purchased. 

- **Avoidable ED Quality Measure**: The [methodlogy and value set directory](https://dhs.saccounty.gov/PRI/Documents/Sacramento-Medi-Cal-Managed-Care-Stakeholder-Advisory-Committee/Old%20Info%20-%20Do%20not%20delete/Other/MA-MCMC--DHCS-Reducing-Avoidable-ED-Visits-Rpt-2011-12.pdf) are provided in Appendix A of the linked pdf. 

#### Analysis and Processing Scripts

All code for this project is provided within the `code\` folder. After obtaining data from the state of Louisiana via a DUA, and setting up a directory with the same folder structure and file names, the project can be replicated. 

1. `analysis.py`. This contains all methods required to transform the data, create analytic tables and create the necessary figures and tables. The method `main_for_paper2()` enumrates the order in which each of the individual steps need to be run and running that function alone will fully reproduce the analyses.
2. `test_balance.do`. Runs the balance-test regressions to produce Table 1
3. `paper2_AC_regs_5plans.do`. Runs the main regressions responsible for the Observational Panel of Table 2
4. `paper2_main_regs_RF_5plans.do`. Runs the main regressions responsible for the Randomized Panel of Table 2
5. `paper2_main_regs_IV.do` Runs the IV specification for the main regressions (eTable 13)
