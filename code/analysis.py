# Racial Disparities initial analysis for CommonWealth Fund

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from linearmodels.iv import AbsorbingLS
import statsmodels.api as sm
from matplotlib import ticker
from matplotlib import patches as mpatches
import statsmodels.formula.api as smf
import statsmodels.stats.multitest as multitest # For adjusted p-values

sys.path.append('D:/Groups/YSPH-HPM-Ndumele/Networks/Anthony/Racial_Disparities/code')
sys.path.append('//storage.yale.edu/home/YSPH_HPM-CC0940-MEDSPH2/'
                +'Private Data Pipeline/Pipeline_Anthony/')

import QualityMetrics.Utilization as utilization
import QualityMetrics.BaseFunctions as base
from Analytics.Analytics import subset_b4_time_by_recip
from QualityMetrics.DrugClasses import categorize_drug_class
from project_paths import FS
import WithStata.WithStata as ws
import RawToGold.DataTransform as dt
import RiskAdjustment.RiskAdjust as ra
from Path.paths import Pipeline

import matplotlib.transforms


def main_for_paper1():
    create_lab_and_tests()
    create_ed()
    create_primary_care()
    create_specialty_care()
    create_hvc_drugs()
    create_2016_hhs_hccs()
    create_pregnancy_pain()
    attribute_providers()

    create_analytic_tables()
    paper1_predict_risk()

    paper1_table1()
    paper1_regressions_yearly()
    paper1_lvc_regressions_yearly()
    paper1_regressions_preg()
    paper1_table2()
    paper1_table3()
    paper1_adjusted_pvals()

    paper1_regressions_yearly_by_group()
    paper1_regressions_yearly_by_group_relwhiteM()
    paper1_regressions_yearly_by_group_by_gender()

    paper1_create_atc_spending()
    paper1_regressions_atc()
    paper1_table_appendix_by_atc()


def main_for_paper2():
    paper2_create_primary_care()
    paper2_create_lab_and_tests()
    paper2_create_ed()
    paper2_create_specialty_care()
    paper2_create_hvc_drugs()
    paper2_create_inpatient()
    paper2_create_brand_generic()

    paper2_grab_AA_analytic_tables()
    paper2_augment_AA_analytic_tables()
    paper2_analytic_tables_to_yearly()

    paper2_create_baseline_tables()
    paper2_create_predicted_risk()
    paper2_augment_baseline_tables()

    paper2_run_balance_regressions()
    paper2_run_balance_regressions_race_split()
    paper2_run_main_regressions()
    paper2_run_main_regressions_reweighted()
    paper2_run_main_regressions_hispanic()    # Review Request
    paper2_clean_main_reg_results()           # Output for table 2
    paper2_clean_main_reg_results_reweighted()
    paper2_clean_main_reg_results_hispanic()  # Review Request

    paper2_figure_averages()      # Figure 1
    paper2_figure_planeff()       # Figure 2 + Appendix Version
    paper2_figure_sorting()

    paper2_figure_spend_dist()    # Appendix Figure of Spending Distribution
    paper2_figure_planeff_RF()    # Appendix Figure of Plan Effects on entire population
    paper2_figure_first_stage()   # Appendix Figure of First stage strength

    # Appendix Spending Sensitivities
    paper2_run_spending_sensitivities()
    paper2_clean_spending_sensitivities()

    # For resampled estimates. This takes a while, so you can call it in different kernels with 
    # different ranges i.e. one with rng=(0,300), another with rng = (300,600), ... since each run
    # is independent. 
    paper2_run_main_regressions_AAs_resampled(rng=(0, 1000))
    paper2_run_main_regressions_byplan_BWrace_ACs()
    paper2_clean_main_reg_results_byplan_BWrace_ACs()

    paper2_run_main_regressions_byplan_byrace()
    paper2_clean_main_reg_results_byplan_byrace()

####################################################################################################
########################################### PAPER 1 ################################################
####################################################################################################
def clean_residential_seg_index():
    output_p = FS().derived_p+'clean_residential_seg_index/output/'
    data_p = FS().derived_p+'clean_residential_seg_index/data/'

    for state in ['KS', 'TN', 'LA']:
        df = pd.read_csv(data_p+f'{state}_residential_segregation.csv')
        df.columns = ['county_name', 'seg_index']
        df['state'] = state
        
        df = dt.county_name_to_fips(df, state_col='state', cname_col='county_name',
                                    fips_col='fips')
        print('Incorrectly mapped', df.fips.isnull().sum())
        df.to_pickle(output_p+f'{state}_segregation.pkl')


def create_lab_and_tests():
    output_p = FS().derived_p+'create_lab_and_tests/output/'

    l = [('KS', FS().ks_pipe_gold_path, (2016, 2019)),
         ('TN', FS().tn_pipe_gold_path, (2016, 2018)), 
         ('LA', FS().la_pipe_gold_path, (2016, 2017))]

    for state, path, yr in l:
        file_specs = [(f'claims_details_{year}.pkl', ['CPT', 'HCPCS', 'POS'], 'details') 
                        for year in range(*yr)]

        df = utilization.BETOS_categorization(gold_path=path,
                                              file_specs=file_specs,
                                              keep_paid_only=True, 
                                              level='broad')
        
        dfc = pd.concat([pd.read_pickle(path+f'claims_{year}.pkl') for year in range(*yr)])

        d = {}
        for cat in ['Imaging', 'Tests']:
            claims = df[df.category.eq(cat)].claim_id.unique()
            gp = dfc[dfc.claim_id.isin(claims)]
            d[cat] = (gp.groupby(['recip_id', gp.claim_date.dt.year.rename('year'),
                                  gp.claim_date.dt.month.rename('month')])
                        .agg(cost=('claim_amt_paid', 'sum'),
                             numclms=('claim_id', 'nunique'),
                             numdays=('claim_date', 'nunique')))

        res = pd.concat(d, axis=1)
        res.columns = [f'{x.lower()}_{y}' for x,y in res.columns]
        res = res.reset_index()
        res.to_pickle(output_p+f'{state}_lab_tests.pkl')
           

def create_ed():
    output_p = FS().derived_p+'create_ed/output/'

    l = [('KS', FS().ks_pipe_gold_path, (2016, 2019)),
         ('TN', FS().tn_pipe_gold_path, (2016, 2018)), 
         ('LA', FS().la_pipe_gold_path, (2016, 2017))
        ]

    for state, gold_path, yr in l:
        file_specs = gen_filespecs(state, yr)

        df = utilization.ED_utilization(gold_path=gold_path,
                                        file_specs=file_specs,
                                        HCPCS_CPT=True,
                                        revenue=True,
                                        pos=True,
                                        keep_paid_only=True, 
                                        remove_admissions=True,
                                        limit_one_per_day=False)

        # Now aggregate
        dfc = pd.concat([pd.read_pickle(gold_path+f'claims_{yr}.pkl') for yr in range(*yr)])
        dfc = dfc[dfc.claim_id.isin(df.claim_id)]
        
        dfc = (dfc.groupby(['recip_id', dfc.claim_date.dt.year.rename('year'),
                      dfc.claim_date.dt.month.rename('month')])
              .agg(cost=('claim_amt_paid', 'sum'),
                   numclms=('claim_id', 'nunique'),
                   numdays=('claim_date', 'nunique'))
              .add_prefix('ednoadmit_'))
        dfc.to_pickle(output_p+f'{state}_ed.pkl')


def create_primary_care():
    output_p = FS().derived_p+'create_primary_care/output/'

    l = [('KS', FS().ks_pipe_gold_path, (2013, 2019)),
         ('TN', FS().tn_pipe_gold_path, (2010, 2018)), 
         ('LA', FS().la_pipe_gold_path, (2010, 2017))
        ]

    for state, gold_path, yr in l:
        file_specs = gen_filespecs(state, yr)

        df = utilization.ambulatory_utilization(gold_path=gold_path,
                                                file_specs=file_specs,
                                                keep_paid_only=True, 
                                                spec_list=None)
        
        df.to_pickle(output_p+f'{state}_pc_claims.pkl')
        
        # Now aggregate
        dfc = pd.concat([pd.read_pickle(gold_path+f'claims_{yr}.pkl') for yr in range(*yr)])
        dfc = dfc[dfc.claim_id.isin(df.claim_id)]

        # Count days as 
        dfc = (dfc.groupby(['recip_id', dfc.claim_date.dt.year.rename('year'),
                      dfc.claim_date.dt.month.rename('month')])
              .agg(cost=('claim_amt_paid', 'sum'),
                   numclms=('claim_id', 'nunique'),
                   numdays=('claim_date', 'nunique'))
              .add_prefix('pc_'))
        dfc.to_pickle(output_p+f'{state}_pc.pkl')


def create_mpm():
    """
    Create measures of Medication possession ratio for selected drug classes.
    Only 2016
    """
    output_p = FS().derived_p+'create_mpm/output/'
    l = [('KS', FS().ks_pipe_gold_path),
         ('TN', FS().tn_pipe_gold_path), 
         ('LA', FS().la_pipe_gold_path)
        ]

    # Only for 2016 because asked for in a review
    for state, gold_path in l:
        df = pd.read_pickle(gold_path+f'pharm_2016.pkl')
        df = df[df.claim_paid.eq(True)]

        df = categorize_drug_class(df, return_cost=True)
        
        # Clip days supply if it spills over to next year
        df['exhaust_dt_eff'] = (df['claim_date'] + pd.to_timedelta(df['supply_days'], unit='D')).clip(upper='2017-01-01')
        df['supply_days_eff'] = ((df['exhaust_dt_eff'] - df['claim_date']).dt.days).clip(lower=0)


        gp = df.groupby(['recip_id', 'drug_class']).agg(
                    total_supply_days=('supply_days_eff', 'sum'),
                    first_fill=('claim_date', 'min'))

        # Through end of year calculate MPM
        gp['duration'] = (pd.to_datetime(f'2016-12-31') - gp['first_fill']).dt.days + 1
        gp['mpm'] = gp['total_supply_days']/gp['duration']

        gp = gp['mpm'].unstack(-1).rename_axis(columns=None)
        gp.columns = gp.columns.str.lower()
        gp = gp.add_suffix('_mpm').reset_index().assign(year=2016)
        
        gp.to_pickle(output_p+f'{state}_hvc_drugs_mpm.pkl')
        
        display(gp.filter(like='mpm').mean())


def create_specialty_care():
    """
    Defined as office care not in the PC measure
    """
    output_p = FS().derived_p+'create_specialty_care/output/'
    l = [('KS', FS().ks_pipe_gold_path, range(2016, 2019)),
         ('TN', FS().tn_pipe_gold_path, range(2016, 2018)), 
         ('LA', FS().la_pipe_gold_path, range(2016, 2017))
        ]

    for state, gold_path, years in l:
        dfpc = pd.read_pickle(FS().derived_p+f'create_primary_care/output/{state}_pc_claims.pkl')
        dfc = pd.concat([pd.read_pickle(gold_path+f'claims_{year}.pkl') for year in years])
        dfcd = pd.concat([pd.read_pickle(gold_path+f'claims_details_{year}.pkl') for year in years])

        office = dfcd[dfcd.pos_code.eq('11')].claim_id.unique()
        specs = dfc[dfc.claim_id.isin(office) & dfc.claim_paid.eq(True)
                    & ~dfc.claim_id.isin(dfpc.claim_id)]

        # Count days as 
        specs = (specs.groupby(['recip_id', specs.claim_date.dt.year.rename('year'),
                                specs.claim_date.dt.month.rename('month')])
                  .agg(cost=('claim_amt_paid', 'sum'),
                       numclms=('claim_id', 'nunique'),
                       numdays=('claim_date', 'nunique'))
                  .add_prefix('spec_'))
        specs.to_pickle(output_p+f'{state}_spec.pkl')


def create_hvc_drugs():
    output_p = FS().derived_p+'create_hvc_drugs/output/'

    l = [('KS', FS().ks_pipe_gold_path, (2016, 2019)),
         ('TN', FS().tn_pipe_gold_path, (2016, 2018)), 
         ('LA', FS().la_pipe_gold_path, (2016, 2017))
        ]

    for state, gold_path, yr in l:
        df = pd.concat([pd.read_pickle(gold_path+f'pharm_{year}.pkl') for year in range(*yr)],
                       ignore_index=True)
        df = df[df.claim_paid.eq(True)]
        
        df = categorize_drug_class(df, return_cost=True)
        
        df = (df.groupby(['recip_id', df.claim_date.dt.year.rename('year'),
                          df.claim_date.dt.month.rename('month'), 'drug_class'])
                .agg(cost=('claim_amt_paid', 'sum'),
                     numclms=('claim_id', 'nunique'),
                     supplydays=('supply_days', 'sum'))
                 .unstack(-1))

        df.columns = [f'{y.lower()}_{x}' for x,y in df.columns]
        df = df.reset_index()

        dfany = df.filter(like='numclms').gt(0).astype(int)
        dfany = dfany.rename(columns={x: x.replace('_numclms', '_any') for x in dfany.columns})
        df = pd.concat([df, dfany], axis=1)

        df.to_pickle(output_p+f'{state}_hvc_drugs.pkl')


def create_atc_spending():
    output_p = FS().derived_p+'create_atc_spending/output/'
    atc_path = ('//storage.yale.edu/home/YSPH_HPM-CC0940-MEDSPH2/'
                 +'Private Data Pipeline/Resource_Data/RxNorm/Gold/')

    dfatc = pd.read_pickle(atc_path+'ndc_to_atc.pkl')
    atc3s =['N06A', 'R06A', 'N03A', 'N06B', 'N02B', 
            'J01C', 'A02B', 'R03C', 'N05A', 'N02A', 
            'R03B', 'S01A', 'N05B', 'C10A', 'A10B',
            'C07A', 'M03B', 'C09A', 'R03D', 'R02A']


    l = [('KS', FS().ks_pipe_gold_path, range(2016, 2019)),
         ('TN', FS().tn_pipe_gold_path, range(2016, 2018)), 
         ('LA', FS().la_pipe_gold_path, range(2016, 2017))
        ]

    for state, gold_path, years in l:
        df = pd.concat([pd.read_pickle(gold_path+f'pharm_{year}.pkl') for year in years])
        df = df[df.claim_paid.eq(True)]
        df = df.merge(dfatc[['drug_ndc', 'atc_3', 'atc_3_desc']], how='left')
        df = df[df.atc_3.isin(atc3s)]
        
        res = (df.groupby(['recip_id', df.claim_date.dt.year.rename('year'), 'atc_3'])
                 .agg(cost=('claim_amt_paid', 'sum'),
                      numclms=('claim_id', 'nunique'),
                      numdays=('claim_date', 'nunique'))
                 .unstack(-1)
                 .rename_axis(columns=[None, None]))

        res.columns = [f'atc3_{atc.lower()}_{out}' for out, atc in res.columns]
        
        res.to_pickle(output_p+f'{state}_atc_spend.pkl')


def create_hhs_hccs():
    output_p = FS().derived_p+'create_hhs_hccs/output/'

    l = [('KS', FS().ks_pipe_gold_path, range(2013, 2019)),
         ('TN', FS().tn_pipe_gold_path, range(2010, 2018)), 
         ('LA', FS().la_pipe_gold_path, range(2010, 2017))]

    for state, gold_path, years in l:
        for year in years:
            df = pd.read_pickle(gold_path+f'diagnosis_{year}.pkl')
            hccs = ra.HHS_hccs_from_diagnosis_df(df, gold_path)
            hccs.to_pickle(output_p+f'{state}_{year}.pkl')


def create_hyp_indicator():
    output_p = FS().derived_p+'create_hyp_indicator/output/'
     
    l = [('KS', FS().ks_pipe_gold_path),
         ('TN', FS().tn_pipe_gold_path),
         ('LA', FS().la_pipe_gold_path)
        ]
     
    for state, gold_path in l:
        df = pd.read_pickle(gold_path+'diagnosis_2016.pkl')

        df = df[df.diag_code.eq('I10')]
     
        df = df[['recip_id']].drop_duplicates().assign(cond_hypertension=1).reset_index(drop=True)
        df['year'] = 2016
        df.to_pickle(output_p+f'{state}_hyp_indicator.pkl')


def create_pregnancy_pain():
    """
    Create measures of pain medications after pregnancy. 
        - Any Pain
        - Opioid
        - Days to medication
    """
    output_p = FS().derived_p+'create_pregnancy_pain/output/'

    l = [('KS', FS().ks_pipe_gold_path, FS().ks_pipe_analytic_path),
         ('TN', FS().tn_pipe_gold_path, FS().tn_pipe_analytic_path), 
         ('LA', FS().la_pipe_gold_path, FS().la_pipe_analytic_path)
        ]

    for state, gold_path, a_path in l:
        # PRevious year for 90 day lookback to exclude pregs with opioids before. 
        dfp = pd.concat([pd.read_pickle(gold_path+f'pharm_{yr}.pkl') for yr in [2015, 2016]])
        dfp = dfp[dfp.claim_paid.eq(True)]
        
        # For KS and LA where we have info, use the prescription write date. 
        if state == 'KS':
            dfp['claim_date'] = dfp['date_rx_write']
        elif state == 'LA':
            dfp['date_rx_dispense'] = dfp['claim_date']
            dfp['claim_date'] = dfp['date_rx_write']

        dfc = pd.read_pickle(gold_path+f'claims_2016.pkl')
        dfdrg = pd.read_pickle(a_path+f'3M_DRG_output_2016.pkl')
        dfatc = pd.read_pickle(Pipeline().path+'Resource_Data/RxNorm/Gold/ndc_to_atc.pkl')

        # Bring over ATC We want `'N02'`
        dfp = dfp.merge(dfatc[['drug_ndc', 'atc_2', 'atc_2_desc', 'atc_3', 'atc_3_desc']], 
                        on='drug_ndc', how='left')

        deliveries = dfdrg[dfdrg.drg.isin(['560', '540', '541'])].copy()
        deliveries['claim_date'] = deliveries.claim_id.map(dfc.set_index('claim_id').claim_date)
        # Exclude deliveries for which we might not be able to see prescriptions
        deliveries = deliveries[deliveries.claim_date.le('2016-12-17')]

        # Lots of same day deliveries, but just keep one per recip anyway
        deliveries = deliveries.drop_duplicates('recip_id', keep='last')

        pain = dfp[dfp.atc_2.eq('N02')]

        m = pd.merge_asof((deliveries.sort_values('claim_date')
                                     .rename(columns={'claim_date': 'claim_date_birth'})),
                          (pain.sort_values('claim_date')
                                .rename(columns={'claim_date': 'claim_date_pain'})),
                          by='recip_id',
                          left_on='claim_date_birth', right_on='claim_date_pain',
                          direction='forward',
                          suffixes=['_birth', '_pain'])

        # Check if they are previously on an opioid. Exclude these deliveries as previous
        # opioid before deliveries can complicate prescriptions of pain medication
        m = pd.merge_asof(m,
                          (pain[pain.atc_3.eq('N02A')].sort_values('claim_date')
                                .rename(columns={'claim_date': 'claim_date_prior'})),
                          by='recip_id',
                          left_on='claim_date_birth', right_on='claim_date_prior',
                          direction='backward',
                          allow_exact_matches=False,
                          suffixes=['', '_prior'])
        m = m[~(m.claim_date_birth-m.claim_date_prior).dt.days.lt(90)]

        # Define relevant outcomes
        s = (m['claim_date_pain'] - m['claim_date_birth']).dt.days
        s = s.where(s.le(7))

        m['days_to_pain_write'] = s
        m['days_to_pain_fill'] = ((m['date_rx_dispense']-m['claim_date_birth'] ).dt.days
                                    .where(s.notnull()))
        m['ever_pain'] = s.notnull().astype(int)
        m['opioid'] = m['atc_3'].where(s.notnull()).eq('N02A').astype(int)
        m['analgesic'] = m['atc_3'].where(s.notnull()).eq('N02B').astype(int)
        m['supply_days'] = m['supply_days'].where(s.notnull())

        out_cols = ['recip_id', 'drg', 'severity_of_illness', 'days_to_pain_write', 
                    'days_to_pain_fill', 'ever_pain', 'opioid', 'analgesic', 'supply_days']

        m[out_cols].to_pickle(output_p+f'{state}_preg_pain.pkl')
        # Sanity Checks by state
        print(state, m.shape)
        display(m[out_cols].iloc[:,-6:].mean())


def attribute_providers():
    output_p = FS().derived_p+'attribute_providers/output/'

    l = [('KS', FS().ks_pipe_gold_path, range(2013, 2019)),
         ('TN', FS().tn_pipe_gold_path, range(2010, 2018)), 
         ('LA', FS().la_pipe_gold_path, range(2010, 2017))
        ]

    nppes_p = '//storage.yale.edu/home/YSPH_HPM-CC0940-MEDSPH2/Private Data Pipeline/NPPES/Gold/'
    dfnp = pd.read_pickle(nppes_p+'nppes_providers.pkl')
    dfnp = (dfnp[['npi', 'prov_pract_zip', 'taxonomy_code_prim']]
             .rename(columns={'npi': 'npi_prov_bill'}))

    for state, path, years in l:
        df = pd.concat([pd.read_pickle(path+f'claims_{year}.pkl') for year in years])
        df = df.sort_values('claim_date')
        df['year'] = df.claim_date.dt.year
        print(state, np.round(df['npi_prov_bill'].isnull().mean()*100,2))
        
        provs = dt.fast_mode(df, key_cols=['recip_id', 'year'], value_col='npi_prov_bill')
        provs = provs.merge(dfnp, how='left')
        provs = dt.zip_to_county(provs, 'prov_pract_zip', 'prov_pract_fips')

        provs.to_pickle(output_p+f'{state}_provs_attr.pkl')


def create_analytic_tables():
    """
    Make MM analytic table for disparities in 2016 across three states
    """

    output_p = FS().derived_p+'create_analytic_tables/output/'

    l = [('KS', FS().ks_pipe_analytic_path, FS().ks_pipe_gold_path, range(2016, 2019)),
         ('TN', FS().tn_pipe_analytic_path, FS().tn_pipe_gold_path, range(2016, 2018)), 
         ('LA', FS().la_pipe_analytic_path, FS().la_pipe_gold_path, range(2016, 2017))
        ]

    for state, path, gpath, years in l:
        for year in years:
            df = pd.read_pickle(path+f'analytic_table_{year}.pkl')
            df['state'] = state

            # FOR TN Remove Tenncare and FFS
            if state == 'TN':
                df = df[~(df.plan_id.isnull() | df.plan_id.isin(['62042793911', '62042793904']))]

            df = df.rename(columns={'all_pharm_cost': 'pharm_cost',
                                    'all_pharm_numclms': 'pharm_numclms',
                                    'all_nonph_cost': 'medical_cost',
                                    'all_nonph_numclms': 'medical_numclms'})
            
            print(f'Starting: {state}: {df.recip_id.nunique():,}')
            # Remove anyone > 65
            df = df[df.recip_age.between(0, 65)]
            print(f'After removing 65: {state}: {df.recip_id.nunique():,}')

            # Remove enrollees who have ZIP codes outside of the state
            # All Fips within a state
            fips = dt.county_name_to_fips(pd.DataFrame({'county': ['statewide'], 'state': [state]}), 
                                               state_col='state', cname_col='county',
                                               fips_col='fips', statewide=True)
            # All Zips within that state
            fips = dt.county_to_zip(fips, fips_col='fips', zip_col='zip')
            df = df[df.recip_zip.isin(fips.zip)]
            print(f'After removing Bad ZIPs: {state}: {df.recip_id.nunique():,}')

            if state == 'TN':
                dfpl = pd.read_pickle(FS().tn_pipe_gold_path+'plan.pkl')
                dfpl['plan_name'] = dfpl.plan_name.str.split(' - ').str[0]

                df['plan_id'] = df['plan_id'].map(dfpl.set_index('plan_id')['plan_name'])
            elif state == 'KS':
                df['plan_id'] = df['plan_id'].str.rstrip('AB')
            else:
                # Create a FFS category
                df['plan_id'] = df['plan_id'].fillna('FFS')

            df['plan_id_g'] = df.groupby('plan_id').ngroup()

            # Keep only those continuously enrolled for the entire year
            if state == 'TN' and year == 2017:
                df = df[df.month.between(1, 6)]
                df = df[df.groupby('recip_id')['recip_id'].transform('size').eq(6)]
            else:
                df = df[df.groupby('recip_id')['recip_id'].transform('size').eq(12)]

            print(f'After requiring continuous enrollment: {state}: {df.recip_id.nunique():,}')

            # Bring over CRG:
            dfcrg = pd.read_pickle(path+f'3M_output_{year}_crg_200.pkl')
            dfcrg = dfcrg.drop_duplicates('recip_id')

            dfcrg['health_status'] = dfcrg['AggregatedCrg3'].astype('str').str[0]
            df['health_status'] = df['recip_id'].map(dfcrg.set_index('recip_id')['health_status'])
            df['crg3'] = df['recip_id'].map(dfcrg.set_index('recip_id')['AggregatedCrg3'])
            # So that we don't adjust for any use set 10 and 11 to the same group, i.e. 10
            df['crg3_edit'] = df['crg3'].replace(11, 10)

            df['crg2'] = df['recip_id'].map(dfcrg.set_index('recip_id')['AggregatedCrg2'])
            df = df[df['health_status'].ne(0)]  # Remove people with incorrect health status, small #

            # Race, Age and Gender controls
            df['recip_race_g'] = map_race(df, state)
            df['recip_age_g'] = (df['recip_age']//5).clip(upper=12).mul(5)
            df['recip_gender_g'] = df.recip_gender.map({'M': 0, 'F': 1})    

            print(f'Limit to Black/White: {state} {year}: {df[df.recip_race_g.isin([0,1])].recip_id.nunique():,}')

            # Create Total Outcome
            df['all_cost'] = df['medical_cost'] + df['pharm_cost']

            # Create brand/Generic Utilization/spending
            dfp = pd.read_pickle(gpath+f'pharm_{year}.pkl')
            dfp = dfp[dfp.claim_paid.eq(True)]

            # Create total pharm days supply
            dfp1 = (dfp.groupby(['recip_id', dfp.claim_date.dt.year.rename('year'),
                                dfp.claim_date.dt.month.rename('month')]) 
                      .agg(pharm_supplydays=('supply_days', 'sum')))
            df = df.merge(dfp1, on=['recip_id', 'year', 'month'], how='left')
            df[dfp1.columns] = df[dfp1.columns].fillna(0)

            dfp = (dfp.groupby(['recip_id', dfp.claim_date.dt.year.rename('year'),
                               dfp.claim_date.dt.month.rename('month'), 
                               dfp['drug_is_generic'].map({True: 'generic', False: 'brand'})])
                      .agg(cost=('claim_amt_paid', 'sum'),
                           numclms=('claim_id', 'nunique'),
                           numdays=('claim_date', 'nunique'),
                           supplydays=('supply_days', 'sum'))
                      .unstack(-1))
            dfp.columns = [f'{y}_{x}' for x,y in dfp.columns]
            df = df.merge(dfp, on=['recip_id', 'year', 'month'], how='left')
            df[dfp.columns] = df[dfp.columns].fillna(0)


            # Get inpatient cost 
            dfc = pd.read_pickle(gpath+f'claims_{year}.pkl')
            dfc = dfc[dfc.claim_paid.eq(True)]

            # Different logic by state to get Inpatient
            if state == 'KS':
                dfc = dfc[dfc.claim_type.eq('I')]
            elif state == 'LA':
                dfc = dfc[dfc.yale_cos.isin(['inpat_psych_hosp', 'inpat_hosp'])]
            elif state == 'TN':
                dfid = pd.read_pickle(gpath+f'inpatient_details_{year}.pkl')
                dfc = dfc[dfc.claim_id.isin(dfid.claim_id)]

            # Get MM Cost/Utilization
            dfc = (dfc.groupby(['recip_id', dfc.claim_date.dt.year.rename('year'),
                                dfc.claim_date.dt.month.rename('month')])
                      .agg(inpat_cost=('claim_amt_paid', 'sum'),
                           inpat_numclms=('claim_id', 'nunique'),
                           inpat_numdays=('claim_date', 'nunique')))

            df = df.merge(dfc, on=['recip_id', 'year', 'month'], how='left')
            for col in ['inpat_cost', 'inpat_numclms', 'inpat_numdays']:
                df[col] = df[col].fillna(0)

            # Non-medical minus inpatient. 
            df['other_cost'] = df['medical_cost'] - df['inpat_cost']

            # Bring over labs and tests
            dflt = pd.read_pickle(FS().derived_p+f'create_lab_and_tests/output/{state}_lab_tests.pkl')
            adnl_cols = dflt.columns.difference(['recip_id', 'year', 'month']).tolist()
            df = df.merge(dflt, on=['recip_id', 'year', 'month'], how='left')
            df[adnl_cols] = df[adnl_cols].fillna(0)
            # Create combined measure
            df['tests_imaging_numdays'] = df[['tests_numdays', 'imaging_numdays']].sum(axis=1)

            # Bring over PC
            dflt = pd.read_pickle(FS().derived_p+f'create_primary_care/output/{state}_pc.pkl')
            adnl_cols = dflt.columns.difference(['recip_id', 'year', 'month']).tolist()
            df = df.merge(dflt, on=['recip_id', 'year', 'month'], how='left')
            df[adnl_cols] = df[adnl_cols].fillna(0)
            df['pc_any'] = df['pc_numdays'].clip(upper=1)

            # Bring over speciality
            dflt = pd.read_pickle(FS().derived_p+f'create_specialty_care/output/{state}_spec.pkl')
            adnl_cols = dflt.columns.difference(['recip_id', 'year', 'month']).tolist()
            df = df.merge(dflt, on=['recip_id', 'year', 'month'], how='left')
            df[adnl_cols] = df[adnl_cols].fillna(0)

            # Bring over ED
            dflt = pd.read_pickle(FS().derived_p+f'create_ed/output/{state}_ed.pkl')
            adnl_cols = dflt.columns.difference(['recip_id', 'year', 'month']).tolist()
            df = df.merge(dflt, on=['recip_id', 'year', 'month'], how='left')
            df[adnl_cols] = df[adnl_cols].fillna(0)

            # Bring HVC Drugs
            dflt = pd.read_pickle(FS().derived_p+f'create_hvc_drugs/output/{state}_hvc_drugs.pkl')
            adnl_cols = dflt.columns.difference(['recip_id', 'year', 'month']).tolist()
            df = df.merge(dflt, on=['recip_id', 'year', 'month'], how='left')
            df[adnl_cols] = df[adnl_cols].fillna(0)

            # Bring over Avoidable ED Outcome
            dfed = pd.read_pickle(path+'ED_claims_avoidable.pkl')
            dfed['year'] = dfed.claim_date.dt.year
            dfed['month'] = dfed.claim_date.dt.month
            dfed = dfed[dfed.year.eq(year)]
            dfed = (dfed.groupby(['year', 'month', 'recip_id'])
                        .agg(ed_avoid_numdays=('claim_date', 'nunique')))
            df = df.merge(dfed, on=['recip_id', 'year', 'month'], how='left')
            df['ed_avoid_numdays'] = df['ed_avoid_numdays'].fillna(0, downcast='infer')

            # Bring over HHS HCCs
            dfhhs = pd.read_pickle(FS().derived_p+f'create_hhs_hccs/output/{state}_{year}.pkl')
            df = df.merge(dfhhs, on='recip_id', how='left')
            hcc_cols = [x for x in df.columns if x.startswith('hcc_')]
            df[hcc_cols] = df[hcc_cols].fillna(0)

            # Bring over eligibility category
            df = map_eligibility(df, state)

            # Bring over the area deprivation index
            dfadi = pd.read_pickle(Pipeline().path
                                   +f'Resource_Data/NeighborAtlas/Gold/{state}_adi_zip_2015.pkl')
            dfadi = dfadi.groupby('zip5')['adi_natnl'].mean()
            df['adi_natnl'] = df['recip_zip'].map(dfadi)

            # Urban vs. Rural
            dfurb = (pd.read_pickle(Pipeline().path
                                    +'Resource_Data/Census/Gold/county_urban_rural_classification.pkl')
                       .set_index('fips')['urban_category'])
            df['urban_category'] = df['recip_county_fips'].map(dfurb)

            df.to_pickle(output_p+f'{state}_analytic_table_mm_{year}.pkl')

            ####
            # Yearly Table
            #####
            sumcols = [x for x in df.columns if '_cost' in x or '_numclms' in x or '_numdays' in x
                                             or '_supplydays' in x]
            firstcols = ['recip_gender', 'recip_race', 'recip_race_g', 'recip_age',
                         'recip_gender_g', 'health_status', 'crg3', 'crg3_edit', 'crg2', 'state']
            mode_cols = ['recip_age_g', 'plan_id', 'plan_id_g', 'recip_zip', 'recip_county_fips', 
                         'elig_cat']
            maxcols = ([x for x in df.columns if x.startswith('hcc_')] 
                        + ['pc_any', 'statins_any', 'antihypertensive_any', 'diabetes_any',
                           'asthma_any'])

            agg_ds = {x: 'sum' for x in sumcols}
            agg_df = {x: 'first' for x in firstcols}
            agg_dm = {x: 'max' for x in maxcols}
            agg_d = {**agg_ds, **agg_df, **agg_dm}

            dfy = df.groupby('recip_id').agg(agg_d).reset_index()
            dfy['year'] = year
            for col in mode_cols:
                s = dt.fast_mode(df, key_cols=['recip_id'], value_col=col)
                dfy = dfy.merge(s, on='recip_id', how='left')

            # Ensure ADI/Urbanicity based on modal zip/county
            dfy['adi_natnl'] = dfy['recip_zip'].map(dfadi)
            dfy['urban_category'] = dfy['recip_county_fips'].map(dfurb)

            # Bring over attributed provider
            dfprov = pd.read_pickle(FS().derived_p+f'attribute_providers/output/{state}_provs_attr.pkl')
            dfprov = dfprov.rename(columns={'npi_prov_bill': 'attr_prov'})
            # Fill missing with a state specific ID so missing in LA is not same as missing in TN
            dfy = dfy.merge(dfprov, on=['recip_id', 'year'], how='left')
            dfy['attr_prov'] = dfy['attr_prov'].fillna(f'missing_{state}')

            # Bring over Segregation Index
            dfseg = pd.read_pickle(FS().derived_p
                                   +f'clean_residential_seg_index/output/{state}_segregation.pkl')
            dfseg = dfseg.set_index('fips')['seg_index']
            dfy['seg_index'] = dfy.recip_county_fips.map(dfseg)

            # Also bring over yearly quality
            quality_l = [('capch_1to2', ['CAP_CH_1to2_HEDIS.pkl']),
                         ('capch_2to6', ['CAP_CH_2to6_HEDIS.pkl']),
                         ('capch_7to12', ['CAP_CH_7to12_HEDIS.pkl']),
                         ('capch_12to19', ['CAP_CH_12to19_HEDIS.pkl']),
                         ('awc_ch', ['AWC_CH_HEDIS.pkl']),
                         ('awc_adolesc', ['AWC_ADOLESC_HEDIS.pkl']),
                         ('capch_all', ['CAP_CH_1to2_HEDIS.pkl', 'CAP_CH_2to6_HEDIS.pkl',
                                    'CAP_CH_7to12_HEDIS.pkl', 'CAP_CH_12to19_HEDIS.pkl']),
                         ('awc_all', ['AWC_CH_HEDIS.pkl', 'AWC_ADOLESC_HEDIS.pkl']),
                         ('bcs', ['BCS_AD_HEDIS.pkl']),
                         ('ha1c', ['HA1C_HEDIS.pkl']),
                         ('ssd', ['SSD_AD_HEDIS.pkl']),
                         ('amr', ['AMR_HEDIS.pkl']),
                         ('poud', ['OUD_AD_HEDIS.pkl'])
                        ]

            for metric, f_list in quality_l:
                dfq = pd.concat([pd.read_pickle(path+f) for f in f_list])
                dfq = dfq[dfq['year'].eq(year)]
                dfq = dfq.rename(columns={'numer': metric})
                dfy = dfy.merge(dfq[['recip_id', metric]], on='recip_id', how='left')

            # Add quality measures but only for some states that we trust
            state_quality_l = [('chl_all', ['CHL_AD_HEDIS.pkl', 'CHL_CH_HEDIS.pkl'], ['KS', 'LA']),
                               ('chl_ad', ['CHL_AD_HEDIS.pkl'], ['KS', 'LA']),
                               ('chl_ch', ['CHL_CH_HEDIS.pkl'], ['KS', 'LA']),
                               ('ccs', ['CCS_AD_HEDIS.pkl'], ['LA']),
                               ('adv', ['ADV_HEDIS.pkl'], ['LA', 'KS'])
                               ]

            for metric, f_list, state_l in state_quality_l:
                dfq = pd.concat([pd.read_pickle(path+f) for f in f_list])
                dfq = dfq[dfq['year'].eq(year)]
                dfq = dfq.rename(columns={'numer': metric})

                if state in state_l:
                    dfy = dfy.merge(dfq[['recip_id', metric]], on='recip_id', how='left')
                else:
                    dfy[metric] = np.NaN


            # Handle MPM separately since multiple measures
            dfq = pd.read_pickle(path+'MPM_AD_HEDIS.pkl')
            dfq = dfq[dfq.year.eq(year)]
            dfq['val_set_name'] = dfq['val_set_name'].map({'ACE Inhibitor/ARB Medications': 'mpm_acearb',
                                                           'Diuretic Medications': 'mpm_diur'})
            dfq = (dfq.pivot(index='recip_id', columns='val_set_name', values='numer')
                      .rename_axis(columns=None))
            dfy = dfy.merge(dfq, on='recip_id', how='left')

            # Add hacked versions of MPM for high value drugs
            if year == 2016:
                dfq = pd.read_pickle(FS().derived_p+f'create_mpm/output/{state}_hvc_drugs_mpm.pkl')
                dfy = dfy.merge(dfq, on=['recip_id', 'year'], how='left')

            # Create some LVC:
            dfq = pd.read_pickle(path+'URI_HEDIS.pkl')
            dfq = dfq[dfq.claim_date.dt.year.eq(year)]
            # Turn into improper
            dfq['numer'] = 1-dfq['numer']
            dfq = dfq.groupby('recip_id').numer.sum().to_frame('lvc_uri').reset_index()
            dfy = dfy.merge(dfq, on='recip_id', how='left')

            # Bring over PQIs
            PQIs = ['PQI01', 'PQI05', 'PQI08', 'PQI15']
            dfpqi = pd.concat([pd.read_pickle(path+f'{pqi}.pkl') for pqi in PQIs])
            dfpqi = dfpqi[dfpqi.claim_date.dt.year.eq(year)]
            dfpqi = dfpqi.groupby('recip_id').claim_date.nunique().to_frame('pqi_numdays')

            dfy = dfy.merge(dfpqi, on='recip_id', how='left')
            # Should only be applicable to 18-65
            dfy.loc[dfy.recip_age.ge(18) & dfy['pqi_numdays'].isnull(), 'pqi_numdays'] = 0
            dfy.loc[dfy.recip_age.lt(18), 'pqi_numdays'] = np.NaN

            # DO each LVC seaprately
            pqi_l = [('pqi_diab', ['PQI01.pkl']),
                     ('pqi_chf', ['PQI08.pkl']),
                     ('pqi_copd', ['PQI05.pkl', 'PQI15.pkl'])]

            for metric, f_list in pqi_l:
                dfpqi = pd.concat([pd.read_pickle(path+f) for f in f_list])
                dfpqi = dfpqi[dfpqi.claim_date.dt.year.eq(year)]
                dfpqi = dfpqi.groupby('recip_id').claim_date.nunique().to_frame(f'{metric}_numdays')

                dfy = dfy.merge(dfpqi, on='recip_id', how='left')
                # Should only be applicable to 18-65
                dfy.loc[dfy.recip_age.ge(18) & dfy[f'{metric}_numdays'].isnull(), f'{metric}_numdays'] = 0
                dfy.loc[dfy.recip_age.lt(18), f'{metric}_numdays'] = np.NaN


            # Bring over the rest of routine LVC:
            lvc_l =[('lvc_ct_rhin', 'LVC_ct_scan_for_acute_uncomp_rhinosinusitis.pkl'),
                    ('lvc_abd_ct', 'LVC_abdomen_ct_combined_studies.pkl'),
                    ('lvc_arthro', 'LVC_arthroscopic_surgery_for_knee_arthritis.pkl'),
                    ('lvc_brain_ct', 'LVC_simultaneous_brain_and_sinus_ct.pkl'),
                    ('lvc_hi_sync', 'LVC_head_imaging_for_syncope.pkl'),
                    ('lvc_thorax', 'LVC_thorax_ct_combined_studies.pkl'),
                    ('lvc_hi_ha', 'LVC_head_imaging_uncomp_headache.pkl'),
                    ('lvc_si_bp', 'LVC_spinal_injections_for_back_pain.pkl'),
                    ('lvc_eeg_ha', 'LVC_eeg_for_headaches.pkl'),
                    ('lvc_im_bp', 'LVC_imaging_for_nonspecific_lbp_HEDIS.pkl')
                   ]
            for cname, f in lvc_l:
                dfq = pd.read_pickle(path+f)
                dfq = dfq[dfq.claim_date.dt.year.eq(year)]
                dfq = (dfq.groupby('recip_id').agg({'numer': 'mean', 'denom': 'sum'})
                          .add_prefix(f'{cname}_'))
                dfy = dfy.merge(dfq, how='left', on='recip_id')

            # Bring over pregnancy related pain medication
            dfpr = pd.read_pickle(FS().derived_p+'create_pregnancy_pain/output/'
                                  +f'{state}_preg_pain.pkl')
            dfy = dfy.merge(dfpr, on='recip_id', how='left')

            # Bring over ATC spending
            dfatc = pd.read_pickle(FS().derived_p+'create_atc_spending/output/'
                                   +f'{state}_atc_spend.pkl')
            dfatc = dfatc[dfatc.index.get_level_values('year') == year]

            dfy = dfy.merge(dfatc, on='recip_id', how='left')
            dfy[dfatc.columns] = dfy[dfatc.columns].fillna(0)

            # Add indicators for hypertension
            if year == 2016:
                dfq = pd.read_pickle(FS().derived_p+f'create_hyp_indicator/output/{state}_hyp_indicator.pkl')
                dfy = dfy.merge(dfq, on=['recip_id', 'year'], how='left')
                dfy['cond_hypertension'] = dfy['cond_hypertension'].fillna(0)

            # Create HCC categories.
            grps = [('asthma', ['hcc_161.2','hcc_161.1', 'hcc_160']),
                    ('depressive_bipolar_psychotic', ['hcc_88', 'hcc_87.1', 'hcc_87.2', 'hcc_90', 
                                                      'hcc_102', 'hcc_103']),
                    ('diabetes', ['hcc_20', 'hcc_21']),
                    ('drug_sud', ['hcc_81', 'hcc_82', 'hcc_83', 'hcc_84']),
                    ('seizure', ['hcc_120']),
                    ('pregnancy', ['hcc_203', 'hcc_204', 'hcc_205', 'hcc_207',
                                    'hcc_208', 'hcc_209', 'hcc_210', 'hcc_211', 'hcc_212']),
                    ('cardiovascular_condition', ['hcc_131', 'hcc_139', 'hcc_138',
                                                   'hcc_142', 'hcc_137', 'hcc_135', 'hcc_132',
                                                   'hcc_130', 'hcc_129', 'hcc_128', 'hcc_127',
                                                   'hcc_126', 'hcc_125'])
                   ]
            for cond, cols in grps:
                dfy[f'cond_{cond}'] = dfy[cols].eq(1).any(1).astype(int)
            dfy['cond_any'] = dfy.filter(like='hcc_').eq(1).any(1).astype(int)
            dfy['cond_N'] = dfy.filter(like='hcc_').sum(axis=1)
            dfy['cond_N_condnl'] = dfy.filter(like='hcc_').sum(axis=1).replace(0, np.NaN)

            dfy.to_pickle(output_p+f'{state}_analytic_table_yearly_{year}.pkl')


def paper1_predict_risk():
    """
    Measure of risk based on HCCs and demographic controls (except race)
    """
    output_p, temp_p, log_p, code_p = gen_paths(FS().analysis_p+'paper1_predict_risk/') 

    path_pre = FS().derived_p+f'create_analytic_tables/output/'
    d = {state: pd.read_pickle(path_pre+f'{state}_analytic_table_yearly_2016.pkl') 
         for state in ['LA', 'KS', 'TN']}

    out_cols = ['all_cost', 'recip_id']
    cont_cols = (['recip_age', 'recip_age_g', 'recip_gender_g', 'recip_zip', 'elig_cat', 'state', 'recip_race_g']
                 + [x for x in d['LA'].columns if 'hcc_' in x])

    res = pd.concat([gp[out_cols+cont_cols] for _, gp in d.items()])

    # Only keep enrollees for which we report results
    res = res[res.recip_race_g.isin([0,1])]
    res['recip_age_g'] = res['recip_age']//1
    res['elig_cat_g'] = res.groupby('elig_cat').ngroup()
    res.to_csv(temp_p+'analytic_table.csv', chunksize=10**6)

    ws.stata_do(do_file=code_p+'predict_risk.do',
                params=[temp_p+'analytic_table.csv',
                        output_p+f'predicted_risk.dta'],
                log_path=log_p)

    dfs = pd.read_stata(output_p+f'predicted_risk.dta')
    dfs['xb_all_cost_norm'] = dfs['xb_all_cost']/dfs['xb_all_cost'].mean()


    dfs[['recip_id', 'recip_race_g', 'all_cost', 'elig_cat', 
         'xb_all_cost', 'xb_all_cost_norm']].to_pickle(output_p+'predicted_risk.pkl')

    # Delete temp file
    os.remove(temp_p+'analytic_table.csv')
    os.remove(output_p+f'predicted_risk.dta')


def paper1_predicted_risk_bias_plot():
    """

    """
    output_p = FS().analysis_p+'paper1_predicted_risk_bias_plot/output/'

    df = pd.read_pickle(FS().analysis_p+'paper1_predict_risk/output/predicted_risk.pkl')
    df['pct'] = pd.qcut(df['xb_all_cost'], 50, labels=False)
    df['dcl'] = pd.qcut(df['xb_all_cost'], 10, labels=False)

    cd = {'White': '#BA4B2F', 'Black': '#6699CC'}

    for truncate in [True, False]:
        fig, ax = plt.subplots(figsize=(6, 7))
        ax.tick_params(labelsize=12, axis='both')

        s1 = (df.groupby([df['pct']*2+1, df['recip_race_g'].map({0: 'White', 1: 'Black'})])
              ['all_cost'].agg(['mean', 'sem']).unstack(-1))

        s = (df.groupby([df['dcl']*10+5, df['recip_race_g'].map({0: 'White', 1: 'Black'})])
                ['all_cost'].agg(['mean', 'sem']).unstack(-1))

        (s1.xs('mean', axis=1)
           .rename(columns={'Black': 'Non-Hispanic Black enrollees',
                            'White': 'Non-Hispanic White enrollees'})
           .plot(ax=ax, lw=0, ms=4, marker='+', color=[cd['Black'], cd['White']]))

        for race in ['White', 'Black']:
            ax.errorbar(x=s.index, y=s[('mean', race)], yerr=s[('sem', race)]*1.96,
                        lw=0, marker='o', elinewidth=3, ms=5, color=cd[race])

        for race in ['White', 'Black']:
            lowess = sm.nonparametric.lowess(np.log(s1[('mean', race)]), s1.index, frac=0.29)
            ax.plot(lowess[:, 0], np.exp(lowess[:, 1]), color=cd[race])


        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[0:2], labels[0:2], title=None, fontsize=12)
        ax.set_xlabel('Percentiles of Predicted Risk', fontsize=13)
        ax.set_ylabel('Annual Spending ($)', fontsize=13)

        ax.grid(zorder=-2, alpha=0.5, color='#ABABAB')

        ax.set_yscale('log', basey=10)
        if truncate:
            ax.set_ylim(500, 17000)

        ax.set_yticks([500, 1000, 3000, 8000, 20000])
        ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())

        plt.tight_layout()
        plt.savefig(output_p+f'prediction_bias{"_truncated"*truncate}.png', dpi=300)
        plt.savefig(output_p+f'prediction_bias{"_truncated"*truncate}.eps', dpi=300)
        plt.savefig(output_p+f'prediction_bias{"_truncated"*truncate}.jpg', dpi=300)
        plt.show()


def paper1_predicted_risk_bias_plot_age_split():
    """

    """
    output_p = FS().analysis_p+'paper1_predicted_risk_bias_plot_age_split/output/'

    df = pd.read_pickle(FS().analysis_p+'paper1_predict_risk/output/predicted_risk.pkl')

    # Merge for recip age
    path_pre = FS().derived_p+f'create_analytic_tables/output/'
    d = {state: pd.read_pickle(path_pre+f'{state}_analytic_table_yearly_2016.pkl') 
         for state in ['LA', 'KS', 'TN']}
    age = pd.concat([gp[['recip_id', 'recip_age']] for _, gp in d.items()])
    df = df.merge(age, on=['recip_id'], how='left')

    # Create Deciles
    df['pct'] = pd.qcut(df.xb_all_cost.rank(method='first'), 50, labels=False)
    df['dcl'] = pd.qcut(df.xb_all_cost.rank(method='first'), 10, labels=False)

    grpr = df['recip_age'].le(18).map({True: 'child', False: 'adult'})
    for truncate in [True]:
        fig, axs = plt.subplots(figsize=(9, 10.5), ncols=2, sharey=True)
        axs = axs.flatten()
        # ax.tick_params(labelsize=12, axis='both')
        cd = {'White': '#BA4B2F', 'Black': '#6699CC'}
        axis = {'child': 0, 'adult': 1}

        for label, gp in df.groupby(grpr):
            ax = axis[label]
            
            # Print
            print(f'{label.title()}: axis {ax}')
                    
            # Group data for graphs
            s1 = (gp.groupby([gp['pct']*2+1, gp['recip_race_g'].map({0: 'White', 1: 'Black'})])
                  ['all_cost'].agg(['mean', 'sem']).unstack(-1))

            s = (gp.groupby([gp['dcl']*10+5, gp['recip_race_g'].map({0: 'White', 1: 'Black'})])
                    ['all_cost'].agg(['mean', 'sem']).unstack(-1))
            
            # Plot scatter plot
            (s1.xs('mean', axis=1)
               .rename(columns={'Black': 'Non-Hispanic Black enrollees',
                                'White': 'Non-Hispanic White enrollees'})
               .plot(ax=axs[ax], lw=0, ms=4, marker='+', color=[cd['Black'], cd['White']]))
            
            # Plot error bar area
            for race in ['White', 'Black']:
                axs[ax].errorbar(x=s.index, y=s[('mean', race)], yerr=s[('sem', race)]*1.96,
                            lw=0, marker='o', elinewidth=3, ms=5, color=cd[race])

            for race in ['White', 'Black']:
                lowess = sm.nonparametric.lowess(np.log(s1[('mean', race)]), s1.index, frac=0.29)
                axs[ax].plot(lowess[:, 0], np.exp(lowess[:, 1]), color=cd[race])

            # Plot add-ons
            axs[ax].set_title(label.title(), fontsize=13, fontweight='bold')
            axs[ax].set_xlabel('Percentiles of Estimated Risk', fontsize=13)
            axs[ax].set_ylabel('Annual Spending ($)', fontsize=13)
            
            axs[ax].grid(zorder=-2, alpha=0.5, color='#ABABAB')
            
            axs[ax].set_yscale('log', basey=10)
            
            axs[ax].set_yticks([500, 1000, 3000, 8000, 20000])
            axs[ax].get_yaxis().set_major_formatter(ticker.ScalarFormatter())
            
            # Legend
            h, l = axs[ax].get_legend_handles_labels()
            axs[ax].get_legend().remove()

        # Truncate y-axis
        if truncate:
            plt.setp(axs, ylim=(500,17000))
            
        # X-axis Title
        #fig.text(0.5, 0.04, 'common X', ha='center')

        # Legend
        fig.legend(h,l, loc='lower center',  bbox_to_anchor=(0.5, -0.08, 0, 0), title=None, fontsize=12)

        # Save
        plt.tight_layout()
        plt.savefig(output_p+f'prediction_bias{"_truncated"*truncate}.png', dpi=300)
        plt.savefig(output_p+f'prediction_bias{"_truncated"*truncate}.eps', dpi=300)
        plt.savefig(output_p+f'prediction_bias{"_truncated"*truncate}.jpg', dpi=300)
        plt.show()


def paper1_table1():
    """
    Create summary statistics of the population across the three states
    """

    output_p = FS().analysis_p+'paper1_table1/output/'

    path_pre = FS().derived_p+f'create_analytic_tables/output/'
    d = {state: pd.read_pickle(path_pre+f'{state}_analytic_table_yearly_2016.pkl') 
         for state in ['LA', 'KS', 'TN']}

    # Create eligibility booleans
    for k,gp in d.items():
        d[k] = pd.concat([gp, pd.get_dummies(gp['elig_cat']).add_prefix('elig_')], axis=1)

    # Create Urban Boolean/FFS
    for k,gp in d.items():
        gp['urban'] = gp['urban_category'].eq('urban').astype(int)
        gp['mmc'] = gp.plan_id.ne('FFS').astype(int)
        d[k] = gp

    # Moved this into create analytic tables
    # # Create HCC categories.
    # grps = [('asthma', ['hcc_161.2','hcc_161.1', 'hcc_160']),
    #         ('depressive_bipolar_psychotic', ['hcc_88', 'hcc_87.1', 'hcc_87.2', 'hcc_90', 
    #                                           'hcc_102', 'hcc_103']),
    #         ('diabetes', ['hcc_20', 'hcc_21']),
    #         ('drug_sud', ['hcc_81', 'hcc_82', 'hcc_83', 'hcc_84']),
    #         ('seizure', ['hcc_120']),
    #         ('pregnancy', ['hcc_203', 'hcc_204', 'hcc_205', 'hcc_207',
    #                         'hcc_208', 'hcc_209', 'hcc_210', 'hcc_211', 'hcc_212']),
    #         ('cardiovascular_condition', ['hcc_131', 'hcc_139', 'hcc_138',
    #                                        'hcc_142', 'hcc_137', 'hcc_135', 'hcc_132',
    #                                        'hcc_130', 'hcc_129', 'hcc_128', 'hcc_127',
    #                                        'hcc_126', 'hcc_125'])
    #        ]
    # for k,gp in d.items():
    #     for cond, cols in grps:
    #         gp[f'cond_{cond}'] = gp[cols].eq(1).any(1).astype(int)
    #     gp['cond_any'] = gp.filter(like='hcc_').eq(1).any(1).astype(int)
    #     gp['cond_N'] = gp.filter(like='hcc_').sum(axis=1)
    #     gp['cond_N_condnl'] = gp.filter(like='hcc_').sum(axis=1).replace(0, np.NaN)
    #     d[k] = gp     

    # Bring over predicted risk
    dfr = pd.read_pickle(FS().analysis_p+'paper1_predict_risk/output/predicted_risk.pkl')
    dfr1 = dfr.set_index('recip_id')['xb_all_cost']
    dfr2 = dfr.set_index('recip_id')['xb_all_cost_norm']
    for k,gp in d.items():
        gp['pred_risk'] = gp.recip_id.map(dfr1)
        gp['pred_risk_norm'] = gp.recip_id.map(dfr2)
        d[k] = gp    

    # Run regressions to recover Means and STDs (and p-value) for the regressions
    outcomes = ['recip_age', 'recip_gender_g', 'adi_natnl', 'urban', 'mmc', 'seg_index', 
                'elig_disability',  'elig_child', 'elig_adult', 'elig_other',
                'cond_any', 'pred_risk', 'pred_risk_norm', 'cond_N', 'cond_N_condnl',
                'cond_asthma', 'cond_depressive_bipolar_psychotic', 'cond_diabetes',
                'cond_drug_sud', 'cond_seizure', 'cond_pregnancy', 'cond_cardiovascular_condition']

    dres = {}
    for outcome in outcomes:
        res = pd.concat([df[['recip_race_g', outcome]] for _,df in d.items()])
        res = res[res.recip_race_g.isin([0,1])]

        mod = smf.ols(f'{outcome} ~ C(recip_race_g) - 1', data=res)
        fit = mod.fit(cov_type='HC2')

        # To scale standard errors to standard devs. 
        s = pd.Series(np.sqrt(res.groupby('recip_race_g').size().to_numpy()), 
                      index=fit.bse.index)
        stds = np.sqrt(np.diag(fit.cov_HC2))*s

        sout = pd.concat([fit.params.to_frame('coeff'),
                          stds.to_frame('stds')], axis=1)
        sout.index = sout.index.map({'C(recip_race_g)[0]': 'White', 'C(recip_race_g)[1]': 'Black'})
        N = res.groupby('recip_race_g')[outcome].sum()
        N.index = N.index.map({0: 'White', 1: 'Black'})
        
        sout = pd.concat([sout, N.to_frame('counts')], axis=1)

        # Now do the overall population
        mod = smf.ols(f'{outcome} ~ 1', data=res)
        fit = mod.fit(cov_type='HC2')

        # To scale standard errors to standard devs. 
        s = pd.Series(np.sqrt(res.shape[0]), index=fit.bse.index)
        stds = np.sqrt(np.diag(fit.cov_HC2))*s

        sout2 = pd.concat([fit.params.to_frame('coeff'),
                          stds.to_frame('stds')], axis=1)
        sout2.index = sout2.index.map({'Intercept': 'All'})
        N = pd.Series(res[outcome].sum(), index=['All'])
        sout2 = pd.concat([sout2, N.to_frame('counts')], axis=1)
        

        dres[outcome] = pd.concat([sout, sout2])

    res = pd.concat(dres, axis=1).stack(0).unstack(0)


    val_l = ['recip_age', 'seg_index', 'adi_natnl', 'pred_risk', 'pred_risk_norm', 'cond_N',
             'cond_N_condnl']

    for race in ['Black', 'White', 'All']:
        res[('value', race)] = np.where(res.index.isin(val_l), res[('coeff', race)], 
                                        res[('counts', race)])
        res[('parenthetical', race)] = np.where(res.index.isin(val_l), res[('stds', race)], 
                                                res[('coeff', race)])

    res = res.sort_index(axis=1)
    res.to_csv(output_p+'sample_avg.csv')


def paper1_table1_bystate():
    """
    Create summary statistics of the population across the three states
    """

    output_p = FS().analysis_p+'paper1_table1_bystate/output/'

    path_pre = FS().derived_p+f'create_analytic_tables/output/'
    d = {state: pd.read_pickle(path_pre+f'{state}_analytic_table_yearly_2016.pkl') 
         for state in ['LA', 'KS', 'TN']}

    # Create eligibility booleans
    for k,gp in d.items():
        d[k] = pd.concat([gp, pd.get_dummies(gp['elig_cat']).add_prefix('elig_')], axis=1)

    # Create Urban Boolean/FFS
    for k,gp in d.items():
        gp['urban'] = gp['urban_category'].eq('urban').astype(int)
        gp['mmc'] = gp.plan_id.ne('FFS').astype(int)
        d[k] = gp

    # Moved this into create analytic tables
    # # Create HCC categories.
    # grps = [('asthma', ['hcc_161.2','hcc_161.1', 'hcc_160']),
    #         ('depressive_bipolar_psychotic', ['hcc_88', 'hcc_87.1', 'hcc_87.2', 'hcc_90', 
    #                                           'hcc_102', 'hcc_103']),
    #         ('diabetes', ['hcc_20', 'hcc_21']),
    #         ('drug_sud', ['hcc_81', 'hcc_82', 'hcc_83', 'hcc_84']),
    #         ('seizure', ['hcc_120']),
    #         ('pregnancy', ['hcc_203', 'hcc_204', 'hcc_205', 'hcc_207',
    #                         'hcc_208', 'hcc_209', 'hcc_210', 'hcc_211', 'hcc_212']),
    #         ('cardiovascular_condition', ['hcc_131', 'hcc_139', 'hcc_138',
    #                                        'hcc_142', 'hcc_137', 'hcc_135', 'hcc_132',
    #                                        'hcc_130', 'hcc_129', 'hcc_128', 'hcc_127',
    #                                        'hcc_126', 'hcc_125'])
    #        ]
    # for k,gp in d.items():
    #     for cond, cols in grps:
    #         gp[f'cond_{cond}'] = gp[cols].eq(1).any(1).astype(int)
    #     gp['cond_any'] = gp.filter(like='hcc_').eq(1).any(1).astype(int)
    #     gp['cond_N'] = gp.filter(like='hcc_').sum(axis=1)
    #     gp['cond_N_condnl'] = gp.filter(like='hcc_').sum(axis=1).replace(0, np.NaN)
    #     d[k] = gp     

    # Bring over predicted risk
    dfr = pd.read_pickle(FS().analysis_p+'paper1_predict_risk/output/predicted_risk.pkl')
    dfr1 = dfr.set_index('recip_id')['xb_all_cost']
    dfr2 = dfr.set_index('recip_id')['xb_all_cost_norm']
    for k,gp in d.items():
        gp['pred_risk'] = gp.recip_id.map(dfr1)
        gp['pred_risk_norm'] = gp.recip_id.map(dfr2)
        d[k] = gp    

    #Run regressions to recover Means and STDs (and p-value) for the regressions
    outcomes = ['recip_age', 'recip_gender_g', 'adi_natnl', 'urban', 'mmc', 'seg_index', 
                'elig_disability',  'elig_child', 'elig_adult', 'elig_other',
                'cond_any', 'pred_risk', 'pred_risk_norm', 'cond_N', 'cond_N_condnl',
                'cond_asthma', 'cond_depressive_bipolar_psychotic', 'cond_diabetes',
                'cond_drug_sud', 'cond_seizure', 'cond_pregnancy', 'cond_cardiovascular_condition']


    for state, df in d.items():
        dres = {}
        for outcome in outcomes:
            res = df[['recip_race_g', outcome]]
            res = res[res.recip_race_g.isin([0,1])]

            mod = smf.ols(f'{outcome} ~ C(recip_race_g) - 1', data=res)
            fit = mod.fit(cov_type='HC2')

            # To scale standard errors to standard devs. 
            s = pd.Series(np.sqrt(res.groupby('recip_race_g').size().to_numpy()), 
                          index=fit.bse.index)
            stds = np.sqrt(np.diag(fit.cov_HC2))*s

            sout = pd.concat([fit.params.to_frame('coeff'),
                              stds.to_frame('stds')], axis=1)
            sout.index = sout.index.map({'C(recip_race_g)[0]': 'White', 'C(recip_race_g)[1]': 'Black'})
            N = res.groupby('recip_race_g')[outcome].sum()
            N.index = N.index.map({0: 'White', 1: 'Black'})
            sout = pd.concat([sout, N.to_frame('counts')], axis=1)

            dres[outcome] = sout
        print(state, res.groupby('recip_race_g').size())
        res = pd.concat(dres, axis=1).stack(0).unstack(0)

        val_l = ['recip_age', 'seg_index', 'adi_natnl', 'pred_risk', 'pred_risk_norm', 'cond_N',
                 'cond_N_condnl']

        for race in ['Black', 'White']:
            res[('value', race)] = np.where(res.index.isin(val_l), res[('coeff', race)], 
                                            res[('counts', race)])
            res[('parenthetical', race)] = np.where(res.index.isin(val_l), res[('stds', race)], 
                                                    res[('coeff', race)])

        res = res.sort_index(axis=1)
        res.to_csv(output_p+f'sample_avg_{state}.csv')


def paper1_regressions_yearly():
    """
    Run regressions for paper 1 using the yearly files for 2016
    """

    output_p, temp_p, log_p, code_p = gen_paths(FS().analysis_p+'paper1_regressions_yearly/')

    path_pre = FS().derived_p+f'create_analytic_tables/output/'
    d = {state: pd.read_pickle(path_pre+f'{state}_analytic_table_yearly_2016.pkl') 
         for state in ['LA', 'KS', 'TN']}
        
    out_cols = ['all_cost', 'medical_cost', 'pharm_cost', 'pc_any', 
                'pc_numdays', 'pharm_numclms', 'ednoadmit_numdays', 'inpat_numdays', 'spec_numdays',
                'antihypertensive_numclms', 'diabetes_numclms',
                'statins_numclms', 'asthma_numclms', 
                'antihypertensive_any', 'diabetes_any',
                'statins_any', 'asthma_any', 
                'ed_avoid_numdays', 'lvc_uri', 'tests_imaging_numdays',
                'capch_all', 'capch_1to2', 'capch_2to6', 'capch_7to12', 'capch_12to19',
                'awc_all', 'awc_ch', 'awc_adolesc', 'ha1c', 'adv',
                'chl_ad', 'chl_ch', 'chl_all', 'ccs',
                'bcs', 'ssd', 'mpm_acearb', 'mpm_diur', 'amr', 'poud',
                'pqi_numdays', 'pqi_diab_numdays', 'pqi_chf_numdays', 'pqi_copd_numdays']

    cont_cols = (['recip_race_g', 'recip_age_g', 'recip_gender_g', 'recip_zip', 'elig_cat',
                  'state', 'plan_id_g']
                 + [x for x in d['LA'].columns if 'hcc_' in x])

    res = pd.concat([gp[out_cols+cont_cols] for _, gp in d.items()])
    # So plan control within state
    res['plan_id_g'] = res.groupby(['plan_id_g', 'state']).ngroup()

    # Only keep enrollees for which we report results
    res = res[res.recip_race_g.isin([0, 1])]

    # Other spending outcomes
    res['any_cost'] = res['all_cost'].gt(0).astype(int)
    res['all_cost_log'] = np.log1p(res['all_cost'])
    res['all_cost_wins40'] = res['all_cost'].clip(upper=40000)
    res['all_cost_wins125'] = res['all_cost'].clip(upper=125000)

    res.to_csv(temp_p+'analytic_table.csv', chunksize=10**6)

    ws.stata_do(do_file=code_p+'paper1_main_regs.do',
                params=[temp_p+'analytic_table.csv',
                        output_p+'paper1_main_results.csv'],
                log_path=log_p)

    # Delete temp file
    os.remove(temp_p+'analytic_table.csv')


def paper1_regressions_yearly_age_split():
    """
    Run regressions for paper 1 using the yearly files for 2016
    """

    output_p, temp_p, log_p, code_p = gen_paths(FS().analysis_p+'paper1_regressions_yearly_age_split/')

    path_pre = FS().derived_p+f'create_analytic_tables/output/'
    d = {state: pd.read_pickle(path_pre+f'{state}_analytic_table_yearly_2016.pkl') 
         for state in ['LA', 'KS', 'TN']}
        
    out_cols = ['all_cost', 'medical_cost', 'pharm_cost', 'pc_any', 
                'pc_numdays', 'pharm_numclms', 'ednoadmit_numdays', 'inpat_numdays', 'spec_numdays',
                'antihypertensive_numclms', 'diabetes_numclms',
                'statins_numclms', 'asthma_numclms', 
                'antihypertensive_any', 'diabetes_any',
                'statins_any', 'asthma_any', 
                'ed_avoid_numdays', 'lvc_uri', 'tests_imaging_numdays',
                'capch_all', 'capch_1to2', 'capch_2to6', 'capch_7to12', 'capch_12to19',
                'awc_all', 'awc_ch', 'awc_adolesc', 'ha1c', 'adv',
                'chl_ad', 'chl_ch', 'chl_all', 'ccs',
                'bcs', 'ssd', 'mpm_acearb', 'mpm_diur', 'amr', 'poud',
                'pqi_numdays', 'pqi_diab_numdays', 'pqi_chf_numdays', 'pqi_copd_numdays']

    cont_cols = (['recip_race_g', 'recip_age_g', 'recip_age', 'recip_gender_g', 'recip_zip', 'elig_cat',
                  'state', 'plan_id_g', 
                  'cond_asthma', 'cond_diabetes', 'cond_cardiovascular_condition', 'cond_hypertension',
                  'asthma_mpm', 'diabetes_mpm', 'statins_mpm', 'antihypertensive_mpm']
                 + [x for x in d['LA'].columns if 'hcc_' in x])

    res = pd.concat([gp[out_cols+cont_cols] for _, gp in d.items()])
    # So plan control within state
    res['plan_id_g'] = res.groupby(['plan_id_g', 'state']).ngroup()

    # Only keep enrollees for which we report results
    res = res[res.recip_race_g.isin([0, 1])]

    # Other spending outcomes
    res['any_cost'] = res['all_cost'].gt(0).astype(int)
    res['all_cost_log'] = np.log1p(res['all_cost'])
    res['all_cost_wins40'] = res['all_cost'].clip(upper=40000)
    res['all_cost_wins125'] = res['all_cost'].clip(upper=125000)

    # Conditional HCCs - Any
    res['asthma_any_condnl'] = res['asthma_any'].where(res['cond_asthma'].eq(True))
    res['diabetes_any_condnl'] = res['diabetes_any'].where(res['cond_diabetes'].eq(True))
    res['statins_any_condnl'] = res['statins_any'].where(res['cond_cardiovascular_condition'].eq(True))
    res['antihypertensive_any_condnl'] = res['antihypertensive_any'].where(res['cond_hypertension'].eq(True))

    # Conditional HCCs - Number of claims
    res['asthma_numclms_condnl'] = res['asthma_numclms'].where(res['cond_asthma'].eq(True))
    res['diabetes_numclms_condnl'] = res['diabetes_numclms'].where(res['cond_diabetes'].eq(True))
    res['statins_numclms_condnl'] = res['statins_numclms'].where(res['cond_cardiovascular_condition'].eq(True))
    res['antihypertensive_numclms_condnl'] = res['antihypertensive_numclms'].where(res['cond_hypertension'].eq(True))

    # Conditional MPM measures
    res['asthma_mpm_condnl'] = res['asthma_mpm'].where(res['cond_asthma'].eq(True))
    res['diabetes_mpm_condnl'] = res['diabetes_mpm'].where(res['cond_diabetes'].eq(True))
    res['statins_mpm_condnl'] = res['statins_mpm'].where(res['cond_cardiovascular_condition'].eq(True))
    res['antihypertensive_mpm_condnl'] = res['antihypertensive_mpm'].where(res['cond_hypertension'].eq(True))

    grpr = res['recip_age'].le(18).map({True: 'child', False: 'adult'})
    for label, gp in res.groupby(grpr):

        gp.to_csv(temp_p+f'analytic_table_{label}.csv', chunksize=10**6)

        cols = (out_cols
                + ['asthma_any_condnl', 'diabetes_any_condnl', 'statins_any_condnl', 'antihypertensive_any_condnl',
                   'asthma_mpm_condnl', 'diabetes_mpm_condnl', 'statins_mpm_condnl', 'antihypertensive_mpm_condnl',
                   'asthma_numclms_condnl', 'diabetes_numclms_condnl', 'statins_numclms_condnl', 'antihypertensive_numclms_condnl']
                + ['any_cost'])
        Nreg = (gp[cols + ['recip_race_g']].groupby('recip_race_g')[cols].count()
                .T
                .rename(columns={0:'White', 1:'Black'}))      
        Nreg.to_csv(output_p+f'num_observations_{label}.csv')

        ws.stata_do(do_file=code_p+'paper1_main_regs.do',
                    params=[temp_p+f'analytic_table_{label}.csv',
                            output_p+f'paper1_main_results_{label}.csv'],
                    log_path=log_p)

        # Delete temp file
        os.remove(temp_p+f'analytic_table_{label}.csv')


def paper1_regressions_yearly_CRG():
    """
    Run regressions for paper 1 using the yearly files
    """

    output_p, temp_p, log_p, code_p = gen_paths(FS().analysis_p+'paper1_regressions_yearly_CRG/')

    path_pre = FS().derived_p+f'create_analytic_tables/output/'
    d = {state: pd.read_pickle(path_pre+f'{state}_analytic_table_yearly_2016.pkl') 
         for state in ['LA', 'KS', 'TN']}
        
    out_cols = ['all_cost', 'medical_cost', 'pharm_cost', 'pc_any', 
                'pc_numdays', 'pharm_numclms', 'ednoadmit_numdays', 'inpat_numdays', 'spec_numdays',
                'antihypertensive_numclms', 'diabetes_numclms',
                'statins_numclms',
                'ed_avoid_numdays', 'lvc_uri', 'tests_imaging_numdays',
                'capch_all', 'capch_1to2', 'capch_2to6', 'capch_7to12', 'capch_12to19',
                'awc_all', 'awc_ch', 'awc_adolesc', 'ha1c', 'adv',
                'chl_ad', 'chl_ch', 'chl_all', 'ccs',
                'bcs', 'ssd', 'mpm_acearb', 'mpm_diur', 'amr', 'poud',
                'pqi_numdays', 'pqi_diab_numdays', 'pqi_chf_numdays', 'pqi_copd_numdays']

    cont_cols = (['recip_race_g', 'recip_age_g', 'recip_gender_g', 'recip_zip', 'elig_cat',
                  'state', 'plan_id_g', 'health_status']
                 + [x for x in d['LA'].columns if 'crg' in x])

    res = pd.concat([gp[out_cols+cont_cols] for _, gp in d.items()])
    # So plan control within state
    res['plan_id_g'] = res.groupby(['plan_id_g', 'state']).ngroup()

    # Only keep enrollees for which we report results
    res = res[res.recip_race_g.isin([0,1])]

    # Other spending outcomes
    res['any_cost'] = res['all_cost'].gt(0).astype(int)
    res['all_cost_log'] = np.log1p(res['all_cost'])
    res['all_cost_wins40'] = res['all_cost'].clip(upper=40000)
    res['all_cost_wins125'] = res['all_cost'].clip(upper=125000)

    res.to_csv(temp_p+'analytic_table.csv', chunksize=10**6)

    ws.stata_do(do_file=code_p+'paper1_main_regs_crg.do',
                params=[temp_p+'analytic_table.csv',
                        output_p+'paper1_main_results_crg.csv'],
                log_path=log_p)

    # Delete temp file
    os.remove(temp_p+'analytic_table.csv')


def paper1_regressions_preg():
    """
    Run regressions for paper 1 using the yearly files
    """

    output_p, temp_p, log_p, code_p = gen_paths(FS().analysis_p+'paper1_regressions_preg/')

    path_pre = FS().derived_p+f'create_analytic_tables/output/'
    d = {state: pd.read_pickle(path_pre+f'{state}_analytic_table_yearly_2016.pkl') 
         for state in ['LA', 'KS', 'TN']}
        
    out_cols = ['days_to_pain_write', 'days_to_pain_fill', 'ever_pain', 
                'opioid', 'analgesic', 'supply_days']
    cont_cols = (['recip_race_g', 'recip_age_g', 'recip_gender_g', 'recip_zip', 'elig_cat',
                  'state', 'plan_id_g', 'drg', 'severity_of_illness']
                 + [x for x in d['LA'].columns if 'hcc_' in x])

    res = pd.concat([gp[out_cols+cont_cols] for _, gp in d.items()])
    # So plan control within state
    res['plan_id_g'] = res.groupby(['plan_id_g', 'state']).ngroup()

    # Only keep enrollees for which we report results
    res = res[res.recip_race_g.isin([0,1])]
    # ALso keep only pregnancy enrollees
    res = res[res.ever_pain.notnull()]

    res.to_csv(temp_p+'analytic_table.csv', chunksize=10**6)

    ws.stata_do(do_file=code_p+'paper1_preg_regs.do',
                params=[temp_p+'analytic_table.csv',
                        output_p+'paper1_main_results.csv'],
                log_path=log_p)

    # Delete temp file
    os.remove(temp_p+'analytic_table.csv')


def paper1_regressions_atc():
    """
    Run regressions for paper 1 using the yearly files
    """

    output_p, temp_p, log_p, code_p = gen_paths(FS().analysis_p+'paper1_regressions_atc/')

    path_pre = FS().derived_p+f'create_analytic_tables/output/'
    d = {state: pd.read_pickle(path_pre+f'{state}_analytic_table_yearly_2016.pkl') 
         for state in ['LA', 'KS', 'TN']}
        
    out_cols = [x for x in d['LA'].columns if 'atc3_' in x and '_numclms' in x]
    cont_cols = (['recip_race_g', 'recip_age_g', 'recip_gender_g', 'recip_zip', 'elig_cat',
                  'state', 'plan_id_g', 'drg', 'severity_of_illness']
                 + [x for x in d['LA'].columns if 'hcc_' in x])

    res = pd.concat([gp[out_cols+cont_cols] for _, gp in d.items()])
    # So plan control within state
    res['plan_id_g'] = res.groupby(['plan_id_g', 'state']).ngroup()

    # Only keep enrollees for which we report results
    res = res[res.recip_race_g.isin([0,1])]

    res.to_csv(temp_p+'analytic_table.csv', chunksize=10**6)

    ws.stata_do(do_file=code_p+'paper1_atc_regs.do',
                params=[temp_p+'analytic_table.csv',
                        output_p+'paper1_atc_results.csv'],
                log_path=log_p)

    # Delete temp file
    os.remove(temp_p+'analytic_table.csv')


def paper1_regressions_atc_with_attr_prov():
    """
    Run regressions for paper 1 using the yearly files
    """

    output_p, temp_p, log_p, code_p = gen_paths(FS().analysis_p
                                                +'paper1_regressions_atc_with_attr_prov/')

    path_pre = FS().derived_p+f'create_analytic_tables/output/'
    d = {state: pd.read_pickle(path_pre+f'{state}_analytic_table_yearly_2016.pkl') 
         for state in ['LA', 'KS', 'TN']}
        
    out_cols = [x for x in d['LA'].columns if 'atc3_' in x and '_numclms' in x]
    cont_cols = (['recip_race_g', 'recip_age_g', 'recip_gender_g', 'recip_zip', 'elig_cat',
                  'state', 'plan_id_g', 'drg', 'severity_of_illness', 'attr_prov']
                 + [x for x in d['LA'].columns if 'hcc_' in x])

    res = pd.concat([gp[out_cols+cont_cols] for _, gp in d.items()])
    # So plan control within state
    res['plan_id_g'] = res.groupby(['plan_id_g', 'state']).ngroup()

    # Only keep enrollees for which we report results
    res = res[res.recip_race_g.isin([0,1])]

    res.to_csv(temp_p+'analytic_table.csv', chunksize=10**6)

    ws.stata_do(do_file=code_p+'paper1_atc_regs_with_prov.do',
                params=[temp_p+'analytic_table.csv',
                        output_p+'paper1_atc_results.csv'],
                log_path=log_p)

    # Delete temp file
    os.remove(temp_p+'analytic_table.csv')


def paper1_lvc_regressions_yearly():
    """
    Run regressions for paper 1 using the yearly files
    """

    output_p, temp_p, log_p, code_p = gen_paths(FS().analysis_p+'paper1_lvc_regressions_yearly/')

    path_pre = FS().derived_p+f'create_analytic_tables/output/'
    d = {state: pd.read_pickle(path_pre+f'{state}_analytic_table_yearly_2016.pkl') 
         for state in ['LA', 'KS', 'TN']}
        
    out_cols = ['lvc_ct_rhin', 'lvc_abd_ct', 'lvc_arthro', 'lvc_brain_ct',
                'lvc_hi_sync', 'lvc_thorax', 'lvc_hi_ha', 'lvc_si_bp',
                'lvc_eeg_ha', 'lvc_im_bp']
    cont_cols = (['recip_race_g', 'recip_age_g', 'recip_gender_g', 'recip_zip', 'elig_cat',
                  'state', 'plan_id_g', 'attr_prov']
                 + [x for x in d['LA'].columns if 'hcc_' in x])

    res = pd.concat([gp[[f'{x}_numer' for x in out_cols]
                        +[f'{x}_denom' for x in out_cols]
                        +cont_cols] for _, gp in d.items()])
    # So plan control within state
    res['plan_id_g'] = res.groupby(['plan_id_g', 'state']).ngroup()

    # Only keep enrollees for which we report results
    res = res[res.recip_race_g.isin([0,1])]
    res.to_csv(temp_p+'analytic_table.csv', chunksize=10**6)

    ws.stata_do(do_file=code_p+'paper1_lvc_regs.do',
                params=[temp_p+'analytic_table.csv',
                        output_p+'paper1_lvc_results.csv'],
                log_path=log_p)

    # Delete temp file
    os.remove(temp_p+'analytic_table.csv')


def paper1_regressions_yearly_with_attr_prov():
    """
    Run regressions for paper 1 using the yearly files
    """

    output_p, temp_p, log_p, code_p = gen_paths(FS().analysis_p
                                               +'paper1_regressions_yearly_with_attr_prov/')

    path_pre = FS().derived_p+f'create_analytic_tables/output/'
    d = {state: pd.read_pickle(path_pre+f'{state}_analytic_table_yearly_2016.pkl') 
         for state in ['LA', 'KS', 'TN']}
        
    out_cols = ['all_cost', 'medical_cost', 'pharm_cost', 'pc_any', 
                'pc_numdays', 'pharm_numclms', 'ednoadmit_numdays', 'inpat_numdays', 'spec_numdays',
                'antihypertensive_numclms', 'diabetes_numclms',
                'statins_numclms', 'asthma_numclms', 
                'antihypertensive_any', 'diabetes_any',
                'statins_any', 'asthma_any', 
                'ed_avoid_numdays', 'lvc_uri', 'tests_imaging_numdays',
                'capch_all', 'capch_1to2', 'capch_2to6', 'capch_7to12', 'capch_12to19',
                'awc_all', 'awc_ch', 'awc_adolesc', 'ha1c', 'adv',
                'chl_ad', 'chl_ch', 'chl_all', 'ccs',
                'bcs', 'ssd', 'mpm_acearb', 'mpm_diur', 'amr', 'poud',
                'pqi_numdays', 'pqi_diab_numdays', 'pqi_chf_numdays', 'pqi_copd_numdays']
    cont_cols = (['recip_race_g', 'recip_age_g', 'recip_gender_g', 'recip_zip', 'elig_cat',
                  'state', 'plan_id_g', 'attr_prov']
                 + [x for x in d['LA'].columns if 'hcc_' in x])

    res = pd.concat([gp[out_cols+cont_cols] for _, gp in d.items()])
    # So plan control within state
    res['plan_id_g'] = res.groupby(['plan_id_g', 'state']).ngroup()

    # Only keep enrollees for which we report results
    res = res[res.recip_race_g.isin([0,1])]

    # Other outcomes of cost
    res['any_cost'] = res['all_cost'].gt(0).astype(int)
    res['all_cost_log'] = np.log1p(res['all_cost'])
    res['all_cost_wins40'] = res['all_cost'].clip(upper=40000)
    res['all_cost_wins125'] = res['all_cost'].clip(upper=125000)

    res.to_csv(temp_p+'analytic_table.csv', chunksize=10**6)

    ws.stata_do(do_file=code_p+'paper1_main_regs_with_prov.do',
                params=[temp_p+'analytic_table.csv',
                        output_p+'paper1_main_results_with_prov.csv'],
                log_path=log_p)

    # Delete temp file
    os.remove(temp_p+'analytic_table.csv')


def paper1_regressions_yearly_with_attr_prov_age_split():
    """
    Run regressions for paper 1 using the yearly files
    """

    output_p, temp_p, log_p, code_p = gen_paths(FS().analysis_p
                                               +'paper1_regressions_yearly_with_attr_prov_age_split/')

    path_pre = FS().derived_p+f'create_analytic_tables/output/'
    d = {state: pd.read_pickle(path_pre+f'{state}_analytic_table_yearly_2016.pkl') 
         for state in ['LA', 'KS', 'TN']}
        
    out_cols = ['all_cost', 'medical_cost', 'pharm_cost', 'pc_any', 
                'pc_numdays', 'pharm_numclms', 'ednoadmit_numdays', 'inpat_numdays', 'spec_numdays',
                'antihypertensive_numclms', 'diabetes_numclms',
                'statins_numclms', 'asthma_numclms', 
                'antihypertensive_any', 'diabetes_any',
                'statins_any', 'asthma_any', 
                'ed_avoid_numdays', 'lvc_uri', 'tests_imaging_numdays',
                'capch_all', 'capch_1to2', 'capch_2to6', 'capch_7to12', 'capch_12to19',
                'awc_all', 'awc_ch', 'awc_adolesc', 'ha1c', 'adv',
                'chl_ad', 'chl_ch', 'chl_all', 'ccs',
                'bcs', 'ssd', 'mpm_acearb', 'mpm_diur', 'amr', 'poud',
                'pqi_numdays', 'pqi_diab_numdays', 'pqi_chf_numdays', 'pqi_copd_numdays']
    cont_cols = (['recip_race_g', 'recip_age_g', 'recip_age', 'recip_gender_g', 'recip_zip', 'elig_cat',
                  'state', 'plan_id_g', 'attr_prov',
                  'cond_asthma', 'cond_diabetes', 'cond_cardiovascular_condition', 'cond_hypertension',
                  'asthma_mpm', 'diabetes_mpm', 'statins_mpm', 'antihypertensive_mpm']
                 + [x for x in d['LA'].columns if 'hcc_' in x])

    res = pd.concat([gp[out_cols+cont_cols] for _, gp in d.items()])
    # So plan control within state
    res['plan_id_g'] = res.groupby(['plan_id_g', 'state']).ngroup()

    # Only keep enrollees for which we report results
    res = res[res.recip_race_g.isin([0,1])]

    # Other outcomes of cost
    res['any_cost'] = res['all_cost'].gt(0).astype(int)
    res['all_cost_log'] = np.log1p(res['all_cost'])
    res['all_cost_wins40'] = res['all_cost'].clip(upper=40000)
    res['all_cost_wins125'] = res['all_cost'].clip(upper=125000)

    # Conditional HCCs - Any
    res['asthma_any_condnl'] = res['asthma_any'].where(res['cond_asthma'].eq(True))
    res['diabetes_any_condnl'] = res['diabetes_any'].where(res['cond_diabetes'].eq(True))
    res['statins_any_condnl'] = res['statins_any'].where(res['cond_cardiovascular_condition'].eq(True))
    res['antihypertensive_any_condnl'] = res['antihypertensive_any'].where(res['cond_hypertension'].eq(True))

    # Conditional HCCs - Number of claims
    res['asthma_numclms_condnl'] = res['asthma_numclms'].where(res['cond_asthma'].eq(True))
    res['diabetes_numclms_condnl'] = res['diabetes_numclms'].where(res['cond_diabetes'].eq(True))
    res['statins_numclms_condnl'] = res['statins_numclms'].where(res['cond_cardiovascular_condition'].eq(True))
    res['antihypertensive_numclms_condnl'] = res['antihypertensive_numclms'].where(res['cond_hypertension'].eq(True))

    # Conditional MPM measures
    res['asthma_mpm_condnl'] = res['asthma_mpm'].where(res['cond_asthma'].eq(True))
    res['diabetes_mpm_condnl'] = res['diabetes_mpm'].where(res['cond_diabetes'].eq(True))
    res['statins_mpm_condnl'] = res['statins_mpm'].where(res['cond_cardiovascular_condition'].eq(True))
    res['antihypertensive_mpm_condnl'] = res['antihypertensive_mpm'].where(res['cond_hypertension'].eq(True))

    grpr = res['recip_age'].le(18).map({True: 'child', False: 'adult'})
    for label, gp in res.groupby(grpr):

        gp.to_csv(temp_p+f'analytic_table_{label}.csv', chunksize=10**6)

        ws.stata_do(do_file=code_p+'paper1_main_regs_with_prov.do',
                    params=[temp_p+f'analytic_table_{label}.csv',
                            output_p+f'paper1_main_results_with_prov_{label}.csv'],
                    log_path=log_p)

        # Delete temp file
        os.remove(temp_p+f'analytic_table_{label}.csv')


def paper1_regressions_yearly_with_attr_prov_CRG():
    """
    Run regressions for paper 1 using the yearly files
    """

    output_p, temp_p, log_p, code_p = gen_paths(FS().analysis_p
                                               +'paper1_regressions_yearly_with_attr_prov_CRG/')

    path_pre = FS().derived_p+f'create_analytic_tables/output/'
    d = {state: pd.read_pickle(path_pre+f'{state}_analytic_table_yearly_2016.pkl') 
         for state in ['LA', 'KS', 'TN']}
        
    out_cols = ['all_cost', 'medical_cost', 'pharm_cost', 'pc_any', 
                'pc_numdays', 'pharm_numclms', 'ednoadmit_numdays', 'inpat_numdays', 'spec_numdays',
                'antihypertensive_numclms', 'diabetes_numclms',
                'statins_numclms',
                'ed_avoid_numdays', 'lvc_uri', 'tests_imaging_numdays',
                'capch_all', 'capch_1to2', 'capch_2to6', 'capch_7to12', 'capch_12to19',
                'awc_all', 'awc_ch', 'awc_adolesc', 'ha1c', 'adv',
                'chl_ad', 'chl_ch', 'chl_all', 'ccs',
                'bcs', 'ssd', 'mpm_acearb', 'mpm_diur', 'amr', 'poud',
                'pqi_numdays', 'pqi_diab_numdays', 'pqi_chf_numdays', 'pqi_copd_numdays']
    cont_cols = (['recip_race_g', 'recip_age_g', 'recip_gender_g', 'recip_zip', 'elig_cat',
                  'state', 'plan_id_g', 'attr_prov']
                 + [x for x in d['LA'].columns if 'crg' in x])

    res = pd.concat([gp[out_cols+cont_cols] for _, gp in d.items()])
    # So plan control within state
    res['plan_id_g'] = res.groupby(['plan_id_g', 'state']).ngroup()

    # Only keep enrollees for which we report results
    res = res[res.recip_race_g.isin([0,1])]

    # Other outcomes of cost
    res['any_cost'] = res['all_cost'].gt(0).astype(int)
    res['all_cost_log'] = np.log1p(res['all_cost'])
    res['all_cost_wins40'] = res['all_cost'].clip(upper=40000)
    res['all_cost_wins125'] = res['all_cost'].clip(upper=125000)

    res.to_csv(temp_p+'analytic_table.csv', chunksize=10**6)

    ws.stata_do(do_file=code_p+'paper1_main_regs_with_prov_crg.do',
                params=[temp_p+'analytic_table.csv',
                        output_p+'paper1_main_results_with_prov_crg.csv'],
                log_path=log_p)

    # Delete temp file
    os.remove(temp_p+'analytic_table.csv')


def paper1_regressions_subset_with_attr_prov():
    """
    Run regressions for paper 1 using the yearly files for only the susbet that has an attributed
    provider 
    """

    output_p, temp_p, log_p, code_p = gen_paths(FS().analysis_p
                                               +'paper1_regressions_subset_with_attr_prov/')

    path_pre = FS().derived_p+f'create_analytic_tables/output/'
    d = {state: pd.read_pickle(path_pre+f'{state}_analytic_table_yearly_2016.pkl') 
         for state in ['LA', 'KS', 'TN']}
        
    out_cols = ['all_cost', 'medical_cost', 'pharm_cost', 'pc_any', 
                'pc_numdays',  'ednoadmit_numdays', 
                'ed_avoid_numdays']
    cont_cols = (['recip_race_g', 'recip_age_g', 'recip_gender_g', 'recip_zip', 'elig_cat',
                  'state', 'plan_id_g', 'attr_prov']
                 + [x for x in d['LA'].columns if 'hcc_' in x])

    res = pd.concat([gp[out_cols+cont_cols] for _, gp in d.items()])
    # So plan control within state
    res['plan_id_g'] = res.groupby(['plan_id_g', 'state']).ngroup()

    # Only keep enrollees for which we report results
    res = res[res.recip_race_g.isin([0,1])]

    # Remove people without attribution
    res = res[~res.attr_prov.isin(['missing_LA', 'missing_TN', 'missing_KS'])]

    res.to_csv(temp_p+'analytic_table.csv', chunksize=10**6)

    ws.stata_do(do_file=code_p+'paper1_main_regs_with_prov.do',
                params=[temp_p+'analytic_table.csv',
                        output_p+'paper1_main_results_with_prov.csv'],
                log_path=log_p)

    # Delete temp file
    os.remove(temp_p+'analytic_table.csv')


def paper1_regressions_yearly_by_group():
    """
    Run regressions for paper 1 using the yearly files
    """

    output_p, temp_p, log_p, code_p = gen_paths(FS().analysis_p
                                                +'paper1_regressions_yearly_by_group/')

    path_pre = FS().derived_p+f'create_analytic_tables/output/'
    d = {state: pd.read_pickle(path_pre+f'{state}_analytic_table_yearly_2016.pkl') 
         for state in ['LA', 'KS', 'TN']}
        
    out_cols = ['all_cost', 'pc_numdays', 'ed_avoid_numdays']
    cont_cols = (['recip_race_g', 'recip_age_g', 'recip_gender_g', 'recip_zip', 'elig_cat',
                  'state', 'plan_id_g', 'urban_category', 'seg_index', 'adi_natnl']
                 + [x for x in d['LA'].columns if 'hcc_' in x])

    res = pd.concat([gp[out_cols+cont_cols] for _, gp in d.items()])
    # So plan control within state
    res['plan_id_g'] = res.groupby(['plan_id_g', 'state']).ngroup()

    # Only keep enrollees for which we report results
    res = res[res.recip_race_g.isin([0,1])]

    # Map so Stata can do factors
    res['state_g'] = res['state'].map({'LA': 1, 'KS': 2, 'TN': 3})
    res['elig_cat_g'] = res['elig_cat'].map({'disability': 1, 'child': 2, 
                                             'adult': 3, 'other': 4})
    res['urban_category_g'] = res['urban_category'].map({'urban': 1, 'rural': 2})

    # Create HCC categories.
    grps = [('asthma', ['hcc_161.2','hcc_161.1', 'hcc_160']),
            ('dep', ['hcc_88', 'hcc_87.1', 'hcc_87.2', 'hcc_90', 'hcc_102', 'hcc_103']),
            ('diabetes', ['hcc_20', 'hcc_21']),
            ('drug_sud', ['hcc_81', 'hcc_82', 'hcc_83', 'hcc_84']),
            ('seizure', ['hcc_120']),
            ('pregnancy', ['hcc_203', 'hcc_204', 'hcc_205', 'hcc_207',
                            'hcc_208', 'hcc_209', 'hcc_210', 'hcc_211', 'hcc_212']),
            ('cvd', ['hcc_131', 'hcc_139', 'hcc_138',
                     'hcc_142', 'hcc_137', 'hcc_135', 'hcc_132',
                     'hcc_130', 'hcc_129', 'hcc_128', 'hcc_127',
                     'hcc_126', 'hcc_125'])
           ]
    for cond, cols in grps:
        res[f'cond_{cond}'] = res[cols].eq(1).any(1).astype(int)
    res['cond_any'] = res.filter(like='hcc_').eq(1).any(1).astype(int)
    res['cond_none'] = (~res.filter(like='hcc_').eq(1).any(1)).astype(int)

    # Number of conditions, 0,1,2,3,4,5+
    res['health_n'] = pd.cut(res.filter(like='hcc').sum(1), 
                             [0,1,2,3,4,5,np.inf], 
                             labels=False, include_lowest=True, right=False)

    res['seg_index_n'] = pd.qcut(res['seg_index'], 2, labels=False)
    res['adi_index_n'] = pd.qcut(res['adi_natnl'], 2, labels=False)

    res.to_csv(temp_p+'analytic_table.csv', chunksize=10**6)

    ws.stata_do(do_file=code_p+'paper1_main_regs_by_group.do',
                params=[temp_p+'analytic_table.csv',
                        output_p+'paper1_main_results_by_group.csv'],
                log_path=log_p)

    # Delete temp file
    os.remove(temp_p+'analytic_table.csv')


def paper1_regressions_yearly_by_group_relwhiteM():
    """
    Run regressions for paper 1 using the yearly files
    """

    output_p, temp_p, log_p, code_p = gen_paths(FS().analysis_p
                                                +'paper1_regressions_yearly_by_group_relwhiteM/')

    path_pre = FS().derived_p+f'create_analytic_tables/output/'
    d = {state: pd.read_pickle(path_pre+f'{state}_analytic_table_yearly_2016.pkl') 
         for state in ['LA', 'KS', 'TN']}
        
    out_cols = ['all_cost', 'pc_numdays', 'ed_avoid_numdays']
    cont_cols = (['recip_race_g', 'recip_age_g', 'recip_gender_g', 'recip_zip', 'elig_cat',
                  'state', 'plan_id_g', 'urban_category', 'seg_index']
                 + [x for x in d['LA'].columns if 'hcc_' in x])

    res = pd.concat([gp[out_cols+cont_cols] for _, gp in d.items()])
    # So plan control within state
    res['plan_id_g'] = res.groupby(['plan_id_g', 'state']).ngroup()

    # Only keep enrollees for which we report results
    res = res[res.recip_race_g.isin([0,1])]

    # Map so Stata can do factors
    res['state_g'] = res['state'].map({'LA': 1, 'KS': 2, 'TN': 3})
    res['elig_cat_g'] = res['elig_cat'].map({'disability': 1, 'child': 2, 
                                             'adult': 3, 'other': 4})
    res['urban_category_g'] = res['urban_category'].map({'urban': 1, 'rural': 2})

    # Create HCC categories.
    grps = [('asthma', ['hcc_161.2','hcc_161.1', 'hcc_160']),
            ('dep', ['hcc_88', 'hcc_87.1', 'hcc_87.2', 'hcc_90', 'hcc_102', 'hcc_103']),
            ('diabetes', ['hcc_20', 'hcc_21']),
            ('drug_sud', ['hcc_81', 'hcc_82', 'hcc_83', 'hcc_84']),
            ('seizure', ['hcc_120']),
            ('pregnancy', ['hcc_203', 'hcc_204', 'hcc_205', 'hcc_207',
                            'hcc_208', 'hcc_209', 'hcc_210', 'hcc_211', 'hcc_212']),
            ('cvd', ['hcc_131', 'hcc_139', 'hcc_138',
                     'hcc_142', 'hcc_137', 'hcc_135', 'hcc_132',
                     'hcc_130', 'hcc_129', 'hcc_128', 'hcc_127',
                     'hcc_126', 'hcc_125'])
           ]
    for cond, cols in grps:
        res[f'cond_{cond}'] = res[cols].eq(1).any(1).astype(int)
    res['cond_any'] = res.filter(like='hcc_').eq(1).any(1).astype(int)
    res['cond_none'] = (~res.filter(like='hcc_').eq(1).any(1)).astype(int)

    # Number of conditions, 0,1,2,3,4,5+
    res['health_n'] = pd.cut(res.filter(like='hcc').sum(1), 
                             [0,1,2,3,4,5,np.inf], 
                             labels=False, include_lowest=True, right=False)

    res['seg_index_n'] = pd.qcut(res['seg_index'], 5, labels=False)

    # 3-White Male, 1-White Female, 2-Black Male, 0-Black Female
    res['recip_race_gen_g'] = 3+(res.recip_race_g + 2*res.recip_gender_g)*-1

    res.to_csv(temp_p+'analytic_table.csv', chunksize=10**6)

    ws.stata_do(do_file=code_p+'paper1_main_regs_by_group_relwhiteM.do',
                params=[temp_p+'analytic_table.csv',
                        output_p+'paper1_main_results_by_group.csv'],
                log_path=log_p)

    # Delete temp file
    os.remove(temp_p+'analytic_table.csv')


def paper1_regressions_yearly_by_group_by_gender():
    """
    Run regressions for paper 1 using the yearly files
    """

    output_p, temp_p, log_p, code_p = gen_paths(FS().analysis_p
                                                +'paper1_regressions_yearly_by_group_by_gender/')

    path_pre = FS().derived_p+f'create_analytic_tables/output/'
    d = {state: pd.read_pickle(path_pre+f'{state}_analytic_table_yearly_2016.pkl') 
         for state in ['LA', 'KS', 'TN']}
        
    out_cols = ['all_cost', 'pc_numdays', 'ed_avoid_numdays']
    cont_cols = (['recip_race_g', 'recip_age_g', 'recip_gender_g', 'recip_zip', 'elig_cat',
                  'state', 'plan_id_g', 'urban_category', 'seg_index', 'adi_natnl']
                 + [x for x in d['LA'].columns if 'hcc_' in x])

    res = pd.concat([gp[out_cols+cont_cols] for _, gp in d.items()])
    # So plan control within state
    res['plan_id_g'] = res.groupby(['plan_id_g', 'state']).ngroup()

    # Only keep enrollees for which we report results
    res = res[res.recip_race_g.isin([0,1])]

    # Map so Stata can do factors
    res['state_g'] = res['state'].map({'LA': 1, 'KS': 2, 'TN': 3})
    res['elig_cat_g'] = res['elig_cat'].map({'disability': 1, 'child': 2, 
                                             'adult': 3, 'other': 4})
    res['urban_category_g'] = res['urban_category'].map({'urban': 1, 'rural': 2})

    # Create HCC categories.
    grps = [('asthma', ['hcc_161.2','hcc_161.1', 'hcc_160']),
            ('dep', ['hcc_88', 'hcc_87.1', 'hcc_87.2', 'hcc_90', 'hcc_102', 'hcc_103']),
            ('diabetes', ['hcc_20', 'hcc_21']),
            ('drug_sud', ['hcc_81', 'hcc_82', 'hcc_83', 'hcc_84']),
            ('seizure', ['hcc_120']),
            ('pregnancy', ['hcc_203', 'hcc_204', 'hcc_205', 'hcc_207',
                            'hcc_208', 'hcc_209', 'hcc_210', 'hcc_211', 'hcc_212']),
            ('cvd', ['hcc_131', 'hcc_139', 'hcc_138',
                     'hcc_142', 'hcc_137', 'hcc_135', 'hcc_132',
                     'hcc_130', 'hcc_129', 'hcc_128', 'hcc_127',
                     'hcc_126', 'hcc_125'])
           ]
    for cond, cols in grps:
        res[f'cond_{cond}'] = res[cols].eq(1).any(1).astype(int)
    res['cond_any'] = res.filter(like='hcc_').eq(1).any(1).astype(int)
    res['cond_none'] = (~res.filter(like='hcc_').eq(1).any(1)).astype(int)

    # Number of conditions, 0,1,2,3,4,5+
    res['health_n'] = pd.cut(res.filter(like='hcc').sum(1), 
                             [0,1,2,3,4,5,np.inf], 
                             labels=False, include_lowest=True, right=False)

    res['seg_index_n'] = pd.qcut(res['seg_index'], 2, labels=False)
    res['adi_index_n'] = pd.qcut(res['adi_natnl'], 2, labels=False)

    for gender, gp in res.groupby('recip_gender_g'):
        gp.to_csv(temp_p+'analytic_table.csv', chunksize=10**6)

        suffix = {0: '_male', 1: '_female'}.get(gender)
        ws.stata_do(do_file=code_p+'paper1_main_regs_by_group_by_gender.do',
                    params=[temp_p+'analytic_table.csv',
                            output_p+f'paper1_main_results_by_group{suffix}.csv'],
                    log_path=log_p)

        # Delete temp file
        os.remove(temp_p+'analytic_table.csv')


def paper1_recover_provider_intensity():
    """
    Run spending regression, but recover the provider effects
    """

    output_p, temp_p, log_p, code_p = gen_paths(FS().analysis_p
                                               +'paper1_recover_provider_intensity/')

    path_pre = FS().derived_p+f'create_analytic_tables/output/'
    d = {state: pd.read_pickle(path_pre+f'{state}_analytic_table_yearly_2016.pkl') 
         for state in ['LA', 'KS', 'TN']}

    out_cols = ['all_cost']
    cont_cols = (['recip_race_g', 'recip_age_g', 'recip_gender_g', 'recip_zip', 'elig_cat',
                  'state', 'plan_id_g', 'attr_prov', 'recip_id']
                 + [x for x in d['LA'].columns if 'hcc_' in x])

    res = pd.concat([gp[out_cols+cont_cols] for _, gp in d.items()])
    # So plan control within state
    res['plan_id_g'] = res.groupby(['plan_id_g', 'state']).ngroup()

    # Only keep enrollees for which we report results
    res = res[res.recip_race_g.isin([0,1])]

    res.to_csv(temp_p+'analytic_table.csv', chunksize=10**6)

    ws.stata_do(do_file=code_p+'paper1_recover_prov_intensity.do',
                params=[temp_p+'analytic_table.csv',
                        output_p+'paper1_prov_intensity_reg.csv',
                        output_p+'paper1_prov_intensity_table.dta'],
                log_path=log_p)

    # Delete temp file
    os.remove(temp_p+'analytic_table.csv')


def paper1_recover_provider_racial_composition():
    """
    Recover provider racial composition of panel, relative to leave-out. 
    """

    output_p, temp_p, log_p, code_p = gen_paths(FS().analysis_p
                                               +'paper1_recover_provider_racial_composition/')

    path_pre = FS().derived_p+f'create_analytic_tables/output/'
    d = {state: pd.read_pickle(path_pre+f'{state}_analytic_table_yearly_2016.pkl') 
         for state in ['LA', 'KS', 'TN']}

    cont_cols = (['recip_race_g', 'recip_age_g', 'recip_gender_g', 'recip_zip', 'elig_cat',
                  'state', 'plan_id_g', 'attr_prov', 'recip_id']
                 + [x for x in d['LA'].columns if 'hcc_' in x])

    res = pd.concat([gp[cont_cols] for _, gp in d.items()])
    # So plan control within state
    res['plan_id_g'] = res.groupby(['plan_id_g', 'state']).ngroup()

    # Only keep enrollees for which we report results
    res = res[res.recip_race_g.isin([0,1])]

    # Create the outcome 
    res['is_black'] = res.recip_race_g.eq(1).astype(int)

    res.to_csv(temp_p+'analytic_table.csv', chunksize=10**6)

    ws.stata_do(do_file=code_p+'paper1_recover_prov_composition.do',
                params=[temp_p+'analytic_table.csv',
                        output_p+'paper1_prov_composition_reg.csv',
                        output_p+'paper1_prov_composition_table.dta'],
                log_path=log_p)

    # Delete temp file
    os.remove(temp_p+'analytic_table.csv')


def paper1_table2():
    """
    Create Table of Main regression results
    """

    output_p = FS().analysis_p+'paper1_table2/output/'

    file_l = [('main', FS().analysis_p+'paper1_regressions_yearly/output/', 
              'paper1_main_results.csv'),
              ('lvc', FS().analysis_p+'paper1_lvc_regressions_yearly/output/',
               'paper1_lvc_results.csv'),
              ('preg', FS().analysis_p+'paper1_regressions_preg/output/',
               'paper1_main_results.csv'),
              ('crg', FS().analysis_p+'paper1_regressions_yearly_CRG/output/',
                'paper1_main_results_crg.csv'),
              ('crg_prov', FS().analysis_p+'paper1_regressions_yearly_with_attr_prov_CRG/output/',
                'paper1_main_results_with_prov_crg.csv')]

    l = []
    for suffix, data_p, f in file_l:
        df = pd.read_csv(data_p+f, sep='\t')

        df = df[df.variable.isin(['_cons', '1.recip_race_g'])]
        df['variable'] = df['variable'].map({'_cons': 'White', '1.recip_race_g': 'Black'})
        df['pval'] = pd.to_numeric(df.pval, errors='coerce')
        df['pval_fmt'] = df['pval'].apply(round_pval)


        df = df.pivot(index=['regression', 'outcome'], columns='variable', 
                      values=['coeff', 'se', 'lb', 'ub', 'pval', 'pval_fmt'])
        l.append(df)

    df = pd.concat(l)
    for reg, gp in df.groupby('regression'):
        gp.xs(reg, level=0).to_csv(output_p+f'{reg}_table.csv')


def paper1_table2_age_split():
    """
    Create Table of Main regression results
    Dan to do 
    """

    output_p = FS().analysis_p+'paper1_table2_age_split/output/'

    # Referene the 
    # Loop over adult and child
    for label in ['adult', 'child']:
        file_l = [('main', FS().analysis_p+'paper1_regressions_yearly_age_split/output/', 
                  f'paper1_main_results_{label}.csv')]

        l = []
        for suffix, data_p, f in file_l:
            df = pd.read_csv(data_p+f, sep='\t')

            df = df[df.variable.isin(['_cons', '1.recip_race_g'])]
            df['variable'] = df['variable'].map({'_cons': 'White', '1.recip_race_g': 'Black'})
            df['pval'] = pd.to_numeric(df.pval, errors='coerce')
            df['pval_fmt'] = df['pval'].apply(round_pval)


            df = df.pivot(index=['regression', 'outcome'], columns='variable', 
                          values=['coeff', 'se', 'lb', 'ub', 'pval', 'pval_fmt'])
            l.append(df)

        df = pd.concat(l)
        for reg, gp in df.groupby('regression'):
            gp.xs(reg, level=0).to_csv(output_p+f'{reg}_table_{label}.csv')


def paper1_table3():
    """
    Create Table of Main regression results
    """

    output_p = FS().analysis_p+'paper1_table3/output/'

    data_p = FS().analysis_p+'paper1_regressions_yearly_with_attr_prov/output/'
    df = pd.read_csv(data_p+'paper1_main_results_with_prov.csv', sep='\t')

    df = df[df.variable.isin(['_cons', '1.recip_race_g'])]
    df['variable'] = df['variable'].map({'_cons': 'White', '1.recip_race_g': 'Black'})
    df['pval'] = pd.to_numeric(df.pval, errors='coerce')
    df['pval_fmt'] = df['pval'].apply(round_pval)


    df = df.pivot(index=['regression', 'outcome'], columns='variable', 
                  values=['coeff', 'se', 'lb', 'ub', 'pval', 'pval_fmt'])
    for reg, gp in df.groupby('regression'):
        gp.xs(reg, level=0).to_csv(output_p+f'{reg}_table.csv')


def paper1_table3_age_split():
    """
    Create Table of Main regression results
    """

    output_p = FS().analysis_p+'paper1_table3_age_split/output/'
    data_p = FS().analysis_p+'paper1_regressions_yearly_with_attr_prov_age_split/output/'

    for label in ['adult', 'child']:
        df = pd.read_csv(data_p+f'paper1_main_results_with_prov_{label}.csv', sep='\t')

        df = df[df.variable.isin(['_cons', '1.recip_race_g'])]
        df['variable'] = df['variable'].map({'_cons': 'White', '1.recip_race_g': 'Black'})
        df['pval'] = pd.to_numeric(df.pval, errors='coerce')
        df['pval_fmt'] = df['pval'].apply(round_pval)


        df = df.pivot(index=['regression', 'outcome'], columns='variable', 
                      values=['coeff', 'se', 'lb', 'ub', 'pval', 'pval_fmt'])
        for reg, gp in df.groupby('regression'):
            gp.xs(reg, level=0).to_csv(output_p+f'{reg}_table_{label}.csv')


def paper1_adjusted_pvals():
    """
    Adjust p-values within domain for the Provider regresions
    """

    output_p = FS().analysis_p+'paper1_adjusted_pvals/output/'

    data_p = FS().analysis_p+'paper1_regressions_yearly_with_attr_prov/output/'
    df = pd.read_csv(data_p+'paper1_main_results_with_prov.csv', sep='\t')

    df = df[df.variable.isin(['_cons', '1.recip_race_g'])]
    df['variable'] = df['variable'].map({'_cons': 'White', '1.recip_race_g': 'Black'})
    df['pval'] = pd.to_numeric(df.pval, errors='coerce')
    df = df[df.variable.eq('Black')]


    dflvc = pd.read_csv(FS().analysis_p+'paper1_lvc_regressions_yearly/output/'+
                        'paper1_lvc_results.csv', sep='\t')
    dflvc = dflvc[dflvc.regression.eq('demo_health_prov')]
    dflvc = dflvc[dflvc.variable.isin(['_cons', '1.recip_race_g'])]
    dflvc['variable'] = dflvc['variable'].map({'_cons': 'White', '1.recip_race_g': 'Black'})
    dflvc['pval'] = pd.to_numeric(dflvc.pval, errors='coerce')
    dflvc = dflvc[dflvc.variable.eq('Black')]

    df = pd.concat([df, dflvc])
    df = df.set_index('outcome')

    domains = [('cost', ['all_cost', 'medical_cost', 'pharm_cost', 'any_cost']),
               ('util', ['pc_numdays', 'pc_any', 'spec_numdays', 'inpat_numdays',
                         'tests_imaging_numdays', 'ednoadmit_numdays']),
               ('sel_drugs', ['pharm_numclms', 'antihypertensive_numclms', 'statins_numclms',
                              'diabetes_numclms', 'asthma_numclms']),
               ('prim_prev', ['awc_all', 'bcs', 'ccs', 'chl_all']),
               ('acute', ['ha1c', 'ssd', 'poud', 'amr', 'ed_avoid_numdays']),
               ('lvc', ['lvc_abd_ct', 'lvc_brain_ct', 'lvc_thorax',
                        'lvc_hi_sync', 'lvc_hi_ha', 'lvc_im_bp', 'lvc_uri'])
              ]

    l = []
    for label, outcomes in domains:
        res = multitest.multipletests(df.loc[outcomes, 'pval'], method='fdr_bh')[1]
        l.append(pd.Series(res, index=outcomes))
    padj = pd.concat(l).to_frame('pval_adj')

    padj['pval_adj_fmt'] = padj['pval_adj'].apply(round_pval)
    padj.to_csv(output_p+'adjusted_pvals.csv')


def paper1_adjusted_pvals_age_split():
    """
    Adjust p-values within domain for the Provider regresions
    """

    output_p = FS().analysis_p+'paper1_adjusted_pvals_age_split/output/'
    data_p = FS().analysis_p+'paper1_regressions_yearly_with_attr_prov_age_split/output/'

    for age in ['adult', 'child']:
        df = pd.read_csv(data_p+f'paper1_main_results_with_prov_{age}.csv', sep='\t')

        df = df[df.variable.isin(['_cons', '1.recip_race_g'])]
        df['variable'] = df['variable'].map({'_cons': 'White', '1.recip_race_g': 'Black'})
        df['pval'] = pd.to_numeric(df.pval, errors='coerce')
        df = df[df.variable.eq('Black')]

        df = df.set_index('outcome')

        domains = [('cost', ['all_cost', 'medical_cost', 'pharm_cost', 'any_cost']),
                   ('util', ['pc_numdays', 'pc_any', 'spec_numdays', 'inpat_numdays',
                             'tests_imaging_numdays', 'ednoadmit_numdays']),
                   ('sel_drugs', ['pharm_numclms', 'antihypertensive_numclms_condnl', 'statins_numclms_condnl',
                                  'diabetes_numclms_condnl', 'asthma_numclms_condnl']),
                   ('prim_prev', ['awc_all', 'bcs', 'ccs', 'chl_all']),
                   ('acute', ['ha1c', 'ssd', 'poud', 'amr', 'ed_avoid_numdays']),
                   ('test', ['amr', 'ed_avoid_numdays', 'awc_all', 'chl_all'])
                  ]

        l = []
        for label, outcomes in domains:
            outcomes = df.index.intersection(outcomes)
            if not outcomes.empty:
                res = multitest.multipletests(df.loc[outcomes, 'pval'], method='fdr_bh')[1]
                l.append(pd.Series(res, index=outcomes))
        padj = pd.concat(l).to_frame('pval_adj')

        padj['pval_adj_fmt'] = padj['pval_adj'].apply(round_pval)
        padj.to_csv(output_p+f'adjusted_pvals_{age}.csv')


def paper1_table_appendix_only_attributable():
    output_p = FS().analysis_p+'paper1_table_appendix_only_attributable/output/'

    data_p = FS().analysis_p+'paper1_regressions_subset_with_attr_prov/output/'
    df = pd.read_csv(data_p+'paper1_main_results_with_prov.csv', sep='\t')

    df = df[df.variable.isin(['_cons', '1.recip_race_g'])]
    df['variable'] = df['variable'].map({'_cons': 'White', '1.recip_race_g': 'Black'})
    df['pval'] = pd.to_numeric(df.pval, errors='coerce')
    df['pval_fmt'] = df['pval'].apply(round_pval)


    df = df.pivot(index=['regression', 'outcome'], columns='variable', 
                  values=['coeff', 'se', 'lb', 'ub', 'pval', 'pval_fmt'])
    for reg, gp in df.groupby('regression'):
        gp.xs(reg, level=0).to_csv(output_p+f'{reg}_table.csv')


def paper1_table_appendix_by_atc():
    output_p = FS().analysis_p+'paper1_table_appendix_by_atc/output/'

    data_l = [FS().analysis_p+'paper1_regressions_atc/output/',
              FS().analysis_p+'paper1_regressions_atc_with_attr_prov/output/']
    for data_p in data_l:
        df = pd.read_csv(data_p+'paper1_atc_results.csv', sep='\t')

        df = df[df.variable.isin(['_cons', '1.recip_race_g'])]
        df['variable'] = df['variable'].map({'_cons': 'White', '1.recip_race_g': 'Black'})
        df['pval'] = pd.to_numeric(df.pval, errors='coerce')
        df['pval_fmt'] = df['pval'].apply(round_pval)


        df = df.pivot(index=['regression', 'outcome'], columns='variable', 
                      values=['coeff', 'se', 'lb', 'ub', 'pval', 'pval_fmt'])
        for reg, gp in df.groupby('regression'):
            gp.xs(reg, level=0).to_csv(output_p+f'{reg}_table.csv')


def paper1_figure1():
    """
    Create Paper 1 Figure 1 of spending effects across states
    """

    output_p = FS().analysis_p+'paper1_figure1/output/'

    path_pre = FS().derived_p+f'create_analytic_tables/output/'
    d = {state: pd.read_pickle(path_pre+f'{state}_analytic_table_yearly_2016.pkl') 
         for state in ['LA', 'KS', 'TN']}
    df = pd.concat(d)

    df = df[df.recip_race_g.isin([0, 1])]
    df['log_cost'] = np.log1p(df['all_cost'])

    # Residualize Log Spending
    absorb = ['recip_age_g', 'recip_gender_g', 'elig_cat', 'state', 'recip_zip']
    absorb = df[absorb].astype('category')
    s = df['log_cost']

    # Residualize log-spending
    mod = AbsorbingLS(s, pd.Series(1, index=s.index), absorb=absorb)
    res = mod.fit()

    s2 = s - res.fitted_values['fitted_values'] + s.mean()

    df['resid'] = s2

    # Get the densities for the residualized and normal log spending
    # for all non-zero obs.
    l = [(0, 'log_cost'), (1, 'log_cost'), (0, 'resid'), (1, 'resid')]
    d = {}
    for race, col in l:
        sp = df.loc[df.all_cost.ne(0) & df.recip_race_g.eq(race), col]
        axis = sp.plot.kde()
        d[(race, col, 'x')] = axis.get_children()[0]._x
        d[(race, col, 'y')] = axis.get_children()[0]._y
        d[(race, col, 'scale')] = len(sp)
        d[(race, col, 'data')] = sp
        plt.clf()

    cd = {0: '#BA4B2F', 1: '#6699CC'}
    fig, ax = plt.subplots(figsize=(10, 5), ncols=2,
                           gridspec_kw = {'width_ratios': [0.4, 1]})
    ax = ax.flatten()
    plt.subplots_adjust(wspace=0.2)

    for axis in ax:
        axis.tick_params(which='major', axis='both', labelsize=12)

    s0 = df['all_cost'].eq(0).groupby(df['recip_race_g']).mean().mul(100)
    s0.index = s0.index.map({0: 'White', 1: 'Black'})
    s0.plot(kind='bar', ec='k', ax=ax[0], rot=0, color=cd.values())
    ax[0].set_xlabel(None)
    ax[0].set_title('% of Population with\n$0 Annual Spending', loc='left')

    bdiff = s0["Black"] - s0["White"]
    ax[0].text(-0.4, 14.7, (r'$\beta$ = ' + f'{bdiff:,.2f}'), fontsize=13)


    for race, col in [(0, 'log_cost'), (1, 'log_cost')]:
        ax[1].plot(d[(race, col, 'x')], d[(race, col, 'y')], color=cd[race])
        ax[1].axvline(d[(race, col, 'data')].mean(), color=cd[race], linestyle='--')

    bdiff = d[(1, "log_cost", "data")].mean() - d[(0, "log_cost", "data")].mean()
    ax[1].text(2, 0.29, (r'$\beta$ = ' + f'{bdiff:,.2f}'), fontsize=13)

    ax[1].set_xlim(1.5, 13)
    ax[1].set_title('Distribution of Log Spending for Positive Spending', fontsize=12, loc='left')
    ax[1].set_ylabel('Density', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_p+'spend_dist.png', dpi=300)
    plt.show()


def paper1_figure2():
    output_p = FS().analysis_p+'paper1_figure2/output/'

    loop_l = [(FS().analysis_p+'paper1_regressions_yearly_by_group/output/',
               'paper1_main_results_by_group.csv', 'all'),
              (FS().analysis_p+'paper1_regressions_yearly_by_group_by_gender/output/',
               'paper1_main_results_by_group_male.csv', 'male'),
              (FS().analysis_p+'paper1_regressions_yearly_by_group_by_gender/output/',
               'paper1_main_results_by_group_female.csv', 'female')
             ]

    for data_p, data_f, suffix in loop_l:
        df = pd.read_csv(data_p+data_f, sep='\t')
        df['entity'] = df['variable'].str.extract(r'.*\#(\d+).*')
        df = df[df.entity.notnull()]

        df.loc[df.outcome.isin(['pc_numdays', 'ed_avoid_numdays']), ['coeff', 'se']]*=100

        # Fix Conditions so all on one plot
        s = df[df.regression.str.startswith('cond')]
        df = df[~df.regression.str.startswith('cond')]

        s = s[s.variable.str.contains('#1')]
        s['entity'] = s.groupby('regression', sort=False).ngroup()+1
        # Becasue Men are missing pregnancy need to adjust counts. 
        if suffix == 'male':
            s.loc[s.entity.ge(8), 'entity'] =  s.loc[s.entity.ge(8), 'entity'] + 1
        s['regression'] = 'cond'

        df = pd.concat([df, s])
        df = df[~df.regression.isin(['health_n', 'plan'])]
        df['entity'] = pd.to_numeric(df['entity'])

        d_map = {'urban': {1: 'Urban', 2: 'Rural'},
               'elig': {1: 'Disability', 2: 'Child',  
                        3: 'Adult'},
               'segidx': {0: 'Below Median\nSegregation', 1: 'Above Median\nSegregation'},
               'adiidx': {0: 'Below Median\nArea Deprivation', 1: 'Above Median\nArea Deprivation'},
               'state': {1: 'State 1', 2: 'State 2', 3: 'State 3'},
               'cond': {1: 'No Conditions', 2: 'Any Condition', 3: 'Asthma', 4: 'Depression',
                        5: 'Diabetes', 6: 'Drug/Substance Use', 7: 'Seizures',
                        8: 'Pregnancy', 9: 'Cardiovascular\nConditions'}}


        l = ['State 1', 'State 2', 'State 3', 'Urban', 'Rural',
           'Below Median\nSegregation', 'Above Median\nSegregation',
           'Below Median\nArea Deprivation', 'Above Median\nArea Deprivation',
           '',
           'Disability', 'Child', 'Adult',
           '',
           'No Conditions', 'Any Condition', 'Asthma', 'Depression',
           'Diabetes', 'Drug/Substance Use', 'Seizures', 'Pregnancy', 
           'Cardiovascular\nConditions']
        l = l[::-1]

        dfl = pd.concat([(pd.DataFrame.from_dict(v, orient='index', columns=['label'])
                          .rename_axis(index='entity')
                          .reset_index()
                          .assign(regression=k)) for k,v in d_map.items()])

        df = df.merge(dfl, how='left')

        fig, ax = plt.subplots(figsize=(10, 10), ncols=3)
        ax = ax.flatten()
        plt.subplots_adjust(wspace=0.04)

        if suffix == 'all':
            ilims = {0: (-1300, 1300), 1: (-260, 260), 2: (-18, 18)}
        else:
            ilims = {0: (-1900, 1900), 1: (-260, 260), 2: (-18, 18)}
        itits = {0: 'Health Care Spending', 1: 'PC Utilization', 2: 'Avoidable ED'}
        xlabel = {0: '$ annually', 1: 'No, per 100 enrollees', 2: 'No, per 100 enrollees'}

        for i, outcome in enumerate(['all_cost', 'pc_numdays', 'ed_avoid_numdays']):
          gp = df[df.outcome.eq(outcome)]
          gp = gp.set_index('label').reindex(l)

          ax[i].errorbar(x=gp['coeff'], 
                         y=np.arange(len(gp)), 
                         xerr=1.96*gp['se'], 
                         lw=0, ms=5, marker='o', elinewidth=2, zorder=2)
          ax[i].axvline(0,0,1, color='#676767', linestyle='--', lw=2, zorder=-1)

          for myl in range(len(l)):
              if l[myl]:
                  ax[i].axhline(myl, 0, 1, color='#ABABAB', linestyle='--', lw=0.8, zorder=0)  

          ax[i].set_ylim(-0.5, len(l)+0.5)    
          ax[i].set_xlim(ilims.get(i))
          ax[i].set_title(itits.get(i), fontsize=12, loc='left')
          ax[i].set_xlabel(xlabel.get(i), fontsize=12)
          if i != 0:
              ax[i].set_yticks([])
              ax[i].set_yticklabels([])

        ax[0].set_yticks([x for x in range(len(l)) if l[x]])
        ax[0].set_yticklabels([x for x in l if x], style='italic')

        labs = fig.text(s='Geography', x=0.052, y=0.927, fontdict={'fontsize': 14})
        labs = fig.text(s='Eligibility', x=0.068, y=0.546, fontdict={'fontsize': 14}) 
        labs = fig.text(s='Health Status', x=0.028, y=0.395, fontdict={'fontsize': 14}) 

        y = 0.914
        fontsize=10
        labs = fig.text(s='Non-Hispanic\nBlack enrollees\nspend less', x=0.224, y=y, fontdict={'fontsize': fontsize},
                        ha='center')
        labs = fig.text(s='Non-Hispanic\nBlack enrollees\nspend more', x=0.366, y=y, fontdict={'fontsize': fontsize},
                        ha='center')

        labs = fig.text(s='Non-Hispanic\nBlack enrollees\nuse less', x=0.505, y=y, fontdict={'fontsize': fontsize},
                        ha='center')
        labs = fig.text(s='Non-Hispanic\nBlack enrollees\nuse more', x=0.65, y=y, fontdict={'fontsize': fontsize},
                        ha='center')

        labs = fig.text(s='Non-Hispanic\nBlack enrollees\nuse less', x=0.782, y=y, fontdict={'fontsize': fontsize},
                        ha='center')
        labs = fig.text(s='Non-Hispanic\nBlack enrollees\nuse more', x=0.93, y=y, fontdict={'fontsize': fontsize},
                        ha='center')


        plt.tight_layout()
        plt.savefig(output_p+f'effects_by_group_{suffix}.png', dpi=300)
        plt.savefig(output_p+f'effects_by_group_{suffix}.eps', dpi=300)
        plt.savefig(output_p+f'effects_by_group_{suffix}.jpg', dpi=300)
        plt.show()


def paper1_figure_spend_dist():
    output_p = FS().analysis_p+'paper1_figure_spend_dist/output/'

    path_pre = FS().derived_p+f'create_analytic_tables/output/'
    d = {state: pd.read_pickle(path_pre+f'{state}_analytic_table_yearly_2016.pkl') 
         for state in ['LA', 'KS', 'TN']}
    df = pd.concat(d)

    df = df[df.recip_race_g.isin([0, 1])]

    s = df['all_cost']
    figsize=(7,6)
    fontsize=11
    title_fontsize=12
    xlim=10000
    xbin=200
    hspace=0.65
    wspace=0.1
    grid=True
    color='#7C5295'

    distbins = np.arange(0, xlim+xbin, xbin)

    fig, ax = plt.subplots(figsize=figsize, nrows=2, ncols=2)
    ax = ax.flatten()
    plt.subplots_adjust(hspace=hspace, wspace=wspace)

    for i in range(len(ax)):
        ax[i].tick_params(which='major', axis='both', labelsize=fontsize)

    # Spending distribution     
    s.plot(kind='hist', ax=ax[0], ec='k', bins=distbins,
           weights=np.ones_like(s)*100./len(s),
           zorder=4, color=color)
    ax[0].set_xlim(0, xlim)
    ax[0].set_xlabel('Spending ($)', fontsize=fontsize)
    ax[0].set_ylabel('Percent', fontsize=fontsize)
    ax[0].set_title('Panel A. All spending', fontsize=title_fontsize)
    ax[0].locator_params(axis='x', nbins=3) 

    # Percent 0 vs non-zero Spending.
    s2 = s.gt(0).value_counts(normalize=True)*100
    s2.index = s2.index.map({True: 'Positive\nspending', 
                             False: 'Zero\nspending'})
    s2.plot(kind='bar', ax=ax[1], ec='k', zorder=4, color=color)
    ax[1].set_title('Panel B. Fraction with\npositive spending', fontsize=title_fontsize)
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=0)
    ax[1].set_ylabel('Percent', fontsize=fontsize)

    # Box Plot of Spending
    s.rename('All Spending').plot(kind='box', ax=ax[2], patch_artist=True,
                        color=dict(boxes=color, whiskers=color, medians='black', caps=color))
    ax[2].set_yscale('log')
    ax[2].set_ylim(0.01, 10**7)
    ax[2].set_ylabel('Spending ($)', fontsize=fontsize)
    ax[2].set_title('Panel C. Box and Whisker Plot\nof All spending', fontsize=title_fontsize)
        
    # # Spending dist | spending > 0 
    # s3 = s[s.gt(0)]
    # s3.plot(kind='hist', ax=ax[2], ec='k', bins=distbins,
    #         weights=np.ones_like(s3)*100./len(s3),
    #         zorder=4, color=color)
    # ax[2].set_xlim(0, xlim)
    # ax[2].set_xlabel('Spending ($)', fontsize=fontsize)
    # ax[2].set_ylabel('Percent', fontsize=fontsize)
    # ax[2].set_title('Panel C. All spending\nif spending > 0', fontsize=title_fontsize)
    # ax[2].locator_params(axis='x', nbins=3) 

    # Dist of log Spending + Normal dist fit. 
    slog = np.log(s[s.gt(0)])
    diff = 0.4 # Size of the bins for log plot
    x = np.arange(-10, 30, diff)
    slog.plot(kind='hist', ax=ax[3], ec='k', bins=x,
              weights=np.ones_like(slog)*100./len(slog),
              zorder=4, color=color)
    ax[3].set_xlabel('Log(spending)', fontsize=fontsize)
    ax[3].set_title('Panel D. Log(spending)\nif spending > 0', fontsize=title_fontsize)
    ax[3].set_ylabel('Percent', fontsize=fontsize)
    ax[3].set_xlim(-1, 14)

    fit = norm.fit(slog)
    xmin, xmax = ax[3].get_xlim()
    p = norm.pdf(x, fit[0], fit[1])
    # Scale fit to be percent based.
    ax[3].plot(x, p*100*diff, 'k', linewidth=2, zorder=5)

    if grid:
        for i in range(len(ax)):
            ax[i].grid(axis='y', alpha=0.4, zorder=2)
    plt.tight_layout()
    plt.savefig(output_p+'spending_Distribution.png', dpi=300)
    plt.show()


def paper1_app_race_map_table():
    """
    Where to get calculations for race in states for appendix. 
    """

    # LA
    df = pd.read_pickle(FS().la_pipe_raw_path+'person_details.pkl.gz')
    s = df.drop_duplicates('NEW_ELIG_ID')['ELS_Race'].value_counts(normalize=True)
    s.mul(100).round(1).sort_index()

    # TN
    df = pd.read_pickle(FS().tn_pipe_raw_path+'eligibility.pkl.gz')
    s = df.drop_duplicates('sak_recip').dsc_race.value_counts(normalize=True)

    # KS
    df = pd.read_pickle(FS().ks_pipe_raw_path+'enrollment_2016.pkl.gz')
    s = df.MEMBER_RACE.value_counts(normalize=True).mul(100).round(1)


def paper1_app_state_compare_table():
    """
    Table comparing our states to the rest of the states. 
    """
    output_p, temp_p, log_p, code_p = gen_paths(FS().analysis_p
                                                +'paper1_app_state_compare_table/') 
    data_p = FS().analysis_p+'paper1_app_state_compare_table/data/'

    def wght_mean(gp, valcol, wcol):
        """ Calculate the weighted mean in a groupby. Deals with dropping NaN """
        gp = gp.dropna(subset=[valcol, wcol])
        return np.average(gp[valcol], weights=gp[wcol])


    ahrf_p = '//storage.yale.edu/home/YSPH_HPM-CC0940-MEDSPH2/Private Data Pipeline/Resource_Data/AHRF/Gold/'
    census_p = '//storage.yale.edu/home/YSPH_HPM-CC0940-MEDSPH2/Private Data Pipeline/Resource_Data/Census/Gold/'
    file = 'AHRF.pkl'

    df = pd.read_pickle(ahrf_p+file)

    # Keep only 50 states 
    df = df[~df['State Name'].isin(['Guam', 'Puerto Rico', 
                                    'US Virgin Islands', 'Dist. of Columbia'])]

    # Create an indicator for whether or not in our sample
    df['in_sample'] = df['State Name'].isin(['Kansas', 'Tennessee', 'Louisiana']).astype(int)

    # Use only < 65 population based on 2017 census estimates for weights. AHRF doesn't have.
    dfc = pd.read_pickle(census_p+'census_2010_population_by_demographics.pkl')

    dfc['nonwhite_0to64'] = dfc['total_0to64'] - (dfc['WA_MALE_0to64'] + dfc['WA_FEMALE_0to64'])
    dfc['non-white'] = dfc['nonwhite_0to64']/dfc['total_0to64']*100
    dfc['non_hisp_white'] = (dfc['NHWA_MALE_0to64'] + dfc['NHWA_FEMALE_0to64'])/dfc['total_0to64']*100
    dfc['non_hisp_black'] = (dfc['NHBA_MALE_0to64'] + dfc['NHBA_FEMALE_0to64'])/dfc['total_0to64']*100
    dfc['hispanic'] = (dfc['H_MALE_0to64'] + dfc['H_FEMALE_0to64'])/dfc['total_0to64']*100
    dfc = dfc.rename(columns={'total_0to64': 'pop_est_2017'})           

    mcols = ['fips', 'nonwhite_0to64', 'non-white', 'non_hisp_white', 'non_hisp_black',
             'hispanic', 'pop_est_2017']
    df = df.merge(dfc[mcols], how='left', on='fips')

    df['non-white'] = df.fips.map(dfc.set_index('fips')['non-white'])

    # Quick sanity check:
    display(df.groupby('in_sample')['State Name'].nunique())


    # Create Population total and weight within state. 
    df['state_poptot'] = df.groupby('State Name')['pop_est_2017'].transform('sum')
    df['weight'] = df['pop_est_2017']/df['state_poptot']

    df['metro'] = df['CBSA Indicator Code 0 = Not, 1 = Metro, 2 = Micro 2017'].map(
                        {'0': 'non-Metro', '1': 'Metro', '2': 'Micro'})

    df['mddo_2016_pct'] = (df[['Tot Active M.D.s Non-Fed and Fed 2016',
                               'Tot Active D.O.s Non-Fed and Fed 2016']]
                               .apply(pd.to_numeric, errors='coerce').sum(1)
                               .div(df['pop_est_2017']).mul(1000))
    df['hosp_pct'] = (pd.to_numeric(df['Total Number Hospitals 2016'], errors='coerce')
                        .div(df['pop_est_2017']).mul(100_000))

    df = df.rename(columns={'% Persons 25+ w/<HS Diploma 2012-16': 'per_25_lthsdip',
                            '% Persons 25+ w/4+ Yrs College 2012-16': 'per_25_4morecol'})


    # Make the State Level Averages
    l = [] # Will hold all of the Frames to Concat

    x = (df.groupby(['State Name', 'in_sample', 'metro'])
                .pop_est_2017.sum())
    x = (x/x.groupby(level=0).sum()).multiply(100).unstack(-1)

    l.append(x)
    l.append(df.groupby(['State Name', 'in_sample']).fips.nunique().to_frame('Nfips'))

    avg_l = [('Percent Persons in Poverty 2016', '% Poverty'),
             ('mddo_2016_pct', 'mddo_2016_pct'),
             ('per_25_4morecol', 'per_25_4morecol'),
             ('hosp_pct', 'hosp_pct'),
            ] 
    for valcol, rename in avg_l:
        df[valcol] = pd.to_numeric(df[valcol], errors='coerce')
        print(valcol, df[valcol].isnull().mean()*100)
        l.append(df.groupby(['State Name', 'in_sample'])
                      .apply(wght_mean, valcol=valcol, wcol='weight')
                      .to_frame(rename))


    for col in ['non_hisp_white', 'non_hisp_black', 'hispanic']:
        l.append(df.groupby(['State Name', 'in_sample'])
                      .apply(wght_mean, valcol=col, wcol='weight')
                      .to_frame(col))


    dfs = pd.concat(l, axis=1)

    display(dfs[dfs.index.get_level_values(1) == 1].T.round(2))
    display(dfs.fillna(0).groupby(level=1).mean().T.round(2))

    res = pd.concat([dfs[dfs.index.get_level_values(1) == 1].T,
                     dfs.fillna(0).groupby(level=1).mean().T], axis=1)
    res.to_csv(output_p+'averages.csv', sep=',')


    df_fs = pd.concat([df[['fips', 'State Name', 'in_sample', 
                           'weight', 'state_poptot', 'pop_est_2017',
                           'non_hisp_black', 'non_hisp_white', 'hispanic']
                           +[x[0] for x in avg_l]],
                       pd.get_dummies(df['metro']).multiply(100)], axis=1)

    df_fs.to_csv(temp_p+'state_char.csv', index=False)

    ws.stata_do(do_file=code_p+'state_regressions.do',
                params=[temp_p+'state_char.csv', 
                        temp_p+f'state_char_results.txt'],
                log_path=log_p)

    t = pd.read_csv(temp_p+f'state_char_results.txt', sep='\t')
    t.to_csv(output_p+'reg_results.csv', index=False)


    # Separate state level averages for coverage
    s = pd.read_csv(data_p+'kff_raw_data_coverage.csv').drop(columns=['Footnotes', 'Total'])
    s1 = pd.read_csv(data_p+'kff_raw_data_mco.csv').drop(columns=['Footnotes'])

    s = pd.concat([s.set_index('Location'), s1.set_index('Location')], axis=1)
    s['in_sample'] = s.index.isin(['Kansas', 'Tennessee', 'Louisiana']).astype(int)
    s = s.drop(columns=['Comprehensive Risk-Based Managed Care Enrollees'])
    s = s.apply(pd.to_numeric, errors='coerce')

    res1 = pd.concat([s[s.in_sample.eq(1)].T.mul(100).drop('in_sample'), 
                       s.groupby('in_sample').mean().mul(100).T], axis=1)

    res1.to_csv(output_p+'coverage_averages.csv')

    s.columns = s.columns.str.replace('\s', '', regex=True).str.replace('-', '').str.lower()
    s.to_csv(temp_p+'state_coverage.csv', index=False)

    ws.stata_do(do_file=code_p+'state_cov_regressions.do',
                params=[temp_p+'state_coverage.csv', 
                        temp_p+f'state_cov_results.txt'],
                log_path=log_p)

    t = pd.read_csv(temp_p+f'state_cov_results.txt', sep='\t')
    cols = ['Coeff', '95per_lo', '95per_up']
    t[cols] = t[cols]*100
    t.to_csv(output_p+'cov_reg_results.csv', index=False)

    # Adjusted P-values
    df1 = pd.concat([pd.read_csv(output_p+'reg_results.csv'),
                     pd.read_csv(output_p+'cov_reg_results.csv')])
    df1['p_adj'] = multitest.multipletests(df1['P(t)'], method='fdr_bh')[1]
    df1['p_rnd'] = df1['P(t)'].apply(round_pval)
    df1['p__adj_rnd'] = df1['p_adj'].apply(round_pval)
    df1.to_csv(output_p+'reg_results_all.csv', index=False)


####################################################################################################
################################## PAPER 1 Possible Expansion ######################################
####################################################################################################
def create_analytic_tables_truncated():
    """
    Shoter version of tabels with select outcomes 2010-2018 where we have data to assess whether
    or not we can expand the study
    """

    output_p = FS().derived_p+'create_analytic_tables_truncated/output/'

    l = [('KS', FS().ks_pipe_analytic_path, FS().ks_pipe_gold_path, range(2013, 2019)),
         ('TN', FS().tn_pipe_analytic_path, FS().tn_pipe_gold_path, range(2010, 2018)), 
         ('LA', FS().la_pipe_analytic_path, FS().la_pipe_gold_path, range(2010, 2017))
        ]

    for state, path, gpath, years in l:
        for year in years:
            df = pd.read_pickle(path+f'analytic_table_{year}.pkl')
            df['state'] = state

            # FOR TN Remove Tenncare and FFS
            if state == 'TN':
                df = df[~(df.plan_id.isnull() | df.plan_id.isin(['62042793911', '62042793904']))]

            df = df.rename(columns={'all_pharm_cost': 'pharm_cost',
                                    'all_pharm_numclms': 'pharm_numclms',
                                    'all_nonph_cost': 'medical_cost',
                                    'all_nonph_numclms': 'medical_numclms'})

            print(f'Starting: {state}: {df.recip_id.nunique():,}')
            # Remove anyone > 65
            df = df[df.recip_age.between(0, 65)]
            print(f'After removing 65: {state}: {df.recip_id.nunique():,}')

            # Remove enrollees who have ZIP codes outside of the state
            # All Fips within a state
            fips = dt.county_name_to_fips(pd.DataFrame({'county': ['statewide'], 'state': [state]}), 
                                               state_col='state', cname_col='county',
                                               fips_col='fips', statewide=True)
            # All Zips within that state
            fips = dt.county_to_zip(fips, fips_col='fips', zip_col='zip')
            df = df[df.recip_zip.isin(fips.zip)]
            print(f'After removing Bad ZIPs: {state}: {df.recip_id.nunique():,}')

            if state == 'TN':
                dfpl = pd.read_pickle(FS().tn_pipe_gold_path+'plan.pkl')
                dfpl['plan_name'] = dfpl.plan_name.str.split(' - ').str[0]

                df['plan_id'] = df['plan_id'].map(dfpl.set_index('plan_id')['plan_name'])
            elif state == 'KS':
                df['plan_id'] = df['plan_id'].str.rstrip('AB')
            else:
                # Create a FFS category
                df['plan_id'] = df['plan_id'].fillna('FFS')

            df['plan_id_g'] = df.groupby('plan_id').ngroup()

            # Keep only those continuously enrolled for the entire year
            if state == 'TN' and year == 2017:
                df = df[df.month.between(1, 6)]
                df = df[df.groupby('recip_id')['recip_id'].transform('size').eq(6)]
            else:
                df = df[df.groupby('recip_id')['recip_id'].transform('size').eq(12)]

            print(f'After requiring continuous enrollment: {state}: {df.recip_id.nunique():,}')

            
            # Race, Age and Gender controls
            df['recip_race_g'] = map_race(df, state)
            df['recip_age_g'] = (df['recip_age']//5).clip(upper=12).mul(5)
            df['recip_gender_g'] = df.recip_gender.map({'M': 0, 'F': 1})    

            print(f'Limit to Black/White: {state}: {df[df.recip_race_g.isin([0,1])].recip_id.nunique():,}')

            # Create Total Outcome
            df['all_cost'] = df['medical_cost'] + df['pharm_cost']
            
            # Bring over PC
            dflt = pd.read_pickle(FS().derived_p+f'create_primary_care/output/{state}_pc.pkl')
            adnl_cols = dflt.columns.difference(['recip_id', 'year', 'month']).tolist()
            df = df.merge(dflt, on=['recip_id', 'year', 'month'], how='left')
            df[adnl_cols] = df[adnl_cols].fillna(0)
            df['pc_any'] = df['pc_numdays'].clip(upper=1)
            
            
            # Bring over Avoidable ED Outcome
            dfed = pd.read_pickle(path+'ED_claims_avoidable.pkl')
            dfed['year'] = dfed.claim_date.dt.year
            dfed['month'] = dfed.claim_date.dt.month
            dfed = dfed[dfed.year.eq(year)]
            dfed = (dfed.groupby(['year', 'month', 'recip_id'])
                        .agg(ed_avoid_numdays=('claim_date', 'nunique')))
            df = df.merge(dfed, on=['recip_id', 'year', 'month'], how='left')
            df['ed_avoid_numdays'] = df['ed_avoid_numdays'].fillna(0, downcast='infer')
            
            # Bring over HHS HCCs
            dfhhs = pd.read_pickle(FS().derived_p+f'create_hhs_hccs/output/{state}_{year}.pkl')
            df = df.merge(dfhhs, on='recip_id', how='left')
            hcc_cols = [x for x in df.columns if x.startswith('hcc_')]
            df[hcc_cols] = df[hcc_cols].fillna(0)

            # Bring over eligibility category
            df = map_eligibility(df, state)

            ####
            # Yearly Table
            #####
            sumcols = [x for x in df.columns if '_cost' in x or '_numclms' in x or '_numdays' in x
                                             or '_supplydays' in x]
            firstcols = ['recip_gender', 'recip_race', 'recip_race_g', 'recip_age',
                         'recip_gender_g', 'state']
            mode_cols = ['recip_age_g', 'plan_id', 'plan_id_g', 'recip_zip', 'recip_county_fips', 
                         'elig_cat']
            maxcols = ([x for x in df.columns if x.startswith('hcc_')] 
                        + ['pc_any'])

            agg_ds = {x: 'sum' for x in sumcols}
            agg_df = {x: 'first' for x in firstcols}
            agg_dm = {x: 'max' for x in maxcols}
            agg_d = {**agg_ds, **agg_df, **agg_dm}

            dfy = df.groupby('recip_id').agg(agg_d).reset_index()
            dfy['year'] = year
            for col in mode_cols:
                s = dt.fast_mode(df, key_cols=['recip_id'], value_col=col)
                dfy = dfy.merge(s, on='recip_id', how='left')        
            
            # Because 2017 TN is only 6 months:
            if year == 2017 and state == 'TN':
                dfy[sumcols] = dfy[sumcols]*2
            
            
            # Bring over attributed provider
            dfprov = pd.read_pickle(FS().derived_p+f'attribute_providers/output/{state}_provs_attr.pkl')
            dfprov = dfprov.rename(columns={'npi_prov_bill': 'attr_prov'})
            # Fill missing with a state specific ID so missing in LA is not same as missing in TN
            dfy = dfy.merge(dfprov, on=['recip_id', 'year'], how='left')
            dfy['attr_prov'] = dfy['attr_prov'].fillna(f'missing_{state}')        
            
            dfy.to_pickle(output_p+f'{state}_analytic_table_yearly_{year}.pkl')


def paper1_truncated_regressions_yearly_with_attr_prov():
    """
    Run regressions for paper 1 using the yearly files
    """

    output_p, temp_p, log_p, code_p = gen_paths(FS().analysis_p
                                            +'paper1_truncated_regressions_yearly_with_attr_prov/')

    path_pre = FS().derived_p+f'create_analytic_tables_truncated/output/'

    for year in range(2010, 2019):
        files = [x for x in os.listdir(path_pre) if str(year) in x]
        print(files)
        res = pd.concat([pd.read_pickle(path_pre+f) for f in files])

        out_cols = ['all_cost', 'pc_numdays', 'ed_avoid_numdays']
        cont_cols = (['recip_race_g', 'recip_age_g', 'recip_gender_g', 'recip_zip', 'elig_cat',
                      'state', 'plan_id_g', 'attr_prov']
                     + [x for x in res.columns if 'hcc_' in x])

        res = res[out_cols+cont_cols]
        # So plan control within state
        res['plan_id_g'] = res.groupby(['plan_id_g', 'state']).ngroup()

        # Only keep enrollees for which we report results
        res = res[res.recip_race_g.isin([0,1])]
        res.groupby('recip_race_g')[out_cols].mean().to_pickle(output_p+f'{year}_averages.pkl')

        res.to_csv(temp_p+f'analytic_table_{year}.csv', chunksize=10**6)

        ws.stata_do(do_file=code_p+'paper1_main_regs_with_prov.do',
                    params=[temp_p+f'analytic_table_{year}.csv',
                            output_p+f'paper1_main_results_with_prov_{year}.csv'],
                    log_path=log_p)

         # Delete temp file
        os.remove(temp_p+f'analytic_table_{year}.csv')


def paper1_truncated_regressions_yearly():
    """
    Run regressions for paper 1 using the yearly files
    """

    output_p, temp_p, log_p, code_p = gen_paths(FS().analysis_p
                                            +'paper1_truncated_regressions_yearly/')

    path_pre = FS().derived_p+f'create_analytic_tables_truncated/output/'

    for year in range(2010, 2019):
        files = [x for x in os.listdir(path_pre) if str(year) in x]
        print(files)
        res = pd.concat([pd.read_pickle(path_pre+f) for f in files])

        out_cols = ['all_cost', 'pc_numdays', 'ed_avoid_numdays']
        cont_cols = (['recip_race_g', 'recip_age_g', 'recip_gender_g', 'recip_zip', 'elig_cat',
                      'state', 'plan_id_g', 'attr_prov']
                     + [x for x in res.columns if 'hcc_' in x])

        res = res[out_cols+cont_cols]
        # So plan control within state
        res['plan_id_g'] = res.groupby(['plan_id_g', 'state']).ngroup()

        # Only keep enrollees for which we report results
        res = res[res.recip_race_g.isin([0,1])]
        res.groupby('recip_race_g')[out_cols].mean().to_pickle(output_p+f'{year}_averages.pkl')

        res.to_csv(temp_p+f'analytic_table_{year}.csv', chunksize=10**6)

        ws.stata_do(do_file=code_p+'paper1_main_regs_with_prov.do',
                    params=[temp_p+f'analytic_table_{year}.csv',
                            output_p+f'paper1_main_results_with_prov_{year}.csv'],
                    log_path=log_p)

         # Delete temp file
        os.remove(temp_p+f'analytic_table_{year}.csv')


def paper1_truncated_regressions_yearly_demo_only():
    """
    Run regressions for paper 1 using the yearly files
    """

    output_p, temp_p, log_p, code_p = gen_paths(FS().analysis_p
                                            +'paper1_truncated_regressions_yearly_demo_only/')

    path_pre = FS().derived_p+f'create_analytic_tables_truncated/output/'

    for year in range(2010, 2019):
        files = [x for x in os.listdir(path_pre) if str(year) in x]
        print(files)
        res = pd.concat([pd.read_pickle(path_pre+f) for f in files])

        out_cols = ['all_cost', 'pc_numdays', 'ed_avoid_numdays']
        cont_cols = (['recip_race_g', 'recip_age_g', 'recip_gender_g', 'recip_zip', 'elig_cat',
                      'state', 'plan_id_g', 'attr_prov']
                     + [x for x in res.columns if 'hcc_' in x])

        res = res[out_cols+cont_cols]
        # So plan control within state
        res['plan_id_g'] = res.groupby(['plan_id_g', 'state']).ngroup()

        # Only keep enrollees for which we report results
        res = res[res.recip_race_g.isin([0,1])]
        res.groupby('recip_race_g')[out_cols].mean().to_pickle(output_p+f'{year}_averages.pkl')

        res.to_csv(temp_p+f'analytic_table_{year}.csv', chunksize=10**6)

        ws.stata_do(do_file=code_p+'paper1_main_regs_with_prov.do',
                    params=[temp_p+f'analytic_table_{year}.csv',
                            output_p+f'paper1_main_results_with_prov_{year}.csv'],
                    log_path=log_p)

         # Delete temp file
        os.remove(temp_p+f'analytic_table_{year}.csv')


def paper1_disparities_over_time():
    output_p = FS().analysis_p +'paper1_disparities_over_time/output/'
    data_p = FS().analysis_p+'paper1_truncated_regressions_yearly_with_attr_prov/output/'

    l = []
    for year in range(2013, 2016):
        dfa = pd.read_pickle(data_p+f'{year}_averages.pkl')
        dfc = pd.read_csv(data_p+f'paper1_main_results_with_prov_{year}.csv', sep='\t')

        dfc = (dfc[dfc.variable.eq('1.recip_race_g')]
               .drop(columns=['regression', 'pval', 'variable'])
               .assign(year=year)
               .set_index(['outcome', 'year']))
        dfa = dfa.loc[1]

        dfc['White Mean'] = dfc.index.get_level_values('outcome').map(dfa)
        dfc = dfc[['White Mean', 'coeff', 'se', 'lb', 'ub']]
        
        m = dfc.index.get_level_values('outcome').isin(['ed_avoid_numdays', 'pc_numdays'])
        dfc.loc[m, :] = dfc.loc[m, :]*100
        s = dfc[['coeff', 'lb', 'ub']].div(dfc['White Mean'], axis=0).mul(100).round(1)
        dfc['% Disparitity'] = (s['coeff'].astype(str) + ' [' + s['lb'].astype(str)
                                + ', ' + s['ub'].astype(str)+']')
        l.append(dfc)

    res = pd.concat(l)
    res = res.sort_index()

    fig, ax = plt.subplots(figsize=(6,10), nrows=3)
    ax = ax.flatten()
    plt.subplots_adjust(hspace=0.3)
    for i, (idx, gp) in enumerate(res.groupby('outcome')):
        gp = gp.xs(idx, level='outcome')
        ax[i].errorbar(x=gp.index, y=gp.coeff, yerr=1.96*gp.se,
                       marker='o', lw=0, elinewidth=2, capsize=2, zorder=4)
        ax[i].axhline(0,0,1, color='black', lw=2, linestyle='--', zorder=2)
        ax[i].set_title(idx, fontsize=12)
    ax[0].set_ylim(-600,100)
    ax[1].set_ylim(-3,5)
    ax[2].set_ylim(-100,10)


####################################################################################################
############################################ PAPER 2 ###############################################
####################################################################################################
def paper2_create_primary_care():
    output_p = FS().derived_p+'paper2_create_primary_care/output/'

    l = [('TN', FS().tn_pipe_gold_path, (2010, 2016)), 
         ('LA', FS().la_pipe_gold_path, (2012, 2015))
        ]

    for state, gold_path, yr in l:
        file_specs = gen_filespecs(state, yr)

        df = utilization.ambulatory_utilization(gold_path=gold_path,
                                                file_specs=file_specs,
                                                keep_paid_only=True, 
                                                spec_list=None)
        
        df.to_pickle(output_p+f'{state}_pc_claims.pkl')
        
        # Now aggregate
        dfc = pd.concat([pd.read_pickle(gold_path+f'claims_{yr}.pkl') for yr in range(*yr)])
        dfc = dfc[dfc.claim_id.isin(df.claim_id)]

        # Count days as 
        dfc = (dfc.groupby(['recip_id', dfc.claim_date.dt.year.rename('year'),
                            dfc.claim_date.dt.month.rename('month')])
              .agg(cost=('claim_amt_paid', 'sum'),
                   numclms=('claim_id', 'nunique'),
                   numdays=('claim_date', 'nunique'))
              .add_prefix('pc_'))
        dfc.to_pickle(output_p+f'{state}_pc.pkl')


def paper2_create_lab_and_tests():
    output_p = FS().derived_p+'paper2_create_lab_and_tests/output/'

    l = [('TN', FS().tn_pipe_gold_path, (2010, 2016)), 
         ('LA', FS().la_pipe_gold_path, (2012, 2015))
        ]

    for state, path, yr in l:
        file_specs = [(f'claims_details_{year}.pkl', ['CPT', 'HCPCS', 'POS'], 'details') 
                        for year in range(*yr)]

        df = utilization.BETOS_categorization(gold_path=path,
                                              file_specs=file_specs,
                                              keep_paid_only=True, 
                                              level='broad')
        
        dfc = pd.concat([pd.read_pickle(path+f'claims_{year}.pkl') for year in range(*yr)])

        d = {}
        for cat in ['Imaging', 'Tests']:
            claims = df[df.category.eq(cat)].claim_id.unique()
            gp = dfc[dfc.claim_id.isin(claims)]
            d[cat] = (gp.groupby(['recip_id', gp.claim_date.dt.year.rename('year'),
                                  gp.claim_date.dt.month.rename('month')])
                        .agg(cost=('claim_amt_paid', 'sum'),
                             numclms=('claim_id', 'nunique'),
                             numdays=('claim_date', 'nunique')))

        res = pd.concat(d, axis=1)
        res.columns = [f'{x.lower()}_{y}' for x,y in res.columns]
        res = res.reset_index()
        res.to_pickle(output_p+f'{state}_lab_tests.pkl')


def paper2_create_ed():
    output_p = FS().derived_p+'paper2_create_ed/output/'

    l = [('TN', FS().tn_pipe_gold_path, (2010, 2016)), 
         ('LA', FS().la_pipe_gold_path, (2012, 2015))
        ]

    for state, gold_path, yr in l:
        file_specs = gen_filespecs(state, yr)

        df = utilization.ED_utilization(gold_path=gold_path,
                                        file_specs=file_specs,
                                        HCPCS_CPT=True,
                                        revenue=True,
                                        pos=True,
                                        keep_paid_only=True, 
                                        remove_admissions=True,
                                        limit_one_per_day=False)

        # Now aggregate
        dfc = pd.concat([pd.read_pickle(gold_path+f'claims_{yr}.pkl') for yr in range(*yr)])
        dfc = dfc[dfc.claim_id.isin(df.claim_id)]
        
        dfc = (dfc.groupby(['recip_id', dfc.claim_date.dt.year.rename('year'),
                      dfc.claim_date.dt.month.rename('month')])
              .agg(cost=('claim_amt_paid', 'sum'),
                   numclms=('claim_id', 'nunique'),
                   numdays=('claim_date', 'nunique'))
              .add_prefix('ednoadmit_'))
        dfc.to_pickle(output_p+f'{state}_ed.pkl')


def paper2_create_specialty_care():
    """
    Defined as office care not in the PC measure
    """
    output_p = FS().derived_p+'paper2_create_specialty_care/output/'

    l = [('TN', FS().tn_pipe_gold_path, (2010, 2016)), 
         ('LA', FS().la_pipe_gold_path, (2012, 2015))
        ]

    for state, gold_path, yr in l:
        dfpc = pd.read_pickle(FS().derived_p
                              +f'paper2_create_primary_care/output/{state}_pc_claims.pkl')

        dfc = pd.concat([pd.read_pickle(gold_path+f'claims_{year}.pkl') for year in range(*yr)],
                        sort=False)
        dfcd = pd.concat([pd.read_pickle(gold_path+f'claims_details_{year}.pkl') 
                          for year in range(*yr)], sort=False)

        office = dfcd[dfcd.pos_code.eq('11')].claim_id.unique()
        specs = dfc[dfc.claim_id.isin(office) & dfc.claim_paid.eq(True)
                    & ~dfc.claim_id.isin(dfpc.claim_id)]

        # Count days as 
        specs = (specs.groupby(['recip_id', specs.claim_date.dt.year.rename('year'),
                                specs.claim_date.dt.month.rename('month')])
                  .agg(cost=('claim_amt_paid', 'sum'),
                       numclms=('claim_id', 'nunique'),
                       numdays=('claim_date', 'nunique'))
                  .add_prefix('spec_'))
        specs.to_pickle(output_p+f'{state}_spec.pkl')


def paper2_create_hvc_drugs():
    output_p = FS().derived_p+'paper2_create_hvc_drugs/output/'

    l = [('TN', FS().tn_pipe_gold_path, (2010, 2016)), 
         ('LA', FS().la_pipe_gold_path, (2012, 2015))
        ]

    for state, gold_path, yr in l:
        df = pd.concat([pd.read_pickle(gold_path+f'pharm_{year}.pkl') for year in range(*yr)],
                       ignore_index=True)
        df = df[df.claim_paid.eq(True)]
        
        df = categorize_drug_class(df, return_cost=True)
        
        df = (df.groupby(['recip_id', df.claim_date.dt.year.rename('year'),
                          df.claim_date.dt.month.rename('month'), 'drug_class'])
                .agg(cost=('claim_amt_paid', 'sum'),
                     numclms=('claim_id', 'nunique'))
                 .unstack(-1))

        df.columns = [f'{y.lower()}_{x}' for x,y in df.columns]
        df = df.reset_index()

        df.to_pickle(output_p+f'{state}_hvc_drugs.pkl')


def paper2_create_brand_generic():
    output_p = FS().derived_p+'paper2_create_brand_generic/output/'

    l = [('TN', FS().tn_pipe_gold_path, (2010, 2016)), 
         ('LA', FS().la_pipe_gold_path, (2012, 2015))
        ]

    for state, gold_path, yr in l:

        # Create brand/Generic Utilization/spending
        dfp = pd.concat([pd.read_pickle(gold_path+f'pharm_{year}.pkl') for year in range(*yr)],
                       ignore_index=True)
        dfp = dfp[dfp.claim_paid.eq(True)]

        dfp = (dfp.groupby(['recip_id', dfp.claim_date.dt.year.rename('year'),
                           dfp.claim_date.dt.month.rename('month'), 
                           dfp['drug_is_generic'].map({True: 'generic', False: 'brand'})])
                  .agg(cost=('claim_amt_paid', 'sum'),
                       numclms=('claim_id', 'nunique'),
                       numdays=('claim_date', 'nunique'))
                  .unstack(-1))

        dfp.columns = [f'{y}_{x}' for x,y in dfp.columns]
        dfp.to_pickle(output_p+f'{state}_brand_generic.pkl')


def paper2_create_inpatient():
    output_p = FS().derived_p+'paper2_create_inpatient/output/'

    l = [('TN', FS().tn_pipe_gold_path, (2010, 2016)), 
         ('LA', FS().la_pipe_gold_path, (2012, 2015))
        ]

    for state, gold_path, yr in l:

        # Create brand/Generic Utilization/spending
        dfc = pd.concat([pd.read_pickle(gold_path+f'claims_{year}.pkl') for year in range(*yr)],
                       ignore_index=True)
        dfc = dfc[dfc.claim_paid.eq(True)]

        # Different logic by state to get Inpatient
        if state == 'LA':
            dfc = dfc[dfc.yale_cos.isin(['inpat_psych_hosp', 'inpat_hosp'])]
        elif state == 'TN':
            dfid = pd.concat([pd.read_pickle(gold_path+f'inpatient_details_{year}.pkl') 
                              for year in range(*yr)], ignore_index=True)
            dfc = dfc[dfc.claim_id.isin(dfid.claim_id)]

        # Get MM Cost/Utilization
        dfc = (dfc.groupby(['recip_id', dfc.claim_date.dt.year.rename('year'),
                            dfc.claim_date.dt.month.rename('month')])
                  .agg(inpat_cost=('claim_amt_paid', 'sum'),
                       inpat_numclms=('claim_id', 'nunique'),
                       inpat_numdays=('claim_date', 'nunique')))

        dfc.to_pickle(output_p+f'{state}_inpatient.pkl')


def paper2_grab_AA_analytic_tables():
    output_p = FS().derived_p+'paper2_grab_AA_analytic_tables/output/'

    # LA Three year Sample
    f = ('D:/Groups/YSPH-HPM-Ndumele/Networks/Anthony/LA_Plan_Effects/trunk/'
         +'pccm_v_mco/add_adnl_outcomes/output/threeyear_analytic_mm.pkl')
    df = pd.read_pickle(f)

    # Make instruments the plans, not the MCO/PCCM
    pland = {'4508073': 3, '4508067': 1, '4508090': 5, '4508063': 2, '4508062': 4}
    df['plan_id_asgn'] = df['plan_id_asgn_five'].map(pland)
    df['plan_id'] = df['plan_id_five'].map(pland)

    # Remove some AAs with a weird plan in later years
    to_drop = (df[~df.plan_id.isin([1,2,3,4,5])].recip_id.unique())
    df = df[~df.recip_id.isin(to_drop)].copy()

    # Remove anyone over 65 during study period
    to_drop = (df[df.recip_age.ge(65)].recip_id.unique())
    df = df[~df.recip_id.isin(to_drop)].copy()

    # Race, Age and Gender controls
    df['recip_race_og'] = df.recip_race.copy()
    df['recip_race_g'] = map_race(df, 'LA')
    df['recip_age_g'] = (df['recip_age']//5).clip(upper=12).mul(5)
    df['recip_gender_g'] = df.recip_gender.map({'M': 0, 'F': 1})  

    df['plan_race_asgn'] = df['plan_id_asgn']*10+df['recip_race_g']
    df['plan_race'] = df['plan_id']*10+df['recip_race_g']

    # Reconcile name differences across projects 
    df['all_cost'] = df['all_clms_amt_paid']
    df['pharm_cost'] = df['pharm_amt_paid']
    df['medical_cost'] = df['all_cost'] - df['pharm_cost']
    df['pharm_numclms'] = df['pharm_num_clms']
    df['medical_numclms'] = df['all_clms_num_clms'] - df['pharm_num_clms']

    df = map_eligibility(df, 'LA')

    keep_cols = ['recip_id', 'year', 'month', 'date', 'anchor_date', 'asgn_type', 'med_elig_cat',
                 'med_elig_type_case', 'mos_aft_anch', 'plan_id', 'plan_id_asgn',
                 'plan_id_asgn_five', 'plan_id_five', 'recip_race', 'recip_race_g',
                 'recip_birthday', 'recip_age', 'recip_age_g', 'recip_race_og', 'recip_gender', 'recip_gender_g',
                 'plan_race_asgn', 'plan_race', 'recip_zip', 'recip_county_fips', 'uor', 'segment',
                 'elig_cat', 'all_cost', 'pharm_cost', 'medical_cost', 'pharm_numclms',
                 'medical_numclms']

    df = df[keep_cols]
    print(df.groupby('asgn_type').recip_id.nunique())
    df.to_pickle(output_p+'LA_sample.pkl')

    # TN Disabled Sample
    f = ('D:/Groups/YSPH-HPM-Ndumele/Networks/Anthony/TN_Network/trunk/'
         +'derived/create_true_analytic_table_with_AC/output/analytic_table_12M_withAC.pkl')
    df = pd.read_pickle(f)

    # Make instruments the plans, not the planXregion
    # Make numeric so works with Stata. 
    pland = {'UnitedHealth': 1, 'AmeriGroup': 2, 'BlueCare': 3}
    df['plan_id_asgn'] = df['plan_id_asgn_name'].map(pland)
    df['plan_id'] = df['plan_id_name'].map(pland)

    # Race, Age and Gender controls
    df['recip_race_og'] = df.recip_race.copy()
    df['recip_race_g'] = map_race(df, 'TN')
    df['recip_age_g'] = (df['recip_age']//5).clip(upper=12).mul(5)
    df['recip_gender_g'] = df.recip_gender.map({'M': 0, 'F': 1})  

    df['plan_race_asgn'] = df['plan_id_asgn']*10+df['recip_race_g']
    df['plan_race'] = df['plan_id']*10+df['recip_race_g']


    df = map_eligibility(df, 'TN')
    df['medical_cost'] = df['all_nonph_cost']
    df['pharm_cost'] = df['all_pharm_cost']
    df['all_cost'] = df['cost_tot']
    df['pharm_numclms'] = df['all_pharm_numclms']
    df['medical_numclms'] = df['all_nonph_numclms'] - df['all_pharm_numclms']

    df = df[[x for x in keep_cols if x not in ['plan_id_five', 'plan_id_asgn_five']
            + ['plan_id_name', 'plan_id_asgn_name']]]
    print(df.groupby('asgn_type').recip_id.nunique())
    df.to_pickle(output_p+'TN_sample.pkl')


def paper2_augment_AA_analytic_tables():
    """
    Add various outcomes to the analytic tables for the AAs in LA and TN
    """

    output_p = FS().derived_p+'paper2_augment_AA_analytic_tables/output/'

    l = [('TN', FS().tn_pipe_analytic_path, FS().tn_pipe_gold_path, (2010, 2016)), 
         ('LA', FS().la_pipe_analytic_path, FS().la_pipe_gold_path, (2012, 2015))]

    for state, apath, gpath, yr in l:
        df = pd.read_pickle(FS().derived_p
                            +f'paper2_grab_AA_analytic_tables/output/{state}_sample.pkl')

        # Bring over CRG:
        dfcrg = pd.concat([pd.read_pickle(apath+f'3M_output_{year}_crg_200.pkl').assign(year=year)
                           for year in range(*yr)])
        dfcrg = dfcrg.drop_duplicates(['recip_id', 'year'])

        dfcrg['health_status'] = dfcrg['AggregatedCrg3'].astype('str').str[0]
        dfcrg = dfcrg.rename(columns={'AggregatedCrg3': 'crg3', 'AggregatedCrg2': 'crg2'})
        df = df.merge(dfcrg[['recip_id', 'year', 'health_status', 'crg3', 'crg2']],
                      on=['recip_id', 'year'], how='left')

        # Bring over inpatient
        dflt = pd.read_pickle(FS().derived_p
                              +f'paper2_create_inpatient/output/{state}_inpatient.pkl')
        adnl_cols = dflt.columns.difference(['recip_id', 'year', 'month']).tolist()
        df = df.merge(dflt, on=['recip_id', 'year', 'month'], how='left')
        df[adnl_cols] = df[adnl_cols].fillna(0)
        # Non-medical minus inpatient. 
        df['other_cost'] = df['medical_cost'] - df['inpat_cost']

        # Bring over brand/generic
        dflt = pd.read_pickle(FS().derived_p
                              +f'paper2_create_brand_generic/output/{state}_brand_generic.pkl')
        adnl_cols = dflt.columns.difference(['recip_id', 'year', 'month']).tolist()
        df = df.merge(dflt, on=['recip_id', 'year', 'month'], how='left')
        df[adnl_cols] = df[adnl_cols].fillna(0)

        # Bring over labs and tests
        dflt = pd.read_pickle(FS().derived_p
                              +f'paper2_create_lab_and_tests/output/{state}_lab_tests.pkl')
        adnl_cols = dflt.columns.difference(['recip_id', 'year', 'month']).tolist()
        df = df.merge(dflt, on=['recip_id', 'year', 'month'], how='left')
        df[adnl_cols] = df[adnl_cols].fillna(0)
        # Create combined measure
        df['tests_imaging_numdays'] = df[['tests_numdays', 'imaging_numdays']].sum(axis=1)

        # Bring over PC
        dflt = pd.read_pickle(FS().derived_p+f'paper2_create_primary_care/output/{state}_pc.pkl')
        adnl_cols = dflt.columns.difference(['recip_id', 'year', 'month']).tolist()
        df = df.merge(dflt, on=['recip_id', 'year', 'month'], how='left')
        df[adnl_cols] = df[adnl_cols].fillna(0)
        df['pc_any'] = df['pc_numdays'].clip(upper=1)

        # Bring over speciality
        dflt = pd.read_pickle(FS().derived_p
                              +f'paper2_create_specialty_care/output/{state}_spec.pkl')
        adnl_cols = dflt.columns.difference(['recip_id', 'year', 'month']).tolist()
        df = df.merge(dflt, on=['recip_id', 'year', 'month'], how='left')
        df[adnl_cols] = df[adnl_cols].fillna(0)

        # Bring over ED
        dflt = pd.read_pickle(FS().derived_p
                              +f'paper2_create_ed/output/{state}_ed.pkl')
        adnl_cols = dflt.columns.difference(['recip_id', 'year', 'month']).tolist()
        df = df.merge(dflt, on=['recip_id', 'year', 'month'], how='left')
        df[adnl_cols] = df[adnl_cols].fillna(0)

        # Bring HVC Drugs
        dflt = pd.read_pickle(FS().derived_p
                              +f'paper2_create_hvc_drugs/output/{state}_hvc_drugs.pkl')
        adnl_cols = dflt.columns.difference(['recip_id', 'year', 'month']).tolist()
        df = df.merge(dflt, on=['recip_id', 'year', 'month'], how='left')
        df[adnl_cols] = df[adnl_cols].fillna(0)

        # Bring over Avoidable ED Outcome
        dfed = pd.read_pickle(apath+'ED_claims_avoidable.pkl')
        dfed['year'] = dfed.claim_date.dt.year
        dfed['month'] = dfed.claim_date.dt.month
        dfed = (dfed.groupby(['year', 'month', 'recip_id'])
                    .agg(ed_avoid_numdays=('claim_date', 'nunique')))
        df = df.merge(dfed, on=['recip_id', 'year', 'month'], how='left')
        df['ed_avoid_numdays'] = df['ed_avoid_numdays'].fillna(0, downcast='infer')

        # Bring over the area deprivation index
        dfadi = pd.read_pickle(Pipeline().path
                               +f'Resource_Data/NeighborAtlas/Gold/{state}_adi_zip_2015.pkl')
        dfadi = dfadi.groupby('zip5')['adi_natnl'].mean()
        df['adi_natnl'] = df['recip_zip'].map(dfadi)

        # Urban vs. Rural
        dfurb = (pd.read_pickle(Pipeline().path
                                +'Resource_Data/Census/Gold/county_urban_rural_classification.pkl')
                   .set_index('fips')['urban_category'])
        df['urban_category'] = df['recip_county_fips'].map(dfurb)

        # Bring over HHS HCCs
        dfhhs = pd.concat([pd.read_pickle(apath+f'HCCS_HHS_{year}.pkl').assign(year=year)
                           for year in range(*yr)])
        df = df.merge(dfhhs, on=['recip_id', 'year'], how='left')
        hcc_cols = [x for x in df.columns if x.startswith('hcc_')]
        df[hcc_cols] = df[hcc_cols].fillna(0)

        df.to_pickle(output_p+f'{state}_analytic_table_mm.pkl')


def paper2_analytic_tables_to_yearly():
    """
    Convert the LA table from MM to yearly with proper aggregations
    """

    output_p = FS().derived_p+'paper2_analytic_tables_to_yearly/output/'
    qpath = FS().la_pipe_analytic_path

    df = pd.read_pickle(FS().derived_p+'paper2_augment_AA_analytic_tables/output/'
                        +'LA_analytic_table_mm.pkl')

    ####
    # Yearly Table
    #####
    sumcols = [x for x in df.columns if '_cost' in x or '_numclms' in x or '_numdays' in x
                                     or '_supplydays' in x]
    firstcols = ['recip_gender', 'recip_race', 'recip_race_g', 'recip_age',
                 'recip_gender_g', 'health_status', 'crg3', 'crg2' ,'asgn_type', 
                 'plan_id_asgn', 'plan_race_asgn', 'uor']
    modecols = ['recip_age_g', 'plan_id', 'recip_zip', 'recip_county_fips', 
                 'elig_cat', 'plan_race']
    maxcols = ([x for x in df.columns if x.startswith('hcc_')] 
                + ['pc_any'])

    agg_ds = {x: 'sum' for x in sumcols}
    agg_df = {x: 'first' for x in firstcols}
    agg_dm = {x: 'max' for x in maxcols}
    agg_d = {**agg_ds, **agg_df, **agg_dm}

    dfy = df.groupby(['recip_id', 'year']).agg(agg_d).reset_index()

    for col in modecols:
        s = dt.fast_mode(df, key_cols=['recip_id', 'year'], value_col=col)
        dfy = dfy.merge(s, on=['recip_id', 'year'], how='left')

    # Winsorize cost
    for col in ['all_cost', 'pharm_cost', 'medical_cost']:
        dfy[f'{col}_wins50'] = dfy[col].clip(upper=50000)
        dfy[f'{col}_wins25'] = dfy[col].clip(upper=25000)
        dfy[f'{col}_wins100'] = dfy[col].clip(upper=100000)
        dfy[f'{col}_log1p'] = np.log1p(dfy[col])

    # Also bring over yearly quality 
    quality_l = [('capch_1to2', ['CAP_CH_1to2_HEDIS.pkl']),
                 ('capch_2to6', ['CAP_CH_2to6_HEDIS.pkl']),
                 ('capch_7to12', ['CAP_CH_7to12_HEDIS.pkl']),
                 ('capch_12to19', ['CAP_CH_12to19_HEDIS.pkl']),
                 ('awc_ch', ['AWC_CH_HEDIS.pkl']),
                 ('awc_adolesc', ['AWC_ADOLESC_HEDIS.pkl']),
                 ('capch_all', ['CAP_CH_1to2_HEDIS.pkl', 'CAP_CH_2to6_HEDIS.pkl',
                                'CAP_CH_7to12_HEDIS.pkl', 'CAP_CH_12to19_HEDIS.pkl']),
                 ('awc_all', ['AWC_CH_HEDIS.pkl', 'AWC_ADOLESC_HEDIS.pkl']),
                 ('wcv_ch', ['WCV_CH_HEDIS.pkl']),
                 ('wcv_3_ch', ['WCV_CH_3_HEDIS.pkl']),
                 ('wcv_12_ch', ['WCV_CH_12_HEDIS.pkl']),
                 ('wcv_18_ch', ['WCV_CH_18_HEDIS.pkl']),
                 ('aap_ad', ['AAP_AD_HEDIS.pkl']),
                 ('chl_all', ['CHL_AD_HEDIS.pkl', 'CHL_CH_HEDIS.pkl']),
                 ('bcs', ['BCS_AD_HEDIS.pkl']),
                 ('ha1c', ['HA1C_HEDIS.pkl']),
                 ('ssd', ['SSD_AD_HEDIS.pkl']),
                 ('amr', ['AMR_HEDIS.pkl']),
                 ('add', ['ADD_CH_HEDIS.pkl'])
                ]

    for metric, f_list in quality_l:
        dfq = pd.concat([pd.read_pickle(qpath+f) for f in f_list])
        dfq = dfq.rename(columns={'numer': metric})
        dfy = dfy.merge(dfq[['recip_id', metric, 'year']], on=['recip_id', 'year'], how='left')

    # Bring over baseline HCCS
    dfh = pd.read_pickle(qpath+'HCCS_HHS_2011.pkl')
    dfh = dfh.set_index('recip_id').add_prefix('b_')
    dfy = dfy.merge(dfh, on='recip_id', how='left')
    dfy[dfh.columns] = dfy[dfh.columns].fillna(0)

    dfy.to_pickle(output_p+'LA_analytic_table_yearly.pkl')
    

def paper2_create_baseline_tables():
    """
    Create baseline table a time of assignment
    """

    output_p = FS().derived_p+'paper2_create_baseline_tables/output/'

    l = [('TN', FS().tn_pipe_analytic_path, FS().tn_pipe_gold_path, (2010, 2016)), 
         ('LA', FS().la_pipe_analytic_path, FS().la_pipe_gold_path, (2012, 2015))
        ]

    for state, apath, gpath, yr in l:
        df = pd.read_pickle(FS().derived_p+'paper2_augment_AA_analytic_tables/'
                            +f'output/{state}_analytic_table_mm.pkl')
        # Get variables at time of assignment
        df = df[df.mos_aft_anch.eq(0)]
        
        # Remove any outcomes that come from post assignment
        drop_cols = ([x for x in df.columns if 'hcc_' in x or '_numclms' in x
                      or '_cost' in x or '_numdays' in x]
                     + ['health_status', 'crg2', 'crg3', 'pc_any'])
        df = df.drop(columns=drop_cols)
        
        df.to_pickle(output_p+f'{state}_baseline.pkl')


def paper2_grab_previous_diag_and_pharm():
    """
    For use in baseline risk prediction. 
    """
    output_p = FS().derived_p+'paper2_grab_previous_diag_and_pharm/output/'

    l = [('TN', FS().tn_pipe_gold_path, (2010, 2016)), 
         ('LA', FS().la_pipe_gold_path, (2011, 2015))
        ]

    for state, gpath, yr in l:
        df = pd.read_pickle(FS().derived_p+'paper2_create_baseline_tables/output/'
                            +f'{state}_baseline.pkl')

        # Get prior claims details and diagnoses
        for ftype in ['diagnosis', 'pharm']:
            l = []
            for year in range(*yr):
                gp = pd.read_pickle(gpath+f'{ftype}_{year}.pkl')
                l.append(gp)
            gp = pd.concat(l, sort=False, ignore_index=False)
            gp = subset_b4_time_by_recip(gp, df_anch=df, interval=(0, 1), unit='year')         
            gp.to_pickle(output_p+f'{state}_{ftype}.pkl')


def paper2_create_predicted_risk():
    """
    Create measures of spending 'risk' from baseline features
    """
    from sklearn.linear_model import LassoCV

    output_p = FS().derived_p+'paper2_create_predicted_risk/output/'

    for state in ['LA', 'TN']:
        dfo = pd.read_pickle(FS().derived_p+'paper2_augment_AA_analytic_tables/'
                             +f'output/{state}_analytic_table_mm.pkl')    
        # Mean spending post assignment
        dfo = dfo.groupby('recip_id')['all_cost'].mean()
        
        dfb = pd.read_pickle(FS().derived_p+'paper2_create_baseline_tables/output/'
                             +f'{state}_baseline.pkl')

        # Limit to Black/White enrollees
        dfb = dfb[dfb.recip_race_g.isin([0,1])]
        # Winsorize at 50K yearly 
        dfb['risk_outcome'] = dfb.recip_id.map(dfo).clip(upper=50000/12)
        
        for ftype,col in [('diagnosis', 'diag_code'), ('pharm', 'drug_ndc')]:
            gp = pd.read_pickle(FS().derived_p+'paper2_grab_previous_diag_and_pharm/output/'
                                +f'{state}_{ftype}.pkl')
            gp = gp[gp.groupby(col).recip_id.transform('nunique').ge(dfb.recip_id.nunique()*0.0035)]

            gp = (pd.crosstab(gp.recip_id, gp[col]).clip(upper=1)
                    .rename_axis(columns=None).add_prefix(f'{col}_'))
            dfb = dfb.merge(gp, on='recip_id', how='left')
            for fcol in [x for x in dfb.columns if x.startswith(f'{col}_')]:
                dfb[fcol] = dfb[fcol].fillna(0, downcast='int')
                
        ### LASSO Model SHOULD REMOVE RECIPIENT RACE
        columns = ['asgn_type', 'uor', 'recip_age_g', 'recip_gender_g',
                   'elig_cat', 'recip_county_fips']
        diag = True
        pharm = True

        outcome = 'risk_outcome'

        np.random.seed(42)
        train = dfb.copy()
        # Create the Dummies Matrices for Fitting
        df_train = pd.get_dummies(train[columns], columns=columns)

        if diag:
            df_train = pd.concat([df_train, train.filter(like='diag_code').astype(int)], axis=1)
        if pharm:
            df_train = pd.concat([df_train, train.filter(like='drug_ndc').astype(int)], axis=1)
        X_train = df_train.to_numpy()
        Y_train = train[outcome].to_numpy()

        reg = LassoCV(cv=5, random_state=0, max_iter=10000).fit(X_train, Y_train)
        print('Score:', reg.score(X_train, Y_train))

        pred = pd.Series(reg.predict(X_train), index=df_train.index)
        pred = pd.concat([dfb[['recip_id']], pred.to_frame('predicted_risk')], axis=1)
        pred.to_pickle(output_p+f'{state}_predicted_risk.pkl')


def paper2_app_race_mapping():
    output_p = FS().analysis_p+'paper2_app_race_mapping/output/'

    df = pd.read_pickle(FS().derived_p+'paper2_grab_AA_analytic_tables/output/LA_sample.pkl')
    df = df.drop_duplicates('recip_id')

    dfp = pd.read_pickle(FS().la_pipe_raw_path+'person_details.pkl.gz')
    # Race is unique w/r/t NEW_ELIG_ID
    dfp = dfp.drop_duplicates('NEW_ELIG_ID')
    dfp = dfp[dfp.NEW_ELIG_ID.isin(df.recip_id.unique())]

    print('N Enrollees:', dfp.NEW_ELIG_ID.nunique())

    d = {'0': 'Not declared',
         '1': 'White',                                                                               
         '2': 'Black or African American',                                                                              
         '3': 'American Indian or Alaskan Native',                                                                                 
         '4': 'Asian',                                                                                  
         '5': 'Hispanic or Latino (no other race info)',                                                                                                  
         '6': 'Native Hawaiian or Other Pacific Islander',                                                                                              
         '7': 'Hispanic or Latino and one or more other races',                                                                                                                 
         '8': 'More than one race indicated (and not Hispanic or Latino)',                                                                     
         '9': 'Unknown'}

    s = dfp.ELS_Race.value_counts(normalize=True).mul(100).round(1).reset_index()
    s['ELS_Race'] = s['ELS_Race'].map(d)
    s = s.sort_values('ELS_Race')
    s

    s.to_csv(output_p+'Race_proportions.csv', index=False)


def paper2_augment_baseline_tables():
    """
    Create baseline table a time of assignment
    """

    output_p = FS().derived_p+'paper2_augment_baseline_tables/output/'

    for state in ['TN', 'LA']:
        df = pd.read_pickle(FS().derived_p+'paper2_create_baseline_tables/output/'
                            +f'{state}_baseline.pkl')
        dfr = pd.read_pickle(FS().derived_p+'paper2_create_predicted_risk/output/'
                            +f'{state}_predicted_risk.pkl')
        df = df.merge(dfr, on='recip_id', how='left')

        df.to_pickle(output_p+f'{state}_baseline.pkl')


def paper2_run_balance_regressions():
    """
    Balance check on LA
    """   
    def clean_baseline_table(df):
        cols = ['Fstat', 'pval', 'se']
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
        df = df

        # Contains F-stat and P-val
        s1 = (df.loc[df.Nobs.notnull(), ['Baseline', 'Fstat', 'pval']]
                .set_index(['Baseline']))

        # Holds individual plan effects
        s2 = df.loc[df.Nobs.isnull(), ['Baseline', 'coeff', 'se']]

        s2['plan'] = s2.Baseline.str.split('.').str[0].str.rsplit('_', n=1).str[1]
        s2 = s2[~s2.plan.eq('cons')].copy()

        s2['Baseline'] = s2.Baseline.str.split('.').str[0].str.rsplit('_', n=1).str[0]

        s2 = (s2.pivot_table(index=['Baseline'], columns='plan', values='coeff',
                             aggfunc='first')
                .rename_axis(None, axis=1))

        res = pd.concat([s2, s1], axis=1, sort=False)

        return res


    output_p, temp_p, log_p, code_p = gen_paths(FS().analysis_p
                                                +'paper2_run_balance_regressions/')

    df = pd.read_pickle(FS().derived_p+'paper2_augment_baseline_tables\output\LA_baseline.pkl')

    df = df[df.recip_race_g.isin([0,1])].copy()
    df['plan_id_test'] = df.plan_id_asgn.where(df.asgn_type.eq('AA')).fillna(df.plan_id)

    df['is_black'] = df.recip_race_g
    df['is_white'] = (1-df.recip_race_g)
    df['is_female'] = df.recip_gender_g
    df['is_male'] = (1-df.recip_gender_g)
    df['age_le5'] = df['recip_age'].le(5).astype(int)
    df['age_6to17'] = df['recip_age'].between(5, 18).astype(int)
    df['age_18to64'] = df['recip_age'].between(18, 64).astype(int)

    # Bring over baseline health 
    dfh = (pd.read_pickle(FS().la_pipe_analytic_path+'HCCS_HHS_2011.pkl')
             .set_index('recip_id'))

    df = df.merge(dfh, on='recip_id', how='left')
    df[dfh.columns] = df[dfh.columns].fillna(0)

    # Create HCC categories.
    grps = [('asthma', ['hcc_161.2','hcc_161.1', 'hcc_160']),
            ('dep_bip_psych', ['hcc_88', 'hcc_87.1', 'hcc_87.2', 'hcc_90', 
                               'hcc_102', 'hcc_103']),
            ('diabetes', ['hcc_20', 'hcc_21']),
            ('drug_sud', ['hcc_81', 'hcc_82', 'hcc_83', 'hcc_84']),
            ('pregnancy', ['hcc_203', 'hcc_204', 'hcc_205', 'hcc_207',
                            'hcc_208', 'hcc_209', 'hcc_210', 'hcc_211', 'hcc_212']),
            ('cardiov_cond', ['hcc_131', 'hcc_139', 'hcc_138',
                              'hcc_142', 'hcc_137', 'hcc_135', 'hcc_132',
                              'hcc_130', 'hcc_129', 'hcc_128', 'hcc_127',
                              'hcc_126', 'hcc_125'])
           ]
    for cond, cols in grps:
        df[f'cond_{cond}'] = df[cols].eq(1).any(axis=1).astype(int)
        df['cond_any'] = df.filter(like='hcc_').eq(1).any(axis=1).astype(int)
        df['cond_n'] = df.filter(like='hcc_').sum(axis=1)  

    # Gets some raw numbers
    ncols = ['age_le5', 'age_6to17', 'age_18to64', 'is_female', 'is_male', 'is_black', 'is_white',
             'cond_asthma', 'cond_dep_bip_psych', 'cond_diabetes',
             'cond_pregnancy', 'cond_cardiov_cond']

    res = df.groupby(['asgn_type', 'plan_id_test'])[ncols].sum().stack().unstack([0,1])
    res.to_csv(output_p+'plan_Ns.csv')

    smeans = df.groupby('asgn_type')[ncols+['predicted_risk']].mean().T
    smeans.to_csv(output_p+'sample_means.csv')

    (df.groupby(['asgn_type', 'plan_id_test']).size().unstack(-1)
        .to_csv(output_p+'plan_enrollment.csv'))

    for idx, gp in df.groupby('asgn_type'):
        gp.to_csv(temp_p+f'{idx}.csv', sep=',')

        ws.stata_do(do_file=code_p+'test_balance.do',
                    params=[temp_p+f'{idx}.csv',
                            output_p+f'{idx}_balance_results.csv'],
                    log_path=log_p)

        # Delete temp file
        os.remove(temp_p+f'{idx}.csv')

    dfaa = pd.read_csv(output_p+'AA_balance_results.csv', sep='\t')
    dfac = pd.read_csv(output_p+'Choice_balance_results.csv', sep='\t')

    avgac = dfac[dfac.Baseline.str.contains('_cons')]
    avgac.index  = avgac.Baseline.str.split('__').str[0]

    avgaa = dfaa[dfaa.Baseline.str.contains('_cons')]
    avgaa.index  = avgaa.Baseline.str.split('__').str[0]

    dfaa = clean_baseline_table(dfaa)
    dfac = clean_baseline_table(dfac)

    dfaa['SD'] = dfaa.drop(columns=['Fstat', 'pval']).std(1).round(6).apply(lambda x: f'{x:.6f}')
    dfac['SD'] = dfac.drop(columns=['Fstat', 'pval']).std(1).round(6).apply(lambda x: f'{x:.6f}')

    dfaa['pval_fmt'] = dfaa['pval'].copy().apply(round_pval)
    dfac['pval_fmt'] = dfac['pval'].copy().apply(round_pval)

    dfaa['Mean'] = avgaa['coeff']
    dfac['Mean'] = avgac['coeff']

    s = df.groupby('asgn_type')[dfaa.index].sum().T.astype(int).applymap(lambda x: f'{x:,}')
    dfaa['Num'] = s['AA']
    dfac['Num'] = s['Choice']

    dfaa['pval_adj'] = multitest.multipletests(dfaa['pval'], method='fdr_bh')[1]
    dfac['pval_adj'] = multitest.multipletests(dfac['pval'], method='fdr_bh')[1]

    dfaa['pval_adj_fmt'] = dfaa['pval_adj'].apply(round_pval)
    dfac['pval_adj_Fmt'] = dfac['pval_adj'].apply(round_pval)


    dfaa.to_csv(output_p+'AA_table.csv')
    dfac.to_csv(output_p+'AC_table.csv')


def paper2_run_balance_regressions_race_split():
    """
    Balance check on LA
    """   
    def clean_baseline_table(df):
        cols = ['Fstat', 'pval', 'se']
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
        df = df

        # Contains F-stat and P-val
        s1 = (df.loc[df.Nobs.notnull(), ['Baseline', 'Fstat', 'pval']]
                .set_index(['Baseline']))

        # Holds individual plan effects
        s2 = df.loc[df.Nobs.isnull(), ['Baseline', 'coeff', 'se']]

        s2['plan'] = s2.Baseline.str.split('.').str[0].str.rsplit('_', n=1).str[1]
        s2 = s2[~s2.plan.eq('cons')].copy()

        s2['Baseline'] = s2.Baseline.str.split('.').str[0].str.rsplit('_', n=1).str[0]

        s2 = (s2.pivot_table(index=['Baseline'], columns='plan', values='coeff',
                             aggfunc='first')
                .rename_axis(None, axis=1))

        res = pd.concat([s2, s1], axis=1, sort=False)

        return res


    ## Prep analytic table
    output_p, temp_p, log_p, code_p = gen_paths(FS().analysis_p
                                                +'paper2_run_balance_regressions_race_split/')

    df = pd.read_pickle(FS().derived_p+'paper2_augment_baseline_tables\output\LA_baseline.pkl')

    df = df[df.recip_race_g.isin([0,1])]
    df['plan_id_test'] = df.plan_id_asgn.where(df.asgn_type.eq('AA')).fillna(df.plan_id)

    df['is_black'] = df.recip_race_g
    df['is_female'] = df.recip_gender_g
    df['age_le5'] = df['recip_age'].le(5).astype(int)
    df['age_6to17'] = df['recip_age'].between(5, 18).astype(int)
    df['age_18to64'] = df['recip_age'].between(18, 64).astype(int)

    # Bring over baseline health 
    dfh = (pd.read_pickle(FS().la_pipe_analytic_path+'HCCS_HHS_2011.pkl')
             .set_index('recip_id'))

    df = df.merge(dfh, on='recip_id', how='left')
    df[dfh.columns] = df[dfh.columns].fillna(0)
    df = df.copy()

    # Create HCC categories.
    grps = [('asthma', ['hcc_161.2','hcc_161.1', 'hcc_160']),
            ('dep_bip_psych', ['hcc_88', 'hcc_87.1', 'hcc_87.2', 'hcc_90', 
                               'hcc_102', 'hcc_103']),
            ('diabetes', ['hcc_20', 'hcc_21']),
            ('drug_sud', ['hcc_81', 'hcc_82', 'hcc_83', 'hcc_84']),
            ('pregnancy', ['hcc_203', 'hcc_204', 'hcc_205', 'hcc_207',
                            'hcc_208', 'hcc_209', 'hcc_210', 'hcc_211', 'hcc_212']),
            ('cardiov_cond', ['hcc_131', 'hcc_139', 'hcc_138',
                              'hcc_142', 'hcc_137', 'hcc_135', 'hcc_132',
                              'hcc_130', 'hcc_129', 'hcc_128', 'hcc_127',
                              'hcc_126', 'hcc_125'])
           ]
    for cond, cols in grps:
        df[f'cond_{cond}'] = df[cols].eq(1).any(axis=1).astype(int)
        df['cond_any'] = df.filter(like='hcc_').eq(1).any(axis=1).astype(int)
        df['cond_n'] = df.filter(like='hcc_').sum(axis=1)  

    ## Loop by race
    grpr = df['is_black'].eq(1).map({True: 'blk', False: 'wht'})
    for label, dfs in df.groupby(grpr):
      
        # Gets some raw numbers
        ncols = ['age_le5', 'age_6to17', 'age_18to64', 'is_female',
                 'cond_asthma', 'cond_dep_bip_psych', 'cond_diabetes',
                 'cond_pregnancy', 'cond_cardiov_cond']

        res = dfs.groupby(['asgn_type', 'plan_id_test'])[ncols].sum().stack().unstack([0,1])
        res.to_csv(output_p+f'{label}_3_plan_Ns.csv')

        smeans = dfs.groupby('asgn_type')[ncols+['predicted_risk']].mean().T
        smeans.to_csv(output_p+f'{label}_2_sample_means.csv')

        (dfs.groupby(['asgn_type', 'plan_id_test']).size().unstack(-1)
            .to_csv(output_p+f'{label}_1_plan_enrollment.csv'))

        for idx, gp in dfs.groupby('asgn_type'):
            gp.to_csv(temp_p+f'{label}_{idx}.csv', sep=',')

            ws.stata_do(do_file=code_p+'test_balance.do',
                        params=[temp_p+f'{label}_{idx}.csv',
                                output_p+f'{label}_{idx}_balance_results.csv'],
                        log_path=log_p)

            # Delete temp file
            os.remove(temp_p+f'{label}_{idx}.csv')

        dfaa = pd.read_csv(output_p+f'{label}_AA_balance_results.csv', sep='\t')
        dfac = pd.read_csv(output_p+f'{label}_Choice_balance_results.csv', sep='\t')

        avgac = dfac[dfac.Baseline.str.contains('_cons')]
        avgac.index  = avgac.Baseline.str.split('__').str[0]

        avgaa = dfaa[dfaa.Baseline.str.contains('_cons')]
        avgaa.index  = avgaa.Baseline.str.split('__').str[0]

        dfaa = clean_baseline_table(dfaa)
        dfac = clean_baseline_table(dfac)

        dfaa['SD'] = dfaa.drop(columns=['Fstat', 'pval']).std(1).round(6).apply(lambda x: f'{x:.6f}')
        dfac['SD'] = dfac.drop(columns=['Fstat', 'pval']).std(1).round(6).apply(lambda x: f'{x:.6f}')

        dfaa['pval_fmt'] = dfaa['pval'].copy().apply(round_pval)
        dfac['pval_fmt'] = dfac['pval'].copy().apply(round_pval)

        dfaa['Mean'] = avgaa['coeff']
        dfac['Mean'] = avgac['coeff']

        s = dfs.groupby('asgn_type')[dfaa.index].sum().T.astype(int).applymap(lambda x: f'{x:,}')
        dfaa['Num'] = s['AA']
        dfac['Num'] = s['Choice']

        dfaa['pval_adj'] = multitest.multipletests(dfaa['pval'], method='fdr_bh')[1]
        dfac['pval_adj'] = multitest.multipletests(dfac['pval'], method='fdr_bh')[1]

        dfaa['pval_adj_fmt'] = dfaa['pval_adj'].apply(round_pval)
        dfac['pval_adj_Fmt'] = dfac['pval_adj'].apply(round_pval)


        dfaa.to_csv(output_p+f'{label}_5_AA_baseline.csv')
        dfac.to_csv(output_p+f'{label}_4_AC_baseline.csv')


def paper2_print_sample_overall_characteristics():
    df = pd.read_pickle(FS().derived_p+'paper2_analytic_tables_to_yearly/output/'
                        +'LA_analytic_table_yearly.pkl')
    df = df[df.recip_race_g.isin([0, 1])]
    dfc = df.copy()
    df = df[df.year.eq(2012)]

    age = df.recip_age.agg(['mean', 'std'])
    race = df.recip_race.value_counts(normalize=True).mul(100).round(1)

    print(f'N_recips: {df.recip_id.nunique()}')
    print(f'Age, mean [SD]: {age.loc["mean"]:,.1f} [{age.loc["std"]:,.1f}]')
    print(f'Female: {df.recip_gender.eq("F").mean()*100:,.1f}')
    print(f'Black: {race.loc["Black"]:,.1f}')
    print(f'White: {race.loc["White"]:,.1f}')
    print(f'Adult: {df.recip_age.gt(18).mean()*100:,.1f}')

    adults = df[df.recip_age.gt(18)].recip_id.unique()
    s = dfc.groupby(dfc.recip_id.isin(adults)).all_cost_wins50.sum()
    adult_spend = (s/s.sum()).mul(100).round(1).loc[True]
    print(f'Adult Spend Prop: {adult_spend}')


def paper2_run_balance_regressions_race_split_diff():
    output_p, temp_p, log_p, code_p = gen_paths(FS().analysis_p+'paper2_run_balance_regressions_race_split_diff/')

    df = pd.read_pickle(FS().derived_p+'paper2_augment_baseline_tables\output\LA_baseline.pkl')

    df = df[df.recip_race_g.isin([0, 1])]
    df['plan_id_test'] = df.plan_id_asgn.where(df.asgn_type.eq('AA')).fillna(df.plan_id)
    df['plan_id_test_race'] = df['plan_id_test']*10+df['recip_race_g']

    df['is_black'] = df.recip_race_g
    df['is_female'] = df.recip_gender_g
    df['age_le5'] = df['recip_age'].le(5).astype(int)
    df['age_6to17'] = df['recip_age'].between(5, 18).astype(int)
    df['age_18to64'] = df['recip_age'].between(18, 64).astype(int)

    # Bring over baseline health 
    dfh = (pd.read_pickle(FS().la_pipe_analytic_path+'HCCS_HHS_2011.pkl')
             .set_index('recip_id'))

    df = df.merge(dfh, on='recip_id', how='left')
    df[dfh.columns] = df[dfh.columns].fillna(0)
    df = df.copy()

    # Create HCC categories.
    grps = [('asthma', ['hcc_161.2','hcc_161.1', 'hcc_160']),
            ('dep_bip_psych', ['hcc_88', 'hcc_87.1', 'hcc_87.2', 'hcc_90', 
                               'hcc_102', 'hcc_103']),
            ('diabetes', ['hcc_20', 'hcc_21']),
            ('drug_sud', ['hcc_81', 'hcc_82', 'hcc_83', 'hcc_84']),
            ('pregnancy', ['hcc_203', 'hcc_204', 'hcc_205', 'hcc_207',
                            'hcc_208', 'hcc_209', 'hcc_210', 'hcc_211', 'hcc_212']),
            ('cardiov_cond', ['hcc_131', 'hcc_139', 'hcc_138',
                              'hcc_142', 'hcc_137', 'hcc_135', 'hcc_132',
                              'hcc_130', 'hcc_129', 'hcc_128', 'hcc_127',
                              'hcc_126', 'hcc_125'])
           ]
    for cond, cols in grps:
        df[f'cond_{cond}'] = df[cols].eq(1).any(axis=1).astype(int)
        df['cond_any'] = df.filter(like='hcc_').eq(1).any(axis=1).astype(int)
        df['cond_n'] = df.filter(like='hcc_').sum(axis=1)  


    for idx, gp in df.groupby('asgn_type'):
        gp.to_csv(temp_p+f'{idx}.csv', sep=',')

        ws.stata_do(do_file=code_p+'test_balance_diff.do',
                    params=[temp_p+f'{idx}.csv',
                            output_p+f'{idx}_balance_results.csv'],
                    log_path=log_p)

        #Delete temp file
        os.remove(temp_p+f'{idx}.csv')

    dfs = df.copy()

    paper_outcomes = ['age_le5', 'age_6to17', 'age_18to64', 'predicted_risk', 
                      'is_female', 'cond_asthma', 'cond_dep_bip_psych', 'cond_diabetes',
                      'cond_pregnancy', 'cond_cardiov_cond']


    for f, asgn_type in [('AA_balance_results.csv', 'AA'),
                         ('Choice_balance_results.csv', 'Choice')]:
        df = pd.read_csv(output_p+f, sep='\t')
        df['variable'] = df['variable'].str.split(':').str[-1]
        df['lb'] = pd.to_numeric(df['coeff'], errors='coerce') + 1.96*pd.to_numeric(df['se'], errors='coerce')
        df['ub'] = pd.to_numeric(df['coeff'], errors='coerce') - 1.96*pd.to_numeric(df['se'], errors='coerce')
        white_avg = dfs.loc[dfs.recip_race_g.eq(0) & dfs.asgn_type.eq(asgn_type), 
                            df.outcome.unique()].mean()

        for reg, gp in df.groupby('regression'):
            pval = gp.loc[gp.variable.eq('plan_diff'), ['outcome', 'pval']].set_index('outcome')
            keeps = ([f'{x}1.plan_id_test_race' for x in range(1, 6)]  
                     + ['51o.plan_id_test_race', '1.recip_race_g'])
            gp = gp[gp.variable.isin(keeps)]
            gp = gp.pivot(index='outcome', columns='variable',
                          values=['coeff', 'se', 'lb', 'ub']).apply(pd.to_numeric)
            gp['pval'] = pd.to_numeric(pval['pval'], errors='coerce')
            gp['pval_fmt'] = gp['pval'].apply(round_pval)

            sadj = pd.Series(multitest.multipletests(gp.reindex(paper_outcomes)['pval'], method='fdr_bh')[1],
                             index=paper_outcomes)
            gp['pval_adj'] = sadj
            gp['pval_adj_fmt'] = gp['pval_adj'].apply(round_pval)

            gp['white_avg'] = white_avg
            if f != 'AA_IV_results_5plans.csv':
                gp['range'] = gp.xs('coeff', axis=1).filter(like='plan_race').apply(np.ptp, axis=1)
            else:
                gp['range'] = gp.xs('coeff', axis=1).filter(like='plan_id#').apply(np.ptp, axis=1)
            gp.to_csv(output_p+f'{reg}_results_{asgn_type}.csv')


def paper2_run_main_regressions():
    """
    Run the main paper regressions
    """        
    output_p, temp_p, log_p, code_p = gen_paths(FS().analysis_p
                                                +'paper2_run_main_regressions/')

    df = pd.read_pickle(FS().derived_p+'paper2_analytic_tables_to_yearly/output/'
                        +'LA_analytic_table_yearly.pkl')
    dfch = df[df.asgn_type.eq('Choice') & df.recip_race_g.isin([0,1])]
    dfch.to_csv(temp_p+'Choice.csv', chunksize=10**5, index=False)

    df = df[df.asgn_type.eq('AA') & df.recip_race_g.isin([0,1])]
    df.to_csv(temp_p+'AA.csv', chunksize=10**5, index=False)

    ws.stata_do(do_file=code_p+'paper2_main_regs_RF.do',
                params=[temp_p+'AA.csv',
                        output_p+'AA_RF_results.csv'],
                log_path=log_p)

    ws.stata_do(do_file=code_p+'paper2_main_regs_RF_5plans.do',
                params=[temp_p+'AA.csv',
                        output_p+'AA_RF_results_5plans.csv'],
                log_path=log_p)

    ws.stata_do(do_file=code_p+'paper2_main_regs_IV.do',
                params=[temp_p+'AA.csv',
                        output_p+'AA_IV_results_5plans.csv'],
                log_path=log_p)

    ws.stata_do(do_file=code_p+'paper2_AC_regs_5plans.do',
                params=[temp_p+'Choice.csv',
                        output_p+'AC_results_5plans.csv'],
                log_path=log_p)

    # Delete temp file
    for file in ['AA', 'Choice']:
        os.remove(temp_p+f'{file}.csv')


def paper2_run_main_regressions_nbreg():
    """
    Run the main paper regressions
    """        
    output_p, temp_p, log_p, code_p = gen_paths(FS().analysis_p
                                                +'paper2_run_main_regressions_nbreg/')

    df = pd.read_pickle(FS().derived_p+'paper2_analytic_tables_to_yearly/output/'
                        +'LA_analytic_table_yearly.pkl')
    dfch = df[df.asgn_type.eq('Choice') & df.recip_race_g.isin([0,1])]
    dfch.to_csv(temp_p+'Choice.csv', chunksize=10**5, index=False)

    df = df[df.asgn_type.eq('AA') & df.recip_race_g.isin([0,1])]
    df.to_csv(temp_p+'AA.csv', chunksize=10**5, index=False)

    ws.stata_do(do_file=code_p+'paper2_main_regs_RF_5plans.do',
                params=[temp_p+'AA.csv',
                        output_p+'AA_RF_results_5plans.csv'],
                log_path=log_p)

    ws.stata_do(do_file=code_p+'paper2_AC_regs_5plans.do',
                params=[temp_p+'Choice.csv',
                        output_p+'AC_results_5plans.csv'],
                log_path=log_p)

    # Delete temp file
    for file in ['AA', 'Choice']:
        os.remove(temp_p+f'{file}.csv')


def paper2_run_main_regressions_interacted():
    """
    Run the main paper regressions
    """        
    output_p, temp_p, log_p, code_p = gen_paths(FS().analysis_p
                                                +'paper2_run_main_regressions_interacted/')

    df = pd.read_pickle(FS().derived_p+'paper2_analytic_tables_to_yearly/output/'
                        +'LA_analytic_table_yearly.pkl')
    dfch = df[df.asgn_type.eq('Choice') & df.recip_race_g.isin([0,1])]
    dfch.to_csv(temp_p+'Choice.csv', chunksize=10**5, index=False)

    ws.stata_do(do_file=code_p+'paper2_AC_regs_5plans_int.do',
                params=[temp_p+'Choice.csv',
                        output_p+'AC_results_5plans.csv'],
                log_path=log_p)

    # Delete temp file
    for file in ['Choice']:
        os.remove(temp_p+f'{file}.csv')


def paper2_clean_main_reg_results_interacted():

    output_p = FS().analysis_p+'paper2_clean_main_regression_results_interacted/output/'


    paper_outcomes = ['all_cost_wins50', 'pc_numdays', 'pharm_numclms', 'ed_avoid_numdays']

    # Get average among White enrollees in sample 
    dfs = pd.read_pickle(FS().derived_p+'paper2_analytic_tables_to_yearly/output/'
                        +'LA_analytic_table_yearly.pkl')
    dfs = dfs[dfs.recip_race_g.isin([0, 1])]

    for label, f, asgn_type in [('_AC_5plan', 'AC_results_5plans.csv', 'Choice')]:
        df = pd.read_csv(FS().analysis_p+'paper2_run_main_regressions_interacted/output/'+f, sep='\t')
        df['variable'] = df['variable'].str.split(':').str[-1]
        df['lb'] = pd.to_numeric(df['coeff'], errors='coerce') + 1.96*pd.to_numeric(df['se'], errors='coerce')
        df['ub'] = pd.to_numeric(df['coeff'], errors='coerce') - 1.96*pd.to_numeric(df['se'], errors='coerce')
        white_avg = dfs.loc[dfs.recip_race_g.eq(0) & dfs.asgn_type.eq(asgn_type), 
                            df.outcome.unique()].mean()

        for reg, gp in df.groupby('regression'):
            pval = gp.loc[gp.variable.eq('plan)diff'), ['outcome', 'pval']].set_index('outcome')
            keeps = ([f'{x}1.plan_race_asgn' for x in range(1, 6)] 
                     + [f'{x}1.plan_race' for x in range(1, 6)] 
                     + ['51o.plan_race_asgn', '1.recip_race_g', '11o.plan_race'])
            gp = gp[gp.variable.isin(keeps)]
            gp = gp.pivot(index='outcome', columns='variable',
                          values=['coeff', 'se', 'lb', 'ub']).apply(pd.to_numeric)
            gp['pval'] = pd.to_numeric(pval['pval'], errors='coerce')
            gp['pval_fmt'] = gp['pval'].apply(round_pval)

            sadj = pd.Series(multitest.multipletests(gp.reindex(paper_outcomes)['pval'], method='fdr_bh')[1],
                             index=paper_outcomes)
            gp['pval_adj'] = sadj
            gp['pval_adj_fmt'] = gp['pval_adj'].apply(round_pval)

            gp['white_avg'] = white_avg
            if f != 'AA_IV_results_5plans.csv':
                gp['range'] = gp.xs('coeff', axis=1).filter(like='plan_race').apply(np.ptp, axis=1)
            else:
                gp['range'] = gp.xs('coeff', axis=1).filter(like='plan_id#').apply(np.ptp, axis=1)
            gp.to_csv(output_p+f'{reg}_results{label}.csv')


def paper2_run_main_regressions_hispanic():
    """
    Run the main paper regressions
    """        
    output_p, temp_p, log_p, code_p = gen_paths(FS().analysis_p
                                                +'paper2_run_main_regressions_hispanic/')

    df = pd.read_pickle(FS().derived_p+'paper2_analytic_tables_to_yearly/output/'
                        +'LA_analytic_table_yearly.pkl')
    # Keep White & Hispanic
    dfch = df[df.asgn_type.eq('Choice') & df.recip_race_g.isin([0, 2])]
    dfch.to_csv(temp_p+'Choice.csv', chunksize=10**5, index=False)

    df = df[df.asgn_type.eq('AA') & df.recip_race_g.isin([0, 2])]
    df.to_csv(temp_p+'AA.csv', chunksize=10**5, index=False)

    # ws.stata_do(do_file=code_p+'paper2_main_regs_RF.do',
    #             params=[temp_p+'AA.csv',
    #                     output_p+'AA_RF_results.csv'],
    #             log_path=log_p)

    ws.stata_do(do_file=code_p+'paper2_main_regs_RF_5plans.do',
                params=[temp_p+'AA.csv',
                        output_p+'AA_RF_results_5plans.csv'],
                log_path=log_p)

    # ws.stata_do(do_file=code_p+'paper2_main_regs_IV.do',
    #             params=[temp_p+'AA.csv',
    #                     output_p+'AA_IV_results_5plans.csv'],
    #             log_path=log_p)

    ws.stata_do(do_file=code_p+'paper2_AC_regs_5plans.do',
                params=[temp_p+'Choice.csv',
                        output_p+'AC_results_5plans.csv'],
                log_path=log_p)

    # Delete temp file
    for file in ['AA', 'Choice']:
        os.remove(temp_p+f'{file}.csv')


def paper2_run_main_regressions_black_hispanic_pooled():
    """
    Run the main paper regressions
    """        
    output_p, temp_p, log_p, code_p = gen_paths(FS().analysis_p
                                                +'paper2_run_main_regressions_black_hispanic_pooled/')

    df = pd.read_pickle(FS().derived_p+'paper2_analytic_tables_to_yearly/output/'
                        +'LA_analytic_table_yearly.pkl')
    # Keep White & Hispanic
    dfch = df[df.asgn_type.eq('Choice') & df.recip_race_g.isin([0, 1, 2])]
    dfch['recip_race_g'] = dfch['recip_race_g'].map({0: 0, 1: 1, 2:1})
    dfch.to_csv(temp_p+'Choice.csv', chunksize=10**5, index=False)

    df = df[df.asgn_type.eq('AA') & df.recip_race_g.isin([0, 1, 2])]
    df['recip_race_g'] = df['recip_race_g'].map({0: 0, 1: 1, 2:1})
    df.to_csv(temp_p+'AA.csv', chunksize=10**5, index=False)

    ws.stata_do(do_file=code_p+'paper2_main_regs_RF_5plans.do',
                params=[temp_p+'AA.csv',
                        output_p+'AA_RF_results_5plans.csv'],
                log_path=log_p)

    ws.stata_do(do_file=code_p+'paper2_AC_regs_5plans.do',
                params=[temp_p+'Choice.csv',
                        output_p+'AC_results_5plans.csv'],
                log_path=log_p)

    # Delete temp file
    for file in ['AA', 'Choice']:
        os.remove(temp_p+f'{file}.csv')


def paper2_run_main_regressions_reweighted():
    """
    Run the main paper regressions re-weighting the AAs to look like the choosers. Because the 
    main outcome of interest is the disparity we re-weight the Black AAs to look like the 
    Black Choosers and the White AAs to look like the White choosers. 

    Because the weighting only impacts the AAs there is no need to re-run the Choosers regressions

    Currenly re-weights on # of HCCS/Age buckets/Sex/Race

    """        
    output_p, temp_p, log_p, code_p = gen_paths(FS().analysis_p
                                                +'paper2_run_main_regressions_reweighted/')

    df = pd.read_pickle(FS().derived_p+'paper2_analytic_tables_to_yearly/output/'
                        +'LA_analytic_table_yearly.pkl')

    # Keep only Black and White Enrollees
    df = df[df.recip_race_g.isin([0,1])]

    # Create a category of # of HCCS [0, 1, 2, 3+]
    hccs = [x for x in df.columns if 'hcc_' in x and not x.startswith('b_')]
    df['num_hccs'] = df[hccs].sum(1)
    df['hcc_g'] = pd.cut(df['num_hccs'], [-np.inf,0,1,2,np.inf], labels=['0', '1','2','3+'])
    df['recip_age_g_weight'] = pd.cut(df['recip_age_g'], [-np.inf, 10, 20, 40, np.inf])

    df = add_weights(df, ['hcc_g', 'recip_race_g', 'recip_age_g_weight', 'recip_gender'], 'weight')

    # Sanity check to make sure it worked and weights aren't extreme
    print(df.groupby('asgn_type').weight.describe().to_string())
    df = df.drop(columns=['hcc_g', 'num_hccs'])

    df = df[df.asgn_type.eq('AA')]
    df.to_csv(temp_p+'AA.csv', chunksize=10**5, index=False)

    ws.stata_do(do_file=code_p+'paper2_main_regs_RF.do',
                params=[temp_p+'AA.csv',
                        output_p+'AA_RF_results_reweighted.csv'],
                log_path=log_p)

    ws.stata_do(do_file=code_p+'paper2_main_regs_RF_5plans.do',
                params=[temp_p+'AA.csv',
                        output_p+'AA_RF_results_5plans_reweighted.csv'],
                log_path=log_p)

    ws.stata_do(do_file=code_p+'paper2_main_regs_IV.do',
                params=[temp_p+'AA.csv',
                        output_p+'AA_IV_results_5plans_reweighted.csv'],
                log_path=log_p)

    # Delete temp file
    for file in ['AA']:
        os.remove(temp_p+f'{file}.csv')


def paper2_run_main_regressions_byplan_BWrace_ACs_HCCs():
    """
    Mains RF regressions to recapture demeaned means for each plan and race group
    """
    output_p, temp_p, log_p, code_p = gen_paths(FS().analysis_p
                                                +'paper2_run_main_regressions_byplan_BWrace_ACs_HCCs/')

    df = pd.read_pickle(FS().derived_p+'paper2_analytic_tables_to_yearly/output/'
                        +'LA_analytic_table_yearly.pkl')
    #Include only black and white
    df = df[df.recip_race_g.isin([0,1])].copy()

    cols = (['recip_id', 'plan_id', 'plan_id_asgn', 'asgn_type']
            + ['pc_numdays', 'pharm_numclms', 'ed_avoid_numdays', 'ednoadmit_numdays', 'all_cost_wins50']
            + ['recip_age_g', 'recip_gender_g', 'recip_race_g', 'plan_race']
            + [x for x in df.columns if 'hcc_' in x]
            + ['uor'])
    df = df[cols]

    # Just choosers
    dfch = df[df.asgn_type.eq('Choice')]
    dfch.to_csv(temp_p+'Allrace_Choice.csv', chunksize=10**5, index=False)

    ws.stata_do(do_file=code_p+'paper2_AC_regs_5plans.do',
                params=[temp_p+'Allrace_Choice.csv',
                        output_p+f'Allrace_Choice_results_5plans.csv',
                        " ".join(['pc_numdays', 'pharm_numclms', 'ed_avoid_numdays', 'ednoadmit_numdays', 'all_cost_wins50'])],
                log_path=log_p)

    ws.stata_do(do_file=code_p+'paper2_AC_interacted_regs_5plans.do',
                params=[temp_p+'Allrace_Choice.csv',
                        output_p+f'Allrace_Choice_interacted_results_5plans.csv',
                        " ".join(['pc_numdays', 'pharm_numclms', 'ed_avoid_numdays', 'ednoadmit_numdays', 'all_cost_wins50'])],
                log_path=log_p)

    # Delete temp file
    for file in ['Allrace_Choice']:
        os.remove(temp_p+f'{file}.csv')


# Appx Tables 9 and 10 (By plan by race regressions, also necessary for the MAD tables)
def paper2_run_main_regressions_byplan_byrace_HCCs():
    """
    Mains RF regressions to recapture demeaned means for each plan and race group
    """
    output_p, temp_p, log_p, code_p = gen_paths(FS().analysis_p
                                                +'paper2_run_main_regressions_byplan_byrace_HCCs/')

    df = pd.read_pickle(FS().derived_p+'paper2_analytic_tables_to_yearly/output/'
                        +'LA_analytic_table_yearly.pkl')
    df = df[df.recip_race_g.isin([0,1])].copy()

    cols = (['recip_id', 'plan_id', 'plan_id_asgn', 'asgn_type']
            + ['pc_numdays', 'pharm_numclms', 'ed_avoid_numdays', 'ednoadmit_numdays', 'all_cost_wins50']
            + ['recip_age_g', 'recip_gender_g', 'recip_race_g', 'plan_race']
            + [x for x in df.columns if 'hcc_' in x]
            + ['uor'])
    df = df[cols]

    grpr = df['recip_race_g'].eq(1).map({True: 'blk', False: 'wht'})
    for label, gp in df.groupby(grpr):

        dfch = gp[gp.asgn_type.eq('Choice')]
        dfch.to_csv(temp_p+'Choice.csv', chunksize=10**5, index=False)

        dfaa = gp[gp.asgn_type.eq('AA')]
        dfaa.to_csv(temp_p+'AA.csv', chunksize=10**5, index=False)

        ws.stata_do(do_file=code_p+'paper2_main_regs_RF_5plans.do',
                    params=[temp_p+'AA.csv',
                            output_p+f'{label}_AA_RF_results_5plans.csv',
                            " ".join(['pc_numdays', 'pharm_numclms', 'ed_avoid_numdays', 'ednoadmit_numdays', 'all_cost_wins50'])],
                    log_path=log_p)

        ws.stata_do(do_file=code_p+'paper2_AC_regs_5plans.do',
                    params=[temp_p+'Choice.csv',
                            output_p+f'{label}_AC_results_5plans.csv',
                            " ".join(['pc_numdays', 'pharm_numclms', 'ed_avoid_numdays', 'ednoadmit_numdays', 'all_cost_wins50'])],
                    log_path=log_p)

        # Delete temp file
        for file in ['AA', 'Choice']:
            os.remove(temp_p+f'{file}.csv')


# Appx Tables 9 and 10 (By plan by race regressions, also necessary for the MAD tables)
def paper2_clean_main_reg_results_byplan_byrace_HCCs(iteration=False,
  import_p='paper2_run_main_regressions_byplan_byrace_HCCs/output/', 
  output_p='paper2_clean_main_regression_results_byplan_byrace_HCCs/output/'):
    """
    Import and clean Stata output for by plan by race regressions - coefs, not demeaned means
    """
    output_p = FS().analysis_p+output_p

    paper_outcomes = ['all_cost_wins50', 'pc_numdays', 'pharm_numclms', 'ed_avoid_numdays']
    text = 'Failed'

    for i, r in enumerate(['wht', 'blk']):

        for label, f, asgn_type in [('_5plan', f'{r}_AA_RF_results_5plans.csv', 'AA'), 
                                    ('_AC_5plan', f'{r}_AC_results_5plans.csv', 'Choice')]:
            if iteration is not False:
                f = f.replace('.csv', f'_{iteration}.csv')
            try:
                df = pd.read_csv(FS().analysis_p+import_p+f, sep='\t')

                for reg, gp in df.groupby('regression'):
                    pval = gp.loc[gp.variable.eq('plan)diff'), ['outcome', 'pval']].set_index('outcome')
                    keeps = ([f'{x}.plan_id' for x in range(1, 6)] 
                           + [f'{x}.plan_id_asgn' for x in range(1, 6)] 
                           + ['1b.plan_id', '1b.plan_id_asgn'])
                    gp = gp[gp.variable.isin(keeps)]
                    gp = gp.pivot(index='outcome', columns='variable',
                                  values=['coeff', 'se', 'lb', 'ub']).apply(pd.to_numeric)
                    gp['pval'] = pd.to_numeric(pval['pval'], errors='coerce')
                    gp['pval_fmt'] = gp['pval'].apply(round_pval)

                    sadj = pd.Series(multitest.multipletests(gp.reindex(paper_outcomes)['pval'], method='fdr_bh')[1],
                                     index=paper_outcomes)
                    gp['pval_adj'] = sadj
                    gp['pval_adj_fmt'] = gp['pval_adj'].apply(round_pval)

                    # Get means
                    if iteration is False:
                        dfs = pd.read_pickle(FS().derived_p+'paper2_analytic_tables_to_yearly/output/'
                                            +'LA_analytic_table_yearly.pkl')
                        dfs = dfs[dfs.recip_race_g.eq(i)]
                        gp['avg'] = dfs.loc[dfs.asgn_type.eq(asgn_type), df.outcome.unique()].mean()

                    gp['range'] = gp.xs('coeff', axis=1).filter(like='plan').apply(np.ptp, axis=1)
                    gp['std_dev'] = gp.xs('coeff', axis=1).filter(like='plan').std(axis=1)
                    if iteration is not False:
                        file = f'{r}_{reg}_results{label}_{iteration}.csv'
                        text = f'Iteration {iteration} done'
                    else:
                        file = f'{r}_{reg}_results{label}.csv'
                        text = 'Done'
                    gp.loc[paper_outcomes].to_csv(output_p+file)  
            except FileNotFoundError as e:
                pass
        if i == 1:
            print(text)   


def paper2_clean_main_reg_results_byplan_BWrace_ACs_HCCs(
  import_p='paper2_run_main_regressions_byplan_BWrace_ACs_HCCs/output/', 
  output_p='paper2_clean_main_regression_results_byplan_BWrace_ACs_HCCs/output/'):
    """
    Import and clean Stata output for by plan by race regressions - coefs, not demeaned means
    """
    output_p = FS().analysis_p+output_p

    paper_outcomes = ['all_cost_wins50', 'pc_numdays', 'pharm_numclms', 'ed_avoid_numdays']
    text = 'Failed'

    for i, r in enumerate(['Allrace_Choice', 'Allrace_Choice_interacted']):
        label = '_AC_5plan', 
        f = f'{r}_results_5plans.csv'
 
        df = pd.read_csv(FS().analysis_p+import_p+f, sep='\t')

        for reg, gp in df.groupby('regression'):
            pval = gp.loc[gp.variable.eq('plan)diff'), ['outcome', 'pval']].set_index('outcome')
            keeps = ([f'{x}.plan_id' for x in range(1, 6)] 
                   + [f'{x}.plan_id_asgn' for x in range(1, 6)] 
                   + ['1b.plan_id', '1b.plan_id_asgn'])
            gp = gp[gp.variable.isin(keeps)]
            gp = gp.pivot(index='outcome', columns='variable',
                          values=['coeff', 'se', 'lb', 'ub']).apply(pd.to_numeric)
            gp['pval'] = pd.to_numeric(pval['pval'], errors='coerce')
            gp['pval_fmt'] = gp['pval'].apply(round_pval)

            sadj = pd.Series(multitest.multipletests(gp.reindex(paper_outcomes)['pval'], method='fdr_bh')[1],
                             index=paper_outcomes)
            gp['pval_adj'] = sadj
            gp['pval_adj_fmt'] = gp['pval_adj'].apply(round_pval)

            # Get means
            dfs = pd.read_pickle(FS().derived_p+'paper2_analytic_tables_to_yearly/output/'
                                +'LA_analytic_table_yearly.pkl')
            # Full file
            if r == 'Allrace':
                dfs = dfs
            # Subset to choosers
            if r in ['Allrace_Choice', 'Allrace_Choice_interacted']:
                dfs = dfs[dfs.asgn_type.eq('Choice')]

            print(i, dfs.loc[:, df.outcome.unique()].mean())
            gp['avg'] = dfs.loc[:, df.outcome.unique()].mean()

            gp['range'] = gp.xs('coeff', axis=1).filter(like='plan').apply(np.ptp, axis=1)
            gp['std_dev'] = gp.xs('coeff', axis=1).filter(like='plan').std(axis=1)
            
            file = f'{r}_{reg}_results{label}.csv'
            gp.loc[paper_outcomes].to_csv(output_p+file)  


def paper2_clean_main_reg_results_nbreg():
    """
    
    """
    output_p = FS().analysis_p+'paper2_clean_main_regression_results_nbreg/output/'


    paper_outcomes = ['all_cost_wins50', 'pc_numdays', 'pharm_numclms', 'ed_avoid_numdays']

    # Get average among White enrollees in sample 
    dfs = pd.read_pickle(FS().derived_p+'paper2_analytic_tables_to_yearly/output/'
                        +'LA_analytic_table_yearly.pkl')
    dfs = dfs[dfs.recip_race_g.isin([0, 1])]

    for label, f, asgn_type in [('_5plan', 'AA_RF_results_5plans.csv', 'AA'),
                                ('_AC_5plan', 'AC_results_5plans.csv', 'Choice')]:
        df = pd.read_csv(FS().analysis_p+'paper2_run_main_regressions_nbreg/output/'+f, sep='\t')
        df['variable'] = df['variable'].str.split(':').str[-1]
        df['lb'] = pd.to_numeric(df['coeff'], errors='coerce') + 1.96*pd.to_numeric(df['se'], errors='coerce')
        df['ub'] = pd.to_numeric(df['coeff'], errors='coerce') - 1.96*pd.to_numeric(df['se'], errors='coerce')
        white_avg = dfs.loc[dfs.recip_race_g.eq(0) & dfs.asgn_type.eq(asgn_type), 
                            df.outcome.unique()].mean()

        for reg, gp in df.groupby('regression'):
            pval = gp.loc[gp.variable.eq('plan)diff'), ['outcome', 'pval']].set_index('outcome')
            if f != 'AA_IV_results_5plans.csv':
                keeps = ([f'{x}1.plan_race_asgn' for x in range(1, 6)] 
                         + [f'{x}1.plan_race' for x in range(1, 6)] 
                         + ['51o.plan_race_asgn', '1.recip_race_g'])
            else:
                keeps = ['1b.plan_id#1.recip_race_g', '2.plan_id#1.recip_race_g',
                         '3.plan_id#1.recip_race_g', '4.plan_id#1.recip_race_g',
                         '5.plan_id#1.recip_race_g']
            gp = gp[gp.variable.isin(keeps)]
            gp = gp.pivot(index='outcome', columns='variable',
                          values=['coeff', 'se', 'lb', 'ub']).apply(pd.to_numeric)
            gp['pval'] = pd.to_numeric(pval['pval'], errors='coerce')
            gp['pval_fmt'] = gp['pval'].apply(round_pval)

            sadj = pd.Series(multitest.multipletests(gp.reindex(paper_outcomes)['pval'], method='fdr_bh')[1],
                             index=paper_outcomes)
            gp['pval_adj'] = sadj
            gp['pval_adj_fmt'] = gp['pval_adj'].apply(round_pval)

            gp['white_avg'] = white_avg
            if f != 'AA_IV_results_5plans.csv':
                gp['range'] = gp.xs('coeff', axis=1).filter(like='plan_race').apply(np.ptp, axis=1)
            else:
                gp['range'] = gp.xs('coeff', axis=1).filter(like='plan_id#').apply(np.ptp, axis=1)
            gp.to_csv(output_p+f'{reg}_results{label}.csv')


def paper2_clean_main_reg_results():
    """
    
    """
    output_p = FS().analysis_p+'paper2_clean_main_regression_results/output/'


    paper_outcomes = ['all_cost_wins50', 'pc_numdays', 'pharm_numclms', 'asthma_numclms',
                      'diabetes_numclms', 'statins_numclms', 'antihypertensive_numclms', 'ed_avoid_numdays',
                      'amr', 'wcv_3_ch', 'wcv_12_ch', 'wcv_18_ch', 'aap_ad', 'chl_all', 'add', 'ha1c']
    #paper_outcomes = ['all_cost_wins50', 'ed_avoid_numdays', 'pc_numdays', 'pharm_numclms']

    # Get average among White enrollees in sample 
    dfs = pd.read_pickle(FS().derived_p+'paper2_analytic_tables_to_yearly/output/'
                        +'LA_analytic_table_yearly.pkl')
    dfs = dfs[dfs.recip_race_g.isin([0, 1])]

    for label, f, asgn_type in [('', 'AA_RF_results.csv', 'AA'), 
                                ('_5plan', 'AA_RF_results_5plans.csv', 'AA'),
                                ('_AC_5plan', 'AC_results_5plans.csv', 'Choice'),
                                ('_AA_5plan_IV', 'AA_IV_results_5plans.csv', 'AA')]:
        df = pd.read_csv(FS().analysis_p+'paper2_run_main_regressions/output/'+f, sep='\t')
        white_avg = dfs.loc[dfs.recip_race_g.eq(0) & dfs.asgn_type.eq(asgn_type), 
                            df.outcome.unique()].mean()

        for reg, gp in df.groupby('regression'):
            pval = gp.loc[gp.variable.eq('plan)diff'), ['outcome', 'pval']].set_index('outcome')
            if f != 'AA_IV_results_5plans.csv':
                keeps = ([f'{x}1.plan_race_asgn' for x in range(1, 6)] 
                         + [f'{x}1.plan_race' for x in range(1, 6)] 
                         + ['51o.plan_race_asgn', '1.recip_race_g'])
            else:
                keeps = ['1b.plan_id#1.recip_race_g', '2.plan_id#1.recip_race_g',
                         '3.plan_id#1.recip_race_g', '4.plan_id#1.recip_race_g',
                         '5.plan_id#1.recip_race_g']
            gp = gp[gp.variable.isin(keeps)]
            gp = gp.pivot(index='outcome', columns='variable',
                          values=['coeff', 'se', 'lb', 'ub']).apply(pd.to_numeric)
            gp['pval'] = pd.to_numeric(pval['pval'], errors='coerce')
            gp['pval_fmt'] = gp['pval'].apply(round_pval)

            sadj = pd.Series(multitest.multipletests(gp.reindex(paper_outcomes)['pval'], method='fdr_bh')[1],
                             index=paper_outcomes)
            gp['pval_adj'] = sadj
            gp['pval_adj_fmt'] = gp['pval_adj'].apply(round_pval)

            gp['white_avg'] = white_avg
            if f != 'AA_IV_results_5plans.csv':
                gp['range'] = gp.xs('coeff', axis=1).filter(like='plan_race').apply(np.ptp, axis=1)
            else:
                gp['range'] = gp.xs('coeff', axis=1).filter(like='plan_id#').apply(np.ptp, axis=1)
            gp.to_csv(output_p+f'{reg}_results{label}.csv')


def paper2_clean_main_reg_results_hispanic():
    """
    
    """
    output_p = FS().analysis_p+'paper2_clean_main_regression_results_hispanic/output/'

    paper_outcomes = ['all_cost_wins50', 'pc_numdays', 'pharm_numclms', 'asthma_numclms',
                      'diabetes_numclms', 'statins_numclms', 'antihypertensive_numclms', 'ed_avoid_numdays',
                      'amr', 'wcv_3_ch', 'wcv_12_ch', 'wcv_18_ch', 'aap_ad', 'chl_all', 'add']
    #paper_outcomes = ['all_cost_wins50', 'ed_avoid_numdays', 'pc_numdays', 'pharm_numclms']

    # Get average among White enrollees in sample 
    dfs = pd.read_pickle(FS().derived_p+'paper2_analytic_tables_to_yearly/output/'
                        +'LA_analytic_table_yearly.pkl')
    dfs = dfs[dfs.recip_race_g.isin([0, 2])]

    for label, f, asgn_type in [('_5plan', 'AA_RF_results_5plans.csv', 'AA'),
                                ('_AC_5plan', 'AC_results_5plans.csv', 'Choice')]:
        df = pd.read_csv(FS().analysis_p+'paper2_run_main_regressions_hispanic/output/'+f, sep='\t')
        white_avg = dfs.loc[dfs.recip_race_g.eq(0) & dfs.asgn_type.eq(asgn_type), 
                            df.outcome.unique()].mean()

        for reg, gp in df.groupby('regression'):
            pval = gp.loc[gp.variable.eq('plan)diff'), ['outcome', 'pval']].set_index('outcome')
            if f != 'AA_IV_results_5plans.csv':
                keeps = ([f'{x}2.plan_race_asgn' for x in range(1, 6)] 
                         + [f'{x}2.plan_race' for x in range(1, 6)] 
                         + ['52o.plan_race_asgn', '1.recip_race_g'])
            else:
                keeps = ['1b.plan_id#2.recip_race_g', '2.plan_id#2.recip_race_g',
                         '3.plan_id#2.recip_race_g', '4.plan_id#2.recip_race_g',
                         '5.plan_id#2.recip_race_g']
            gp = gp[gp.variable.isin(keeps)]
            gp = gp.pivot(index='outcome', columns='variable',
                          values=['coeff', 'se', 'lb', 'ub']).apply(pd.to_numeric)
            gp['pval'] = pd.to_numeric(pval['pval'], errors='coerce')
            gp['pval_fmt'] = gp['pval'].apply(round_pval)

            sadj = pd.Series(multitest.multipletests(gp.reindex(paper_outcomes)['pval'], method='fdr_bh')[1],
                             index=paper_outcomes)
            gp['pval_adj'] = sadj
            gp['pval_adj_fmt'] = gp['pval_adj'].apply(round_pval)

            gp['white_avg'] = white_avg
            if f != 'AA_IV_results_5plans.csv':
                gp['range'] = gp.xs('coeff', axis=1).filter(like='plan_race').apply(np.ptp, axis=1)
            else:
                gp['range'] = gp.xs('coeff', axis=1).filter(like='plan_id#').apply(np.ptp, axis=1)
            gp.to_csv(output_p+f'{reg}_results{label}.csv')


def paper2_clean_main_reg_results_black_hispanic_pooled():
    """
    
    """
    output_p = FS().analysis_p+'paper2_clean_main_regression_results_black_hispanic_pooled/output/'

    paper_outcomes = ['all_cost_wins50', 'ed_avoid_numdays', 'pc_numdays', 'pharm_numclms']

    # Get average among White enrollees in sample 
    dfs = pd.read_pickle(FS().derived_p+'paper2_analytic_tables_to_yearly/output/'
                        +'LA_analytic_table_yearly.pkl')
    dfs = dfs[dfs.recip_race_g.isin([0, 1, 2])]
    dfs['recip_race_g'] = dfs['recip_race_g'].map({0: 0, 1: 1, 2:1})

    for label, f, asgn_type in [('_5plan', 'AA_RF_results_5plans.csv', 'AA'),
                                ('_AC_5plan', 'AC_results_5plans.csv', 'Choice')]:
        df = pd.read_csv(FS().analysis_p+'paper2_run_main_regressions_black_hispanic_pooled/output/'+f, sep='\t')
        white_avg = dfs.loc[dfs.recip_race_g.eq(0) & dfs.asgn_type.eq(asgn_type), 
                            df.outcome.unique()].mean()

        for reg, gp in df.groupby('regression'):
            pval = gp.loc[gp.variable.eq('plan)diff'), ['outcome', 'pval']].set_index('outcome')
            if f != 'AA_IV_results_5plans.csv':
                keeps = ([f'{x}2.plan_race_asgn' for x in range(1, 6)] 
                         + [f'{x}2.plan_race' for x in range(1, 6)] 
                         + ['52o.plan_race_asgn', '1.recip_race_g'])
            else:
                keeps = ['1b.plan_id#2.recip_race_g', '2.plan_id#2.recip_race_g',
                         '3.plan_id#2.recip_race_g', '4.plan_id#2.recip_race_g',
                         '5.plan_id#2.recip_race_g']
            gp = gp[gp.variable.isin(keeps)]
            gp = gp.pivot(index='outcome', columns='variable',
                          values=['coeff', 'se', 'lb', 'ub']).apply(pd.to_numeric)
            gp['pval'] = pd.to_numeric(pval['pval'], errors='coerce')
            gp['pval_fmt'] = gp['pval'].apply(round_pval)

            sadj = pd.Series(multitest.multipletests(gp.reindex(paper_outcomes)['pval'], method='fdr_bh')[1],
                             index=paper_outcomes)
            gp['pval_adj'] = sadj
            gp['pval_adj_fmt'] = gp['pval_adj'].apply(round_pval)

            gp['white_avg'] = white_avg
            if f != 'AA_IV_results_5plans.csv':
                gp['range'] = gp.xs('coeff', axis=1).filter(like='plan_race').apply(np.ptp, axis=1)
            else:
                gp['range'] = gp.xs('coeff', axis=1).filter(like='plan_id#').apply(np.ptp, axis=1)
            gp.to_csv(output_p+f'{reg}_results{label}.csv')


def paper2_clean_main_reg_results_reweighted():
    """
    Clean up the regression results table for the re-weighted analysis
    """
    output_p = FS().analysis_p+'paper2_clean_main_regression_results_reweighted/output/'

    paper_outcomes = ['adhd_numclms', 'all_cost_wins50',
                      'asthma_numclms', 'capch_all', 'ed_avoid_numdays',
                      'pc_numdays', 'pharm_numclms']

    # Get average among White enrollees in sample. 
    # Need to calculate a weighted average here. 
    dfs = pd.read_pickle(FS().derived_p+'paper2_analytic_tables_to_yearly/output/'
                         +'LA_analytic_table_yearly.pkl')
    dfs = dfs[dfs.recip_race_g.isin([0, 1])]

    # Should really do this in one place. Make sure this is the exact same weighting used 
    # in `paper2_run_main_regressions_reweighted`
    hccs = [x for x in dfs.columns if 'hcc_' in x and not x.startswith('b_')]
    dfs['num_hccs'] = dfs[hccs].sum(1)
    dfs['hcc_g'] = pd.cut(dfs['num_hccs'], [-np.inf,0,1,2,np.inf], labels=['0', '1','2','3+'])
    dfs['recip_age_g_weight'] = pd.cut(dfs['recip_age_g'], [-np.inf, 10, 20, 40, np.inf])

    dfs = add_weights(dfs, ['hcc_g', 'recip_race_g', 'recip_age_g_weight', 'recip_gender'], 'weight')

    # No choosers because they are identical as unweighted (all weights == 1)
    for label, f, asgn_type in [('', 'AA_RF_results_reweighted.csv', 'AA'), 
                                ('_5plan', 'AA_RF_results_5plans_reweighted.csv', 'AA'),
                                ('_AA_5plan_IV', 'AA_IV_results_5plans_reweighted.csv', 'AA')
                                ]:
        df = pd.read_csv(FS().analysis_p+'paper2_run_main_regressions_reweighted/output/'+f, 
                         sep='\t')
        
        # Need to calculate a weighted average separately for each column. 
        davg = {}
        for col in df.outcome.unique():
            data = dfs.loc[dfs.recip_race_g.eq(0) & dfs[['weight', col]].notnull().all(1)]
            davg[col] = np.average(data[col], weights=data['weight'])

        white_avg = pd.Series(davg)

        for reg, gp in df.groupby('regression'):
            pval = gp.loc[gp.variable.eq('plan)diff'), ['outcome', 'pval']].set_index('outcome')
            if f != 'AA_IV_results_5plans_reweighted.csv':
                keeps = ([f'{x}1.plan_race_asgn' for x in range(1, 6)] 
                         + [f'{x}1.plan_race' for x in range(1, 6)] 
                         + ['51o.plan_race_asgn', '1.recip_race_g'])
            else:
                keeps = ['1b.plan_id#1.recip_race_g', '2.plan_id#1.recip_race_g',
                         '3.plan_id#1.recip_race_g', '4.plan_id#1.recip_race_g',
                         '5.plan_id#1.recip_race_g']
            gp = gp[gp.variable.isin(keeps)]
            gp = gp.pivot(index='outcome', columns='variable',
                          values=['coeff', 'se', 'lb', 'ub']).apply(pd.to_numeric)
            gp['pval'] = pd.to_numeric(pval['pval'], errors='coerce')
            gp['pval_fmt'] = gp['pval'].apply(round_pval)

            sadj = pd.Series(multitest.multipletests(gp.reindex(paper_outcomes)['pval'], method='fdr_bh')[1],
                             index=paper_outcomes)
            gp['pval_adj'] = sadj
            gp['pval_adj_fmt'] = gp['pval_adj'].apply(round_pval)

            gp['white_avg'] = white_avg
            if f != 'AA_IV_results_5plans_reweighted.csv':
                gp['range'] = gp.xs('coeff', axis=1).filter(like='plan_race').apply(np.ptp, axis=1)
            else:
                gp['range'] = gp.xs('coeff', axis=1).filter(like='plan_id#').apply(np.ptp, axis=1)
            gp.to_csv(output_p+f'{reg}_results{label}.csv')


# Appx Tables 9 and 10 (By plan by race regressions, also necessary for the MAD tables)
def paper2_run_main_regressions_byplan_BWrace_ACs():
    """
    Mains RF regressions to recapture demeaned means for each plan and race group
    """
    output_p, temp_p, log_p, code_p = gen_paths(FS().analysis_p
                                                +'paper2_run_main_regressions_byplan_BWrace_ACs/')

    df = pd.read_pickle(FS().derived_p+'paper2_analytic_tables_to_yearly/output/'
                        +'LA_analytic_table_yearly.pkl')
    #Include only Black and White race categories
    df = df[df.recip_race_g.isin([0,1])].copy()

    cols = (['recip_id', 'plan_id', 'plan_id_asgn', 'asgn_type']
            + ['pc_numdays', 'pharm_numclms', 'ed_avoid_numdays', 'ednoadmit_numdays', 'all_cost_wins50']
            + ['recip_age_g', 'recip_gender_g', 'recip_race_g', 'plan_race']
            + ['uor'])
    df = df[cols]

    # Just choosers
    dfch = df[df.asgn_type.eq('Choice')]
    dfch.to_csv(temp_p+'Allrace_Choice.csv', chunksize=10**5, index=False)

    # AAs and choosers
    df.to_csv(temp_p+'Allrace.csv', chunksize=10**5, index=False)

    ws.stata_do(do_file=code_p+'paper2_AC_regs_5plans.do',
                params=[temp_p+'Allrace.csv',
                        output_p+f'Allrace_results_5plans.csv',
                        " ".join(['pc_numdays', 'pharm_numclms', 'ed_avoid_numdays', 'ednoadmit_numdays', 'all_cost_wins50'])],
                log_path=log_p)

    ws.stata_do(do_file=code_p+'paper2_AC_regs_5plans.do',
                params=[temp_p+'Allrace_Choice.csv',
                        output_p+f'Allrace_Choice_results_5plans.csv',
                        " ".join(['pc_numdays', 'pharm_numclms', 'ed_avoid_numdays', 'ednoadmit_numdays', 'all_cost_wins50'])],
                log_path=log_p)

    ws.stata_do(do_file=code_p+'paper2_AC_regs_5plans_interacted.do',
                params=[temp_p+'Allrace_Choice.csv',
                        output_p+f'Allrace_Choice_interacted_results_5plans.csv',
                        " ".join(['pc_numdays', 'pharm_numclms', 'ed_avoid_numdays', 'ednoadmit_numdays', 'all_cost_wins50'])],
                log_path=log_p)


    # Delete temp file
    for file in ['Allrace', 'Allrace_Choice']:
        os.remove(temp_p+f'{file}.csv')


# Appx Tables 9 and 10 (By plan by race regressions, also necessary for the MAD tables)
def paper2_clean_main_reg_results_byplan_BWrace_ACs(
  import_p='paper2_run_main_regressions_byplan_BWrace_ACs/output/', 
  output_p='paper2_clean_main_regression_results_byplan_BWrace_ACs/output/'):
    """
    Import and clean Stata output for by plan by race regressions - coefs, not demeaned means
    """
    output_p = FS().analysis_p+output_p

    paper_outcomes = ['pc_numdays', 'pharm_numclms', 'ed_avoid_numdays', 'ednoadmit_numdays', 'all_cost_wins50']
    text = 'Failed'

    for i, r in enumerate(['Allrace', 'Allrace_Choice', 'Allrace_Choice_interacted']):
        label = '_AC_5plan', 
        f = f'{r}_results_5plans.csv'
 
        df = pd.read_csv(FS().analysis_p+import_p+f, sep='\t')

        for reg, gp in df.groupby('regression'):
            pval = gp.loc[gp.variable.eq('plan)diff'), ['outcome', 'pval']].set_index('outcome')
            keeps = ([f'{x}.plan_id' for x in range(1, 6)] 
                   + [f'{x}.plan_id_asgn' for x in range(1, 6)] 
                   + ['1b.plan_id', '1b.plan_id_asgn'])
            gp = gp[gp.variable.isin(keeps)]
            gp = gp.pivot(index='outcome', columns='variable',
                          values=['coeff', 'se', 'lb', 'ub']).apply(pd.to_numeric)
            gp['pval'] = pd.to_numeric(pval['pval'], errors='coerce')
            gp['pval_fmt'] = gp['pval'].apply(round_pval)

            sadj = pd.Series(multitest.multipletests(gp.reindex(paper_outcomes)['pval'], method='fdr_bh')[1],
                             index=paper_outcomes)
            gp['pval_adj'] = sadj
            gp['pval_adj_fmt'] = gp['pval_adj'].apply(round_pval)

            # Get means
            dfs = pd.read_pickle(FS().derived_p+'paper2_analytic_tables_to_yearly/output/'
                                +'LA_analytic_table_yearly.pkl')
            dfs = dfs[dfs.recip_race_g.isin([0,1])].copy()

            # Full file
            if r == 'Allrace':
                dfs = dfs
            # Subset to choosers
            if r in ['Allrace_Choice', 'Allrace_Choice_interacted']:
                dfs = dfs[dfs.asgn_type.eq('Choice')]

            print(i, dfs.loc[:, df.outcome.unique()].mean())
            gp['avg'] = dfs.loc[:, df.outcome.unique()].mean()

            gp['range'] = gp.xs('coeff', axis=1).filter(like='plan').apply(np.ptp, axis=1)
            gp['std_dev'] = gp.xs('coeff', axis=1).filter(like='plan').std(axis=1)
            
            file = f'{r}_{reg}_results{label}.csv'
            gp.loc[paper_outcomes].to_csv(output_p+file)  


def paper2_run_main_regressions_byplan_byrace():
    """
    Mains RF regressions to recapture demeaned means for each plan and race group
    """
    output_p, temp_p, log_p, code_p = gen_paths(FS().analysis_p
                                                +'paper2_run_main_regressions_byplan_byrace/')

    df = pd.read_pickle(FS().derived_p+'paper2_analytic_tables_to_yearly/output/'
                        +'LA_analytic_table_yearly.pkl')
    df = df[df.recip_race_g.isin([0,1])].copy()

    cols = (['recip_id', 'plan_id', 'plan_id_asgn', 'asgn_type']
            + ['pc_numdays', 'pharm_numclms', 'ed_avoid_numdays', 'ednoadmit_numdays', 'all_cost_wins50']
            + ['recip_age_g', 'recip_gender_g', 'recip_race_g', 'plan_race']
            + ['uor'])
    df = df[cols]

    grpr = df['recip_race_g'].eq(1).map({True: 'blk', False: 'wht'})
    for label, gp in df.groupby(grpr):

        dfch = gp[gp.asgn_type.eq('Choice')]
        dfch.to_csv(temp_p+'Choice.csv', chunksize=10**5, index=False)

        dfaa = gp[gp.asgn_type.eq('AA')]
        dfaa.to_csv(temp_p+'AA.csv', chunksize=10**5, index=False)

        ws.stata_do(do_file=code_p+'paper2_main_regs_RF_5plans.do',
                    params=[temp_p+'AA.csv',
                            output_p+f'{label}_AA_RF_results_5plans.csv',
                            " ".join(['pc_numdays', 'pharm_numclms', 'ed_avoid_numdays', 'ednoadmit_numdays', 'all_cost_wins50'])],
                    log_path=log_p)

        ws.stata_do(do_file=code_p+'paper2_AC_regs_5plans.do',
                    params=[temp_p+'Choice.csv',
                            output_p+f'{label}_AC_results_5plans.csv',
                            " ".join(['pc_numdays', 'pharm_numclms', 'ed_avoid_numdays', 'ednoadmit_numdays', 'all_cost_wins50'])],
                    log_path=log_p)

        # Delete temp file
        for file in ['AA', 'Choice']:
            os.remove(temp_p+f'{file}.csv')


# Appx Tables 9 and 10 (By plan by race regressions, also necessary for the MAD tables)
def paper2_clean_main_reg_results_byplan_byrace(iteration=False,
  import_p='paper2_run_main_regressions_byplan_byrace/output/', 
  output_p='paper2_clean_main_regression_results_byplan_byrace/output/'):
    """
    Import and clean Stata output for by plan by race regressions - coefs, not demeaned means
    """
    output_p = FS().analysis_p+output_p

    paper_outcomes = ['pc_numdays', 'pharm_numclms', 'ed_avoid_numdays', 'ednoadmit_numdays', 'all_cost_wins50']
    text = 'Failed'

    for i, r in enumerate(['wht', 'blk']):

        for label, f, asgn_type in [('_5plan', f'{r}_AA_RF_results_5plans.csv', 'AA'), 
                                    ('_AC_5plan', f'{r}_AC_results_5plans.csv', 'Choice')]:
            if iteration is not False:
                f = f.replace('.csv', f'_{iteration}.csv')
            try:
                df = pd.read_csv(FS().analysis_p+import_p+f, sep='\t')

                for reg, gp in df.groupby('regression'):
                    pval = gp.loc[gp.variable.eq('plan)diff'), ['outcome', 'pval']].set_index('outcome')
                    keeps = ([f'{x}.plan_id' for x in range(1, 6)] 
                           + [f'{x}.plan_id_asgn' for x in range(1, 6)] 
                           + ['1b.plan_id', '1b.plan_id_asgn'])
                    gp = gp[gp.variable.isin(keeps)]
                    gp = gp.pivot(index='outcome', columns='variable',
                                  values=['coeff', 'se', 'lb', 'ub']).apply(pd.to_numeric)
                    gp['pval'] = pd.to_numeric(pval['pval'], errors='coerce')
                    gp['pval_fmt'] = gp['pval'].apply(round_pval)

                    sadj = pd.Series(multitest.multipletests(gp.reindex(paper_outcomes)['pval'], method='fdr_bh')[1],
                                     index=paper_outcomes)
                    gp['pval_adj'] = sadj
                    gp['pval_adj_fmt'] = gp['pval_adj'].apply(round_pval)

                    # Get means
                    if iteration is False:
                        dfs = pd.read_pickle(FS().derived_p+'paper2_analytic_tables_to_yearly/output/'
                                            +'LA_analytic_table_yearly.pkl')
                        dfs = dfs[dfs.recip_race_g.eq(i)]
                        gp['avg'] = dfs.loc[dfs.asgn_type.eq(asgn_type), df.outcome.unique()].mean()

                    gp['range'] = gp.xs('coeff', axis=1).filter(like='plan').apply(np.ptp, axis=1)
                    gp['std_dev'] = gp.xs('coeff', axis=1).filter(like='plan').std(axis=1)
                    if iteration is not False:
                        file = f'{r}_{reg}_results{label}_{iteration}.csv'
                        text = f'Iteration {iteration} done'
                    else:
                        file = f'{r}_{reg}_results{label}.csv'
                        text = 'Done'
                    gp.loc[paper_outcomes].to_csv(output_p+file)  
            except FileNotFoundError as e:
                pass
        if i == 1:
            print(text)               


def paper2_figure2_score_comparison():
    """
    Figure of sorting in observational sample, demeaned on age, sex and UoR
    """
    output_p = FS().analysis_p+'paper2_figure2_score_comparison/output/'

    td = {'pc_numdays': 'Primary Care Visits\n$\mathit{Score \> as \> percent \> of \> population \> mean}$',
          'pharm_numclms': 'Prescriptions\n$\mathit{Score \> as \> percent \> of \> population \> mean}$', 
          'ed_avoid_numdays': 'Low-Acuity ED Visits\n$\mathit{Score \> as \> percent \> of \> population \> mean}$', 
          'all_cost_wins50': 'Total Spending\n$\mathit{Score \> as \> percent \> of \> population \> mean}$'}

    td2 = {'pc_numdays': 'Primary Care Visits\n$\mathit{Absolute \> difference \> in \>  percentage \> plan \> scores}$',
           'pharm_numclms': 'Prescriptions\n$\mathit{Absolute \> difference \> in \>  percentage \> plan \> scores}$', 
           'ed_avoid_numdays': 'Low-Acuity ED Visits\n$\mathit{Absolute \> difference \> in \>  percentage \> plan \> scores}$', 
           'all_cost_wins50': 'Total Spending\n$\mathit{Absolute \> difference \> in \>  percentage \> plan \> scores}$'}

    idl = {'obs_pooled': 'Observational\nPooled',
           'obs_strat': 'Observational\nStratified',
           'AA_black': 'Randomized\nBlack'}

    styles = {0: {'marker': 's', 'color': '#6870c8', 'ms': 5},
              1: {'marker': 'o', 'color': '#af953c', 'ms': 5},
              2: {'marker': '*', 'color': '#a44f9a', 'ms': 7},
              3: {'marker': 'v', 'color': '#56ae6c', 'ms': 5},
              4: {'marker': 'D', 'color': '#ba4a4f', 'ms': 4}}

    plan_rename = {'1b.plan_id_asgn': '1b.plan_id',
                  '2.plan_id_asgn': '2.plan_id',
                  '3.plan_id_asgn': '3.plan_id',
                  '4.plan_id_asgn': '4.plan_id',
                  '5.plan_id_asgn': '5.plan_id'
                  }


    pl = [('obs_pooled', FS().analysis_p+"paper2_clean_main_regression_results_byplan_BWrace_ACs/output/Allrace_Choice_interacted_age_sex_results('_AC_5plan',).csv"),
          ('obs_strat', FS().analysis_p+'paper2_clean_main_regression_results_byplan_byrace/output/blk_age_sex_results_AC_5plan.csv'),
          ('AA_black', FS().analysis_p+'paper2_clean_main_regression_results_byplan_byrace/output/blk_age_sex_results_5plan.csv'), 
         ]

    l = []
    for sample, f in pl:
        df = pd.read_csv(f, skiprows=[2], header=[0,1])
        df['sample'] = sample
        df = df.rename(columns=plan_rename)
        l.append(df)

    df = pd.concat(l)
    df.columns = [f'{x*(not "Unnamed" in x)}_{y*(not "Unnamed" in y)}'.strip('_') for x,y in df.columns]
    df = df.reindex(['variable', 'sample'] + df.columns.difference(['variable', 'sample']).tolist(), axis=1)

    df = df[df.variable.ne('ednoadmit_numdays')]
    df = df.reset_index(drop=True)

    pcols = [x for x in df.columns if 'coeff_' in x]
    secols = [x for x in df.columns if 'se_' in x]
    df[pcols] = df[pcols].sub(df[pcols].mean(1), axis=0).div(df['avg'], axis=0)
    df[secols] = df[secols].div(df['avg'], axis=0)


    df = df.set_index(['sample', 'variable'])[pcols+secols]
    df.columns = [f'Plan {x}' for x in range(1,6)]+[f'Plan {x}_se' for x in range(1,6)]

    # Scale variables to %
    df = df*100
    dfse = df[[f'Plan {x}_se' for x in range(1,6)]].copy()
    df = df[[f'Plan {x}' for x in range(1,6)]].copy()



    fig, ax = plt.subplots(figsize=(6, 6), nrows=2, ncols=2)
    ax = ax.flatten()
    plt.subplots_adjust(hspace=0.00, wspace=0.25)

    for axis in ax:
        axis.tick_params(axis='both', which='major', labelsize=7.5)

    for i,variable in enumerate(['pc_numdays', 'pharm_numclms', 'ed_avoid_numdays', 'all_cost_wins50']):
        gp = df[df.index.get_level_values('variable') == variable]
        gpse = dfse.xs(variable, level=1)

        gp = gp.reset_index(level=1, drop=True)

        gp.index = [idl.get(x) for x in gp.index]

        for k, col in enumerate(gp.columns):
            ax[i].errorbar(x=np.array([0,1,2])+(k-2)*0.03, 
                             y=gp[col], lw=0,
                             yerr=1.96*gpse[f'{col}_se'], elinewidth=1, capthick=1, capsize=2,
                             **styles[k], label=col)

        ax[i].set_title(td.get(variable), loc='left', fontsize=8.5)
        ax[i].set_xticks([0,1,2])
        if i < 2 :
            ax[i].set_xticklabels([])
        else:           
            ax[i].set_xticklabels(['Observational\nPooled\nEnrollees', 'Observational\nStratified\nBlack Enrollees',
                                      'Randomized\nBlack Enrollees'])

    ax[0].legend()
    handles, labels = ax[0].get_legend_handles_labels()
    # remove the errorbars
    handles = [h[0] for h in handles]
    # use them in the legend
    ax[0].legend(handles, labels, loc=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(output_p+'Figure3.pdf', dpi=300)
    plt.savefig(output_p+'no_RA_black.png', dpi=300)
    plt.show()



    fig, ax = plt.subplots(figsize=(6, 6), nrows=2, ncols=2)
    ax = ax.flatten()
    plt.subplots_adjust(hspace=0.00, wspace=0.25)

    for axis in ax:
        axis.tick_params(axis='both', which='major', labelsize=8)

    dd = {}
    for i,variable in enumerate(['pc_numdays', 'pharm_numclms', 'ed_avoid_numdays', 'all_cost_wins50']):
        gp = df[df.index.get_level_values('variable') == variable]

        score_diff = np.abs((gp - gp.loc['AA_black']).drop(('AA_black', variable)))
        score_diff = score_diff.reset_index(level=1, drop=True)
        # Rename for plotting
        score_diff.index = [{'obs_pooled': 'Compared to\nObservational\nPooled',
                             'obs_strat': 'Compared to\nObservational\nStratified'}.get(x) for x in score_diff.index]
        score_diff.T.plot(ax=ax[i], kind='bar', legend=False, rot=0, ec='k', zorder=4)    
        dd[variable] = score_diff


        ax[i].set_title(td2.get(variable), loc='left', fontsize=8)
        if i < 2 :
            ax[i].set_xticklabels([])


    ax[0].legend()
    handles, labels = ax[0].get_legend_handles_labels()
    # remove the errorbars
    handles = [h[0] for h in handles]
    # use them in the legend
    ax[0].legend(handles, labels, loc=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(output_p+'no_RA_black_score.png', dpi=300)
    plt.show()
    pd.concat(dd).T.to_csv(output_p+'Score_diff_black.csv')


    pl = [('obs_pooled', FS().analysis_p+"paper2_clean_main_regression_results_byplan_BWrace_ACs/output/Allrace_Choice_interacted_age_sex_results('_AC_5plan',).csv"),
          ('obs_strat', FS().analysis_p+'paper2_clean_main_regression_results_byplan_byrace/output/wht_age_sex_results_AC_5plan.csv'),
          ('AA_white', FS().analysis_p+'paper2_clean_main_regression_results_byplan_byrace/output/wht_age_sex_results_5plan.csv'), 
         ]

    l = []
    for sample, f in pl:
        df = pd.read_csv(f, skiprows=[2], header=[0,1])
        df['sample'] = sample
        df = df.rename(columns=plan_rename)
        l.append(df)

    df = pd.concat(l)
    df.columns = [f'{x*(not "Unnamed" in x)}_{y*(not "Unnamed" in y)}'.strip('_') for x,y in df.columns]
    df = df.reindex(['variable', 'sample'] + df.columns.difference(['variable', 'sample']).tolist(), axis=1)

    df = df[df.variable.ne('ednoadmit_numdays')]
    df = df.reset_index(drop=True)

    pcols = [x for x in df.columns if 'coeff_' in x]
    secols = [x for x in df.columns if 'se_' in x]
    df[pcols] = df[pcols].sub(df[pcols].mean(1), axis=0).div(df['avg'], axis=0)
    df[secols] = df[secols].div(df['avg'], axis=0)


    df = df.set_index(['sample', 'variable'])[pcols+secols]
    df.columns = [f'Plan {x}' for x in range(1,6)]+[f'Plan {x}_se' for x in range(1,6)]

    # Scale variables to %
    df = df*100
    dfse = df[[f'Plan {x}_se' for x in range(1,6)]].copy()
    df = df[[f'Plan {x}' for x in range(1,6)]].copy()

    fig, ax = plt.subplots(figsize=(6, 6), nrows=2, ncols=2)
    ax = ax.flatten()
    plt.subplots_adjust(hspace=0.00, wspace=0.25)

    for axis in ax:
        axis.tick_params(axis='both', which='major', labelsize=7.5)

    for i,variable in enumerate(['pc_numdays', 'pharm_numclms', 'ed_avoid_numdays', 'all_cost_wins50']):
        gp = df[df.index.get_level_values('variable') == variable]
        gpse = dfse.xs(variable, level=1)

        gp = gp.reset_index(level=1, drop=True)

        gp.index = [idl.get(x) for x in gp.index]

        for k, col in enumerate(gp.columns):
            ax[i].errorbar(x=np.array([0,1,2])+(k-2)*0.03, lw=0, 
                             y=gp[col], 
                             yerr=1.96*gpse[f'{col}_se'], elinewidth=1, capthick=1, capsize=2,
                             **styles[k], label=col)


        ax[i].set_title(td.get(variable), loc='left', fontsize=8.5)
        ax[i].set_xticks([0,1,2])
        if i < 2 :
            ax[i].set_xticklabels([])
        else:           
            ax[i].set_xticklabels(['Observational\nPooled\nEnrollees', 'Observational\nStratified\nWhite Enrollees',
                                      'Randomized\nWhite Enrollees'])


    ax[0].legend()
    handles, labels = ax[0].get_legend_handles_labels()
    # remove the errorbars
    handles = [h[0] for h in handles]
    # use them in the legend

    ax[0].legend(handles, labels, loc=0, fontsize=6)

    plt.tight_layout()
    plt.savefig(output_p+'no_RA_white.png', dpi=300)
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 6), nrows=2, ncols=2)
    ax = ax.flatten()
    plt.subplots_adjust(hspace=0.00, wspace=0.25)

    for axis in ax:
        axis.tick_params(axis='both', which='major', labelsize=8)

    dd = {}
    for i,variable in enumerate(['pc_numdays', 'pharm_numclms', 'ed_avoid_numdays', 'all_cost_wins50']):
        gp = df[df.index.get_level_values('variable') == variable]

        score_diff = np.abs((gp - gp.loc['AA_white']).drop(('AA_white', variable)))
        score_diff = score_diff.reset_index(level=1, drop=True)
        # Rename for plotting
        score_diff.index = [{'obs_pooled': 'Compared to\nObservational\nPooled',
                             'obs_strat': 'Compared to\nObservational\nStratified'}.get(x) for x in score_diff.index]
        score_diff.T.plot(ax=ax[i], kind='bar', legend=False, rot=0, ec='k', zorder=4)    
        dd[variable] = score_diff


        ax[i].set_title(td2.get(variable), loc='left', fontsize=8)
        if i < 2 :
            ax[i].set_xticklabels([])


    ax[0].legend()
    handles, labels = ax[0].get_legend_handles_labels()
    # remove the errorbars
    handles = [h[0] for h in handles]
    # use them in the legend
    ax[0].legend(handles, labels, loc=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(output_p+'no_RA_white_score.png', dpi=300)
    plt.show()
    pd.concat(dd).T.to_csv(output_p+'Score_diff_white.csv')


def paper2_figure2_score_comparison_riskAdjusted():
    """
    Figure of sorting in observational sample, demeaned on age, sex and UoR
    """
    output_p = FS().analysis_p+'paper2_figure2_score_comparison_riskAdjusted/output/'

    td = {'pc_numdays': 'Primary Care Visits\n$\mathit{Score \> as \> percent \> of \> population \> mean}$',
          'pharm_numclms': 'Prescriptions\n$\mathit{Score \> as \> percent \> of \> population \> mean}$', 
          'ed_avoid_numdays': 'Low-Acuity ED Visits\n$\mathit{Score \> as \> percent \> of \> population \> mean}$', 
          'all_cost_wins50': 'Total Spending\n$\mathit{Score \> as \> percent \> of \> population \> mean}$'}

    td2 = {'pc_numdays': 'Primary Care Visits\n$\mathit{Absolute \> difference \> in \>  percentage \> plan \> scores}$',
           'pharm_numclms': 'Prescriptions\n$\mathit{Absolute \> difference \> in \>  percentage \> plan \> scores}$', 
           'ed_avoid_numdays': 'Low-Acuity ED Visits\n$\mathit{Absolute \> difference \> in \>  percentage \> plan \> scores}$', 
           'all_cost_wins50': 'Total Spending\n$\mathit{Absolute \> difference \> in \>  percentage \> plan \> scores}$'}

    idl = {'obs_pooled': 'Observational\nPooled',
           'obs_strat': 'Observational\nStratified',
           'AA_black': 'Randomized\nBlack'}

    styles = {0: {'marker': 's', 'color': '#6870c8', 'ms': 5},
              1: {'marker': 'o', 'color': '#af953c', 'ms': 5},
              2: {'marker': '*', 'color': '#a44f9a', 'ms': 7},
              3: {'marker': 'v', 'color': '#56ae6c', 'ms': 5},
              4: {'marker': 'D', 'color': '#ba4a4f', 'ms': 4}}

    plan_rename = {'1b.plan_id_asgn': '1b.plan_id',
                  '2.plan_id_asgn': '2.plan_id',
                  '3.plan_id_asgn': '3.plan_id',
                  '4.plan_id_asgn': '4.plan_id',
                  '5.plan_id_asgn': '5.plan_id'
                  }


    pl = [('obs_pooled', FS().analysis_p+"paper2_clean_main_regression_results_byplan_BWrace_ACs_HCCs/output/Allrace_Choice_interacted_age_sex_results('_AC_5plan',).csv"),
          ('obs_strat', FS().analysis_p+'paper2_clean_main_regression_results_byplan_byrace_HCCs/output/blk_age_sex_results_AC_5plan.csv'),
          ('AA_black', FS().analysis_p+'paper2_clean_main_regression_results_byplan_byrace_HCCs/output/blk_age_sex_results_5plan.csv'), 
         ]

    l = []
    for sample, f in pl:
        df = pd.read_csv(f, skiprows=[2], header=[0,1])
        df['sample'] = sample
        df = df.rename(columns=plan_rename)
        l.append(df)

    df = pd.concat(l)
    df.columns = [f'{x*(not "Unnamed" in x)}_{y*(not "Unnamed" in y)}'.strip('_') for x,y in df.columns]
    df = df.reindex(['variable', 'sample'] + df.columns.difference(['variable', 'sample']).tolist(), axis=1)

    df = df[df.variable.ne('ednoadmit_numdays')]
    df = df.reset_index(drop=True)

    pcols = [x for x in df.columns if 'coeff_' in x]
    secols = [x for x in df.columns if 'se_' in x]
    df[pcols] = df[pcols].sub(df[pcols].mean(1), axis=0).div(df['avg'], axis=0)
    df[secols] = df[secols].div(df['avg'], axis=0)


    df = df.set_index(['sample', 'variable'])[pcols+secols]
    df.columns = [f'Plan {x}' for x in range(1,6)]+[f'Plan {x}_se' for x in range(1,6)]

    # Scale variables to %
    df = df*100
    dfse = df[[f'Plan {x}_se' for x in range(1,6)]].copy()
    df = df[[f'Plan {x}' for x in range(1,6)]].copy()



    fig, ax = plt.subplots(figsize=(6, 6), nrows=2, ncols=2)
    ax = ax.flatten()
    plt.subplots_adjust(hspace=0.00, wspace=0.25)

    for axis in ax:
        axis.tick_params(axis='both', which='major', labelsize=7.5)

    for i,variable in enumerate(['pc_numdays', 'pharm_numclms', 'ed_avoid_numdays', 'all_cost_wins50']):
        gp = df[df.index.get_level_values('variable') == variable]
        gpse = dfse.xs(variable, level=1)

        gp = gp.reset_index(level=1, drop=True)

        gp.index = [idl.get(x) for x in gp.index]

        for k, col in enumerate(gp.columns):
            ax[i].errorbar(x=np.array([0,1,2])+(k-2)*0.03, lw=0, 
                             y=gp[col], 
                             yerr=1.96*gpse[f'{col}_se'], elinewidth=1, capthick=1, capsize=2,
                             **styles[k], label=col)

        ax[i].set_title(td.get(variable), loc='left', fontsize=8.5)
        ax[i].set_xticks([0,1,2])
        if i < 2 :
            ax[i].set_xticklabels([])
        else:           
            ax[i].set_xticklabels(['Observational\nPooled\nEnrollees', 'Observational\nStratified\nBlack Enrollees',
                                      'Randomized\nBlack Enrollees'])

    ax[0].legend()
    handles, labels = ax[0].get_legend_handles_labels()
    # remove the errorbars
    handles = [h[0] for h in handles]
    # use them in the legend
    ax[0].legend(handles, labels, loc=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(output_p+'RA_black.png', dpi=300)
    plt.show()



    fig, ax = plt.subplots(figsize=(6, 6), nrows=2, ncols=2)
    ax = ax.flatten()
    plt.subplots_adjust(hspace=0.00, wspace=0.25)

    for axis in ax:
        axis.tick_params(axis='both', which='major', labelsize=8)

    dd = {}
    for i,variable in enumerate(['pc_numdays', 'pharm_numclms', 'ed_avoid_numdays', 'all_cost_wins50']):
        gp = df[df.index.get_level_values('variable') == variable]

        score_diff = np.abs((gp - gp.loc['AA_black']).drop(('AA_black', variable)))
        score_diff = score_diff.reset_index(level=1, drop=True)
        # Rename for plotting
        score_diff.index = [{'obs_pooled': 'Compared to\nObservational\nPooled',
                             'obs_strat': 'Compared to\nObservational\nStratified'}.get(x) for x in score_diff.index]
        score_diff.T.plot(ax=ax[i], kind='bar', legend=False, rot=0, ec='k', zorder=4)    
        dd[variable] = score_diff
        
        
        ax[i].set_title(td2.get(variable), loc='left', fontsize=8)
        if i < 2 :
            ax[i].set_xticklabels([])


    ax[0].legend()
    handles, labels = ax[0].get_legend_handles_labels()
    # remove the errorbars
    handles = [h[0] for h in handles]
    # use them in the legend
    ax[0].legend(handles, labels, loc=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(output_p+'RA_black_score.png', dpi=300)
    plt.show()
    pd.concat(dd).T.to_csv(output_p+'Score_diff_black.csv')


    pl = [('obs_pooled', FS().analysis_p+"paper2_clean_main_regression_results_byplan_BWrace_ACs_HCCs/output/Allrace_Choice_interacted_age_sex_results('_AC_5plan',).csv"),
          ('obs_strat', FS().analysis_p+'paper2_clean_main_regression_results_byplan_byrace_HCCs/output/wht_age_sex_results_AC_5plan.csv'),
          ('AA_white', FS().analysis_p+'paper2_clean_main_regression_results_byplan_byrace_HCCs/output/wht_age_sex_results_5plan.csv'), 
         ]

    l = []
    for sample, f in pl:
        df = pd.read_csv(f, skiprows=[2], header=[0,1])
        df['sample'] = sample
        df = df.rename(columns=plan_rename)
        l.append(df)

    df = pd.concat(l)
    df.columns = [f'{x*(not "Unnamed" in x)}_{y*(not "Unnamed" in y)}'.strip('_') for x,y in df.columns]
    df = df.reindex(['variable', 'sample'] + df.columns.difference(['variable', 'sample']).tolist(), axis=1)

    df = df[df.variable.ne('ednoadmit_numdays')]
    df = df.reset_index(drop=True)

    pcols = [x for x in df.columns if 'coeff_' in x]
    secols = [x for x in df.columns if 'se_' in x]
    df[pcols] = df[pcols].sub(df[pcols].mean(1), axis=0).div(df['avg'], axis=0)
    df[secols] = df[secols].div(df['avg'], axis=0)


    df = df.set_index(['sample', 'variable'])[pcols+secols]
    df.columns = [f'Plan {x}' for x in range(1,6)]+[f'Plan {x}_se' for x in range(1,6)]

    # Scale variables to %
    df = df*100
    dfse = df[[f'Plan {x}_se' for x in range(1,6)]].copy()
    df = df[[f'Plan {x}' for x in range(1,6)]].copy()

    fig, ax = plt.subplots(figsize=(6, 6), nrows=2, ncols=2)
    ax = ax.flatten()
    plt.subplots_adjust(hspace=0.00, wspace=0.25)

    for axis in ax:
        axis.tick_params(axis='both', which='major', labelsize=7.5)

    for i,variable in enumerate(['pc_numdays', 'pharm_numclms', 'ed_avoid_numdays', 'all_cost_wins50']):
        gp = df[df.index.get_level_values('variable') == variable]
        gpse = dfse.xs(variable, level=1)

        gp = gp.reset_index(level=1, drop=True)

        gp.index = [idl.get(x) for x in gp.index]

        for k, col in enumerate(gp.columns):
            ax[i].errorbar(x=np.array([0,1,2])+(k-2)*0.03, lw=0, 
                             y=gp[col], 
                             yerr=1.96*gpse[f'{col}_se'], elinewidth=1, capthick=1, capsize=2,
                             **styles[k], label=col)


        ax[i].set_title(td.get(variable), loc='left', fontsize=8.5)
        ax[i].set_xticks([0,1,2])
        if i < 2 :
            ax[i].set_xticklabels([])
        else:           
            ax[i].set_xticklabels(['Observational\nPooled\nEnrollees', 'Observational\nStratified\nWhite Enrollees',
                                      'Randomized\nWhite Enrollees'])


    ax[0].legend()
    handles, labels = ax[0].get_legend_handles_labels()
    # remove the errorbars
    handles = [h[0] for h in handles]
    # use them in the legend

    ax[0].legend(handles, labels, loc=0, fontsize=6)

    plt.tight_layout()
    plt.savefig(output_p+'RA_white.png', dpi=300)
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 6), nrows=2, ncols=2)
    ax = ax.flatten()
    plt.subplots_adjust(hspace=0.00, wspace=0.25)

    for axis in ax:
        axis.tick_params(axis='both', which='major', labelsize=8)

    dd = {}
    for i,variable in enumerate(['pc_numdays', 'pharm_numclms', 'ed_avoid_numdays', 'all_cost_wins50']):
        gp = df[df.index.get_level_values('variable') == variable]

        score_diff = np.abs((gp - gp.loc['AA_white']).drop(('AA_white', variable)))
        score_diff = score_diff.reset_index(level=1, drop=True)
        # Rename for plotting
        score_diff.index = [{'obs_pooled': 'Compared to\nObservational\nPooled',
                             'obs_strat': 'Compared to\nObservational\nStratified'}.get(x) for x in score_diff.index]
        score_diff.T.plot(ax=ax[i], kind='bar', legend=False, rot=0, ec='k', zorder=4)    
        dd[variable] = score_diff
        

        ax[i].set_title(td2.get(variable), loc='left', fontsize=8)
        if i < 2 :
            ax[i].set_xticklabels([])


    ax[0].legend()
    handles, labels = ax[0].get_legend_handles_labels()
    # remove the errorbars
    handles = [h[0] for h in handles]
    # use them in the legend
    ax[0].legend(handles, labels, loc=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(output_p+'RA_white_score.png', dpi=300)
    plt.show()
    pd.concat(dd).T.to_csv(output_p+'Score_diff_white.csv')


def paper2_figure_planeff():
    output_p = FS().analysis_p+'paper2_figure_planeff/output/'

    # WHat is the base regression used for estimates reported in the paper
    base_reg = 'age_sex_elig'

    d = {}
    dpval = {}
    for f, asgn_type in [('AA_RF_results_5plans.csv', 'AA'),
                         ('AC_results_5plans.csv', 'Choice')]:
        df = pd.read_csv(FS().analysis_p+'paper2_run_main_regressions/output/'+f, sep='\t')
        df = df[df.regression.eq(base_reg)]

        pval = df.loc[df.variable.eq('plan)diff'), ['outcome', 'pval']].set_index('outcome')
        pval['pval'] = pval['pval'].apply(pd.to_numeric, errors='coerce').apply(round_pval)
        keeps = ([f'{x}1.plan_race_asgn' for x in range(1, 6)] 
                 + [f'{x}1.plan_race' for x in range(1, 6)] 
                 + ['51o.plan_race_asgn', '1.recip_race_g'])
        df = df[df.variable.isin(keeps)]
        df['variable'] = df['variable'].str[0]
        df = df.pivot(index='outcome', columns='variable',
                     values=['coeff', 'se']).apply(pd.to_numeric)
        d[asgn_type] = df
        dpval[asgn_type] = pval


    outcomes = ['pc_numdays', 'pharm_numclms', 'ed_avoid_numdays', 'all_cost_wins50']
    title_d = {'all_cost_wins50': 'Total Spending\n($ per enrollee per year)', 
               'pc_numdays': 'Primary Care Visits\n(per 100 enrollees per year)', 
               'pharm_numclms': 'Prescriptions\n(per 100 enrollees per year)', 
               'ed_avoid_numdays': 'Avoidable ED Visits\n(per 100 enrollees per year)'}
    color_d = {'AA': '#4781BE', 'Choice': '#F2C463'}
    label_d = {'AA': 'Randomized', 'Choice': 'Observational'}
    keeps = ([f'{x}1.plan_race_asgn' for x in range(1, 6)]
             + [f'{x}1.plan_race' for x in range(1, 6)])


    fig, ax = plt.subplots(figsize=(9, 8), nrows=2, ncols=2)
    ax = ax.flatten()
    plt.subplots_adjust(hspace=0.2, wspace=0.27)
    for axis in ax:
        axis.tick_params(axis='both', which='major', labelsize=10.5)

    for i, outcome in enumerate(outcomes):

        ax[i].grid(color='#ABABAB', zorder=0, alpha=0.8, linestyle='--', axis='both')
        ax[i].set_title(title_d[outcome], fontsize=12)

        for sample in ['AA', 'Choice']:
            gpp = d[sample].loc[outcome]
            if outcome != 'all_cost_wins50':
                gpp = gpp*100
            eb = ax[i].errorbar(x=gpp.xs('coeff'),
                                y=gpp.xs('coeff').index.astype(int)+0.15*{'AA': -1, 'Choice': 1}.get(sample), 
                                xerr=gpp.xs('se')*1.96,
                                lw=0, elinewidth=2, capthick=2, marker='o',
                                color=color_d[sample], capsize=3,
                                label=f'{label_d[sample]}\nF-stat p-val: {dpval[sample].loc[outcome, "pval"]}')
            ax[i].axvline(0,0,1, color='black', linestyle='--')
            ax[i].legend(fontsize=7.9, framealpha=1)
            if i % 2 == 1:
                ax[i].set_yticklabels([None])
            else: 
                ax[i].set_yticklabels([f'Plan {i}' for i in range(0,6)])

    plt.tight_layout()
    plt.savefig(output_p+'plan_eff.png', dpi=300)
    plt.show()

    output_p = FS().analysis_p+'paper2_figure_planeff/output/'

    d = {}
    dpval = {}
    for f, asgn_type in [('AA_RF_results_5plans.csv', 'AA'),
                         ('AC_results_5plans.csv', 'Choice')]:
        df = pd.read_csv(FS().analysis_p+'paper2_run_main_regressions/output/'+f, sep='\t')
        df = df[df.regression.isin(['demo_health', base_reg])]

        pval = (df.loc[df.variable.eq('plan)diff'), ['outcome', 'pval', 'regression']]
                  .set_index(['outcome', 'regression']))
        pval['pval'] = pval['pval'].apply(pd.to_numeric, errors='coerce').apply(round_pval)
        keeps = ([f'{x}1.plan_race_asgn' for x in range(1, 6)] 
                 + [f'{x}1.plan_race' for x in range(1, 6)] 
                 + ['51o.plan_race_asgn', '1.recip_race_g'])
        df = df[df.variable.isin(keeps)]
        df['variable'] = df['variable'].str[0]
        df = df.pivot(index=['outcome', 'regression'], columns='variable',
                      values=['coeff', 'se']).apply(pd.to_numeric)
        d[asgn_type] = df
        dpval[asgn_type] = pval


    outcomes = ['pc_numdays', 'pharm_numclms', 'ed_avoid_numdays', 'all_cost_wins50']
    title_d = {'all_cost_wins50': 'Total Spending\n($ per enrollee per year)', 
               'pc_numdays': 'Primary Care Visits\n(per 100 enrollees per year)', 
               'pharm_numclms': 'Prescriptions\n(per 100 enrollees per year)', 
               'ed_avoid_numdays': 'Avoidable ED Visits\n(per 100 enrollees per year)'}
    color_d = {'AA': '#4781BE', 'Choice': '#F2C463'}
    label_d = {'AA': 'Randomized', 'Choice': 'Observational'}
    keeps = ([f'{x}1.plan_race_asgn' for x in range(1, 6)]
             + [f'{x}1.plan_race' for x in range(1, 6)])


    fig, ax = plt.subplots(figsize=(10, 9), nrows=2, ncols=2)
    ax = ax.flatten()
    plt.subplots_adjust(hspace=0.2, wspace=0.27)
    for axis in ax:
        axis.tick_params(axis='both', which='major', labelsize=10.5)

        
    for i, outcome in enumerate(outcomes):

        ax[i].grid(color='#ABABAB', zorder=0, alpha=0.8, linestyle='--', axis='both')
        ax[i].set_title(title_d[outcome], fontsize=12)

        for sample in ['AA', 'Choice']:
            gpp = d[sample].loc[outcome]
            if outcome != 'all_cost_wins50':
                gpp = gpp*100
            for reg in [base_reg, 'demo_health']:
                gppp = gpp.loc[reg]
                eb = ax[i].errorbar(x=gppp.xs('coeff'),
                                    y=(gppp.xs('coeff').index.astype(int)
                                       +0.2*{'AA': -1, 'Choice': 1}.get(sample)
                                       +0.07*{base_reg: -1, 'demo_health': 1}.get(reg)), 
                                    xerr=gppp.xs('se')*1.96,
                                    lw=0, elinewidth=1, capthick=2, 
                                    marker={base_reg: 'x', 'demo_health': 'o'}.get(reg), 
                                    ms=5,
                                    color=color_d[sample], capsize=3,
                                    label=f'{label_d[sample]}\nF-stat p-val: {dpval[sample].loc[(outcome, reg), "pval"]}'
                                    )
                if reg == 'raw':
                    eb[-1][0].set_linestyle('--')
                
            ax[i].axvline(0,0,1, color='black', linestyle='--')
            ax[i].legend(fontsize=7.8, framealpha=1)
            if i % 2 == 1:
                ax[i].set_yticklabels([None])
            else: 
                ax[i].set_yticklabels([f'Plan {i}' for i in range(0,6)])

    plt.tight_layout()
    plt.savefig(output_p+'plan_eff_appendix.png', dpi=300)
    plt.show()


def paper2_figure_averages():
    """
    Figure of causal plan effects vs sorting in observational sample. 
    """
    output_p = FS().analysis_p+'paper2_figure_averages/output/'


    df = pd.read_pickle(FS().derived_p+'paper2_analytic_tables_to_yearly/output/'
                        +'LA_analytic_table_yearly.pkl')
    df = df[df.recip_race_g.isin([0,1])].copy()

    outcomes = ['pc_numdays', 'pharm_numclms', 'ed_avoid_numdays', 'all_cost_wins50']
    title_d = {'all_cost_wins50': 'Total Spending\n($ per enrollee per year)', 
               'pc_numdays': 'Primary Care Visits\n(per 100 enrollees per year)', 
               'pharm_numclms': 'Prescriptions\n(per 100 enrollees per year)', 
               'ed_avoid_numdays': 'Avoidable ED Visits\n(per 100 enrollees per year)'}
    color_d = {0: '#4781BE', 1: '#F2C463'}
    label_d = {0: 'Non-Hispanic White enrollees', 1: 'Non-Hispanic Black enrollees'}
    keeps = ([f'{x}1.plan_race_asgn' for x in range(1, 6)]
             + [f'{x}1.plan_race' for x in range(1, 6)])

    overall = df.groupby('recip_race_g')[[*title_d.keys()]].mean().stack().unstack(1)
    overall = overall.stack().rename('Overall\nPopulation')
    df = df.groupby(['recip_race_g', 'plan_id'])[[*title_d.keys()]].mean().stack().unstack(1)
    df = pd.concat([overall, df], axis=1)
    m = df.index.get_level_values(1) != 'all_cost_wins50'
    df.loc[m] *=100
    df.columns = ['Overall\nPopulation'] + [f'Plan {i}' for i in df.columns[1:].astype(int).astype(str)]

    fig, ax = plt.subplots(figsize=(10, 8), nrows=2, ncols=2)
    ax = ax.flatten()
    plt.subplots_adjust(hspace=0.2, wspace=0.27)
    for axis in ax:
        axis.tick_params(axis='both', which='major', labelsize=10.5)

    for i, outcome in enumerate(outcomes):
        ax[i].grid(color='#ABABAB', zorder=0, alpha=0.8, linestyle='--', axis='y')
        ax[i].set_title(title_d[outcome], fontsize=12)
        ax[i].axvline(0.5, 0, 1, color='black', linestyle='--')

        gp = df.xs(outcome, level=1)
        for sample, soff in zip([0, 1], [-1, 1]):
            gpp = gp.loc[sample]
            eb = ax[i].bar(gpp.index, height=gpp, width=0.25*soff, align='edge', zorder=2,
                           ec='k', color=color_d[sample], label=label_d[sample])


    handles, labels = ax[-1].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc='center', ncol=2,
                     bbox_to_anchor=(0.5, -0.01),fancybox=False, shadow=False)
    plt.tight_layout()
    plt.savefig(output_p+'plan_avg.png', dpi=300, bbox_extra_artists=[leg], bbox_inches='tight')
    plt.show()


def paper2_run_spending_sensitivities():
    """
    Balance check on LA
    """        
    output_p, temp_p, log_p, code_p = gen_paths(FS().analysis_p
                                                +'paper2_run_spending_sensitivities/')

    df = pd.read_pickle(FS().derived_p+'paper2_analytic_tables_to_yearly/output/'
                        +'LA_analytic_table_yearly.pkl')

    df = df[df.asgn_type.eq('AA') & df.recip_race_g.isin([0,1])]
    df.to_csv(temp_p+'AA.csv', chunksize=10**5, index=False)

    ws.stata_do(do_file=code_p+'paper2_spending_regs_RF_5plans.do',
                params=[temp_p+'AA.csv',
                        output_p+'AA_RF_results_5plans.csv'],
                log_path=log_p)

    # Delete temp file
    for file in ['AA']:
        os.remove(temp_p+f'{file}.csv')


def paper2_clean_spending_sensitivities():
    """
    
    """
    output_p = FS().analysis_p+'paper2_clean_spending_sensitivities/output/'

    paper_outcomes = ['adhd_numclms', 'all_cost_wins50',
                      'asthma_numclms', 'capch_all', 'ed_avoid_numdays',
                      'pc_numdays', 'pharm_numclms']

    # Get average among White enrollees in sample 
    dfs = pd.read_pickle(FS().derived_p+'paper2_analytic_tables_to_yearly/output/'
                        +'LA_analytic_table_yearly.pkl')
    dfs = dfs[dfs.recip_race_g.isin([0, 1])]

    for label, f, asgn_type in [('_5plan_spend', 'AA_RF_results_5plans.csv', 'AA')]:
        df = pd.read_csv(FS().analysis_p+'paper2_run_spending_sensitivities/output/'+f, sep='\t')
        white_avg = dfs.loc[dfs.recip_race_g.eq(0) & dfs.asgn_type.eq(asgn_type), 
                            df.outcome.unique()].mean()

        for reg, gp in df.groupby('regression'):
            pval = gp.loc[gp.variable.eq('plan)diff'), ['outcome', 'pval']].set_index('outcome')
            keeps = ([f'{x}1.plan_race_asgn' for x in range(1, 6)] 
                     + [f'{x}1.plan_race' for x in range(1, 6)] 
                     + ['51o.plan_race_asgn', '1.recip_race_g'])

            gp = gp[gp.variable.isin(keeps)]
            gp = gp.pivot(index='outcome', columns='variable',
                          values=['coeff', 'se', 'lb', 'ub']).apply(pd.to_numeric)
            gp['pval'] = pd.to_numeric(pval['pval'], errors='coerce')
            gp['pval_fmt'] = gp['pval'].apply(round_pval)

            sadj = pd.Series(multitest.multipletests(gp.reindex(paper_outcomes)['pval'], method='fdr_bh')[1],
                             index=paper_outcomes)
            gp['pval_adj'] = sadj
            gp['pval_adj_fmt'] = gp['pval_adj'].apply(round_pval)

            gp['white_avg'] = white_avg
            gp['range'] = gp.xs('coeff', axis=1).filter(like='plan_race').apply(np.ptp, axis=1)

            gp.to_csv(output_p+f'{reg}_results{label}.csv')


def paper2_figure_spend_dist():
    output_p = FS().analysis_p+'paper2_figure_spend_dist/output/'

    df = pd.read_pickle(FS().derived_p+'paper2_analytic_tables_to_yearly/output/'
                        +'LA_analytic_table_yearly.pkl')

    df = df[df.asgn_type.eq('AA') & df.recip_race_g.isin([0,1])]


    fix, ax = plt.subplots(figsize=(8,4), ncols=2)
    ax = ax.flatten()
    for axis in ax:
        axis.tick_params(axis='both', which='major', labelsize=10.5)
        
    xlim=15000
    xbin=150
    distbins = np.arange(0, xlim+xbin, xbin)
    fontsize=11
    title_fontsize=12


    s = df['all_cost']
    s.plot(kind='hist', ax=ax[0], ec='k', bins=distbins,
           weights=np.ones_like(s)*100./len(df),
           zorder=4)
    ax[0].set_xlabel('Spending ($)', fontsize=fontsize)
    ax[0].set_ylabel('Percent', fontsize=fontsize)
    ax[0].set_title('Panel A. All Spending', fontsize=title_fontsize)



    # Dist of log Spending + Normal dist fit. 
    slog = np.log(1+s[s.gt(0)])
    diff = 0.25 # Size of the bins for log plot
    x = np.arange(-10, 30, diff)
    np.log(1+s).plot(kind='hist', ax=ax[1], ec='k', bins=x,
                     weights=np.ones_like(s)*100./len(s),
                     zorder=4)
    ax[1].set_xlabel('Log($1 + Spending)', fontsize=fontsize)
    ax[1].set_title('Panel B. Log Transformation\nof Spending', fontsize=title_fontsize)
    ax[1].set_ylabel('Percent', fontsize=fontsize)
    ax[1].set_xlim(-1, 15)

    fit = norm.fit(slog)
    xmin, xmax = ax[1].get_xlim()
    p = norm.pdf(x, fit[0], fit[1])
    # Scale fit to be percent based.
    ax[1].plot(x, p*100*diff, 'k', linewidth=2, zorder=5)

    plt.tight_layout()
    plt.savefig(output_p+'spend_dist.png', dpi=300)
    plt.show()


def paper2_figure_planeff_RF():
    output_p = FS().analysis_p+'paper2_figure_planeff_RF/output/'

    outcomes = [ 
                ('pc_numdays', 'PC Utilization'), 
                ('pharm_numclms', 'Prescribing'),
                ('ed_avoid_numdays', ' Avoidable ED'),
                ('all_cost_wins50', 'Spending')]

    title_d = {'all_cost_wins50': 'Total Spending\n($ per enrollee per year)', 
               'pc_numdays': 'Primary Care Visits\n(per 100 enrollees per year)', 
               'pharm_numclms': 'Prescriptions\n(per 100 enrollees per year)', 
               'ed_avoid_numdays': 'Avoidable ED Visits\n(per 100 enrollees per year)'}

    df = pd.read_csv(FS().analysis_p+'paper2_plan_eff_and_by_grp/output/AA_RF_results.csv', sep='\t')
    pval = df.loc[df.variable.eq('plan)diff') & df.regression.eq('all'), 
                  ['outcome', 'pval']].set_index('outcome')
    pval['pval'] = pval['pval'].apply(pd.to_numeric).apply(round_pval)

    df = df[df.variable.str.contains('plan_id_asgn')]
    df = df[df.regression.eq('all') & df.outcome.isin([x[0] for x in outcomes])]
    df['grp'] = df['variable'].str.split('.').str[0].str.rstrip('b')
    df = df.set_index('grp')
    df[['coeff', 'se']] = df[['coeff', 'se']].apply(pd.to_numeric, errors='coerce')

    fig, ax = plt.subplots(figsize=(9, 9), nrows=2, ncols=2)
    plt.subplots_adjust(wspace=0.05, hspace=0.23)
    ax=ax.flatten()
    for axis in ax:
        axis.tick_params(axis='both', which='major', labelsize=11.5)

        
    for i, (outcome, label) in enumerate(outcomes):
        gp = df[df.outcome.eq(outcome)]
        if outcome != 'all_cost_wins50':
            gp = gp*100
        eb = ax[i].errorbar(x=gp['coeff'],
                            y=gp.index.astype(int), 
                            xerr=gp['se']*1.96,
                            zorder=3,
                            lw=0, elinewidth=2, capthick=2, marker='o',
                            capsize=3,
                            label=f'F-stat p-val: {pval.loc[outcome, "pval"]}')
        ax[i].grid(zorder=1)
        ax[i].legend(fontsize=9)
        ax[i].locator_params(axis='y', nbins=5) 
        ax[i].set_title(title_d[outcome], fontsize=12)
        
        xlim = np.max(np.abs(ax[i].get_xlim()))
        ax[i].set_xlim(-xlim, xlim)
        
        if i % 2 == 1:
            ax[i].set_yticklabels([None])
        else: 
            ax[i].set_yticklabels([''] + [f'Plan {i}' for i in gp.index])    
        
    plt.tight_layout()
    plt.savefig(output_p+'RF_plan_effects.png', dpi=300)
    plt.show()    


def paper2_figure_first_stage():
    output_p = FS().analysis_p+'paper2_figure_first_stage/output/'

    df = pd.read_pickle(FS().derived_p+'paper2_analytic_tables_to_yearly/output/'
                        +'LA_analytic_table_yearly.pkl')
    df = df[df.asgn_type.eq('AA') & df.recip_race_g.isin([0,1])]

    df['in_asgn'] = df.plan_id_asgn == df.plan_id

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.tick_params(axis='both', which='major', labelsize=11.5)
    s = df.groupby('plan_id_asgn').in_asgn.mean().mul(100)
    s.index = 'Plan ' + s.index.astype(int).astype(str)

    s.plot(kind='bar', ax=ax, ec='k', rot=0, zorder=2)
    ax.set_xlabel(None)
    ax.set_ylabel('Share Remaining\nin Assigned Plan', fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', zorder=-2, )

    plt.tight_layout()
    plt.savefig(output_p+'first_stage.png', dpi=300)
    plt.show()


def paper2_run_main_regressions_AAs_resampled(rng=(0,1000)):
    """
    Run the main paper regressions, but with the AAs resampled down to the same plan distribution
    as the active choosers, based on the chosen plan as of 2012 Runs 1000 resamples.

    Because there are many regressions to run, this only runs the 5 plan RF identification for the
    AAs, and at the time of writing this will only use the age/sex/elig adjustments. 

    We resample with replacement, so the same AA recip can be included multiple times. 
    """        
    output_p, temp_p, log_p, code_p = gen_paths(FS().analysis_p
                                                +'paper2_run_main_regressions_AAs_resampled/')

    df = pd.read_pickle(FS().derived_p+'paper2_analytic_tables_to_yearly/output/'
                        +'LA_analytic_table_yearly.pkl')
    dfch = df[df.asgn_type.eq('Choice') & df.recip_race_g.isin([0,1])]
    # Chooser Distribution, base it on 2012. 
    nCH = dfch[dfch.year.eq(2012)].plan_id.value_counts()


    df = df[df.asgn_type.eq('AA') & df.recip_race_g.isin([0,1])]

    for i in range(*rng):
        # Set a deterministic seed, but also one that is different for each resample. 
        np.random.seed(i*6)

        # Randomly resample same share in each plan based on 2013 Choosers. 
        recips = []
        for plan, N in nCH.iteritems():
            recips.extend(
                    np.random.choice(df.loc[df.plan_id_asgn.eq(plan), 'recip_id'].unique(), N,
                                     replace=True).tolist())

        # Because replace=True need to merge to duplicate people who are included twice. 
        gp = df.merge(pd.Series(recips, name='recip_id'))

        gp.to_csv(temp_p+f'AA_{i}.csv', index=False)

        ws.stata_do(do_file=code_p+'paper2_main_regs_RF_5plans.do',
                    params=[temp_p+f'AA_{i}.csv',
                            output_p+f'AA_RF_results_5plans_run{i}.csv'],
                    log_path=log_p)

        os.remove(temp_p+f'AA_{i}.csv')


def paper2_run_main_regressions_fine_age():
    """
    Run the main paper regressions
    """        
    output_p, temp_p, log_p, code_p = gen_paths(FS().analysis_p
                                                +'paper2_run_main_regressions_fine_age/')

    df = pd.read_pickle(FS().derived_p+'paper2_analytic_tables_to_yearly/output/'
                        +'LA_analytic_table_yearly.pkl')

    #####
    ##### Redefine age controls to be age in years
    #####
    df['recip_age_g'] = (df.recip_age//1).astype(int)

    dfch = df[df.asgn_type.eq('Choice') & df.recip_race_g.isin([0,1])]
    dfch.to_csv(temp_p+'Choice.csv', chunksize=10**5, index=False)

    df = df[df.asgn_type.eq('AA') & df.recip_race_g.isin([0,1])]
    df.to_csv(temp_p+'AA.csv', chunksize=10**5, index=False)


    ws.stata_do(do_file=code_p+'paper2_main_regs_RF_5plans.do',
                params=[temp_p+'AA.csv',
                        output_p+'AA_RF_results_5plans.csv'],
                log_path=log_p)

    ws.stata_do(do_file=code_p+'paper2_AC_regs_5plans.do',
                params=[temp_p+'Choice.csv',
                        output_p+'AC_results_5plans.csv'],
                log_path=log_p)

    # Delete temp file
    for file in ['AA', 'Choice']:
        os.remove(temp_p+f'{file}.csv')


def paper2_clean_main_reg_results_fine_age():
    """
    
    """
    output_p = FS().analysis_p+'paper2_clean_main_regression_results_fine_age/output/'

    paper_outcomes = ['all_cost_wins50', 'ed_avoid_numdays', 'pc_numdays', 'pharm_numclms']

    # Get average among White enrollees in sample 
    dfs = pd.read_pickle(FS().derived_p+'paper2_analytic_tables_to_yearly/output/'
                        +'LA_analytic_table_yearly.pkl')
    dfs = dfs[dfs.recip_race_g.isin([0, 1])]

    for label, f, asgn_type in [ ('_5plan', 'AA_RF_results_5plans.csv', 'AA'),
                                ('_AC_5plan', 'AC_results_5plans.csv', 'Choice')]:
        df = pd.read_csv(FS().analysis_p+'paper2_run_main_regressions_fine_age/output/'+f, sep='\t')
        white_avg = dfs.loc[dfs.recip_race_g.eq(0) & dfs.asgn_type.eq(asgn_type), 
                            df.outcome.unique()].mean()

        for reg, gp in df.groupby('regression'):
            pval = gp.loc[gp.variable.eq('plan)diff'), ['outcome', 'pval']].set_index('outcome')
            if f != 'AA_IV_results_5plans.csv':
                keeps = ([f'{x}1.plan_race_asgn' for x in range(1, 6)] 
                         + [f'{x}1.plan_race' for x in range(1, 6)] 
                         + ['51o.plan_race_asgn', '1.recip_race_g'])
            else:
                keeps = ['1b.plan_id#1.recip_race_g', '2.plan_id#1.recip_race_g',
                         '3.plan_id#1.recip_race_g', '4.plan_id#1.recip_race_g',
                         '5.plan_id#1.recip_race_g']
            gp = gp[gp.variable.isin(keeps)]
            gp = gp.pivot(index='outcome', columns='variable',
                          values=['coeff', 'se', 'lb', 'ub']).apply(pd.to_numeric)
            gp['pval'] = pd.to_numeric(pval['pval'], errors='coerce')
            gp['pval_fmt'] = gp['pval'].apply(round_pval)

            sadj = pd.Series(multitest.multipletests(gp.reindex(paper_outcomes)['pval'], method='fdr_bh')[1],
                             index=paper_outcomes)
            gp['pval_adj'] = sadj
            gp['pval_adj_fmt'] = gp['pval_adj'].apply(round_pval)

            gp['white_avg'] = white_avg
            if f != 'AA_IV_results_5plans.csv':
                gp['range'] = gp.xs('coeff', axis=1).filter(like='plan_race').apply(np.ptp, axis=1)
            else:
                gp['range'] = gp.xs('coeff', axis=1).filter(like='plan_id#').apply(np.ptp, axis=1)
            gp.to_csv(output_p+f'{reg}_results{label}.csv')


############################################
#### MAIN FIGURES
############################################
def paper2_figure1_demeaned_barplot():
    """
    Figure of sorting in observational sample, demeaned on age, sex and UoR
    """
    output_p = FS().analysis_p+'paper2_figure1_demeaned_barplot/output/'


    df = pd.read_pickle(FS().derived_p+'paper2_analytic_tables_to_yearly/output/'
                        +'LA_analytic_table_yearly.pkl')
    df = df[df.recip_race_g.isin([0, 1])].copy()

    outcomes = ['pc_numdays', 'pharm_numclms', 'ed_avoid_numdays', 'all_cost_wins50']
    title_d = {'all_cost_wins50': 'Total Spending\n($ per enrollee per year)', 
               'pc_numdays': 'Primary Care Visits\n(per 100 enrollees per year)', 
               'pharm_numclms': 'Prescriptions\n(per 100 enrollees per year)', 
               'ed_avoid_numdays': 'Low-acuity ED visits\n(per 100 enrollees per year)',
               'ednoadmit_numdays': 'ED Visits\n(per 100 enrollees per year)'}
    color_d = {'White': '#4781BE', 'Black': '#F2C463'}
    label_d = {'White': 'Non-Hispanic White enrollees', 'Black': 'Non-Hispanic Black enrollees'}
    height=0.6

    ## Prep dataframe
    r = {}
    # Only keep Active Choosers
    df = df[df.asgn_type.eq('Choice')]
    plan = 'plan_id'
    # Manually create the plan-race dummies, so we can de-mean each separately
    dummies = (pd.get_dummies(df[plan].astype(int).astype(str) 
                              + '_' + df['recip_race'].str.lower()).add_prefix('plan_'))
    df = (pd.concat([df, dummies],axis=1))
    # Manually create the race dummies, so we can de-mean each separately
    dummies = (pd.get_dummies(df['recip_race'].str.lower()))
    df = (pd.concat([df, dummies],axis=1))

    ## Regressions - By plan and by race
    results = {} 
    for y in outcomes: 
        # De-mean the outcome and each of the 10 dummy columns, add back in overall mean
        FEs = [f'plan_{i}_{x}' for i in range(1,6) for x in ['black', 'white']]
        df[FEs] = df[FEs].astype(int)
        cols = [y] + FEs
        d = {}
        for col in cols:
            res = smf.ols(f'{col} ~ C(uor) + C(recip_age_g) + C(recip_gender_g) -1', data=df).fit()
            d[col] = df[col] - res.fittedvalues + df[col].mean()
        # Now run a regression with the de-meaned dummies as separate effects, the coefficient on those will become the 
        # plan effects interacted with race, net of the controls
        res = smf.ols(f'{y} ~ {"+".join(FEs)} - 1', data=pd.concat(d, axis=1)).fit()
        results[y] = res.params
    # Transform results into dataframe
    res = pd.DataFrame.from_dict(results).reset_index()
    res['plan'] = res['index'].str[:6].str.title().str.replace('_', ' ')
    res['race'] = res['index'].str[6:].str.title().str.replace('_', ' ').str.strip()
    r['byplan'] = (res.drop(columns='index')
                      .melt(['race', 'plan'], var_name='outcomes', value_name='')
                      .pivot(index=['race', 'outcomes'], columns='plan')
                      .sort_index())
    r['byplan'].columns = r['byplan'].columns.get_level_values(1)

    ## Regressions - By race
    results = {} 
    for y in outcomes: 
        # De-mean the outcome and each of the 2 dummy columns, add back in overall mean
        FEs = ['black', 'white']
        df[FEs] = df[FEs].astype(int)
        cols = [y] + FEs
        d = {}
        for col in cols:
            res = smf.ols(f'{col} ~ C(uor) + C(recip_age_g) + C(recip_gender_g) -1', data=df).fit()
            d[col] = df[col] - res.fittedvalues + df[col].mean()

        # Now run a regression with the de-meaned dummies as separate effects, the coefficient on those will become the 
        # plan effects interacted with race, net of the controls
        res = smf.ols(f'{y} ~ {"+".join(FEs)} - 1', data=pd.concat(d, axis=1)).fit()
        results[y] = res.params
    # Transform results into dataframe
    res = pd.DataFrame.from_dict(results).reset_index()
    res['race'] = res['index'].str.title()
    r['overall'] = (res.drop(columns='index')
                       .melt(['race'], var_name='outcomes', value_name='')
                       .set_index(['race', 'outcomes'])
                       .sort_index())
    r['overall'].columns = ['Overall']
    r['overall'].columns.names = ['plan']


    ## Concatenate all of the results and plot barplots
    # Concat and prepare
    res = pd.concat(r, axis=1)
    res.columns = res.columns.get_level_values(1)
    # Multiply by 100
    m = res.index.get_level_values(1) != 'all_cost_wins50'
    res.loc[m] *=100
    # Organize columns
    res = res[['Overall']+[c for c in res.columns if 'Plan' in c]]

    ## Plot
    fig, ax = plt.subplots(figsize=(10, 8), nrows=2, ncols=2)
    ax = ax.flatten()
    plt.subplots_adjust(hspace=0.2, wspace=0.27)
    for axis in ax:
        axis.tick_params(axis='both', which='major', labelsize=10.5)

    for i, outcome in enumerate(outcomes):
        ax[i].grid(color='#ABABAB', zorder=0, alpha=0.8, linestyle='--', axis='y')
        ax[i].set_title(title_d[outcome], fontsize=12)
        ax[i].axvline(0.5, 0, 1, color='black', linestyle='--')

        gp = res.xs(outcome, level=1)
        ymin = gp.min().min()*height
        for sample, soff in zip(['Black', 'White'], [-1, 1]):
            gpp = gp.loc[sample]
            eb = ax[i].bar(gpp.index, height=gpp, width=0.25*soff, align='edge', zorder=2,
                           ec='k', color=color_d[sample], label=label_d[sample])
        ax[i].set_ylim(ymin)

    handles, labels = ax[-1].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc='center', ncol=2,
                     bbox_to_anchor=(0.5, -0.01),fancybox=False, shadow=False)
    plt.tight_layout()
    plt.savefig(output_p+'Figure2.eps', dpi=300, bbox_extra_artists=[leg], bbox_inches='tight')
    plt.savefig(output_p+'plan_demeaned.png', dpi=300, bbox_extra_artists=[leg], bbox_inches='tight')
    plt.show()


def paper2_figure1_demeaned_barplot_AAs():
    output_p = FS().analysis_p+'paper2_figure1_demeaned_barplot_AAs/output/'


    df = pd.read_pickle(FS().derived_p+'paper2_analytic_tables_to_yearly/output/'
                        +'LA_analytic_table_yearly.pkl')
    df = df[df.recip_race_g.isin([0, 1])].copy()

    outcomes = ['pc_numdays', 'pharm_numclms', 'ed_avoid_numdays', 'all_cost_wins50']
    title_d = {'all_cost_wins50': 'Total Spending\n($ per enrollee per year)', 
               'pc_numdays': 'Primary Care Visits\n(per 100 enrollees per year)', 
               'pharm_numclms': 'Prescriptions\n(per 100 enrollees per year)', 
               'ed_avoid_numdays': 'Low-acuity ED visits\n(per 100 enrollees per year)',
               'ednoadmit_numdays': 'ED Visits\n(per 100 enrollees per year)'}
    color_d = {'White': '#4781BE', 'Black': '#F2C463'}
    label_d = {'White': 'Non-Hispanic White enrollees', 'Black': 'Non-Hispanic Black enrollees'}
    height=0.6

    ## Prep dataframe
    r = {}
    # Only keep Active Choosers
    df = df[df.asgn_type.eq('AA')]
    plan = 'plan_id_asgn'
    # Manually create the plan-race dummies, so we can de-mean each separately
    dummies = (pd.get_dummies(df[plan].astype(int).astype(str) 
                              + '_' + df['recip_race'].str.lower()).add_prefix('plan_'))
    df = (pd.concat([df, dummies],axis=1))
    # Manually create the race dummies, so we can de-mean each separately
    dummies = (pd.get_dummies(df['recip_race'].str.lower()))
    df = (pd.concat([df, dummies],axis=1))

    ## Regressions - By plan and by race
    results = {} 
    for y in outcomes: 
        # De-mean the outcome and each of the 10 dummy columns, add back in overall mean
        FEs = [f'plan_{i}_{x}' for i in range(1,6) for x in ['black', 'white']]
        df[FEs] = df[FEs].astype(int)
        cols = [y] + FEs
        d = {}
        for col in cols:
            res = smf.ols(f'{col} ~ C(uor) + C(recip_age_g) + C(recip_gender_g) -1', data=df).fit()
            d[col] = df[col] - res.fittedvalues + df[col].mean()
        # Now run a regression with the de-meaned dummies as separate effects, the coefficient on those will become the 
        # plan effects interacted with race, net of the controls
        res = smf.ols(f'{y} ~ {"+".join(FEs)} - 1', data=pd.concat(d, axis=1)).fit()
        results[y] = res.params
    # Transform results into dataframe
    res = pd.DataFrame.from_dict(results).reset_index()
    res['plan'] = res['index'].str[:6].str.title().str.replace('_', ' ')
    res['race'] = res['index'].str[6:].str.title().str.replace('_', ' ').str.strip()
    r['byplan'] = (res.drop(columns='index')
                      .melt(['race', 'plan'], var_name='outcomes', value_name='')
                      .pivot(index=['race', 'outcomes'], columns='plan')
                      .sort_index())
    r['byplan'].columns = r['byplan'].columns.get_level_values(1)

    ## Regressions - By race
    results = {} 
    for y in outcomes: 
        # De-mean the outcome and each of the 2 dummy columns, add back in overall mean
        FEs = ['black', 'white']
        df[FEs] = df[FEs].astype(int)
        cols = [y] + FEs
        d = {}
        for col in cols:
            res = smf.ols(f'{col} ~ C(uor) + C(recip_age_g) + C(recip_gender_g) -1', data=df).fit()
            d[col] = df[col] - res.fittedvalues + df[col].mean()

        # Now run a regression with the de-meaned dummies as separate effects, the coefficient on those will become the 
        # plan effects interacted with race, net of the controls
        res = smf.ols(f'{y} ~ {"+".join(FEs)} - 1', data=pd.concat(d, axis=1)).fit()
        results[y] = res.params
    # Transform results into dataframe
    res = pd.DataFrame.from_dict(results).reset_index()
    res['race'] = res['index'].str.title()
    r['overall'] = (res.drop(columns='index')
                       .melt(['race'], var_name='outcomes', value_name='')
                       .set_index(['race', 'outcomes'])
                       .sort_index())
    r['overall'].columns = ['Overall']
    r['overall'].columns.names = ['plan']


    ## Concatenate all of the results and plot barplots
    # Concat and prepare
    res = pd.concat(r, axis=1)
    res.columns = res.columns.get_level_values(1)
    # Multiply by 100
    m = res.index.get_level_values(1) != 'all_cost_wins50'
    res.loc[m] *=100
    # Organize columns
    res = res[['Overall']+[c for c in res.columns if 'Plan' in c]]

    ## Plot
    fig, ax = plt.subplots(figsize=(10, 8), nrows=2, ncols=2)
    ax = ax.flatten()
    plt.subplots_adjust(hspace=0.2, wspace=0.27)
    for axis in ax:
        axis.tick_params(axis='both', which='major', labelsize=10.5)

    for i, outcome in enumerate(outcomes):
        ax[i].grid(color='#ABABAB', zorder=0, alpha=0.8, linestyle='--', axis='y')
        ax[i].set_title(title_d[outcome], fontsize=12)
        ax[i].axvline(0.5, 0, 1, color='black', linestyle='--')

        gp = res.xs(outcome, level=1)
        ymin = gp.min().min()*height
        for sample, soff in zip(['Black', 'White'], [-1, 1]):
            gpp = gp.loc[sample]
            eb = ax[i].bar(gpp.index, height=gpp, width=0.25*soff, align='edge', zorder=2,
                           ec='k', color=color_d[sample], label=label_d[sample])
        ax[i].set_ylim(ymin)

    handles, labels = ax[-1].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc='center', ncol=2,
                     bbox_to_anchor=(0.5, -0.01),fancybox=False, shadow=False)
    plt.tight_layout()
    plt.savefig(output_p+'Figure2_AA.eps', dpi=300, bbox_extra_artists=[leg], bbox_inches='tight')
    plt.savefig(output_p+'plan_demeaned_AA.png', dpi=300, bbox_extra_artists=[leg], bbox_inches='tight')
    plt.show()


def paper2_figure1_demeaned_barplot_both():
    output_p = FS().analysis_p+'paper2_figure1_demeaned_barplot_both/output/'


    df = pd.read_pickle(FS().derived_p+'paper2_analytic_tables_to_yearly/output/'
                        +'LA_analytic_table_yearly.pkl')
    df = df[df.recip_race_g.isin([0, 1])].copy()

    outcomes = ['pc_numdays', 'pharm_numclms', 'ed_avoid_numdays', 'all_cost_wins50']
    title_d = {'all_cost_wins50': 'Total Spending\n($ per enrollee per year)', 
               'pc_numdays': 'Primary Care Visits\n(per 100 enrollees per year)', 
               'pharm_numclms': 'Prescriptions\n(per 100 enrollees per year)', 
               'ed_avoid_numdays': 'Low-acuity ED visits\n(per 100 enrollees per year)',
               'ednoadmit_numdays': 'ED Visits\n(per 100 enrollees per year)'}
    color_d = {'White': '#354850', 'Black': '#D5E0EC'}
    label_d = {'White': 'Non-Hispanic White enrollees', 'Black': 'Non-Hispanic Black enrollees'}
    height=0.6

    ## Prep dataframe
    r = {}
    # Only keep Active Choosers
    df = df[df.asgn_type.eq('Choice')]
    plan = 'plan_id'
    # Manually create the plan-race dummies, so we can de-mean each separately
    dummies = (pd.get_dummies(df[plan].astype(int).astype(str) 
                              + '_' + df['recip_race'].str.lower()).add_prefix('plan_'))
    df = (pd.concat([df, dummies],axis=1))
    # Manually create the race dummies, so we can de-mean each separately
    dummies = (pd.get_dummies(df['recip_race'].str.lower()))
    df = (pd.concat([df, dummies],axis=1))

    ## Regressions - By plan and by race
    results = {} 
    for y in outcomes: 
        # De-mean the outcome and each of the 10 dummy columns, add back in overall mean
        FEs = [f'plan_{i}_{x}' for i in range(1,6) for x in ['black', 'white']]
        df[FEs] = df[FEs].astype(int)
        cols = [y] + FEs
        d = {}
        for col in cols:
            res = smf.ols(f'{col} ~ C(uor) + C(recip_age_g) + C(recip_gender_g) -1', data=df).fit()
            d[col] = df[col] - res.fittedvalues + df[col].mean()
        # Now run a regression with the de-meaned dummies as separate effects, the coefficient on those will become the 
        # plan effects interacted with race, net of the controls
        res = smf.ols(f'{y} ~ {"+".join(FEs)} - 1', data=pd.concat(d, axis=1)).fit()
        results[y] = res.params
    # Transform results into dataframe
    res = pd.DataFrame.from_dict(results).reset_index()
    res['plan'] = res['index'].str[:6].str.title().str.replace('_', ' ')
    res['race'] = res['index'].str[6:].str.title().str.replace('_', ' ').str.strip()
    r['byplan'] = (res.drop(columns='index')
                      .melt(['race', 'plan'], var_name='outcomes', value_name='')
                      .pivot(index=['race', 'outcomes'], columns='plan')
                      .sort_index())
    r['byplan'].columns = r['byplan'].columns.get_level_values(1)

    ## Regressions - By race
    results = {} 
    for y in outcomes: 
        # De-mean the outcome and each of the 2 dummy columns, add back in overall mean
        FEs = ['black', 'white']
        df[FEs] = df[FEs].astype(int)
        cols = [y] + FEs
        d = {}
        for col in cols:
            res = smf.ols(f'{col} ~ C(uor) + C(recip_age_g) + C(recip_gender_g) -1', data=df).fit()
            d[col] = df[col] - res.fittedvalues + df[col].mean()

        # Now run a regression with the de-meaned dummies as separate effects, the coefficient on those will become the 
        # plan effects interacted with race, net of the controls
        res = smf.ols(f'{y} ~ {"+".join(FEs)} - 1', data=pd.concat(d, axis=1)).fit()
        results[y] = res.params
    # Transform results into dataframe
    res = pd.DataFrame.from_dict(results).reset_index()
    res['race'] = res['index'].str.title()
    r['overall'] = (res.drop(columns='index')
                       .melt(['race'], var_name='outcomes', value_name='')
                       .set_index(['race', 'outcomes'])
                       .sort_index())
    r['overall'].columns = ['Overall']
    r['overall'].columns.names = ['plan']


    ## Concatenate all of the results and plot barplots
    # Concat and prepare
    res_c = pd.concat(r, axis=1)
    res_c.columns = res_c.columns.get_level_values(1)
    # Multiply by 100
    m = res_c.index.get_level_values(1) != 'all_cost_wins50'
    res_c.loc[m] *=100
    # Organize columns
    res_c = res_c[['Overall']+[c for c in res_c.columns if 'Plan' in c]]


    # Auto Assignees
    df = pd.read_pickle(FS().derived_p+'paper2_analytic_tables_to_yearly/output/'
                        +'LA_analytic_table_yearly.pkl')
    df = df[df.recip_race_g.isin([0, 1])].copy()

    ## Prep dataframe
    r = {}
    # Only keep Active Choosers
    df = df[df.asgn_type.eq('AA')]
    plan = 'plan_id_asgn'
    # Manually create the plan-race dummies, so we can de-mean each separately
    dummies = (pd.get_dummies(df[plan].astype(int).astype(str) 
                              + '_' + df['recip_race'].str.lower()).add_prefix('plan_'))
    df = (pd.concat([df, dummies],axis=1))
    # Manually create the race dummies, so we can de-mean each separately
    dummies = (pd.get_dummies(df['recip_race'].str.lower()))
    df = (pd.concat([df, dummies],axis=1))

    ## Regressions - By plan and by race
    results = {} 
    for y in outcomes: 
        # De-mean the outcome and each of the 10 dummy columns, add back in overall mean
        FEs = [f'plan_{i}_{x}' for i in range(1,6) for x in ['black', 'white']]
        df[FEs] = df[FEs].astype(int)
        cols = [y] + FEs
        d = {}
        for col in cols:
            res = smf.ols(f'{col} ~ C(uor) + C(recip_age_g) + C(recip_gender_g) -1', data=df).fit()
            d[col] = df[col] - res.fittedvalues + df[col].mean()
        # Now run a regression with the de-meaned dummies as separate effects, the coefficient on those will become the 
        # plan effects interacted with race, net of the controls
        res = smf.ols(f'{y} ~ {"+".join(FEs)} - 1', data=pd.concat(d, axis=1)).fit()
        results[y] = res.params
    # Transform results into dataframe
    res = pd.DataFrame.from_dict(results).reset_index()
    res['plan'] = res['index'].str[:6].str.title().str.replace('_', ' ')
    res['race'] = res['index'].str[6:].str.title().str.replace('_', ' ').str.strip()
    r['byplan'] = (res.drop(columns='index')
                      .melt(['race', 'plan'], var_name='outcomes', value_name='')
                      .pivot(index=['race', 'outcomes'], columns='plan')
                      .sort_index())
    r['byplan'].columns = r['byplan'].columns.get_level_values(1)

    ## Regressions - By race
    results = {} 
    for y in outcomes: 
        # De-mean the outcome and each of the 2 dummy columns, add back in overall mean
        FEs = ['black', 'white']
        df[FEs] = df[FEs].astype(int)
        cols = [y] + FEs
        d = {}
        for col in cols:
            res = smf.ols(f'{col} ~ C(uor) + C(recip_age_g) + C(recip_gender_g) -1', data=df).fit()
            d[col] = df[col] - res.fittedvalues + df[col].mean()

        # Now run a regression with the de-meaned dummies as separate effects, the coefficient on those will become the 
        # plan effects interacted with race, net of the controls
        res = smf.ols(f'{y} ~ {"+".join(FEs)} - 1', data=pd.concat(d, axis=1)).fit()
        results[y] = res.params
    # Transform results into dataframe
    res = pd.DataFrame.from_dict(results).reset_index()
    res['race'] = res['index'].str.title()
    r['overall'] = (res.drop(columns='index')
                       .melt(['race'], var_name='outcomes', value_name='')
                       .set_index(['race', 'outcomes'])
                       .sort_index())
    r['overall'].columns = ['Overall']
    r['overall'].columns.names = ['plan']


    ## Concatenate all of the results and plot barplots
    # Concat and prepare
    res = pd.concat(r, axis=1)
    res.columns = res.columns.get_level_values(1)
    # Multiply by 100
    m = res.index.get_level_values(1) != 'all_cost_wins50'
    res.loc[m] *=100
    # Organize columns
    res = res[['Overall']+[c for c in res.columns if 'Plan' in c]]

    color_d = {'White': '#E9D9A3', 'Black': '#D5E0EC'}

    for outcome in outcomes:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True, width_ratios=[1, 1])
        fig.subplots_adjust(wspace=0.14)  # adjust space between Axes

        ax1.tick_params(labelsize=13)
        ax2.tick_params(labelsize=13)
        gp = res.xs(outcome, level=1)
        gp_c = res_c.xs(outcome, level=1)

        c_min = gp_c.min().min()*0.75
        c_max = gp_c.max().max()*1.04

        a_min = gp.min().min()*0.75
        a_max = gp.max().max()*1.04
        
        escale = 2.26
        capsize=2.5
        for sample, soff in zip(['Black', 'White'], [-1, 1]):
            gpp = gp.loc[sample]
            gpp_c = gp_c.loc[sample]

            bar_diff = gp.loc['Black'] - gp.loc['White']
            bar_diff_c = gp_c.loc['Black'] - gp_c.loc['White']

            # Left Figure
            ax1.bar(gpp_c.index, gpp_c*1, width=0.25*soff, align='edge', zorder=23,
                    ec=None, color=color_d[sample])

            if (bar_diff_c.mean() < 0) & (sample == 'Black'):
                l, caps, c = ax1.errorbar(x=np.arange(len(gpp_c.index))-0.25/2, y=gp_c.loc['Black']+np.abs(bar_diff_c)/2, 
                                          yerr=np.abs(bar_diff_c)/escale, lw=0, elinewidth=1.5, capthick=1.5, capsize=capsize, 
                                          color='#354850', zorder=44)
                for ci, cap in enumerate(caps):
                    if ci==0:
                        cap.set_marker("v")
                    if ci==1:
                        cap.set_marker("^")

            if (bar_diff_c.mean() > 0) & (sample == 'White'):
                l, caps, c = ax1.errorbar(x=np.arange(len(gpp_c.index))+0.25/2, y=gp_c.loc['White']+np.abs(bar_diff_c)/2, 
                             yerr=np.abs(bar_diff_c)/escale, lw=0, elinewidth=1.5, capthick=1.5, capsize=capsize,
                             color='#354850', zorder=44)    
                for ci, cap in enumerate(caps):
                    if ci==0:
                        cap.set_marker("v")
                    if ci==1:
                        cap.set_marker("^")
                
                
            # Right Figure
            ax2.bar(gpp.index, gpp, width=0.25*soff, align='edge', zorder=23,
                     ec=None, color=color_d[sample], label=label_d[sample])

            if (bar_diff.mean() < 0) & (sample == 'Black'):
                l, caps, c = ax2.errorbar(x=np.arange(len(gpp.index))-0.25/2, y=gp.loc['Black']+np.abs(bar_diff)/2, 
                                          yerr=np.abs(bar_diff)/escale, lw=0, elinewidth=1.5, capthick=1.5, capsize=capsize,
                                          color='#354850', zorder=44, label='Racial difference')
                for ci, cap in enumerate(caps):
                    if ci==0:
                        cap.set_marker("v")
                    if ci==1:
                        cap.set_marker("^")
                
            if (bar_diff.mean() > 0) & (sample == 'White'):
                l, caps, c = ax2.errorbar(x=np.arange(len(gpp.index))+0.25/2, y=gp.loc['White']+np.abs(bar_diff)/2, 
                                          yerr=np.abs(bar_diff)/escale, lw=0, elinewidth=1.5, capthick=1.5, capsize=capsize,
                                          color='#354850', zorder=44, label='Racial difference')  
                for ci, cap in enumerate(caps):
                    if ci==0:
                        cap.set_marker("v")
                    if ci==1:
                        cap.set_marker("^")

            if outcome == 'pc_numdays':
                #ax2.text(x=-0.4, y=475, s='Racial difference', ha='left', fontsize=14, color='Gold')
                ax2.annotate('Racial difference', 
                             xy=(0.82, 350),
                             xytext=(-0.4, 475),
                             color='#354850',
                             fontsize=12,
                             arrowprops=dict( arrowstyle="->" , color='#354850'))
                
            p_min = min(a_min, c_min)
            p_max = max(a_max, c_max)
            ax1.set_ylim(-p_max, -p_min)          # Obs. outliers
            ax2.set_ylim(p_min, p_max)            # AA outliers

            ax2.set_title('Randomized', ha='center', size=14)
            ax1.set_title('Observational', ha='center', size=14)
            ax1.grid(axis='y', color='#2c2c2c', zorder=-11, alpha=0.2)
            ax2.grid(axis='y', color='#2c2c2c', zorder=-11, alpha=0.2)
            
        fig.savefig(output_p+f'{outcome}_both.pdf', dpi=300, bbox_inches='tight')
        handles, labels = ax2.get_legend_handles_labels()
        
        leg = fig.legend(handles, labels, loc='center', ncol=2,
                         bbox_to_anchor=(0.5, -0.01),fancybox=False, shadow=False) 

        def export_legend(legend, filename=output_p+"legend.pdf", expand=[-5,-5,5,5]):
            fig  = legend.figure
            fig.canvas.draw()
            bbox  = legend.get_window_extent()
            bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
            bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(filename, dpi="figure", bbox_inches=bbox)

        export_legend(leg)
        plt.show()



    # for outcome in outcomes:
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True, width_ratios=[1, 1])
    #     fig.subplots_adjust(wspace=0.14)  # adjust space between Axes

    #     ax1.tick_params(labelsize=13)
    #     ax2.tick_params(labelsize=13)
    #     gp = res.xs(outcome, level=1)
    #     gp_c = res_c.xs(outcome, level=1)

    #     c_min = gp_c.min().min()*0.75
    #     c_max = gp_c.max().max()*1.04

    #     a_min = gp.min().min()*0.75
    #     a_max = gp.max().max()*1.04
    #     for sample, soff in zip(['Black', 'White'], [-1, 1]):
    #         gpp = gp.loc[sample]
    #         gpp_c = gp_c.loc[sample]

    #         bar_diff = gp.loc['Black'] - gp.loc['White']
    #         bar_diff_c = gp_c.loc['Black'] - gp_c.loc['White']

    #         # Left Figure
    #         ax1.bar(gpp_c.index, gpp_c*1, width=0.25*soff, align='edge', zorder=23,
    #                 ec='k', color=color_d[sample])

    #         if (bar_diff_c.mean() < 0) & (sample == 'Black'):
    #             ax1.bar(gpp_c.index, bar_diff_c, width=-0.25, align='edge', zorder=14,
    #                     ec='k', color='#E9D9A3', bottom=gp_c.loc['White'])

    #         if (bar_diff_c.mean() > 0) & (sample == 'White'):
    #             ax1.bar(gpp_c.index, bar_diff_c*-1, width=0.25, align='edge', zorder=14,
    #                     ec='k', color='#E9D9A3', bottom=gp_c.loc['Black'])            


    #         # Right Figure
    #         ax2.bar(gpp.index, gpp, width=0.25*soff, align='edge', zorder=2,
    #                  ec='k', color=color_d[sample], label=label_d[sample])

    #         if (bar_diff.mean() < 0) & (sample == 'Black'):
    #             ax2.bar(gpp.index, bar_diff*-1, width=-0.25, align='edge', zorder=4,
    #                     ec='k', color='#E9D9A3', bottom=gp.loc['Black'], label='Racial difference')
                
    #         if (bar_diff.mean() > 0) & (sample == 'White'):
    #             ax2.bar(gpp.index, bar_diff, width=0.25, align='edge', zorder=4,
    #                     ec='k', color='#E9D9A3', bottom=gp.loc['White'], label='Racial difference')           

                
    #         p_min = min(a_min, c_min)
    #         p_max = max(a_max, c_max)
    #         ax1.set_ylim(-p_max, -p_min)          # Obs. outliers
    #         ax2.set_ylim(p_min, p_max)            # AA outliers

    #         ax2.set_title('Randomized', ha='center', size=14)
    #         ax1.set_title('Observational', ha='center', size=14)
    #         ax1.grid(axis='y', color='#2c2c2c', zorder=-11, alpha=0.2)
    #         ax2.grid(axis='y', color='#2c2c2c', zorder=-11, alpha=0.2)
            
    #     fig.savefig(output_p+f'{outcome}_both.pdf', dpi=300, bbox_inches='tight')
    #     handles, labels = ax2.get_legend_handles_labels()
    #     leg = fig.legend(handles, labels, loc='center', ncol=2,
    #                      bbox_to_anchor=(0.5, -0.01),fancybox=False, shadow=False) 

    #     def export_legend(legend, filename=output_p+"legend.pdf", expand=[-5,-5,5,5]):
    #         fig  = legend.figure
    #         fig.canvas.draw()
    #         bbox  = legend.get_window_extent()
    #         bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    #         bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    #         fig.savefig(filename, dpi="figure", bbox_inches=bbox)

    #     export_legend(leg)
    #     plt.show()





    # for outcome in outcomes:
    #     fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, width_ratios=[1, 1])
    #     fig.subplots_adjust(wspace=0.14)  # adjust space between Axes

    #     gp = res.xs(outcome, level=1)
    #     gp_c = res_c.xs(outcome, level=1)

    #     c_min = gp_c.min().min()*0.75
    #     c_max = gp_c.max().max()*1.04

    #     a_min = gp.min().min()*0.75
    #     a_max = gp.max().max()*1.04
    #     for sample, soff in zip(['Black', 'White'], [-1, 1]):
    #         gpp = gp.loc[sample]
    #         gpp_c = gp_c.loc[sample]

    #         bar_diff = gp.loc['Black'] - gp.loc['White']
    #         bar_diff_c = gp_c.loc['Black'] - gp_c.loc['White']

    #         # Left Figure
    #         ax1.barh(gpp_c.index, gpp_c*-1, height=0.25*soff, align='edge', zorder=23,
    #                  ec='k', color=color_d[sample])

    #         if (bar_diff_c.mean() < 0) & (sample == 'Black'):
    #             ax1.barh(gpp_c.index, bar_diff_c*-1, height=-0.25, align='edge', zorder=14,
    #                      ec='k', color='#E9D9A3', left=gp_c.loc['White']*-1)

    #         if (bar_diff_c.mean() > 0) & (sample == 'White'):
    #             ax1.barh(gpp_c.index, bar_diff_c, height=0.25, align='edge', zorder=14,
    #                      ec='k', color='#E9D9A3', left=gp_c.loc['Black']*-1)            


    #         # Right Figure
    #         ax2.barh(gpp.index, gpp, height=0.25*soff, align='edge', zorder=2,
    #                  ec='k', color=color_d[sample], label=label_d[sample])


    #         if (bar_diff.mean() < 0) & (sample == 'Black'):
    #             ax2.barh(gpp.index, bar_diff*-1, height=-0.25, align='edge', zorder=4,
    #                      ec='k', color='#E9D9A3', left=gp.loc['Black'], label='Racial difference')

    #         if (bar_diff.mean() > 0) & (sample == 'White'):
    #             ax2.barh(gpp.index, bar_diff, height=0.25, align='edge', zorder=4,
    #                      ec='k', color='#E9D9A3', left=gp.loc['White'], label='Racial difference')           

    #         p_min = min(a_min, c_min)
    #         p_max = max(a_max, c_max)
    #         ax1.set_xlim(-p_max, -p_min)          # Obs. outliers
    #         ax2.set_xlim(p_min, p_max)            # AA outliers

    #         ax2.text(a_max*0.75, 5.6, 'Randomized', ha='center', size=11)
    #         ax1.text(-c_max*0.75, 5.6, 'Observational', ha='center', size=11)
    #         ax1.grid(axis='x', color='#2c2c2c', zorder=-11, alpha=0.2)
    #         ax2.grid(axis='x', color='#2c2c2c', zorder=-11, alpha=0.2)
            
    #         ax1.locator_params(axis='x', nbins=3)
    #         ax2.locator_params(axis='x', nbins=3)       
    #         ax1.xaxis.set_major_formatter(lambda x, pos: f'{abs(x):g}')
            
    #     fig.savefig(output_p+f'{outcome}_both.pdf', dpi=300, bbox_inches='tight')
    #     handles, labels = ax2.get_legend_handles_labels()
    #     leg = fig.legend(handles, labels, loc='center', ncol=2,
    #                      bbox_to_anchor=(0.5, -0.01),fancybox=False, shadow=False) 

    #     def export_legend(legend, filename=output_p+"legend.pdf", expand=[-5,-5,5,5]):
    #         fig  = legend.figure
    #         fig.canvas.draw()
    #         bbox  = legend.get_window_extent()
    #         bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    #         bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    #         fig.savefig(filename, dpi="figure", bbox_inches=bbox)

    #     export_legend(leg)
    #     plt.show()



    # for outcome in outcomes:
    #     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, width_ratios=[2, 0.7, 2])
    #     fig.subplots_adjust(wspace=0.01)  # adjust space between Axes

    #     gp = res.xs(outcome, level=1)
    #     gp_c = res_c.xs(outcome, level=1)

    #     c_min = gp_c.min().min()*0.75
    #     c_max = gp_c.max().max()*1.05

    #     a_min = gp.min().min()*0.75
    #     a_max = gp.max().max()*1.05
    #     for sample, soff in zip(['Black', 'White'], [-1, 1]):
    #         gpp = gp.loc[sample]
    #         gpp_c = gp_c.loc[sample]
            
    #         bar_diff = gp.loc['Black'] - gp.loc['White']
    #         bar_diff_c = gp_c.loc['Black'] - gp_c.loc['White']

    #         # Left Figure
    #         ax1.barh(gpp_c.index, gpp_c*-1, height=0.25*soff, align='edge', zorder=2,
    #                  ec=None, color=color_d[sample], alpha=0.6)
            
    #         if (bar_diff_c.mean() < 0) & (sample == 'Black'):
    #             ax1.barh(gpp_c.index, bar_diff_c*-1, height=-0.25, align='edge', zorder=4,
    #                      ec=None, color='black', left=gp_c.loc['White']*-1)

    #         if (bar_diff_c.mean() > 0) & (sample == 'White'):
    #             ax1.barh(gpp_c.index, bar_diff_c, height=0.25, align='edge', zorder=4,
    #                      ec=None, color='black', left=gp_c.loc['Black']*-1)            
                
                
    #         # Middle Figure
    #         ax2.barh(gpp_c.index, gpp_c*-1, height=0.25*soff, align='edge', zorder=2,
    #                  ec=None, color=color_d[sample], alpha=0.6)
    #         ax2.barh(gpp.index, gpp, height=0.25*soff, align='edge', zorder=2,
    #                  ec=None, color=color_d[sample], label=label_d[sample], alpha=0.6)

    #         # Right Figure
    #         ax3.barh(gpp.index, gpp, height=0.25*soff, align='edge', zorder=2,
    #                  ec=None, color=color_d[sample], label=label_d[sample], alpha=0.6)


    #         if (bar_diff.mean() < 0) & (sample == 'Black'):
    #             ax3.barh(gpp.index, bar_diff*-1, height=-0.25, align='edge', zorder=4,
    #                      ec=None, color='black', left=gp.loc['Black'], label='Difference')

    #         if (bar_diff.mean() > 0) & (sample == 'White'):
    #             ax3.barh(gpp.index, bar_diff, height=0.25, align='edge', zorder=4,
    #                      ec=None, color='black', left=gp.loc['White'], label='Difference')           
            
            
    #         ax1.set_xlim(-c_max, -c_min)          # Obs. outliers
    #         ax2.set_xlim(-0.2*c_min, 0.2*a_min)   # Middle
    #         ax3.set_xlim(a_min, a_max)            # AA outliers

    #         ax1.spines.right.set_visible(False)
    #         ax2.spines.left.set_visible(False)
    #         ax2.spines.right.set_visible(False)
    #         ax3.spines.left.set_visible(False)
    #         ax2.tick_params(labelleft=False, left=False, labelright=False)
    #         ax2.set_xticks([0])
    #         ax3.tick_params(left=False)

    #         ax2.set_title(title_d[outcome], fontsize=12)
    #         ax1.axhline(0.5, 0, 1, color='black', linestyle='--')
    #         ax2.axhline(0.5, 0, 1, color='black', linestyle='--')
    #         ax3.axhline(0.5, 0, 1, color='black', linestyle='--')
    #         ax2.axvline(0, 0, 1, color='black', lw=0.5, zorder=-2)

    #         d = .4  # proportion of vertical to horizontal extent of the slanted line
    #         kwargs = dict(marker=[(-1, -d), (1, d)], markersize=7,
    #                       linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    #         ax1.plot([1, 1], [1, 0], transform=ax1.transAxes, **kwargs)
    #         ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
    #         ax2.plot([0, 1], [0, 0], transform=ax2.transAxes, **kwargs)
    #         ax3.plot([0, 0], [1, 0], transform=ax3.transAxes, **kwargs)

    #         ax1.set_ylim(-0.5, 6)

    #         ax3.text(a_max*0.75, 5.6, 'Randomized', ha='center', size=11)
    #         ax1.text(-c_max*0.75, 5.6, 'Observational', ha='center', size=11)

    #         labels = [item.get_text().lstrip('−') for item in ax1.get_xticklabels()]
    #         ax1.set_xticklabels(labels)
    #         ax1.locator_params(axis='x', nbins=3)
    #         ax3.locator_params(axis='x', nbins=3)       

    #     fig.savefig(output_p+f'{outcome}_both.pdf', dpi=300, bbox_inches='tight')
    #     handles, labels = ax3.get_legend_handles_labels()
    #     leg = fig.legend(handles, labels, loc='center', ncol=2,
    #                      bbox_to_anchor=(0.5, -0.01),fancybox=False, shadow=False) 
        
    #     def export_legend(legend, filename=output_p+"legend.pdf", expand=[-5,-5,5,5]):
    #         fig  = legend.figure
    #         fig.canvas.draw()
    #         bbox  = legend.get_window_extent()
    #         bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    #         bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    #         fig.savefig(filename, dpi="figure", bbox_inches=bbox)

    #     export_legend(leg)
    #     plt.show()    
        

def paper2_figure1_demeaned_barplot_hispanic():
    """
    Figure of sorting in observational sample, demeaned on age, sex and UoR
    """
    output_p = FS().analysis_p+'paper2_figure1_demeaned_barplot_hispanic/output/'


    df = pd.read_pickle(FS().derived_p+'paper2_analytic_tables_to_yearly/output/'
                        +'LA_analytic_table_yearly.pkl')
    df = df[df.recip_race_g.isin([0, 1, 2])].copy()
    # 2 is Hispanic, but labeled as other
    df['recip_race'] = df['recip_race'].map({'Other': 'Hispanic'}).fillna(df.recip_race)

    outcomes = ['pc_numdays', 'pharm_numclms', 'ed_avoid_numdays', 'all_cost_wins50']
    title_d = {'all_cost_wins50': 'Total Spending\n($ per enrollee per year)', 
               'pc_numdays': 'Primary Care Visits\n(per 100 enrollees per year)', 
               'pharm_numclms': 'Prescriptions\n(per 100 enrollees per year)', 
               'ed_avoid_numdays': 'Low-acuity ED visits\n(per 100 enrollees per year)',
               'ednoadmit_numdays': 'ED Visits\n(per 100 enrollees per year)'}
    color_d = {'White': '#4781BE', 'Black': '#F2C463', 'Hispanic': '#AcAcAc'}
    label_d = {'White': 'Non-Hispanic White enrollees', 'Black': 'Non-Hispanic Black enrollees', 
               'Hispanic': 'Hispanic enrollees'}
    height=0.6

    ## Prep dataframe
    r = {}
    # Only keep Active Choosers
    df = df[df.asgn_type.eq('Choice')]
    plan = 'plan_id'
    # Manually create the plan-race dummies, so we can de-mean each separately
    dummies = (pd.get_dummies(df[plan].astype(int).astype(str) 
                              + '_' + df['recip_race'].str.lower()).add_prefix('plan_'))

    df = (pd.concat([df, dummies],axis=1))
    # Manually create the race dummies, so we can de-mean each separately
    dummies = (pd.get_dummies(df['recip_race'].str.lower()))
    df = (pd.concat([df, dummies],axis=1))

    ## Regressions - By plan and by race
    results = {} 
    for y in outcomes: 
        # De-mean the outcome and each of the 10 dummy columns, add back in overall mean
        FEs = [f'plan_{i}_{x}' for i in range(1,6) for x in ['black', 'white', 'hispanic']]
        df[FEs] = df[FEs].astype(int)
        cols = [y] + FEs
        d = {}
        for col in cols:
            res = smf.ols(f'{col} ~ C(uor) + C(recip_age_g) + C(recip_gender_g) -1', data=df).fit()
            d[col] = df[col] - res.fittedvalues + df[col].mean()
        # Now run a regression with the de-meaned dummies as separate effects, the coefficient on those will become the 
        # plan effects interacted with race, net of the controls
        res = smf.ols(f'{y} ~ {"+".join(FEs)} - 1', data=pd.concat(d, axis=1)).fit()
        results[y] = res.params
    # Transform results into dataframe
    res = pd.DataFrame.from_dict(results).reset_index()
    res['plan'] = res['index'].str[:6].str.title().str.replace('_', ' ')
    res['race'] = res['index'].str[6:].str.title().str.replace('_', ' ').str.strip()
    r['byplan'] = (res.drop(columns='index')
                      .melt(['race', 'plan'], var_name='outcomes', value_name='')
                      .pivot(index=['race', 'outcomes'], columns='plan')
                      .sort_index())
    r['byplan'].columns = r['byplan'].columns.get_level_values(1)

    ## Regressions - By race
    results = {} 
    for y in outcomes: 
        # De-mean the outcome and each of the 2 dummy columns, add back in overall mean
        FEs = ['black', 'white', 'hispanic']
        df[FEs] = df[FEs].astype(int)
        cols = [y] + FEs
        d = {}
        for col in cols:
            res = smf.ols(f'{col} ~ C(uor) + C(recip_age_g) + C(recip_gender_g) -1', data=df).fit()
            d[col] = df[col] - res.fittedvalues + df[col].mean()

        # Now run a regression with the de-meaned dummies as separate effects, the coefficient on those will become the 
        # plan effects interacted with race, net of the controls
        res = smf.ols(f'{y} ~ {"+".join(FEs)} - 1', data=pd.concat(d, axis=1)).fit()
        results[y] = res.params
    # Transform results into dataframe
    res = pd.DataFrame.from_dict(results).reset_index()
    res['race'] = res['index'].str.title() 
    r['overall'] = (res.drop(columns='index')
                       .melt(['race'], var_name='outcomes', value_name='')
                       .set_index(['race', 'outcomes'])
                       .sort_index())
    r['overall'].columns = ['Overall']
    r['overall'].columns.names = ['plan']


    ## Concatenate all of the results and plot barplots
    # Concat and prepare
    res = pd.concat(r, axis=1)
    res.columns = res.columns.get_level_values(1)
    # Multiply by 100
    m = res.index.get_level_values(1) != 'all_cost_wins50'
    res.loc[m] *=100
    # Organize columns
    res = res[['Overall']+[c for c in res.columns if 'Plan' in c]]


    ## Plot
    fig, ax = plt.subplots(figsize=(10, 8), nrows=2, ncols=2)
    ax = ax.flatten()
    plt.subplots_adjust(hspace=0.2, wspace=0.27)
    for axis in ax:
        axis.tick_params(axis='both', which='major', labelsize=10.5)

    for i, outcome in enumerate(outcomes):
        ax[i].grid(color='#ABABAB', zorder=0, alpha=0.8, linestyle='--', axis='y')
        ax[i].set_title(title_d[outcome], fontsize=12)
        ax[i].axvline(0.5, 0, 1, color='black', linestyle='--')

        gp = res.xs(outcome, level=1)
        ymin = gp.min().min()*height
        for sample, offs in zip(['Black', 'Hispanic', 'White'], [-0.2, 0, 0.2]):
            gpp = gp.loc[sample]
            eb = ax[i].bar(np.array(range(len(gpp.index)))+offs, height=gpp, align='center', width=0.2, zorder=2,
                           ec='k', color=color_d[sample], label=label_d[sample])
        ax[i].set_ylim(ymin)
        ax[i].set_xticks(range(len(gpp.index)))
        ax[i].set_xticklabels(gpp.index)
    handles, labels = ax[-1].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc='center', ncol=2,
                     bbox_to_anchor=(0.5, -0.02),fancybox=False, shadow=False)
    plt.tight_layout()
    plt.savefig(output_p+'Figure2.eps', dpi=300, bbox_extra_artists=[leg], bbox_inches='tight')
    plt.savefig(output_p+'plan_demeaned.png', dpi=300, bbox_extra_artists=[leg], bbox_inches='tight')
    plt.show()


def paper2_figure1_demeaned_barplot_black_hispanic_pooled():
    """
    Figure of sorting in observational sample, demeaned on age, sex and UoR
    """
    output_p = FS().analysis_p+'paper2_figure1_demeaned_barplot_black_hispanic_pooled/output/'


    df = pd.read_pickle(FS().derived_p+'paper2_analytic_tables_to_yearly/output/'
                        +'LA_analytic_table_yearly.pkl')
    df = df[df.recip_race_g.isin([0, 1, 2])].copy()
    # 2 is Hispanic, but labeled as other
    df['recip_race'] = df['recip_race'].map({'Other': 'Black'}).fillna(df.recip_race)

    outcomes = ['pc_numdays', 'pharm_numclms', 'ed_avoid_numdays', 'all_cost_wins50']
    title_d = {'all_cost_wins50': 'Total Spending\n($ per enrollee per year)', 
               'pc_numdays': 'Primary Care Visits\n(per 100 enrollees per year)', 
               'pharm_numclms': 'Prescriptions\n(per 100 enrollees per year)', 
               'ed_avoid_numdays': 'Low-acuity ED visits\n(per 100 enrollees per year)',
               'ednoadmit_numdays': 'ED Visits\n(per 100 enrollees per year)'}
    color_d = {'White': '#4781BE', 'Black': '#F2C463'}
    label_d = {'White': 'Non-Hispanic White enrollees', 'Black': 'Non-Hispanic Black and Hispanic enrollees'}
    height=0.6

    ## Prep dataframe
    r = {}
    # Only keep Active Choosers
    df = df[df.asgn_type.eq('Choice')]
    plan = 'plan_id'
    # Manually create the plan-race dummies, so we can de-mean each separately
    dummies = (pd.get_dummies(df[plan].astype(int).astype(str) 
                              + '_' + df['recip_race'].str.lower()).add_prefix('plan_'))

    df = (pd.concat([df, dummies],axis=1))
    # Manually create the race dummies, so we can de-mean each separately
    dummies = (pd.get_dummies(df['recip_race'].str.lower()))
    df = (pd.concat([df, dummies],axis=1))

    ## Regressions - By plan and by race
    results = {} 
    for y in outcomes: 
        # De-mean the outcome and each of the 10 dummy columns, add back in overall mean
        FEs = [f'plan_{i}_{x}' for i in range(1,6) for x in ['black', 'white']]
        df[FEs] = df[FEs].astype(int)
        cols = [y] + FEs
        d = {}
        for col in cols:
            res = smf.ols(f'{col} ~ C(uor) + C(recip_age_g) + C(recip_gender_g) -1', data=df).fit()
            d[col] = df[col] - res.fittedvalues + df[col].mean()
        # Now run a regression with the de-meaned dummies as separate effects, the coefficient on those will become the 
        # plan effects interacted with race, net of the controls
        res = smf.ols(f'{y} ~ {"+".join(FEs)} - 1', data=pd.concat(d, axis=1)).fit()
        results[y] = res.params
    # Transform results into dataframe
    res = pd.DataFrame.from_dict(results).reset_index()
    res['plan'] = res['index'].str[:6].str.title().str.replace('_', ' ')
    res['race'] = res['index'].str[6:].str.title().str.replace('_', ' ').str.strip()
    r['byplan'] = (res.drop(columns='index')
                      .melt(['race', 'plan'], var_name='outcomes', value_name='')
                      .pivot(index=['race', 'outcomes'], columns='plan')
                      .sort_index())
    r['byplan'].columns = r['byplan'].columns.get_level_values(1)

    ## Regressions - By race
    results = {} 
    for y in outcomes: 
        # De-mean the outcome and each of the 2 dummy columns, add back in overall mean
        FEs = ['black', 'white']
        df[FEs] = df[FEs].astype(int)
        cols = [y] + FEs
        d = {}
        for col in cols:
            res = smf.ols(f'{col} ~ C(uor) + C(recip_age_g) + C(recip_gender_g) -1', data=df).fit()
            d[col] = df[col] - res.fittedvalues + df[col].mean()

        # Now run a regression with the de-meaned dummies as separate effects, the coefficient on those will become the 
        # plan effects interacted with race, net of the controls
        res = smf.ols(f'{y} ~ {"+".join(FEs)} - 1', data=pd.concat(d, axis=1)).fit()
        results[y] = res.params
    # Transform results into dataframe
    res = pd.DataFrame.from_dict(results).reset_index()
    res['race'] = res['index'].str.title()
    r['overall'] = (res.drop(columns='index')
                       .melt(['race'], var_name='outcomes', value_name='')
                       .set_index(['race', 'outcomes'])
                       .sort_index())
    r['overall'].columns = ['Overall']
    r['overall'].columns.names = ['plan']


    ## Concatenate all of the results and plot barplots
    # Concat and prepare
    res = pd.concat(r, axis=1)
    res.columns = res.columns.get_level_values(1)
    # Multiply by 100
    m = res.index.get_level_values(1) != 'all_cost_wins50'
    res.loc[m] *=100
    # Organize columns
    res = res[['Overall']+[c for c in res.columns if 'Plan' in c]]


    ## Plot
    fig, ax = plt.subplots(figsize=(10, 8), nrows=2, ncols=2)
    ax = ax.flatten()
    plt.subplots_adjust(hspace=0.2, wspace=0.27)
    for axis in ax:
        axis.tick_params(axis='both', which='major', labelsize=10.5)

    for i, outcome in enumerate(outcomes):
        ax[i].grid(color='#ABABAB', zorder=0, alpha=0.8, linestyle='--', axis='y')
        ax[i].set_title(title_d[outcome], fontsize=12)
        ax[i].axvline(0.5, 0, 1, color='black', linestyle='--')

        gp = res.xs(outcome, level=1)
        ymin = gp.min().min()*height
        for sample, soff in zip(['Black', 'White'], [-1, 1]):
            gpp = gp.loc[sample]
            eb = ax[i].bar(gpp.index, height=gpp, width=0.25*soff, align='edge', zorder=2,
                           ec='k', color=color_d[sample], label=label_d[sample])
        ax[i].set_ylim(ymin)

    handles, labels = ax[-1].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc='center', ncol=2,
                     bbox_to_anchor=(0.5, -0.02),fancybox=False, shadow=False)
    plt.tight_layout()
    plt.savefig(output_p+'Figure2.eps', dpi=300, bbox_extra_artists=[leg], bbox_inches='tight')
    plt.savefig(output_p+'plan_demeaned.png', dpi=300, bbox_extra_artists=[leg], bbox_inches='tight')
    plt.show()


def paper2_fig_age_dist():
    output_p = FS().analysis_p+'paper2_fig_age_dist/output/'

    df = pd.read_pickle(FS().derived_p+'paper2_analytic_tables_to_yearly/output/'
                        +'LA_analytic_table_yearly.pkl')
    df = df[df.recip_race_g.isin([0, 1])].copy()

    fig, ax = plt.subplots(figsize=(10, 4), nrows=1, ncols=2)
    ax = ax.flatten()
    plt.subplots_adjust(wspace=0.15)
    color_d = {'White': '#4781BE', 'Black': '#F2C463'}

    for axis in ax:
        axis.tick_params(axis='both', which='major', labelsize=10)
        
    _ = ax[0].hist(df[df.asgn_type.eq('AA') & df.recip_race_g.eq(0)].recip_age, 
                   bins=np.arange(-0.5, 100.5, 1), density=True, ec=None, 
                   alpha=0.7, color='#4781BE', label='Non-Hispanic White Enrollees')
    _ = ax[0].hist(df[df.asgn_type.eq('AA') & df.recip_race_g.eq(1)].recip_age, 
                   bins=np.arange(-0.5, 100.5, 1), density=True, ec=None, 
                   alpha=0.7, color='#F2C463', label='Non-Hispanic Black Enrollees')

    _ = ax[1].hist(df[df.asgn_type.eq('Choice') & df.recip_race_g.eq(0)].recip_age, 
                   bins=np.arange(-0.5, 100.5, 1), density=True, ec=None, 
                   alpha=0.7, color='#4781BE')
    _ = ax[1].hist(df[df.asgn_type.eq('Choice') & df.recip_race_g.eq(1)].recip_age, 
                   bins=np.arange(-0.5, 100.5, 1), density=True, ec=None, 
                   alpha=0.7, color='#F2C463')


    for axis in ax:
        axis.set_xlim(-1, 65)
        axis.set_xlabel('Age')
        axis.set_ylim(0, 0.075)
        
    ax[0].set_title('Randomized Population', ha='right')  
    ax[1].set_title('Observational Population', ha='right') 
    ax[0].legend()
    ax[0].set_ylabel('Proportion of Population (%)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_p+'age_dist.png', dpi=300)
    plt.show()

####################################################################################################
####################################################################################################
def map_race(df, state):
    """
    Map race to a consistent sent of variables across states. White=0, Black=1, Hispanic=2;
    this way estimates are relative to white. 
    """
    # KS is missing this information, so create a column. 
    if state == 'KS':
        df['recip_is_hispanic'] = False

    # Create new "Race" variable that includes hispanic
    s = pd.Series(np.where(df['recip_is_hispanic'].eq(True), 'Hispanic', df['recip_race']),
                  index=df.index)
    s = s.fillna('Other')

    # So that we can set estimates relative to White in Stata regressions
    s = s.map({'White': 0, 'Black': 1, 'Hispanic': 2, 'Other': 3, 'Asian': 4, 
               'Native American or Alaska Native': 5,
               'Native Hawaiian or Pacific Islander': 6})

    return s


def map_eligibility(df, state):
    """
    Create eligibility dummies for each state based on state specific logic.
    NO PREGNANCY
    """

    choices = ['disability', 'child', 'adult']
    if state == 'LA':
        conds = [df['med_elig_cat'].eq('04'),
                 (df['med_elig_type_case'].isin(['007', '014', '015', '055',])
                   | (df['med_elig_cat'].isin(['03', '13']) & df['recip_age'].lt(19))),
                 (df['med_elig_type_case'].isin(['001'])
                  | df['med_elig_cat'].eq('50')
                  | (df['med_elig_cat'].isin(['03', '13']) 
                     & df['recip_age'].between(19, 65)))
                  ]
    if state == 'TN':
        conds = [df['med_elig_type_case'].isin(['Aid to Disabled- CN', 'Aid to Disabled- CN and QMB',
                                                'Aid to Blind- CN', 'Aid to Disabled- No Money-CN',
                                                'Aid to Blind- CN and QMB', 'Aid to Disabled- MN',
                                                'Aid To The Disabled - Spend Down',
                                                'TennCare Disabled- Uninsured']),
                 (df['med_elig_type_case'].isin(['AFDC-CN', 'AFDC-No Money-CN',
                                        'Pregnant Women/Child of Specified Age under Pov Level'])
                  & df['recip_age'].lt(19)),
                 (df['med_elig_type_case'].isin(['AFDC-CN', 'AFDC-No Money-CN',
                                        'Pregnant Women/Child of Specified Age under Pov Level'])
                  & df['recip_age'].between(19, 65))
                ]
    if state == 'KS':
        s = df['med_elig_cat'].str.split('_', expand=True)
        s1 = df['med_elig_type_case'].str.split('_', expand=True)

        conds = [s1.isin(['11', '12', '21', '22', '26', 'H4']).any(1),
                 (s.isin(['TXXI', 'P21']).any(1) | s1.isin(['73', '95', 'G6', 'B9']).any(1)
                  | ( s.eq('TXIX').any(1) & s1.isin(['43', '45', '46', '36', '34', '62', '67']).any(1)
                     & df.recip_age.lt(19))),
                 (s1.isin(['36', 'G8']).any(1) & df['recip_age'].between(19, 65))
                ]

    df['elig_cat'] = np.select(conds, choices, default='other')
    return df


def round_pval(x):
    x = pd.to_numeric(x)
    if x < 0.001:
        return '<0.001'
    elif x >= 0.001 and x < 0.01:
        return f'{round(x,3):.3f}'
    else:
        return f'{round(x,2):.2f}'


def add_stars(x):
    x = pd.to_numeric(x, errors='coerce')
    if x < 0.01:
        return '**'
    elif x < 0.05:
        return '*'
    else:
        return ''


def add_weights(df, wcols, wname):
    s = df.groupby(['asgn_type']+wcols).size()
    s = s/s.groupby(level=0).sum()
    weights = (s.xs('Choice')/s.xs('AA')).to_frame(wname)

    df = df.merge(weights, how='left', on=weights.index.names)
    df.loc[df.asgn_type.eq('Choice'), weights.columns] = 1
    return df


####################################################################################################
####################################################################################################
def gen_paths(base_p):
    """
    Gets all 4 paths for analysis folder

    Returns:
        output_p, temp_p, log_p, code_p
    """
    return (base_p+'output/', base_p+'temp/', base_p+'logs/', base_p+'code/')    


def gen_filespecs(state, years):
    if state == 'TN':
        filespecs = (
            [(f'diagnosis_{year}.pkl', ['ICD9CM', 'ICD10CM'], 'diagnosis') 
                for year in range(*years)]
             + [(f'claims_details_{year}.pkl', ['CPT', 'HCPCS', 'UBREV', 'POS'], 'details') 
                for year in range(*years)]
             + [(f'pharm_{year}.pkl', ['NDC'], 'pharmacy') 
                for year in range(*years)]
             + [(f'inpatient_details_{year}.pkl', ['ICD9PCS', 'ICD10PCS'], 'details') 
                for year in range(*years)])
    elif state == 'LA':
        filespecs =  (
            [(f'diagnosis_{year}.pkl', ['ICD9CM', 'ICD10CM'], 'diagnosis') 
                for year in range(*years)]
             + [(f'claims_details_{year}.pkl', ['CPT', 'HCPCS', 'POS'], 'details') 
                for year in range(*years)]
             + [(f'pharm_{year}.pkl', ['NDC'], 'pharmacy') 
                for year in range(*years)]
             + [(f'surgery_{year}.pkl', ['ICD9PCS', 'ICD10PCS'], 'diagnosis') 
                for year in range(*years)]
             + [(f'revenue_{year}.pkl', ['UBREV'], 'diagnosis') 
                 for year in range(*years)])
    elif state == 'KS':
        filespecs = (
            [(f'diagnosis_{year}.pkl', ['ICD9CM', 'ICD10CM'], 'diagnosis') 
                for year in range(*years)]
             + [(f'claims_details_{year}.pkl', ['CPT', 'HCPCS', 'POS', 'UBREV'], 'details') 
                for year in range(*years)]
             + [(f'pharm_{year}.pkl', ['NDC'], 'pharmacy') 
                for year in range(*years)]
             + [(f'surgery_{year}.pkl', ['ICD9PCS', 'ICD10PCS'], 'diagnosis') 
                for year in range(*years)])

    return filespecs
