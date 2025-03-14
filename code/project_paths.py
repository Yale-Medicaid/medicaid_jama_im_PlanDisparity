class FS():
    """
    Class which contains all of the run script paths. 
    """
    def __init__(self):
        # All of the LA Pipeline paths
        self.la_pipe_main_path = ('//storage.yale.edu/home/YSPH_HPM-CC0940-MEDSPH2/'
                                  +'Private Data Pipeline/Healthcare_Data/LA/')
        self.tn_pipe_main_path = ('//storage.yale.edu/home/YSPH_HPM-CC0940-MEDSPH2/'
                                  +'Private Data Pipeline/Healthcare_Data/TN/')
        self.ks_pipe_main_path = ('//storage.yale.edu/home/YSPH_HPM-CC0940-MEDSPH2/'
                                  +'Private Data Pipeline/Healthcare_Data/KS/')
        self.mi_pipe_main_path = ('//storage.yale.edu/home/YSPH_HPM-CC0940-MEDSPH2/'
                                  +'Private Data Pipeline/Healthcare_Data/MI/')


        self.la_pipe_raw_path = self.la_pipe_main_path+ 'Raw/'
        self.la_pipe_std_path = self.la_pipe_main_path+ 'Standard/'
        self.la_pipe_gold_path = self.la_pipe_main_path+ 'Gold/'
        self.la_pipe_inter_path = self.la_pipe_main_path+ 'Intermediate/'
        self.la_pipe_analytic_path = self.la_pipe_main_path+ 'Analytic_Tables/'

        self.tn_pipe_raw_path = self.tn_pipe_main_path+ 'Raw/'
        self.tn_pipe_std_path = self.tn_pipe_main_path+ 'Standard/'
        self.tn_pipe_gold_path = self.tn_pipe_main_path+ 'Gold/'
        self.tn_pipe_inter_path = self.tn_pipe_main_path+ 'Intermediate/'
        self.tn_pipe_analytic_path = self.tn_pipe_main_path+ 'Analytic_Tables/'

        self.ks_pipe_raw_path = self.ks_pipe_main_path+ 'Raw/'
        self.ks_pipe_std_path = self.ks_pipe_main_path+ 'Standard/'
        self.ks_pipe_gold_path = self.ks_pipe_main_path+ 'Gold/'
        self.ks_pipe_inter_path = self.ks_pipe_main_path+ 'Intermediate/'
        self.ks_pipe_analytic_path = self.ks_pipe_main_path+ 'Analytic_Tables/'


        self.mi_pipe_raw_path = self.mi_pipe_main_path+ 'Raw/'
        self.mi_pipe_std_path = self.mi_pipe_main_path+ 'Standard/'
        self.mi_pipe_gold_path = self.mi_pipe_main_path+ 'Gold/'
        self.mi_pipe_inter_path = self.mi_pipe_main_path+ 'Intermediate/'
        self.mi_pipe_analytic_path = self.mi_pipe_main_path+ 'Analytic_Tables/'

        self.trunk_p = 'D:/Groups/YSPH-HPM-Ndumele/Networks/Anthony/Racial_Disparities/trunk/'
        self.raw_p = self.trunk_p + 'raw/'
        self.derived_p = self.trunk_p + 'derived/'
        self.analysis_p = self.trunk_p + 'analysis/'