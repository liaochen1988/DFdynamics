import pandas as pd
import numpy as np
from scipy.spatial.distance import cityblock
from scipy import interpolate

def get_count_table(file_name_count, unstacked=False):
    df_count = pd.read_csv(file_name_count)
    if unstacked==True:
        df_count = pd.pivot_table(df_count, values = 'Count', index=['SampleID'], columns = 'ASV',fill_value=0).reset_index()
        df_count.columns.name = ''
        df_count.set_index('SampleID',drop=True,inplace=True)
    return df_count

def get_qpcr_table(file_name_qpcr):
    df_qpcr = pd.read_csv(file_name_qpcr, index_col=0)
    df_qpcr=df_qpcr.groupby(df_qpcr.index).mean()
    df_qpcr=df_qpcr[df_qpcr['Value']>0].reset_index()
    return df_qpcr

def get_sample_table(file_name_sample):
    df_sample = pd.read_csv(file_name_sample)
    return df_sample

def get_MSKCC_16S_relative_abundance_table(df_count):
    df_count_unstacked = pd.pivot_table(df_count, values = 'Count', index=['SampleID'], columns = 'ASV',fill_value=0).reset_index()
    df_count_unstacked.columns.name = ''
    df_count_unstacked.set_index('SampleID',drop=True,inplace=True)
    df_relative_abundance = df_count_unstacked.div(df_count_unstacked.sum(axis=1), axis=0)
    return df_relative_abundance

def group_counts_by_taxonomy(df_count, df_tax, tax_level='ASV', square_form=False, square_form_relative_abundance=True):
    assert tax_level in ['Kingdom','Phylum','Class','Order','Family','Genus','Species','ASV']

    # replace ASV-level annotation with specified taxonomy
    if tax_level != 'ASV':
        if 'ASV' in df_tax.columns:
            df_count_tax = pd.merge(df_count, df_tax[['ASV',tax_level]], left_on='ASV', right_on='ASV', how='left')
        else:
            df_count_tax = pd.merge(df_count, df_tax[[tax_level]], left_on='ASV', right_on='ASV', how='left')
        df_count_tax = df_count_tax[['SampleID','Count',tax_level]]
    else:
        df_count_tax = df_count

    # use squareform
    if square_form==True:
        df_count_tax = pd.pivot_table(df_count_tax, values = 'Count', index=['SampleID'], columns=tax_level, fill_value=0, aggfunc=np.sum).reset_index()
        df_count_tax.columns.name = ''
        df_count_tax.set_index('SampleID',drop=True,inplace=True)
        if square_form_relative_abundance==True:
            df_count_tax = df_count_tax.div(df_count_tax.sum(axis=1), axis=0)

    return df_count_tax

def find_overlap_samples(df1, df2, df3=None):
    if df3 is None:
        common_samples = list(set(df1.SampleID).intersection(set(df2.SampleID)))
        df1 = df1[df1.SampleID.isin(common_samples)]
        df2 = df2[df2.SampleID.isin(common_samples)]
        return df1, df2
    else:
        common_samples = list(set(df1.SampleID).intersection(set(df2.SampleID)).intersection(set(df3.SampleID)))
        df1 = df1[df1.SampleID.isin(common_samples)]
        df2 = df2[df2.SampleID.isin(common_samples)]
        df3 = df3[df3.SampleID.isin(common_samples)]
        return df1, df2, df3

def get_consecutive_sample_dict(df_sample):
    # key is sample at day t and value is sample at day t+1
    consecutive_sample_dict = {}
    # loop over all patients
    unique_pids = list(np.unique(df_sample.PatientID))
    for pid in unique_pids:
        # get all samples of a patient
        df_sample_pid = df_sample[df_sample.PatientID==pid]
        # sort samples by time point
        df_sample_pid = df_sample_pid.sort_values(by='Timepoint', axis=0)
        # loop over samples of the patient pid
        for k,curr_s in enumerate(df_sample_pid.SampleID):
            if k<len(df_sample_pid.index)-1:
                next_s = df_sample_pid.iloc[k+1]['SampleID']
                if df_sample_pid.iloc[k+1]['Timepoint']-df_sample_pid.iloc[k]['Timepoint']==1:
                    assert curr_s not in consecutive_sample_dict
                    consecutive_sample_dict[curr_s] = next_s

    return consecutive_sample_dict

def get_MSKCC_drug_table(file_name_drug, anti_infective=None, convert_to_single_day_entry=True):
    df_drug = pd.read_csv(file_name_drug)
    if anti_infective is not None:
        df_drug = df_drug[df_drug.AntiInfective==anti_infective].reset_index(drop=True)

    if convert_to_single_day_entry==False:
        return df_drug
    else:
        df_drug_single_day = []
        for index in df_drug.index:
            start_tps = df_drug.loc[index,'StartTimepoint']
            stop_tps = df_drug.loc[index,'StopTimepoint']
            start_day = df_drug.loc[index,'StartDayRelativeToNearestHCT']
            for timepoint in np.arange(start_tps,stop_tps+1):
                res = list(df_drug.loc[index])
                res.append(timepoint)
                res.append(timepoint-start_tps+start_day)
                df_drug_single_day.append(res)
        df_drug_single_day = pd.DataFrame(df_drug_single_day, columns=list(df_drug.columns)+['Timepoint','DayRelativeToNearestHCT'])
        df_drug_single_day = df_drug_single_day.drop(['StartTimepoint','StopTimepoint','StartDayRelativeToNearestHCT','StopDayRelativeToNearestHCT'], axis=1)
        return df_drug_single_day

def get_MSKCC_taxonomy_table(file_name_taxonomy, index_col=None):
    if index_col is None:
        df_tax = pd.read_csv(file_name_taxonomy)
    else:
        df_tax = pd.read_csv(file_name_taxonomy, index_col=index_col)
    return df_tax

def fill_unclassified_taxa(df_taxonomy):
    ranks = ['Kingdom','Phylum','Class','Order','Family','Genus']
    for index in df_taxonomy.index:
        row = list(df_taxonomy.loc[index,ranks])
        for k in np.arange(len(row)):
            if str(row[k]) == 'nan' and k != 0:
                row[k] = row[k-1]
        df_taxonomy.loc[index,ranks] = row
    return df_taxonomy

def compute_absolute_abundance(df_relative_abundance, df_qpcr):
    if 'SampleID' in df_qpcr.columns:
        df_qpcr2 = df_qpcr.set_index('SampleID',drop=True)
    else:
        df_qpcr2 = df_qpcr
    df_absolute_abundance = pd.merge(df_relative_abundance, df_qpcr2, how='inner', left_index=True, right_index=True)
    df_absolute_abundance = df_absolute_abundance.mul(df_absolute_abundance['Value'], axis=0)
    df_absolute_abundance.drop(columns=['Value'], inplace=True)
    return df_absolute_abundance

def get_most_abundant_taxa_group(df_count, df_taxonomy, df_qpcr=None, rank='ASV', n=-1):
    if rank=='ASV':
        df_relative_abundance = get_MSKCC_16S_relative_abundance_table(df_count)
    else:
        df_count_tax = pd.merge(df_count, df_taxonomy, how='left')
        df_count_tax = df_count_tax[['SampleID','Count',rank]]
        df_count_tax = df_count_tax.fillna('Other') # group unclassified rank with other
        df_relative_abundance = pd.pivot_table(df_count_tax, values = 'Count', index=['SampleID'], columns=rank, fill_value=0, aggfunc=np.sum).reset_index()
        df_relative_abundance.columns.name = ''
        df_relative_abundance.set_index('SampleID',drop=True,inplace=True)
        df_relative_abundance = df_relative_abundance.div(df_relative_abundance.sum(axis=1), axis=0)

    if df_qpcr is None:
        df_relative_abundance_mean = df_relative_abundance.mean(axis=0).sort_values(ascending=False)
        if n==-1:
            return list(df_relative_abundance_mean.index), df_relative_abundance
        else:
            assert n>=0
            return list(df_relative_abundance_mean.index[0:n]), df_relative_abundance
    else:
        df_absolute_abundance = compute_absolute_abundance(df_relative_abundance, df_qpcr)
        df_absolute_abundance_mean = df_absolute_abundance.mean(axis=0).sort_values(ascending=False)
        if n==-1:
            return list(df_absolute_abundance_mean.index), df_relative_abundance, df_absolute_abundance
        else:
            assert n>=0
            return list(df_absolute_abundance_mean.index[0:n]), df_relative_abundance, df_absolute_abundance

def rank_microbiome_trajectory_similarity(patient_in_focus, start_day, stop_day, df_sample, df_relative_abundance, umap_components=10, nrun=1):
    # select samples based on start_day and stop_day
    df_sample_selected = df_sample[(df_sample.DayRelativeToNearestHCT==start_day) | (df_sample.DayRelativeToNearestHCT==stop_day)]
    unique_pids = list(set(df_sample_selected.PatientID))
    pids_to_keep = []
    for pid in unique_pids:
        # keep one sample if multiple samples exist for the sample patient on the same day
        curr_df = df_sample_selected[df_sample_selected.PatientID==pid].drop_duplicates(subset=['DayRelativeToNearestHCT'])
        if len(curr_df.index)>1:
            pids_to_keep.append(pid)

    # make sure focus patient have microbiome samples on the start and stop day
    assert patient_in_focus in pids_to_keep

    # get all samples belong to these patients
    df_sample_selected = df_sample[(df_sample.PatientID.isin(pids_to_keep)) & (df_sample.DayRelativeToNearestHCT>=start_day) & (df_sample.DayRelativeToNearestHCT<=stop_day)].reset_index(drop=True)
    df_relative_abundance_selected = df_relative_abundance.loc[df_sample_selected.SampleID]

    # get Umap coordinates of these samples
    df_dist = None
    for kk in np.arange(nrun):
        np.random.seed(42)
        reducer = umap.UMAP(metric="manhattan", n_components=umap_components)
        embedding = reducer.fit_transform(df_relative_abundance_selected)

        # fit B-spline
        bspline = {}
        for pid in pids_to_keep:
            curr_df = df_sample_selected[df_sample_selected.PatientID==pid].drop_duplicates(subset=['DayRelativeToNearestHCT']).sort_values('DayRelativeToNearestHCT',ascending=True)
            curr_embedd = embedding[curr_df.index]
            input_data = [curr_embedd[:,0]]
            for k in np.arange(1,nc):
                input_data.append(curr_embedd[:,k])
            if curr_embedd.shape[0]==2:
                tck,u = interpolate.splprep(input_data,k=1,s=0)
            elif curr_embedd.shape[0]==3:
                tck,u = interpolate.splprep(input_data,k=2,s=0)
            else:
                tck,u = interpolate.splprep(input_data,k=3,s=0)
            u=np.linspace(0,1,num=50,endpoint=True)

            # rows: u
            # columns: components
            bspline[pid] = pd.DataFrame(interpolate.splev(u,tck), columns=u).transpose()

        # compute distance between B-splines
        df_patient_in_focus = bspline[patient_in_focus]
        curr_df_dist = []
        for pid,df_pid in bspline.items():
            if pid != patient_in_focus:
                dist_ave = 0.0
                for u in df_pid.index:
                    dist_ave += cityblock(df_pid.loc[u], df_patient_in_focus.loc[u])
                dist_ave /= len(df_pid.index)
                curr_df_dist.append([pid, dist_ave])
        curr_df_dist = pd.DataFrame(curr_df_dist, columns=['PatientID',str(kk)]).set_index('PatientID',drop=True)

        # merge results
        if kk==0:
            df_dist = curr_df_dist
        else:
            df_dist = pd.merge(df_dist, curr_df_dist, left_index=True, right_index=True, how='inner')

    # add mean colume and sort by their values
    df_dist['mean'] = df_dist.mean(axis=1)
    df_dist = df_dist.sort_values('mean')

    return df_dist

# sense could be presence_absence (administered/not administered), total_count (total number of days a drug was administered), exact (consider temporal order)
def select_patients_with_similar_drug_administration(patient_in_focus, start_tps, stop_tps, df_sample, df_drug, sense='presence_absence', max_drug_administration_distance=0):

    # some logistic checks
    assert stop_tps>=start_tps

    # unique drug factors
    unique_drug_factors = list(set(df_drug.Factor))

    # get drug vector for patient pid_k
    df_drug_selected = df_drug[(df_drug.PatientID==patient_in_focus) & (df_drug.Timepoint >=start_tps) & (df_drug.Timepoint <= stop_tps)]
    if sense=='presence_absence':
        drug_dict_in_focus = {dfac:0 for dfac in unique_drug_factors}
        for dfac in df_drug_selected.Factor:
            drug_dict_in_focus[dfac] = 1
        drug_vec_patient_in_focus = [drug_dict_in_focus[dfac] for dfac in unique_drug_factors]
    elif sense=='total_count':
        drug_dict_in_focus = {dfac:0 for dfac in unique_drug_factors}
        for dfac in df_drug_selected.Factor:
            drug_dict_in_focus[dfac] += 1
        drug_vec_patient_in_focus = [drug_dict_in_focus[dfac] for dfac in unique_drug_factors]
    elif sense=='exact':
        drug_dict_in_focus = {dfac:[0]*(stop_tps-start_tps+1) for dfac in unique_drug_factors}
        for dfac,tps in zip(df_drug_selected.Factor, df_drug_selected.Timepoint):
            drug_dict_in_focus[dfac][tps-start_tps] = 1
        drug_vec_patient_in_focus = []
        for dfac in unique_drug_factors:
            drug_vec_patient_in_focus.extend(drug_dict_in_focus[dfac])
    else:
        print('unknown sense %s'%(sense))
        raise

    unique_pids = list(set(df_sample.PatientID))
    res = []
    for curr_patient in unique_pids:
        curr_df_sample = df_sample[df_sample.PatientID==curr_patient].drop_duplicates(subset=['Timepoint']).sort_values(['Timepoint']).reset_index(drop=True)
        # find pairs of start and stop points in this patient that have the same length as [start_tps, stop_tps]
        for curr_start_tps in curr_df_sample.Timepoint:
            curr_stop_tps = curr_start_tps + stop_tps - start_tps
            if curr_stop_tps in list(curr_df_sample.Timepoint):
                curr_df_drug = df_drug[(df_drug.PatientID==curr_patient) & (df_drug.Timepoint >=curr_start_tps) & (df_drug.Timepoint <= curr_stop_tps)]
                if sense=='presence_absence':
                    curr_drug_dict = {dfac:0 for dfac in unique_drug_factors}
                    for dfac in curr_df_drug.Factor:
                        curr_drug_dict[dfac] = 1
                    curr_drug_vec = [curr_drug_dict[dfac] for dfac in unique_drug_factors]
                elif sense=='total_count':
                    curr_drug_dict = {dfac:0 for dfac in unique_drug_factors}
                    for dfac in curr_df_drug.Factor:
                        curr_drug_dict[dfac] += 1
                    curr_drug_vec = [curr_drug_dict[dfac] for dfac in unique_drug_factors]
                elif sense=='exact':
                    curr_drug_dict = {dfac:[0]*(curr_stop_tps-curr_start_tps+1) for dfac in unique_drug_factors}
                    for dfac,tps in zip(curr_df_drug.Factor, curr_df_drug.Timepoint):
                        curr_drug_dict[dfac][tps-curr_start_tps] = 1
                    curr_drug_vec = []
                    for dfac in unique_drug_factors:
                        curr_drug_vec.extend(curr_drug_dict[dfac])
                else:
                    print('unknown sense %s'%(sense))
                    raise

                # get distance between drug administration
                drug_vec_distance = cityblock(drug_vec_patient_in_focus, curr_drug_vec)
                res.append([patient_in_focus, start_tps, stop_tps, curr_patient, curr_start_tps, curr_stop_tps, drug_vec_distance])

    df_res = pd.DataFrame(res, columns=['PatientQuery','StartTimepointQuery','StopTimepointQuery','PatientMatch','StartTimepointMatch','StopTimepointMatch','DrugAdministrationDistance'])
    df_res = df_res[df_res.DrugAdministrationDistance <= max_drug_administration_distance]
    return df_res
