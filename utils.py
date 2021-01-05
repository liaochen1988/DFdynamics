import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.interpolate import CubicSpline
from scipy.integrate import odeint, ode, solve_ivp
from scipy.stats import pearsonr, spearmanr
import scipy.stats.kde as kde
from scipy.spatial import distance
from skbio.stats.ordination import pcoa
import warnings
warnings.filterwarnings("ignore")

# Note: all samples in df_meta should be consistent in their diets
def data_processing_scfa(
    df_meta, # meta data
    df_bac,  # relative abundace or absolute abundance of gut microbiome
    df_scfa, # SCFA measurement
    target_scfa, # dependent variable(s) in the regression
    topN,    # keep only the most abundance N taxa in the model
    exclude_group, # group of mice excluded from model training
    exclude_vendor, # vendor of mice excluded from model training
    use_pcoa # convert bacterial abundance to pcoa
):
    # exlucde mice group
    if exclude_group is not None:
        assert exclude_group in ['A','B','C','D','E']
        df_meta_sliced = df_meta[df_meta.RandomizedGroup != exclude_group]
    else:
        df_meta_sliced = deepcopy(df_meta)

    # exlucde mice group
    if exclude_vendor is not None:
        assert exclude_vendor in ['Hunan','Shanghai','Guangdong','Beijing']
        df_meta_sliced = df_meta_sliced[df_meta_sliced.Vendor != exclude_vendor]

    # select target scfa
    for scfa_ in target_scfa:
        if scfa_ not in list(df_scfa.columns):
            print('unknown scfa: %s'%(target_scfa))
            raise

    # make sure that meta data and SCFA have the same samples in the same order
    shared_samples = [x for x in df_meta_sliced.index if x in list(df_scfa.index)]
    df_scfa_sliced = df_scfa.loc[shared_samples, target_scfa]
    df_meta_sliced = df_meta_sliced.loc[shared_samples]

    # calculate SCFA derivatives
    df_scfa_sliced = pd.merge(df_meta_sliced, df_scfa_sliced, left_index=True, right_index=True, how='inner')
    df_scfa_deriv = deepcopy(df_scfa_sliced)
    for curr_subject in set(df_scfa_deriv.SubjectID):
        curr_df = df_scfa_sliced[df_scfa_sliced.SubjectID==curr_subject].sort_values(by='Day')
        xdata = np.array(curr_df['Day'])
        for scfa_ in target_scfa:
            ydata = np.array(curr_df[scfa_])
            cs = CubicSpline(xdata, ydata)
            csd1 = cs.derivative(nu=1)
            ydata_d1 = csd1(xdata)
            df_scfa_deriv.loc[df_scfa_deriv.SubjectID==curr_subject, scfa_] = ydata_d1

    # keep only samples in df_meta_sliced for bacterial abundance data
    df_bac_sliced = df_bac.loc[df_meta_sliced.index]

    # run PCOA if needed
    if use_pcoa:
        # keep the topN principle components
        bac_dist = distance.squareform(distance.pdist(df_bac_sliced, metric="braycurtis"))
        df_bac_dist = pd.DataFrame(bac_dist, index = df_bac_sliced.index, columns = df_bac_sliced.index)
        ndim = np.min([topN,len(df_bac_sliced.index)])
        OrdinationResults = pcoa(df_bac_dist.values, number_of_dimensions=ndim)
        df_bac_sliced = pd.DataFrame(OrdinationResults.samples.values, index=df_bac_sliced.index, columns=['PC%d'%(n) for n in np.arange(1,ndim+1)])
    else:
        # keep the topN taxa based on averaged abundance
        df_bac_sliced_T = df_bac_sliced.T
        df_bac_sliced_T['mean'] = df_bac_sliced_T.mean(axis=1)
        df_bac_sliced_T = df_bac_sliced_T.sort_values(by=['mean'], axis=0, ascending=False)
        df_bac_sliced_T = df_bac_sliced_T.drop('mean', axis=1)
        df_bac_sliced = df_bac_sliced_T.iloc[0:topN].T
    selected_topN_bac = list(df_bac_sliced.columns)

    # calculate Microbiome derivative
    df_bac_meta = pd.merge(df_meta_sliced, df_bac_sliced, left_index=True, right_index=True, how='inner')
    df_bac_deriv = deepcopy(df_bac_meta)
    for curr_subject in set(df_bac_deriv.SubjectID):
        curr_df = df_bac_meta[df_bac_meta.SubjectID==curr_subject].sort_values(by='Day')
        xdata = np.array(curr_df['Day'])
        for bac_ in selected_topN_bac:
            ydata = np.array(curr_df[bac_])
            cs = CubicSpline(xdata, ydata)
            csd1 = cs.derivative(nu=1)
            ydata_d1 = csd1(xdata)
            df_bac_deriv.loc[df_bac_deriv.SubjectID==curr_subject, bac_] = ydata_d1
    df_bac_deriv = df_bac_deriv[selected_topN_bac]

    return selected_topN_bac, df_meta_sliced, df_bac_sliced, df_bac_deriv, df_scfa_sliced, df_scfa_deriv

def binarize_categories(lst):
    lines = []
    all_days = [0, 1, 2, 3, 5, 8, 10, 13, 19, 25, 31]
    for elem in lst:
        curr_line = [0]*len(all_days)
        curr_line[all_days.index(elem)] = 1
        lines.append(curr_line)
    df_line = pd.DataFrame(lines, columns=all_days)
    return df_line

def train_scfa_dynamics_model(
    df_meta, # meta data
    df_bac,  # relative abundace or absolute abundance of gut microbiome
    df_scfa, # SCFA measurement
    target_scfa, # dependent variable(s) in the regression
    topN=40, # keep only the most abundance N taxa in the model
    exclude_group=None, # group of mice excluded from model training
    exclude_vendor=None, # group of mice excluded from model training
    model='Correlation',# regression model
    opt_params = None, # optimal model parameters
    use_deriv_scfa=True, # whether using dSCFA/dt as the dependent variable
    use_deriv_microbiome=False, # whether dMicrobiome/dt as the independent variable
                                # if None, use day as the sole indepedent variable
    use_pcoa=False
):
    # get processed input data
    selected_topN_bac, df_meta_sliced, df_bac_sliced, df_bac_deriv, df_scfa_sliced, df_scfa_deriv = data_processing_scfa(df_meta, df_bac, df_scfa, target_scfa, topN, exclude_group, exclude_vendor, use_pcoa)

    # train specified model on the data
    if model=='Correlation':
        lines = []
        for scfa_ in target_scfa:
            if use_deriv_microbiome is None:
                df_day_pres = binarize_categories(list(df_scfa_sliced.Day))
                for d in df_day_pres.columns:
                    if use_deriv_scfa:
                        Y_var = np.asarray(list(df_scfa_deriv[scfa_]))
                    else:
                        Y_var = np.asarray(list(df_scfa_sliced[scfa_]))
                    X_var = np.asarray(list(df_day_pres[d]))
                    corr_p, pvalue_p = pearsonr(Y_var, X_var)
                    corr_s, pvalue_s = spearmanr(Y_var, X_var)
                    lines.append([scfa_, d, corr_p, pvalue_p, corr_s, pvalue_s])
            else:
                for t in selected_topN_bac:
                    if use_deriv_scfa:
                        Y_var = np.asarray(list(df_scfa_deriv[scfa_]))
                    else:
                        Y_var = np.asarray(list(df_scfa_sliced[scfa_]))
                    if use_deriv_microbiome:
                        X_var = np.asarray(list(df_bac_deriv[t]))
                    else:
                        X_var = np.asarray(list(df_bac_sliced[t]))
                    corr_p, pvalue_p = pearsonr(Y_var, X_var)
                    corr_s, pvalue_s = spearmanr(Y_var, X_var)
                    lines.append([scfa_, t, corr_p, pvalue_p, corr_s, pvalue_s])
        df_output = pd.DataFrame(lines, columns=['SCFA','Feature','PearsonR','PearsonP','SpearmanR','SpearmanP'])
        return df_output
    elif model=='ElasticNet':
        lines = []
        regression_model = {}
        for scfa_ in target_scfa:
            if use_deriv_scfa:
                Y_var = np.asarray(list(df_scfa_deriv[scfa_]))
            else:
                Y_var = np.asarray(list(df_scfa_sliced[scfa_]))
            if use_deriv_microbiome is None:
                df_day_pres = binarize_categories(list(df_scfa_sliced.Day))
                X_var = np.asarray(df_day_pres.values)
            else:
                if use_deriv_microbiome:
                    X_var = np.asarray(df_bac_deriv.values)
                else:
                    X_var = np.asarray(df_bac_sliced.values)

            if opt_params is None:
                l1_ratio = [1e-4, .1, .3, .5, .7, .9, .95, .99, 1]
                clf = ElasticNetCV(
                    eps=1e-4,
                    n_alphas=10000,
                    cv=5,
                    random_state=0,
                    max_iter=100000,
                    tol=1e-6,
                    l1_ratio=l1_ratio,
                    n_jobs=-1
                ).fit(X_var, Y_var)
                best_l1_ratio = clf.l1_ratio_
                best_alpha = clf.alpha_
            else:
                best_l1_ratio = opt_params[0]
                best_alpha = opt_params[1]
            reg = ElasticNet(
                l1_ratio=best_l1_ratio,
                alpha=best_alpha,
                random_state=0,
                max_iter=100000,
                tol=1e-6
            )
            clf = reg.fit(X_var, Y_var)
            reg.feature_names = selected_topN_bac
            regression_model[scfa_] = reg
            lines.append([scfa_, best_alpha, best_l1_ratio, clf.score(X_var, Y_var)]+list(clf.coef_))
        if use_deriv_microbiome is None:
            df_output = pd.DataFrame(lines, columns=['SCFA','BestAlpha','BestL1Ratio','R2']+list(df_day_pres.columns))
        else:
            df_output = pd.DataFrame(lines, columns=['SCFA','BestAlpha','BestL1Ratio','R2']+selected_topN_bac)
        return df_output, regression_model
    elif model=='RandomForest':
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 5)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 30, num = 3)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]

        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}

        lines_opt = []
        lines_reg = []
        regression_model = {}
        for scfa_ in target_scfa:
            if use_deriv_scfa:
                Y_var = np.asarray(list(df_scfa_deriv[scfa_]))
            else:
                Y_var = np.asarray(list(df_scfa_sliced[scfa_]))
            if use_deriv_microbiome is None:
                df_day_pres = binarize_categories(list(df_scfa_sliced.Day))
                X_var = np.asarray(df_day_pres.values)
            else:
                if use_deriv_microbiome:
                    X_var = np.asarray(df_bac_deriv.values)
                else:
                    X_var = np.asarray(df_bac_sliced.values)

            if opt_params is None:
                # grid search
                rf = RandomForestRegressor()
                rf_random = GridSearchCV(
                    estimator = rf,
                    param_grid = random_grid,
                    cv = 5,
                    verbose=2,
                    n_jobs = -1)

                # fit the random search model
                rf_random.fit(X_var, Y_var)
                lines_opt.append([scfa_,
                                 rf_random.best_params_['n_estimators'],
                                 rf_random.best_params_['max_features'],
                                 rf_random.best_params_['max_depth'],
                                 rf_random.best_params_['min_samples_split'],
                                 rf_random.best_params_['min_samples_leaf'],
                                 rf_random.best_params_['bootstrap']])

                # run RF using optimal parameter
                reg = RandomForestRegressor(
                    random_state=0,
                    bootstrap=rf_random.best_params_['bootstrap'],
                    max_depth=None if rf_random.best_params_['max_depth']=='nan' else rf_random.best_params_['max_depth'],
                    max_features=rf_random.best_params_['max_features'],
                    min_samples_leaf=rf_random.best_params_['min_samples_leaf'],
                    min_samples_split=rf_random.best_params_['min_samples_split'],
                    n_estimators=rf_random.best_params_['n_estimators'],
                    n_jobs=-1)
            else:
                reg = RandomForestRegressor(
                    random_state=0,
                    bootstrap=list(opt_params.loc[opt_params.SCFA==scfa_,'bootstrap'])[0],
                    max_depth=None if str(list(opt_params.loc[opt_params.SCFA==scfa_,'max_depth'])[0])=='nan' else list(opt_params.loc[opt_params.SCFA==scfa_,'max_depth'])[0],
                    max_features=list(opt_params.loc[opt_params.SCFA==scfa_,'max_features'])[0],
                    min_samples_leaf=list(opt_params.loc[opt_params.SCFA==scfa_,'min_samples_leaf'])[0],
                    min_samples_split=list(opt_params.loc[opt_params.SCFA==scfa_,'min_samples_split'])[0],
                    n_estimators=list(opt_params.loc[opt_params.SCFA==scfa_,'n_estimators'])[0],
                    n_jobs=-1
                )
            clf = reg.fit(X_var, Y_var)
            if use_deriv_microbiome is None:
                reg.feature_names = deepcopy(list(df_day_pres.columns)) # add feature names
            else:
                reg.feature_names = deepcopy(selected_topN_bac) # add feature names
            lines_reg.append([scfa_, clf.score(X_var, Y_var)]+ list(clf.feature_importances_))
            regression_model[scfa_] = deepcopy(reg)
        df_output_opt = pd.DataFrame(lines_opt, columns=['SCFA','n_estimators','max_features','max_depth','min_samples_split','min_samples_leaf','bootstrap'])
        if use_deriv_microbiome is None:
            df_output_reg = pd.DataFrame(lines_reg, columns=['SCFA','R2']+list(df_day_pres.columns))
        else:
            df_output_reg = pd.DataFrame(lines_reg, columns=['SCFA','R2']+selected_topN_bac)
        return df_output_reg, df_output_opt, regression_model
    else:
        print('unknown method: %s'%(method))
        raise


def generate_stan_files_for_fiber_respones(
    df_bac, # 16S data (relative or absolute), rows are samples, columns are taxa
    df_meta, # meta data, rows are samples, columns are SubjectID, Day, and Dose
    prefix, # prefix of stan file name
    topN=20, # select the topN taxa to run bayesian regression
    stan_path='/Users/liaoc/Documents/cmdstan-2.24.1/projects/microbiome_fiber_response_LD'
):
    # stan program does not support certain symbols, replace them with meaningful ones
    df_bac_renamed = deepcopy(df_bac)
    df_bac_renamed.columns = [c.replace('/','_slash_').replace(' ','_space_').replace('[','_leftsquarebracket_').replace(']','_rightsquarebracket_').replace('-','_dash_').replace('.','_dot_').replace('(','_leftroundbracket').replace(')','_rightroundbracket_') for c in df_bac_renamed.columns]

    # add pseudo abundances to zeros
    for sample_id in df_bac_renamed.index:
        sample = np.array(df_bac_renamed.loc[sample_id])
        minval = np.min(sample[np.nonzero(sample)]) # minimum non-zero value
        sample[sample==0] = minval
        df_bac_renamed.loc[sample_id] = sample

    # select the topN most abundant taxa
    df_bac_renamed_T = df_bac_renamed.loc[df_meta.index].T
    df_bac_renamed_T['mean'] = df_bac_renamed_T.mean(axis=1)
    df_bac_renamed_T = df_bac_renamed_T.sort_values(by=['mean'],axis=0,ascending=False)
    df_bac_renamed_T = df_bac_renamed_T.drop('mean', axis=1)
    df_bac_renamed = df_bac_renamed_T.iloc[0:topN].T

    # normalize bacterial abundance (maximum -> 1)
    selected_bacterial_taxa = list(df_bac_renamed.columns)
    df_bac_renamed_w_meta = pd.merge(df_meta, df_bac_renamed/df_bac_renamed.max().max(), left_index=True, right_index=True, how='inner')

    # if there are duplicate samples for the same subject, average the data
    df_bac_renamed_w_meta = df_bac_renamed_w_meta.groupby(['SubjectID','Day']).agg(np.mean).reset_index()

    # remove samples that have single data (at least two data is required)
    subjects_to_remove = []
    for curr_subject in set(df_bac_renamed_w_meta.SubjectID):
        if len(df_bac_renamed_w_meta[df_bac_renamed_w_meta.SubjectID==curr_subject])<2:
            subjects_to_remove.append(curr_subject)
    df_bac_renamed_w_meta = df_bac_renamed_w_meta[~df_bac_renamed_w_meta.SubjectID.isin(subjects_to_remove)]

    # calculate log-derivatives of bacterial abundance
    df_bac_deriv = deepcopy(df_bac_renamed_w_meta)
    for curr_subject in set(df_bac_deriv.SubjectID):
        curr_df = df_bac_deriv[df_bac_deriv.SubjectID==curr_subject].sort_values(by='Day')
        for taxon in selected_bacterial_taxa:
            xdata = np.array(curr_df['Day'])
            ydata = np.array(curr_df[taxon])
            cs = CubicSpline(xdata, ydata)
            csd1 = cs.derivative(nu=1)
            ydata_d1 = csd1(xdata)
            df_bac_deriv.loc[df_bac_deriv.SubjectID==curr_subject, taxon] = ydata_d1

    # construct regression matrix
    Ymat = df_bac_deriv[selected_bacterial_taxa].values
    Ymat = Ymat.flatten(order='F')
    Ymat = StandardScaler().fit_transform(Ymat.reshape(-1,1)).reshape(1,-1)[0] # standardize

    Xmat = np.zeros(shape=(topN*len(df_bac_deriv.index), (topN+2)*topN))
    for k in np.arange(topN):
        Xmat[k*len(df_bac_deriv.index):(k+1)*len(df_bac_deriv.index),k*(topN+2)] = 1
        Xmat[k*len(df_bac_deriv.index):(k+1)*len(df_bac_deriv.index),k*(topN+2)+1] = df_bac_deriv.Dose.values
        Xmat[k*len(df_bac_deriv.index):(k+1)*len(df_bac_deriv.index),k*(topN+2)+2:(k+1)*(topN+2)] = df_bac_renamed_w_meta[selected_bacterial_taxa].values

    # write data to stan program files
    json_str = '{\n"N" : %d,\n'%(len(Ymat))
    json_str += '\"dlogX\" : [%s],\n'%(','.join(list(Ymat.astype(str))))
    for k1,c1 in enumerate(selected_bacterial_taxa):
        # growth rate
        json_str += '\"growth_rate_%s\" : [%s],\n'%(c1,','.join(list(Xmat[:,k1*(topN+2)].astype(str))))
        # diet response
        json_str += '\"fiber_response_%s\" : [%s],\n'%(c1,','.join(list(Xmat[:,k1*(topN+2)+1].astype(str))))
        # bacterial interactions
        for k2,c2 in enumerate(selected_bacterial_taxa):
            v = list(Xmat[:,k1*(topN+2)+2+k2].astype(str))
            json_str += '\"pairwise_interaction_%s_%s\" : [%s]'%(c1,c2,','.join(v))
            if c1 == selected_bacterial_taxa[-1] and c2 == selected_bacterial_taxa[-1]:
                json_str += '\n}'
            else:
                json_str += ',\n'
    text_file = open("%s/%s.data.json"%(stan_path, prefix), "w")
    text_file.write("%s" % json_str)
    text_file.close()

    # write stan program
    # data block
    model_str = 'data {\n'
    model_str += '\tint<lower=0> N;\n'
    model_str += '\tvector[N] dlogX;\n'
    for c1 in selected_bacterial_taxa:
        model_str += '\tvector[N] growth_rate_%s;\n'%(c1)
        model_str += '\tvector[N] fiber_response_%s;\n'%(c1)
        for c2 in selected_bacterial_taxa:
            model_str += '\tvector[N] pairwise_interaction_%s_%s;\n'%(c1,c2)
    model_str += '}\n'

    # parameter block
    model_str += 'parameters {\n\treal<lower=0,upper=1> sigma;\n'
    for c1 in selected_bacterial_taxa:
        model_str += '\treal alpha__%s;\n'%(c1) # growth rate
        model_str += '\treal epsilon__%s;\n'%(c1) # inulin response
        for c2 in selected_bacterial_taxa:
            model_str += '\treal beta__%s_%s;\n'%(c1,c2)
    model_str += '}\n'

    # model block
    model_str += 'model {\n\tsigma ~ uniform(0,1);\n'
    for c1 in selected_bacterial_taxa:
        model_str += '\talpha__%s ~ normal(0,1);\n'%(c1) # growth rate
        model_str += '\tepsilon__%s ~ normal(0,1);\n'%(c1) # inulin response
        for c2 in selected_bacterial_taxa:
            model_str += '\tbeta__%s_%s ~ normal(0,1);\n'%(c1,c2)
    model_str += '\tdlogX ~ normal('
    for c1 in selected_bacterial_taxa:
        model_str += 'alpha__%s*growth_rate_%s+'%(c1,c1) # growth rate
        model_str += 'epsilon__%s*fiber_response_%s+'%(c1,c1) # inulin response
        for c2 in selected_bacterial_taxa:
            if c1 == selected_bacterial_taxa[-1] and c2 == selected_bacterial_taxa[-1]:
                model_str += 'beta__%s_%s*pairwise_interaction_%s_%s'%(c1,c2,c1,c2)
            else:
                model_str += 'beta__%s_%s*pairwise_interaction_%s_%s+'%(c1,c2,c1,c2)
    model_str += ', sigma);\n}'
    text_file = open("%s/%s.stan"%(stan_path, prefix), "w")
    text_file.write("%s" % model_str)
    text_file.close()

    return selected_bacterial_taxa

def hpd_grid(sample, alpha=0.05, roundto=2):
    """Calculate highest posterior density (HPD) of array for given alpha.
    The HPD is the minimum width Bayesian credible interval (BCI).
    The function works for multimodal distributions, returning more than one mode
    Parameters
    ----------

    sample : Numpy array or python list
        An array containing MCMC samples
    alpha : float
        Desired probability of type I error (defaults to 0.05)
    roundto: integer
        Number of digits after the decimal point for the results
    Returns
    ----------
    hpd: array with the lower

    """
    sample = np.asarray(sample)
    sample = sample[~np.isnan(sample)]
    # get upper and lower bounds
    l = np.min(sample)
    u = np.max(sample)
    density = kde.gaussian_kde(sample)
    x = np.linspace(l, u, 2000)
    y = density.evaluate(x)
    #y = density.evaluate(x, l, u) waitting for PR to be accepted
    xy_zipped = zip(x, y/np.sum(y))
    xy = sorted(xy_zipped, key=lambda x: x[1], reverse=True)
    xy_cum_sum = 0
    hdv = []
    for val in xy:
        xy_cum_sum += val[1]
        hdv.append(val[0])
        if xy_cum_sum >= (1-alpha):
            break
    hdv.sort()
    diff = (u-l)/20  # differences of 5%
    hpd = []
    hpd.append(round(min(hdv), roundto))
    for i in range(1, len(hdv)):
        if hdv[i]-hdv[i-1] >= diff:
            hpd.append(round(hdv[i-1], roundto))
            hpd.append(round(hdv[i], roundto))
    hpd.append(round(max(hdv), roundto))
    ite = iter(hpd)
    hpd = list(zip(ite, ite))
    modes = []
    for value in hpd:
        x_hpd = x[(x > value[0]) & (x < value[1])]
        y_hpd = y[(x > value[0]) & (x < value[1])]
        modes.append(round(x_hpd[np.argmax(y_hpd)], roundto))
    return hpd, x, y, modes

def get_rf_training_error(
    df_meta,
    df_scfa,
    df_bac,
    target_scfa,
    topN,
    exclude_group,
    exclude_vendor,
    use_deriv_scfa,
    use_deriv_microbiome,
    use_pcoa
):
    df_train = None

    # get best RF model hyperparameters
    if exclude_group is not None:
        df_opt_paras = pd.read_csv('optimal_rf_parameters_exclude_group%s.csv'%(exclude_group), index_col=0)
    if exclude_vendor is not None:
        df_opt_paras = pd.read_csv('optimal_rf_parameters_exclude_vendor%s.csv'%(exclude_vendor), index_col=0)

    # retrain the model
    selected_topN_bac, df_meta_sliced, df_bac_sliced, df_bac_deriv, df_scfa_sliced, df_scfa_deriv = data_processing_scfa(
        df_meta, df_bac, df_scfa, target_scfa, topN=topN, exclude_group=exclude_group, exclude_vendor=exclude_vendor, use_pcoa=use_pcoa)
    _,_,reg = train_scfa_dynamics_model(
        df_meta=df_meta,
        df_bac=df_bac,
        df_scfa=df_scfa,
        target_scfa=target_scfa,
        topN=topN,
        exclude_group=exclude_group,
        exclude_vendor=exclude_vendor,
        model='RandomForest',
        opt_params = df_opt_paras,
        use_deriv_scfa=use_deriv_scfa,
        use_deriv_microbiome=use_deriv_microbiome,
        use_pcoa=use_pcoa
    )

    # get predicted SCFA derivative of the same training dataset
    for scfa_ in target_scfa:
        if use_deriv_scfa:
            df_train_tmp = deepcopy(df_scfa_deriv)
        else:
            df_train_tmp = deepcopy(df_scfa_sliced)
        df_train_tmp = df_train_tmp[[x for x in df_train_tmp.columns if x not in list(set(target_scfa)-set([scfa_]))]]
        df_train_tmp = df_train_tmp.rename({scfa_:'SCFA_observed'}, axis=1)
        df_train_tmp['SCFA_mol'] = scfa_
        if use_deriv_microbiome is None:
            X_var = np.asarray(binarize_categories(list(df_scfa_sliced.Day)).values)
        else:
            if use_deriv_microbiome:
                X_var = np.asarray(df_bac_deriv.values)
            else:
                X_var = np.asarray(df_bac_sliced.values)
        df_train_tmp['SCFA_predicted'] = reg[scfa_].predict(X_var)

        if df_train is None:
            df_train = df_train_tmp
        else:
            df_train = pd.concat([df_train, df_train_tmp], ignore_index=True)

    df_train['RelativeError_SCFA'] = (df_train['SCFA_predicted']-df_train['SCFA_observed'])/df_train['SCFA_observed']*100
    return df_train

def get_rf_prediction_error(
    df_meta,
    df_scfa,
    df_bac,
    target_scfa,
    prediction_type,
    topN,
    use_deriv_scfa,
    use_deriv_microbiome,
    use_pcoa,
    is_plot=False,
    save_fig=False
):
    df_pred = None
    if prediction_type=='intrapolation':
        exclude_set = np.sort(list(set(df_meta.RandomizedGroup)))
        exclude_set = [lett for lett in exclude_set if lett in ['A','B','C','D']] # we never use group E as exclude set
        if is_plot:
            fig, ax = plt.subplots(figsize=(20, 3*len(exclude_set)), nrows=len(exclude_set), ncols=4, sharex=True)
    elif prediction_type=='extrapolation':
        exclude_set = np.sort(list(set(df_meta.Vendor)))
        if is_plot:
            fig, ax = plt.subplots(figsize=(20, 3*len(exclude_set)), nrows=len(exclude_set), ncols=5, sharex=True)
    else:
        print('unknown prediction type %s'%(prediction_type))
        raise

    if is_plot:
        scfa_color={'Acetate':'#DB5E56', 'Butyrate':'#56DB5E', 'Propionate':'#5E56DB'}

    # get SCFA and microbiome derivative for all mice
    # keep all taxa at this step
    selected_topN_bac, df_meta_sliced, df_bac_sliced, df_bac_deriv, df_scfa_sliced, df_scfa_deriv = data_processing_scfa(df_meta=df_meta, df_bac=df_bac, df_scfa=df_scfa, target_scfa=target_scfa, topN=len(df_bac.columns), exclude_group=None, exclude_vendor=None, use_pcoa=use_pcoa)

    # rename columns of df_scfa_deriv
    df_scfa_sliced = df_scfa_sliced.rename({x:x+'_value_observed' for x in target_scfa}, axis=1)
    df_scfa_deriv = df_scfa_deriv.rename({x:x+'_deriv_observed' for x in target_scfa}, axis=1)

    # get prediction for excluded dataset in training
    for idx_i,to_exclude in enumerate(exclude_set):
        # get trained model
        if prediction_type=='intrapolation':
            df_opt_paras = pd.read_csv('intrapolation/optimal_rf_parameters_exclude_group%s.csv'%(to_exclude), index_col=0)
        elif prediction_type=='extrapolation':
            df_opt_paras = pd.read_csv('extrapolation/optimal_rf_parameters_exclude_vendor%s.csv'%(to_exclude), index_col=0)
        else:
            print('unknown prediction type %s'%(prediction_type))
            raise
        _,_,reg = train_scfa_dynamics_model(
            df_meta=df_meta,
            df_bac=df_bac,
            df_scfa=df_scfa,
            target_scfa=target_scfa,
            topN=topN,
            exclude_group=to_exclude if prediction_type=='intrapolation' else None,
            exclude_vendor=to_exclude if prediction_type=='extrapolation' else None,
            model='RandomForest',
            opt_params=df_opt_paras,
            use_deriv_scfa=use_deriv_scfa,
            use_deriv_microbiome=use_deriv_microbiome,
            use_pcoa=use_pcoa
        )

        # rejoin sliced tables but only keep samples in the test dataset
        df_sliced_ext = deepcopy(df_meta_sliced)
        if prediction_type=='intrapolation':
            df_sliced_ext = df_sliced_ext[df_sliced_ext.RandomizedGroup==to_exclude]
        elif prediction_type=='extrapolation':
            df_sliced_ext = df_sliced_ext[df_sliced_ext.Vendor==to_exclude]
        else:
            print('unknown prediction type %s'%(prediction_type))
            raise
        df_sliced_ext = pd.merge(df_sliced_ext, df_scfa_sliced, left_index=True, right_index=True, how='inner')
        df_sliced_ext = pd.merge(df_sliced_ext, df_scfa_deriv, left_index=True, right_index=True, how='inner')
        if use_deriv_microbiome is not None:
            if use_deriv_microbiome:
                df_sliced_ext = pd.merge(df_sliced_ext, df_bac_deriv, left_index=True, right_index=True, how='inner')
            else:
                df_sliced_ext = pd.merge(df_sliced_ext, df_bac_sliced, left_index=True, right_index=True, how='inner')
        else:
            df_day_pres = binarize_categories(list(df_sliced_ext.Day))
            for d in df_day_pres.columns:
                df_sliced_ext[d] = list(df_day_pres[d])

        # remove duplicate columns
        df_sliced_ext = df_sliced_ext.drop([c for c in df_sliced_ext.columns if '_x' in str(c) or '_y' in str(c) or '_z' in str(c)], axis=1)

        # predict SCFA derivative and SCFA value
        all_mice = set(df_sliced_ext['SubjectID'])
        for scfa_ in target_scfa:
            topN_taxa = reg[scfa_].feature_names
            X_var = np.asarray(df_sliced_ext[topN_taxa].values)
            if use_deriv_scfa:
                df_sliced_ext['%s_deriv_predicted'%(scfa_)] = reg[scfa_].predict(X_var)
            else:
                df_sliced_ext['%s_value_predicted'%(scfa_)] = reg[scfa_].predict(X_var)

            for idx_j, curr_mice in enumerate(all_mice):
                df_tmp = df_sliced_ext[df_sliced_ext.SubjectID==curr_mice].sort_values(by='Day')
                if use_deriv_scfa:
                    df_tmp = df_tmp[['SubjectID','Vendor','Day','RandomizedGroup',scfa_+'_value_observed',scfa_+'_deriv_observed',scfa_+'_deriv_predicted']+topN_taxa]
                    df_tmp = df_tmp.rename({scfa_+'_value_observed':'SCFA_value_observed',
                                            scfa_+'_deriv_observed':'SCFA_deriv_observed',
                                            scfa_+'_deriv_predicted':'SCFA_deriv_predicted'},
                                           axis=1)
                else:
                    df_tmp = df_tmp[['SubjectID','Vendor','Day','RandomizedGroup',scfa_+'_value_observed',scfa_+'_deriv_observed',scfa_+'_value_predicted']+topN_taxa]
                    df_tmp = df_tmp.rename({scfa_+'_value_observed':'SCFA_value_observed',
                                            scfa_+'_deriv_observed':'SCFA_deriv_observed',
                                            scfa_+'_value_predicted':'SCFA_value_predicted'},
                                           axis=1)
                df_tmp['SCFA_mol'] = scfa_

                if use_deriv_scfa==False:
                    # calcualte derivative using predicted SCFA value
                    xdata = np.asarray(list(df_tmp['Day']))
                    ydata = np.asarray(list(df_tmp['SCFA_value_predicted']))
                    cs = CubicSpline(xdata, ydata)
                    csd1 = cs.derivative(nu=1)
                    ydata_d1 = csd1(xdata)
                    df_tmp['SCFA_deriv_predicted'] = ydata_d1
                else:
                    # integration is needed to calculate SCFA value from SCFA derivative
                    init_scfa_value = df_tmp.loc[df_tmp.Day==0,'SCFA_value_observed'].values[0]

                    # get cubic spliner for each bacterial taxa
                    tck = {}
                    for taxa in topN_taxa:
                        tck[taxa] = CubicSpline(df_tmp.Day, df_tmp[taxa])

                    # remove taxa in df_tmp
                    df_tmp = df_tmp[[x for x in df_tmp.columns if x not in topN_taxa]]

                    # function to be used in ODE solver
                    def f_solve_ivp(t, y, reg_scfa_, tck):
                        x = []
                        for feature in reg_scfa_.feature_names:
                            x.append(tck[feature](t))
                        deriv = reg_scfa_.predict([x])
                        if y<=0 and deriv<0:
                            deriv=0
                        return deriv

                    # solve the model
                    xcorr = np.linspace(0,31,311)
                    sol = solve_ivp(f_solve_ivp, [0,31], [init_scfa_value], args=(reg[scfa_], tck), method='RK45', t_eval=xcorr)
                    sol_t = sol.t
                    sol_y = sol.y[0]

                    # interpolate the values on the day of observataion
                    cs = CubicSpline(sol_t, sol_y)
                    df_tmp['SCFA_value_predicted'] = cs(df_tmp.Day)

                if df_pred is None:
                    df_pred = df_tmp
                else:
                    df_pred = pd.concat([df_pred, df_tmp], ignore_index=True)

                # remove topN taxa
                df_pred = df_pred[[x for x in df_pred.columns if x not in topN_taxa]]

                if is_plot:
                    _ = ax[idx_i,idx_j].scatter(df_tmp.Day, df_tmp.SCFA_value_observed, marker='o', color=scfa_color[scfa_], s=100, label=scfa_)
                    _ = ax[idx_i,idx_j].plot(df_tmp.Day, df_tmp.SCFA_value_predicted, '-', marker="s", markersize=10, color=scfa_color[scfa_])
                    _ = ax[idx_i,idx_j].set_title(curr_mice)
                    if idx_i==0 and idx_j==0:
                        ax[idx_i,idx_j].legend()
                    ax[idx_i,idx_j].set_xlim([-3.2,32])
                    if idx_j==0:
                        ax[idx_i,idx_j].set_ylabel('SCFA', fontsize=15)
                    if idx_i==3:
                        ax[idx_i,idx_j].set_xlabel('Day', fontsize=15)

    if is_plot:
        plt.tight_layout()
        plt.rcParams['svg.fonttype'] = 'none'

    if save_fig:
        fig.savefig("rf_prediction_error_%s.svg"%(prediction_type), format="svg")

    df_pred['RelativeError_SCFA_value'] = (df_pred['SCFA_value_predicted']-df_pred['SCFA_value_observed'])/df_pred['SCFA_value_observed']*100
    df_pred['RelativeError_SCFA_deriv'] = (df_pred['SCFA_deriv_predicted']-df_pred['SCFA_deriv_observed'])/df_pred['SCFA_deriv_observed']*100
    return df_pred

def smape(A, F):
    return 100/len(A) * np.sum(np.abs(F - A) / (np.abs(A) + np.abs(F)))
