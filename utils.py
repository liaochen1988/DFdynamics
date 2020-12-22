import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.interpolate import CubicSpline
from scipy.stats import pearsonr, spearmanr
import scipy.stats.kde as kde
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
    use_deriv_scfa, # use derivative for SCFA
    use_deriv_microbiome # use derivative for microbiome
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
    target_scfa_sliced = [x for x in target_scfa if x in list(df_scfa.columns)]
    if len(target_scfa_sliced) == 0:
        print('cannot find any SCFA.')
        raise

    # make sure that meta data and SCFA have the same samples in the same order
    shared_samples = [x for x in df_meta_sliced.index if x in list(df_scfa.index)]
    df_scfa_sliced = df_scfa.loc[shared_samples, target_scfa_sliced]
    df_meta_sliced = df_meta_sliced.loc[shared_samples]

    # calculate SCFA derivatives
    df_scfa_meta = pd.merge(df_meta_sliced, df_scfa_sliced, left_index=True, right_index=True, how='inner')
    df_scfa_deriv = deepcopy(df_scfa_meta)
    if use_deriv_scfa:
        for curr_subject in set(df_scfa_deriv.SubjectID):
            curr_df = df_scfa_meta[df_scfa_meta.SubjectID==curr_subject].sort_values(by='Day')
            xdata = np.array(curr_df['Day'])
            for scfa_ in target_scfa_sliced:
                ydata = np.array(curr_df[scfa_])
                cs = CubicSpline(xdata, ydata)
                csd1 = cs.derivative(nu=1)
                ydata_d1 = csd1(xdata)
                df_scfa_deriv.loc[df_scfa_deriv.SubjectID==curr_subject, scfa_] = ydata_d1

    # keep only samples in df_meta_sliced for bacterial abundance data
    df_bac_sliced = df_bac.loc[df_meta_sliced.index]

    # select the topN taxa based on averaged abundance
    df_bac_sliced_T = df_bac_sliced.T
    df_bac_sliced_T['mean'] = df_bac_sliced_T.mean(axis=1)
    df_bac_sliced_T = df_bac_sliced_T.sort_values(by=['mean'], axis=0, ascending=False)
    df_bac_sliced_T = df_bac_sliced_T.drop('mean', axis=1)
    df_bac_sliced = df_bac_sliced_T.iloc[0:topN].T
    selected_topN_bac = list(df_bac_sliced.columns)

    # calculate Microbiome derivative
    df_bac_meta = pd.merge(df_meta_sliced, df_bac_sliced, left_index=True, right_index=True, how='inner')
    df_bac_deriv = deepcopy(df_bac_meta)
    if use_deriv_microbiome:
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

    return target_scfa_sliced, selected_topN_bac, df_meta_sliced, df_bac_sliced, df_bac_deriv, df_scfa_sliced, df_scfa_deriv

def train_scfa_dynamics_model(
    df_meta, # meta data
    df_bac,  # relative abundace or absolute abundance of gut microbiome
    df_scfa, # SCFA measurement
    target_scfa, # dependent variable(s) in the regression
    topN=40, # keep only the most abundance N taxa in the model
    normalize_X=True, # normalize maximum of bacterial abundance to 1
    exclude_group=None, # group of mice excluded from model training
    exclude_vendor=None, # group of mice excluded from model training
    model='Correlation',# regression model
    opt_params = None, # optimal model parameters
    addVar=None, # options: time or SCFA
    use_deriv_scfa=True, # whether using dSCFA/dt as the dependent variable
    use_deriv_microbiome=False, # whether dMicrobiome/dt as the independent variable
):
    # get processed input data
    target_scfa_sliced, selected_topN_bac, df_meta_sliced, df_bac_sliced, df_bac_deriv, df_scfa_sliced, df_scfa_deriv = data_processing_scfa(df_meta, df_bac, df_scfa, target_scfa, topN, exclude_group, exclude_vendor, use_deriv_scfa, use_deriv_microbiome)

    # set X variable column names
    if addVar is not None:
        X_var_names = selected_topN_bac + ['AddVar']
    else:
        X_var_names = selected_topN_bac

    # train specified model on the data
    if model=='Correlation':
        if normalize_X:
            df_bac_deriv = df_bac_deriv/df_bac_deriv.max().max()
        lines = []
        for scfa_ in target_scfa_sliced:
            for t in selected_topN_bac:
                corr_p, pvalue_p = pearsonr(df_scfa_deriv[scfa_], df_bac_deriv[t])
                corr_s, pvalue_s = spearmanr(df_scfa_deriv[scfa_], df_bac_deriv[t])
                lines.append([scfa_, t, corr_p, pvalue_p, corr_s, pvalue_s])
        df_output = pd.DataFrame(lines, columns=['SCFA','Taxa','PearsonR','PearsonP','SpearmanR','SpearmanP'])
        return df_output
    elif model=='ElasticNet':
        lines = []
        regression_model = {}
        for scfa_ in target_scfa_sliced:
            if addVar is not None:
                if addVar=='SCFA':
                    X_var = np.concatenate((np.asarray(df_bac_deriv.values), np.asarray(df_scfa_sliced[scfa_]).reshape(-1,1)), 1)
                elif addVar=='time':
                    X_var = np.concatenate((np.asarray(df_bac_deriv.values), np.asarray(df_meta_sliced['Day']).reshape(-1,1)), 1)
                else:
                    print('unknown additional variable %s'%(addVar))
                    raise
            else:
                X_var = np.asarray(df_bac_deriv.values)
            Y_var = np.asarray(list(df_scfa_deriv[scfa_]))

            if normalize_X:
                X_var = X_var/X_var.max().max()

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
            reg.feature_names = X_var_names
            regression_model[scfa_] = reg
            lines.append([scfa_, best_alpha, best_l1_ratio, clf.score(X_var, Y_var)]+list(clf.coef_))
        df_output = pd.DataFrame(lines, columns=['SCFA','BestAlpha','BestL1Ratio','R2']+X_var_names)
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
        for scfa_ in target_scfa_sliced:
            if addVar is not None:
                if addVar=='SCFA':
                    X_var = np.concatenate((np.asarray(df_bac_deriv.values), np.asarray(df_scfa_sliced[scfa_]).reshape(-1,1)), 1)
                elif addVar=='time':
                    X_var = np.concatenate((np.asarray(df_bac_deriv.values), np.asarray(df_meta_sliced['Day']).reshape(-1,1)), 1)
                else:
                    print('unknown additional variable %s'%(addVar))
                    raise
            else:
                X_var = np.asarray(df_bac_deriv.values)
            Y_var = np.asarray(list(df_scfa_deriv[scfa_]))

            if normalize_X:
                X_var = X_var/X_var.max().max()

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
            reg.feature_names = deepcopy(X_var_names) # add feature names
            lines_reg.append([scfa_, clf.score(X_var, Y_var)]+ list(clf.feature_importances_))
            regression_model[scfa_] = deepcopy(reg)
        df_output_opt = pd.DataFrame(lines_opt, columns=['SCFA','n_estimators','max_features','max_depth','min_samples_split','min_samples_leaf','bootstrap'])
        df_output_reg = pd.DataFrame(lines_reg, columns=['SCFA','R2']+X_var_names)
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
