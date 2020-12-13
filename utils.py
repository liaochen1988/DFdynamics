import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.interpolate import CubicSpline
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings("ignore")

# Note: all samples in df_meta should be consistent in their diets
def data_processing_scfa(
    df_meta, # meta data
    df_bac,  # relative abundace or absolute abundance of gut microbiome
    df_scfa, # SCFA measurement
    target_scfa, # dependent variable(s) in the regression
    topN,    # keep only the most abundance N taxa in the model
    normalize_X, # normalize maximum of bacterial abundance to 1
    exclude_group, # group of mice excluded from model training
    exclude_vendor # vendor of mice excluded from model training
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
    for curr_mice in set(df_scfa_deriv.MiceID):
        curr_df = df_scfa_meta[df_scfa_meta.MiceID==curr_mice].sort_values(by='Day')
        xdata = np.array(curr_df['Day'])
        for scfa_ in target_scfa_sliced:
            ydata = np.array(curr_df[scfa_])
            cs = CubicSpline(xdata, ydata)
            csd1 = cs.derivative(nu=1)
            ydata_d1 = csd1(xdata)
            df_scfa_deriv.loc[df_scfa_deriv.MiceID==curr_mice, scfa_] = ydata_d1

    # keep only samples in df_meta_sliced for bacterial abundance data
    df_bac_sliced = df_bac.loc[df_meta_sliced.index]

    # select the topN taxa based on averaged abundance
    df_bac_sliced_T = df_bac_sliced.T
    df_bac_sliced_T['mean'] = df_bac_sliced_T.mean(axis=1)
    df_bac_sliced_T = df_bac_sliced_T.sort_values(by=['mean'], axis=0, ascending=False)
    df_bac_sliced_T = df_bac_sliced_T.drop('mean', axis=1)
    df_bac_sliced = df_bac_sliced_T.iloc[0:topN].T

    # normalize max value of bacterial abundance to 1
    if normalize_X:
        df_bac_sliced = df_bac_sliced/df_bac_sliced.max().max()

    return target_scfa_sliced, df_meta_sliced, df_bac_sliced, df_scfa_sliced, df_scfa_deriv

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
    feedback=False # if True, add SCFA feedback, i.e., dSCFA/dt = f(microbiome, SCFA)
):
    # get processed input data
    target_scfa_sliced, df_meta_sliced, df_bac_sliced, df_scfa_sliced, df_scfa_deriv = data_processing_scfa(df_meta, df_bac, df_scfa, target_scfa, topN, normalize_X, exclude_group, exclude_vendor)

    # train specified model on the data
    if model=='Correlation':
        lines = []
        for scfa_ in target_scfa_sliced:
            for t in list(df_bac_sliced.columns):
                corr_p, pvalue_p = pearsonr(df_scfa_deriv[scfa_], df_bac_sliced[t])
                corr_s, pvalue_s = spearmanr(df_scfa_deriv[scfa_], df_bac_sliced[t])
                lines.append([scfa_, t, corr_p, pvalue_p, corr_s, pvalue_s])
        df_output = pd.DataFrame(lines, columns=['SCFA','Taxa','PearsonR','PearsonP','SpearmanR','SpearmanP'])
        return df_output
    elif model=='ElasticNet':
        lines = []
        regression_model = {}
        for scfa_ in target_scfa_sliced:
            if feedback:
                X_var = np.concatenate((np.asarray(df_bac_sliced.values), np.asarray(df_scfa_sliced[scfa_]).reshape(-1,1)), 1)
                X_var_names = list(df_bac_sliced.columns) + ['SCFA_fdb']
            else:
                X_var = np.asarray(df_bac_sliced.values)
                X_var_names = list(df_bac_sliced.columns)
            Y_var = np.asarray(list(df_scfa_deriv[scfa_]))
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
            if feedback:
                X_var = np.concatenate((np.asarray(df_bac_sliced.values), np.asarray(df_scfa_sliced[scfa_]).reshape(-1,1)), 1)
                X_var_names = list(df_bac_sliced.columns) + ['SCFA_fdb']
            else:
                X_var = np.asarray(df_bac_sliced.values)
                X_var_names = list(df_bac_sliced.columns)
            Y_var = np.asarray(list(df_scfa_deriv[scfa_]))

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
            reg.feature_names = X_var_names # add feature names
            lines_reg.append([scfa_, clf.score(X_var, Y_var)]+ list(clf.feature_importances_))
            regression_model[scfa_] = reg
        df_output_opt = pd.DataFrame(lines_opt, columns=['SCFA','n_estimators','max_features','max_depth','min_samples_split','min_samples_leaf','bootstrap'])
        df_output_reg = pd.DataFrame(lines_reg, columns=['SCFA','R2']+X_var_names)
        return df_output_reg, df_output_opt, regression_model
    else:
        print('unknown method: %s'%(method))
        raise
