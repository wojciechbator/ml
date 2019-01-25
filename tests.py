import pandas as pd
import time
import numpy as np
import warnings

warnings.filterwarnings('ignore')

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold, RFE, SelectPercentile, SelectKBest, \
    SelectFromModel, f_classif
from sklearn.linear_model import LassoCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import recall_score, precision_score, f1_score


def get_data(filename, labels_last=True, header=None, sep=','):
    ds = pd.read_csv(f'./datasets/{filename}', header=header, sep=sep)

    char_cols = ds.dtypes.pipe(lambda x: x[x == 'object']).index
    for col in char_cols:
        ds[col] = pd.factorize(ds[col])[0]

    features = ds.iloc[:, 0:-1] if labels_last else ds.iloc[:, 1:len(ds)]
    labels = ds.iloc[:, -1] if labels_last else ds.iloc[:, 0]

    return features, labels


# DATASETS

iris_ds = get_data('iris.data')
nursery_ds = get_data('nursery.data')
transfusion_ds = get_data('transfusion.data')
eighthr_ds = get_data('eighthr.data')
sonar_ds = get_data('sonar.data')
congress_ds = get_data('house-votes-84.data')
tae_ds = get_data('tae.data')
flag_ds = get_data('flag.data')
bridges_ds = get_data('bridges.data')
bands_ds = get_data('bands.data')
cmc_ds = get_data('cmc.data')
krkopt_ds = get_data('krkopt.data')
spambase_ds = get_data('spambase.data')
primary_tumor_ds = get_data('primary-tumor.data')
lymphography_ds = get_data('lymphography.data')
hayes_roth_ds = get_data('hayes-roth.data')
haberman_ds = get_data('haberman.data')
glass_ds = get_data('glass.data')
dermatology_ds = get_data('dermatology.data')
hepatitis_ds = get_data('hepatitis.data')
hungarian_heart_disease_ds = get_data('hungarian_heart_disease.data')
balance_scale_ds = get_data('balance-scale.data')
breast_cancer_ds = get_data('breast-cancer.data')
adult_ds = get_data('adult.data')
wine_ds = get_data('wine.data', False)
wine_quality_red_ds = get_data('winequality-red.csv', header=0, sep=';')
bank_ds = get_data('bank.csv', header=0, sep=';')
car_ds = get_data('car.data')
heart_disease_ds = get_data('processed.cleveland.data')
abalone_ds = get_data('abalone.data')


# FEATURE SELECTORS

# select from model
estimator = LassoCV(cv=5)
sfm_selector = SelectFromModel(estimator)
select_from_model = ('SelectFromModel', sfm_selector)

# select k best
skb_selector = SelectKBest(f_classif, k=2)
select_k_best = ('SelectKBest', skb_selector)

# recursive feature elimination
estimator = LassoCV(cv=5)
rfe_selector = RFE(estimator)
recursive_feature_elimination = ('RecursiveFeatureElimination', rfe_selector)

# variance threshold
vt_selector = VarianceThreshold(threshold=.5)
variance_threshold = ('VarianceThreshold', vt_selector)

# select percentile
sp_selector = SelectPercentile(f_classif, percentile=10)
select_percentile = ('SelectPercentile', sp_selector)

selectors = [
    select_from_model,
    select_k_best,
    recursive_feature_elimination,
    variance_threshold,
    select_percentile
]


# CLASSIFIERS

nb_datasets = [
    ('wine_quality_red', wine_quality_red_ds),
    ('nursery', nursery_ds),
    ('balance scale', balance_scale_ds),
    ('king rook vs king', krkopt_ds),
    ('bank', bank_ds),
    ('dermatology', dermatology_ds),
    ('hayes roth', hayes_roth_ds),
    ('bridges', bridges_ds),
    ('flag', flag_ds),
    ('sonar', sonar_ds)
]
nb_clf = ('NaiveBayes', GaussianNB(), nb_datasets)

svm_datasets = [
    ('glass', glass_ds),
    ('iris', iris_ds),
    ('car', car_ds),
    ('hepatitis', hepatitis_ds),
    ('primary tumor', primary_tumor_ds),
    ('heart disease', heart_disease_ds),
    ('abalone', abalone_ds),
    ('cmc', cmc_ds),
    ('tae', tae_ds),
    ('eighthr', eighthr_ds)
]
svm_clf = ('SVM', SVC(kernel='linear', gamma='auto'), svm_datasets)

rf_datasets = [
    ('wine', wine_ds),
    ('adult', adult_ds),
    ('breast cancer', breast_cancer_ds),
    ('haberman', haberman_ds),
    ('lymphography', lymphography_ds),
    ('spambase', spambase_ds),
    ('bands', bands_ds),
    ('congress', congress_ds),
    ('transfusion', transfusion_ds)
]
rf_clf = ('RandomForest', RandomForestClassifier(n_estimators=10), rf_datasets)

classifiers = [
    nb_clf,
    svm_clf,
    rf_clf
]


# MAGIC

def run():
    k = 10
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    for clf_name, clf, datasets in classifiers:
        csv_measure = []
        csv_df = pd.DataFrame(data=csv_measure)
        print('\n############################################')
        print(clf_name)
        for ds_name, ds in datasets:
            print('============================================')
            print(ds_name)

            metrics = {}
            for train_index, test_index in kf.split(ds[0]):

                f_train = ds[0].iloc[train_index]
                l_train = ds[1].iloc[train_index]
                f_test = ds[0].iloc[test_index]
                l_test = ds[1].iloc[test_index]

                for sel_name, sel in selectors:
                    if sel_name not in metrics:
                        metrics[sel_name] = {}
                        metrics[sel_name]['no_of_features'] = f_test.shape[1]
                        metrics[sel_name]['no_of_features_selected'] = 0
                        metrics[sel_name]['accuracy'] = 0
                        metrics[sel_name]['time_elapsed_ms'] = 0
                        metrics[sel_name]['precision'] = 0
                        metrics[sel_name]['recall'] = 0
                        metrics[sel_name]['f1'] = 0
                    
                    sel_start = time.time()
                    sel_f_train, sel_f_test = selector_fit_and_transform(sel, f_train, f_test,
                                                                         l_train)
                    clf.fit(sel_f_train, l_train)

                    pred = clf.predict(sel_f_test)

                    sel_end = time.time()

                    metrics[sel_name]['no_of_features_selected'] += len(sel_f_test[0])
                    metrics[sel_name]['accuracy'] += clf.score(sel_f_test, l_test)
                    metrics[sel_name]['time_elapsed_ms'] += int(round((sel_end - sel_start)*1000))
                    metrics[sel_name]['precision'] += precision_score(l_test.values, pred, average='macro')
                    metrics[sel_name]['recall'] += recall_score(l_test.values, pred, average='macro')
                    metrics[sel_name]['f1'] += f1_score(l_test.values, pred, average='macro')
            
            print('--------------------------------------------')
            for key, value in metrics.items():
                print(key)

                accuracy = value['accuracy'] / k
                time_elapsed_ms = value['time_elapsed_ms'] / k 
                no_of_features_selected = round(value['no_of_features_selected'] / k)
                precision = value['precision'] / k
                recall = value['recall'] / k
                f1 = value['f1'] / k
                
                print('accuracy', accuracy)
                print('no_of_features', value['no_of_features'])
                print('no_of_features_selected', no_of_features_selected)
                print('precision', precision)
                print('recall', recall)
                print('f1', f1)
                csv_measure.append({
                    'dataset_name': ds_name,
                    'selector_name': key,
                    'accuracy': round(accuracy, 3),
                    'time_elapsed_ms': time_elapsed_ms,
                    'no_of_features': value['no_of_features'],
                    'no_of_features_selected': no_of_features_selected,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })
                print('--------------------------------------------')

        print('############################################')

        output_csv_file_name = f'{clf_name}_selectors_measurements.csv'
        csv_df = pd.DataFrame(data=csv_measure)
        csv_df.to_csv(output_csv_file_name, index=False, sep=',', header=True,
                      columns=['dataset_name', 'selector_name', 'accuracy', 'time_elapsed_ms'], encoding='utf-8')


def selector_fit_and_transform(selector, f_train, f_test, l_train):
    selector.fit(f_train, l_train)
    sel_features_train = selector.transform(f_train)
    sel_features_test = selector.transform(f_test)
    return sel_features_train, sel_features_test


if __name__ == "__main__":
    run()
