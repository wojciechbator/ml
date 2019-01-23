import pandas as pd
import time

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, RFE, SelectPercentile, SelectKBest, \
    SelectFromModel, f_classif
from sklearn.linear_model import LassoCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.svm import SVC

# TODO: Add databases (total 30 needed)
# TODO: Calculate new metrics
# TODO: Cross Validation after everything else is set up
# TODO: Save metrics to csv instead of printing them


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
parkinsons_ds = get_data('parkinsons.data')
spambase_ds = get_data('spambase.data')
primary_tumor_ds =get_data('primary-tumor.data')
lymphography_ds = get_data('lymphography.data')
lung_cancer_ds = get_data('lung-cancer.data')
hayes_roth_ds = get_data('hayes-roth.data')
haberman_ds = get_data('haberman.data')
ionosphere_ds = get_data('ionosphere.data')
glass_ds = get_data('glass.data')
dermatology_ds = get_data('dermatology.data')
hepatitis_ds = get_data('hepatitis.data')
tic_tac_toe_ds = get_data('tic-tac-toe.data')
agaricus_lepiota_ds = get_data('agaricus-lepiota.data')
zoo_ds = get_data('zoo.data')
hungarian_heart_disease_ds = get_data('hungarian_heart_disease.data')
kr_vs_kp_ds = get_data('kr-vs-kp.data')
anneal_ds = get_data('anneal.data')
balance_scale_ds = get_data('balance-scale.data')
breast_cancer_ds = get_data('breast-cancer.data')
adult_ds = get_data('adult.data')
wine_ds = get_data('wine.data', False)
wine_quality_red_ds = get_data('winequality-red.csv', header=0, sep=';')
soybean_ds = get_data('soybean.data')
bank_ds = get_data('bank.csv', header=0, sep=';')
car_ds = get_data('car.data')
heart_disease_ds = get_data('processed.cleveland.data')
poker_hand_ds = get_data('poker-hand-testing.data')
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
vt_selector = VarianceThreshold()
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
    ('ionosphere', ionosphere_ds),
    ('tic-tac-toe', tic_tac_toe_ds),
    ('king rook vs king pawn', kr_vs_kp_ds),
    ('bank', bank_ds),
    ('dermatology', dermatology_ds),
    ('hayes roth', hayes_roth_ds),
    ('parkinsons', parkinsons_ds)
]
nb_clf = ('NaiveBayes', GaussianNB(), nb_datasets)

svm_datasets = [
    ('glass', glass_ds),
    ('iris', iris_ds),
    ('car', car_ds),
    ('hepatitis', hepatitis_ds),
    ('zoo', zoo_ds),
    ('primary tumor', primary_tumor_ds),
    ('agaricus lepiota', agaricus_lepiota_ds),
    ('heart disease', heart_disease_ds),
    ('abalone', abalone_ds),
    ('lung cancer', lung_cancer_ds)
]
svm_clf = ('SVM', SVC(kernel='linear', gamma='auto'), svm_datasets)

rf_datasets = [
    ('wine', wine_ds),
    ('adult', adult_ds),
    ('anneal', anneal_ds),
    ('soybean', soybean_ds),
    ('poker hand', poker_hand_ds),
    ('hungarian heart disease', hungarian_heart_disease_ds),
    ('breast cancer', breast_cancer_ds),
    ('haberman', haberman_ds),
    ('lymphography', lymphography_ds),
    ('spambase', spambase_ds)
]
rf_clf = ('RandomForest', RandomForestClassifier(n_estimators=10), rf_datasets)

classifiers = [
    nb_clf,
    svm_clf,
    rf_clf
]


# MAGIC

def run():
    for clf_name, clf, datasets in classifiers:
        csv_measure = []
        csv_df = pd.DataFrame(data=csv_measure)
        print('\n############################################')
        print(clf_name)
        for ds_name, ds in datasets:
            print('============================================')
            print(ds_name)
            f_train, f_test, l_train, l_test = train_test_split(ds[0], ds[1],
                                                                test_size=0.33)

            print('--------------------------------------------')
            for sel_name, sel in selectors:
                sel_start = time.time()
                sel_f_train, sel_f_test = selector_fit_and_transform(sel, f_train, f_test,
                                                                     l_train)
                clf.fit(sel_f_train, l_train)

                print('accuracy', clf.score(sel_f_test, l_test))
                sel_end = time.time()
                csv_measure.append(
                    {'dataset_name': ds_name, 'selector_name': sel_name, 'accuracy': round(clf.score(sel_f_test, l_test), 3), 'time_elapsed_ms': int(round((sel_end - sel_start)*1000))})
                print('--------------------------------------------')

        print('############################################')

        output_csv_file_name = f'{clf_name}_selectors_measurements.csv'
        csv_df = pd.DataFrame(data=csv_measure)
        csv_df.to_csv(output_csv_file_name, index=False,
                      sep=',', header=True, columns=['dataset_name', 'selector_name', 'accuracy', 'time_elapsed_ms'], encoding='utf-8')


def selector_fit_and_transform(selector, f_train, f_test, l_train):
    selector.fit(f_train, l_train)
    sel_features_train = selector.transform(f_train)
    sel_features_test = selector.transform(f_test)
    return sel_features_train, sel_features_test


if __name__ == "__main__":
    run()
