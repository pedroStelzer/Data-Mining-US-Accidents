import os

try:
    from google.colab import drive
    IN_COLAB = True
except Exception:
    IN_COLAB = False

import mlflow
from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from sklearn_lvq import GLVQ
except ImportError:
    GLVQ = None

from src.train import run_baseline, run_random_search
from src.pipeline import get_pipeline
from src.data_ingestion import load_data


def main():
    if IN_COLAB:
        drive.mount('/content/drive')
        mlflow.set_tracking_uri('file:///content/drive/MyDrive/mlruns')
    else:
        local_mlflow_path = os.path.abspath(os.path.expanduser('~/mlruns'))
        os.makedirs(local_mlflow_path, exist_ok=True)
        mlflow.set_tracking_uri('file:///' + local_mlflow_path.replace(os.sep, '/'))
        print(f'Usando MLflow local em: {local_mlflow_path}')

    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # Ajustar as classes de y para serem zero-indexed
    y_train = y_train - 1
    y_val = y_val - 1
    y_test = y_test - 1

    feature_flags = {
        'duration': True,
        'wind': True,
        'weather': True,
        'geo': True,
        'drop_columns': True,
        'infrastructure': True,
    }

    base_pipe = get_pipeline(feature_flags=feature_flags)

    def create_named_pipe(model_obj):
        new_pipe = clone(base_pipe)
        new_pipe.set_params(model=model_obj)
        return new_pipe

    # baselines = {
    #     'Logistic_Regression': create_named_pipe(LogisticRegression(max_iter=1000)),
    #     'Decision_Tree': create_named_pipe(DecisionTreeClassifier(max_depth=5)),
    #     'KNN': create_named_pipe(KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
    #     'Naive_Bayes': create_named_pipe(GaussianNB()),
    #     'MLP': create_named_pipe(MLPClassifier(max_iter=300)),
    #     'Random_Forest': create_named_pipe(RandomForestClassifier(n_estimators=100, n_jobs=-1)),
    #     'AdaBoost': create_named_pipe(AdaBoostClassifier()),
    #     'Bagging': create_named_pipe(BaggingClassifier()),
    #     'SVM': create_named_pipe(SVC(kernel='linear', max_iter=1000, probability=True)),
    # }

    # if XGBClassifier is not None:
    #     baselines['XGBoost'] = create_named_pipe(XGBClassifier(n_jobs=-1))
    # else:
    #     print('XGBoost não encontrado; pulando XGBoost.')

    # if GLVQ is not None:
    #     baselines['LVQ_GLVQ'] = create_named_pipe(GLVQ(distance_type='squared-euclidean', random_state=42))
    # else:
    #     print('GLVQ não encontrado; pulando GLVQ.')

    estimators = [
        ('lr', LogisticRegression(max_iter=1000)),
        ('dt', DecisionTreeClassifier(max_depth=5)),
        ('rf', RandomForestClassifier(n_estimators=50))
    ]

    # baselines['Voting'] = create_named_pipe(VotingClassifier(estimators=estimators, voting='soft'))
    # baselines['Stacking'] = create_named_pipe(StackingClassifier(estimators=estimators, final_estimator=LogisticRegression()))

    rf_search_pipe = create_named_pipe(RandomForestClassifier())
    rf_param_dist = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
        'model__bootstrap': [True, False],
    }

    run_random_search(
        experiment_name='US_Accidents_RandomSearch_RandomForest',
        pipeline=rf_search_pipe,
        param_distributions=rf_param_dist,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        feature_flags=feature_flags,
        n_iter=12,
        cv=3,
        scoring='accuracy',
    )

    if XGBClassifier is not None:
        xgb_search_pipe = create_named_pipe(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1))
        xgb_param_dist = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [3, 6, 9],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__subsample': [0.6, 0.8, 1.0],
        }
        run_random_search(
            experiment_name='US_Accidents_RandomSearch_XGBoost',
            pipeline=xgb_search_pipe,
            param_distributions=xgb_param_dist,
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            feature_flags=feature_flags,
            n_iter=12,
            cv=3,
            scoring='accuracy',
        )

    # Additional RandomizedSearch runs for other models
    # Logistic Regression
    lr_pipe = create_named_pipe(LogisticRegression(max_iter=1000))
    lr_param_dist = {
        'model__C': [0.01, 0.1, 1.0, 10.0],
        'model__penalty': ['l2'],
        'model__solver': ['lbfgs']
    }
    run_random_search(
        experiment_name='US_Accidents_RandomSearch_LogisticRegression',
        pipeline=lr_pipe,
        param_distributions=lr_param_dist,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        feature_flags=feature_flags,
        n_iter=8,
        cv=3,
        scoring='accuracy',
    )

    # Decision Tree
    dt_pipe = create_named_pipe(DecisionTreeClassifier())
    dt_param_dist = {
        'model__max_depth': [None, 5, 10, 20],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }
    run_random_search(
        experiment_name='US_Accidents_RandomSearch_DecisionTree',
        pipeline=dt_pipe,
        param_distributions=dt_param_dist,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        feature_flags=feature_flags,
        n_iter=8,
        cv=3,
        scoring='accuracy',
    )

    # KNN
    knn_pipe = create_named_pipe(KNeighborsClassifier())
    knn_param_dist = {
        'model__n_neighbors': [3, 5, 7, 11],
        'model__weights': ['uniform', 'distance'],
        'model__p': [1, 2]
    }
    run_random_search(
        experiment_name='US_Accidents_RandomSearch_KNN',
        pipeline=knn_pipe,
        param_distributions=knn_param_dist,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        feature_flags=feature_flags,
        n_iter=8,
        cv=3,
        scoring='accuracy',
    )

    # Naive Bayes
    nb_pipe = create_named_pipe(GaussianNB())
    nb_param_dist = {
        'model__var_smoothing': [1e-9, 1e-8, 1e-7]
    }
    run_random_search(
        experiment_name='US_Accidents_RandomSearch_NaiveBayes',
        pipeline=nb_pipe,
        param_distributions=nb_param_dist,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        feature_flags=feature_flags,
        n_iter=6,
        cv=3,
        scoring='accuracy',
    )

    # MLP
    mlp_pipe = create_named_pipe(MLPClassifier(max_iter=300))
    mlp_param_dist = {
        'model__hidden_layer_sizes': [(50,), (100,), (50,50)],
        'model__alpha': [1e-4, 1e-3, 1e-2],
        'model__learning_rate_init': [1e-3, 1e-2]
    }
    run_random_search(
        experiment_name='US_Accidents_RandomSearch_MLP',
        pipeline=mlp_pipe,
        param_distributions=mlp_param_dist,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        feature_flags=feature_flags,
        n_iter=8,
        cv=3,
        scoring='accuracy',
    )

    # AdaBoost
    ada_pipe = create_named_pipe(AdaBoostClassifier())
    ada_param_dist = {
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.5, 1.0, 1.5]
    }
    run_random_search(
        experiment_name='US_Accidents_RandomSearch_AdaBoost',
        pipeline=ada_pipe,
        param_distributions=ada_param_dist,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        feature_flags=feature_flags,
        n_iter=8,
        cv=3,
        scoring='accuracy',
    )

    # Bagging
    bag_pipe = create_named_pipe(BaggingClassifier())
    bag_param_dist = {
        'model__n_estimators': [10, 50, 100],
        'model__max_samples': [0.5, 0.75, 1.0],
        'model__max_features': [0.5, 0.75, 1.0]
    }
    run_random_search(
        experiment_name='US_Accidents_RandomSearch_Bagging',
        pipeline=bag_pipe,
        param_distributions=bag_param_dist,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        feature_flags=feature_flags,
        n_iter=8,
        cv=3,
        scoring='accuracy',
    )

    # SVM
    svm_pipe = create_named_pipe(SVC(probability=True, max_iter=1000))
    svm_param_dist = {
        'model__C': [0.1, 1.0, 10.0],
        'model__kernel': ['linear', 'rbf'],
        'model__gamma': ['scale', 'auto']
    }
    run_random_search(
        experiment_name='US_Accidents_RandomSearch_SVM',
        pipeline=svm_pipe,
        param_distributions=svm_param_dist,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        feature_flags=feature_flags,
        n_iter=8,
        cv=3,
        scoring='accuracy',
    )

    # LVQ (if available) - skip if not installed
    if GLVQ is not None:
        lvq_pipe = create_named_pipe(GLVQ(distance_type='squared-euclidean', random_state=42))
        # GLVQ has limited tunable params in many implementations; perform a light search if possible
        try:
            lvq_param_dist = {
                'model__prototypes_per_class': [1, 2, 3]
            }
            run_random_search(
                experiment_name='US_Accidents_RandomSearch_LVQ',
                pipeline=lvq_pipe,
                param_distributions=lvq_param_dist,
                X_train=X_train,
                X_val=X_val,
                y_train=y_train,
                y_val=y_val,
                feature_flags=feature_flags,
                n_iter=6,
                cv=3,
                scoring='accuracy',
            )
        except Exception:
            print('Não foi possível rodar RandomizedSearch para LVQ (parâmetros desconhecidos). Pulando.')
    else:
        print('GLVQ não instalado; pulando RandomizedSearch para LVQ.')

    # Voting classifier search (tune voting type)
    voting_pipe = create_named_pipe(VotingClassifier(estimators=estimators))
    voting_param_dist = {
        'model__voting': ['soft', 'hard']
    }
    run_random_search(
        experiment_name='US_Accidents_RandomSearch_Voting',
        pipeline=voting_pipe,
        param_distributions=voting_param_dist,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        feature_flags=feature_flags,
        n_iter=4,
        cv=3,
        scoring='accuracy',
    )

    # Stacking classifier search (tune final_estimator C when logistic)
    stacking_pipe = create_named_pipe(StackingClassifier(estimators=estimators, final_estimator=LogisticRegression()))
    stacking_param_dist = {
        'model__final_estimator__C': [0.01, 0.1, 1.0, 10.0]
    }
    run_random_search(
        experiment_name='US_Accidents_RandomSearch_Stacking',
        pipeline=stacking_pipe,
        param_distributions=stacking_param_dist,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        feature_flags=feature_flags,
        n_iter=6,
        cv=3,
        scoring='accuracy',
    )

    # run_baseline(
    #     experiment_name='US_Accidents_Baseline_Feature_Engineering_V1',
    #     models_dict=baselines,
    #     X_train=X_train,
    #     X_val=X_val,
    #     y_train=y_train,
    #     y_val=y_val,
    #     feature_flags=feature_flags,
    # )


if __name__ == '__main__':
    main()

