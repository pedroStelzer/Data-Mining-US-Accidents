import os
import itertools

try:
    from google.colab import drive
    IN_COLAB = True
except Exception:
    IN_COLAB = False

import mlflow
from sklearn.model_selection import StratifiedKFold
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

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

from src.train import run_baseline, run_random_search
from src.pipeline import get_pipeline
from src.data_ingestion import load_data

# ------------------------------------------------------- #
# --------------- INICIO DOS EXPERIMENTOS --------------- #
# ------------------------------------------------------- #

def run_all_experiments(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    feature_flags=None,
    resampling_method='none',
    use_pca=False,
    pca_n_components=0.95,
    experiment_group='hptuning'
):
    # 1. Define o nome do experimento baseado na FASE/GRUPO atual
    phase_experiment_name = f"US_Accidents_Phase_{experiment_group}"

    if feature_flags is None:
        feature_flags = {
            'duration': True,
            'wind': True,
            'weather': True,
            'geo': True,
            'drop_columns': True,
            'infrastructure': True,
        }

    def create_named_pipe(model_obj):
        new_pipe = clone(get_pipeline(
            feature_flags=feature_flags,
            model=model_obj,
            use_pca=use_pca,
            pca_n_components=pca_n_components,
            balancer_method=resampling_method,
        ))
        return new_pipe
    
    estimators = [
        ('lr', LogisticRegression(max_iter=1000)),
        ('dt', DecisionTreeClassifier(max_depth=5)),
        ('rf', RandomForestClassifier(n_estimators=50))
    ]

    # Criamos um gerador de splits fixo com 5 Folds (Adequado para Wilcoxon)
    cv_estatistico = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # =========================================================
    # LVQ (Com Patch de Compatibilidade Ativo)
    # =========================================================
    
    # --- PATCH DE COMPATIBILIDADE PARA O GLVQ ---
    try:
        from sklvq import GLVQ
        import sklvq.models._base 
    except ImportError:
        GLVQ = None

    if GLVQ is not None:
        import sklearn.base
        import sklearn.utils
        from sklearn.utils import check_X_y

        # 1. Patch para o momento do Treino (.fit)
        def _compat_validate_data(self, X, y=None, ensure_2d=True, dtype="numeric", accept_sparse=False, **kwargs):
            kwargs.pop('force_all_finite', None)
            kwargs.pop('allow_nd', None)
            kwargs.pop('cast_to_ndarray', None)
            if y is not None:
                X_validated, y_validated = check_X_y(X, y, accept_sparse=accept_sparse, dtype=dtype, ensure_2d=ensure_2d, **kwargs)
                self.n_features_in_ = X_validated.shape[1]
                return X_validated, y_validated
            return X

        if not hasattr(sklearn.base.BaseEstimator, "_validate_data"):
            sklearn.base.BaseEstimator._validate_data = _compat_validate_data

        # 2. Patch para o momento da Predição (.predict)
        if not hasattr(sklearn.utils, '_original_check_array'):
            sklearn.utils._original_check_array = sklearn.utils.check_array

        def _compat_check_array(array, *args, **kwargs):
            if 'force_all_finite' in kwargs:
                # Traduz o parâmetro antigo para o que o Scikit-Learn atual espera
                kwargs['ensure_all_finite'] = kwargs.pop('force_all_finite')
            return sklearn.utils._original_check_array(array, *args, **kwargs)

        # Substitui na origem global
        sklearn.utils.check_array = _compat_check_array
        sklearn.utils.validation.check_array = _compat_check_array
        
        sklvq.models._base.check_array = _compat_check_array
    # ---------------------------------------------------------
    baselines = {}

    try:
        from sklvq import GLVQ
        baselines['LVQ_GLVQ'] = create_named_pipe(GLVQ(distance_type='squared-euclidean', random_state=42))
    except ImportError:
            print("  ⚠ GLVQ não encontrado, pulando do baseline.")
    
    run_baseline(
        experiment_name=phase_experiment_name,
        models_dict=baselines,
        X_train=X_train,
        X_test=X_val,
        y_train=y_train,
        y_test=y_val,
        feature_flags=feature_flags,
        resampling_method=resampling_method,
        experiment_group=experiment_group
    )

    
    # =========================================================================
    #  PATCH DE COMPATIBILIDADE PARA O GLVQ (Scikit-Learn 1.6+)
    # =========================================================================
    try:
        from sklvq import GLVQ
        import sklvq.models._base  # Arquivo interno do sklvq onde o erro acontece
    except ImportError:
        GLVQ = None

    if GLVQ is not None:
        import sklearn.base
        import sklearn.utils
        import sklearn.utils.validation
        import inspect
        from sklearn.utils import check_X_y, check_array

        # 1. Proteção: Salva as funções de validação originais do Scikit-Learn
        if not hasattr(sklearn.utils, '_original_check_array'):
            sklearn.utils._original_check_array = sklearn.utils.check_array
        if not hasattr(sklearn.utils, '_original_check_X_y'):
            sklearn.utils._original_check_X_y = sklearn.utils.check_X_y

        # 2. Varredura dinâmica de parâmetros aceitos pela sua versão do Sklearn
        _sig_check = inspect.signature(sklearn.utils._original_check_array)
        _has_ensure_all_finite = 'ensure_all_finite' in _sig_check.parameters
        _has_force_all_finite = 'force_all_finite' in _sig_check.parameters

        def _clean_and_translate_kwargs(kwargs):
            """Traduz force_all_finite e remove parâmetros que quebram o Sklearn novo."""
            if 'force_all_finite' in kwargs:
                val = kwargs.pop('force_all_finite')
                if _has_ensure_all_finite:
                    kwargs['ensure_all_finite'] = val
                elif _has_force_all_finite:
                    kwargs['force_all_finite'] = val
            
            # Remove parâmetros antigos injetados pelo sklvq que o Sklearn atual rejeita
            kwargs.pop('allow_nd', None)
            kwargs.pop('cast_to_ndarray', None)
            return kwargs

        # 3. Criação dos wrappers de validação compatíveis
        def _compat_check_array(array, *args, **kwargs):
            kwargs = _clean_and_translate_kwargs(kwargs)
            return sklearn.utils._original_check_array(array, *args, **kwargs)

        def _compat_check_X_y(X, y, *args, **kwargs):
            kwargs = _clean_and_translate_kwargs(kwargs)
            return sklearn.utils._original_check_X_y(X, y, *args, **kwargs)

        # Substituição global para o sklvq interceptar os dados limpos
        sklearn.utils.check_array = _compat_check_array
        sklearn.utils.validation.check_array = _compat_check_array
        sklvq.models._base.check_array = _compat_check_array

        sklearn.utils.check_X_y = _compat_check_X_y
        sklearn.utils.validation.check_X_y = _compat_check_X_y
        sklvq.models._base.check_X_y = _compat_check_X_y

        # 4. Reconstrução cirúrgica do método '_validate_data' que sumiu
        _SENTINEL = "no_validation"

        def _compat_validate_data(self, X=_SENTINEL, y=_SENTINEL, reset=True, validate_separately=False, **kwargs):
            # Evita o erro de ambiguidade do NumPy checando strings com isinstance
            X_is_sentinel = isinstance(X, str) and X == _SENTINEL
            y_is_sentinel = (y is None) or (isinstance(y, str) and y == _SENTINEL)
            
            # Momento do .fit() -> Valida X e y juntos
            if not X_is_sentinel and not y_is_sentinel:
                X_validated, y_validated = _compat_check_X_y(X, y, **kwargs)
                if reset:
                    self.n_features_in_ = X_validated.shape[1]
                return X_validated, y_validated
            
            # Momento do .predict() -> Valida apenas o X de entrada
            if not X_is_sentinel:
                X_validated = _compat_check_array(X, **kwargs)
                if reset and hasattr(X_validated, "shape") and len(X_validated.shape) > 1:
                    self.n_features_in_ = X_validated.shape[1]
                return X_validated
            
            return X

        # Garante o patch na classe base geral do Sklearn
        if not hasattr(sklearn.base.BaseEstimator, "_validate_data"):
            sklearn.base.BaseEstimator._validate_data = _compat_validate_data
        
        # Alvo Principal: Força a injeção diretamente na classe GLVQ e variantes do sklvq
        GLVQ._validate_data = _compat_validate_data
        
        if hasattr(sklvq, 'GMLVQ'):
            sklvq.GMLVQ._validate_data = _compat_validate_data
        if hasattr(sklvq, 'LGMLVQ'):
            sklvq.LGMLVQ._validate_data = _compat_validate_data

    # =========================================================================

        lvq_pipe = create_named_pipe(
            GLVQ(
                distance_type='squared-euclidean',
                random_state=42
            )
        )

        try:
            lvq_param_dist = {
                'model__prototype_n_per_class': [1, 2, 3]
            }

            run_random_search(
                experiment_name=phase_experiment_name,
                model_name='LVQ',
                pipeline=lvq_pipe,
                param_distributions=lvq_param_dist,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                feature_flags=feature_flags,
                resampling_method=resampling_method,
                use_pca=use_pca,
                pca_n_components=pca_n_components,
                experiment_group=experiment_group,
                n_iter=3,
                cv=cv_estatistico,
                scoring='f1_macro',
            )

        except Exception as e:
            print(
                f'Não foi possível rodar RandomizedSearch '
                f'para LVQ. Erro interno: {e}. Pulando.'
            )
    else:
        print('GLVQ não instalado; pulando RandomizedSearch para LVQ.')

    # =========================================================
    # RANDOM FOREST
    # =========================================================
    rf_search_pipe = create_named_pipe(RandomForestClassifier())

    rf_param_dist = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
        'model__bootstrap': [True, False],
    }

    run_random_search(
        experiment_name=phase_experiment_name,
        model_name='RandomForest',
        pipeline=rf_search_pipe,
        param_distributions=rf_param_dist,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_flags=feature_flags,
        resampling_method=resampling_method,
        use_pca=use_pca,
        pca_n_components=pca_n_components,
        experiment_group=experiment_group,
        n_iter=15,
        cv=cv_estatistico,
        scoring='f1_macro',
    )


    # =========================================================
    # XGBOOST
    # =========================================================
    if XGBClassifier is not None:
        xgb_search_pipe = create_named_pipe(
            XGBClassifier(
                use_label_encoder=False,
                eval_metric='mlogloss',
                n_jobs=1
            )
        )

        xgb_param_dist = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [3, 6, 9],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__subsample': [0.6, 0.8, 1.0],
        }

        run_random_search(
            experiment_name=phase_experiment_name,
            model_name='XGBoost',
            pipeline=xgb_search_pipe,
            param_distributions=xgb_param_dist,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_flags=feature_flags,
            resampling_method=resampling_method,
            use_pca=use_pca,
            pca_n_components=pca_n_components,
            experiment_group=experiment_group,
            n_iter=12,
            cv=cv_estatistico,
            scoring='f1_macro',
        )

    # =========================================================
    # LOGISTIC REGRESSION
    # =========================================================
    lr_search_pipe = create_named_pipe(
        LogisticRegression(max_iter=1000)
    )

    lr_param_dist = {
        'model__C': [0.01, 0.1, 1.0, 10.0],
        'model__penalty': ['l2'],
        'model__solver': ['lbfgs']
    }

    run_random_search(
        experiment_name=phase_experiment_name,
        model_name='LogisticRegression',
        pipeline=lr_search_pipe,
        param_distributions=lr_param_dist,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_flags=feature_flags,
        resampling_method=resampling_method,
        use_pca=use_pca,
        pca_n_components=pca_n_components,
        experiment_group=experiment_group,
        n_iter=4,
        cv=cv_estatistico,
        scoring='f1_macro',
    )

    # =========================================================
    # DECISION TREE
    # =========================================================
    dt_pipe = create_named_pipe(
        DecisionTreeClassifier()
    )

    dt_param_dist = {
        'model__max_depth': [None, 5, 10, 20],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }

    run_random_search(
        experiment_name=phase_experiment_name,
        model_name='DecisionTree',
        pipeline=dt_pipe,
        param_distributions=dt_param_dist,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_flags=feature_flags,
        resampling_method=resampling_method,
        use_pca=use_pca,
        pca_n_components=pca_n_components,
        experiment_group=experiment_group,
        n_iter=8,
        cv=cv_estatistico,
        scoring='f1_macro',
    )

    # =========================================================
    # KNN
    # =========================================================
    knn_pipe = create_named_pipe(
        KNeighborsClassifier()
    )

    knn_param_dist = {
        'model__n_neighbors': [3, 5, 7, 11],
        'model__weights': ['uniform', 'distance'],
        'model__p': [1, 2]
    }

    run_random_search(
        experiment_name=phase_experiment_name,
        model_name='KNN',
        pipeline=knn_pipe,
        param_distributions=knn_param_dist,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_flags=feature_flags,
        resampling_method=resampling_method,
        use_pca=use_pca,
        pca_n_components=pca_n_components,
        experiment_group=experiment_group,
        n_iter=2,
        cv=cv_estatistico,
        scoring='f1_macro',
    )


    # =========================================================
    # NAIVE BAYES
    # =========================================================
    nb_pipe = create_named_pipe(
        GaussianNB()
    )

    nb_param_dist = {
        'model__var_smoothing': [1e-9, 1e-8, 1e-7]
    }

    run_random_search(
        experiment_name=phase_experiment_name,
        model_name='NaiveBayes',
        pipeline=nb_pipe,
        param_distributions=nb_param_dist,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_flags=feature_flags,
        resampling_method=resampling_method,
        use_pca=use_pca,
        pca_n_components=pca_n_components,
        experiment_group=experiment_group,
        n_iter=3,
        cv=cv_estatistico,
        scoring='f1_macro',
    )

    # =========================================================
    # MLP
    # =========================================================
    mlp_pipe = create_named_pipe(
        MLPClassifier(max_iter=300)
    )

    mlp_param_dist = {
        'model__hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'model__alpha': [1e-4, 1e-3, 1e-2],
        'model__learning_rate_init': [1e-3, 1e-2]
    }

    run_random_search(
        experiment_name=phase_experiment_name,
        model_name='MLP',
        pipeline=mlp_pipe,
        param_distributions=mlp_param_dist,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_flags=feature_flags,
        resampling_method=resampling_method,
        use_pca=use_pca,
        pca_n_components=pca_n_components,
        experiment_group=experiment_group,
        n_iter=5,
        cv=cv_estatistico,
        scoring='f1_macro',
    )

    # =========================================================
    # ADABOOST
    # =========================================================
    ada_pipe = create_named_pipe(
        AdaBoostClassifier()
    )

    ada_param_dist = {
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.5, 1.0, 1.5]
    }

    run_random_search(
        experiment_name=phase_experiment_name,
        model_name='AdaBoost',
        pipeline=ada_pipe,
        param_distributions=ada_param_dist,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_flags=feature_flags,
        resampling_method=resampling_method,
        use_pca=use_pca,
        pca_n_components=pca_n_components,
        experiment_group=experiment_group,
        n_iter=8,
        cv=cv_estatistico,
        scoring='f1_macro',
    )

    # =========================================================
    # BAGGING
    # =========================================================
    bag_pipe = create_named_pipe(
        BaggingClassifier()
    )

    bag_param_dist = {
        'model__n_estimators': [10, 50, 100],
        'model__max_samples': [0.5, 0.75, 1.0],
        'model__max_features': [0.5, 0.75, 1.0]
    }

    run_random_search(
        experiment_name=phase_experiment_name,
        model_name='Bagging',
        pipeline=bag_pipe,
        param_distributions=bag_param_dist,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_flags=feature_flags,
        resampling_method=resampling_method,
        use_pca=use_pca,
        pca_n_components=pca_n_components,
        experiment_group=experiment_group,
        n_iter=8,
        cv=cv_estatistico,
        scoring='f1_macro',
    )

    # =========================================================
    # SVM
    # =========================================================
    svm_pipe = create_named_pipe(
        SVC(probability=True, max_iter=1000)
    )

    svm_param_dist = {
        'model__C': [0.1, 1.0, 10.0],
        'model__kernel': ['linear', 'rbf'],
        'model__gamma': ['scale', 'auto']
    }

    run_random_search(
        experiment_name=phase_experiment_name,
        model_name='SVM',
        pipeline=svm_pipe,
        param_distributions=svm_param_dist,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_flags=feature_flags,
        resampling_method=resampling_method,
        use_pca=use_pca,
        pca_n_components=pca_n_components,
        experiment_group=experiment_group,
        n_iter=3,
        cv=cv_estatistico,
        scoring='f1_macro',
    )

    
    # =========================================================
    # VOTING
    # =========================================================
    voting_pipe = create_named_pipe(
        VotingClassifier(estimators=estimators)
    )

    voting_param_dist = {
        'model__voting': ['soft', 'hard']
    }

    run_random_search(
        experiment_name=phase_experiment_name,
        model_name='Voting',
        pipeline=voting_pipe,
        param_distributions=voting_param_dist,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_flags=feature_flags,
        resampling_method=resampling_method,
        use_pca=use_pca,
        pca_n_components=pca_n_components,
        experiment_group=experiment_group,
        n_iter=2,
        cv=cv_estatistico,
        scoring='f1_macro',
    )

    # =========================================================
    # STACKING
    # =========================================================
    stacking_pipe = create_named_pipe(
        StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression()
        )
    )

    stacking_param_dist = {
        'model__final_estimator__C': [0.01, 0.1, 1.0, 10.0]
    }

    run_random_search(
        experiment_name=phase_experiment_name,
        model_name='Stacking',
        pipeline=stacking_pipe,
        param_distributions=stacking_param_dist,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_flags=feature_flags,
        resampling_method=resampling_method,
        use_pca=use_pca,
        pca_n_components=pca_n_components,
        experiment_group=experiment_group,
        n_iter=4,
        cv=cv_estatistico,
        scoring='f1_macro',
    )

    # =========================================================
    # LIGHTGBM
    # =========================================================
    if LGBMClassifier is not None:

        lgb_search_pipe = create_named_pipe(
            LGBMClassifier(
                verbose=-1,
                force_col_wise=True
            )
        )

        lgb_param_dist = {
            'model__num_leaves': [15, 31, 63, 127, 255],
            'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'model__feature_fraction': [0.6, 0.8, 1.0],
            'model__bagging_fraction': [0.6, 0.8, 1.0],
            'model__min_data_in_leaf': [5, 10, 20, 50],
            'model__lambda_l1': [0, 0.1, 1.0],
            'model__lambda_l2': [0, 0.1, 1.0]
        }

        run_random_search(
            experiment_name=phase_experiment_name,
            model_name='LightGBM',
            pipeline=lgb_search_pipe,
            param_distributions=lgb_param_dist,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_flags=feature_flags,
            resampling_method=resampling_method,
            use_pca=use_pca,
            pca_n_components=pca_n_components,
            experiment_group=experiment_group,
            n_iter=12,
            cv=3,
            scoring='f1_macro',
        )

    else:
        print(
            'LightGBM não encontrado; '
            'pulando RandomizedSearch para LightGBM.'
        )

# -------------------------------------------------------- #
# ----------------- FIM DOS EXPERIMENTOS ----------------- #
# -------------------------------------------------------- #


def generate_feature_engineering_ablation_configs(
    feature_flags=None,
    include_baseline=True,
    include_all_disabled=False,
):
    """Gera configurações de ablação de feature engineering."""
    if feature_flags is None:
        feature_flags = {
            'duration': True,
            'wind': True,
            'weather': True,
            'geo': True,
            'drop_columns': True,
            'infrastructure': True,
        }

    configs = []
    if include_baseline:
        configs.append((feature_flags.copy(), 'all_enabled'))

    for key in sorted(feature_flags.keys()):
        config = feature_flags.copy()
        config[key] = False
        configs.append((config, f'no_{key}'))

    if include_all_disabled:
        configs.append(({k: False for k in feature_flags}, 'all_disabled'))

    return configs


def run_feature_engineering_ablation(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    feature_flags=None,
    resampling_method='none',
    use_pca=False,
    pca_n_components=0.95,
    experiment_group='feature_engineering_ablation',
    include_all_disabled=False,
):
    cv_estatistico = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    estimators = [
        ('lr', LogisticRegression(max_iter=1000)),
        ('dt', DecisionTreeClassifier(max_depth=5)),
        ('rf', RandomForestClassifier(n_estimators=50))
    ]

    configs = generate_feature_engineering_ablation_configs(
        feature_flags=feature_flags,
        include_baseline=True,
        include_all_disabled=include_all_disabled,
    )

    for config, suffix in configs:
        group_name = f"{experiment_group}_{suffix}"
        phase_experiment_name = f"US_Accidents_Ablation_{group_name}"
        
        print(f"\n Executando experimento de FE: {suffix} (group={group_name})")
        
        # Helper local para criar os pipelines respeitando o 'config' (flag de features) desta iteração
        def create_ablation_pipe(model_obj):
            return clone(get_pipeline(
                feature_flags=config,  # <-- Crucial: passa o config atual do loop!
                model=model_obj,
                use_pca=use_pca,
                pca_n_components=pca_n_components,
                balancer_method=resampling_method,
            ))
        
        rf_search_pipe = create_ablation_pipe(RandomForestClassifier())

        rf_param_dist = {}

        run_random_search(
            experiment_name=phase_experiment_name,
            model_name='RandomForest',
            pipeline=rf_search_pipe,
            param_distributions=rf_param_dist,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_flags=feature_flags,
            resampling_method=resampling_method,
            use_pca=use_pca,
            pca_n_components=pca_n_components,
            experiment_group=experiment_group,
            n_iter=15,
            cv=cv_estatistico,
            scoring='f1_macro',
        )


        # =========================================================
        # XGBOOST
        # =========================================================
        if XGBClassifier is not None:
            xgb_search_pipe = create_ablation_pipe(
                XGBClassifier(
                    use_label_encoder=False,
                    eval_metric='mlogloss',
                    n_jobs=1
                )
            )

            xgb_param_dist = {}

            run_random_search(
                experiment_name=phase_experiment_name,
                model_name='XGBoost',
                pipeline=xgb_search_pipe,
                param_distributions=xgb_param_dist,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                feature_flags=feature_flags,
                resampling_method=resampling_method,
                use_pca=use_pca,
                pca_n_components=pca_n_components,
                experiment_group=experiment_group,
                n_iter=12,
                cv=cv_estatistico,
                scoring='f1_macro',
            )

        # =========================================================
        # LOGISTIC REGRESSION
        # =========================================================
        lr_search_pipe = create_ablation_pipe(
            LogisticRegression(max_iter=1000)
        )

        lr_param_dist = {}

        run_random_search(
            experiment_name=phase_experiment_name,
            model_name='LogisticRegression',
            pipeline=lr_search_pipe,
            param_distributions=lr_param_dist,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_flags=feature_flags,
            resampling_method=resampling_method,
            use_pca=use_pca,
            pca_n_components=pca_n_components,
            experiment_group=experiment_group,
            n_iter=4,
            cv=cv_estatistico,
            scoring='f1_macro',
        )

        # =========================================================
        # DECISION TREE
        # =========================================================
        dt_pipe = create_ablation_pipe(
            DecisionTreeClassifier()
        )

        dt_param_dist = {}

        run_random_search(
            experiment_name=phase_experiment_name,
            model_name='DecisionTree',
            pipeline=dt_pipe,
            param_distributions=dt_param_dist,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_flags=feature_flags,
            resampling_method=resampling_method,
            use_pca=use_pca,
            pca_n_components=pca_n_components,
            experiment_group=experiment_group,
            n_iter=8,
            cv=cv_estatistico,
            scoring='f1_macro',
        )

        # =========================================================
        # KNN
        # =========================================================
        knn_pipe = create_ablation_pipe(
            KNeighborsClassifier()
        )

        knn_param_dist = {}

        run_random_search(
            experiment_name=phase_experiment_name,
            model_name='KNN',
            pipeline=knn_pipe,
            param_distributions=knn_param_dist,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_flags=feature_flags,
            resampling_method=resampling_method,
            use_pca=use_pca,
            pca_n_components=pca_n_components,
            experiment_group=experiment_group,
            n_iter=2,
            cv=cv_estatistico,
            scoring='f1_macro',
        )


        # =========================================================
        # NAIVE BAYES
        # =========================================================
        nb_pipe = create_ablation_pipe(
            GaussianNB()
        )

        nb_param_dist = {}

        run_random_search(
            experiment_name=phase_experiment_name,
            model_name='NaiveBayes',
            pipeline=nb_pipe,
            param_distributions=nb_param_dist,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_flags=feature_flags,
            resampling_method=resampling_method,
            use_pca=use_pca,
            pca_n_components=pca_n_components,
            experiment_group=experiment_group,
            n_iter=3,
            cv=cv_estatistico,
            scoring='f1_macro',
        )

        # =========================================================
        # MLP
        # =========================================================
        mlp_pipe = create_ablation_pipe(
            MLPClassifier(max_iter=300)
        )

        mlp_param_dist = {}

        run_random_search(
            experiment_name=phase_experiment_name,
            model_name='MLP',
            pipeline=mlp_pipe,
            param_distributions=mlp_param_dist,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_flags=feature_flags,
            resampling_method=resampling_method,
            use_pca=use_pca,
            pca_n_components=pca_n_components,
            experiment_group=experiment_group,
            n_iter=5,
            cv=cv_estatistico,
            scoring='f1_macro',
        )

        # =========================================================
        # ADABOOST
        # =========================================================
        ada_pipe = create_ablation_pipe(
            AdaBoostClassifier()
        )

        ada_param_dist = {}

        run_random_search(
            experiment_name=phase_experiment_name,
            model_name='AdaBoost',
            pipeline=ada_pipe,
            param_distributions=ada_param_dist,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_flags=feature_flags,
            resampling_method=resampling_method,
            use_pca=use_pca,
            pca_n_components=pca_n_components,
            experiment_group=experiment_group,
            n_iter=8,
            cv=cv_estatistico,
            scoring='f1_macro',
        )

        # =========================================================
        # BAGGING
        # =========================================================
        bag_pipe = create_ablation_pipe(
            BaggingClassifier()
        )

        bag_param_dist = {}

        run_random_search(
            experiment_name=phase_experiment_name,
            model_name='Bagging',
            pipeline=bag_pipe,
            param_distributions=bag_param_dist,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_flags=feature_flags,
            resampling_method=resampling_method,
            use_pca=use_pca,
            pca_n_components=pca_n_components,
            experiment_group=experiment_group,
            n_iter=8,
            cv=cv_estatistico,
            scoring='f1_macro',
        )

        # =========================================================
        # SVM
        # =========================================================
        svm_pipe = create_ablation_pipe(
            SVC(probability=True, max_iter=1000)
        )

        svm_param_dist = {}

        run_random_search(
            experiment_name=phase_experiment_name,
            model_name='SVM',
            pipeline=svm_pipe,
            param_distributions=svm_param_dist,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_flags=feature_flags,
            resampling_method=resampling_method,
            use_pca=use_pca,
            pca_n_components=pca_n_components,
            experiment_group=experiment_group,
            n_iter=3,
            cv=cv_estatistico,
            scoring='f1_macro',
        )

        
        # =========================================================
        # VOTING
        # =========================================================
        voting_pipe = create_ablation_pipe(
            VotingClassifier(estimators=estimators)
        )

        voting_param_dist = {}

        run_random_search(
            experiment_name=phase_experiment_name,
            model_name='Voting',
            pipeline=voting_pipe,
            param_distributions=voting_param_dist,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_flags=feature_flags,
            resampling_method=resampling_method,
            use_pca=use_pca,
            pca_n_components=pca_n_components,
            experiment_group=experiment_group,
            n_iter=2,
            cv=cv_estatistico,
            scoring='f1_macro',
        )

        # =========================================================
        # STACKING
        # =========================================================
        stacking_pipe = create_ablation_pipe(
            StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression()
            )
        )

        stacking_param_dist = {}

        run_random_search(
            experiment_name=phase_experiment_name,
            model_name='Stacking',
            pipeline=stacking_pipe,
            param_distributions=stacking_param_dist,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_flags=feature_flags,
            resampling_method=resampling_method,
            use_pca=use_pca,
            pca_n_components=pca_n_components,
            experiment_group=experiment_group,
            n_iter=4,
            cv=cv_estatistico,
            scoring='f1_macro',
        )

        # =========================================================
        # LIGHTGBM
        # =========================================================
        if LGBMClassifier is not None:

            lgb_search_pipe = create_ablation_pipe(
                LGBMClassifier(
                    verbose=-1,
                    force_col_wise=True
                )
            )

            lgb_param_dist = {}

            run_random_search(
                experiment_name=phase_experiment_name,
                model_name='LightGBM',
                pipeline=lgb_search_pipe,
                param_distributions=lgb_param_dist,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                feature_flags=feature_flags,
                resampling_method=resampling_method,
                use_pca=use_pca,
                pca_n_components=pca_n_components,
                experiment_group=experiment_group,
                n_iter=12,
                cv=cv_estatistico,
                scoring='f1_macro',
            )

        else:
            print(
                'LightGBM não encontrado; '
                'pulando RandomizedSearch para LightGBM.'
            )

def run_all_experiments_variations(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    feature_flags=None,
    resampling_method='none',
    use_pca=False,
    pca_n_components=0.95,
    experiment_group='hptuning'
):
    # Define o nome do experimento baseado na FASE/GRUPO atual
    phase_experiment_name = f"US_Accidents_Phase_{experiment_group}"

    if feature_flags is None:
        feature_flags = {
            'duration': True,
            'wind': True,
            'weather': True,
            'geo': True,
            'drop_columns': True,
            'infrastructure': True,
        }

    def create_named_pipe(model_obj):
        new_pipe = clone(get_pipeline(
            feature_flags=feature_flags,
            model=model_obj,
            use_pca=use_pca,
            pca_n_components=pca_n_components,
            balancer_method=resampling_method,
        ))
        return new_pipe
    
    estimators = [
        ('lr', LogisticRegression(max_iter=1000)),
        ('dt', DecisionTreeClassifier(max_depth=5)),
        ('rf', RandomForestClassifier(n_estimators=50))
    ]

    # Criamos um gerador de splits fixo com 5 Folds
    cv_estatistico = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # =========================================================
    # RANDOM FOREST
    # =========================================================
    rf_search_pipe = create_named_pipe(RandomForestClassifier())

    rf_param_dist = {}

    run_random_search(
        experiment_name=phase_experiment_name,
        model_name='RandomForest',
        pipeline=rf_search_pipe,
        param_distributions=rf_param_dist,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_flags=feature_flags,
        resampling_method=resampling_method,
        use_pca=use_pca,
        pca_n_components=pca_n_components,
        experiment_group=experiment_group,
        n_iter=15,
        cv=cv_estatistico,
        scoring='f1_macro',
    )


    # =========================================================
    # XGBOOST
    # =========================================================
    if XGBClassifier is not None:
        xgb_search_pipe = create_named_pipe(
            XGBClassifier(
                use_label_encoder=False,
                eval_metric='mlogloss',
                n_jobs=1
            )
        )

        xgb_param_dist = {}

        run_random_search(
            experiment_name=phase_experiment_name,
            model_name='XGBoost',
            pipeline=xgb_search_pipe,
            param_distributions=xgb_param_dist,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_flags=feature_flags,
            resampling_method=resampling_method,
            use_pca=use_pca,
            pca_n_components=pca_n_components,
            experiment_group=experiment_group,
            n_iter=12,
            cv=cv_estatistico,
            scoring='f1_macro',
        )

    # =========================================================
    # LOGISTIC REGRESSION
    # =========================================================
    lr_search_pipe = create_named_pipe(
        LogisticRegression(max_iter=1000)
    )

    lr_param_dist = {}

    run_random_search(
        experiment_name=phase_experiment_name,
        model_name='LogisticRegression',
        pipeline=lr_search_pipe,
        param_distributions=lr_param_dist,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_flags=feature_flags,
        resampling_method=resampling_method,
        use_pca=use_pca,
        pca_n_components=pca_n_components,
        experiment_group=experiment_group,
        n_iter=4,
        cv=cv_estatistico,
        scoring='f1_macro',
    )

    # =========================================================
    # DECISION TREE
    # =========================================================
    dt_pipe = create_named_pipe(
        DecisionTreeClassifier()
    )

    dt_param_dist = {}

    run_random_search(
        experiment_name=phase_experiment_name,
        model_name='DecisionTree',
        pipeline=dt_pipe,
        param_distributions=dt_param_dist,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_flags=feature_flags,
        resampling_method=resampling_method,
        use_pca=use_pca,
        pca_n_components=pca_n_components,
        experiment_group=experiment_group,
        n_iter=8,
        cv=cv_estatistico,
        scoring='f1_macro',
    )

    # =========================================================
    # KNN
    # =========================================================
    knn_pipe = create_named_pipe(
        KNeighborsClassifier()
    )

    knn_param_dist = {}

    run_random_search(
        experiment_name=phase_experiment_name,
        model_name='KNN',
        pipeline=knn_pipe,
        param_distributions=knn_param_dist,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_flags=feature_flags,
        resampling_method=resampling_method,
        use_pca=use_pca,
        pca_n_components=pca_n_components,
        experiment_group=experiment_group,
        n_iter=2,
        cv=cv_estatistico,
        scoring='f1_macro',
    )


    # =========================================================
    # NAIVE BAYES
    # =========================================================
    nb_pipe = create_named_pipe(
        GaussianNB()
    )

    nb_param_dist = {}

    run_random_search(
        experiment_name=phase_experiment_name,
        model_name='NaiveBayes',
        pipeline=nb_pipe,
        param_distributions=nb_param_dist,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_flags=feature_flags,
        resampling_method=resampling_method,
        use_pca=use_pca,
        pca_n_components=pca_n_components,
        experiment_group=experiment_group,
        n_iter=3,
        cv=cv_estatistico,
        scoring='f1_macro',
    )

    # =========================================================
    # MLP
    # =========================================================
    mlp_pipe = create_named_pipe(
        MLPClassifier(max_iter=300)
    )

    mlp_param_dist = {}

    run_random_search(
        experiment_name=phase_experiment_name,
        model_name='MLP',
        pipeline=mlp_pipe,
        param_distributions=mlp_param_dist,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_flags=feature_flags,
        resampling_method=resampling_method,
        use_pca=use_pca,
        pca_n_components=pca_n_components,
        experiment_group=experiment_group,
        n_iter=5,
        cv=cv_estatistico,
        scoring='f1_macro',
    )

    # =========================================================
    # ADABOOST
    # =========================================================
    ada_pipe = create_named_pipe(
        AdaBoostClassifier()
    )

    ada_param_dist = {}

    run_random_search(
        experiment_name=phase_experiment_name,
        model_name='AdaBoost',
        pipeline=ada_pipe,
        param_distributions=ada_param_dist,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_flags=feature_flags,
        resampling_method=resampling_method,
        use_pca=use_pca,
        pca_n_components=pca_n_components,
        experiment_group=experiment_group,
        n_iter=8,
        cv=cv_estatistico,
        scoring='f1_macro',
    )

    # =========================================================
    # BAGGING
    # =========================================================
    bag_pipe = create_named_pipe(
        BaggingClassifier()
    )

    bag_param_dist = {}

    run_random_search(
        experiment_name=phase_experiment_name,
        model_name='Bagging',
        pipeline=bag_pipe,
        param_distributions=bag_param_dist,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_flags=feature_flags,
        resampling_method=resampling_method,
        use_pca=use_pca,
        pca_n_components=pca_n_components,
        experiment_group=experiment_group,
        n_iter=8,
        cv=cv_estatistico,
        scoring='f1_macro',
    )

    # =========================================================
    # SVM
    # =========================================================
    svm_pipe = create_named_pipe(
        SVC(probability=True, max_iter=1000)
    )

    svm_param_dist = {}

    run_random_search(
        experiment_name=phase_experiment_name,
        model_name='SVM',
        pipeline=svm_pipe,
        param_distributions=svm_param_dist,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_flags=feature_flags,
        resampling_method=resampling_method,
        use_pca=use_pca,
        pca_n_components=pca_n_components,
        experiment_group=experiment_group,
        n_iter=3,
        cv=cv_estatistico,
        scoring='f1_macro',
    )

    
    # =========================================================
    # VOTING
    # =========================================================
    voting_pipe = create_named_pipe(
        VotingClassifier(estimators=estimators)
    )

    voting_param_dist = {}

    run_random_search(
        experiment_name=phase_experiment_name,
        model_name='Voting',
        pipeline=voting_pipe,
        param_distributions=voting_param_dist,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_flags=feature_flags,
        resampling_method=resampling_method,
        use_pca=use_pca,
        pca_n_components=pca_n_components,
        experiment_group=experiment_group,
        n_iter=2,
        cv=cv_estatistico,
        scoring='f1_macro',
    )
    
    # =========================================================
    # STACKING
    # =========================================================
    stacking_pipe = create_named_pipe(
        StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression()
        )
    )

    stacking_param_dist = {}

    run_random_search(
        experiment_name=phase_experiment_name,
        model_name='Stacking',
        pipeline=stacking_pipe,
        param_distributions=stacking_param_dist,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_flags=feature_flags,
        resampling_method=resampling_method,
        use_pca=use_pca,
        pca_n_components=pca_n_components,
        experiment_group=experiment_group,
        n_iter=4,
        cv=cv_estatistico,
        scoring='f1_macro',
    )

    # =========================================================
    # LIGHTGBM
    # =========================================================
    if LGBMClassifier is not None:

        lgb_search_pipe = create_named_pipe(
            LGBMClassifier(
                verbose=-1,
                force_col_wise=True
            )
        )

        lgb_param_dist = {}

        run_random_search(
            experiment_name=phase_experiment_name,
            model_name='LightGBM',
            pipeline=lgb_search_pipe,
            param_distributions=lgb_param_dist,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_flags=feature_flags,
            resampling_method=resampling_method,
            use_pca=use_pca,
            pca_n_components=pca_n_components,
            experiment_group=experiment_group,
            n_iter=12,
            cv=cv_estatistico,
            scoring='f1_macro',
        )

    else:
        print(
            'LightGBM não encontrado; '
            'pulando RandomizedSearch para LightGBM.'
        )
