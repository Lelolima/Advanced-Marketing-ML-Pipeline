import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, mean_squared_error, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import NotFittedError
import xgboost as xgb
from datetime import datetime, timedelta
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import mlflow
import mlflow.sklearn
import logging
import joblib
from typing import Dict, Tuple, Any, List
import warnings
warnings.filterwarnings('ignore')

class EnhancedMarketingML:
    def __init__(self, log_path: str = 'marketing_ml.log'):
        """
        Inicializa o modelo com logging e validações
        """
        self.setup_logging(log_path)
        self.data = None
        self.models = {}
        self.preprocessors = {}
        self.feature_importance = None
        self.required_columns = None
        self.metrics_history = []
        
    def setup_logging(self, log_path: str) -> None:
        """
        Configura sistema de logging
        """
        logging.basicConfig(
            filename=log_path,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def validar_dados(self, df: pd.DataFrame) -> bool:
        """
        Valida o DataFrame de entrada
        """
        try:
            assert isinstance(df, pd.DataFrame), "Input deve ser um DataFrame"
            assert len(df) > 0, "DataFrame está vazio"
            assert not df.isnull().any().any(), "Dados contêm valores nulos"
            
            # Validações específicas
            if 'idade' in df.columns:
                assert df['idade'].between(18, 80).all(), "Idades devem estar entre 18 e 80"
            if 'valor_interesse' in df.columns:
                assert df['valor_interesse'].between(0, 10000).all(), "Valores devem estar entre 0 e 10000"
                
            self.logger.info("Validação de dados concluída com sucesso")
            return True
        except AssertionError as e:
            self.logger.error(f"Erro na validação dos dados: {str(e)}")
            raise
            
    def tratar_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trata outliers usando IQR
        """
        df_clean = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
            
        self.logger.info(f"Outliers tratados em {len(numeric_cols)} colunas")
        return df_clean

    def criar_dados_avancados(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Cria dataset mais complexo com features adicionais
        """
        np.random.seed(42)
        
        # Features base
        data = self._gerar_features_base(n_samples)
        
        # Features avançadas
        data = self._adicionar_features_avancadas(data)
        
        # Criar target com mais nuance
        data = self._gerar_target_avancado(data)
        
        self.data = pd.DataFrame(data)
        self.validar_dados(self.data)
        self.data = self.tratar_outliers(self.data)
        
        self.logger.info(f"Dataset criado com {n_samples} amostras e {len(data.keys())} features")
        return self.data
    
    def _gerar_features_base(self, n_samples: int) -> Dict:
        """
        Gera as features base do dataset
        """
        dates = pd.date_range(end=datetime.now(), periods=n_samples, freq='H')
        
        return {
            'data': dates,
            'categoria': np.random.choice(
                ['Transição de Carreira', 'Estudante', 'Sem Experiência'],
                n_samples, p=[0.40, 0.35, 0.25]
            ),
            'idade': np.clip(np.concatenate([
                np.random.normal(30, 3, int(n_samples * 0.6)),
                np.random.normal(20, 2, int(n_samples * 0.2)),
                np.random.normal(45, 5, int(n_samples * 0.2))
            ]), 18, 55).astype(int),
            'recursos_disponiveis': np.random.choice([True, False], n_samples, p=[0.3, 0.7]),
            'forma_pagamento': np.random.choice(['PIX', 'Cartão de Crédito'], n_samples, p=[0.6, 0.4]),
            'nivel_tech': np.random.choice(['Básico', 'Intermediário', 'Avançado'], n_samples, p=[0.5, 0.3, 0.2]),
            'interesse_ferramentas': np.random.choice(['Power BI', 'SQL', 'Python', 'Todos'], n_samples, p=[0.3, 0.2, 0.2, 0.3])
        }
    
    def _adicionar_features_avancadas(self, data: Dict) -> Dict:
        """
        Adiciona features mais complexas ao dataset
        """
        n_samples = len(data['idade'])
        
        # Features comportamentais
        data.update({
            'tempo_site': np.random.exponential(5, n_samples),
            'paginas_visitadas': np.random.poisson(4, n_samples),
            'interacoes_email': np.random.poisson(2, n_samples),
            'cliques_anuncios': np.random.poisson(3, n_samples),
            'visualizacoes_video': np.random.poisson(2, n_samples),
            'compartilhamentos_social': np.random.poisson(1, n_samples),
            
            # Features financeiras
            'valor_interesse': np.random.uniform(1000, 5000, n_samples),
            'orcamento_declarado': np.random.uniform(800, 6000, n_samples),
            
            # Features de origem
            'origem_trafego': np.random.choice(
                ['Orgânico', 'Social', 'Email', 'Ads', 'Referral'],
                n_samples, p=[0.25, 0.2, 0.2, 0.25, 0.1]
            ),
            'dispositivo': np.random.choice(
                ['Mobile', 'Desktop', 'Tablet'],
                n_samples, p=[0.5, 0.4, 0.1]
            )
        })
        
        return data
    
    def _gerar_target_avancado(self, data: Dict) -> Dict:
        """
        Gera target mais sofisticado baseado em múltiplas variáveis
        """
        # Calculando score de engajamento
        engagement_score = (
            (data['tempo_site'] > 5) * 0.2 +
            (data['paginas_visitadas'] > 3) * 0.15 +
            (data['interacoes_email'] > 1) * 0.15 +
            (data['cliques_anuncios'] > 2) * 0.1 +
            (data['visualizacoes_video'] > 1) * 0.1 +
            (data['compartilhamentos_social'] > 0) * 0.1 +
            (data['recursos_disponiveis'] == True) * 0.2
        )
        
        # Ajustando probabilidade baseado em outros fatores
        tech_bonus = np.where(data['nivel_tech'] == 'Avançado', 0.1, 
                            np.where(data['nivel_tech'] == 'Intermediário', 0.05, 0))
        
        interesse_bonus = np.where(data['interesse_ferramentas'] == 'Todos', 0.1, 0)
        
        probabilidade_final = np.clip(engagement_score + tech_bonus + interesse_bonus, 0, 1)
        
        data['converteu'] = np.random.binomial(1, probabilidade_final)
        data['engagement_score'] = engagement_score
        
        return data
    
    def preprocessar_dados(self) -> Tuple:
        """
        Prepara os dados para modelagem com preprocessamento avançado
        """
        # Definindo features
        cat_features = [
            'categoria', 'forma_pagamento', 'nivel_tech', 
            'interesse_ferramentas', 'origem_trafego', 'dispositivo'
        ]
        num_features = [
            'idade', 'tempo_site', 'paginas_visitadas', 'interacoes_email',
            'cliques_anuncios', 'visualizacoes_video', 'compartilhamentos_social',
            'valor_interesse', 'orcamento_declarado', 'engagement_score'
        ]
        
        # Features temporais
        self.data['hora_dia'] = self.data['data'].dt.hour
        self.data['dia_semana'] = self.data['data'].dt.dayofweek
        self.data['mes'] = self.data['data'].dt.month
        num_features.extend(['hora_dia', 'dia_semana', 'mes'])
        
        # Preprocessamento
        X = self.data[cat_features + num_features]
        y_conversion = self.data['converteu']
        y_value = self.data['valor_interesse']
        
        # Pipeline de preprocessamento robusto
        numeric_transformer = Pipeline([
            ('scaler', RobustScaler())
        ])
        
        categorical_transformer = Pipeline([
            ('onehot', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_features),
                ('cat', categorical_transformer, cat_features)
            ])
        
        return X, y_conversion, y_value, preprocessor
    
    def otimizar_hiperparametros(self, X: pd.DataFrame, y: pd.Series, pipeline: Pipeline) -> Dict:
        """
        Otimiza hiperparâmetros usando GridSearchCV
        """
        param_grid = {
            'classifier__max_depth': [3, 4, 5],
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.01, 0.1]
        }
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        self.logger.info(f"Melhores parâmetros encontrados: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    
    def treinar_modelos(self) -> Dict:
        """
        Treina os modelos com validação cruzada e balanceamento
        """
        X, y_conversion, y_value, preprocessor = self.preprocessar_dados()
        
        # Divisão treino-teste
        X_train, X_test, y_conv_train, y_conv_test, y_val_train, y_val_test = train_test_split(
            X, y_conversion, y_value, test_size=0.2, random_state=42
        )
        
        # Pipeline de conversão com SMOTE
        conversion_pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss'
            ))
        ])
        
        # Pipeline de valor
        value_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor())
        ])
        
        # Otimização e treinamento
        with mlflow.start_run():
            # Modelo de conversão
            best_conversion_model = self.otimizar_hiperparametros(
                X_train, y_conv_train, conversion_pipeline
            )
            conv_pred = best_conversion_model.predict(X_test)
            
            # Modelo de valor
            value_pipeline.fit(X_train, y_val_train)
            val_pred = value_pipeline.predict(X_test)
            
            # Métricas
            conv_report = classification_report(y_conv_test, conv_pred, output_dict=True)
            val_mse = mean_squared_error(y_val_test, val_pred)
            
            # Log métricas
            mlflow.log_metrics({
                'accuracy': conv_report['accuracy'],
                'f1_score': conv_report['1']['f1-score'],
                'mse_valor': val_mse
            })
            
            # Log modelos
            mlflow.sklearn.log_model(best_conversion_model, "conversion_model")
            mlflow.sklearn.log_model(value_pipeline, "value_model")
        
        self.models['conversion'] = best_conversion_model
        self.models['value'] = value_pipeline
        
        # Segmentação de clientes
        X_transformed = preprocessor.fit_transform(X)
        kmeans = KMeans(n_clusters=4, random_state=42)
        clusters = kmeans.fit_predict(X_transformed)
        
        self.models['cluster'] = kmeans
        
        return {
            'conversion_metrics': conv_report,
            'value_prediction_mse': val_mse,
            'silhouette_score': silhouette_score(X_transformed, clusters)
        }
    
    def avaliar_modelo(self) -> Dict:
        """
        Realiza uma avaliação completa do modelo
        """
        X, y_conversion, _, _ = self.preprocessar_dados()
        
        # Validação cruzada
        cv_scores = cross_val_score(
            self.models['conversion'],
            X, y_conversion,
            cv=5,
            scoring='f1'
        )
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.models['conversion'].