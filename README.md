# Advanced Marketing ML Pipeline

Pipeline de machine learning para marketing em um único script Python (`enhanced-ml-pipeline.py`).

> **Atenção: os dados são sintéticos.** O script gera um dataset simulado (`np.random`, semente 42) de leads com
> features demográficas, comportamentais e financeiras. Não há leitura de dados reais; as métricas impressas servem
> apenas para demonstrar o fluxo.

## O que o código faz

1. **Geração e validação de dados** (`criar_dados_avancados`): dataset sintético, validação (nulos, faixas de idade e
   valor) e recorte de outliers por IQR nas colunas numéricas (exceto o alvo `converteu`).
2. **Pré-processamento** (`preprocessar_dados`): `RobustScaler` nas numéricas, `OneHotEncoder` nas categóricas e
   features temporais (hora, dia da semana, mês).
3. **Modelos** (`treinar_modelos`):
   - classificação de conversão com XGBoost + SMOTE, hiperparâmetros por `GridSearchCV` (F1, 5 folds);
   - regressão de `valor_interesse` com `GradientBoostingRegressor`;
   - segmentação com KMeans (k=4) e silhouette score.
4. **Avaliação** (`avaliar_modelo`): F1 por validação cruzada (5 folds) e importância das features do classificador.
5. **Rastreamento com MLflow**: métricas e modelos registrados localmente.

Não existe dashboard Streamlit nem visualizações Plotly neste repositório (versões anteriores do README citavam isso).

## Requisitos

- Python 3.10+ (testado com 3.12)
- Dependências em `requirements.txt` (versões fixadas e testadas)

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt          # execução
pip install -r requirements-dev.txt      # execução + testes/lint
```

## Execução

```bash
python enhanced-ml-pipeline.py
```

Imprime acurácia, MSE, silhouette, F1 médio de validação cruzada e as 10 features mais importantes. Gera
`marketing_ml.log` e os registros do MLflow no diretório atual.

### Variáveis de ambiente (opcionais)

| Variável | Uso | Padrão |
|---|---|---|
| `MLFLOW_TRACKING_URI` | Onde o MLflow grava runs e modelos | padrão do MLflow (`sqlite:///mlflow.db` no MLflow 3) |

Não há segredos nem chaves de API; por isso não há `.env.example`.

## Testes e lint

```bash
ruff check .
pytest -q tests
```

Os testes executam o pipeline ponta a ponta com amostra reduzida (MLflow em SQLite temporário). O CI
(`.github/workflows/ci.yml`) roda lint, compilação e testes em cada push/PR.

## Relação com outros repositórios

O repositório `Marketing-Analytics` trata do mesmo domínio (leads de cursos, perfis "Transição de carreira",
"Estudante" etc.) com uma pipeline ETL mais simples. Os projetos se sobrepõem, mas são independentes.

## Licença

O README original declarava MIT, mas não há arquivo `LICENSE` no repositório.
