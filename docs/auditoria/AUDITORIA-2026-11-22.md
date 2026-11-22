# Auditoria — Advanced-Marketing-ML-Pipeline — 2026-11-22

## Estado anterior
- Um script (`enhanced-ml-pipeline.py`, 348 linhas) e README. Sem testes, sem CI (`gh run list` vazio), sem `requirements.txt` (embora o README mandasse instalá-lo).
- README prometia dashboard Streamlit e visualizações Plotly: nada disso existe (só imports não usados).
- Dados 100% sintéticos (`np.random`, seed 42).

## Problemas encontrados

| # | Sev. | Problema | Evidência |
|---|---|---|---|
| 1 | P0 | **Arquivo truncado**: termina no meio de `avaliar_modelo` (`'feature': self.models['conversion'].`) → `SyntaxError`, o script nunca executou | `ast.parse` → `SyntaxError: '{' was never closed` (linha 348); `pytest` antes: 1 failed, 6 errors |
| 2 | P1 | `OneHotEncoder(sparse=False)`: parâmetro removido no scikit-learn ≥ 1.4 (TypeError) | código; teste do pré-processador com sklearn 1.9.1 |
| 3 | P1 | **Vazamento de dados**: `valor_interesse` era feature *e* alvo do regressor (MSE 85 em escala ~1,3 milhão) | execução real antes: `MSE valor_interesse: 85.37`; depois: `1437094.95` |
| 4 | P1 | `criar_dados_avancados(n)` quebra para `n` não múltiplo de 5 (arrays de `idade` com tamanho ≠ n) | teste parametrizado `n=1001, 333` |
| 5 | P1 | Com MLflow 3, `log_model` falha: file store em modo manutenção e tipos não confiáveis no skops | execução real: `MlflowException ... untrusted types` |
| 6 | P2 | Outlier clipping por IQR aplicado também ao alvo binário `converteu` (zeraria o alvo se taxa < 25%) | leitura do código |
| 7 | P2 | `freq='H'` obsoleto no pandas; imports pesados não usados (streamlit, plotly, joblib) | código |
| 8 | P2 | Sem testes/CI/requirements; README infiel | — |

Segredos: nenhum (regex em todo o histórico; gitleaks não instalado). Vulnerabilidades: `pip-audit -r requirements.txt` → nenhuma conhecida.

## Mudanças
- `avaliar_modelo` concluída de forma mínima: F1 por CV (já presente) + importância de features do classificador treinado (nome das features via `get_feature_names_out`). Adicionado `main()`/`if __name__ == '__main__'`.
- `sparse_output=False`, `freq='h'`, `idade` com exatamente `n` linhas, alvo fora do clipping, regressor sem a própria coluna-alvo.
- `log_model(..., name=..., skops_trusted_types=[tipos gerados pelo próprio pipeline])`.
- Imports não usados removidos; `requirements.txt` fixado; `requirements-dev.txt`; `ruff.toml`; `.gitignore`.
- 8 testes (`tests/test_pipeline.py`), CI (ruff, compileall, pytest; `contents: read`; actions por SHA). README reescrito.

## Testes executados (Windows, Python 3.12)
- Antes: `pytest -q tests` → 1 failed, 6 errors (SyntaxError na importação).
- Depois: `pytest -q tests` → **8 passed** (~5 min; GridSearch + MLflow em SQLite temporário). `ruff check .` OK.
- O teste de vazamento (`test_modelo_de_valor_nao_usa_o_proprio_alvo`) foi escrito junto com a correção; não foi executado contra o código antigo, mas a execução real antes/depois (MSE 85 → 1,4 milhão) evidencia o vazamento.
- Execução real `python enhanced-ml-pipeline.py` (1000 amostras): acurácia de teste 0,6750; F1 médio CV 0,6926 (±0,0308); silhouette 0,0763; MSE 1.437.094,95.

## Riscos residuais / decisões do dono
- O MSE alto é esperado: no gerador sintético `valor_interesse` é uniforme e independente das demais features, portanto não é previsível. A métrica antiga só parecia boa por vazamento.
- Silhouette 0,08: a segmentação KMeans não encontra grupos bem separados nos dados sintéticos.
- Dashboard Streamlit não existe (não foi inventado). Sem arquivo LICENSE.
- Sobreposição de domínio com `Marketing-Analytics` (não unificado).
