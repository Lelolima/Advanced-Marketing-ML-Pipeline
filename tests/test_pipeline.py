import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location("enhanced_ml_pipeline", ROOT / "enhanced-ml-pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def ml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///" + (tmp_path / "mlflow.db").as_posix())
    mod = _load_module()
    return mod.EnhancedMarketingML(log_path=str(tmp_path / "test.log"))


def test_modulo_compila():
    # Antes da correção o arquivo terminava truncado (SyntaxError).
    _load_module()


@pytest.mark.parametrize("n", [1000, 1001, 333])
def test_dataset_tem_n_linhas_para_qualquer_n(ml, n):
    df = ml.criar_dados_avancados(n)
    assert len(df) == n
    assert set(df["converteu"].unique()) <= {0, 1}


def test_preprocessador_aceita_sklearn_atual(ml):
    ml.criar_dados_avancados(200)
    X, y, _, pre = ml.preprocessar_dados()
    Xt = pre.fit_transform(X)
    assert Xt.shape[0] == 200


def test_validacao_rejeita_nulos(ml):
    import pandas as pd

    with pytest.raises(AssertionError):
        ml.validar_dados(pd.DataFrame({"idade": [20, None]}))


def test_treino_e_avaliacao_ponta_a_ponta(ml):
    ml.criar_dados_avancados(400)
    metricas = ml.treinar_modelos()
    assert 0.0 <= metricas["conversion_metrics"]["accuracy"] <= 1.0
    assert metricas["value_prediction_mse"] >= 0
    assert -1.0 <= metricas["silhouette_score"] <= 1.0
    aval = ml.avaliar_modelo()
    assert 0.0 <= aval["cv_f1_mean"] <= 1.0
    fi = aval["feature_importance"]
    assert len(fi) > 0 and {"feature", "importance"} <= set(fi.columns)


def test_modelo_de_valor_nao_usa_o_proprio_alvo(ml):
    ml.criar_dados_avancados(300)
    ml.treinar_modelos()
    nomes = ml.models["value"].named_steps["preprocessor"].get_feature_names_out()
    assert not any("valor_interesse" in n for n in nomes)
