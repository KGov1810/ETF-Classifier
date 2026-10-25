from pathlib import Path

import yaml

from etf_classifier.config import PROJECT_ROOT
from etf_classifier.features.build import main


def _config(tmp_path, settings) -> str:
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(settings, allow_unicode=True), encoding="utf-8")
    return str(path)


def test_estimate_only_never_calls_azure(sample_csv, azure_settings, fake_client, tmp_path, capsys):
    code = main(["--csv", str(sample_csv), "--config", _config(tmp_path, azure_settings), "--estimate-only"],
                client=fake_client)
    assert code == 0
    assert fake_client.calls == []
    assert "à calculer" in capsys.readouterr().out


def test_refusing_the_confirmation_cancels_without_spending(sample_csv, azure_settings, fake_client, tmp_path):
    code = main(["--csv", str(sample_csv), "--config", _config(tmp_path, azure_settings)],
                client=fake_client, input_fn=lambda prompt: "n")
    assert code == 1
    assert fake_client.calls == []


def test_yes_builds_and_saves_the_pipeline(sample_csv, azure_settings, fake_client, tmp_path, capsys):
    code = main(["--csv", str(sample_csv), "--config", _config(tmp_path, azure_settings), "--yes"],
                client=fake_client)
    output = capsys.readouterr().out
    assert code == 0
    assert fake_client.calls
    saved = list(Path(azure_settings["features"]["artifacts_dir"]).glob("*/pipeline.joblib"))
    assert len(saved) == 1
    assert "embeddings" in output and "Coût réel" in output


def test_second_build_costs_nothing(sample_csv, azure_settings, fake_client, tmp_path, capsys):
    config = _config(tmp_path, azure_settings)
    main(["--csv", str(sample_csv), "--config", config, "--yes"], client=fake_client)
    calls = len(fake_client.calls)
    main(["--csv", str(sample_csv), "--config", config], client=fake_client, input_fn=lambda p: "n")
    assert len(fake_client.calls) == calls
    assert "0 texte(s) à calculer" in capsys.readouterr().out


def test_no_azure_option_builds_the_local_blocks(sample_csv, azure_settings, tmp_path, capsys):
    code = main(["--csv", str(sample_csv), "--config", _config(tmp_path, azure_settings), "--no-azure"])
    assert code == 0
    assert "embeddings" not in capsys.readouterr().out.split("Blocs construits")[1]


def test_env_example_contains_no_secret():
    content = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    values = dict(line.split("=", 1) for line in content.splitlines() if line and not line.startswith("#"))
    assert values["AZURE_OPENAI_API_KEY"] == ""
    assert "AZURE_OPENAI_ENDPOINT" in values
