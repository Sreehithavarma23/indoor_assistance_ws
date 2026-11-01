"""Speech config loads and stays separate from NLU catalogs."""

from src.config_loader import load_speech


def test_speech_config_has_pipeline_sections():
    config = load_speech()
    assert config["sample_rate"] == 16000
    assert "preprocess" in config
    assert "vad" in config
    assert "asr" in config
    assert config["asr"]["model_size"]
    assert "intent" not in config
