import yaml

from imgclf.config import load_config


def test_load_config_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    test_config = {"name": "image-classifier", "version": 1}

    with open(config_file, "w") as file:
        yaml.safe_dump(test_config, file)

    config = load_config(str(config_file))

    assert config == test_config
