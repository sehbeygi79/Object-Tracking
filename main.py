import yaml
from types import SimpleNamespace

from object_tracking import ObjectTracking


def load_configs(config_path="configs.yaml"):
    try:
        with open(config_path, "r") as f:
            configs = yaml.safe_load(f)
        return SimpleNamespace(**configs)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None


if __name__ == "__main__":
    cfg = load_configs(config_path="configs.yaml")
    if cfg is None:
        exit(1)

    try:
        object_tracking = ObjectTracking(cfg)
        object_tracking.run()
    except Exception as e:
        print(f"An error occurred during video processing: {e}")
        exit(1)
