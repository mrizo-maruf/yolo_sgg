import argparse

from yolo_sgg.core.config import load_config, save_default_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Run modular YOLO-SGG pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to OmegaConf yaml")
    parser.add_argument("--write-default-config", type=str, default=None, help="Write default config yaml and exit")
    args = parser.parse_args()

    if args.write_default_config:
        path = save_default_config(args.write_default_config)
        print(f"Wrote default config to: {path}")
        return 0

    from yolo_sgg.core.pipeline import TrackingSceneGraphPipeline

    cfg = load_config(args.config)
    pipeline = TrackingSceneGraphPipeline(cfg)
    pipeline.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
