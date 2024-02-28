import argparse
import yaml
import os

pardir = os.path.dirname(os.path.dirname(__file__))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=["MTL", "Seven"], default="MTL", help="dataset name"
    )
    args = parser.parse_args()

    return args


def setup(args):
    # load config
    args.config_path = pardir + "/config/{}_Config.yaml".format(args.dataset)
    print("Load config yaml from %s" % args.config_path)
    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
        for key, val in config.items():
            setattr(args, key, val)


if __name__ == "__main__":
    print(pardir)
