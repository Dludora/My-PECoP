import argparse
import yaml
import os

pardir = os.path.dirname(os.path.dirname(__file__))


class Parser:
    def __init__(self):
        self.init_args()
        self.setup()
        self.check_config()

    def init_args(self):
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--benchmark", choices=["MTL", "Seven", "MTL_Pace"], help="dataset name"
        )
        parser.add_argument(
            "--resume",
            default=False,
            help="autoresume training from exp dir (interrupted by accident)",
        )
        parser.add_argument(
            "--exp_name", default="default", help="experiment name for this run"
        )
        parser.add_argument(
            "--subset", default="train", choices=["train", "test"], help="train or test"
        )
        parser.add_argument(
            "--class_idx",
            type=int,
            default=1,
            choices=[1, 2, 3, 4, 5, 6],
            help="class index for AQA-7",
        )
        parser.add_argument("--gpu", type=str, help="gpu id for training and testing")
        args = parser.parse_args()

        self.args = args

    def setup(self):
        # load config
        self.args.config_path = pardir + "/config/{}.yaml".format(self.args.benchmark)
        self.get_config()
        self.merge_config()

    def get_config(self):
        print(
            "------------------------\n"
            + "Load yaml from %s\n" % self.args.config_path
            + "------------------------\n"
        )
        with open(self.args.config_path) as f:
            self.config = yaml.load(f, Loader=yaml.Loader)

    def merge_config(self):
        for key, val in self.config.items():
            setattr(self.args, key, val)

    def check_config(self):
        assert self.args.benchmark in ["MTL", "Seven", "MTL_Pace"]
        self.args.exp_path = os.path.join(
            pardir, "exps", self.args.benchmark, self.args.exp_name
        )
        if self.args.benchmark == "Seven":
            print(f"Using CLASS idx {self.args.class_idx}")
            self.args.exp_path = os.path.join(
                self.args.exp_path, str(self.args.class_idx)
            )
        if self.args.benchmark == "JIGSAWS":
            print(f"Using CLASS idx {self.args.class_idx}")
            self.args.exp_path = os.path.join(
                self.args.exp_path, str(self.args.class_idx) + "_" + str(self.args.fold)
            )
