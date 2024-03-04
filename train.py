from utils.processor import Processor
from utils.parser import Parser


def main():
    args = Parser().args

    processor = Processor(args)

    processor.start()


if __name__ == "__main__":
    main()
