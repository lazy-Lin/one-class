import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("anomalib_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.anomalib_args:
        raise SystemExit("请传入 anomalib 子命令参数")
    command = [sys.executable, "-m", "anomalib", *args.anomalib_args]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
