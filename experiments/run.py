import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description='Run a repository experiment')
    parser.add_argument('task', choices=('cec2022', 'mstar', 'atrnet_star'))
    args = parser.parse_args(sys.argv[1:2])
    remaining = sys.argv[2:]
    script = Path(__file__).resolve().parent / ('run_' + args.task + '.py')
    process = subprocess.Popen([sys.executable, str(script), *remaining])
    try:
        return process.wait()
    except KeyboardInterrupt:
        return process.wait()


if __name__ == '__main__':
    raise SystemExit(main())
