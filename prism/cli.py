import argparse, sys
from .run import run_all

def main(argv=sys.argv[1:]):
    p = argparse.ArgumentParser(prog="prism")
    p.add_argument("--ticker", required=True)
    p.add_argument("--start", default="2000-01-01")
    p.add_argument("--end",   default=None)
    args = p.parse_args(argv)

    run_all(args.ticker.upper(), args.start, args.end)

if __name__ == "__main__":
    main()
