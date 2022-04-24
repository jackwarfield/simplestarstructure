### import statements ###
import argparse as ap

### main function ###
def main(args: ap.Namespace) -> int:

  return 0

###############################################################################
###############################################################################

if __name__ == "__main__":
  parser = ap.ArgumentParser(description="")
  _= parser.add_argument(
      "-e",
      "--example",
      help="",
      default="",
      type=str,
      )
  args = parser.parse_args()

  raise SystemExit(main(args))
