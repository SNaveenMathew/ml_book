import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--type", type =  str, default = "constrained")
args = parser.parse_args()
file = "test_quadratic.py" if args.type == "constrained" else "test_quadratic_unconstrained.py"

with open("output_" + args.type + ".txt", "w") as output:
	subprocess.call(["python", file], stdout = output)
