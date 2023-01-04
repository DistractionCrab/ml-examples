import sys
import importlib

program = importlib.import_module(sys.argv[1])
program.main(sys.argv[2:])

