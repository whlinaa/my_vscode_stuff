import argparse
# # %%
# import sys
# if len(sys.argv)<=1:
#     sys.argv.append('hello world')
# print(sys.argv)
# # np.random.rand()
# # print(sys.exit(123))
# # for x in sys.argv:
# #     print(x)
# # %%
# print("hello")

# print(sys.argc)

# import sys
# try:
#     arg = sys.argv[1]
# except IndexError:
#     arg = "default string"

# print(arg[::-1])

# reverse_exc.py
# %%

# import sys

# try:
#     arg = sys.argv[1]
# except IndexError:
#     raise SystemExit(f"Usage: {sys.argv[0]} <string_to_reverse>")
# print(arg[::-1])
# %%
# sha1sum.py

# import sys
# import hashlib
# import os

# # # M1
# # data = sys.argv[1]
# # m = hashlib.sha1()
# # m.update(bytes(data, 'utf-8')) # string to byte conversion
# # print(m.hexdigest())

# # M2
# data = os.fsencode(sys.argv[1])
# m = hashlib.sha1()
# m.update(data) 
# print(m.hexdigest())
# %%
# import sys
# import hashlib
# import os

# args = sys.argv[1:]

# opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
# args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

# if "-c" in opts:
#     print(" ".join(arg.capitalize() for arg in args))
# elif "-u" in opts:
#     print(" ".join(arg.upper() for arg in args))
# elif "-l" in opts:
#     print(" ".join(arg.lower() for arg in args))
# else:
#     raise SystemExit(f"Usage: {sys.argv[0]} (-c | -u | -l) <arguments>...")
# %%
# import nbclient
# import nbformat
# from nbparameterise import (
#     extract_parameters, replace_definitions, parameter_values
# )

# with open("/Users/whlin/Library/CloudStorage/OneDrive-HKUSTConnect/Documents/python/sample_nb.ipynb") as f:
#     nb = nbformat.read(f, as_version=4)

# # Get a list of Parameter objects
# orig_parameters = extract_parameters(nb)

# # Update one or more parameters
# params = parameter_values(orig_parameters, arg1='GOOG', arg2=123343)

# # Make a notebook object with these definitions
# new_nb = replace_definitions(nb, params)

# # Execute the notebook with the new parameters
# new = nbclient.execute(new_nb)

# with open('/Users/whlin/Downloads/', 'w') as output:
#     nbformat.write(new, output)
# %%
# import sys

# opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
# args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

# if "-c" in opts:
#     print(" ".join(arg.capitalize() for arg in args))
# elif "-u" in opts:
#     print(" ".join(arg.upper() for arg in args))
# elif "-l" in opts:
#     print(" ".join(arg.lower() for arg in args))
# else:
#     raise SystemExit(f"Usage: {sys.argv[0]} (-c | -u | -l) <arguments>...")

# %%
# myls.py
# Import the argparse library
# import argparse

# import os
# import sys

# # Create the parser
# my_parser = argparse.ArgumentParser(description='List the content of a folder')

# # Add the arguments
# my_parser.add_argument('Path',
#                        metavar='path',
#                        type=str,
#                        help='the path to list')

# my_parser.add_argument('quant',
#                        metavar='quant',
#                        type=int,
#                        help='a value')

# # Execute the parse_args() method
# args = my_parser.parse_args()

# input_path = args.Path


# print(f"{args.Path = }")
# print(f"{args.quant = }")


# if not os.path.isdir(input_path):
#     print('The path specified does not exist')
#     sys.exit()


# print('\n'.join(os.listdir(input_path)))
# %%
# fromfile_example.py
# import argparse
# my_parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
# my_parser.add_argument('a',
#                        help='first argument')
# my_parser.add_argument('b',
#                        help='second argument')
# my_parser.add_argument('c',
#                        help='third argument')
# my_parser.add_argument('d',
#                        help='fourth argument')
# my_parser.add_argument('e',
#                        help='fifth argument')
# my_parser.add_argument('-v',
#                        '--verbose',
#                        action='store_true',
#                        help='an optional argument')
# # Execute parse_args()

# print(my_parser.v)

# args = my_parser.parse_args()
# print('If you read this line it means that you have provided ''all the parameters')
# %%
# # abbrev_example.py
# import argparse

# # my_parser = argparse.ArgumentParser(description='List the content of a folder', add_help=False)
# my_parser = argparse.ArgumentParser(description='List the content of a folder')
# my_parser.add_argument('--input', '-i', '-in', action='store', type=int, required=True, help='my input')
# my_parser.add_argument('--id', action='store', type=int)

# args = my_parser.parse_args()

# print(dir(args))

# # print(f"{args.i = }")
# # print(f"{args.id = }")
# %%
# Create the parser
# my_parser = argparse.ArgumentParser(description='List the content of a folder',
                                    # add_help=False)
# # %%
# # myls.py
# # Import the argparse library
# import argparse
# import os
# import sys
# # Create the parser
# my_parser = argparse.ArgumentParser(description='List the content of a folder')
# # Add the arguments
# my_parser.add_argument('Path',
#                        metavar='path',
#                        type=str,
#                        help='the path to list')
# my_parser.add_argument('-l',
#                        '--long',
#                        action='store_true',
#                        help='enable the long listing format')
# # Execute parse_args()
# args = my_parser.parse_args()

# print(dir(args))
# print(vars(args))

# # input_path = args.Path
# # if not os.path.isdir(input_path):
# #     print('The path specified does not exist')
# #     sys.exit()
# # for line in os.listdir(input_path):
# #     if args.long:  # Simplified long listing
# #         size = os.stat(os.path.join(input_path, line)).st_size
# #         line = '%10d  %s' % (size, line)
# #     print(line)                                    
# %%
# actions_example.py
import argparse

my_parser = argparse.ArgumentParser()
my_parser.version = '1.0'
my_parser.add_argument('-a', '--add', '--addition', action='store', help='store the input')
my_parser.add_argument('-b', action='store_const', const=42, help='store a pre-defined constant')
my_parser.add_argument('-c', action='store_true', help='store true if set') # store True value if specified
my_parser.add_argument('-d', action='store_false', help='store false if set')
my_parser.add_argument('-e', action='append', help='append the input if set') # 
my_parser.add_argument('-f', action='append_const', const=42, help='append a constant if set') # given a constant, append it n times if the option is specified n times
my_parser.add_argument('-g', action='count', help='count how many times this option is set') # count how many times `-g` has been specified
my_parser.add_argument('-i', action='help', help='another -h') # give another option to show help page
my_parser.add_argument('-j', action='version') # show version if specified 
my_parser.add_argument('-k', action='store', required=True, help='required argument')
my_parser.add_argument('-l', action='store', nargs=3, help='required 3 arguments')
my_parser.add_argument('-m', action='store', nargs='*', help='>=0 arguments')
my_parser.add_argument('-n', action='store', nargs='+', help='>=1 arguments')
my_parser.add_argument('-o', action='store', nargs='?', help='0 or 1 argument')
my_parser.add_argument('-p', action='store', nargs='?', help='0 or 1 argument, with default', default=12345)

args = my_parser.parse_args()
print(vars(args))