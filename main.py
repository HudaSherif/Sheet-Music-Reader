
import argparse
import os
import datetime

from projectFunctions import *
# Initialize parser
parser = argparse.ArgumentParser()

parser.add_argument("inputfolder", help = "Input File")
parser.add_argument("outputfolder", help = "Output File")

args = parser.parse_args()

with open(f"{args.outputfolder}/Output.txt", "w") as text_file:
    text_file.write("Input Folder: %s" % args.inputfolder)
    text_file.write("Output Folder: %s" % args.outputfolder)
    text_file.write("Date: %s" % datetime.datetime.now())
    
input_folder_path = args.inputfolder
output_folder_path = args.outputfolder

inputArr = os.listdir(input_folder_path)

for inputFile in inputArr:
    out_filename = ".".join(inputFile.split('.')[:-1]) + '.txt'
    try:
        generate_output_file(input_folder_path + '/' + inputFile, output_folder_path + '/' + out_filename)
    except:
        try:
            with open(output_folder_path + '/' + out_filename, 'x') as f:
                f.write("[]")
                f.close()
        except:
            with open(output_folder_path + '/' + out_filename, 'w') as f:
                f.write("[]")
                f.close()

print('Finished !!') 
