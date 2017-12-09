import re
import fileinput
import sys

# Read all data_set files
data_set = "mark"
data_train =  open('./data/' + data_set + '.train.txt').read()
data_valid =  open('./data/' + data_set + '.valid.txt').read()
data_test =  open('./data/' + data_set + '.test.txt').read()
# Put spaces between all words and punctuation. Everything lowercase
pre_data_train = re.sub( r'([a-zA-Z])([,.!;:?])', r'\1 \2', data_train).lower()
pre_data_valid = re.sub( r'([a-zA-Z])([,.!;:?])', r'\1 \2', data_valid).lower()
pre_data_test = re.sub( r'([a-zA-Z])([,.!;:?])', r'\1 \2', data_test).lower()
# Print out formatted text
print(pre_data_train)
print(pre_data_valid)
print(pre_data_test)
# Write new files
file = open("./data/pre." + data_set + ".train.txt", 'w')
file.write(pre_data_train)
file.close()
file = open("./data/pre." + data_set + ".valid.txt", 'w')
file.write(pre_data_valid)
file.close()
file = open("./data/pre." + data_set + ".test.txt", 'w')
file.write(pre_data_test)
file.close()
# Addes white space at the beginning of file
for line in fileinput.input(["./data/pre." + data_set + ".train.txt"], inplace=True):
    sys.stdout.write(' {l}'.format(l=line))
for line in fileinput.input(["./data/pre." + data_set + ".valid.txt"], inplace=True):
    sys.stdout.write(' {l}'.format(l=line))
for line in fileinput.input(["./data/pre." + data_set + ".test.txt"], inplace=True):
    sys.stdout.write(' {l}'.format(l=line))
