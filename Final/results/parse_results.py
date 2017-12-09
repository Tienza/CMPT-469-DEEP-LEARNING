import os
import re
import webbrowser

best_result = 701.019
best_validation = 1273.906
best_sentence = ""
best_file = ""

for filename in os.listdir(os.getcwd()):
    if (filename == "parse_results.py"):
        pass
    else:
        text = open(filename, 'r').read()
        working_validation = re.findall('Valid\sPerplexity:\s(\d+.\d+)', text)
        test_perplexity = re.findall('Test Perplexity:\s(\d+.\d+)', text)
        working_sentence = re.findall('Sample sentence:\s(.*)', text)
        working_result = float(test_perplexity[0])
        if best_result > working_result:
            best_result = working_result
            best_validation = working_validation[len(working_validation) - 1]
            best_sentence = working_sentence[len(working_sentence) - 1]
            best_file = filename

print("File Name: " + best_file)
print("Last Sentence: " + best_sentence)
print("Best Valid Perplexity: " + best_validation)
print("Best Test Perplexity: " + str(best_result))

webbrowser.open(best_file)