# by Dr. Pablo Rivas
# and @Tienza
# (R) 2017

import datetime
import pandas as pd

# Index of the company that we want he stock of
index = 'GOOG'

# Date standardization format
dateparse = lambda x: pd.datetime.strptime(x, '%m/%d/%Y')

# Parse stock price
sp = pd.read_csv('all_stocks_5yr.csv', parse_dates=['Date'], date_parser=dateparse)
stck = sp[sp['Name'] == index]

# Processing for news headlines
nws = pd.read_csv('Combined_News_DJIA.csv', parse_dates=['Date'], date_parser=dateparse)

# Open the dataset file
text_file = open('formatted_dataset.csv', 'w')

# Insert column titles
text_file.write('Date' + '^')
text_file.write('Month' + '^')
text_file.write('Weekday' + '^')
text_file.write('Headline' + '^')
text_file.write('ClosePriceUSD' + '\n')

for (index, row) in nws.iterrows():
    the_date = row['Date']
    day_of_week = datetime.datetime.strptime(str(the_date), '%Y-%m-%d %H:%M:%S').strftime('%w')
    month = datetime.datetime.strptime(str(the_date), '%Y-%m-%d %H:%M:%S').strftime('%-m')
    stck_by_dt = stck[stck['Date'] == the_date]
    if not stck_by_dt.empty:
        clsprc = stck_by_dt['Close']

    # for all top 25 news
        for x in range(25):
            idx = x + 1
            col = 'Top' + str(idx)

            nhl = row[col]
            nhl = nhl.replace('\n', '')
            # Write the data
            text_file.write(str(the_date) + '^')
            text_file.write(month + '^')
            text_file.write(day_of_week + '^')
            text_file.write(''.join(list(nhl)) + '^')
            text_file.write(str(list(clsprc)[0]) + '\n')

# Close the dataset file
text_file.close()

print('Finished Processing Data!')