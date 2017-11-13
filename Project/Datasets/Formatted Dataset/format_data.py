# by Dr. Pablo Rivas 
# and @Tienza
# (R) 2017

import pandas as pd                                                                

# Index of the company that we want he stock of
index = "GOOG"      

# Date standardization format                                              
dateparse = lambda x: pd.datetime.strptime(x, '%m/%d/%Y')

# Parse stock price                        
sp = pd.read_csv('all_stocks_5yr.csv', parse_dates = ['Date'], date_parser = dateparse)
stck = sp[sp['Name'] == index]                                                      
                                                                                   
# Processing for news headlines                                                           
nws = pd.read_csv('Combined_News_DJIA.csv', parse_dates = ['Date'], date_parser = dateparse)
                                                                                   
# Open the dataset file                                                                      
text_file = open("formatted_dataset.csv", "w")      

# Insert column titles        
text_file.write("Date" + "^")
text_file.write("Headline" + "^")
text_file.write("ClosePriceUSD" + "\n")

''' 
  Iterate through the dates of the news headlines dataset and get all the stocks for the
  specified index where the date matches the date of the headline. If the close price is
  not empty the process it for all 25 new headlines
'''
for index, row in nws.iterrows():                                                  
  theDate = row['Date']                                                            
  stckbydt = stck[stck['Date'] == theDate]                                           
  if not stckbydt.empty:                                                           
    clsprc = stckbydt['Close']                                                     
    # for all top 25 news                                                          
    for x in range(25):                                                            
      idx = x+1                                                                    
      col = 'Top' + str(idx)                                                       
      # print col                                                                  
      nhl = row[col]                                                               
      nhl = nhl.replace('\n', '')                                                               
      text_file.write(str(theDate) + "^")                                          
      text_file.write("".join(list(nhl)) + "^")                                    
      text_file.write(str(list(clsprc)[0]) + "\n")                                 
                                                                                   
# Close the dataset file                                                                                 
text_file.close()

print("Finished Processing Data!")