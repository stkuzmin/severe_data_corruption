Random forest regression under severe data corruption.
Severely damaged digital data having 79 features;  4 output target parameters.
Almost each of the features has severe damage (includes from 20% to 80% undefined NAN values ), therefore it is impossible to immediately find primary features that uniquely determine the output. The Train_Data_200k training data has 200k entries. It is necessary to predict test data test_data_100k. It is also necessary to determine the top 10 significant tags. 
I found that top tags are Tag_01 to Tag_10 in a row
accuracy on train set: 0.9957
accuracy on test set: 0.9912
