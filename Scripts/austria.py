from ElecEuro import *


read_and_clean("Austria", "/Users/ethan/Dev/Thesis Work/Data/Bloomberg/VTP_data.xlsx", "BZN|AT")

timeperiod_differences("/Users/Ethan/Dev/Thesis Work/Data/Austria/combined_data.csv", "Austria", log_adj=30)

betas_over_time("/Users/Ethan/Dev/Thesis Work/Data/Austria/combined_data.csv", "Austria")