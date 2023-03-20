from ElecEuro import *

countries = ["Austria", "Spain", "France", "Netherlands", "Germany", "Italy"]

if __name__=="__main__":
        for country in countries:
                if(country == "Italy"):
                        betas_over_time(country, log_adj=30, timeperiod_length=365, roll_forward=30)
                else:
                        betas_over_time(country, log_adj=20, timeperiod_length=365, roll_forward=30)
