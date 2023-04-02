from ElecEuro import *

countries = ["Germany", "Netherlands"]

if __name__ == "__main__":
    for country in countries:
        data = pd.read_csv(os.path.join("/Users", "Ethan", "Dev", "Thesis Work", "Data", country, "combined_data.csv"))
        aid_regression(data, country, time_start=8, time_end=9, log_adj=5)