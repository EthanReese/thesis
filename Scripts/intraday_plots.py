from ElecEuro import *

countries = ["Austria", "Spain", "France", "Netherlands", "Germany", "Italy"]

if __name__ == "__main__":
    for country in countries:
        intraday_price_plot(country)