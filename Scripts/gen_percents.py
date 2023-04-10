from ElecEuro import *

countries = ["France", "Germany", "Netherlands", "Spain"]

if __name__ == "__main__":
    for country in countries:
        gens = gen_read_clean(country)
        gens = gens[gens.index.hour == 18]
        gens = gens.reset_index()

        gens = gens.drop(columns = ["Percent Gas", "Total", "Start"])

        gens = gens.applymap(float)
        total_by_day = gens.sum(axis=1)

        percentage = gens.div(total_by_day, axis=0)
        print(country)

        print(percentage.mean())