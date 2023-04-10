from ElecEuro import *
import multiprocessing

countries = ["France", "Germany", "Netherlands", "Spain"]

def Processing(country):
    data = pd.read_csv(os.path.join("/Users", "Ethan", "Dev", "Thesis Work", "Data", country, "combined_data.csv"))
    aid_regression(data, country, time_start=18, time_end=19, log_adj=0, verbose="a")

if __name__ == "__main__":
        lst = []
        with multiprocessing.Pool() as pool:
                lst = pool.map(Processing, countries)
        
        print(lst)