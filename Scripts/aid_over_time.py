from ElecEuro import *
import multiprocessing


countries = ["France", "Germany", "Netherlands", "Spain"]

if __name__ == "__main__":
    with multiprocessing.Pool() as pool:
        lst = pool.map(aid_timeperiod_differences, countries)
        print(lst)