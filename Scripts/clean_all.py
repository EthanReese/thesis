from ElecEuro import *
import os

countries = ["Austria", "Spain", "France", "Netherlands", "Germany", "Italy"]

area_suffixes = {
        "France": "BZN|FR",
        "Austria": "BZN|AT",
        "Spain": "BZN|ES",
        "Netherlands":"BZN|NL",
        "Germany":"BZN|DE-LU",
        "Italy": "Italy (IT)"
}

if __name__=="__main__":
        for country in countries:
                lng_path = os.path.join("/Users", "Ethan", "Dev", "Thesis Work", "Data", "Bloomberg", "{}_lng.xlsx".format(country))
                read_and_clean(country, lng_path, area_suffixes[country])
