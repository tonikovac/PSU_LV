naziv = "song.txt"

try:
    with open(naziv, 'r', encoding='utf-8') as datoteka:
        rijeci = {}

        for red in datoteka:
            red = red.strip().lower()
            for rijec in red.split():
                rijec = rijec.strip(".,!?;:\"()[]{}")
                if rijec:
                    rijeci[rijec] = rijeci.get(rijec, 0) + 1

    rjecnik = [rijec for rijec, brojac in rijeci.items() if brojac == 1]

    print(f"Broj riječi: {len(rjecnik)}")
    print("Riječi:")
    for rijec in rjecnik:
        print(rijec)

except FileNotFoundError:
    print("Datoteka nije pronađena.")
