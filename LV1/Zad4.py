ime = input("Ime datoteke: ")

try:
    with open(ime, 'r') as datoteka:
        total = 0.0
        count = 0

        for red in datoteka:
            if red.startswith("X-DSPAM-Confidence:"):
                number = float(red.strip().split(":")[1])
                total += number
                count += 1

        if count > 0:
            average = total / count
            print(f"Average X-DSPAM-Confidence: {average}")
        else:
            print("Nisu pronađene linije s 'X-DSPAM-Confidence:'.")

except FileNotFoundError:
    print("Datoteka nije pronađena.")
