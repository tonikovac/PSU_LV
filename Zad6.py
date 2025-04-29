naziv = "SMSSpamCollection.txt"

try:
    with open(naziv, 'r', encoding='utf-8') as file:
        poruke = file.readlines()

    hamBroj = 0
    hamPoruka = 0
    spamBroj = 0
    spamPoruka = 0
    spamUpitnik = 0

    for red in poruke:
        red = red.strip()
        if not red:
            continue
        if "\t" in red:
            label, message = red.split("\t", 1)
        else:
            continue

        brojRijeci = len(message.split())

        if label.lower() == "ham":
            hamBroj += brojRijeci
            hamPoruka += 1
        elif label.lower() == "spam":
            spamBroj += brojRijeci
            spamPoruka += 1
            if message.endswith("!"):
                spamUpitnik += 1

    prosjekHam =     hamBroj / hamPoruka if hamPoruka else 0
    prosjekSpam = spamBroj / spamPoruka if spamPoruka else 0

    print(f"a)")
    print(f"Prosječan broj riječi u ham porukama: {prosjekHam:.2f}")
    print(f"Prosječan broj riječi u spam porukama: {prosjekSpam:.2f}")

    print(f"\nb)")
    print(f"Broj spam poruka koje završavaju uskličnikom: {spamUpitnik}")

except FileNotFoundError:
    print("Datoteka nije pronađena.")
