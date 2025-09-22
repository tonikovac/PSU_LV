lista = []

while True:
    broj = input()
    if  broj == "Done":
        break
    try:
        broj = float(broj)
        lista.append(broj)
    except:
        print("Nije unesen broj.")

if lista:
    print(f"\nUnijeli ste {len(lista)} brojeva.")
    print(f"Srednja vrijednost: {sum(lista) / len(lista)}")
    print(f"Minimalna vrijednost: {min(lista)}")
    print(f"Maksimalna vrijednost: {max(lista)}")
    lista.sort()
    print(f"{lista}")
else:
    print("Nije unesen ni jedan broj.")
