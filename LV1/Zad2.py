try:
    ocjena = float(input())
    if ocjena > 1.0 or ocjena < 0:
        print("Uneseni broj nema trazenu vrijednost.")
    elif ocjena < 0.6:
       print("F")
    elif ocjena < 0.7:
       print("D")
    elif ocjena < 0.8:
       print("C")
    elif ocjena < 0.9:
       print("B")
    else:
       print("A")
except:
  print("Nije unesen broj.")