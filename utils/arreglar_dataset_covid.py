import pandas as pd
import sys

def main():
    if len(sys.argv) > 1:
        ruta_csv = sys.argv[1]
        df = pd.read_csv(ruta_csv)
        df["Filaname"] = df["Filaname"].apply(lambda fn: fn + ".png")
        df = df[["Filaname", 'Label']]
        df.to_csv(ruta_csv)

    else:
        print("Por favor, especifique una ruta del dataset")

if __name__ == "__main__":
    main()
