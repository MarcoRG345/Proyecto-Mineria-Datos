import pandas as pd
import glob
from pathlib import Path

def integrate_datasets():
    print("Iniciando integración de datasets...")
    # Usamos path relativo a la carpeta src donde se ejecutará
    files = sorted(glob.glob("../data/raw/Cierre_agricola_mun_*.csv"))
    
    dfs = []
    for f in files:
        print(f"Procesando {Path(f).name}...")
        df = pd.read_csv(f, encoding='latin-1', low_memory=False)
        
        # Unificar nombres de columnas
        if 'Nomcultivo Sin Um' in df.columns:
            df.rename(columns={'Nomcultivo Sin Um': 'Nomcultivo'}, inplace=True)
        if 'Preciomediorural' in df.columns:
            df.rename(columns={'Preciomediorural': 'Precio'}, inplace=True)
            
        dfs.append(df)
        
    print("Concatenando datos...")
    df_final = pd.concat(dfs, ignore_index=True)
    
    output_path = "../data/raw/siap_2010_2024.csv"
    print(f"Guardando resultado en {output_path} (Filas: {len(df_final)}, Columnas: {len(df_final.columns)})...")
    df_final.to_csv(output_path, index=False, encoding='latin-1')
    print("¡Integración completada exitosamente!")

if __name__ == "__main__":
    integrate_datasets()
