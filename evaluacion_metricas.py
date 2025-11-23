import os
import pathlib
import numpy as np
import pandas as pd
import faiss
import extractores as ext

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


IMAGES_TEST_PATH = "Imgs test"   
K_VECINOS = 5

ROOT_PATH = pathlib.Path(__file__).resolve().parent
TEST_PATH = ROOT_PATH / IMAGES_TEST_PATH
DB_PATH = ROOT_PATH / "database"
DB_FILE = DB_PATH / "db.csv"

INDEX_MAP = {
    "Histograma":      "caracteristicas_hist.index",
    "SIFT (BoW)":      "caracteristicas_sift.index",
    "Haralick":        "caracteristicas_haralick.index",
    "VGG19":           "caracteristicas_vgg19.index",
    "MobileNetV2":     "caracteristicas_mobilenetv2.index",
}


def obtener_clase(ruta_nombre):
    nombre_archivo = pathlib.Path(str(ruta_nombre)).name
    if '_' in nombre_archivo:
        return nombre_archivo.split('_')[0]
    return pathlib.Path(nombre_archivo).stem

def calcular_matriz_confusion(predicciones, clases_reales_unicas):
    y_real = []
    y_pred = []
    
    for real, lista_preds in predicciones:
        for pred in lista_preds:
            y_real.append(real)
            y_pred.append(pred)
            
    df_conf = pd.crosstab(
        pd.Series(y_real, name='Real'),
        pd.Series(y_pred, name='Predicho'),
        normalize='index'
    )
    return df_conf


def evaluacion():
    print("--- INICIANDO EVALUACIÓN COMPLETA (5 EXTRACTORES) ---")
    
    if not TEST_PATH.exists() or not DB_FILE.exists():
        print("Revisa que existan la carpeta de test y db.csv")
        return
    
    df_bd = pd.read_csv(DB_FILE)
    df_bd['clase_real'] = df_bd['image'].apply(obtener_clase)
    clases_unicas = sorted(df_bd['clase_real'].unique())
    
    imagenes_test = []
    for patron in ["*.jpg", "*.jpeg", "*.png"]:
        imagenes_test.extend(TEST_PATH.glob(patron))
    
    if not imagenes_test:
        print("La carpeta de test está vacía.")
        return

    print(f"Procesando {len(imagenes_test)} imágenes de prueba...\n")

    resumen_general = []

    for nombre_extractor, archivo_indice in INDEX_MAP.items():
        ruta_indice = DB_PATH / archivo_indice
        
        if not ruta_indice.exists():
            print(f"No se encontró el índice para {nombre_extractor}")
            continue
            
        indice_faiss = faiss.read_index(str(ruta_indice))
        
        datos_para_matriz = []
        precisiones = []
        estadisticas_clases = {}

        print(f"Evaluando: {nombre_extractor}...")
        
        for ruta_imagen in imagenes_test:
            try:
                clase_consulta = obtener_clase(ruta_imagen)
                if clase_consulta not in estadisticas_clases:
                    estadisticas_clases[clase_consulta] = []

                vector = None
                if nombre_extractor == "Histograma":
                    vector = ext.extraer_histograma(ruta_imagen)
                elif nombre_extractor == "SIFT (BoW)":
                    libro_codigos = DB_PATH / "sift_codebook.faiss"
                    vector = ext.extraer_sift_bow(ruta_imagen, str(libro_codigos), n_clusters=256)
                elif nombre_extractor == "Haralick":
                    vector = ext.extraer_haralick(ruta_imagen)
                elif nombre_extractor == "VGG19":
                    vector = ext.extraer_vgg19(ruta_imagen, dispositivo="cpu")
                elif nombre_extractor == "MobileNetV2":
                    vector = ext.extraer_mobilenetv2(ruta_imagen, dispositivo="cpu")
                
                if vector is None: continue

                vector = vector.astype("float32")
                faiss.normalize_L2(vector)
                _, I = indice_faiss.search(vector, K_VECINOS)
                
                preds_query = []
                aciertos = 0
                
                for idx_bd in I[0]:
                    clase_recuperada = df_bd.iloc[idx_bd]['clase_real']
                    preds_query.append(clase_recuperada)
                    
                    if clase_recuperada.lower() == clase_consulta.lower():
                        aciertos += 1
                
                precision_actual = aciertos / K_VECINOS
                precisiones.append(precision_actual)
                datos_para_matriz.append((clase_consulta, preds_query))
                estadisticas_clases[clase_consulta].append(precision_actual)

            except Exception as e:
                print(f"Error en {ruta_imagen.name}: {e}")

        promedio_global = np.mean(precisiones)
 
        promedios_clase = {k: np.mean(v) for k, v in estadisticas_clases.items()}
        mejor_clase = max(promedios_clase, key=promedios_clase.get)
        peor_clase = min(promedios_clase, key=promedios_clase.get)

        resumen_general.append({
            "Extractor": nombre_extractor, 
            "Global Precision@5": f"{promedio_global:.2%}",
            "Mejor Clase": f"{mejor_clase} ({promedios_clase[mejor_clase]:.0%})",
            "Peor Clase": f"{peor_clase} ({promedios_clase[peor_clase]:.0%})"
        })
        
        print(f"\nMATRIZ DE CONFUSIÓN: {nombre_extractor}")
        matriz = calcular_matriz_confusion(datos_para_matriz, clases_unicas)
        print(matriz.applymap(lambda x: f"{x:.0%}").to_string())
        print("-" * 60 + "\n")

    print("\n" + "="*80)
    print(f" TABLA COMPARATIVA FINAL (Precision@{K_VECINOS})")
    print("="*80)
    df_resumen = pd.DataFrame(resumen_general)
    print(df_resumen.to_string(index=False))
    print("="*80)

if __name__ == "__main__":
    evaluacion()