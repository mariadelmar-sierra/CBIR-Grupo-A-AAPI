import os
# Solución al error de librerías duplicadas
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pathlib
import numpy as np
import pandas as pd
import faiss
import extractores as ext

# --------------------------------------------------------
# CONFIGURACIÓN Y RUTAS
# --------------------------------------------------------
NOMBRE_CARPETA_TEST = "Imgs test"   
K_VECINOS = 5

RUTA_RAIZ = pathlib.Path(__file__).resolve().parent
RUTA_TEST = RUTA_RAIZ / NOMBRE_CARPETA_TEST
RUTA_BD = RUTA_RAIZ / "database"
ARCHIVO_BD = RUTA_BD / "db.csv"

# AHORA SÍ: Todos los extractores activos
MAPA_INDICES = {
    "Histograma":      "caracteristicas_hist.index",
    "Haralick":        "caracteristicas_haralick.index",
    "SIFT (BoW)":      "caracteristicas_sift.index",
    "VGG19":           "caracteristicas_vgg19.index",
    "MobileNetV2":     "caracteristicas_mobilenetv2.index",
}

# --------------------------------------------------------
# FUNCIONES AUXILIARES
# --------------------------------------------------------
def obtener_clase_desde_nombre(ruta_o_nombre):
    """ Extrae 'Lilly' de 'Lilly_1.jpg' """
    nombre_archivo = pathlib.Path(str(ruta_o_nombre)).name
    if '_' in nombre_archivo:
        return nombre_archivo.split('_')[0]
    return pathlib.Path(nombre_archivo).stem

def calcular_matriz_confusion(predicciones, clases_reales_unicas):
    """ Genera la matriz de confusión normalizada """
    y_real = []
    y_pred = []
    
    for real, lista_preds in predicciones:
        for pred in lista_preds:
            y_real.append(real)
            y_pred.append(pred)
            
    df_conf = pd.crosstab(
        pd.Series(y_real, name='Real'),
        pd.Series(y_pred, name='Predicho'),
        normalize='index' # Porcentajes por fila
    )
    return df_conf

# --------------------------------------------------------
# EVALUACIÓN PRINCIPAL
# --------------------------------------------------------
def evaluar_todo():
    print("--- INICIANDO EVALUACIÓN COMPLETA (5 EXTRACTORES) ---")
    
    if not RUTA_TEST.exists() or not ARCHIVO_BD.exists():
        print("[ERROR] Revisa que existan la carpeta de test y db.csv")
        return
    
    # 1. Cargar DB
    df_bd = pd.read_csv(ARCHIVO_BD)
    df_bd['clase_real'] = df_bd['image'].apply(obtener_clase_desde_nombre)
    clases_unicas = sorted(df_bd['clase_real'].unique())
    
    # 2. Cargar imágenes de test
    imagenes_test = []
    for patron in ["*.jpg", "*.jpeg", "*.png"]:
        imagenes_test.extend(RUTA_TEST.glob(patron))
    
    if not imagenes_test:
        print("[ERROR] La carpeta de test está vacía.")
        return

    print(f"Procesando {len(imagenes_test)} imágenes de prueba...\n")

    resumen_general = []

    for nombre_extractor, archivo_indice in MAPA_INDICES.items():
        ruta_indice = RUTA_BD / archivo_indice
        
        if not ruta_indice.exists():
            print(f"[SKIP] No se encontró el índice para {nombre_extractor}")
            continue
            
        indice_faiss = faiss.read_index(str(ruta_indice))
        
        datos_para_matriz = []
        precisiones = []
        estadisticas_clases = {} # Para saber mejor/peor clase

        print(f"Evaluando: {nombre_extractor}...")
        
        for ruta_imagen in imagenes_test:
            try:
                clase_consulta = obtener_clase_desde_nombre(ruta_imagen)
                if clase_consulta not in estadisticas_clases:
                    estadisticas_clases[clase_consulta] = []

                # --- LÓGICA COMPLETA DE EXTRACCIÓN ---
                vector = None
                if nombre_extractor == "Histograma":
                    vector = ext.extraer_histograma(ruta_imagen)
                elif nombre_extractor == "Haralick":
                    vector = ext.extraer_haralick(ruta_imagen)
                elif nombre_extractor == "SIFT (BoW)":
                    libro_codigos = RUTA_BD / "sift_codebook.faiss"
                    vector = ext.extraer_sift_bow(ruta_imagen, str(libro_codigos), n_clusters=256)
                elif nombre_extractor == "VGG19":
                    vector = ext.extraer_vgg19(ruta_imagen, dispositivo="cpu")
                elif nombre_extractor == "MobileNetV2":
                    vector = ext.extraer_mobilenetv2(ruta_imagen, dispositivo="cpu")
                
                if vector is None: continue

                # Normalizar y Buscar
                vector = vector.astype("float32")
                faiss.normalize_L2(vector)
                _, I = indice_faiss.search(vector, K_VECINOS)
                
                # Analizar resultados
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

        # --- CÁLCULOS FINALES DEL EXTRACTOR ---
        promedio_global = np.mean(precisiones)
        
        # Calcular mejor y peor clase
        promedios_clase = {k: np.mean(v) for k, v in estadisticas_clases.items()}
        mejor_clase = max(promedios_clase, key=promedios_clase.get)
        peor_clase = min(promedios_clase, key=promedios_clase.get)

        resumen_general.append({
            "Extractor": nombre_extractor, 
            "Global P@5": f"{promedio_global:.2%}",
            "Mejor Clase": f"{mejor_clase} ({promedios_clase[mejor_clase]:.0%})",
            "Peor Clase": f"{peor_clase} ({promedios_clase[peor_clase]:.0%})"
        })
        
        # --- IMPRIMIR MATRIZ DE CONFUSIÓN ---
        print(f"\n>> MATRIZ DE CONFUSIÓN: {nombre_extractor}")
        matriz = calcular_matriz_confusion(datos_para_matriz, clases_unicas)
        print(matriz.applymap(lambda x: f"{x:.0%}").to_string())
        print("-" * 60 + "\n")

    # --- IMPRIMIR TABLA RESUMEN FINAL ---
    print("\n" + "="*80)
    print(f" TABLA COMPARATIVA FINAL (Precision@{K_VECINOS})")
    print("="*80)
    df_resumen = pd.DataFrame(resumen_general)
    print(df_resumen.to_string(index=False))
    print("="*80)

if __name__ == "__main__":
    evaluar_todo()