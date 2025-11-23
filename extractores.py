import os, io, glob, pathlib
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import glob
import faiss
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
import torch
from torchvision import models, transforms


def cargar_imagen_rgb(entrada):
    """
    Carga una imagen desde distintos formatos (PIL.Image, bytes, archivo o ruta) 
    y la convierte a formato RGB (PIL.Image).
    """
    if isinstance(entrada, Image.Image):
        imagen = entrada
    elif isinstance(entrada, (bytes, bytearray)):
        imagen = Image.open(io.BytesIO(entrada))
    elif hasattr(entrada, 'read'):
        imagen = Image.open(entrada)
    else:
        imagen = Image.open(str(entrada))
    
    imagen_rgb = imagen.convert('RGB')
    return imagen_rgb

def convertir_cv2(imagen_pil):
    """
    Convierte una imagen PIL (RGB) a NumPy RGB 
    compatible con OpenCV.
    """
    return np.array(imagen_pil)

def normalizar_l2(matriz):
    """
    Aplica normalización L2 fila por fila a una matriz NumPy.
    Devuelve la matriz con cada fila normalizada (norma = 1).
    """
    matriz = matriz.astype('float32', copy=False)
    norma = np.linalg.norm(matriz, axis=1, keepdims=True) + 1e-10
    return matriz / norma


# 1) HISTOGRAMA
def extraer_histograma(imagen_entrada, bins=(16, 16, 16)):
    """
    Calcula el histograma de color de una imagen.
    Aplica normalización raíz cuadrada y normalización L2 (fila con norma = 1).
    """
    imagen_pil = cargar_imagen_rgb(imagen_entrada)
    imagen_rgb = convertir_cv2(imagen_pil)
    imagen_lab = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2LAB)
    
    histograma = cv2.calcHist(
        [imagen_lab], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256]
    ).flatten()
    
    histograma = np.sqrt(histograma).astype('float32')
    histograma_normalizado = normalizar_l2(histograma[None, :])
    return histograma_normalizado


# 2) HARALICK
def extraer_haralick(imagen_entrada):
    """
    Calcula las 6 características estadísticas de Haralick 
    basadas en la matriz de co-ocurrencia de niveles de gris (GLCM).
    Aplica normalización L2 al vector resultante.
    """
    imagen_pil = cargar_imagen_rgb(imagen_entrada)
    imagen_gris = np.array(imagen_pil.convert("L"))
    imagen_gris = img_as_ubyte(imagen_gris)

    glcm = graycomatrix(
        imagen_gris,
        distances=[1, 2],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        symmetric=True,
        normed=True
    )

    propiedades = [
        graycoprops(glcm, prop).mean()
        for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    ]

    vector = np.array(propiedades, dtype="float32")
    vector_normalizado = normalizar_l2(vector[None, :])
    return vector_normalizado


# 3) SIFT + Bag of Words
def aplicar_rootsift(descriptores):
    """
    Aplica la normalización RootSIFT a los descriptores SIFT.
    """
    if descriptores is None or len(descriptores) == 0:
        return None

    descriptores = descriptores.astype('float32')
    descriptores /= (np.sum(descriptores, axis=1, keepdims=True) + 1e-10)
    return np.sqrt(descriptores)


CACHE_SIFT = {"index": None, "path": None, "n_clusters": None, "sift_model": None}

def extraer_sift_bow(imagen_entrada, ruta_codebook, n_clusters=256):
    """
    Extrae un vector de características basado en el modelo Bag of Words (BoW)
    usando descriptores SIFT con normalización RootSIFT.
    """
    # Cargar o actualizar el modelo FAISS y el detector SIFT en caché
    if (CACHE_SIFT["index"] is None) or (CACHE_SIFT["path"] != ruta_codebook):
        CACHE_SIFT["index"] = faiss.read_index(ruta_codebook)
        CACHE_SIFT["path"] = ruta_codebook
        CACHE_SIFT["n_clusters"] = n_clusters
        CACHE_SIFT["sift_model"] = cv2.SIFT_create()

    # Preprocesar la imagen
    imagen_pil = cargar_imagen_rgb(imagen_entrada)
    imagen_rgb = convertir_cv2(imagen_pil)
    imagen_gris = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2GRAY)

    # Extraer descriptores SIFT y aplicar RootSIFT
    _, descriptores = CACHE_SIFT["sift_model"].detectAndCompute(imagen_gris, None)
    descriptores = aplicar_rootsift(descriptores)

    # Crear histograma de ocurrencias (BoW)
    histograma = np.zeros((CACHE_SIFT["n_clusters"],), dtype='float32')

    if descriptores is not None and len(descriptores) > 0:
        _, indices = CACHE_SIFT["index"].search(descriptores.astype('float32'), 1)
        for idx in indices.flatten():
            histograma[idx] += 1.0

    # Normalización del histograma
    histograma /= (np.sum(histograma) + 1e-10)
    histograma = np.sqrt(histograma)
    histograma_normalizado = normalizar_l2(histograma[None, :])
    return histograma_normalizado


# 4) VGG19
CACHE_VGG19 = {"modelo": None, "preprocesador": None, "dispositivo": None}

def cargar_vgg19(dispositivo: str = "cpu"):
    """
    Carga el modelo VGG19 preentrenado y el pipeline de preprocesamiento.
    Usa caché para evitar recargar el modelo múltiples veces.
    """
    if CACHE_VGG19["modelo"] is None:
        pesos = models.VGG19_Weights.IMAGENET1K_V1
        base = torch.nn.Sequential(*list(models.vgg19(weights=pesos).features.children())[:-2])

        modelo = torch.nn.Sequential(
            base,
            torch.nn.AdaptiveAvgPool2d((1, 1))
        ).to(dispositivo).eval()

        preprocesador = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=pesos.transforms().mean, std=pesos.transforms().std)
        ])

        # Guardar en caché
        CACHE_VGG19["modelo"] = modelo
        CACHE_VGG19["preprocesador"] = preprocesador
        CACHE_VGG19["dispositivo"] = torch.device(dispositivo)

        # Desactivar gradientes
        for parametro in modelo.parameters():
            parametro.requires_grad = False

    return CACHE_VGG19["modelo"], CACHE_VGG19["preprocesador"], CACHE_VGG19["dispositivo"]


def extraer_vgg19(imagen_entrada, dispositivo: str = "cpu"):
    """
    Extrae un vector de características usando el modelo VGG19 preentrenado.
    """
    modelo, preprocesador, dev = cargar_vgg19(dispositivo)
    imagen_pil = cargar_imagen_rgb(imagen_entrada)
    tensor = preprocesador(imagen_pil).unsqueeze(0).to(dev)

    with torch.no_grad():
        caracteristicas = modelo(tensor).flatten(1).cpu().numpy().astype("float32")
    caracteristicas_normalizadas = normalizar_l2(caracteristicas)

    return caracteristicas_normalizadas


# 5) MobileNetV2 (GAP → 1280-D)
CACHE_MOBILENETV2 = {"modelo": None, "preprocesador": None, "dispositivo": None}

def cargar_mobilenetv2(dispositivo: str = "cpu"):
    """
    Carga el modelo MobileNetV2 preentrenado y el pipeline de preprocesamiento.
    Usa caché para evitar recargar el modelo múltiples veces.
    """
    if CACHE_MOBILENETV2["modelo"] is None:
        pesos = models.MobileNet_V2_Weights.IMAGENET1K_V2
        base = models.mobilenet_v2(weights=pesos).features

        modelo = torch.nn.Sequential(
            base,
            torch.nn.AdaptiveAvgPool2d((1, 1))
        ).to(dispositivo).eval()

        preprocesador = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=pesos.transforms().mean, std=pesos.transforms().std)
        ])

        # Guardar en caché
        CACHE_MOBILENETV2["modelo"] = modelo
        CACHE_MOBILENETV2["preprocesador"] = preprocesador
        CACHE_MOBILENETV2["dispositivo"] = torch.device(dispositivo)

        # Desactivar gradientes
        for parametro in modelo.parameters():
            parametro.requires_grad = False

    return (CACHE_MOBILENETV2["modelo"], CACHE_MOBILENETV2["preprocesador"], CACHE_MOBILENETV2["dispositivo"])


def extraer_mobilenetv2(imagen_entrada, dispositivo: str = "cpu"):
    """
    Extrae un vector de características usando el modelo MobileNetV2 preentrenado.
    """
    modelo, preprocesador, dev = cargar_mobilenetv2(dispositivo)
    imagen_pil = cargar_imagen_rgb(imagen_entrada)
    tensor = preprocesador(imagen_pil).unsqueeze(0).to(dev)

    with torch.no_grad():
        caracteristicas = modelo(tensor).flatten(1).cpu().numpy().astype("float32")
    caracteristicas_normalizadas = normalizar_l2(caracteristicas)
    return caracteristicas_normalizadas


# db.csv + .index
def guardar_indice_faiss(vectores, ruta_salida):
    """
    Crea y guarda un índice FAISS entrenado usando la métrica de coseno.
    """
    vectores = vectores.astype('float32')
    vectores = normalizar_l2(vectores)
    d = vectores.shape[1]

    # Determinar número de listas (clusters)
    n_listas = max(10, min(50, vectores.shape[0] // 10))

    # Crear y entrenar índice FAISS
    cuantizador = faiss.IndexFlatIP(d)
    indice = faiss.IndexIVFFlat(cuantizador, d, n_listas, faiss.METRIC_INNER_PRODUCT)

    indice.train(vectores)
    indice.add(vectores)

    faiss.write_index(indice, str(ruta_salida))
    print(f"[OK] Índice FAISS guardado en {ruta_salida} (N={vectores.shape[0]}, D={d}, n_listas={n_listas})")


def entrenar_codebook_sift(rutas_imagenes, ruta_codebook_salida, n_clusters=256, limite_muestras=400):
    """
    Entrena un codebook (diccionario visual) para BoW con descriptores SIFT,
    usando K-Means de FAISS, y lo guarda como un índice plano (IndexFlatL2).
    """
    sift = cv2.SIFT_create()
    todos_descriptores = []
    contador_imagenes = 0

    for ruta in rutas_imagenes:
        imagen_pil = cargar_imagen_rgb(ruta)
        imagen_rgb = convertir_cv2(imagen_pil)
        imagen_gris = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2GRAY)
        _, descriptores = sift.detectAndCompute(imagen_gris, None)
        descriptores = aplicar_rootsift(descriptores)

        if descriptores is not None and len(descriptores) > 0:
            todos_descriptores.append(descriptores)
            contador_imagenes += 1

        if contador_imagenes >= limite_muestras:
            break

    if not todos_descriptores:
        raise RuntimeError("No se pudieron obtener descriptores SIFT para entrenar el codebook.")

    X = np.vstack(todos_descriptores).astype('float32')
    print(f"[*] Entrenando KMeans SIFT con {X.shape[0]} descriptores…")

    kmeans = faiss.Kmeans(d=X.shape[1], k=n_clusters, niter=20, verbose=True)
    kmeans.train(X)

    indice = faiss.IndexFlatL2(X.shape[1])
    indice.add(kmeans.centroids)

    faiss.write_index(indice, str(ruta_codebook_salida))
    print(f"[OK] Codebook SIFT guardado en {ruta_codebook_salida}")


# FUNCIÓN PRINCIPAL
def crear_indices_csv():
    """
    Genera la base de datos de imágenes y los índices FAISS de características.
    """
    print("[INFO] Iniciando creación de base de datos e índices...")

    # Directorios principales
    ROOT = pathlib.Path(__file__).resolve().parent
    IMAGES_DIR = ROOT / "images"
    DB_DIR = ROOT / "database"
    DB_DIR.mkdir(exist_ok=True, parents=True)

    # Rutas y parámetros
    DB_CSV = DB_DIR / "db.csv"
    CODEBOOK_PATH = DB_DIR / "sift_codebook.faiss"
    N_CLUSTERS = 256
    LIMITE_MUESTRAS_SIFT = 400

    # Buscar imágenes
    rutas_imagenes = sorted([
        p for p in glob.glob(str(IMAGES_DIR / "**/*"), recursive=True)
        if os.path.isfile(p) and p.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if not rutas_imagenes:
        raise SystemExit("No se encontraron imágenes en ./images")

    rutas_relativas = [os.path.relpath(p, IMAGES_DIR) for p in rutas_imagenes]
    pd.DataFrame({
        "id": np.arange(len(rutas_relativas)),
        "image": rutas_relativas
    }).to_csv(DB_CSV, index=False)
    print(f"[OK] CSV creado: {DB_CSV} (N={len(rutas_relativas)})")

    # Entrenar codebook SIFT si no existe
    if not CODEBOOK_PATH.exists():
        entrenar_codebook_sift(
            rutas_imagenes,
            CODEBOOK_PATH,
            n_clusters=N_CLUSTERS,
            limite_muestras=LIMITE_MUESTRAS_SIFT
        )

    # Extraer descriptores
    descriptores_hist = []
    descriptores_haralick = []
    descriptores_sift = []
    descriptores_vgg19 = []
    descriptores_mobilenet = []

    for ruta in rutas_imagenes:
        descriptores_hist.append(extraer_histograma(ruta))
        descriptores_haralick.append(extraer_haralick(ruta))
        descriptores_sift.append(extraer_sift_bow(ruta, str(CODEBOOK_PATH), n_clusters=N_CLUSTERS))
        descriptores_vgg19.append(extraer_vgg19(ruta, dispositivo='cpu'))
        descriptores_mobilenet.append(extraer_mobilenetv2(ruta, dispositivo='cpu'))

    # Guardar índices FAISS
    guardar_indice_faiss(np.vstack(descriptores_hist),       DB_DIR / "caracteristicas_hist.index")
    guardar_indice_faiss(np.vstack(descriptores_haralick),   DB_DIR / "caracteristicas_haralick.index")
    guardar_indice_faiss(np.vstack(descriptores_sift),       DB_DIR / "caracteristicas_sift.index")
    guardar_indice_faiss(np.vstack(descriptores_vgg19),      DB_DIR / "caracteristicas_vgg19.index")
    guardar_indice_faiss(np.vstack(descriptores_mobilenet),  DB_DIR / "caracteristicas_mobilenetv2.index")

    print("\n[FIN] Base de datos creada exitosamente en ./database")
    

if __name__ == "__main__":
    crear_indices_csv()