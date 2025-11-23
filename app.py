import os
import pathlib
import time
import numpy as np
import pandas as pd
import faiss
from PIL import Image
import streamlit as st
from streamlit_cropper import st_cropper
import extractores as ext

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
st.set_page_config(layout="wide")

ROOT_PATH = pathlib.Path(__file__).resolve().parent
IMAGES_PATH = ROOT_PATH / "images"
DB_PATH = ROOT_PATH / "database"
DB_FILE = DB_PATH / "db.csv"

INDEX_MAP = {
    "Histograma":      "caracteristicas_hist.index",
    "SIFT (BoW)":      "caracteristicas_sift.index",
    "Haralick":        "caracteristicas_haralick.index",
    "VGG19":           "caracteristicas_vgg19.index",
    "MobileNetV2":     "caracteristicas_mobilenetv2.index",
}

def comprobar_y_crear_base():
    DB_PATH.mkdir(exist_ok=True, parents=True)
    indices_faltan = False

    if not DB_FILE.exists():
        indices_faltan = True

    for nombre in INDEX_MAP.values():
        if not (DB_PATH / nombre).exists():
            indices_faltan = True

    if indices_faltan:
        st.warning("No se encontraron los índices o el CSV. Generando base de datos...")
        ext.crear_indices_csv()
        st.success("Base de datos creada correctamente.")
    else:
        print("Base de datos existente.")

@st.cache_data
def cargar_imagenes():
    df = pd.read_csv(DB_FILE)
    return list(df.image.values)

@st.cache_resource
def cargar_indices():
    indices = {}
    for k, fname in INDEX_MAP.items():
        indices[k] = faiss.read_index(str(DB_PATH / fname))
    return indices

def procesar_imagen_query(pil_img, extractor):
    if extractor == "Histograma":
        return ext.extraer_histograma(pil_img)
    elif extractor == "Haralick":
        return ext.extraer_haralick(pil_img)
    elif extractor == "SIFT (BoW)":
        codebook = DB_PATH / "sift_codebook.faiss"
        return ext.extraer_sift_bow(pil_img, str(codebook), n_clusters=256)
    elif extractor == "VGG19":
        return ext.extraer_vgg19(pil_img, dispositivo="cpu")
    elif extractor == "MobileNetV2":
        return ext.extraer_mobilenetv2(pil_img, dispositivo="cpu")
    else:
        raise ValueError("Extractor no válido.")

def recuperar_similares(pil_img, extractor, n_imgs=11):
    indices_faiss = cargar_indices()
    index = indices_faiss[extractor]

    vec = procesar_imagen_query(pil_img, extractor).astype("float32")
    faiss.normalize_L2(vec)
    _, I = index.search(vec, k=n_imgs)
    return I[0]

def main():
    st.markdown(
    """
    <style>
    .stApp {
        background-color: #d3d3d3;  /* Gris medio */
    }
    </style>
    """,
    unsafe_allow_html=True
)

    col1, col2 = st.columns([5, 1])
    with col1:
        st.title("CBIR IMAGE SEARCH")
        st.markdown("**Carga una imagen y selecciona un extractor.**")
    with col2:
        st.image("logo_uni.png", width=160)
 

    comprobar_y_crear_base()
    
    lista_imagenes = cargar_imagenes()

    col1, col2 = st.columns(2)

    with col1:
        st.header("QUERY")
        extractor = st.selectbox("Selecciona extractor:", list(INDEX_MAP.keys()))
        img_file = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])

        if img_file:
            img = Image.open(img_file).convert("RGB")
            cropped = st_cropper(img, realtime_update=True, box_color="#FF0004")
            st.write("Vista previa")
            _ = cropped.thumbnail((200, 200))
            st.image(cropped)

    with col2:
        st.header("RESULT")
        if img_file:
            t0 = time.time()
            resultados = recuperar_similares(cropped, extractor, n_imgs=11)
            t1 = time.time()

            col3, col4 = st.columns(2)
            with col3:
                for i in range(1, 11, 2): 
                    img = Image.open(IMAGES_PATH / lista_imagenes[resultados[i]])
                    st.image(img, caption=f"Resultado {i+1}", use_column_width=True)

            with col4:
                for i in range(2, 11, 2): 
                    img = Image.open(IMAGES_PATH / lista_imagenes[resultados[i]])
                    st.image(img, caption=f"Resultado {i+1}", use_column_width=True)


            col5, col6, col7 = st.columns(3)
            for col, start in zip([col5, col6, col7], [2, 3, 4]):
                with col:
                    for u in range(start, 11, 3):
                        idx = resultados[u]
                        ruta = IMAGES_PATH / lista_imagenes[idx]
                        if ruta.exists():
                            st.image(Image.open(ruta), use_column_width=True)

if __name__ == "__main__":
    main()
