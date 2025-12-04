<p><img height="80px" src="https://www.upm.es/sfs/Rectorado/Gabinete%20del%20Rector/Logos/UPM/Escudo/EscUpm.jpg" align="left" hspace="0px" vspace="0px"></p>
Grado de Ciencia de Datos e Inteligencia Artificial

Asignatura: Algoritmos y Arquitecturas para el Procesado de Imágenes

# **CBIR-Grupo-A**

Este repositorio contiene una aplicación completa de **búsqueda de imágenes por contenido (CBIR)** basada en distintos extractores de características y FAISS para la indexación.

# Cómo usar este repositorio

## 1. Clonar el repositorio

```bash
git clone https://github.com/mariadelmar-sierra/CBIR-Grupo-A-AAPI.git
cd CBIR-Grupo-A-AAPI
```

## 2. Descargar las imágenes necesarias

Las carpetas `images/` y `Imgs test/` contienen un archivo `.txt` con un enlace de descarga.

Debes:

1. Abrir los archivos `.txt` de cada carpeta.  
2. Descargar las imágenes usando el enlace que aparece en el archivo.  
3. Almacenar todas las imágenes dentro de la **carpeta correspondiente**.  
4. **Eliminar el archivo `.txt`**.

Sin estas imágenes, la aplicación no funcionará, ya que todos los extractores trabajan sobre ellas.

## 3. Crear un entorno virtual (recomendado)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux \ Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

## 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

## 5. Ejecutar la aplicación

```bash
streamlit run app.py
```

### IMPORTANTE: primera ejecución

Durante la primera ejecución:

- La aplicación tardará un poco más.  
- Se generará automáticamente la carpeta `database/`.  
- Esta carpeta contendrá los **índices FAISS** y archivos de características que permiten realizar las búsquedas.

Este proceso solo ocurre la primera vez que se ejecuta la aplicación.

## 6. Evaluación de las métricas (opcional)

```bash
python evaluacion_metricas.py
```
Este script calcula las métricas comparando distintos extractores de características.

## Estructura del proyecto
📁 Proyecto  
├ 📂 images/                  → Conjunto de imágenes (descargar desde enlace.txt)  
├ 📂 Imgs test/               → Imágenes de test (descargar desde enlace.txt)  
├ 📂 database/                → Se genera automáticamente  
├ extractores.py              → Código con los extractores de características  
├ app.py                      → Aplicación Streamlit  
├ evaluacion_metricas.py      → Script de evaluación  
├ requirements.txt            → Dependencias  
└ README.md
