## 🏛️ Arquitectura del Ecosistema N-Metrics

La plataforma funciona mediante la integración continua de tres servicios independientes. Entender la relación entre ellos es clave para mantener la herramienta segura y actualizada:

- **🐍 PyPI (El Motor):** Es el repositorio oficial de Python donde vive tu paquete `n-metrics`. Aquí residen los *cores* matemáticos y la lógica termodinámica. Es **inmutable**: una versión publicada no se puede modificar, solo se pueden publicar versiones nuevas.

- **🐙 GitHub (La Carrocería):** Es tu disco duro en la nube. Almacena únicamente el diseño de la página web (`app.py`) y la lista de dependencias (`requirements.txt`).

- **🎈 Streamlit Cloud (El Concesionario):** Es el servidor web. No almacena código propio. Su trabajo es vigilar tu GitHub; cuando detecta un cambio, lee `app.py`, descarga el motor matemático desde PyPI (leyendo el `requirements.txt`) y muestra la aplicación al mundo.

---

## 🚀 1. Comandos de Inicio y Pruebas en Local

Antes de subir cualquier cambio a internet, es recomendable probar la aplicación en tu propio ordenador.

**Preparación del entorno (Solo la primera vez):**

Bash

```
# 1. Crear un entorno virtual aislado (para no mezclar librerías)
python -m venv venv

# 2. Activar el entorno
# En Windows:
venv\Scripts\activate
# En Mac/Linux:
source venv/bin/activate

# 3. Instalar las dependencias necesarias
pip install -r requirements.txt
```

**Arrancar la aplicación web en tu ordenador:**

Bash

```
streamlit run app.py
```

> Esto abrirá automáticamente una pestaña en tu navegador (normalmente en `http://localhost:8501`) donde podrás probar tu interfaz sin que afecte a la versión pública.

---

## 🔄 2. Flujo de Actualización: Modificar la Interfaz (Frontend)

Si solo quieres cambiar textos, colores, botones o la forma en que se muestran las tablas en la web, el motor de PyPI no necesita enterarse.

**Pasos para actualizar:**

1. Modifica el código en tu archivo `app.py`.

2. Sube los cambios a GitHub con los siguientes comandos en tu terminal:

Bash

```
git add app.py
git commit -m "Mejora en la visualización de las tablas de inferencia"
git push origin main
```

**Resultado:** Streamlit detectará este *push* en GitHub y, en cuestión de segundos, actualizará la página web pública automáticamente. No tienes que hacer nada más.

---

## ⚙️ 3. Flujo de Actualización: Modificar el Motor Matemático (Backend)

Si has descubierto un nuevo teorema, optimizado un algoritmo, o arreglado un error en las funciones termodinámicas (`ni_core.py`, etc.), el proceso requiere actualizar tanto PyPI como GitHub para obligar a Streamlit a descargar el nuevo motor.

**Paso 1: Publicar la nueva versión en PyPI**

1. Modifica tu código fuente local.

2. Sube el número de versión en tu archivo de configuración (por ejemplo, de `1.0.0` a `1.0.1`).

3. Compila y sube el paquete a PyPI usando tu API Token seguro:

Bash

```
# Compilar los archivos de distribución
python -m build

# Subir a PyPI usando Twine
twine upload dist/*
```

**Paso 2: Forzar a Streamlit a actualizar** Como Streamlit no vigila PyPI, debes avisarle desde GitHub.

1. Abre tu archivo `requirements.txt`.

2. Actualiza la línea de tu librería para apuntar a la nueva versión exacta:

Plaintext

```
streamlit
pandas
numpy
n-metrics==1.0.1
```

3. Sube este cambio a GitHub:

Bash

```
git add requirements.txt
git commit -m "Actualización del motor termodinámico a v1.0.1"
git push origin main
```

**Resultado:** Streamlit verá que el archivo de requerimientos ha cambiado. Destruirá el servidor antiguo, construirá uno nuevo, descargará tu versión `1.0.1` desde PyPI y lanzará la web con las nuevas matemáticas aplicadas.

---

## 🛡️ 4. Reglas de Seguridad y Mantenimiento

- **El archivo `.gitignore`:** Asegúrate de tener este archivo en tu GitHub. Debe incluir carpetas de compilación (`build/`, `dist/`, `*.egg-info/`) y carpetas locales (`__pycache__/`, `venv/`) para no saturar tu repositorio con "basura" técnica.

- **Nunca subas secretos:** Las contraseñas, tokens de PyPI o claves API jamás deben escribirse en `app.py` ni en ningún archivo subido a GitHub.

- **Sincronía de versiones:** Acostúmbrate a anclar siempre la versión en `requirements.txt` (ej. `n-metrics==1.0.1`). Si solo escribes `n-metrics`, Streamlit podría instalar una versión antigua almacenada en su memoria caché al reiniciarse.
