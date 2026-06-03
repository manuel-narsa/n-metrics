nmetrics-env\Scripts\activate

Absolutely — here’s a polished, professional **README.md in English**, already customized with your GitHub username **manuel-narsa** and aligned with the N‑Metrics ecosystem you’re building.

You can paste this directly into your repository.

---

# 📘 **README.md — N‑Metrics (English Version)**

```markdown
# N‑Metrics

**N‑Metrics** is a modern statistical ecosystem for analyzing agreement, concordance, and consistency between raters.  
It provides Natural Metrics, simulation tools, specialized visualizations, and benchmarking against traditional reliability coefficients.

The goal of N‑Metrics is to offer a **conceptually transparent**, **universally comparable**, and **robust** framework for reliability analysis, overcoming the limitations of classical models such as Kappa, ICC, or Krippendorff’s Alpha.

---

## ✨ Key Features

### 🔹 Natural Metrics
- Natural Concordance Coefficient (NCC)  
- Natural agreement, consistency, and proximity metrics  
- Natural distances for nominal, ordinal, interval, and continuous scales  

### 🔹 Simulation Engine
- Rater generation with adjustable noise levels  
- Systematic bias simulation  
- Extreme-case scenarios and arbitrary distributions  
- Reproducible synthetic datasets  

### 🔹 Specialized Visualization
- Concordance maps  
- Disagreement heatmaps  
- Sensitivity curves  
- Rater profiles  
- Comparative metric plots  

### 🔹 Benchmarking Tools
Compare Natural Metrics with:
- ICC (all types)  
- Cohen’s and Fleiss’ Kappa  
- Weighted Kappa  
- Krippendorff’s Alpha  
- Kendall’s W  
- Lin’s CCC  

### 🔹 Scientific Ecosystem Integration
Fully compatible with:
- **NumPy**
- **Pandas**
- **SciPy**
- **scikit‑learn**
- **Matplotlib / Seaborn**

---

## 📦 Installation

```bash
pip install n-metrics
```

*(The package will be available on PyPI once version 0.1.0 is published.)*

---

## 🚀 Basic Usage

```python
import nmetrics as nm
import pandas as pd

# Load example data
df = pd.read_csv("ratings.csv")

# Compute the Natural Concordance Coefficient
score = nm.core.NCC(df)

print("NCC:", score)
```

---

## 📁 Package Structure

```
nmetrics/
│
├── core/           # Core Natural Metrics
├── metrics/        # Variants and extensions
├── simulation/     # Simulation engine
├── visualization/  # Specialized plots
├── analysis/       # Analysis tools
├── datasets/       # Example datasets
├── cli/            # Command-line interface
├── utils/          # Internal utilities
└── config/         # Configuration
```

---

## 📚 Documentation

Full documentation will be available soon at:

👉 `https://github.com/manuel-narsa/n-metrics/wiki` [(github.com in Bing)](https://www.bing.com/search?q=%22https%3A%2F%2Fgithub.com%2Fmanuel-narsa%2Fn-metrics%2Fwiki%22)

---

## 🧪 Running Tests

```bash
pytest tests/
```

---

## 🤝 Contributing

Contributions are welcome.  
You can open issues, submit pull requests, or propose new Natural Metrics.

---

## 📄 License

This project is licensed under the **MIT License**.  
See the `LICENSE` file for details.

---

## 🧠 Suggested Citation

If you use N‑Metrics in a scientific publication:

```
Manuel Narsa. (2026). N‑Metrics: A Natural Metrics Ecosystem for Agreement and Concordance Analysis.
https://github.com/manuel-narsa/n-metrics
```

---

## 🌟 Project Vision

N‑Metrics aims to become a **new methodological standard** for reliability analysis by providing:

- universal comparability
- conceptual coherence
- robustness to bias
- intuitive interpretation

A tool designed for researchers, statisticians, clinicians, psychometricians, and data scientists.

```
---

If you want, I can also prepare:

- a **CONTRIBUTING.md**,  
- a **CHANGELOG.md**,  
- a **GitHub Actions workflow** to auto‑publish to PyPI,  
- or the **first functional Python modules** for `nmetrics/core`.

Just tell me where you want to go next.Configurar **TOTP** para PyPI y TestPyPI es muy sencillo, pero hay que seguir los pasos exactos porque ambos sitios funcionan igual. Te lo dejo claro, ordenado y sin ruido para que lo actives en un minuto.
```

---

# 🔐 1. Entra en tu cuenta de PyPI o TestPyPI

Según dónde quieras activarlo:

- **PyPI**: https://pypi.org/account/login
- **TestPyPI**: https://test.pypi.org/account/login

Después ve a:

👉 **Account settings → Two-factor authentication**

---

# 🟦 2. Selecciona “Add TOTP”

Verás dos opciones:

- **Authenticator app (TOTP)**
- **Security key (FIDO2/WebAuthn)**

Elige:

👉 **Add TOTP**

---

# 🟩 3. Escanea el código QR con tu app autenticadora

Puedes usar:

- Google Authenticator
- Microsoft Authenticator
- Authy
- 1Password
- Bitwarden
- FreeOTP

Tu app generará códigos de 6 dígitos que cambian cada 30 segundos.

---

# 🟧 4. Introduce el código de 6 dígitos en PyPI/TestPyPI

PyPI te pedirá:

- un **código actual**
- confirmar la activación

Cuando lo introduzcas correctamente, TOTP quedará activado.

---

# 🟨 5. Guarda los códigos de recuperación

PyPI/TestPyPI te mostrará varios **recovery codes**.

Son IMPORTANTES porque:

- si pierdes tu móvil
- si borras la app
- si cambias de dispositivo

los necesitarás para entrar.

Guárdalos en un sitio seguro.

---

# 🟪 6. A partir de ahora, cada vez que inicies sesión

PyPI/TestPyPI te pedirá:

1. tu usuario
2. tu contraseña
3. un **código TOTP** de tu app

Esto protege tu cuenta contra accesos no autorizados.

---

# ⭐ ¿Y para subir paquetes con Twine?

TOTP **no se usa** para subir paquetes.  
Para eso necesitas un **API token**, que ya estás usando.

TOTP solo protege **el acceso a tu cuenta web**.

---

# Si quieres, puedo ayudarte a:

- activar TOTP paso a paso en tu cuenta real,
- configurar un segundo dispositivo de respaldo,
- entender cómo funcionan los tokens de PyPI,
- o automatizar tus publicaciones con GitHub Actions.

Tú decides hacia dónde seguimos.

Aquí tienes **exactamente los comandos que debes ejecutar**, sin adornos y en el orden correcto, para subir tu paquete a **PyPI** (no TestPyPI).  
Están listos para copiar y pegar.

---

# 🟦 1. Construir los paquetes

Desde la raíz del proyecto:

```bash
python -m build
```

Esto crea los archivos dentro de `dist/`.

---

# 🟩 2. Subirlos a PyPI usando tu API token

Ejecuta:

```bash
twine upload dist/*
```

Twine detectará que PyPI usa tokens y te mostrará:

```
Enter your API token:
```

Ahí pegas tu token completo, por ejemplo:

```
pypi-AgEIcHlwaS5vcmcC...
```

Y pulsas **Enter**.

---

# 🟧 Resultado esperado

Si todo va bien verás algo como:

```
Uploading n_metrics-0.1.0-py3-none-any.whl
Uploading n_metrics-0.1.0.tar.gz
View at:
https://pypi.org/project/n-metrics/
```

En ese momento el nombre **queda reservado** y tu paquete está publicado.

---

Si quieres, puedo revisar tu carpeta `dist/` o tu `pyproject.toml` para asegurar que todo está perfecto antes de subirlo.Perfecto, vamos a construir **los tres pilares** que necesitas para que N‑Metrics tenga un flujo profesional de publicación:

1. **Workflow completo de GitHub Actions para publicar automáticamente en PyPI**
2. **Script PowerShell para automatizar build + upload manual**
3. **Configuración de versiones recomendada para un proyecto serio**

Todo está pensado para que funcione desde ya, sin ajustes extra.

---

# 🟦 1. **Workflow completo de GitHub Actions para publicar automáticamente en PyPI**

Este workflow:

- se ejecuta cuando haces un **tag** (por ejemplo `v0.1.0`)
- construye el paquete
- lo sube a PyPI usando un **API token** guardado en GitHub Secrets
- funciona tanto para PyPI como para TestPyPI (te doy ambas versiones)

## 📄 **Archivo: `.github/workflows/publish.yml`**

```yaml
name: Publish to PyPI

on:
  push:
    tags:
      - "v*"

jobs:
  build-and-publish:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install build tools
        run: |
          python -m pip install --upgrade pip
          pip install build twine

      - name: Build package
        run: python -m build

      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*
```

### 🔐 **Cómo configurarlo**

En tu repositorio GitHub:

1. Ve a **Settings → Secrets → Actions**
2. Crea un secreto llamado:

```
PYPI_API_TOKEN
```

3. Pega tu token de PyPI (empieza por `pypi-...`)

### 🚀 **Cómo publicar una versión**

Solo tienes que crear un tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

Y GitHub Actions hará el resto.

---

# 🟩 2. **Script PowerShell para automatizar build + upload**

Perfecto para publicar manualmente desde tu máquina.

## 📄 **Archivo: `publish.ps1`**

```powershell
# Clean old builds
if (Test-Path "dist") {
    Remove-Item -Recurse -Force dist
}

# Build package
python -m build

# Upload to PyPI
twine upload dist/*
```

### ▶️ **Cómo usarlo**

```powershell
.\publish.ps1
```

Cuando Twine pida:

```
Enter your API token:
```

Pegas tu token de PyPI.

---

# 🟧 3. **Configuración de versiones recomendada**

Para un proyecto científico serio como N‑Metrics, te recomiendo:

## 📌 **Versionado semántico (SemVer)**

```
MAJOR.MINOR.PATCH
```

### ✔️ **PATCH**

Correcciones pequeñas, sin cambios en la API  
Ej: `0.1.1 → 0.1.2`

### ✔️ **MINOR**

Nuevas funciones compatibles  
Ej: `0.1.0 → 0.2.0`

### ✔️ **MAJOR**

Cambios que rompen compatibilidad  
Ej: `0.9.0 → 1.0.0`

---

# ⭐ Flujo recomendado para N‑Metrics

1. Empieza con:

```
0.1.0
```

2. Cada mejora pequeña:

```
0.1.1, 0.1.2, 0.1.3…
```

3. Cuando tengas el **NCC** implementado:

```
0.2.0
```

4. Cuando el ecosistema esté estable:

```
1.0.0
```

---

# ¿Quieres que prepare también…?

- un **CHANGELOG.md** profesional
- un **CONTRIBUTING.md**
- un **pyproject.toml** con versionado automático
- un workflow para publicar **en TestPyPI y PyPI según el tag**
- un sistema de **bumping automático de versión** con GitHub Actions

Solo dime qué quieres y lo montamos.

Puedes hacerlo si quieres, pero **no es obligatorio instalar GitHub en tu PC** para usar GitHub Actions ni para publicar en PyPI.  
Aun así, si quieres trabajar de forma cómoda desde tu máquina, lo ideal es tener:

- **Git** instalado
- Tu PC **conectado a tu cuenta de GitHub**
- (Opcional) **GitHub Desktop** si prefieres interfaz gráfica

Te dejo el camino claro y sin complicaciones.

---

# 🟦 1. Instalar Git (imprescindible)

Descárgalo aquí:

👉 https://git-scm.com/downloads

Durante la instalación, acepta las opciones por defecto.

Comprueba que funciona:

```bash
git --version
```

---

# 🟩 2. Conectar Git con tu cuenta de GitHub

Hay dos formas: **HTTPS + token** (más fácil) o **SSH** (más profesional).  
Te doy la forma sencilla primero.

---

## ✔️ Opción A: Conexión por HTTPS (recomendada para empezar)

1. En GitHub, crea un **Personal Access Token** (PAT):
   
   - Settings → Developer settings → Personal access tokens → Tokens (classic)
   - Generate new token
   - Marca:
     - `repo`
     - `workflow`
   - Cópialo (solo aparece una vez)

2. En tu PC, configura tu usuario:

```bash
git config --global user.name "manuel-narsa"
git config --global user.email "tu-email@loquesea.com"
```

3. La primera vez que hagas `git push`, Git te pedirá:
- **username:** tu usuario de GitHub
- **password:** tu token (no tu contraseña)

Y quedará guardado.

---

## ✔️ Opción B: Conexión por SSH (más profesional)

1. Genera una clave SSH:

```bash
ssh-keygen -t ed25519 -C "tu-email@loquesea.com"
```

Pulsa Enter a todo.

2. Muestra tu clave pública:

```bash
cat ~/.ssh/id_ed25519.pub
```

3. Copia el contenido y añádelo en GitHub:
   
   Settings → SSH and GPG keys → New SSH key

4. Prueba la conexión:

```bash
ssh -T git@github.com
```

Si ves un mensaje de bienvenida, funciona.

---

# 🟧 3. Clonar tu repositorio en tu PC

```bash
git clone https://github.com/manuel-narsa/n-metrics.git
```

o si usas SSH:

```bash
git clone git@github.com:manuel-narsa/n-metrics.git
```

---

# 🟨 4. Trabajar normalmente

Cada vez que hagas cambios:

```bash
git add .
git commit -m "mensaje"
git push
```

Y GitHub Actions se encargará de publicar en PyPI cuando hagas un **tag**.

---

# ⭐ Resumen rápido

| Acción                  | ¿Es obligatorio? | ¿Para qué sirve?                  |
| ----------------------- | ---------------- | --------------------------------- |
| Instalar Git            | ✔️ Sí            | Trabajar con repositorios         |
| Conectar Git con GitHub | ✔️ Sí            | Hacer push/pull                   |
| Instalar GitHub Desktop | ❌ No             | Solo si quieres interfaz gráfica  |
| Instalar GitHub CLI     | ❌ No             | Opcional, para usuarios avanzados |

---

Si quieres, puedo ayudarte a:

- generar tu **token de GitHub**,
- configurar **SSH paso a paso**,
- verificar que tu repositorio está bien clonado,
- o preparar un **script PowerShell** para automatizar commits y pushes.
