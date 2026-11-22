# N-Framework: The Exact Thermodynamics of Consensus

**N-Framework** (`n-metrics` package v0.8.5) is an analytical and computational ecosystem designed to overcome the structural limitations of classical inter-rater agreement metrics (such as Krippendorff's Alpha, Fleiss' Kappa, Intraclass Correlation Coefficient ICC(2,1), or Kendall's W). Developed by **Manuel Narbona Sarria**, the framework redefines rater consensus measurement by replacing traditional variance-based analysis with an exact formulation grounded in **statistical physics and the thermodynamics of geometric consensus**.

---

## 1. Theoretical Foundations and Paradigm Shift

### The Paradox of Classical Metrics
Classical agreement metrics evaluate reliability based on between-subject variance. This premise introduces systematic flaws in real-world research scenarios:
1. **Zero-Variance Paradox (Scale Ceiling):** When there is massive or perfect agreement among raters in a single category (sample homogeneity), variance drops to zero. Consequently, metrics such as ICC or Krippendorff collapse computationally or yield artificially low or undefined indices.
2. **Spatial Blindness and Sample Size Sensitivity:** Traditional estimators vary depending on the number of subjects ($n$), treating agreement as an extensive property dragged down by degrees of freedom.
3. **Population Bias in Small Samples:** In highly specialized audits ($n < 10, m \in [2, 3]$), classical bootstrap methods suffer from severe optimistic or pessimistic biases.

### The Thermodynamic Solution of N-Framework
N-Framework isolates **consensus signatures (macrostates)** from the rating matrix $X \in \mathbb{R}^{n \times m}$, decomposing the structure in combinatorial phase space:
* **Consensus Signature (Macrostate):** Extracts the pure topological configuration of row responses, discarding nominal subject noise and labeling order.
* **Theoretical Multiplicity ($\Omega$):** Computes via exact combinatorics the volume of physical microstates composing each pattern in the sample space.
* **Asymptotic Population Value ($n \to \infty$):** Weights empirical sample frequencies with their theoretical volume in the probability universe, accurately reconstructing true consensus free of statistical noise.
* **Intensive Property:** Consensus under N-Framework behaves as an intensive variable (analogous to temperature), remaining strictly **invariant under row cloning or scaling** ($n \to c \cdot n$).

---

## 2. System Architecture and Engine Modules

The framework core is organized into a clean modular architecture optimized for big data processing, parallel state-cache computation, and an interactive Streamlit web interface (`app.py`).

```
n-metrics/
├── app.py                      # Interactive Streamlit app (9 Tabs, i18n, GUI)
├── data_handler.py             # Streaming ingestion, matrix auto-cleaner, and validation
├── calcular_diccionarios.py    # Parallel SQLite cache generator for macrostates (m, k)
├── pyproject.toml              # Package config, dependencies, and 18 CLI commands
├── CITATION.cff                # Academic citation metadata
├── LICENSE                     # GNU General Public License v3.0
├── nmetrics/
│   ├── interval/
│   │   ├── ni_core.py          # Interval N Coefficient (NI) & anomaly scanner
│   │   ├── aki_core.py         # Classical baseline: Adjusted Krippendorff Interval
│   │   └── icc21_core.py       # Classical baseline: ICC(2,1) (F-ANOVA)
│   ├── nominal/
│   │   ├── nn_core.py          # Nominal N Coefficient (NN) & bounded partitions
│   │   ├── akn_core.py         # Classical baseline: Adjusted Krippendorff Nominal
│   │   └── kf_core.py          # Classical baseline: Fleiss' Kappa
│   └── ordinal/
│       ├── no_core.py          # Ordinal N Coefficient (NO) & entropic mass
│       ├── ako_core.py         # Classical baseline: Adjusted Krippendorff Ordinal
│       └── w_core.py           # Classical baseline: Kendall's W
```

### Key Engine Components

* **`app.py` (Streamlit Orchestrator & i18n):** Integrates a bilingual dynamic translation engine (Spanish/English) via `obtener_idioma_usuario()` and `t(key)`, persistent `st.session_state` management, sidebar control panel, and rendering of 9 interactive modules.
* **`data_handler.py` (Big Data Reader & Heuristic Cleaner):** Processes CSV/TXT files in chunks (*chunksize = 100,000*) preventing RAM overflow. Its intelligent auto-cleaner detects text headers and subject names, trimming the matrix down to pure numeric data and compressing identical rows into frequency tuples using `collections.Counter`.
* **`calcular_diccionarios.py` (SQLite Cache Generator):** Precomputes macrostate dictionaries for rater ($m$) and category ($k$) combinations, storing them in the `n_metrics_cache.db` database (`macroestados_cache` table). It implements parallel execution via `concurrent.futures.ProcessPoolExecutor` for $m < 60$ and safe sequential processing with explicit memory release (`gc.collect()`) for $m \ge 60$.

---

## 3. The Three Consensus Topologies

N-Framework adapts its mathematical engine to the specific variable type and metric distance used in the evaluation process.

| Topology | N-Coefficient (N-Framework) | Control Classical Estimators | Data Type & Rules | Scale Nature |
| :--- | :--- | :--- | :--- | :--- |
| **Interval** | **`NI`** | `AKI` (Krippendorff Interval), `ICC(2,1)` (F-ANOVA) | Continuous or discrete numerical scales ($k_{\text{min}}$ to $k_{\text{max}}$). | Continuous numerical distances where differences between values are constant. |
| **Nominal** | **`NN`** | `AKN` (Krippendorff Nominal), `KF` (Fleiss Kappa) | Pure categorical. Maps text labels to integers $\{1, \dots, k\}$. Requires integer values. | No order or implicit distance between categories. |
| **Ordinal** | **`NO`** | `AKO` (Krippendorff Ordinal), `W` (Kendall's W) | Hierarchical categorical. Requires discrete integer values. | Strict hierarchy between levels with rank distances. |

---

## 4. Statistical Formulations and Mechanics

### 1. Macrostate Extraction and Multiplicity $\Omega$
Given an evaluation matrix with $n$ subjects, $m$ raters, and $k$ scale levels, each subject generates a category assignment frequency distribution $\vec{f} = (f_1, f_2, \dots, f_k)$ such that $\sum_{j=1}^k f_j = m$.

The physical multiplicity (number of microstates generating the same macrostate) is computed using the multinomial formula:

$$\Omega(\vec{f}) = \frac{m!}{f_1! f_2! \dots f_k!}$$

### 2. Thermodynamic Randomness ($P_e$) and Universal Percentile
The engine computes the geometric expectation of chance $P_e$ for the configuration $(n, m, k)$. The **Universal Percentile (%)** evaluates the ordinal position of empirical consensus within the complete combinatorial distribution of possible universes:

$$P_e = \text{Geometric Expectation under Pure Chance}(m, k)$$

A Universal Percentile of 95% indicates that the empirical matrix possesses greater geometric order than 95% of all possible stochastic configurations for the $(n, m, k)$ design.

### 3. Anomaly Scanner and Crystal Floor
The anomaly scanner compares the observed empirical frequency of each macrostate ($P_{\text{emp}}$) against its theoretical thermodynamic probability under pure chance ($P_{\text{azar}}$):

$$Z = \frac{\log_{10}(P_{\text{emp}} / P_{\text{azar}}) - \mu_{Z}}{\sigma_{Z}}$$

* **Over-representation ($P_{\text{emp}} / P_{\text{azar}} > 1$):** Consensus patterns observed more frequently than expected by chance.
* **Under-representation and Crystal Floor ($P_{\text{emp}} / P_{\text{azar}} < 1$):** Detects subjects or responses falling into "crystal floor" zones (thermodynamic black holes), where high entropy or lack of degrees of freedom prevents standard physical projection.

---

## 5. Web Application Modules (`app.py`)

The user interface implemented in `app.py` contains 9 specialized tabs:

1. **Tab 1: Consensus Inference & Anomalies:** Empirical matrix loading, parallel N-Coefficient calculation (`NI`, `NN`, `NO`) alongside classical metrics, weighted Confidence Interval simulation, $P_e$ calculation, universal percentiles, and $Z$-score anomaly scanner.
2. **Tab 2: Coverage Tests:** Stress testing audit to verify the effective rate at which 95% Confidence Intervals capture Population Truth across simulated parallel universes.
3. **Tab 3: Multirange Report:** Interactive explorer analyzing consensus behavior across variations in scale categories $k$ and number of raters $m$.
4. **Tab 4: Invariance & Sensitivity:** Experimental demonstration of the N-Framework intensive property by cloning rows ($n$), showcasing N-Coefficient stability against classical metric drift.
5. **Tab 5: Synthetic Matrix Generator:** Simulation tool to build synthetic matrices by specifying global or subject-level probability patterns and injecting entropic exceptions ("black holes") for diagnostic testing.
6. **Tab 6: Precision Duel (N vs. Classicals):** Benchmarking suite pitting N-Framework against traditional metrics across three edge-case scenarios:
   * *Zero-Variance Paradox:* Evaluation under massive agreement ($target\_N > 0.9$).
   * *Small Samples (Experts):* Audits with $n < 10$ and $m \in [2, 3]$.
   * *Noise & Maximum Entropy:* Chaotic dispersion.
7. **Tab 7: Dictionary & Signature Database:** Explorer and auditor for the SQLite database (`n_metrics_cache.db`), allowing precomputed signature inspection and JSON macrostate dictionary export.
8. **Tab 8: User Manual & Theoretical Foundations:** Comprehensive interactive guide featuring mathematical foundations, glossary, and bilingual documentation.
9. **Tab 9: Authorship & License:** Development credits (Manuel Narbona Sarria), version info (0.8.5), citation format (`CITATION.cff`), and GNU GPL v3.0 license terms.

---

## 6. Installation, Requirements, and CLI

### System Requirements
* **Python:** `>= 3.8`
* **Core Dependencies:** `numpy >= 1.20.0`, `pandas >= 1.3.0`, `scipy >= 1.7.0`, `streamlit`, `plotly`, `Pillow`.

### Installation
```bash
# Install from PyPI
pip install n-metrics

# Or local installation from cloned repository
pip install -e .
```

### Command Line Interface (CLI) Commands

The `pyproject.toml` configuration registers 18 terminal-executable commands:

```bash
# --- Global Comparative Tables ---
nmetrics-interval   # Comparative analysis for interval topology
nmetrics-nominal    # Comparative analysis for nominal topology
nmetrics-ordinal    # Comparative analysis for ordinal topology

# --- N-Framework Individual Metrics ---
nmetrics-ni         # Individual NI analysis (Interval)
nmetrics-nn         # Individual NN analysis (Nominal)
nmetrics-no         # Individual NO analysis (Ordinal)

# --- Classical Individual Baseline Metrics ---
nmetrics-aki        # AKI analysis (Krippendorff Interval)
nmetrics-icc21      # ICC(2,1) analysis
nmetrics-akn        # AKN analysis (Krippendorff Nominal)
nmetrics-kf         # Fleiss' Kappa analysis
nmetrics-ako        # AKO analysis (Krippendorff Ordinal)
nmetrics-w          # Kendall's W analysis

# --- Coverage and Stress Tests ---
nmetrics-ni-cob     # NI coverage test
nmetrics-nn-cob     # Coverage NN
nmetrics-no-cob     # Coverage NO

# --- Comparative Resampling Modules ---
nmetrics-interval-comp
nmetrics-nominal-comp
nmetrics-ordinal-comp
```

---

## 7. Citation and License

### Academic Citation
If you use N-Framework (`n-metrics`) in your academic research or software projects, please cite it using the metadata from `CITATION.cff`:

Narbona-Sarria, M. (2026). The N Framework: Consensus as a State in Configuration Space. Zenodo. https://doi.org/10.5281/zenodo.24426504

### License
This project is licensed under the **GNU General Public License v3.0 (GPLv3)**. For complete terms, see `LICENSE.txt`.
