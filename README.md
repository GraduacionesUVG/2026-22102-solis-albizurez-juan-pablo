# Detección de patrones de ataque lentos y distribuidos utilizando redes neuronales recurrentes: Módulo Support Vector Machines

Trabajo de graduación, Universidad del Valle de Guatemala, Facultad de Ingeniería.
Licenciatura en Ingeniería en Ciencia de la Computación y Tecnologías de la Información.

## Autor

| Nombre | Carné |
|---|---|
| Juan Pablo Solís Albizurez | 22102 |

**Asesor:** Ing. Luis Alberto Suriano

## Descripción

Los ataques *low and slow* (Slowloris y SlowHTTPTest) agotan los recursos de un servidor web
manteniendo conexiones abiertas con muy poco tráfico. Por eso son difíciles de distinguir del
tráfico legítimo. Este proyecto diseña y evalúa un clasificador **SVM (Support Vector Machine)**
que detecta estos ataques a nivel de flujo de red, usando el conjunto de datos CIC-IDS2017.

Su desempeño se contrasta con el de los
módulos , XGBoost y GRU, sobre el mismo subconjunto de prueba.

| Módulo | Autor |
|---|---|
| SVM (este repositorio) | Juan Pablo Solís Albizurez |
| XGBoost | Edwin Ortega Kou |
| GRU | Esteban Zambrano Garoz |

El SVM **no** entrena ni reproduce los modelos XGBoost ni GRU. La comparación usa las métricas
que cada módulo reporta sobre el subconjunto común de prueba.

### Cómo funciona

1. **Limpieza:** filtra las clases BENIGN, Slowloris y SlowHTTPTest, corrige las marcas de tiempo
   y elimina los registros inválidos (443,614 flujos y 59 características).
2. **Partición cronológica por bloques:** 70 % entrenamiento, 15 % validación y 15 % prueba.
3. **Selección de características:** elimina las variables casi constantes, las de información
   mutua casi nula y las redundantes (Spearman > 0.98). Quedan 42 (39 originales y 3 derivadas).
4. **Entrenamiento:** usa un pipeline con transformación logarítmica con signo, estandarización y SVM.
   La búsqueda de hiperparámetros usa validación cruzada temporal y la regla de un error estándar.
5. **Umbral y evaluación:** el umbral se elige en validación (máximo F2 con recall ≥ 0.75).
   La partición de prueba se usa una sola vez, al final.

**Modelo final:** kernel RBF, C = 1, γ = 0.1, umbral −0.832.

### Resultados en prueba (66,544 flujos, 1,175 ataques)

| Recall | Precisión | F1 | ROC-AUC | AP | Falsos positivos | Falsos negativos |
|---|---|---|---|---|---|---|
| 0.9898 | 0.9765 | 0.9831 | 0.9984 | 0.9876 | 28 | 12 |

Comparación en el subconjunto común (66,454 flujos, 1,164 ataques). GRU se reporta como media ±
desviación estándar de cinco semillas:

| Métrica | SVM | XGBoost | GRU |
|---|---|---|---|
| Recall | 0.9897 | 0.9905 | 0.9727 ± 0.0106 |
| Precisión | 0.9763 | 0.9931 | 0.8967 ± 0.0219 |
| F1 | 0.9829 | 0.9918 | 0.9330 ± 0.0106 |
| Falsos positivos | 28 | 8 | 131.2 ± 31.8 |

## Tecnologías utilizadas

- Python 3
- pandas y NumPy: manejo de datos
- scikit-learn: SVM, pipeline, validación cruzada y métricas
- Matplotlib y Seaborn: visualización
- Jupyter Notebook / Google Colab
- Streamlit: demo en vivo
- XGBoost y TensorFlow/Keras: inferencia de los modelos hermanos en la demo
- pytest: pruebas unitarias
- Git y GitHub

## Requisitos previos

- **Python 3.10 o superior**
- **pip**
- **Git**
- **Conjunto de datos CIC-IDS2017.** No se incluye en el repositorio porque pesa varios GB.
  Se descarga de https://www.unb.ca/cic/datasets/ids-2017.html (captura del miércoles 5 de julio de 2017).

## Instalación

1. Clonar el repositorio y entrar a la carpeta del código:

   ```bash
   git clone https://github.com/GraduacionesUVG/2026-22102-solis-albizurez-juan-pablo.git
   cd 2026-22102-solis-albizurez-juan-pablo/src
   ```

2. Crear y activar un entorno virtual:

   ```bash
   python3 -m venv venv
   source venv/bin/activate        # En Windows: venv\Scripts\activate
   ```

3. Instalar las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

4. Copiar los CSV de CIC-IDS2017 a `src/data/raw/`.

5. (Opcional) Copiar `.env.example` a `.env`. El proyecto no necesita variables de entorno para
   funcionar; el archivo solo documenta esto.

## Ejecución

Desde la carpeta `src/`, abre Jupyter:

```bash
jupyter notebook
```

Ejecuta los cuadernos en este orden. Cada uno verifica que los archivos del anterior sean
consistentes antes de continuar:

1. `notebooks/Cleaning/limpieza_low_and_slow.ipynb`: filtrado de clases y limpieza.
2. `notebooks/Cleaning/Analisis_Exploratorio_actualizado.ipynb`: análisis exploratorio (opcional).
3. `notebooks/SVM/01_Analisis_Limpieza_SVM.ipynb`: partición cronológica y selección por multicolinealidad.
4. `notebooks/SVM/02_Feature_Engineering.ipynb`: construcción de características y matrices finales.
5. `notebooks/SVM/03_b_Train_Model.ipynb`: búsqueda de hiperparámetros, umbral y evaluación final.

Para correr las pruebas unitarias, desde `src/`:

```bash
python -m pytest -v
```

### Demo en vivo (Streamlit)

La carpeta `src/live_demo/` contiene una aplicación web que hace inferencia con los modelos ya
entrenados (SVM, XGBoost y GRU) sobre el subconjunto común de prueba. **No necesita descargar
CIC-IDS2017**, porque los modelos y los datos de prueba vienen incluidos.

```bash
cd src/live_demo
pip install -r requirements.txt
streamlit run app.py
```

Se abre en el navegador (http://localhost:8501). En la barra lateral elige el modelo y presiona
**Ejecutar predicciones** para ver las métricas, la matriz de confusión, las curvas ROC y PR y las
predicciones de cada flujo. Para usar el modelo GRU hace falta TensorFlow, incluido en ese
`requirements.txt`.

### En Google Colab (opcional)

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/GraduacionesUVG/2026-22102-solis-albizurez-juan-pablo.git
%cd 2026-22102-solis-albizurez-juan-pablo/src
!pip install -r requirements.txt -q
```

## Capturas

Las capturas del sistema en funcionamiento están en [`capturas/`](capturas/), en el orden de uso:

| # | Contenido |
|---|---|
| [01](capturas/captura-01.png) | Limpieza: carga de datos crudos y filtrado de clases BENIGN + ataques lentos |
| [02](capturas/captura-02.png) | Limpieza: resumen del conjunto procesado |
| [03](capturas/captura-03.png) | Análisis exploratorio: distribución de clases |
| [04](capturas/captura-04.png) | Análisis exploratorio: distribución temporal de los flujos |
| [05](capturas/captura-05.png) | Selección voraz de características por redundancia |
| [06](capturas/captura-06.png) | Resumen de la reducción de características (39 conservadas) |
| [07](capturas/captura-07.png) | Evaluación de las características derivadas candidatas |
| [08](capturas/captura-08.png) | Matrices finales: 42 características y particiones |
| [09](capturas/captura-09.png) | Búsqueda de hiperparámetros y regla de un error estándar |
| [10](capturas/captura-10.png) | Evaluación en prueba: métricas frente a umbral 0 y línea base |
| [11](capturas/captura-11.png) | Evaluación en prueba: matriz de confusión, curvas ROC y PR |
| [12](capturas/captura-12.png) | Resumen de métricas en prueba con intervalos de confianza |
| [13](capturas/captura-13.png) | Ejecución de las pruebas unitarias con pytest |
| [14](capturas/captura-14.png) | Demo en vivo: pantalla inicial y configuración |
| [15](capturas/captura-15.png) | Demo en vivo: métricas y matriz de confusión del SVM |

## Demo

Video demostrativo: [`demo/demo.mp4`](demo/demo.mp4)

## Documentación

- Informe final: [`docs/informe_final.pdf`](docs/informe_final.pdf)
- Consentimiento de uso de foto y video firmado: [`docs/consentimiento_firmado.pdf`](docs/consentimiento_firmado.pdf)
- Carta de aprobación del asesor: [`docs/carta_aprobacion.pdf`](docs/carta_aprobacion.pdf)
- Documentación técnica del desarrollo: [`src/docs/`](src/docs/) (bitácora, pruebas y registros de decisiones de arquitectura)

## Estructura del repositorio

```
2026-22102-solis-albizurez-juan-pablo/
├── capturas/              # Capturas del sistema (captura-01 … captura-15)
├── demo/
│   └── demo.mp4           # Video demostrativo
├── docs/
│   ├── informe_final.pdf
│   ├── consentimiento_firmado.pdf
│   └── carta_aprobacion.pdf
├── src/
│   ├── .env.example       # El proyecto no requiere variables de entorno
│   ├── requirements.txt   # Dependencias de Python
│   ├── conftest.py
│   ├── data/              # raw/, processed/, sequences/ (el dataset no se sube)
│   ├── docs/              # Bitácora, pruebas y ADR
│   ├── live_demo/         # Demo en vivo en Streamlit (app.py, modelos y datos de prueba)
│   ├── notebooks/         # Cleaning/ y SVM/
│   ├── reports/           # Figuras y métricas generadas
│   ├── src/               # Paquete de Python: preprocessing, evaluation, models
│   └── tests/             # Pruebas unitarias e integración
└── README.md
```

## Limitaciones

- Un solo día de captura y un solo origen de ataque.
- Los ataques de prueba pertenecen a las mismas ejecuciones que los de entrenamiento.
- Etiquetas con limitaciones documentadas en CIC-IDS2017 (Engelen et al., 2021).
- El contraste con XGBoost y GRU usa métricas reportadas, no un análisis pareado flujo por flujo.
- Evaluación *offline*; no se probó en un despliegue real.

## Referencias principales

- Sharafaldin, I., Habibi Lashkari, A. y Ghorbani, A. A. (2018). Toward Generating a New Intrusion
  Detection Dataset and Intrusion Traffic Characterization. *ICISSP*, 108–116.
- Engelen, G., Rimmer, V. y Joosen, W. (2021). Troubleshooting an Intrusion Detection Dataset:
  the CICIDS2017 Case Study. *IEEE SPW*, 7–12.
- Cortes, C. y Vapnik, V. (1995). Support-Vector Networks. *Machine Learning*, 20(3), 273–297.

La lista completa está en el informe final.

## Licencia

Trabajo académico de la Universidad del Valle de Guatemala. Todos los derechos reservados al autor.