# Proyecto Clausura — Bolsa de Palabras con MPI (C++)

Cómputo Paralelo y en la Nube · Versión **serial** y versión **paralela (MPI)** del modelo de
Bolsa de Palabras sobre libros descargados dinámicamente de Project Gutenberg. Funciona para
**k libros** y **q procesos**, imprime el **speed-up** y escribe la matriz en **CSV**.

---

## 1. Estructura

```
proyecto/
├── source/
│   ├── bow_core.hpp     # núcleo común: descarga, tokenizador, matriz, CSV, versión SERIAL
│   ├── bow_mpi.cpp      # versión PARALELA (MPI) + línea base serial + speed-up
│   └── bow_serial.cpp   # versión SERIAL como programa independiente
├── data/
│   └── urls_gutenberg.txt   # listado de URLs (ejemplo con los 6 libros del notebook)
├── Makefile             # make  -> build/bow_mpi y build/bow_serial
├── run.sh               # ./run.sh <procesos> <libros> <urls.txt>
└── README.md
```

## 2. Requisitos y compilación

* Compilador C++17 y una implementación de MPI (OpenMPI o MPICH): `mpicxx`, `mpirun`.
* `libcurl` con cabeceras (descarga HTTPS):
  `sudo apt install build-essential openmpi-bin libopenmpi-dev libcurl4-openssl-dev`

```bash
make            # genera build/bow_mpi y build/bow_serial
```

## 3. Ejecución

Entradas (según la especificación): **número de procesos**, **número de libros** y
**listado de URLs en un .txt**.

```bash
./run.sh <procesos> <num_libros> <archivo_urls.txt>
./run.sh 6 6 data/urls_gutenberg.txt              # 6 procesos, 6 libros

# equivalente, sin el script:
mpirun -np 6 build/bow_mpi 6 data/urls_gutenberg.txt

# opciones (opcionales):
#   -o ruta.csv        archivo de salida (por defecto output/bolsa_de_palabras.csv)
#   --sin-stopwords    elimina palabras vacías del inglés (the, of, and, ...)

# sólo la versión serial:
build/bow_serial 6 data/urls_gutenberg.txt
```

> **Sobre "número de procesos":** en MPI ese número lo fija el lanzador (`mpirun -np q`);
> el programa lo lee con `MPI_Comm_size`. `run.sh` simplemente lo recibe como primer argumento.
> Si pides más procesos que núcleos, `run.sh` agrega `--oversubscribe` (OpenMPI).

### Formato del listado de URLs (`data/*.txt`)

* **Una URL por línea**, texto plano. Sin comas, sin comillas, sin encabezado.
* Líneas vacías y líneas que empiezan con `#` se ignoran. Se toleran saltos de línea de Windows.
* Se usan **las primeras k** URLs; el orden del archivo es el orden de las filas de la matriz.
  Si el archivo tiene menos de k URLs válidas, el programa termina con un error claro.
* Puede haber URLs repetidas (cada una genera su propia fila).
* Usa el **texto plano UTF‑8** de Gutenberg: `https://www.gutenberg.org/cache/epub/<ID>/pg<ID>.txt`

```text
# data/urls_gutenberg.txt
https://www.gutenberg.org/cache/epub/46/pg46.txt       # A Christmas Carol
https://www.gutenberg.org/cache/epub/98/pg98.txt       # A Tale of Two Cities
https://www.gutenberg.org/cache/epub/1524/pg1524.txt   # Hamlet
https://www.gutenberg.org/cache/epub/1513/pg1513.txt   # Romeo and Juliet
https://www.gutenberg.org/cache/epub/730/pg730.txt     # Oliver Twist
https://www.gutenberg.org/cache/epub/1515/pg1515.txt   # The Merchant of Venice
```

(Para pruebas sin internet, una línea también puede ser una **ruta local** o `file:///ruta/libro.txt`.)

### Salidas

| Archivo | Contenido |
|---|---|
| `output/bolsa_de_palabras.csv` | Matriz de Bolsa de Palabras. **Fila 1** = vocabulario (columnas, orden alfabético); luego **una fila por libro**, en el orden de las URLs. Sin columna de índice (igual que `matriz_conteo` del notebook). |
| `output/bolsa_de_palabras_libros.txt` | Auxiliar: qué URL corresponde a cada fila (0, 1, ...). |
| Consola | Tiempos paralelo/serial, verificación, **speed-up** y eficiencia. |

Se carga directo con `pd.read_csv("output/bolsa_de_palabras.csv")`.

---

## 4. Modelo de Bolsa de Palabras implementado

Para cada libro: se pasa a **minúsculas**, se separa en **tokens = secuencias de letras/dígitos
ASCII** y se **descartan tokens de 1 carácter**; se cuenta cuántas veces aparece cada token.
El vocabulario es la **unión ordenada alfabéticamente** de todos los tokens; la celda
`(i, j)` es el número de veces que la palabra `j` aparece en el libro `i`.

Esto reproduce lo que hace `CountVectorizer(min_df=0.0, max_df=1.0)` del notebook
(patrón `\b\w\w+\b`, minúsculas, columnas ordenadas), con dos diferencias deliberadas:

* `_` **no** forma parte de las palabras (en Gutenberg `_palabra_` marca cursivas; así se cuenta `palabra`).
* Los bytes no ASCII (comillas tipográficas “ ” ’, guiones largos) actúan como separadores.

Se conserva el encabezado/pie de licencia de Gutenberg (el notebook también lo conserva:
sus textos inician con `project, gutenberg, ebook, ...`). `--sin-stopwords` es opcional.

## 5. Diseño de la versión paralela (MPI)

```
 Proceso 0: lee argumentos + URLs ──Bcast──▶ todos
 Cada proceso r:  libros r, r+q, r+2q, ...   (reparto cíclico)
        descarga ─▶ tokeniza/cuenta ─▶ empaqueta (buffer binario)
 ─────────── Allreduce(error?) + Allreduce(tamaño total) ───────────
 Gather(tamaños) + Gatherv(bytes) ──▶ Proceso 0
 Proceso 0: desempaqueta ─▶ une vocabularios ─▶ matriz k × V ─▶ CSV
```

**Paso a paso (`source/bow_mpi.cpp`)**

1. **Entrada y difusión.** Sólo el proceso 0 valida argumentos y lee el `.txt`. Difunde con
   `MPI_Bcast` (estado → metadatos → lista de URLs). Así **no se necesita sistema de archivos
   compartido**, y si hay error de uso **todos** salen juntos (ningún proceso queda esperando).
2. **Reparto cíclico.** Al proceso `r` le tocan los libros `i = r, r+q, r+2q, …`. Funciona para
   cualquier `k` y `q` (incluido `q = 1` y `q > k`: los procesos sobrantes simplemente no tienen
   libros y envían 0 bytes; se imprime un aviso).
3. **Trabajo local.** Descarga (libcurl, con reintentos) → `count_words` (un solo recorrido del
   texto, tabla de 256 entradas para minúsculas/separadores, `unordered_map`) → `pack_book`.
   La **descarga** y el **conteo** se hacen en paralelo en los q procesos.
4. **Acuerdo de errores (anti‑deadlock).** Antes de cualquier `Gather`, `MPI_Allreduce(MAX)` sobre
   un indicador de fallo y `MPI_Allreduce(SUM)` sobre el tamaño total. Todos conocen el mismo
   valor, así que **todos** abortan limpiamente (código 3) si una descarga falla o si los datos
   excederían el límite de `int` de `MPI_Gatherv`. Ningún proceso sale de un colectivo por un error local.
5. **Recolección.** `MPI_Gather` de los tamaños y `MPI_Gatherv` de los bytes (`MPI_BYTE`) hacia el
   proceso 0. El volumen es pequeño (≈ 100–200 KB por libro: sólo palabras únicas + conteos, no el texto).
6. **Ensamblado (proceso 0).** `unpack_books` (con validación de límites) y `build_matrix`:
   unión de palabras (`unordered_set<string_view>`), orden alfabético, mapa palabra→columna y llenado de
   la matriz. El resultado se confirma al resto con un `MPI_Bcast` de estado.
7. **Línea base serial.** El proceso 0 corre `run_serial` (mismo código del núcleo) mientras los
   demás esperan en `MPI_Ibarrier` + `usleep(1 ms)` (**no** `MPI_Barrier`, que hace *busy‑wait*
   y le robaría CPU al proceso 0, falseando el tiempo serial).
8. **Reporte.** Verifica `matriz_serial == matriz_paralela`, calcula
   `speed-up = T_serial / T_paralelo` y eficiencia `speed-up / q`, y escribe el CSV.

**Otras decisiones para evitar errores de MPI**

* Cada proceso corre **todas** las llamadas colectivas en el mismo orden; las decisiones que cambian
  el flujo se toman con valores comunes (Bcast/Allreduce).
* `sendbuf.reserve(...)` garantiza que `data()` nunca sea `nullptr`; en procesos no raíz los
  parámetros "sólo raíz" de `Gather/Gatherv` se pasan como `nullptr`.
* Contadores y desplazamientos de `Gatherv` son `int`; se valida que el total no pase de `INT_MAX`.
* Una excepción C++ en un proceso llama a `MPI_Abort` (si no, los demás quedarían bloqueados).
* `MPI_Init`/`MPI_Finalize` en `main`; los objetos con destructores (libcurl, buffers) viven dentro de `run()`.

**Qué se mide.** `T_paralelo` va desde una barrera común hasta que el proceso 0 termina de armar la
matriz (incluye descarga, conteo, comunicación y ensamblado). `T_serial` es lo mismo en un solo proceso.
Escribir el CSV queda fuera de ambos. La paralela se ejecuta **antes** que la serial, de modo que las
cachés de DNS/conexión favorecen (si acaso) a la serial: el speed-up reportado es conservador.

**Complejidad.** Con `N` bytes de texto total y `q` procesos: conteo `O(N/q)` por proceso (en el mejor caso),
descargas en paralelo, comunicación `O(P)` con `P` = palabras únicas totales, y ensamblado `O(P + V log V)`
en el proceso 0. El balance es por libro: el tiempo lo marca el proceso con más libros (⌈k/q⌉); el
máximo aprovechamiento se logra con `q = k`.

## 6. Resultados de prueba

Validación hecha en el entorno de desarrollo (OpenMPI 4.1.6, **1 solo núcleo**, 7 libros de ~340 KB
servidos por HTTP local con 0.6 s de latencia simulada para emular la red):

| q | T paralelo | T serial | Speed-up |
|---|-----------:|---------:|---------:|
| 1 | 4.24 s | 4.24 s | 1.00× |
| 2 | 2.44 s | 4.23 s | 1.74× |
| 4 | 1.24 s | 4.23 s | 3.42× |
| 7 | 0.63 s | 4.24 s | 6.69× |
| 9 (> k) | 0.63 s | 4.26 s | 6.76× |

* La matriz fue **idéntica** para todo q, **idéntica** a la versión serial standalone y **exacta** contra una
  implementación independiente en Python (con y sin `--sin-stopwords`).
* Casos de error probados sin bloqueos: URL con HTTP 404, `k` mayor que las URLs del archivo, `k` inválido,
  archivo inexistente, opción desconocida, saltos de línea CRLF y URLs duplicadas.
* **Importante y honesto:** la mejora viene sobre todo de **descargar en paralelo** (la latencia de red
  domina). En esa prueba de 1 núcleo, el conteo *solo* (archivos locales, sin red) no mejora (≈0.8×), como es
  de esperar sin núcleos adicionales. En un equipo con varios núcleos el conteo también escala. Con Gutenberg
  real, la descarga domina el tiempo, por lo que el speed-up > 1.2× es esperable con q ≥ 2; no pude probar
  contra gutenberg.org desde mi entorno, **corre tú la prueba final** y revisa el valor impreso. Los tiempos
  de red varían entre ejecuciones; repite si quedas cerca del umbral.

## 7. Criterios de la rúbrica

| Requisito | Dónde |
|---|---|
| Versión paralela MPI y versión serial | `bow_mpi.cpp`, `bow_serial.cpp` (+ `run_serial` en `bow_core.hpp`) |
| Entrada: nº de procesos, listado de URLs (descarga dinámica) | `-np q` / `run.sh`, `data/*.txt`, libcurl |
| Salida: matriz de Bolsa de Palabras en CSV | `output/bolsa_de_palabras.csv` |
| Calcular e imprimir speed-up | Fase 6 (reporte) de `bow_mpi.cpp` |
| Funciona para k libros y q procesos | reparto cíclico; probado q < k, q = k, q > k |
| Speed-up > 1.2× (si no, −1 punto) | se imprime "CUMPLIDA / NO cumplida" |
| Código documentado (0.1 pt) | comentarios Doxygen‑style en todos los archivos |

## 8. Problemas frecuentes

* `libcurl` / `curl/curl.h` no encontrado → instala `libcurl4-openssl-dev` (Linux) o `brew install curl open-mpi` (macOS).
* OpenMPI: *"not enough slots"* → usa `./run.sh` (agrega `--oversubscribe`) o `mpirun --oversubscribe ...`.
* En contenedores como root: `mpirun --allow-run-as-root ...` (`run.sh` ya lo agrega).
* Error `HTTP 403/429` de Gutenberg → demasiadas peticiones seguidas; espera unos minutos y reintenta.
* Si Gutenberg cambia una URL, abre la página del libro en gutenberg.org y usa el formato `cache/epub/<ID>/pg<ID>.txt`.
