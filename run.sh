#!/usr/bin/env bash
# Uso: ./run.sh <procesos> <num_libros> <archivo_urls.txt> [-o salida.csv] [--sin-stopwords]
# Compila si hace falta y lanza la versión MPI (incluye serial + speed-up).
set -euo pipefail
if [ "$#" -lt 3 ]; then
  echo "Uso: $0 <procesos> <num_libros> <archivo_urls.txt> [-o salida.csv] [--sin-stopwords]" >&2
  exit 2
fi
NP="$1"; K="$2"; URLS="$3"; shift 3
case "$NP" in ''|*[!0-9]*|0) echo "Error: <procesos> debe ser un entero positivo" >&2; exit 2;; esac
DIR="$(cd "$(dirname "$0")" && pwd)"
[ -x "$DIR/build/bow_mpi" ] || make -C "$DIR"
EXTRA=()
# OpenMPI se niega a lanzar más procesos que núcleos, salvo con --oversubscribe.
if mpirun --version 2>&1 | grep -qi "open mpi"; then
  EXTRA+=(--oversubscribe)
  [ "$(id -u)" -eq 0 ] && EXTRA+=(--allow-run-as-root)
fi
exec mpirun ${EXTRA[@]+"${EXTRA[@]}"} -np "$NP" "$DIR/build/bow_mpi" "$K" "$URLS" "$@"
