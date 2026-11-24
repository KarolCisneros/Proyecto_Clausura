/**
 * @file bow_mpi.cpp
 * @brief Versión PARALELA (MPI) de la Bolsa de Palabras + línea base serial + speed-up.
 *
 * Uso:  mpirun -np <q> ./bow_mpi <num_libros> <archivo_urls.txt> [-o salida.csv] [--sin-stopwords]
 *       (o bien:  ./run.sh <q> <num_libros> <archivo_urls.txt>)
 *
 * ESTRATEGIA PARALELA
 * -------------------
 *  1. El proceso 0 lee/valida argumentos y el archivo de URLs, y los difunde
 *     (MPI_Bcast) a todos. Así los demás procesos NO necesitan acceso al archivo
 *     (funciona aunque no haya sistema de archivos compartido).
 *  2. Reparto de libros CÍCLICO: el proceso r trabaja los libros r, r+q, r+2q, ...
 *     Cada proceso, para sus libros: descarga -> tokeniza/cuenta -> empaqueta el
 *     conteo en un buffer binario. Las descargas (lo más lento) ocurren en
 *     paralelo, y el conteo de cada libro también.
 *  3. Se acuerda globalmente (MPI_Allreduce) si hubo errores ANTES de cualquier
 *     Gather: así, si un libro falla, TODOS los procesos abortan limpiamente y
 *     nunca queda un proceso esperando en un colectivo (deadlock).
 *  4. MPI_Gather (tamaños) + MPI_Gatherv (bytes) llevan los conteos parciales al
 *     proceso 0, que los desempaqueta, une vocabularios y arma la matriz k x V.
 *  5. Línea base: el proceso 0 ejecuta la versión SERIAL completa (mismo código
 *     de bow_core.hpp) mientras los demás duermen en una barrera no bloqueante
 *     (para no quitarle CPU). Se verifica que ambas matrices sean idénticas y se
 *     imprime el speed-up = T_serial / T_paralelo.
 *
 * Los tiempos comparan "obtener los datos + contar + armar la matriz"; escribir
 * el CSV queda fuera de ambos (es idéntico en las dos versiones).
 *
 * REGLA DE ORO PARA EVITAR DEADLOCKS: toda decisión que cambie el flujo de
 * llamadas colectivas se toma con un valor que TODOS los procesos conocen (por
 * Bcast o Allreduce). Ningún proceso sale de run() por un error local.
 */
#include <mpi.h>
#include <unistd.h>

#include <climits>
#include <cstdio>

#include "bow_core.hpp"

namespace {

/**
 * Barrera que NO consume CPU mientras espera: MPI_Ibarrier + sondeo con
 * dormidas de 1 ms. (MPI_Barrier normal hace "busy-wait": mientras el proceso 0
 * corre el baseline serial, los otros procesos girarían al 100 % compitiendo por
 * núcleos y falsearían el tiempo serial.)
 */
void idle_barrier(MPI_Comm comm) {
    MPI_Request req;
    MPI_Ibarrier(comm, &req);
    int done = 0;
    while (!done) {
        MPI_Test(&req, &done, MPI_STATUS_IGNORE);
        if (!done) usleep(1000);
    }
}

/// Une las URLs en un solo string "url1\nurl2\n..." para difundirlas de una vez.
std::string join_lines(const std::vector<std::string>& v) {
    std::string s;
    for (const auto& x : v) { s += x; s.push_back('\n'); }
    return s;
}

/// Inversa de join_lines.
std::vector<std::string> split_lines(const std::string& s) {
    std::vector<std::string> v;
    std::size_t start = 0;
    while (start < s.size()) {
        std::size_t e = s.find('\n', start);
        if (e == std::string::npos) e = s.size();
        v.emplace_back(s.substr(start, e - start));
        start = e + 1;
    }
    return v;
}

/// Todo el programa. Devuelve el código de salida (idéntico en todos los procesos).
int run(int argc, char** argv, int rank, int size) {
    const MPI_Comm comm = MPI_COMM_WORLD;

    // ------------------------------------------------------------------
    // Fase 0: argumentos (sólo el proceso 0) y difusión a todos.
    // ------------------------------------------------------------------
    bow::Config              cfg;
    std::vector<std::string> urls;
    std::string              blob;          // URLs unidas por '\n'
    int                      status = 0;    // 0 = ok, 2 = error de uso/entrada

    if (rank == 0) {
        std::string err;
        if (!bow::parse_args(argc, argv, cfg, err)) {
            std::fprintf(stderr, "Error: %s\n", err.c_str());
            bow::print_usage(argv[0]);
            status = 2;
        } else if (!bow::read_url_list(cfg.urls_path, cfg.num_books, urls, err)) {
            std::fprintf(stderr, "Error: %s\n", err.c_str());
            status = 2;
        } else {
            blob = join_lines(urls);
            if (blob.size() >= static_cast<std::size_t>(INT_MAX)) {
                std::fprintf(stderr, "Error: lista de URLs demasiado grande\n");
                status = 2;
            }
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, comm);
    if (status != 0) return status;                       // TODOS salen juntos

    int meta[3] = {cfg.num_books, cfg.remove_stopwords ? 1 : 0, static_cast<int>(blob.size())};
    MPI_Bcast(meta, 3, MPI_INT, 0, comm);
    const int  k         = meta[0];
    const bool stopwords = meta[1] != 0;
    if (rank != 0) blob.resize(static_cast<std::size_t>(meta[2]));
    MPI_Bcast(&blob[0], meta[2], MPI_CHAR, 0, comm);      // meta[2] > 0 siempre (k >= 1)
    if (rank != 0) urls = split_lines(blob);

    if (rank == 0 && size > k)
        std::fprintf(stderr, "Aviso: hay %d procesos pero sólo %d libros; %d proceso(s) quedarán ociosos.\n",
                     size, k, size - k);

    bow::CurlRuntime curl_runtime;

    // ------------------------------------------------------------------
    // Fase 1: PARALELO. Se mide desde una barrera común para que todos arranquen a la vez.
    // ------------------------------------------------------------------
    MPI_Barrier(comm);
    const double t_start = MPI_Wtime();

    std::vector<char> sendbuf;                 // conteos empaquetados de MIS libros
    sendbuf.reserve(1 << 16);                  // reserve>0 => data() nunca es nullptr
    int    local_fail = 0;
    double t_dl = 0.0, t_cnt = 0.0;
    {
        std::string text;
        text.reserve(1 << 20);
        for (int i = rank; i < k; i += size) { // reparto cíclico
            std::string e;
            const double a = MPI_Wtime();
            if (!bow::fetch_text(urls[static_cast<std::size_t>(i)], text, e)) {
                std::fprintf(stderr, "[proceso %d] libro %d (%s): %s\n", rank, i,
                             urls[static_cast<std::size_t>(i)].c_str(), e.c_str());
                local_fail = 1;
                break;                         // no seguimos, pero NO salimos de run(): ver Fase 2
            }
            const double b = MPI_Wtime();
            bow::Counts counts;
            counts.reserve(1 << 14);
            bow::count_words(text, counts, stopwords);
            bow::pack_book(sendbuf, static_cast<std::uint32_t>(i), counts);
            const double c = MPI_Wtime();
            t_dl  += b - a;
            t_cnt += c - b;
        }
    }

    // ------------------------------------------------------------------
    // Fase 2: acuerdo global de errores y de tamaños (antes de Gatherv).
    // ------------------------------------------------------------------
    long long my_size = static_cast<long long>(sendbuf.size());
    long long total   = 0;
    int       any_fail = 0;
    MPI_Allreduce(&local_fail, &any_fail, 1, MPI_INT, MPI_MAX, comm);
    MPI_Allreduce(&my_size, &total, 1, MPI_LONG_LONG, MPI_SUM, comm);
    if (any_fail || total > static_cast<long long>(INT_MAX)) {
        if (rank == 0)
            std::fprintf(stderr, any_fail ? "Abortando: falló la obtención de al menos un libro.\n"
                                          : "Abortando: los datos parciales exceden 2 GB.\n");
        return 3;                              // decisión idéntica en todos los procesos
    }

    // ------------------------------------------------------------------
    // Fase 3: Gather de tamaños + Gatherv de bytes hacia el proceso 0.
    // ------------------------------------------------------------------
    const int        sendcount = static_cast<int>(my_size);
    std::vector<int> counts, displs;
    std::vector<char> recv;
    if (rank == 0) { counts.resize(static_cast<std::size_t>(size)); displs.resize(static_cast<std::size_t>(size)); }
    MPI_Gather(&sendcount, 1, MPI_INT, rank == 0 ? counts.data() : nullptr, 1, MPI_INT, 0, comm);
    if (rank == 0) {
        int off = 0;
        for (int r = 0; r < size; ++r) { displs[static_cast<std::size_t>(r)] = off; off += counts[static_cast<std::size_t>(r)]; }
        recv.resize(static_cast<std::size_t>(off > 0 ? off : 1));
    }
    MPI_Gatherv(sendbuf.data(), sendcount, MPI_BYTE, rank == 0 ? recv.data() : nullptr,
                rank == 0 ? counts.data() : nullptr, rank == 0 ? displs.data() : nullptr, MPI_BYTE, 0, comm);
    std::vector<char>().swap(sendbuf);         // liberar memoria

    // ------------------------------------------------------------------
    // Fase 4: ensamblado en el proceso 0 (desempaquetar + matriz). Su resultado
    // se difunde como código de estado para que nadie quede esperando.
    // ------------------------------------------------------------------
    bow::Matrix par_matrix;
    double      t_par  = 0.0;
    int         asm_ok = 1;
    if (rank == 0) {
        std::vector<bow::Counts> books(static_cast<std::size_t>(k));
        std::string              err;
        if (bow::unpack_books(recv.data(), static_cast<std::size_t>(total), books, err)) {
            par_matrix = bow::build_matrix(books);
        } else {
            std::fprintf(stderr, "Error al desempaquetar resultados: %s\n", err.c_str());
            asm_ok = 0;
        }
        t_par = MPI_Wtime() - t_start;         // el proceso 0 termina el último => define el tiempo
    }
    MPI_Bcast(&asm_ok, 1, MPI_INT, 0, comm);
    if (!asm_ok) return 3;

    // Tiempos máximos entre procesos (el más lento manda en paralelo).
    double mine[3]    = {t_dl, t_cnt, t_dl + t_cnt};
    double maxima[3]  = {0, 0, 0};
    MPI_Reduce(mine, maxima, 3, MPI_DOUBLE, MPI_MAX, 0, comm);

    // ------------------------------------------------------------------
    // Fase 5: línea base SERIAL en el proceso 0; los demás duermen.
    // ------------------------------------------------------------------
    bow::Matrix ser_matrix;
    bow::Timing ser_t;
    bool        ser_ok = false;
    std::string ser_err;
    if (rank == 0) ser_ok = bow::run_serial(urls, stopwords, ser_matrix, ser_t, ser_err);
    idle_barrier(comm);

    if (rank != 0) return 0;

    // ------------------------------------------------------------------
    // Fase 6 (sólo proceso 0): salida y reporte.
    // ------------------------------------------------------------------
    std::string err;
    if (!bow::write_csv(par_matrix, cfg.out_path, err) || !bow::write_book_index(urls, cfg.out_path, err)) {
        std::fprintf(stderr, "Error: %s\n", err.c_str());
        return 4;
    }

    std::printf("==================== BOLSA DE PALABRAS (MPI) ====================\n");
    std::printf("Libros (k)            : %d\n", k);
    std::printf("Procesos (q)          : %d (con trabajo: %d)\n", size, size < k ? size : k);
    std::printf("Matriz                : %zu filas x %zu columnas (vocabulario)\n", par_matrix.rows,
                par_matrix.vocab.size());
    std::printf("Palabras vacías       : %s\n", stopwords ? "eliminadas" : "incluidas");
    std::printf("CSV                   : %s\n", cfg.out_path.c_str());
    std::printf("-----------------------------------------------------------------\n");
    std::printf("PARALELO  total       : %.4f s\n", t_par);
    std::printf("   descarga (máx/proc): %.4f s\n", maxima[0]);
    std::printf("   conteo   (máx/proc): %.4f s\n", maxima[1]);
    std::printf("   comunicación + matriz (resto): %.4f s\n", t_par - maxima[2] > 0 ? t_par - maxima[2] : 0.0);
    if (ser_ok) {
        std::printf("SERIAL    total       : %.4f s\n", ser_t.total);
        std::printf("   descarga           : %.4f s\n", ser_t.download);
        std::printf("   conteo             : %.4f s\n", ser_t.count);
        std::printf("   matriz             : %.4f s\n", ser_t.assemble);
        const bool same = (ser_matrix == par_matrix);
        std::printf("-----------------------------------------------------------------\n");
        std::printf("Verificación serial == paralelo : %s\n", same ? "OK (matrices idénticas)" : "¡DIFIEREN!");
        const double speedup = ser_t.total / t_par;
        std::printf("SPEED-UP  = T_serial / T_paralelo = %.4f / %.4f = %.2fx\n", ser_t.total, t_par, speedup);
        std::printf("Eficiencia = speed-up / q = %.1f %%\n", 100.0 * speedup / size);
        std::printf("Meta del proyecto (speed-up > 1.2x): %s\n", speedup > 1.2 ? "CUMPLIDA" : "NO cumplida");
        if (!same) return 5;
    } else {
        std::fprintf(stderr, "Aviso: falló la línea base serial (%s); no se calcula speed-up.\n", ser_err.c_str());
    }
    std::printf("=================================================================\n");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int rc = 0;
    try {
        rc = run(argc, argv, rank, size);
    } catch (const std::exception& ex) {
        // Una excepción en un solo proceso dejaría a los demás en un colectivo: abortar todo.
        std::fprintf(stderr, "[proceso %d] excepción: %s\n", rank, ex.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return rc;
}
