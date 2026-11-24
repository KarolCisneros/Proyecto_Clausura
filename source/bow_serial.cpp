/**
 * @file bow_serial.cpp
 * @brief Versión SERIAL independiente de la Bolsa de Palabras.
 *
 * Uso:  ./bow_serial <num_libros> <archivo_urls.txt> [-o salida.csv] [--sin-stopwords]
 *
 * Descarga (uno por uno) los primeros k libros del archivo de URLs, cuenta sus
 * palabras y escribe la matriz k x V en CSV. Toda la lógica vive en bow_core.hpp.
 * (La versión MPI, bow_mpi.cpp, ya incluye internamente esta misma versión
 * serial como línea base para imprimir el speed-up.)
 */
#include <cstdio>

#include "bow_core.hpp"

int main(int argc, char** argv) {
    bow::Config cfg;
    std::string err;
    if (!bow::parse_args(argc, argv, cfg, err)) {
        std::fprintf(stderr, "Error: %s\n", err.c_str());
        bow::print_usage(argv[0]);
        return 2;
    }
    std::vector<std::string> urls;
    if (!bow::read_url_list(cfg.urls_path, cfg.num_books, urls, err)) {
        std::fprintf(stderr, "Error: %s\n", err.c_str());
        return 2;
    }

    bow::CurlRuntime curl_runtime;
    bow::Matrix      m;
    bow::Timing      t;
    if (!bow::run_serial(urls, cfg.remove_stopwords, m, t, err)) {
        std::fprintf(stderr, "Error: %s\n", err.c_str());
        return 3;
    }
    if (!bow::write_csv(m, cfg.out_path, err) || !bow::write_book_index(urls, cfg.out_path, err)) {
        std::fprintf(stderr, "Error: %s\n", err.c_str());
        return 4;
    }
    std::printf("Versión SERIAL\n");
    std::printf("  Libros: %d | Vocabulario: %zu | Matriz: %zu x %zu\n", cfg.num_books, m.vocab.size(),
                m.rows, m.vocab.size());
    std::printf("  Tiempo total: %.3f s (descarga %.3f, conteo %.3f, matriz %.3f)\n", t.total, t.download,
                t.count, t.assemble);
    std::printf("  CSV: %s\n", cfg.out_path.c_str());
    return 0;
}
