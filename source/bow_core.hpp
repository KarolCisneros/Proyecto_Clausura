/**
 * @file bow_core.hpp
 * @brief Núcleo común (serial) del proyecto "Bolsa de Palabras con MPI".
 *
 * Este archivo contiene TODO lo que comparten la versión serial (bow_serial.cpp)
 * y la versión paralela (bow_mpi.cpp):
 *
 *   1. Lectura y validación de argumentos y del archivo de URLs.
 *   2. Obtención del texto de cada libro (descarga HTTP(S) con libcurl, o
 *      lectura de un archivo local para pruebas sin red).
 *   3. Tokenización y conteo de palabras (el "modelo de bolsa de palabras").
 *   4. Construcción de la matriz k x V (libros x vocabulario).
 *   5. Escritura del CSV.
 *   6. Empaquetado/desempaquetado binario de conteos (lo usa MPI para enviar
 *      resultados parciales entre procesos).
 *   7. La versión SERIAL completa (run_serial), que además sirve como línea
 *      base para calcular el speed-up.
 *
 * Es "header-only" (todo es `inline`) para que compilar sea trivial:
 *      g++/mpicxx -O3 -std=c++17 archivo.cpp -lcurl
 *
 * Decisión de diseño importante: la versión serial y la paralela comparten el
 * MISMO tokenizador, la MISMA descarga y el MISMO constructor de matriz. Así la
 * comparación de tiempos es justa y la verificación (serial == paralelo) es
 * significativa: lo único que cambia es la distribución del trabajo.
 */
#ifndef BOW_CORE_HPP
#define BOW_CORE_HPP

#include <curl/curl.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace bow {

// ============================================================================
// Tipos básicos
// ============================================================================

/// Conteo de palabras de UN libro: palabra -> número de apariciones.
using Counts = std::unordered_map<std::string, std::uint32_t>;

/// Parámetros de ejecución (comunes a la versión serial y a la paralela).
struct Config {
    int         num_books        = 0;                              ///< k: cuántos libros usar
    std::string urls_path;                                         ///< ruta del .txt con las URLs
    std::string out_path         = "output/bolsa_de_palabras.csv"; ///< CSV de salida
    bool        remove_stopwords = false;                          ///< ¿quitar palabras vacías (the, of, ...)?
};

/// Matriz de Bolsa de Palabras: filas = libros, columnas = vocabulario ordenado.
struct Matrix {
    std::vector<std::string>   vocab;  ///< palabras únicas, orden alfabético (columnas)
    std::size_t                rows = 0;
    std::vector<std::uint32_t> data;   ///< almacenamiento por filas: data[i * vocab.size() + j]

    std::uint32_t at(std::size_t i, std::size_t j) const { return data[i * vocab.size() + j]; }
    bool operator==(const Matrix& o) const {
        return rows == o.rows && vocab == o.vocab && data == o.data;
    }
};

/// Tiempos (segundos) de una ejecución.
struct Timing {
    double download = 0.0;  ///< suma de tiempos de descarga de los libros procesados
    double count    = 0.0;  ///< suma de tiempos de tokenización + conteo
    double assemble = 0.0;  ///< construcción de la matriz (vocabulario + llenado)
    double total    = 0.0;  ///< tiempo total de la ejecución
};

/// Reloj monótono en segundos.
inline double now_sec() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

// ============================================================================
// 1. Argumentos y archivo de URLs
// ============================================================================

inline void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Uso: %s <num_libros> <archivo_urls.txt> [-o salida.csv] [--sin-stopwords]\n"
        "  num_libros        k: cuántas URLs del archivo se usan (las primeras k)\n"
        "  archivo_urls.txt  una URL por línea (líneas vacías y que inician con # se ignoran)\n"
        "  -o salida.csv     ruta del CSV (por defecto output/bolsa_de_palabras.csv)\n"
        "  --sin-stopwords   elimina palabras vacías del inglés (the, of, and, ...)\n",
        prog);
}

/// Lee argv. Devuelve false (y llena `err`) si hay algún problema.
inline bool parse_args(int argc, char** argv, Config& cfg, std::string& err) {
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-o" || a == "--salida") {
            if (i + 1 >= argc) { err = "falta el valor de " + a; return false; }
            cfg.out_path = argv[++i];
        } else if (a == "--sin-stopwords") {
            cfg.remove_stopwords = true;
        } else if (a.size() > 1 && a[0] == '-') {
            err = "opción desconocida: " + a;
            return false;
        } else {
            pos.push_back(a);
        }
    }
    if (pos.size() != 2) {
        err = "se requieren exactamente 2 argumentos: <num_libros> <archivo_urls.txt>";
        return false;
    }
    char* end = nullptr;
    errno = 0;
    long v = std::strtol(pos[0].c_str(), &end, 10);
    if (errno != 0 || *end != '\0' || pos[0].empty() || v <= 0 || v > 100000000L) {
        err = "num_libros debe ser un entero positivo (recibido: '" + pos[0] + "')";
        return false;
    }
    cfg.num_books = static_cast<int>(v);
    cfg.urls_path = pos[1];
    return true;
}

/// Quita espacios/tabuladores/CR/LF de los extremos (tolera archivos de Windows).
inline std::string trim(const std::string& s) {
    const char* ws = " \t\r\n\f\v";
    std::size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    std::size_t e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
}

/**
 * Lee el archivo de URLs. Formato: una fuente por línea; se ignoran líneas
 * vacías y comentarios (#). Se conservan las PRIMERAS k entradas, en orden:
 * la fila i de la matriz corresponde a la i-ésima URL.
 */
inline bool read_url_list(const std::string& path, int k,
                          std::vector<std::string>& urls, std::string& err) {
    std::ifstream in(path);
    if (!in) { err = "no se pudo abrir el archivo de URLs: " + path; return false; }
    urls.clear();
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        urls.push_back(line);
    }
    if (static_cast<int>(urls.size()) < k) {
        err = "el archivo '" + path + "' tiene " + std::to_string(urls.size()) +
              " URL(s) válida(s) pero se pidieron " + std::to_string(k) + " libros";
        return false;
    }
    urls.resize(static_cast<std::size_t>(k));
    return true;
}

// ============================================================================
// 2. Obtención del texto (descarga HTTP(S) o archivo local)
// ============================================================================

/// RAII para libcurl: curl_global_init/cleanup una vez por proceso.
struct CurlRuntime {
    CurlRuntime()  { curl_global_init(CURL_GLOBAL_DEFAULT); }
    ~CurlRuntime() { curl_global_cleanup(); }
    CurlRuntime(const CurlRuntime&)            = delete;
    CurlRuntime& operator=(const CurlRuntime&) = delete;
};

/// Callback de libcurl: agrega los bytes recibidos al std::string del usuario.
inline std::size_t curl_write_cb(char* ptr, std::size_t size, std::size_t nmemb, void* userdata) {
    auto* s = static_cast<std::string*>(userdata);
    s->append(ptr, size * nmemb);
    return size * nmemb;
}

inline bool is_http(const std::string& s) {
    return s.rfind("http://", 0) == 0 || s.rfind("https://", 0) == 0;
}

/// Descarga `url` a memoria. Reintenta (3 veces, con espera creciente) ante
/// errores de red, HTTP 5xx o 429; falla de inmediato ante 4xx (p. ej. 404).
inline bool http_get(const std::string& url, std::string& out, std::string& err) {
    constexpr int kMaxAttempts = 3;
    for (int attempt = 1; attempt <= kMaxAttempts; ++attempt) {
        out.clear();
        CURL* h = curl_easy_init();
        if (!h) { err = "curl_easy_init falló"; return false; }
        curl_easy_setopt(h, CURLOPT_URL, url.c_str());
        curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, curl_write_cb);
        curl_easy_setopt(h, CURLOPT_WRITEDATA, &out);
        curl_easy_setopt(h, CURLOPT_FOLLOWLOCATION, 1L);          // gutenberg.org redirige a veces
        curl_easy_setopt(h, CURLOPT_MAXREDIRS, 10L);
        curl_easy_setopt(h, CURLOPT_USERAGENT, "bolsa-palabras-mpi/1.0 (proyecto academico)");
        curl_easy_setopt(h, CURLOPT_ACCEPT_ENCODING, "");         // permite gzip/deflate si el servidor lo ofrece
        curl_easy_setopt(h, CURLOPT_CONNECTTIMEOUT, 15L);
        curl_easy_setopt(h, CURLOPT_TIMEOUT, 120L);
        curl_easy_setopt(h, CURLOPT_NOSIGNAL, 1L);                // seguro con hilos internos de curl/MPI
        CURLcode rc   = curl_easy_perform(h);
        long     code = 0;
        curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &code);
        curl_easy_cleanup(h);

        if (rc == CURLE_OK && code >= 200 && code < 300) return true;

        err = (rc != CURLE_OK) ? std::string(curl_easy_strerror(rc)) : "HTTP " + std::to_string(code);
        bool retryable = (rc != CURLE_OK) || code >= 500 || code == 429;
        if (!retryable || attempt == kMaxAttempts) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(500 * attempt));
    }
    return false;
}

/// Lee un archivo local completo (modo binario).
inline bool read_local_file(const std::string& path, std::string& out, std::string& err) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { err = "no se pudo abrir el archivo local: " + path; return false; }
    in.seekg(0, std::ios::end);
    std::streamoff n = in.tellg();
    in.seekg(0, std::ios::beg);
    out.assign(static_cast<std::size_t>(n > 0 ? n : 0), '\0');
    if (n > 0) in.read(&out[0], n);
    return true;
}

/// Obtiene el texto de una "fuente": URL http(s) (caso del proyecto) o ruta
/// local / file:// (útil para pruebas sin conexión).
inline bool fetch_text(const std::string& source, std::string& out, std::string& err) {
    if (is_http(source)) return http_get(source, out, err);
    std::string path = source.rfind("file://", 0) == 0 ? source.substr(7) : source;
    return read_local_file(path, out, err);
}

// ============================================================================
// 3. Tokenización y conteo (modelo de bolsa de palabras)
// ============================================================================

/**
 * Tabla de 256 entradas: para cada byte, su versión en minúscula si es
 * alfanumérico ASCII [A-Za-z0-9]; 0 si es separador. Una sola consulta de
 * tabla por byte reemplaza a isalnum()+tolower() (más rápido y sin depender
 * del locale).
 */
inline const std::array<unsigned char, 256>& token_table() {
    static const std::array<unsigned char, 256> t = [] {
        std::array<unsigned char, 256> a{};
        for (int c = '0'; c <= '9'; ++c) a[c] = static_cast<unsigned char>(c);
        for (int c = 'a'; c <= 'z'; ++c) a[c] = static_cast<unsigned char>(c);
        for (int c = 'A'; c <= 'Z'; ++c) a[c] = static_cast<unsigned char>(c - 'A' + 'a');
        return a;
    }();
    return t;
}

/// Lista de stopwords del inglés (la de NLTK). Sólo se usa con --sin-stopwords.
inline const std::unordered_set<std::string>& stopwords() {
    static const std::unordered_set<std::string> s = {
        "i","me","my","myself","we","our","ours","ourselves","you","your","yours","yourself",
        "yourselves","he","him","his","himself","she","her","hers","herself","it","its","itself",
        "they","them","their","theirs","themselves","what","which","who","whom","this","that",
        "these","those","am","is","are","was","were","be","been","being","have","has","had",
        "having","do","does","did","doing","an","the","and","but","if","or","because","as",
        "until","while","of","at","by","for","with","about","against","between","into","through",
        "during","before","after","above","below","to","from","up","down","in","out","on","off",
        "over","under","again","further","then","once","here","there","when","where","why","how",
        "all","any","both","each","few","more","most","other","some","such","no","nor","not",
        "only","own","same","so","than","too","very","can","will","just","don","should","now",
        "ll","re","ve","ain","aren","couldn","didn","doesn","hadn","hasn","haven","isn","ma",
        "mightn","mustn","needn","shan","shouldn","wasn","weren","won","wouldn"};
    return s;
}

/**
 * Cuenta las palabras de `text` y las acumula en `counts`.
 *
 * Reglas (equivalentes a CountVectorizer de scikit-learn en texto ASCII):
 *   - minúsculas;
 *   - token = secuencia máxima de letras/dígitos ASCII;
 *   - se descartan tokens de 1 carácter;
 *   - bytes no ASCII (p. ej. comillas tipográficas en UTF-8) actúan de separador.
 * Un solo recorrido O(n) del texto, sin regex ni copias intermedias.
 */
inline void count_words(const std::string& text, Counts& counts, bool remove_stopwords) {
    const auto&          tbl = token_table();
    const auto&          sw  = stopwords();
    const unsigned char* p   = reinterpret_cast<const unsigned char*>(text.data());
    const std::size_t    n   = text.size();

    std::string tok;
    tok.reserve(64);
    std::size_t i = 0;
    while (i < n) {
        if (!tbl[p[i]]) { ++i; continue; }          // separador: avanzar
        tok.clear();
        while (i < n && tbl[p[i]]) {                // acumular token en minúsculas
            tok.push_back(static_cast<char>(tbl[p[i]]));
            ++i;
        }
        if (tok.size() < 2) continue;               // descartar 1 carácter
        if (remove_stopwords && sw.count(tok)) continue;
        ++counts[tok];                              // inserta con 0 si no existe, luego incrementa
    }
}

// ============================================================================
// 4. Construcción de la matriz libros x vocabulario
// ============================================================================

/**
 * A partir de los conteos de cada libro (books[i] = libro i) construye la
 * matriz densa. El vocabulario es la unión de todas las palabras, ORDENADO
 * alfabéticamente (igual que CountVectorizer.get_feature_names_out()).
 * Costo: O(N) hashing + O(V log V) por el ordenamiento, con N = entradas totales.
 */
inline Matrix build_matrix(const std::vector<Counts>& books) {
    // 1) Unión de palabras (sin copiar strings: string_view sobre las claves).
    std::size_t total_entries = 0;
    for (const auto& b : books) total_entries += b.size();
    std::unordered_set<std::string_view> seen;
    seen.reserve(total_entries / 2 + 16);
    for (const auto& b : books)
        for (const auto& kv : b) seen.insert(std::string_view(kv.first));

    // 2) Orden alfabético = orden de columnas.
    std::vector<std::string_view> sorted(seen.begin(), seen.end());
    std::sort(sorted.begin(), sorted.end());

    // 3) palabra -> columna.
    std::unordered_map<std::string_view, std::size_t> col;
    col.reserve(sorted.size() * 2);
    for (std::size_t j = 0; j < sorted.size(); ++j) col.emplace(sorted[j], j);

    // 4) Llenado. La matriz inicia en ceros; sólo se escriben las entradas no nulas.
    Matrix m;
    m.rows = books.size();
    m.vocab.reserve(sorted.size());
    for (auto sv : sorted) m.vocab.emplace_back(sv);
    m.data.assign(m.rows * m.vocab.size(), 0u);
    for (std::size_t i = 0; i < books.size(); ++i)
        for (const auto& kv : books[i])
            m.data[i * m.vocab.size() + col.at(std::string_view(kv.first))] = kv.second;
    return m;
}

// ============================================================================
// 5. Salida (CSV)
// ============================================================================

/**
 * Escribe el CSV: primera línea = vocabulario (encabezado), luego una fila por
 * libro, en el orden del archivo de URLs. Las palabras sólo contienen [a-z0-9],
 * por lo que no requieren comillas ni escape.
 */
inline bool write_csv(const Matrix& m, const std::string& path, std::string& err) {
    namespace fs = std::filesystem;
    fs::path parent = fs::path(path).parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        fs::create_directories(parent, ec);
        if (ec) { err = "no se pudo crear el directorio " + parent.string() + ": " + ec.message(); return false; }
    }
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { err = "no se pudo abrir para escribir: " + path; return false; }
    std::setvbuf(f, nullptr, _IOFBF, 1 << 20);

    std::string line;
    line.reserve(m.vocab.size() * 8 + 16);
    for (std::size_t j = 0; j < m.vocab.size(); ++j) {      // encabezado
        if (j) line.push_back(',');
        line += m.vocab[j];
    }
    line.push_back('\n');
    std::fwrite(line.data(), 1, line.size(), f);

    char tmp[16];
    for (std::size_t i = 0; i < m.rows; ++i) {               // una fila por libro
        line.clear();
        const std::uint32_t* row = m.data.data() + i * m.vocab.size();
        for (std::size_t j = 0; j < m.vocab.size(); ++j) {
            if (j) line.push_back(',');
            auto r = std::to_chars(tmp, tmp + sizeof(tmp), row[j]);
            line.append(tmp, static_cast<std::size_t>(r.ptr - tmp));
        }
        line.push_back('\n');
        std::fwrite(line.data(), 1, line.size(), f);
    }
    bool ok = (std::fflush(f) == 0) && !std::ferror(f);
    ok = (std::fclose(f) == 0) && ok;
    if (!ok) err = "error de escritura en " + path;
    return ok;
}

/// Archivo auxiliar "<salida>_libros.txt": fila (desde 0) <TAB> fuente.
inline bool write_book_index(const std::vector<std::string>& urls, const std::string& csv_path,
                             std::string& err) {
    std::string path = std::filesystem::path(csv_path).replace_extension("").string() + "_libros.txt";
    std::ofstream out(path);
    if (!out) { err = "no se pudo escribir " + path; return false; }
    out << "fila\tfuente\n";
    for (std::size_t i = 0; i < urls.size(); ++i) out << i << '\t' << urls[i] << '\n';
    return static_cast<bool>(out);
}

// ============================================================================
// 6. Serialización binaria de conteos (para enviarlos por MPI)
// ============================================================================
// Formato de un libro:  u32 indice | u32 n_entradas | n x ( u32 len | bytes | u32 cuenta )
// Se usa el orden de bytes nativo: válido porque todos los procesos MPI de una
// misma ejecución corren en máquinas con la misma arquitectura.

inline void put_u32(std::vector<char>& buf, std::uint32_t v) {
    char tmp[4];
    std::memcpy(tmp, &v, 4);
    buf.insert(buf.end(), tmp, tmp + 4);
}

/// Agrega al buffer el conteo del libro `index`.
inline void pack_book(std::vector<char>& buf, std::uint32_t index, const Counts& c) {
    put_u32(buf, index);
    put_u32(buf, static_cast<std::uint32_t>(c.size()));
    for (const auto& kv : c) {
        put_u32(buf, static_cast<std::uint32_t>(kv.first.size()));
        buf.insert(buf.end(), kv.first.begin(), kv.first.end());
        put_u32(buf, kv.second);
    }
}

/// Recorre un buffer con N libros empaquetados y rellena books[index].
/// Valida límites (un buffer corrupto devuelve false, nunca lee fuera de rango).
inline bool unpack_books(const char* data, std::size_t size, std::vector<Counts>& books,
                         std::string& err) {
    std::size_t off = 0;
    auto get_u32 = [&](std::uint32_t& v) -> bool {
        if (size - off < 4) return false;
        std::memcpy(&v, data + off, 4);
        off += 4;
        return true;
    };
    while (off < size) {
        std::uint32_t idx = 0, n = 0;
        if (!get_u32(idx) || !get_u32(n)) { err = "buffer truncado (cabecera)"; return false; }
        if (idx >= books.size())          { err = "índice de libro fuera de rango"; return false; }
        Counts& c = books[idx];
        c.reserve(n);
        for (std::uint32_t e = 0; e < n; ++e) {
            std::uint32_t len = 0, cnt = 0;
            if (!get_u32(len) || size - off < len) { err = "buffer truncado (palabra)"; return false; }
            std::string w(data + off, len);
            off += len;
            if (!get_u32(cnt))                     { err = "buffer truncado (cuenta)"; return false; }
            c[std::move(w)] += cnt;
        }
    }
    return true;
}

// ============================================================================
// 7. Versión SERIAL completa
// ============================================================================

/**
 * Procesa los k libros uno tras otro en un solo proceso: descarga -> cuenta ->
 * (al final) construye la matriz. Es la línea base del speed-up.
 */
inline bool run_serial(const std::vector<std::string>& urls, bool remove_stopwords,
                       Matrix& out, Timing& t, std::string& err) {
    const double t0 = now_sec();
    std::vector<Counts> books(urls.size());
    std::string text;
    text.reserve(1 << 20);
    for (std::size_t i = 0; i < urls.size(); ++i) {
        std::string e;
        const double a = now_sec();
        if (!fetch_text(urls[i], text, e)) {
            err = "libro " + std::to_string(i) + " (" + urls[i] + "): " + e;
            return false;
        }
        const double b = now_sec();
        count_words(text, books[i], remove_stopwords);
        const double c = now_sec();
        t.download += b - a;
        t.count    += c - b;
    }
    const double d = now_sec();
    out        = build_matrix(books);
    t.assemble = now_sec() - d;
    t.total    = now_sec() - t0;
    return true;
}

}  // namespace bow
#endif  // BOW_CORE_HPP
