#include "openfhe.h"
#include "scheme/ckksrns/ckksrns-fhe.h"
#include "typedef.h"
#include <cassert>
#include <chrono>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>

#include "omp.h"
#define NUM_THREADS 16

// #pragma omp parallel for num_threads(NUM_THREADS)
// int tid = omp_get_thread_num();

using namespace lbcrypto;

// TODO: set conditions so the user does not need to know the logic of the
// following parameters

int setting = 0;  // 0 if 8x8 or 16x16, 1 to perform the "trick" with blocks of
                  // 6x6 or 14x14
int run_conv = 0; // run conv_parallel_encrypted if 1
int run_inv = 0;  // run pixel-wise inversion if 1
size_t partition = 8;      // can be 8 or 16
size_t conv_partition = 6; // 6 if partition = 8 and 14 if partition = 16

class Benchmark {
public:
  Benchmark() { start_point = std::chrono::high_resolution_clock::now(); }
  ~Benchmark() { Stop(); }
  void Stop() {
    std::chrono::time_point<std::chrono::high_resolution_clock> end_point =
        std::chrono::high_resolution_clock::now();

    auto start = std::chrono::time_point_cast<std::chrono::seconds>(start_point)
                     .time_since_epoch()
                     .count();
    auto end = std::chrono::time_point_cast<std::chrono::seconds>(end_point)
                   .time_since_epoch()
                   .count();
    std::cout << " " << (end - start) << " seconds" << std::endl;
  }

private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_point;
};

vec2d_t<float> Q50_8 = {
    {16, 11, 10, 16, 24, 40, 51, 61},     {12, 12, 14, 19, 26, 58, 60, 55},
    {14, 13, 16, 24, 40, 57, 69, 56},     {14, 17, 22, 29, 51, 87, 80, 62},
    {18, 22, 37, 56, 68, 109, 103, 77},   {24, 35, 55, 64, 81, 104, 113, 92},
    {49, 64, 78, 87, 103, 121, 120, 101}, {72, 92, 95, 98, 112, 100, 103, 99}};

vec2d_t<float> Q50_16 = {
    {7, 7, 7, 7, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {7, 7, 7, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 17},
    {7, 7, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 17, 18},
    {7, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 17, 18, 20},
    {7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 17, 18, 20, 22},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 17, 18, 20, 22, 24},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 17, 18, 20, 22, 24, 26},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 17, 18, 20, 22, 24, 26, 28},
    {1, 1, 1, 1, 1, 1, 1, 1, 17, 18, 20, 22, 24, 26, 28, 30},
    {1, 1, 1, 1, 1, 1, 1, 17, 18, 20, 22, 24, 26, 28, 30, 33},
    {1, 1, 1, 1, 1, 1, 17, 18, 20, 22, 24, 26, 28, 30, 33, 36},
    {1, 1, 1, 1, 1, 17, 18, 20, 22, 24, 26, 28, 30, 33, 36, 39},
    {1, 1, 1, 1, 17, 18, 20, 22, 24, 26, 28, 30, 33, 36, 39, 42},
    {1, 1, 1, 17, 18, 20, 22, 24, 26, 28, 30, 33, 36, 39, 42, 45},
    {1, 1, 17, 18, 20, 22, 24, 26, 28, 30, 33, 36, 39, 42, 45, 49},
    {1, 17, 18, 20, 22, 24, 26, 28, 30, 33, 36, 39, 42, 45, 49, 52}};

vec2d_t<float> Q50;

void setQMatrix(vec2d_t<float> Q50_p) {
  for (size_t i = 0; i < Q50_p.size(); i++) {
    vec1d_t<float> aux;
    for (size_t j = 0; j < Q50_p[i].size(); j++) {
      aux.push_back(Q50_p[i][j]);
    }
    Q50.push_back(aux);
  }
}

template <typename T> bool eq_vec1d(const vec1d_t<T> &a, const vec1d_t<T> &b) {
  if (a.size() == b.size()) {
    for (size_t i = 0; i < a.size(); i++) {
      if (a[i] != b[i]) {
        return false;
      }
      continue;
    }
    return true;
  } else {
    return false;
  }
}

template <typename T> bool eq_vec2d(const vec2d_t<T> &a, const vec2d_t<T> &b) {
  if (a.size() == b.size()) {
    for (size_t i = 0; i < a.size(); i++) {
      if (!eq_vec1d(a[i], b[i])) {
        return false;
      }
      continue;
    }
    return true;
  } else {
    return false;
  }
}

template <typename T> bool eq_vec3d(const vec3d_t<T> &a, const vec3d_t<T> &b) {
  if (a.size() == b.size()) {
    for (size_t i = 0; i < a.size(); i++) {
      if (!eq_vec2d(a[i], b[i])) {
        return false;
      }
      continue;
    }
    return true;
  } else {
    return false;
  }
}

template <typename T> void print_vec1d(const vec1d_t<T> &v) {
  std::cout << "[";
  for (const auto &i : v) {
    std::cout << std::to_string(i) << ", ";
  }
  std::cout << "]\n";
}
template <typename T> void print_vec2d(const vec2d_t<T> &v) {
  std::cout << "[\n";
  for (const auto &i : v) {
    std::cout << "\t";
    print_vec1d(i);
    std::cout << ",";
  }
  std::cout << "]\n";
}

template <typename T> void print_vec3d(const vec3d_t<T> &v) {
  std::cout << "[\n";
  for (const auto &i : v) {
    std::cout << "\t";
    print_vec2d(i);
    std::cout << ",";
  }
  std::cout << "]\n";
}

template <typename T> vec2d_t<T> transpose(vec2d_t<T> &a) {
  vec2d_t<T> result(a.size(), vec1d_t<T>(a[0].size()));

  for (size_t i = 0; i < a.size(); i++)
    for (size_t j = 0; j < a[0].size(); j++)
      result[i][j] = a[j][i];

  return result;
}

template <typename T> vec2d_t<T> naive_mul_matrix(vec2d_t<T> a, vec2d_t<T> b) {

  size_t s = a.size();
  vec2d_t<T> result(s, vec1d_t<T>(s, 0));

  for (size_t i = 0; i < s; i++) {
    for (size_t j = 0; j < s; j++) {
      for (size_t k = 0; k < s; k++) {
        result[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  return result;
}
template <typename T> vec2d_t<T> mat_add(vec2d_t<T> &a, vec2d_t<T> &b) {
  size_t N = a.size();
  vec2d_t<T> res(N, vec1d_t<T>(N, 0));
  for (size_t i = 0; i < N; i++)
    for (size_t j = 0; j < N; j++)
      res[i][j] = a[i][j] + b[i][j];
  return res;
}
template <typename T> vec2d_t<T> mat_sub(vec2d_t<T> &a, vec2d_t<T> &b) {
  size_t N = a.size();
  vec2d_t<T> res(N, vec1d_t<T>(N, 0));
  for (size_t i = 0; i < N; i++)
    for (size_t j = 0; j < N; j++)
      res[i][j] = a[i][j] - b[i][j];
  return res;
}
template <typename T> vec2d_t<T> mat_mul(vec2d_t<T> &a, vec2d_t<T> &b) {
  size_t N = a.size();
  if (N == 1)
    return vec2d_t<T>(1, vec1d_t<T>(1, a[0][0] * b[0][0]));
  else {
    size_t N_2 = N / 2;
    vec2d_t<T> a00(N_2, vec1d_t<T>(N_2, 0));
    vec2d_t<T> a01(N_2, vec1d_t<T>(N_2, 0));
    vec2d_t<T> a10(N_2, vec1d_t<T>(N_2, 0));
    vec2d_t<T> a11(N_2, vec1d_t<T>(N_2, 0));

    vec2d_t<T> b00(N_2, vec1d_t<T>(N_2, 0));
    vec2d_t<T> b01(N_2, vec1d_t<T>(N_2, 0));
    vec2d_t<T> b10(N_2, vec1d_t<T>(N_2, 0));
    vec2d_t<T> b11(N_2, vec1d_t<T>(N_2, 0));

    for (size_t i = 0; i < N_2; i++) {
      std::move(a[i].begin(), a[i].begin() + N_2, a00[i].begin());
      std::move(a[i].begin() + N_2, a[i].end(), a01[i].begin());
      std::move(a[i + N_2].begin(), a[i + N_2].begin() + N_2, a10[i].begin());
      std::move(a[i + N_2].begin() + N_2, a[i + N_2].end(), a11[i].begin());
      a[i].clear();

      std::move(b[i].begin(), b[i].begin() + N_2, b00[i].begin());
      std::move(b[i].begin() + N_2, b[i].end(), b01[i].begin());
      std::move(b[i + N_2].begin(), b[i + N_2].begin() + N_2, b10[i].begin());
      std::move(b[i + N_2].begin() + N_2, b[i + N_2].end(), b11[i].begin());
      b[i].clear();
    }

    vec2d_t<T> buffer_0 = mat_sub(b01, b11);

    vec2d_t<T> p(mat_mul(a00, buffer_0));
    buffer_0 = mat_add(a00, a01);
    vec2d_t<T> q = mat_mul(buffer_0, b11);
    buffer_0 = mat_add(a10, a11);
    vec2d_t<T> r = mat_mul(buffer_0, b00);
    buffer_0 = mat_sub(b10, b00);
    vec2d_t<T> s = mat_mul(a11, buffer_0);
    buffer_0 = mat_add(a00, a11);
    vec2d_t<T> buffer_1 = mat_add(b00, b11);
    vec2d_t<T> t = mat_mul(buffer_0, buffer_1);

    buffer_0 = mat_sub(a01, a11);
    a01.clear();
    buffer_1 = mat_add(b10, b11);
    a11.clear();
    b10.clear();
    b11.clear();

    vec2d_t<T> u = mat_mul(buffer_0, buffer_1);

    buffer_0 = mat_sub(a00, a10);
    a00.clear();
    a10.clear();
    buffer_1 = mat_add(b00, b01);
    b00.clear();
    b01.clear();

    vec2d_t<T> v = mat_mul(buffer_0, buffer_1);

    buffer_0 = mat_add(t, s);
    buffer_0 = mat_add(buffer_0, u);
    u.clear();
    vec2d_t<T> r00 = mat_sub(buffer_0, q);
    vec2d_t<T> r01 = mat_add(p, q);
    q.clear();
    vec2d_t<T> r10 = mat_add(r, s);
    s.clear();
    buffer_0 = mat_add(t, p);
    t.clear();
    p.clear();
    buffer_0 = mat_sub(buffer_0, r);
    r.clear();
    vec2d_t<T> r11 = mat_sub(buffer_0, v);
    v.clear();

    vec2d_t<T> res(N, vec1d_t<T>(N, 0));
    for (size_t i = 0; i < N_2; i++) {
      std::move(r00[i].begin(), r00[i].end(), res[i].begin());
      r00[i].clear();
      std::move(r01[i].begin(), r01[i].end(), res[i].begin() + N_2);
      r01[i].clear();
      std::move(r10[i].begin(), r10[i].end(), res[i + N_2].begin());
      r10[i].clear();
      std::move(r11[i].begin(), r11[i].end(), res[i + N_2].begin() + N_2);
      r11[i].clear();
    }
    return res;
  }
}

template <typename T>
vec2d_t<ciphertext_t> mat_add_encrypted_one(vec2d_t<ciphertext_t> &a,
                                            vec2d_t<T> &b, cryptocontext_t cc) {
  size_t N = a.size();
  vec2d_t<ciphertext_t> res =
      vec2d_t<ciphertext_t>(N, vec1d_t<ciphertext_t>(N));
  for (size_t i = 0; i < N; i++)
    for (size_t j = 0; j < N; j++)
      res[i][j] = cc->EvalAdd(a[i][j], b[i][j]);
  return res;
}

vec2d_t<ciphertext_t> mat_add_encrypted_both(vec2d_t<ciphertext_t> &a,
                                             vec2d_t<ciphertext_t> &b,
                                             cryptocontext_t cc) {
  size_t N = a.size();
  vec2d_t<ciphertext_t> res =
      vec2d_t<ciphertext_t>(N, vec1d_t<ciphertext_t>(N));
  for (size_t i = 0; i < N; i++)
    for (size_t j = 0; j < N; j++)
      res[i][j] = cc->EvalAdd(a[i][j], b[i][j]);
  return res;
}

vec2d_t<ciphertext_t> mat_sub_encrypted(vec2d_t<ciphertext_t> &a,
                                        vec2d_t<ciphertext_t> &b,
                                        cryptocontext_t cc) {
  size_t N = a.size();
  vec2d_t<ciphertext_t> res(N, vec1d_t<ciphertext_t>(N));
  for (size_t i = 0; i < N; i++)
    for (size_t j = 0; j < N; j++)
      res[i][j] = cc->EvalSub(a[i][j], b[i][j]);
  return res;
}

void mat_sub_encrypted_in_place(vec2d_t<ciphertext_t> &a,
                                vec2d_t<ciphertext_t> &b, cryptocontext_t cc) {
  size_t N = a.size();
  for (size_t i = 0; i < N; i++)
    for (size_t j = 0; j < N; j++)
      cc->EvalSubInPlace(a[i][j], b[i][j]);
}

template <typename T>
vec2d_t<ciphertext_t> mat_mul_encrypted_one(vec2d_t<ciphertext_t> &a,
                                            vec2d_t<T> &b, cryptocontext_t cc) {
  size_t N = a.size();
  if (N == 1) {
    cc->EvalMultInPlace(a[0][0], b[0][0]);
    return a;
  } else {
    size_t N_2 = N / 2;
    vec2d_t<ciphertext_t> a00(N_2, vec1d_t<ciphertext_t>(N_2));
    vec2d_t<ciphertext_t> a01(N_2, vec1d_t<ciphertext_t>(N_2));
    vec2d_t<ciphertext_t> a10(N_2, vec1d_t<ciphertext_t>(N_2));
    vec2d_t<ciphertext_t> a11(N_2, vec1d_t<ciphertext_t>(N_2));

    vec2d_t<T> b00(N_2, vec1d_t<T>(N_2, 0));
    vec2d_t<T> b01(N_2, vec1d_t<T>(N_2, 0));
    vec2d_t<T> b10(N_2, vec1d_t<T>(N_2, 0));
    vec2d_t<T> b11(N_2, vec1d_t<T>(N_2, 0));

    for (size_t i = 0; i < N_2; i++) {
      std::move(a[i].begin(), a[i].begin() + N_2, a00[i].begin());
      std::move(a[i].begin() + N_2, a[i].end(), a01[i].begin());
      std::move(a[i + N_2].begin(), a[i + N_2].begin() + N_2, a10[i].begin());
      std::move(a[i + N_2].begin() + N_2, a[i + N_2].end(), a11[i].begin());
      a[i].clear();

      std::move(b[i].begin(), b[i].begin() + N_2, b00[i].begin());
      std::move(b[i].begin() + N_2, b[i].end(), b01[i].begin());
      std::move(b[i + N_2].begin(), b[i + N_2].begin() + N_2, b10[i].begin());
      std::move(b[i + N_2].begin() + N_2, b[i + N_2].end(), b11[i].begin());
      b[i].clear();
    }

    vec2d_t<T> tmp = mat_sub(b01, b11);
    vec2d_t<ciphertext_t> p = mat_mul_encrypted_one(a00, tmp, cc);
    vec2d_t<ciphertext_t> buffer = mat_add_encrypted_both(a00, a01, cc);
    vec2d_t<ciphertext_t> q = mat_mul_encrypted_one(buffer, b11, cc);
    buffer = mat_add_encrypted_both(a10, a11, cc);
    vec2d_t<ciphertext_t> r = mat_mul_encrypted_one(buffer, b00, cc);
    tmp = mat_sub(b10, b00);
    vec2d_t<ciphertext_t> s = mat_mul_encrypted_one(a11, tmp, cc);
    buffer = mat_add_encrypted_both(a00, a11, cc);
    tmp = mat_add(b00, b11);
    vec2d_t<ciphertext_t> t = mat_mul_encrypted_one(buffer, tmp, cc);

    mat_sub_encrypted_in_place(a01, a11, cc);
    a11.clear();
    tmp = mat_add(b10, b11);
    vec2d_t<ciphertext_t> u = mat_mul_encrypted_one(a01, tmp, cc);
    a01.clear();
    b10.clear();
    b11.clear();

    mat_sub_encrypted_in_place(a00, a10, cc);

    a10.clear();
    tmp = mat_add(b00, b01);
    vec2d_t<ciphertext_t> v = mat_mul_encrypted_one(a00, tmp, cc);
    a00.clear();

    b00.clear();
    b01.clear();

    buffer = mat_add_encrypted_both(t, s, cc);
    buffer = mat_add_encrypted_both(buffer, u, cc);
    u.clear();
    vec2d_t<ciphertext_t> r00 = mat_sub_encrypted(buffer, q, cc);
    vec2d_t<ciphertext_t> r01 = mat_add_encrypted_both(p, q, cc);
    q.clear();
    vec2d_t<ciphertext_t> r10 = mat_add_encrypted_both(r, s, cc);
    s.clear();
    buffer = mat_add_encrypted_both(t, p, cc);
    t.clear();
    p.clear();
    buffer = mat_sub_encrypted(buffer, r, cc);
    r.clear();
    vec2d_t<ciphertext_t> r11 = mat_sub_encrypted(buffer, v, cc);
    v.clear();
    buffer.clear();

    vec2d_t<ciphertext_t> res =
        vec2d_t<ciphertext_t>(N, vec1d_t<ciphertext_t>(N));
    for (size_t i = 0; i < N_2; i++) {
      std::move(r00[i].begin(), r00[i].end(), res[i].begin());
      r00[i].clear();
      std::move(r01[i].begin(), r01[i].end(), res[i].begin() + N_2);
      r01[i].clear();
      std::move(r10[i].begin(), r10[i].end(), res[i + N_2].begin());
      r10[i].clear();
      std::move(r11[i].begin(), r11[i].end(), res[i + N_2].begin() + N_2);
      r11[i].clear();
    }
    return res;
  }
}

vec2d_t<ciphertext_t> mat_mul_encrypted_both(vec2d_t<ciphertext_t> &a,
                                             vec2d_t<ciphertext_t> &b,
                                             cryptocontext_t cc) {
  size_t N = a.size();
  if (N == 1) {
    return vec2d_t<ciphertext_t>(
        1, vec1d_t<ciphertext_t>(1, cc->EvalMult(a[0][0], b[0][0])));
  } else {
    size_t N_2 = N / 2;
    vec2d_t<ciphertext_t> a00(N_2, vec1d_t<ciphertext_t>(N_2));
    vec2d_t<ciphertext_t> a01(N_2, vec1d_t<ciphertext_t>(N_2));
    vec2d_t<ciphertext_t> a10(N_2, vec1d_t<ciphertext_t>(N_2));
    vec2d_t<ciphertext_t> a11(N_2, vec1d_t<ciphertext_t>(N_2));

    vec2d_t<ciphertext_t> b00(N_2, vec1d_t<ciphertext_t>(N_2, 0));
    vec2d_t<ciphertext_t> b01(N_2, vec1d_t<ciphertext_t>(N_2, 0));
    vec2d_t<ciphertext_t> b10(N_2, vec1d_t<ciphertext_t>(N_2, 0));
    vec2d_t<ciphertext_t> b11(N_2, vec1d_t<ciphertext_t>(N_2, 0));

    for (size_t i = 0; i < N_2; i++) {
      std::move(a[i].begin(), a[i].begin() + N_2, a00[i].begin());
      std::move(a[i].begin() + N_2, a[i].end(), a01[i].begin());
      std::move(a[i + N_2].begin(), a[i + N_2].begin() + N_2, a10[i].begin());
      std::move(a[i + N_2].begin() + N_2, a[i + N_2].end(), a11[i].begin());
      a[i].clear();

      std::move(b[i].begin(), b[i].begin() + N_2, b00[i].begin());
      std::move(b[i].begin() + N_2, b[i].end(), b01[i].begin());
      std::move(b[i + N_2].begin(), b[i + N_2].begin() + N_2, b10[i].begin());
      std::move(b[i + N_2].begin() + N_2, b[i + N_2].end(), b11[i].begin());
      b[i].clear();
    }

    vec2d_t<ciphertext_t> buffer_0 = mat_sub_encrypted(b01, b11, cc);

    vec2d_t<ciphertext_t> p(mat_mul_encrypted_both(a00, buffer_0, cc));
    buffer_0 = mat_add_encrypted_both(a00, a01, cc);
    vec2d_t<ciphertext_t> q = mat_mul_encrypted_both(buffer_0, b11, cc);
    buffer_0 = mat_add_encrypted_both(a10, a11, cc);
    vec2d_t<ciphertext_t> r = mat_mul_encrypted_both(buffer_0, b00, cc);
    buffer_0 = mat_sub_encrypted(b10, b00, cc);
    vec2d_t<ciphertext_t> s = mat_mul_encrypted_both(a11, buffer_0, cc);
    buffer_0 = mat_add_encrypted_both(a00, a11, cc);
    vec2d_t<ciphertext_t> buffer_1 = mat_add_encrypted_both(b00, b11, cc);
    vec2d_t<ciphertext_t> t = mat_mul_encrypted_both(buffer_0, buffer_1, cc);

    buffer_0 = mat_sub_encrypted(a01, a11, cc);
    a01.clear();
    buffer_1 = mat_add_encrypted_both(b10, b11, cc);
    a11.clear();
    b10.clear();
    b11.clear();

    vec2d_t<ciphertext_t> u = mat_mul_encrypted_both(buffer_0, buffer_1, cc);

    buffer_0 = mat_sub_encrypted(a00, a10, cc);
    a00.clear();
    a10.clear();
    buffer_1 = mat_add_encrypted_both(b00, b01, cc);
    b00.clear();
    b01.clear();

    vec2d_t<ciphertext_t> v = mat_mul_encrypted_both(buffer_0, buffer_1, cc);

    buffer_0 = mat_add_encrypted_both(t, s, cc);
    buffer_0 = mat_add_encrypted_both(buffer_0, u, cc);
    u.clear();
    vec2d_t<ciphertext_t> r00 = mat_sub_encrypted(buffer_0, q, cc);
    vec2d_t<ciphertext_t> r01 = mat_add_encrypted_both(p, q, cc);
    q.clear();
    vec2d_t<ciphertext_t> r10 = mat_add_encrypted_both(r, s, cc);
    s.clear();
    buffer_0 = mat_add_encrypted_both(t, p, cc);
    t.clear();
    p.clear();
    buffer_0 = mat_sub_encrypted(buffer_0, r, cc);
    r.clear();
    vec2d_t<ciphertext_t> r11 = mat_sub_encrypted(buffer_0, v, cc);
    v.clear();

    vec2d_t<ciphertext_t> res(N, vec1d_t<ciphertext_t>(N, 0));
    for (size_t i = 0; i < N_2; i++) {
      std::move(r00[i].begin(), r00[i].end(), res[i].begin());
      r00[i].clear();
      std::move(r01[i].begin(), r01[i].end(), res[i].begin() + N_2);
      r01[i].clear();
      std::move(r10[i].begin(), r10[i].end(), res[i + N_2].begin());
      r10[i].clear();
      std::move(r11[i].begin(), r11[i].end(), res[i + N_2].begin() + N_2);
      r11[i].clear();
    }
    return res;
  }
}

bool test_mat_mul() {
  vec2d_t<size_t> a = {{1, 1, 1, 1}, {2, 23, 2, 2}, {3, 3, 3, 3}, {2, 2, 2, 2}};
  vec2d_t<size_t> b = {{1, 1, 1, 1}, {2, 2, 4, 2}, {3, 3, 3, 3}, {2, 2, 2, 2}};
  vec2d_t<size_t> res2(naive_mul_matrix(a, b));
  vec2d_t<size_t> res(mat_mul(a, b));

  print_vec2d(res);
  print_vec2d(res2);
  return eq_vec2d(res, res2);
}

// input here has to be 16-bit signed integer (int16_t)
template <typename T> void level(vec2d_t<T> &img) {
  for (size_t i = 0; i < img.size(); i++) {
    for (size_t j = 0; j < img[i].size(); j++) {
      img[i][j] -= 128;
    }
  }
}

void bench_level(size_t t) {
  vec2d_t<uint8_t> test = {{255, 255, 255, 255},
                           {255, 255, 254, 255},
                           {255, 255, 255, 255},
                           {255, 255, 255, 255}};
  vec2d_t<uint8_t> result;
  auto t0 = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < t; i++) {
    level(test);
  }
  auto t1 = std::chrono::high_resolution_clock::now();

  std::cout
      << "level (" << t << ") iterations: "
      << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
      << "ms\n";
}

// input signed 16-bit integers; output unsigned 16-bit integers
// because we're operating in place im going to use unsigned 16-bit
// img mutated
template <typename T> void unlevel(vec2d_t<T> &img) {
  for (size_t i = 0; i < img.size(); i++) {
    for (size_t j = 0; j < img[i].size(); j++) {
      img[i][j] += 128;
    }
  }
}

template <typename T> void unlevel_rounded(vec2d_t<T> &img) {
#pragma omp parallel for num_threads(NUM_THREADS)
  for (size_t i = 0; i < img.size(); i++) {
    for (size_t j = 0; j < img[i].size(); j++) {
      img[i][j] += 128;
      img[i][j] = std::round(img[i][j]);
    }
  }
}

// img mutated
template <typename T> void level_blocks(vec3d_t<T> &img) {
  for (size_t i = 0; i < img.size(); i++) {
    level(img[i]);
  }
}

// img mutated
template <typename T> void unlevel_blocks(vec3d_t<T> &img) {
  for (size_t i = 0; i < img.size(); i++) {
    unlevel(img[i]);
  }
}

template <typename T> vec3d_t<T> img_to_blocks(vec2d_t<T> &img) {
  vec3d_t<T> blocks;

  const size_t H = img.size();
  const size_t L = img[0].size();
  const size_t l = std::ceil(float(L) / partition);
  const size_t h = std::ceil(float(H) / partition);

  if (l != L / partition || h != H / partition) {
    std::cout << "IMAGE SIZE IS NOT A MULTIPLE OF " << partition << std::endl;
    return blocks;
  }

  for (size_t i = 0; i < l; i++) {
    for (size_t j = 0; j < h; j++) {
      vec2d_t<T> block(partition, vec1d_t<T>(partition));

      for (size_t k = 0; k < partition; k++) {
        for (size_t m = 0; m < partition; m++) {
          block[k][m] = img[partition * j + k][partition * i + m];
        }
      }

      blocks.push_back(block);
    }
  }

  return blocks;
}

template <typename T>
vec2d_t<T> img_from_blocks(const vec3d_t<T> &blocks, size_t bpr) {
  const size_t LR = bpr * partition;
  const size_t LC = int(blocks.size() / bpr * partition);

  vec2d_t<T> img(LC, vec1d_t<T>(LR, 0.0));

  for (size_t j = 0; j < LC; j++) {
    for (size_t i = 0; i < LR; i++) {
      img[j][i] = blocks[floor(i / partition) + floor(j / partition) * bpr]
                        [j % partition][i % partition];
    }
  }
  return img;
}

// img unusable after this
template <typename T> vec3d_t<T> img_to_6x6_blocks(vec2d_t<T> &img) {
  const size_t H = img.size();
  const size_t L = img[0].size();
  const size_t l = floor(float(L) / conv_partition);
  const size_t h = floor(float(H) / conv_partition);

  // std::cout << "H: " << H << " L: " << L << " l: " << l << " h: " << h <<
  //"\n"; std::cout << h * l << "\n";

  vec3d_t<T> blocks(h * l, vec2d_t<T>(partition, vec1d_t<T>(partition)));

  if ((l * conv_partition != L) || (h * conv_partition != H)) {
    std::cout << "Image size is not a multiple of " << conv_partition;
    return blocks;
  }

  // Goal:
  //  - duplicate second and second-last elements of dimension 1 into 1st and
  //  last positions
  // [[1,2,3], [2,3,4], [4,2,1]] -> [[1,2,3], [1,2,3], [2,3,4], [4,2,1],
  // [4,2,1]]
  //  - do the same for elements in dimension 2
  // -> [[1,1,2,3,3], [1,1,2,3,3], [2,2,3,4,4], [4,4,2,1,1], [4,4,2,1,1]]

  vec2d_t<T> padded_img(H + 2, vec1d_t<T>(L + 2));

  // for (size_t i = 0; i < L; i++) {
  //   std::move(img[i].begin(), img[i].end(), (padded_img[i + 1].begin() + 1));
  //   padded_img[i + 1][0] = padded_img[i + 1][1];
  //   padded_img[i + 1][L + 1] = padded_img[i + 1][L];
  // }
  for (size_t i = 0; i < H; i++) {
    for (size_t j = 0; j < L; j++) {
      padded_img[i + 1][j + 1] = img[i][j];
    }
  }

  for (size_t i = 1; i < H + 2; i++) {
    padded_img[i][0] = padded_img[i][1];
    padded_img[i][L + 1] = padded_img[i][L];
  }

  // // we could do this copy at the end
  padded_img[0] = padded_img[1];
  padded_img[H + 1] = padded_img[H];

  // for (size_t j = 0; j < h; j++) {
  //   for (size_t i = 0; i < l; i++) {
  //     for (size_t k = 0; k < partition; k++) {
  //       for (size_t m = 0; m < partition; m++) {
  //         blocks[(2 * j) + i][k][m] = (padded_img[conv_partition * j +
  //         k][conv_partition * i + m]);
  //       }
  //     }
  //   }
  // }

  for (size_t j = 0; j < h; ++j) {
    for (size_t i = 0; i < l; ++i) {
      for (size_t k = 0; k < partition; ++k) {
        for (size_t m = 0; m < partition; ++m) {
          blocks[j * l + i][k][m] =
              padded_img[conv_partition * j + k][conv_partition * i + m];
        }
      }
    }
  }

  return blocks;
}

template <typename T>
vec2d_t<T> img_from_6x6_blocks(const vec3d_t<T> &blocks, size_t bpr) {
  const size_t LR = bpr * conv_partition;
  const size_t LC = int(blocks.size() / bpr * conv_partition);

  // print_vec3d(blocks);
  // std::cout << "printed blocks" << LR << " " << LC << " " << blocks.size() <<
  // " " << bpr << std::endl; std::cout << blocks.size() << " " <<
  // blocks[0].size() << " " << blocks[0][0].size() << std::endl;

  vec2d_t<T> img(LC, vec1d_t<T>(LR, 0.0));

  std::cout << img.size() << " " << img[0].size() << std::endl;

  for (size_t j = 0; j < LC; j++) {
    for (size_t i = 0; i < LR; i++) {
      img[j][i] =
          blocks[floor(i / conv_partition) + floor(j / conv_partition) * bpr]
                [1 + j % conv_partition][1 + i % conv_partition];
    }
  }

  return img;
}

// this should return decimals
template <typename T> vec2d_t<T> get_dct_table(size_t N) {
  vec2d_t<T> table(N, vec1d_t<T>(N));
  table[0] = vec1d_t<T>(N, 1.0 / sqrt(N));
  for (size_t i = 1; i < N; i++) {
    for (size_t k = 0; k < N; k++) {
      table[i][k] = sqrt(2.0 / N) * cos((2 * k + 1) * i * M_PI / (2 * N));
      // std::cout << "table: " << table[i][k] << std::endl;
    }
  }
  return table;
}

template <typename D, typename I> vec3d_t<D> dct_blocks(const vec3d_t<I> &img) {
  vec3d_t<D> result =
      vec3d_t<D>(img.size(), vec2d_t<D>(partition, vec1d_t<D>(partition, 0.0)));
  vec2d_t<D> DCT = get_dct_table<D>(partition);

  for (size_t x = 0; x < img.size(); x++) {

    // Can we do this in one loop?
    vec2d_t<D> aux(partition, vec1d_t<D>(partition, 0.0));
    for (size_t i = 0; i < partition; i++) {
      for (size_t j = 0; j < partition; j++) {
        for (size_t k = 0; k < partition; k++) {
          aux[i][j] += DCT[i][k] * img[x][k][j];
        }
      }
    }

    for (size_t i = 0; i < partition; i++) {
      for (size_t j = 0; j < partition; j++) {
        for (size_t k = 0; k < partition; k++) {
          result[x][i][j] += aux[i][k] * DCT[j][k];
        }
      }
    }
  }

  return result;
}
template <typename D, typename I>
vec3d_t<D> idct_blocks(const vec3d_t<I> &img) {
  vec3d_t<D> result =
      vec3d_t<D>(img.size(), vec2d_t<D>(partition, vec1d_t<D>(partition, 0.0)));
  vec2d_t<D> DCT = get_dct_table<D>(partition);

  for (size_t x = 0; x < img.size(); x++) {

    // Can we do this in one loop?
    vec2d_t<D> aux(partition, vec1d_t<D>(partition, 0.0));
    for (size_t i = 0; i < partition; i++) {
      for (size_t j = 0; j < partition; j++) {
        for (size_t k = 0; k < partition; k++) {
          aux[i][j] += DCT[k][i] * img[x][k][j];
        }
      }
    }

    for (size_t i = 0; i < partition; i++) {
      for (size_t j = 0; j < partition; j++) {
        for (size_t k = 0; k < partition; k++) {
          result[x][i][j] += aux[i][k] * DCT[k][j];
        }
      }
    }
  }

  // for(size_t x = 0; x < 3*85 + 1; x++){
  //     print_vec2d(result[x]);
  //     std::cout << img.size() << std::endl;
  // }

  return result;
}

template <typename I, typename D>
vec3d_t<D> _idct_blocks(const vec3d_t<I> &img) {

  vec3d_t<D> result =
      vec3d_t<D>(img.size(), vec2d_t<D>(partition, vec1d_t<D>(partition, 0.0)));

  vec2d_t<D> DCT = get_dct_table<D>(partition);

  for (size_t x = 0; x < img.size(); x++) {
    vec2d_t<D> aux(partition, vec1d_t<D>(partition, 0.0));

    for (size_t i = 0; i < partition; i++) {
      for (size_t j = 0; j < partition; j++) {
        for (size_t k = 0; k < partition; k++) {
          aux[i][j] += DCT[k][i] * img[x][k][j];
        }
      }
    }

    for (size_t i = 0; i < partition; i++) {
      for (size_t j = 0; j < partition; j++) {
        // this stays because we lose precision when we cast
        D sum = 0.0;
        for (size_t k = 0; k < partition; k++) {
          sum += aux[i][k] * DCT[k][j];
        }
        result[x][i][j] = sum;
      }
    }
  }
  return result;
}

template <typename T> vec1d_t<T> flatten(const vec2d_t<T> &matrix) {
  vec1d_t<T> out;
  for (vec1d_t<T> i : matrix) {
    for (T j : i) {
      out.push_back(j);
    }
  }
  return out;
}

template <typename I, typename D> vec3d_t<I> quantize(const vec3d_t<D> &img) {
  vec3d_t<I> result =
      vec3d_t<I>(img.size(), vec2d_t<I>(partition, vec1d_t<I>(partition)));
  for (size_t k = 0; k < img.size(); k++) {
    for (size_t i = 0; i < partition; i++) {
      for (size_t j = 0; j < partition; j++) {
        result[k][i][j] = (round(img[k][i][j] / Q50[i][j]));
      }
    }
  }
  return result;
}

template <typename T> void dequantize(vec3d_t<T> &img) {
  vec3d_t<T> result(img.size(), vec2d_t<T>(partition, vec1d_t<T>(partition)));
  for (size_t k = 0; k < img.size(); k++) {
    for (size_t i = 0; i < partition; i++) {
      for (size_t j = 0; j < partition; j++) {
        img[k][i][j] *= Q50[i][j];
      }
    }
  }
}

template <typename T> vec1d_t<T> zigzag(vec2d_t<T> &block, size_t trim_at = 0) {
  vec1d_t<T> res;
  size_t res_size = trim_at == 0 ? partition * partition : trim_at;
  res.reserve(res_size);

  size_t i = 0;
  size_t j = 0;

  for (size_t k = 0; k < res_size; k++) {
    res.push_back(std::move(block[i][j]));

    if ((i + j) % 2 == 1) {
      if (i == (block.size() - 1)) {
        j += 1;
      } else if (j == 0) {
        i += 1;
      } else {
        i += 1;
        j -= 1;
      }
    } else {
      if (j == block.size() - 1) {
        i += 1;
      } else if (i == 0) {
        j += 1;
      } else {
        i -= 1;
        j += 1;
      }
    }
  }
  return res;
}

// arr unsable after this
template <typename T> vec2d_t<T> unzigzag(vec1d_t<T> &arr, size_t L) {
  vec2d_t<T> res(L, vec1d_t<T>(L, 0));

  size_t i = 0;
  size_t j = 0;

  while ((i != (L - 1) || j != (L - 1)) && arr.size() >= 1) {
    res[i][j] = *arr.begin();
    arr.erase(arr.begin());
    if ((i + j) % 2 == 1) {
      if (i == (L - 1)) {
        j += 1;
      } else if (j == 0) {
        i += 1;
      } else {
        i += 1;
        j -= 1;
      }
    } else {
      if (j == (L - 1)) {
        i += 1;
      } else if (i == 0) {
        j += 1;
      } else {
        i -= 1;
        j += 1;
      }
    }
  }
  if (arr.size() != 0) {
    res[L - 1][L - 1] = arr[0];
    arr.erase(arr.begin());
  }
  return res;
}

template <typename T> vec2d_t<T> compress(vec3d_t<T> &array, size_t nb_elem) {
  vec2d_t<T> compressed_array(array.size(), vec1d_t<T>(nb_elem));
  for (size_t i = 0; i < array.size(); i++) {
    vec1d_t<T> zig = zigzag(array[i]);
    std::move(zig.begin(), zig.begin() + nb_elem, compressed_array[i].begin());
  }

  return compressed_array;
}

// array unusable after this
template <typename T> vec3d_t<T> decompress(vec2d_t<T> &array) {
  vec3d_t<T> result(array.size(), vec2d_t<T>(partition, vec1d_t<T>(partition)));
  for (size_t i = 0; i < array.size(); i++) {
    vec2d_t<T> r = unzigzag(array[i], partition);
    // TODO: remove this if
    if (setting == 0) {
      r = transpose(r);
    }

    std::move(r.begin(), r.end(), result[i].begin());
  }

  // for(size_t x = 0; x < 3*85 + 1; x++){
  //     print_vec2d(result[x]);
  // }

  // print_vec2d(array);

  return result;
}

template <typename I, typename D>
vec2d_t<I> compress_image_6x6_client_debug(vec2d_t<I> &image, size_t nb_elem) {

  // vec2d_t<I> aux;

  level(image);
  vec3d_t<I> blocks = img_to_6x6_blocks(image);
  vec3d_t<D> d = dct_blocks<D, I>(blocks);
  vec3d_t<I> q = quantize<I, D>(d);
  return compress(q, nb_elem);
  // return aux;
}

template <typename I, typename D>
vec2d_t<I> compress_image_6x6_client(vec2d_t<I> &image, size_t nb_elem) {
  level(image);
  vec3d_t<I> quantized =
      quantize<I, D>(dct_blocks<D, I>(img_to_6x6_blocks(image)));
  return compress(quantized, nb_elem);
}

template <typename I, typename D>
vec2d_t<I> compress_image_client(vec2d_t<I> &image, size_t nb_elem) {
  level(image);
  vec3d_t<I> quantized = quantize<I, D>(dct_blocks<D, I>(img_to_blocks(image)));
  return compress(quantized, nb_elem);
}

template <typename D, typename I>
vec3d_t<D> expand_image_6x6_server(vec2d_t<I> &image) {
  vec3d_t<I> decompressed = decompress(image);
  dequantize(decompressed);
  vec3d_t<D> dec_idct = idct_blocks<D, I>(decompressed);
  unlevel_blocks(dec_idct);
  return dec_idct;
}

template <typename I, typename D>
vec2d_t<I> compress_image_6x6_server(vec3d_t<D> &image, size_t nb_elem) {
  level_blocks(image);

  vec3d_t<I> quantized = quantize<I, D>(dct_blocks<D, D>(image));
  return compress(quantized, nb_elem);
}

template <typename D, typename I>
vec2d_t<D> expand_image_6x6_client(vec2d_t<I> image, size_t bpr) {
  vec3d_t<I> dec = decompress(image);
  dequantize(dec);
  vec2d_t<D> img = img_from_6x6_blocks(idct_blocks<D, I>(dec), bpr);
  // print_vec2d(img);
  // std::cout << "printed image" << std::endl;
  unlevel(img);
  return img;
}

template <typename D, typename I>
vec2d_t<D> expand_image_client(vec2d_t<I> image, size_t bpr) {
  vec3d_t<I> dec = decompress(image);
  dequantize(dec);
  vec2d_t<D> img = img_from_blocks(idct_blocks<D, I>(dec), bpr);
  unlevel_rounded(img);
  return img;
}

template <typename T> vec2d_t<T> conv(vec2d_t<T> matrix, vec2d_t<T> filter) {
  const size_t X = matrix.size();
  const size_t Y = matrix[0].size();

  vec2d_t<T> result(X, vec1d_t<T>(Y, 0));

  for (size_t x = 0; x < X; x++) {
    for (size_t y = 0; y < Y; y++) {

      for (size_t i = 0; i < 3; i++) {

        for (size_t j = 0; j < 3; j++) {
          if ((x == 0 && i == 0) || (x == X - 1 && i == 2) ||
              (y == 0 && j == 0) || (y == Y - 1 && j == 2)) {
            continue;
          } else {
            result[x][y] += matrix[x - 1 + i][y - 1 + j] * filter[i][j];
          }
        }
      }
    }
  }
}

template <typename T>
vec3d_t<T> conv_parallel(vec3d_t<T> blocks, vec2d_t<T> filter) {
  vec3d_t<T> result_blocks;
  for (vec2d_t<T> block : blocks) {

    vec2d_t<T> result = block;
    for (size_t x = 1; x < 7; x++) {
      for (size_t y = 1; y < 7; y++) {
        result[x][y] = 0;
        for (size_t i = 0; i < 3; i++) {
          for (size_t j = 0; j < 3; j++) {
            result[x][y] += block[x - 1 + i][y - 1 + j] * filter[i][j];
          }
        }
      }
    }
    for (size_t x = 1; x < 7; x++) {
      result[x][0] = 2 * result[x][1] - result[x][2];
      result[x][7] = 2 * result[x][6] - result[x][5];
    }
    for (size_t i = 0; i < result[1].size(); i++) {
      result[0][i] = (2 * result[1][i]) - result[2][i];
    }
    for (size_t i = 0; i < result[6].size(); i++) {
      result[7][i] = (2 * result[6][i]) - result[5][i];
    }
    result_blocks.push_back(result);
  }
  return result_blocks;
}

template <typename T>
vec2d_t<T> readCSV(const std::string &filename, char delimiter) {
  vec2d_t<T> data;
  std::ifstream file(filename);

  if (!file) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return data;
  }

  std::string line;
  while (std::getline(file, line)) {
    vec1d_t<T> row;
    std::stringstream ss(line);
    std::string cell;

    while (std::getline(ss, cell, delimiter)) {
      row.push_back(std::stoi(cell));
    }

    data.push_back(row);
  }

  file.close();
  return data;
}

template <typename T>
void writeCSV(const std::string &filename, const vec2d_t<T> &data,
              char delimiter) {
  std::ofstream file(filename);

  if (!file) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return;
  }

  for (const auto &row : data) {
    for (auto it = row.begin(); it != row.end(); ++it) {
      file << *it;
      if (std::next(it) != row.end()) {
        file << delimiter;
      }
    }
    file << "\n";
  }

  file.close();
}

void test() {

  vec2d_t<int16_t> tv2d = {
      {48, 46, 44, 45, 47, 44, 50, 56, 70, 90, 138, 162},
      {49, 50, 40, 46, 47, 75, 62, 69, 94, 128, 162, 193},
      {60, 45, 39, 35, 40, 87, 87, 65, 86, 121, 174, 202},
      {79, 48, 37, 35, 38, 55, 90, 65, 50, 72, 163, 204},
      {93, 76, 47, 40, 42, 68, 112, 77, 56, 66, 157, 204},
      {98, 90, 72, 66, 66, 90, 108, 74, 53, 87, 177, 207},
      {84, 98, 84, 84, 91, 83, 72, 57, 66, 126, 197, 209},
      {60, 79, 86, 90, 80, 76, 55, 65, 113, 173, 207, 211},
      {62, 57, 54, 60, 57, 64, 77, 107, 160, 198, 208, 206},
      {95, 82, 63, 65, 75, 88, 127, 158, 188, 202, 203, 202},
      {120, 108, 101, 102, 116, 137, 163, 186, 197, 198, 202, 205},
      {141, 131, 132, 138, 146, 157, 182, 187, 193, 197, 199, 203},
  };

  vec3d_t<int16_t> blocks = img_to_6x6_blocks(tv2d);

  vec2d_t<int16_t> a = compress_image_6x6_client<int16_t, double>(tv2d, 32);

  vec3d_t<double> d = expand_image_6x6_server<double, int16_t>(a);

  vec2d_t<double> F = {{0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 0.0}};

  vec3d_t<double> cp = conv_parallel(d, F);

  vec2d_t<int16_t> cs = compress_image_6x6_server<int16_t, double>(cp, 32);
  vec2d_t<double> img = expand_image_6x6_client<double, int16_t>(cs, 2);

  print_vec2d(img);
}

// ENCRYPTION WORK

vec1d_t<ciphertext_t> encrypt_image(vec2d_t<int16_t> compressed_image,
                                    usint slots, cryptocontext_t cc,
                                    keypair_t keys, int depth) {

  vec1d_t<ciphertext_t> encrypted_image;

  size_t num_blocks = compressed_image.size();
  size_t cutting_point = compressed_image[0].size();

  for (size_t i = 0; i < cutting_point; i++) {

    vec1d_t<double> doubleVec;

    for (size_t j = 0; j < num_blocks; j++) {
      doubleVec.push_back(static_cast<double>(compressed_image[j][i]));
    }

    plaintext_t ptxt =
        cc->MakeCKKSPackedPlaintext(doubleVec, depth, 0, nullptr, slots);
    ciphertext_t encrypted_position = cc->Encrypt(keys.publicKey, ptxt);
    encrypted_image.push_back(encrypted_position);
  }

  return encrypted_image;
}

template <typename T>
vec2d_t<T> decrypt_image(vec1d_t<ciphertext_t> &encrypted_image,
                         usint num_blocks, usint cutting_point,
                         cryptocontext_t cc, keypair_t keys) {

  vec2d_t<T> compressed_image(num_blocks, vec1d_t<T>(cutting_point));

  plaintext_t output;

  for (size_t i = 0; i < cutting_point; i++) {
    cc->Decrypt(keys.secretKey, encrypted_image[i], &output);
    output->SetLength(num_blocks);
    for (size_t j = 0; j < num_blocks; j++) {
      compressed_image[j][i] = real(output->GetCKKSPackedValue()[j]);
    }
  }
  return compressed_image;
}

// vec2d_t<int16_t> decrypt_image_int(vec1d_t<ciphertext_t> encrypted_image,
// usint num_blocks, usint cutting_point, cryptocontext_t cc,
//                                     keypair_t keys) {

//     vec2d_t<int16_t> compressed_image = vec2d_t<int16_t>(num_blocks,
//     vec1d_t<int16_t>(cutting_point));

//     plaintext_t output;

//     for(size_t i = 0; i < cutting_point; i++) {
//           cc->Decrypt(keys.secretKey, encrypted_image[i], &output);
//           output->SetLength(num_blocks);
//           for (size_t j = 0; j < num_blocks; j++) {
//               compressed_image[j][i] =
//               round(real(output->GetCKKSPackedValue()[j]));
//           }
//     }
//     return compressed_image;
// }

// arr unsable after this
vec2d_t<ciphertext_t> unzigzag_encrypted(vec1d_t<ciphertext_t> &arr, size_t L,
                                         usint slots, cryptocontext_t cc,
                                         keypair_t keys, int depth) {
  vec2d_t<ciphertext_t> res =
      vec2d_t<ciphertext_t>(L, vec1d_t<ciphertext_t>(L, 0));

  // We need to init res with encryptions of 0
  vec1d_t<double> x = {0.0};

  for (size_t i = 0; i < res.size(); i++) {
    for (size_t j = 0; j < res[0].size(); j++) {
      plaintext_t ptxt =
          cc->MakeCKKSPackedPlaintext(x, depth, 0, nullptr, slots);
      ciphertext_t ct = cc->Encrypt(keys.publicKey, ptxt);
      res[i][j] = ct;
    }
  }

  size_t i = 0;
  size_t j = 0;

  while ((i != (L - 1) || j != (L - 1)) && arr.size() >= 1) {
    res[i][j] = *arr.begin();
    arr.erase(arr.begin());
    if ((i + j) % 2 == 1) {
      if (i == (L - 1)) {
        j += 1;
      } else if (j == 0) {
        i += 1;
      } else {
        i += 1;
        j -= 1;
      }
    } else {
      if (j == (L - 1)) {
        i += 1;
      } else if (i == 0) {
        j += 1;
      } else {
        i -= 1;
        j += 1;
      }
    }
  }
  if (arr.size() != 0) {
    res[L - 1][L - 1] = arr[0];
    arr.erase(arr.begin());
  }
  return res;
}

vec2d_t<ciphertext_t>
decompress_encrypted(vec1d_t<ciphertext_t> &encrypted_image, usint slots,
                     cryptocontext_t cc, keypair_t keys, int depth) {
  return unzigzag_encrypted(encrypted_image, partition, slots, cc, keys, depth);
}

vec1d_t<ciphertext_t> compress_encrypted(vec2d_t<ciphertext_t> &array,
                                         size_t nb_elem) {
  return zigzag(array, nb_elem);
}

// TODO: we could unify quantize and dequantize
void quantize_encrypted(vec2d_t<ciphertext_t> &img, cryptocontext_t cc,
                        vec2d_t<double> q50_div) {
  for (size_t i = 0; i < img.size(); i++) {
    for (size_t j = 0; j < img[0].size(); j++) {
      cc->EvalMultInPlace(img[i][j], q50_div[i][j]);
    }
  }
}

void dequantize_encrypted(vec2d_t<ciphertext_t> &img, cryptocontext_t cc) {
  for (size_t i = 0; i < img.size(); i++) {
    for (size_t j = 0; j < img[0].size(); j++) {
      cc->EvalMultInPlace(img[i][j], Q50[i][j]);
    }
  }
}

// todo opt here
template <typename D>
vec2d_t<ciphertext_t> dct_blocks_encrypted(const vec2d_t<ciphertext_t> &img,
                                           usint slots, cryptocontext_t cc,
                                           keypair_t keys, int depth) {

  vec2d_t<ciphertext_t> result =
      vec2d_t<ciphertext_t>(img.size(), vec1d_t<ciphertext_t>(partition, 0));

  vec1d_t<double> x = {0.0};
  plaintext_t ptxt = cc->MakeCKKSPackedPlaintext(x, depth, 0, nullptr, slots);

  //  uncomment this and remove below if partition != img.size()
  // for (size_t i = 0; i < img.size(); i++) {
  //   for (size_t j = 0; j < partition; j++) {
  //     result[i][j] = cc->Encrypt(keys.publicKey, ptxt);
  //   }
  // }

  vec2d_t<D> DCT = get_dct_table<D>(partition);
  vec2d_t<ciphertext_t> aux(partition, vec1d_t<ciphertext_t>(partition));
  ciphertext_t prod;

  // Can we do this in one loop?
  for (size_t i = 0; i < partition; i++) {
    for (size_t j = 0; j < partition; j++) {

      aux[i][j] = cc->Encrypt(keys.publicKey, ptxt);

      for (size_t k = 0; k < partition; k++) {
        prod = cc->EvalMult(img[k][j], DCT[i][k]);
        cc->EvalAddInPlace(aux[i][j], prod);
        // aux[i][j] += DCT[i][k] * img[x][k][j];
      }
    }
  }

  for (size_t i = 0; i < partition; i++) {
    for (size_t j = 0; j < partition; j++) {
      // remove this and uncomment above if partition != img.size()
      result[i][j] = cc->Encrypt(keys.publicKey, ptxt);
      for (size_t k = 0; k < partition; k++) {

        prod = cc->EvalMult(aux[i][k], DCT[j][k]);
        cc->EvalAddInPlace(result[i][j], prod);

        // result[i][j] += aux[i][k] * DCT[j][k];
      }
    }
  }
  return result;
}
// todo opt here
template <typename D>
vec2d_t<ciphertext_t> dct_blocks_encrypted_b(vec2d_t<ciphertext_t> &img,
                                             cryptocontext_t cc) {

  vec2d_t<D> dct_table = get_dct_table<D>(partition);
  vec2d_t<ciphertext_t> result = mat_mul_encrypted_one(img, dct_table, cc);
  dct_table = transpose(dct_table);
  return mat_mul_encrypted_one(result, dct_table, cc);
}

// todo opt here
template <typename D>
vec2d_t<ciphertext_t> idct_blocks_encrypted(const vec2d_t<ciphertext_t> &img,
                                            usint slots, cryptocontext_t cc,
                                            keypair_t keys, int depth) {

  vec2d_t<ciphertext_t> result =
      vec2d_t<ciphertext_t>(img.size(), vec1d_t<ciphertext_t>(partition, 0));

  vec1d_t<double> x = {0.0};
  plaintext_t ptxt = cc->MakeCKKSPackedPlaintext(x, depth, 0, nullptr, slots);

  for (size_t i = 0; i < result.size(); i++) {
    for (size_t j = 0; j < result[0].size(); j++) {
      ciphertext_t ct = cc->Encrypt(keys.publicKey, ptxt);
      result[i][j] = ct;
    }
  }

  vec2d_t<D> DCT = get_dct_table<D>(partition);

  vec2d_t<ciphertext_t> aux(partition, vec1d_t<ciphertext_t>(partition));

  plaintext_t output;

  for (size_t i = 0; i < partition; i++) {
    for (size_t j = 0; j < partition; j++) {

      aux[i][j] = cc->Encrypt(keys.publicKey, ptxt);

      for (size_t k = 0; k < partition; k++) {
        ciphertext_t prod = cc->EvalMult(img[k][j], DCT[k][i]);
        cc->EvalAddInPlace(aux[i][j], prod);
        // cc->Decrypt(keys.secretKey, aux[i][j], &output);
        // output->SetLength(4);
        // std::cout << "aux" << output << std::endl;
      }
    }
  }

  for (size_t i = 0; i < partition; i++) {
    for (size_t j = 0; j < partition; j++) {
      // this stays because we lose precision when we cast
      ciphertext_t sum = cc->Encrypt(keys.publicKey, ptxt);
      for (size_t k = 0; k < partition; k++) {
        ciphertext_t prod = cc->EvalMult(aux[i][k], DCT[k][j]);
        cc->EvalAddInPlace(sum, prod);
        // cc->Decrypt(keys.secretKey, sum, &output);
        // output->SetLength(4);
        // std::cout << "sum" << output << std::endl;
      }
      result[i][j] = sum;
    }
  }
  return result;
}

// todo opt here
template <typename T>
vec2d_t<ciphertext_t> conv_parallel_encrypted(vec2d_t<ciphertext_t> img,
                                              vec2d_t<T> filter, usint slots,
                                              cryptocontext_t cc,
                                              keypair_t keys, int depth) {

  vec2d_t<ciphertext_t> result =
      vec2d_t<ciphertext_t>(img.size(), vec1d_t<ciphertext_t>(partition, 0));

  vec1d_t<double> x = {0.0};
  plaintext_t ptxt = cc->MakeCKKSPackedPlaintext(x, depth, 0, nullptr, slots);

  for (size_t i = 0; i < result.size(); i++) {
    for (size_t j = 0; j < result[0].size(); j++) {
      ciphertext_t ct = cc->Encrypt(keys.publicKey, ptxt);
      result[i][j] = ct;
    }
  }

  // vec2d_t<ciphertext_t> result = img;

  for (size_t x = 1; x < partition - 1; x++) {
    for (size_t y = 1; y < partition - 1; y++) {
      ciphertext_t ct = cc->Encrypt(keys.publicKey, ptxt);
      result[x][y] = ct;
      for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
          ciphertext_t prod =
              cc->EvalMult(img[x - 1 + i][y - 1 + j], filter[i][j]);
          cc->EvalAddInPlace(result[x][y], prod);
          // result[x][y] += img[x - 1 + i][y - 1 + j] * filter[i][j];
        }
      }
    }
  }

  for (size_t x = 1; x < partition - 1; x++) {
    ciphertext_t prod1 = cc->EvalMult(result[x][1], 2);
    ciphertext_t sub1 = cc->EvalSub(prod1, result[x][2]);
    result[x][0] = sub1;
    // result[x][0] = 2 * result[x][1] - result[x][2];
    ciphertext_t prod2 = cc->EvalMult(result[x][conv_partition], 2);
    ciphertext_t sub2 = cc->EvalSub(prod2, result[x][conv_partition - 1]);
    result[x][partition - 1] = sub2;
    // result[x][7] = 2 * result[x][6] - result[x][5];
  }
  for (size_t i = 0; i < result[1].size(); i++) {
    ciphertext_t prod = cc->EvalMult(result[1][i], 2);
    ciphertext_t aux = cc->EvalSub(prod, result[2][i]);
    result[0][i] = aux;
    // result[0][i] = (2 * result[1][i]) - result[2][i];
  }
  for (size_t i = 0; i < result[conv_partition].size(); i++) {
    ciphertext_t prod = cc->EvalMult(result[conv_partition][i], 2);
    ciphertext_t aux = cc->EvalSub(prod, result[conv_partition - 1][i]);
    result[partition - 1][i] = aux;
    // result[7][i] = (2 * result[6][i]) - result[5][i];
  }

  return result;
}

// todo opt here
vec2d_t<ciphertext_t> pixel_inversion_encrypted(vec2d_t<ciphertext_t> img,
                                                usint slots, cryptocontext_t cc,
                                                keypair_t keys, int depth,
                                                int run_inv) {

  if (run_inv == 0) {
    return img;
  }

  vec2d_t<ciphertext_t> result =
      vec2d_t<ciphertext_t>(img.size(), vec1d_t<ciphertext_t>(partition, 0));

  for (size_t i = 0; i < img.size(); i++) {
    for (size_t j = 0; j < partition; j++) {
      result[i][j] = cc->EvalSub(256.0, img[i][j]);
    }
  }

  return result;
}

void level_encrypted(vec2d_t<ciphertext_t> &img, cryptocontext_t cc) {
  for (size_t i = 0; i < img.size(); i++) {
    for (size_t j = 0; j < img[i].size(); j++) {
      cc->EvalAddInPlace(img[i][j], -128);
    }
  }
}

void unlevel_encrypted(vec2d_t<ciphertext_t> &img, cryptocontext_t cc) {
  for (size_t i = 0; i < img.size(); i++) {
    for (size_t j = 0; j < img[i].size(); j++) {
      cc->EvalAddInPlace(img[i][j], 128);
    }
  }
}

template <typename D>
vec2d_t<ciphertext_t> expand_image_6x6_server_encrypted_verbose(
    vec1d_t<ciphertext_t> &encrypted_image, usint slots, cryptocontext_t cc,
    keypair_t keys, int depth) {
  std::cout << "Decompression: ";
  vec2d_t<ciphertext_t> image_decompressed;
  {
    Benchmark bench;
    image_decompressed =
        decompress_encrypted(encrypted_image, slots, cc, keys, depth);
  }
  std::cout << "Dequantization: ";
  {
    Benchmark bench;
    dequantize_encrypted(image_decompressed, cc);
  }
  std::cout << "Inverse dct: ";
  vec2d_t<ciphertext_t> image_idct;
  {
    Benchmark bench;
    image_idct =
        idct_blocks_encrypted<D>(image_decompressed, slots, cc, keys, depth);
  }
  std::cout << "Unlevel: ";
  {
    Benchmark bench;
    unlevel_encrypted(image_idct, cc);
  }
  return image_idct;
}

template <typename D>
vec2d_t<ciphertext_t>
expand_image_6x6_server_encrypted(vec1d_t<ciphertext_t> &encrypted_image,
                                  usint slots, cryptocontext_t cc,
                                  keypair_t keys, int depth) {
  vec2d_t<ciphertext_t> image_decompressed =
      decompress_encrypted(encrypted_image, slots, cc, keys, depth);
  encrypted_image.clear();
  dequantize_encrypted(image_decompressed, cc);
  vec2d_t<ciphertext_t> image_idct =
      idct_blocks_encrypted<D>(image_decompressed, slots, cc, keys, depth);
  image_decompressed.clear();
  unlevel_encrypted(image_idct, cc);
  return image_idct;
}

template <typename D>
vec1d_t<ciphertext_t>
compress_image_6x6_server_encrypted(vec2d_t<ciphertext_t> &image_expanded,
                                    int cutting_point, vec2d_t<D> q50_div,
                                    usint slots, cryptocontext_t cc,
                                    keypair_t keys, int depth) {
  level_encrypted(image_expanded, cc);
  vec2d_t<ciphertext_t> image_dct =
      dct_blocks_encrypted<D>(image_expanded, slots, cc, keys, depth);
  // dct_blocks_encrypted_b<D>(image_expanded, slots, cc, keys, depth);
  image_expanded.clear();
  quantize_encrypted(image_dct, cc, q50_div);
  vec1d_t<ciphertext_t> compressed_server =
      compress_encrypted(image_dct, cutting_point);
  image_dct.clear();
  return compressed_server;
}

int compute_diff_pos(ciphertext_t ctxt1, ciphertext_t ctxt2, int len,
                     cryptocontext_t cc, keypair_t keys) {
  plaintext_t ptxt1;
  plaintext_t ptxt2;
  double max = -1;

  cc->Decrypt(keys.secretKey, ctxt1, &ptxt1);
  ptxt1->SetLength(len);
  cc->Decrypt(keys.secretKey, ctxt2, &ptxt2);
  ptxt2->SetLength(len);

  size_t slots = ptxt1->GetCKKSPackedValue().size();

  for (size_t i = 0; i < slots; i++) {
    auto diff = fabs(real(ptxt1->GetCKKSPackedValue()[i]) -
                     real(ptxt2->GetCKKSPackedValue()[i]));
    if (diff > max) {
      max = diff;
    }
  }

  return max;
}

int compute_diff(vec1d_t<ciphertext_t> &ctxt1, vec1d_t<ciphertext_t> &ctxt2,
                 int len, cryptocontext_t cc, keypair_t keys) {
  for (int i = 0; i < len; i++) {
    int diff = compute_diff_pos(ctxt1[i], ctxt2[i], len, cc, keys);
    if (diff > 0) {
      std::cout << "NO PASS" << std::endl;
      return false;
    }
  }
  std::cout << "PASS" << std::endl;
  return true;
}

int main() {
  // test();

  // CKKS SETUP

  vec1d_t<uint32_t> dim1 = {0, 0};

  // Level budget = {e,d} controls the computational complex_tity of the
  // homomorphic encoding and decoding steps in CKKS bootstrapping, both
  // are
  // homomorphic evaluations of linear transforms. A higher budget allows
  // faster
  // computation of these steps at the expense of using a deeper circuit
  // (consuming more levels). On the other hand, lower budget would be
  // slower
  // but it uses a shallow circuit. Recommended values, found
  // experimentally,
  // are e or d = {1,2,3, or 4} (they do not need to be equal)
  vec1d_t<uint32_t> levelBudget = {1, 1};

#if NATIVEINT == 128
  lbcrypto::ScalingTechnique rescaleTech = FIXEDMANUAL;
  usint dcrtBits = 78;
  usint firstMod = 89; /*firstMod*/
#else
  lbcrypto::ScalingTechnique rescaleTech = lbcrypto::FLEXIBLEAUTOEXT;
  usint dcrtBits = 59;
  usint firstMod = 60; /*firstMod*/
#endif

  auto secretKeyDist =
      lbcrypto::UNIFORM_TERNARY; // Check section 3 to see why uniform
                                 // ternary
                                 // makes sense:
                                 // https://eprint.iacr.org/2020/1118.pdf
  // auto n = 1 << 15;
  // int quality = 90;

  // auto N = cc->GetRingDimension();
  //  std::cout << N << std::endl;
  //  std::cout << *cc->GetCryptoParameters()  <<std::endl;
  //  NativeInteger q =
  //  cc->GetElementParams()->GetParams()[0]->GetModulus().ConvertToInt();

  //  std::cout << q << std::endl;

  //  exit(0);

  //  std::cout << cc->GetElementParams()->GetParams()[0]->GetBigModulus() <<
  //  std::endl;

  //  auto seclevel  = parameters.GetSecurityLevel();
  //  auto seclevel_statistical = parameters.GetStatisticalSecurity();
  //  auto seckeydist = parameters.GetSecretKeyDist();
  //  auto standar_deviation = parameters.GetStandardDeviation();

  //  std::cout << "Security level: " << seclevel << "Statistical security" <<
  //  seclevel_statistical << " Secret key distribution: " << seckeydist <<
  //  "Error standard deviation " << standar_deviation << std::endl;
  // std::cout << dcrtBits << firstMod << std::endl;
  // std::cout << parameters << std::endl;
  // std::cout <<  cc->GetElementParams()->GetModulus() << std::endl;
  //   exit(0);

  vec1d_t<std::string> images = {
      "Sun.bmp.csv", "cardinal2048.bmp.csv", "mandril.bmp.csv",
      "pirate.bmp.csv",       "boat.bmp.csv",         "lake.bmp.csv",
      "mandril256.bmp.csv",   "pirate1024.bmp.csv",   "cameraman.bmp.csv",
      "monument.bmp.csv",     "pirate256.bmp.csv",    "cameraman256.bmp.csv",
      "office.bmp.csv",       "sunset1024.bmp.csv",   "cardinal1024.bmp.csv",
      "livingroom.bmp.csv",   "peppers_gray.bmp.csv", "sunset2048.bmp.csv"};

  for (std::string img_path : images) {

    // OBTAIN IMAGE

    std::cout << img_path << "\n";
    vec2d_t<int16_t> image = readCSV<int16_t>("../data/csvs/" + img_path, ',');

    for (int kk = 1; kk < 4; kk++) {
      if (kk == 0) {
        std::cout << "Setting 1\n";
        setting = 0;
        run_conv = 0;
        run_inv = 1;
        partition = 8;
      } else if (kk == 1) {
        std::cout << "Setting 2\n";
        setting = 0;
        run_conv = 1;
        run_inv = 0;
        partition = 8;
      } else if (kk == 2) {
        std::cout << "Setting 3\n";
        setting = 1;
        run_conv = 0;
        run_inv = 1;
        partition = 16;
        conv_partition = 14;
      } else {
        std::cout << "Setting 4\n";
        setting = 1;
        run_conv = 1;
        run_inv = 0;
        partition = 16;
        conv_partition = 14;
      }

      usint depth = 9;
      size_t cutting_point = 30;

      if (run_conv == 1) { // setting to perform the (m-2)x(m-2) reduction
        if (partition == 16) {
          cutting_point = 70;
        }
      } else { // working directly with blocks of mxm
        depth = 6;
        if (partition == 8) {
          cutting_point = 22;
        } else {
          cutting_point = 63;
        }
      }

      lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;

      parameters.SetMultiplicativeDepth(depth);
      parameters.SetScalingModSize(dcrtBits);
      parameters.SetFirstModSize(firstMod);
      parameters.SetScalingTechnique(rescaleTech);
      parameters.SetSecretKeyDist(secretKeyDist);
      parameters.SetNumLargeDigits(3);
      parameters.SetKeySwitchTechnique(lbcrypto::HYBRID);

      parameters.SetSecurityLevel(lbcrypto::HEStd_128_classic);

      // parameters.SetSecurityLevel(lbcrypto::HEStd_NotSet);
      // parameters.SetRingDim(1<<14);

      cryptocontext_t cc = GenCryptoContext(parameters);

      size_t slots = cc->GetRingDimension() / 2;

      size_t r_size =
          image.size() -
          image.size() %
              (setting == 0
                   ? partition
                   : conv_partition); // smallest size multiple of partition

      image.resize(r_size);
      for (size_t i = 0; i < image.size(); i++) {
        image[i].resize(r_size);
      }

      std::cout << r_size << std::endl;

      // vec2d_t<double> F = {{0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0,
      // 0.0}};
      vec2d_t<double> F = {
          {-1.0, -1.0, -1.0}, {-1.0, 8.0, -1.0}, {-1.0, -1.0, -1.0}};

      // vec2d_t<int16_t> a = compress_image_6x6_client<int16_t, double>(image,
      // cutting_point); vec3d_t<double> d = expand_image_6x6_server<int16_t,
      // double>(a);

      // vec3d_t<double> conv_i = conv_parallel(d,F);
      // std::cout << "Image client convolved: " << std::endl;
      // print_vec3d(conv_i);

      size_t H = image.size();
      size_t Low = image[0].size();

      size_t l =
          std::ceil(double(Low) / ((setting == 0) ? partition : conv_partition));
      size_t h =
          std::ceil(double(H) / ((setting == 0) ? partition : conv_partition));

      std::cout << "H: " << H << std::endl;
      std::cout << "L: " << Low << std::endl;
      std::cout << "l: " << l << std::endl;
      std::cout << "h: " << h << std::endl;

      size_t num_Blocks = l * h;

      //depending on the size of the image and the size of the blocks we will require large ciphertexts
      //we pick the minimum depth that gives us large enough ciphertexts for the necessary amount of blocks
      if(num_Blocks > slots) {
        std::cout << "Setting necessary depth for " << l*h << " blocks" << std::endl;
      }
      while(num_Blocks > slots) {
        depth += 1;
        //std::cout << "depth: " << depth << std::endl;
        parameters.SetMultiplicativeDepth(depth);
        cc = GenCryptoContext(parameters);
        slots = cc->GetRingDimension() / 2;
        //std::cout << "slots" << slots << std::endl;
      }

            // Turn on features
      cc->Enable(lbcrypto::PKE);
      cc->Enable(lbcrypto::KEYSWITCH);
      cc->Enable(lbcrypto::LEVELEDSHE);
      cc->Enable(lbcrypto::ADVANCEDSHE);
      cc->Enable(lbcrypto::FHE);

      auto keys = cc->KeyGen();
      cc->EvalMultKeyGen(keys.secretKey);

      std::cout << "Num of blocks: " << num_Blocks << std::endl;
      std::cout << "Cutting point: " << cutting_point << std::endl;
      std::cout << "Depth: " << depth << std::endl;
      std::cout << "N: " << cc->GetRingDimension() << std::endl;

      setQMatrix((partition == 8) ? Q50_8 : Q50_16);

      // TODO: can we do this outside main?
      vec2d_t<double> q50_div(Q50.size(), vec1d_t<double>(Q50[0].size()));

#pragma omp parallel for num_threads(NUM_THREADS)
      for (size_t i = 0; i < partition; i++) {
        for (size_t j = 0; j < partition; j++) {
          q50_div[i][j] = 1. / Q50[i][j];
        }
      }

      // print_vec2d(q50_div);

      vec2d_t<int16_t> compressed_image;

      

      if (setting == 0) {
        compressed_image =
            compress_image_client<int16_t, double>(image, cutting_point);
      } else {
        compressed_image = compress_image_6x6_client_debug<int16_t, double>(
            image, cutting_point);
      }

      // vec2d_t<int16_t> compressed_image =
      // compress_image_6x6_client_debug<int16_t, double>(image, cutting_point);
      // vec2d_t<int16_t> compressed_image =
      //   compress_image_client<int16_t, double>(image, cutting_point);

      std::cout << "Client encryption: ";
      vec1d_t<ciphertext_t> encrypted_image;
      {
        Benchmark bench;
        encrypted_image =
            encrypt_image(compressed_image, slots, cc, keys, depth);
      }

      vec1d_t<ciphertext_t> encrypted_image_copy =
          encrypt_image(compressed_image, slots, cc, keys, depth);

      plaintext_t output;

      // for(const auto& block: encrypted_image){
      //     cc->Decrypt(keys.secretKey, block, &output);
      //     output->SetLength(cutting_point);
      //     std::cout << output;
      // }

      vec2d_t<ciphertext_t> image_expanded;
      vec2d_t<ciphertext_t> image_expanded_copy;
      vec1d_t<ciphertext_t> compressed_server;

      // Server decompression
      std::cout << "Time server expansion: ";
      {
        Benchmark bench;
        image_expanded = expand_image_6x6_server_encrypted<double>(
            encrypted_image, slots, cc, keys, depth);
        // image_expanded = expand_image_server_encrypted<double>(
        // encrypted_image, slots, cc, keys, depth);
      }

      // std::cout << "image expanded" << std::endl;

      // for(size_t i = 0; i < image_expanded.size(); i++){
      //   for(size_t j = 0; j < image_expanded[0].size(); j++){
      //     std::cout << "i,j" << i << "," << j << std::endl;
      //     cc->Decrypt(keys.secretKey, image_expanded[i][j], &output);
      //     output->SetLength(cutting_point);
      //     std::cout << output;
      //   }
      // }

      // If we run this the correction test will not pass since
      // image_expanded is modified
      vec2d_t<ciphertext_t> img_transformed;
      if (run_conv == 1) {
        std::cout << "Time server convolution: ";
        {
          Benchmark bench;
          img_transformed = conv_parallel_encrypted(image_expanded, F, slots,
                                                    cc, keys, depth);
        }
      } else {
        std::cout << "Time server pixel-wise operation: ";
        {
          Benchmark bench;
          img_transformed = pixel_inversion_encrypted(image_expanded, slots, cc,
                                                      keys, depth, run_inv);
        }
      }

      // for (size_t i = 0; i < img_conv.size(); i++) {
      //   for (size_t j = 0; j < img_conv[0].size(); j++) {
      //     std::cout << "i,j" << i << "," << j << std::endl;
      //     cc->Decrypt(keys.secretKey, img_conv[i][j], &output);
      //     output->SetLength(cutting_point);
      //     std::cout << output;
      //   }
      // }

      std::cout << "Time server compression: ";
      {
        Benchmark bench;
        compressed_server = compress_image_6x6_server_encrypted<double>(
            img_transformed, cutting_point, q50_div, slots, cc, keys, depth);
      }

      // compute_diff(encrypted_image_copy, compressed_server, cutting_point,
      // cc,
      //              keys);

      vec2d_t<double> compressed_server_decrypted = decrypt_image<double>(
          compressed_server, num_Blocks, cutting_point, cc, keys);

      compressed_server.clear();

      vec2d_t<double> expanded_image_client =
          setting == 0 ? expand_image_client<double, double>(
                             compressed_server_decrypted, l)
                       : expand_image_6x6_client<double, double>(
                             compressed_server_decrypted, l);

      compressed_server_decrypted.clear();

      // auto expanded_image_client = expand_image_6x6_client<double,
      // double>(compressed_server_decrypted, l); auto expanded_image_client =
      // expand_image_client<double, double>(compressed_server_decrypted, l);

      writeCSV("../data/csvs/out/output_image_" + img_path, expanded_image_client,
               ',');

      // for(size_t i = 0; i < encrypted_image.size(); i++) {
      //   int diff = compute_diff(encrypted_image_copy[i],
      //   compressed_server[i], encrypted_image_copy.size(), cc, keys);
      //   std::cout << "diference: " << diff << std::endl;
      // }

      // level_encrypted(image_expanded,cc);
      // vec2d_t<ciphertext_t> image_dct =
      // dct_blocks_encrypted<double>(image_expanded, slots, cc, keys, depth);
      // quantize_encrypted(image_dct, cc, q50_div);
      // vec1d_t<ciphertext_t> compressed_server = compress_encrypted(image_dct,
      // cutting_point, slots, cc, keys, depth);

      // for(size_t i = 0; i < compressed_server.size(); i++){
      //   std::cout << "i" << i << std::endl;
      //   cc->Decrypt(keys.secretKey, compressed_server[i], &output);
      //   output->SetLength(cutting_point);
      //   std::cout << output;
      // }

      // for(size_t i = 0; i < compressed_server.size(); i++){
      //   for(size_t j = 0; j < compressed_server[0].size(); j++){
      //     std::cout << "i,j" << i << "," << j << std::endl;
      //     cc->Decrypt(keys.secretKey, compressed_server[i][j], &output);
      //     output->SetLength(cutting_point);
      //     std::cout << output;
      //   }
      // }
    }
  }
  return 0;
}
