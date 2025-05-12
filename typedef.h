#define PROFILE

#include<iostream>
#include<string>
#include<chrono>
#include<complex>
#include<vector>

#include "openfhe.h"

template<typename T>
using vec1d_t = std::vector<T>;
template<typename T>
using vec2d_t = std::vector<std::vector<T>>;
template<typename T>
using vec3d_t = std::vector<std::vector<std::vector<T>>>;

using complex_t = std::complex<double>;


using dcrtpoly_t = lbcrypto::DCRTPoly;
using cryptocontext_t = lbcrypto::CryptoContext<dcrtpoly_t>;
using fheckkrns_t = lbcrypto::FHECKKSRNS;
using plaintext_t = lbcrypto::Plaintext;
using ciphertext_t = lbcrypto::Ciphertext<dcrtpoly_t>;
using keypair_t = lbcrypto::KeyPair<dcrtpoly_t>;
