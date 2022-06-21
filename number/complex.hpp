#pragma once
#ifndef _MATH_BEGIN
#define _MATH_BEGIN namespace math {
#define _MATH_END }
#endif

#include <complex>
#include <concepts>

_STD_BEGIN
template<typename _Ty, typename _Other> requires std::convertible_to<_Other, _Ty> inline 
std::complex<_Ty> operator+(const std::complex<_Ty>& a, const _Other& b) {
	return a + static_cast<_Ty>(b);
}

template<typename _Ty, typename _Other> requires std::convertible_to<_Other, _Ty> inline
std::complex<_Ty> operator-(const std::complex<_Ty>& a, const _Other& b) {
	return a - static_cast<_Ty>(b);
}

template<typename _Ty, typename _Other> requires std::convertible_to<_Other, _Ty> inline
std::complex<_Ty> operator*(const std::complex<_Ty>& a, const _Other& b) {
	return a * static_cast<_Ty>(b);
}

template<typename _Ty, typename _Other> requires std::convertible_to<_Other, _Ty> inline
std::complex<_Ty> operator/(const std::complex<_Ty>& a, const _Other& b) {
	return a / static_cast<_Ty>(b);
}

template<typename _Ty, typename _Other> requires std::convertible_to<_Other, _Ty> inline
std::complex<_Ty> operator+(const _Other& a, const std::complex<_Ty>& b) {
	return static_cast<_Ty>(a) + b;
}

template<typename _Ty, typename _Other> requires std::convertible_to<_Other, _Ty> inline
std::complex<_Ty> operator-(const _Other& a, const std::complex<_Ty>& b) {
	return static_cast<_Ty>(a) - b;
}

template<typename _Ty, typename _Other> requires std::convertible_to<_Other, _Ty> inline
std::complex<_Ty> operator*(const _Other& a, const std::complex<_Ty>& b) {
	return static_cast<_Ty>(a) * b;
}

template<typename _Ty, typename _Other> requires std::convertible_to<_Other, _Ty> inline
std::complex<_Ty> operator/(const _Other& a, const std::complex<_Ty>& b) {
	return static_cast<_Ty>(a) / b;
}

template<typename _Ty, typename _Other> requires std::convertible_to<_Other, _Ty> inline
bool operator==(const std::complex<_Ty>& a, const _Other& b) {
	return a == static_cast<_Ty>(b);
}

template<typename _Ty, typename _Other> requires std::convertible_to<_Other, _Ty> inline
bool operator!=(const std::complex<_Ty>& a, const _Other& b) {
	return a != static_cast<_Ty>(b);
}

template<typename _Ty, typename _Other> requires std::convertible_to<_Other, _Ty> inline
bool operator==(const _Other& a, const std::complex<_Ty>& b) {
	return static_cast<_Ty>(a) == b;
}

template<typename _Ty, typename _Other> requires std::convertible_to<_Other, _Ty> inline
bool operator!=(const _Other& a, const std::complex<_Ty>& b) {
	return static_cast<_Ty>(a) != b;
}
_STD_END

_MATH_BEGIN
template<typename R>
using complex = std::complex<R>;
using std::imag;
using std::real;
using std::sqrt;
using std::abs;
using std::acos;
using std::acosh;
using std::asinh;
using std::asin;
using std::atanh;
using std::atan;
using std::cosh;
using std::exp;
using std::log;
using std::pow;
using std::sinh;
using std::tanh;
using std::arg;
using std::conj;
using std::proj;
using std::cos;
using std::log10;
using std::norm;
using std::polar;
using std::sin;
using std::tan;
inline namespace literals {
	using namespace std::literals::complex_literals;
}
_MATH_END