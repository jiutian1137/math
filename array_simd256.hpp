#pragma once

#include "array_simdcore.hpp"

#include <concepts>// std::convertible_to<..,..>

#include <cmath>

#include <intrin.h>
#ifndef _mm256_neg_pd
#define _mm256_neg_pd(x) _mm256_mul_pd(_mm256_set1_pd(-1.0), x)
#endif
#ifndef _mm256_abs_pd
#define _mm256_abs_pd(x) _mm256_andnot_pd(_mm256_set1_pd(-0.0), x)
#endif
#ifndef _mm256_roundeven_pd
#define _mm256_roundeven_pd(x) _mm256_round_pd(x, _MM_ROUND_MODE_NEAREST) 
#endif


/// Operations of Array, optimized with unroll(16 times & package_stride times) & simd256.
///		math::multi_array<float,size,1,8>
///		math::multi_array<double,size,1,4>
///		math::simd_multi_array<float,size,1,__m256>
///		math::simd_multi_array<double,size,1,__m256d>
///@license Free 
///@review 2022-5-22 
///@author LongJiangnan, Jiang1998Nan@outlook.com 
#define _MATH_ARRAY_SIMD256_HPP_

namespace math {
#ifdef _MATH_MULTI_ARRAY_
	/// z[i] = -x[i].
	template<size_t M, size_t N>
	_array_simdlen4_op_x(double, M, N, -, _mm256_neg_pd, 
		_mm256_loadu_pd, _mm256_storeu_pd)

	/// z[i] = x[i] + y[i].
	template<size_t M, size_t N>
	_array_simdlen4_x_op_y(double, M, N, +, _mm256_add_pd, 
		_mm256_loadu_pd, _mm256_storeu_pd)

	/// z[i] = x[i] - y[i].
	template<size_t M, size_t N>
	_array_simdlen4_x_op_y(double, M, N, -, _mm256_sub_pd, 
		_mm256_loadu_pd, _mm256_storeu_pd)

	/// z[i] = x[i] * y[i].
	template<size_t M, size_t N>
	_array_simdlen4_x_op_y(double, M, N, *, _mm256_mul_pd, 
		_mm256_loadu_pd, _mm256_storeu_pd)

	/// z[i] = x[i] / y[i].
	template<size_t M, size_t N>
	_array_simdlen4_x_op_y(double, M, N, /, _mm256_div_pd, 
		_mm256_loadu_pd, _mm256_storeu_pd)

	/// z[i] = x[i] + yval.
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, double>
	_array_simdlen4_x_op_yval(double, M, N, T2, +, _mm256_add_pd,
		_mm256_loadu_pd, _mm256_storeu_pd, _mm256_set1_pd)

	/// z[i] = x[i] - yval.
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, double>
	_array_simdlen4_x_op_yval(double, M, N, T2, -, _mm256_sub_pd,
		_mm256_loadu_pd, _mm256_storeu_pd, _mm256_set1_pd)

	/// z[i] = x[i] * yval.
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, double>
	_array_simdlen4_x_op_yval(double, M, N, T2, *, _mm256_mul_pd,
		_mm256_loadu_pd, _mm256_storeu_pd, _mm256_set1_pd)

	/// z[i] = x[i] / yval.
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, double>
	_array_simdlen4_x_op_yval(double, M, N, T2, /, _mm256_div_pd,
		_mm256_loadu_pd, _mm256_storeu_pd, _mm256_set1_pd)

	/// z[i] = xval + y[i].
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, double>
	_array_simdlen4_xval_op_y(T2, double, M, N, +, _mm256_add_pd,
		_mm256_loadu_pd, _mm256_storeu_pd, _mm256_set1_pd)

	/// z[i] = xval - y[i].
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, double>
	_array_simdlen4_xval_op_y(T2, double, M, N, -, _mm256_sub_pd,
		_mm256_loadu_pd, _mm256_storeu_pd, _mm256_set1_pd)

	/// z[i] = xval * y[i].
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, double>
	_array_simdlen4_xval_op_y(T2, double, M, N, *, _mm256_mul_pd,
		_mm256_loadu_pd, _mm256_storeu_pd, _mm256_set1_pd)

	/// z[i] = xval / y[i].
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, double>
	_array_simdlen4_xval_op_y(T2, double, M, N, /, _mm256_div_pd,
		_mm256_loadu_pd, _mm256_storeu_pd, _mm256_set1_pd)

	/// x[i] += y[i].
	template<size_t M, size_t N>
	_array_simdlen4_x_assginop_y(double, M, N, +=, _mm256_add_pd,
		_mm256_loadu_pd, _mm256_storeu_pd)

	/// x[i] -= y[i].
	template<size_t M, size_t N>
	_array_simdlen4_x_assginop_y(double, M, N, -=, _mm256_sub_pd,
		_mm256_loadu_pd, _mm256_storeu_pd)

	/// x[i] *= y[i].
	template<size_t M, size_t N>
	_array_simdlen4_x_assginop_y(double, M, N, *=, _mm256_mul_pd,
		_mm256_loadu_pd, _mm256_storeu_pd)

	/// x[i] /= y[i].
	template<size_t M, size_t N>
	_array_simdlen4_x_assginop_y(double, M, N, /=, _mm256_div_pd,
		_mm256_loadu_pd, _mm256_storeu_pd)

	/// x[i] += yval.
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, double>
	_array_simdlen4_x_assginop_yval(double, M, N, T2, +=, _mm256_add_pd,
		_mm256_loadu_pd, _mm256_storeu_pd, _mm256_set1_pd)

	/// x[i] -= yval.
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, double>
	_array_simdlen4_x_assginop_yval(double, M, N, T2, -=, _mm256_sub_pd,
		_mm256_loadu_pd, _mm256_storeu_pd, _mm256_set1_pd)

	/// x[i] *= yval.
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, double>
	_array_simdlen4_x_assginop_yval(double, M, N, T2, *=, _mm256_mul_pd,
		_mm256_loadu_pd, _mm256_storeu_pd, _mm256_set1_pd)

	/// x[i] /= yval.
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, double>
	_array_simdlen4_x_assginop_yval(double, M, N, T2, /=, _mm256_div_pd,
		_mm256_loadu_pd, _mm256_storeu_pd, _mm256_set1_pd)

	using std::fmod;
	using std::min;
	using std::max;
	using std::abs;
	using std::trunc;
	using std::floor;
	using std::ceil;
	using std::round;

	/// z[i] = fmod(x[i], y[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x_y(double, M, N, fmod, _mm256_fmod_pd,
		_mm256_loadu_pd, _mm256_storeu_pd)

	/// z[i] = fmod(x[i], yval).
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, double>
	_array_simdlen4_fn_x_yval(double, M, N, T2, fmod, _mm256_fmod_pd,
		_mm256_loadu_pd, _mm256_storeu_pd, _mm256_set1_pd)

	/// z[i] = fmod(xval, y[i]).
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, double>
	_array_simdlen4_fn_xval_y(T2, double, M, N, fmod, _mm256_fmod_pd,
		_mm256_loadu_pd, _mm256_storeu_pd, _mm256_set1_pd)

	/// z[i] = min(x[i], y[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x_y(double, M, N, min, _mm256_min_pd,
		_mm256_loadu_pd, _mm256_storeu_pd)

	/// z[i] = min(x[i], yval).
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, double>
	_array_simdlen4_fn_x_yval(double, M, N, T2, min, _mm256_min_pd,
		_mm256_loadu_pd, _mm256_storeu_pd, _mm256_set1_pd)

	/// z[i] = min(xval, y[i]).
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, double>
	_array_simdlen4_fn_xval_y(T2, double, M, N, min, _mm256_min_pd,
		_mm256_loadu_pd, _mm256_storeu_pd, _mm256_set1_pd)

	/// z[i] = max(x[i], y[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x_y(double, M, N, max, _mm256_max_pd,
		_mm256_loadu_pd, _mm256_storeu_pd)

	/// z[i] = max(x[i], yval).
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, double>
	_array_simdlen4_fn_x_yval(double, M, N, T2, max, _mm256_max_pd,
		_mm256_loadu_pd, _mm256_storeu_pd, _mm256_set1_pd)

	/// z[i] = max(xval, y[i]).
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, double>
	_array_simdlen4_fn_xval_y(T2, double, M, N, max, _mm256_max_pd,
		_mm256_loadu_pd, _mm256_storeu_pd, _mm256_set1_pd)

	/// y[i] = abs(x[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x(double, M, N, abs, _mm256_abs_pd,
		_mm256_loadu_pd, _mm256_storeu_pd)

	/// y[i] = trunc(x[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x(double, M, N, trunc, _mm256_trunc_pd,
		_mm256_loadu_pd, _mm256_storeu_pd)

	/// y[i] = floor(x[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x(double, M, N, floor, _mm256_floor_pd,
		_mm256_loadu_pd, _mm256_storeu_pd)

	/// y[i] = ceil(x[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x(double, M, N, ceil, _mm256_ceil_pd,
		_mm256_loadu_pd, _mm256_storeu_pd)

	/// y[i] = round(x[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x(double, M, N, round, _mm256_roundeven_pd,
		_mm256_loadu_pd, _mm256_storeu_pd)

	using std::sqrt;
	using std::cbrt;
	using std::exp;
	using std::exp2;
	using std::log;
	using std::log2;
	using std::pow;
	using std::sin;
	using std::cos;
	using std::tan;
	using std::asin;
	using std::acos;
	using std::atan;
	using std::atan2;

	/// y[i] = sqrt(x[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x(double, M, N, sqrt, _mm256_sqrt_pd,
		_mm256_loadu_pd, _mm256_storeu_pd)

	/// y[i] = cbrt(x[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x(double, M, N, cbrt, _mm256_cbrt_pd,
		_mm256_loadu_pd, _mm256_storeu_pd)

	/// y[i] = exp(x[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x(double, M, N, exp, _mm256_exp_pd,
		_mm256_loadu_pd, _mm256_storeu_pd)

	/// y[i] = exp2(x[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x(double, M, N, exp2, _mm256_exp2_pd,
		_mm256_loadu_pd, _mm256_storeu_pd)

	/// y[i] = log(x[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x(double, M, N, log, _mm256_log_pd,
		_mm256_loadu_pd, _mm256_storeu_pd)

	/// y[i] = log2(x[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x(double, M, N, log2, _mm256_log2_pd,
		_mm256_loadu_pd, _mm256_storeu_pd)

	/// z[i] = pow(x[i], y[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x_y(double, M, N, pow, _mm256_pow_pd,
		_mm256_loadu_pd, _mm256_storeu_pd)

	/// z[i] = pow(x[i], yval).
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, double>
	_array_simdlen4_fn_x_yval(double, M, N, T2, pow, _mm256_pow_pd,
		_mm256_loadu_pd, _mm256_storeu_pd, _mm256_set1_pd)

	/// z[i] = pow(xval, y[i]).
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, double>
	_array_simdlen4_fn_xval_y(T2, double, M, N, pow, _mm256_pow_pd,
		_mm256_loadu_pd, _mm256_storeu_pd, _mm256_set1_pd)

	/// y[i] = sin(x[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x(double, M, N, sin, _mm256_sin_pd,
		_mm256_loadu_pd, _mm256_storeu_pd)

	/// y[i] = cos(x[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x(double, M, N, cos, _mm256_cos_pd,
		_mm256_loadu_pd, _mm256_storeu_pd)

	/// y[i] = tan(x[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x(double, M, N, tan, _mm256_tan_pd,
		_mm256_loadu_pd, _mm256_storeu_pd)

	/// y[i] = asin(x[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x(double, M, N, asin, _mm256_asin_pd,
		_mm256_loadu_pd, _mm256_storeu_pd)

	/// y[i] = acos(x[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x(double, M, N, acos, _mm256_acos_pd,
		_mm256_loadu_pd, _mm256_storeu_pd)

	/// y[i] = atan(x[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x(double, M, N, atan, _mm256_atan_pd,
		_mm256_loadu_pd, _mm256_storeu_pd)

	/// z[i] = atan2(x[i], y[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x_y(double, M, N, atan2, _mm256_atan2_pd,
		_mm256_loadu_pd, _mm256_storeu_pd)

	/// z[i] = atan2(x[i], yval).
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, double>
	_array_simdlen4_fn_x_yval(double, M, N, T2, atan2, _mm256_atan2_pd,
		_mm256_loadu_pd, _mm256_storeu_pd, _mm256_set1_pd)

	/// z[i] = atan2(xval, y[i]).
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, double>
	_array_simdlen4_fn_xval_y(T2, double, M, N, atan2, _mm256_atan2_pd,
		_mm256_loadu_pd, _mm256_storeu_pd, _mm256_set1_pd)

#endif
}