#pragma once

#include "array_simdcore.hpp"

#include <concepts>// std::convertible_to<..,..>

#include <cmath>

#include <intrin.h>
#ifndef _mm_neg_ps
#define _mm_neg_ps(x) _mm_mul_ps(_mm_set1_ps(-1.0f), x)
#endif
#ifndef _mm_abs_ps
#define _mm_abs_ps(x) _mm_andnot_ps(_mm_set1_ps(-0.0f), x)
#endif
#ifndef _mm_round_even_ps
#define _mm_round_even_ps(x) _mm_round_ps(x, _MM_ROUND_MODE_NEAREST) 
#endif
#ifndef _mm_neg_pd
#define _mm_neg_pd(x) _mm_mul_pd(_mm_set1_pd(-1.0), x)
#endif
#ifndef _mm_abs_pd
#define _mm_abs_pd(x) _mm_andnot_pd(_mm_set1_pd(-0.0), x)
#endif
#ifndef _mm_round_even_pd
#define _mm_round_even_pd(x) _mm_round_pd(x, _MM_ROUND_MODE_NEAREST) 
#endif


/// Operations of Array, optimized with unroll(16 times & package_stride times) & simd128.
///		math::multi_array<float,size,1,4>
///		math::multi_array<double,size,1,2>
///		math::simd_multi_array<float,size,1,__m128>
///		math::simd_multi_array<double,size,1,__m128d>
///@license Free 
///@review 2022-5-22 
///@author LongJiangnan, Jiang1998Nan@outlook.com 
#define _MATH_ARRAY_SIMD128_

namespace math {
#ifdef _MATH_MULTI_ARRAY_
	// y[i] = -x[i].
	template<size_t M, size_t N>
	_array_simdlen4_op_x(float, M, N, -, _mm_neg_ps, 
		_mm_loadu_ps, _mm_storeu_ps)

	/// z[i] = x[i] + y[i].
	template<size_t M, size_t N>
	_array_simdlen4_x_op_y(float, M, N, +, _mm_add_ps, 
		_mm_loadu_ps, _mm_storeu_ps)

	/// z[i] = x[i] - y[i].
	template<size_t M, size_t N>
	_array_simdlen4_x_op_y(float, M, N, -, _mm_sub_ps, 
		_mm_loadu_ps, _mm_storeu_ps)

	/// z[i] = x[i] * y[i].
	template<size_t M, size_t N>
	_array_simdlen4_x_op_y(float, M, N, *, _mm_mul_ps, 
		_mm_loadu_ps, _mm_storeu_ps)

	/// z[i] = x[i] / y[i].
	template<size_t M, size_t N>
	_array_simdlen4_x_op_y(float, M, N, /, _mm_div_ps, 
		_mm_loadu_ps, _mm_storeu_ps)

	/// z[i] = x[i] + yval.
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, float>
	_array_simdlen4_x_op_yval(float, M, N, T2, +, _mm_add_ps,
		_mm_loadu_ps, _mm_storeu_ps, _mm_set1_ps)

	/// z[i] = x[i] - yval.
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, float>
	_array_simdlen4_x_op_yval(float, M, N, T2, -, _mm_sub_ps,
		_mm_loadu_ps, _mm_storeu_ps, _mm_set1_ps)

	/// z[i] = x[i] * yval.
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, float>
	_array_simdlen4_x_op_yval(float, M, N, T2, *, _mm_mul_ps,
		_mm_loadu_ps, _mm_storeu_ps, _mm_set1_ps)

	/// z[i] = x[i] / yval.
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, float>
	_array_simdlen4_x_op_yval(float, M, N, T2, /, _mm_div_ps,
		_mm_loadu_ps, _mm_storeu_ps, _mm_set1_ps)

	/// z[i] = xval + y[i].
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, float>
	_array_simdlen4_xval_op_y(T2, float, M, N, +, _mm_add_ps,
		_mm_loadu_ps, _mm_storeu_ps, _mm_set1_ps)

	/// z[i] = xval - y[i].
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, float>
	_array_simdlen4_xval_op_y(T2, float, M, N, -, _mm_sub_ps,
		_mm_loadu_ps, _mm_storeu_ps, _mm_set1_ps)

	/// z[i] = xval * y[i].
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, float>
	_array_simdlen4_xval_op_y(T2, float, M, N, *, _mm_mul_ps,
		_mm_loadu_ps, _mm_storeu_ps, _mm_set1_ps)

	/// z[i] = xval / y[i].
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, float>
	_array_simdlen4_xval_op_y(T2, float, M, N, /, _mm_div_ps,
		_mm_loadu_ps, _mm_storeu_ps, _mm_set1_ps)

	/// x[i] += y[i].
	template<size_t M, size_t N>
	_array_simdlen4_x_assginop_y(float, M, N, +=, _mm_add_ps,
		_mm_loadu_ps, _mm_storeu_ps)

	/// x[i] -= y[i].
	template<size_t M, size_t N>
	_array_simdlen4_x_assginop_y(float, M, N, -=, _mm_sub_ps,
		_mm_loadu_ps, _mm_storeu_ps)

	/// x[i] *= y[i].
	template<size_t M, size_t N>
	_array_simdlen4_x_assginop_y(float, M, N, *=, _mm_mul_ps,
		_mm_loadu_ps, _mm_storeu_ps)

	/// x[i] /= y[i].
	template<size_t M, size_t N>
	_array_simdlen4_x_assginop_y(float, M, N, /=, _mm_div_ps,
		_mm_loadu_ps, _mm_storeu_ps)

	/// x[i] += yval.
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, float>
	_array_simdlen4_x_assginop_yval(float, M, N, T2, +=, _mm_add_ps,
		_mm_loadu_ps, _mm_storeu_ps, _mm_set1_ps)

	/// x[i] -= yval.
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, float>
	_array_simdlen4_x_assginop_yval(float, M, N, T2, -=, _mm_sub_ps,
		_mm_loadu_ps, _mm_storeu_ps, _mm_set1_ps)

	/// x[i] *= yval.
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, float>
	_array_simdlen4_x_assginop_yval(float, M, N, T2, *=, _mm_mul_ps,
		_mm_loadu_ps, _mm_storeu_ps, _mm_set1_ps)

	/// x[i] /= yval.
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, float>
	_array_simdlen4_x_assginop_yval(float, M, N, T2, /=, _mm_div_ps,
		_mm_loadu_ps, _mm_storeu_ps, _mm_set1_ps)

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
	_array_simdlen4_fn_x_y(float, M, N, fmod, _mm_fmod_ps,
		_mm_loadu_ps, _mm_storeu_ps)

	/// z[i] = fmod(x[i], yval).
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, float>
	_array_simdlen4_fn_x_yval(float, M, N, T2, fmod, _mm_fmod_ps,
		_mm_loadu_ps, _mm_storeu_ps, _mm_set1_ps)

	/// z[i] = fmod(xval, y[i]).
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, float>
	_array_simdlen4_fn_xval_y(T2, float, M, N, fmod, _mm_fmod_ps,
		_mm_loadu_ps, _mm_storeu_ps, _mm_set1_ps)

	/// z[i] = min(x[i], y[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x_y(float, M, N, min, _mm_min_ps,
		_mm_loadu_ps, _mm_storeu_ps)

	/// z[i] = min(x[i], yval).
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, float>
	_array_simdlen4_fn_x_yval(float, M, N, T2, min, _mm_min_ps,
		_mm_loadu_ps, _mm_storeu_ps, _mm_set1_ps)

	/// z[i] = min(xval, y[i]).
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, float>
	_array_simdlen4_fn_xval_y(T2, float, M, N, min, _mm_min_ps,
		_mm_loadu_ps, _mm_storeu_ps, _mm_set1_ps)

	/// z[i] = max(x[i], y[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x_y(float, M, N, max, _mm_max_ps,
		_mm_loadu_ps, _mm_storeu_ps)

	/// z[i] = max(x[i], yval).
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, float>
	_array_simdlen4_fn_x_yval(float, M, N, T2, max, _mm_max_ps,
		_mm_loadu_ps, _mm_storeu_ps, _mm_set1_ps)

	/// z[i] = max(xval, y[i]).
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, float>
	_array_simdlen4_fn_xval_y(T2, float, M, N, max, _mm_max_ps,
		_mm_loadu_ps, _mm_storeu_ps, _mm_set1_ps)

	/// y[i] = abs(x[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x(float, M, N, abs, _mm_abs_ps,
		_mm_loadu_ps, _mm_storeu_ps)

	/// y[i] = trunc(x[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x(float, M, N, trunc, _mm_trunc_ps,
		_mm_loadu_ps, _mm_storeu_ps)

	/// y[i] = floor(x[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x(float, M, N, floor, _mm_floor_ps,
		_mm_loadu_ps, _mm_storeu_ps)

	/// y[i] = ceil(x[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x(float, M, N, ceil, _mm_ceil_ps,
		_mm_loadu_ps, _mm_storeu_ps)

	/// y[i] = round(x[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x(float, M, N, round, _mm_round_even_ps,
		_mm_loadu_ps, _mm_storeu_ps)

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
	_array_simdlen4_fn_x(float, M, N, sqrt, _mm_sqrt_ps,
		_mm_loadu_ps, _mm_storeu_ps)

	/// y[i] = cbrt(x[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x(float, M, N, cbrt, _mm_cbrt_ps,
		_mm_loadu_ps, _mm_storeu_ps)

	/// y[i] = exp(x[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x(float, M, N, exp, _mm_exp_ps,
		_mm_loadu_ps, _mm_storeu_ps)

	/// y[i] = exp2(x[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x(float, M, N, exp2, _mm_exp2_ps,
		_mm_loadu_ps, _mm_storeu_ps)

	/// y[i] = log(x[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x(float, M, N, log, _mm_log_ps,
		_mm_loadu_ps, _mm_storeu_ps)

	/// y[i] = log2(x[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x(float, M, N, log2, _mm_log2_ps,
		_mm_loadu_ps, _mm_storeu_ps)

	/// z[i] = pow(x[i], y[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x_y(float, M, N, pow, _mm_pow_ps,
		_mm_loadu_ps, _mm_storeu_ps)

	/// z[i] = pow(x[i], yval).
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, float>
	_array_simdlen4_fn_x_yval(float, M, N, T2, pow, _mm_pow_ps,
		_mm_loadu_ps, _mm_storeu_ps, _mm_set1_ps)

	/// z[i] = pow(xval, y[i]).
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, float>
	_array_simdlen4_fn_xval_y(T2, float, M, N, pow, _mm_pow_ps,
		_mm_loadu_ps, _mm_storeu_ps, _mm_set1_ps)

	/// y[i] = sin(x[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x(float, M, N, sin, _mm_sin_ps,
		_mm_loadu_ps, _mm_storeu_ps)

	/// y[i] = cos(x[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x(float, M, N, cos, _mm_cos_ps,
		_mm_loadu_ps, _mm_storeu_ps)

	/// y[i] = tan(x[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x(float, M, N, tan, _mm_tan_ps,
		_mm_loadu_ps, _mm_storeu_ps)

	/// y[i] = asin(x[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x(float, M, N, asin, _mm_asin_ps,
		_mm_loadu_ps, _mm_storeu_ps)

	/// y[i] = acos(x[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x(float, M, N, acos, _mm_acos_ps,
		_mm_loadu_ps, _mm_storeu_ps)

	/// y[i] = atan(x[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x(float, M, N, atan, _mm_atan_ps,
		_mm_loadu_ps, _mm_storeu_ps)

	/// z[i] = atan2(x[i], y[i]).
	template<size_t M, size_t N>
	_array_simdlen4_fn_x_y(float, M, N, atan2, _mm_atan2_ps,
		_mm_loadu_ps, _mm_storeu_ps)

	/// z[i] = atan2(x[i], yval).
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, float>
	_array_simdlen4_fn_x_yval(float, M, N, T2, atan2, _mm_atan2_ps,
		_mm_loadu_ps, _mm_storeu_ps, _mm_set1_ps)

	/// z[i] = atan2(xval, y[i]).
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, float>
	_array_simdlen4_fn_xval_y(T2, float, M, N, atan2, _mm_atan2_ps,
		_mm_loadu_ps, _mm_storeu_ps, _mm_set1_ps)


	// y[i] = -x[i].
	template<size_t M, size_t N>
	_array_simdlen2_op_x(double, M, N, -, _mm_neg_pd, 
		_mm_loadu_pd, _mm_storeu_pd)

	/// z[i] = x[i] + y[i].
	template<size_t M, size_t N>
	_array_simdlen2_x_op_y(double, M, N, +, _mm_add_pd,
		_mm_loadu_pd, _mm_storeu_pd)

	/// z[i] = x[i] - y[i].
	template<size_t M, size_t N>
	_array_simdlen2_x_op_y(double, M, N, -, _mm_sub_pd, 
		_mm_loadu_pd, _mm_storeu_pd)

	/// z[i] = x[i] * y[i].
	template<size_t M, size_t N>
	_array_simdlen2_x_op_y(double, M, N, *, _mm_mul_pd, 
		_mm_loadu_pd, _mm_storeu_pd)

	/// z[i] = x[i] / y[i].
	template<size_t M, size_t N>
	_array_simdlen2_x_op_y(double, M, N, /, _mm_div_pd, 
		_mm_loadu_pd, _mm_storeu_pd)

	/// z[i] = x[i] + yval.
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, double>
	_array_simdlen2_x_op_yval(double, M, N, T2, +, _mm_add_pd,
		_mm_loadu_pd, _mm_storeu_pd, _mm_set1_pd)

	/// z[i] = x[i] - yval.
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, double>
	_array_simdlen2_x_op_yval(double, M, N, T2, -, _mm_sub_pd,
		_mm_loadu_pd, _mm_storeu_pd, _mm_set1_pd)

	/// z[i] = x[i] * yval.
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, double>
	_array_simdlen2_x_op_yval(double, M, N, T2, *, _mm_mul_pd,
		_mm_loadu_pd, _mm_storeu_pd, _mm_set1_pd)

	/// z[i] = x[i] / yval.
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, double>
	_array_simdlen2_x_op_yval(double, M, N, T2, /, _mm_div_pd,
		_mm_loadu_pd, _mm_storeu_pd, _mm_set1_pd)

	/// z[i] = xval + y[i].
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, double>
	_array_simdlen2_xval_op_y(T2, double, M, N, +, _mm_add_pd,
		_mm_loadu_pd, _mm_storeu_pd, _mm_set1_pd)

	/// z[i] = xval - y[i].
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, double>
	_array_simdlen2_xval_op_y(T2, double, M, N, -, _mm_sub_pd,
		_mm_loadu_pd, _mm_storeu_pd, _mm_set1_pd)

	/// z[i] = xval * y[i].
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, double>
	_array_simdlen2_xval_op_y(T2, double, M, N, *, _mm_mul_pd,
		_mm_loadu_pd, _mm_storeu_pd, _mm_set1_pd)

	/// z[i] = xval / y[i].
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, double>
	_array_simdlen2_xval_op_y(T2, double, M, N, /, _mm_div_pd,
		_mm_loadu_pd, _mm_storeu_pd, _mm_set1_pd)

	/// x[i] += y[i].
	template<size_t M, size_t N>
	_array_simdlen2_x_assginop_y(double, M, N, +=, _mm_add_pd,
		_mm_loadu_pd, _mm_storeu_pd)

	/// x[i] -= y[i].
	template<size_t M, size_t N>
	_array_simdlen2_x_assginop_y(double, M, N, -=, _mm_sub_pd,
		_mm_loadu_pd, _mm_storeu_pd)

	/// x[i] *= y[i].
	template<size_t M, size_t N>
	_array_simdlen2_x_assginop_y(double, M, N, *=, _mm_mul_pd,
		_mm_loadu_pd, _mm_storeu_pd)

	/// x[i] /= y[i].
	template<size_t M, size_t N>
	_array_simdlen2_x_assginop_y(double, M, N, /=, _mm_div_pd,
		_mm_loadu_pd, _mm_storeu_pd)

	/// x[i] += yval.
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, double>
	_array_simdlen2_x_assginop_yval(double, M, N, T2, +=, _mm_add_pd,
		_mm_loadu_pd, _mm_storeu_pd, _mm_set1_pd)

	/// x[i] -= yval.
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, double>
	_array_simdlen2_x_assginop_yval(double, M, N, T2, -=, _mm_sub_pd,
		_mm_loadu_pd, _mm_storeu_pd, _mm_set1_pd)

	/// x[i] *= yval.
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, double>
	_array_simdlen2_x_assginop_yval(double, M, N, T2, *=, _mm_mul_pd,
		_mm_loadu_pd, _mm_storeu_pd, _mm_set1_pd)

	/// x[i] /= yval.
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, double>
	_array_simdlen2_x_assginop_yval(double, M, N, T2, /=, _mm_div_pd,
		_mm_loadu_pd, _mm_storeu_pd, _mm_set1_pd)

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
	_array_simdlen2_fn_x_y(double, M, N, fmod, _mm_fmod_pd,
		_mm_loadu_pd, _mm_storeu_pd)

	/// z[i] = fmod(x[i], yval).
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, double>
	_array_simdlen2_fn_x_yval(double, M, N, T2, fmod, _mm_fmod_pd,
		_mm_loadu_pd, _mm_storeu_pd, _mm_set1_pd)

	/// z[i] = fmod(xval, y[i]).
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, double>
	_array_simdlen2_fn_xval_y(T2, double, M, N, fmod, _mm_fmod_pd,
		_mm_loadu_pd, _mm_storeu_pd, _mm_set1_pd)

	/// z[i] = min(x[i], y[i]).
	template<size_t M, size_t N>
	_array_simdlen2_fn_x_y(double, M, N, min, _mm_min_pd,
		_mm_loadu_pd, _mm_storeu_pd)

	/// z[i] = min(x[i], yval).
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, double>
	_array_simdlen2_fn_x_yval(double, M, N, T2, min, _mm_min_pd,
		_mm_loadu_pd, _mm_storeu_pd, _mm_set1_pd)

	/// z[i] = min(xval, y[i]).
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, double>
	_array_simdlen2_fn_xval_y(T2, double, M, N, min, _mm_min_pd,
		_mm_loadu_pd, _mm_storeu_pd, _mm_set1_pd)

	/// z[i] = max(x[i], y[i]).
	template<size_t M, size_t N>
	_array_simdlen2_fn_x_y(double, M, N, max, _mm_max_pd,
		_mm_loadu_pd, _mm_storeu_pd)

	/// z[i] = max(x[i], yval).
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, double>
	_array_simdlen2_fn_x_yval(double, M, N, T2, max, _mm_max_pd,
		_mm_loadu_pd, _mm_storeu_pd, _mm_set1_pd)

	/// z[i] = max(xval, y[i]).
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, double>
	_array_simdlen2_fn_xval_y(T2, double, M, N, max, _mm_max_pd,
		_mm_loadu_pd, _mm_storeu_pd, _mm_set1_pd)

	/// y[i] = abs(x[i]).
	template<size_t M, size_t N>
	_array_simdlen2_fn_x(double, M, N, abs, _mm_abs_pd,
		_mm_loadu_pd, _mm_storeu_pd)

	/// y[i] = trunc(x[i]).
	template<size_t M, size_t N>
	_array_simdlen2_fn_x(double, M, N, trunc, _mm_trunc_pd,
		_mm_loadu_pd, _mm_storeu_pd)

	/// y[i] = floor(x[i]).
	template<size_t M, size_t N>
	_array_simdlen2_fn_x(double, M, N, floor, _mm_floor_pd,
		_mm_loadu_pd, _mm_storeu_pd)

	/// y[i] = ceil(x[i]).
	template<size_t M, size_t N>
	_array_simdlen2_fn_x(double, M, N, ceil, _mm_ceil_pd,
		_mm_loadu_pd, _mm_storeu_pd)

	/// y[i] = round(x[i]).
	template<size_t M, size_t N>
	_array_simdlen2_fn_x(double, M, N, round, _mm_round_even_pd,
		_mm_loadu_pd, _mm_storeu_pd)

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
	_array_simdlen2_fn_x(double, M, N, sqrt, _mm_sqrt_pd,
		_mm_loadu_pd, _mm_storeu_pd)

	/// y[i] = cbrt(x[i]).
	template<size_t M, size_t N>
	_array_simdlen2_fn_x(double, M, N, cbrt, _mm_cbrt_pd,
		_mm_loadu_pd, _mm_storeu_pd)

	/// y[i] = exp(x[i]).
	template<size_t M, size_t N>
	_array_simdlen2_fn_x(double, M, N, exp, _mm_exp_pd,
		_mm_loadu_pd, _mm_storeu_pd)

	/// y[i] = exp2(x[i]).
	template<size_t M, size_t N>
	_array_simdlen2_fn_x(double, M, N, exp2, _mm_exp2_pd,
		_mm_loadu_pd, _mm_storeu_pd)

	/// y[i] = log(x[i]).
	template<size_t M, size_t N>
	_array_simdlen2_fn_x(double, M, N, log, _mm_log_pd,
		_mm_loadu_pd, _mm_storeu_pd)

	/// y[i] = log2(x[i]).
	template<size_t M, size_t N>
	_array_simdlen2_fn_x(double, M, N, log2, _mm_log2_pd,
		_mm_loadu_pd, _mm_storeu_pd)

	/// z[i] = pow(x[i], y[i]).
	template<size_t M, size_t N>
	_array_simdlen2_fn_x_y(double, M, N, pow, _mm_pow_pd,
		_mm_loadu_pd, _mm_storeu_pd)

	/// z[i] = pow(x[i], yval).
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, double>
	_array_simdlen2_fn_x_yval(double, M, N, T2, pow, _mm_pow_pd,
		_mm_loadu_pd, _mm_storeu_pd, _mm_set1_pd)

	/// z[i] = pow(xval, y[i]).
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, double>
	_array_simdlen2_fn_xval_y(T2, double, M, N, pow, _mm_pow_pd,
		_mm_loadu_pd, _mm_storeu_pd, _mm_set1_pd)

	/// y[i] = sin(x[i]).
	template<size_t M, size_t N>
	_array_simdlen2_fn_x(double, M, N, sin, _mm_sin_pd,
		_mm_loadu_pd, _mm_storeu_pd)

	/// y[i] = cos(x[i]).
	template<size_t M, size_t N>
	_array_simdlen2_fn_x(double, M, N, cos, _mm_cos_pd,
		_mm_loadu_pd, _mm_storeu_pd)

	/// y[i] = tan(x[i]).
	template<size_t M, size_t N>
	_array_simdlen2_fn_x(double, M, N, tan, _mm_tan_pd,
		_mm_loadu_pd, _mm_storeu_pd)

	/// y[i] = asin(x[i]).
	template<size_t M, size_t N>
	_array_simdlen2_fn_x(double, M, N, asin, _mm_asin_pd,
		_mm_loadu_pd, _mm_storeu_pd)

	/// y[i] = acos(x[i]).
	template<size_t M, size_t N>
	_array_simdlen2_fn_x(double, M, N, acos, _mm_acos_pd,
		_mm_loadu_pd, _mm_storeu_pd)

	/// y[i] = atan(x[i]).
	template<size_t M, size_t N>
	_array_simdlen2_fn_x(double, M, N, atan, _mm_atan_pd,
		_mm_loadu_pd, _mm_storeu_pd)

	/// z[i] = atan2(x[i], y[i]).
	template<size_t M, size_t N>
	_array_simdlen2_fn_x_y(double, M, N, atan2, _mm_atan2_pd,
		_mm_loadu_pd, _mm_storeu_pd)

	/// z[i] = atan2(x[i], yval).
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, double>
	_array_simdlen2_fn_x_yval(double, M, N, T2, atan2, _mm_atan2_pd,
		_mm_loadu_pd, _mm_storeu_pd, _mm_set1_pd)

	/// z[i] = atan2(xval, y[i]).
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, double>
	_array_simdlen2_fn_xval_y(T2, double, M, N, atan2, _mm_atan2_pd,
		_mm_loadu_pd, _mm_storeu_pd, _mm_set1_pd)
#endif

#ifdef _MATH_SIMD_MULTI_ARRAY_
	template<size_t static_rows, size_t static_columns>
	using multi_array_m128i = simd_multi_array<float, static_rows, static_columns, __m128i>;

	template<size_t static_rows, size_t static_columns>
	using multi_array_m128 = simd_multi_array<float, static_rows, static_columns, __m128>;

	template<size_t static_rows, size_t static_columns>
	using multi_array_m128d = simd_multi_array<double, static_rows, static_columns, __m128d>;

	/// <summary>
	/// __m128
	/// </sumary>

	#define _array_m128_fn_x(M, N, FUNC, PACK_FN) \
	multi_array_m128<M,N> FUNC(const multi_array_m128<M,N> &x) { \
		multi_array_m128<M,N> y; \
		if constexpr (M != 0 && N != 0) { \
			if constexpr (y.actual_size() <= 16) { \
				if constexpr (y.actual_size() > 0) \
					_mm_store_ps(y.data(), PACK_FN( _mm_load_ps(x.data()) )); \
				if constexpr (y.actual_size() > 4) \
					_mm_store_ps(y.data()+4, PACK_FN( _mm_load_ps(x.data()+4) )); \
				if constexpr (y.actual_size() > 8) \
					_mm_store_ps(y.data()+8, PACK_FN( _mm_load_ps(x.data()+8) )); \
				if constexpr (y.actual_size() > 12) \
					_mm_store_ps(y.data()+12, PACK_FN( _mm_load_ps(x.data()+12) )); \
				return( y ); \
			} \
		} \
	\
		size_t       n  = y.actual_size(); \
		float       *yi = y.data(); \
		float const *xi = x.data(); \
		for ( ; n >= 16; n -= 16, yi += 16, xi += 16) { \
			_mm_store_ps(yi, PACK_FN( _mm_load_ps(xi) )); \
			_mm_store_ps(yi+4, PACK_FN( _mm_load_ps(xi+4) )); \
			_mm_store_ps(yi+8, PACK_FN( _mm_load_ps(xi+8) )); \
			_mm_store_ps(yi+12, PACK_FN( _mm_load_ps(xi+12) )); \
		} \
		for ( ; n >= 4; n -= 4, yi += 4, xi += 4) { \
			_mm_store_ps(yi, PACK_FN( _mm_load_ps(xi) )); \
		} \
		return( std::move(y) ); \
	}

	// y[i] = x[i].
	template<size_t M, size_t N>
	_array_m128_fn_x(M, N, operator+, )

	// y[i] = -x[i].
	template<size_t M, size_t N>
	_array_m128_fn_x(M, N, operator-, _mm_neg_ps)

	#define _array_m128_fn_x_y(M, N, FUNC, PACK_FN) \
	multi_array_m128<M,N> FUNC(const multi_array_m128<M,N> &x, const multi_array_m128<M,N> &y) { \
		multi_array_m128<M,N> z; \
		if constexpr (M != 0 && N != 0) { \
			if constexpr (z.actual_size() <= 16) { \
				if constexpr (z.actual_size() > 0) \
					_mm_store_ps(z.data(), PACK_FN( _mm_load_ps(x.data()), _mm_load_ps(y.data()) )); \
				if constexpr (z.actual_size() > 4) \
					_mm_store_ps(z.data()+4, PACK_FN( _mm_load_ps(x.data()+4), _mm_load_ps(y.data()+4) )); \
				if constexpr (z.actual_size() > 8) \
					_mm_store_ps(z.data()+8, PACK_FN( _mm_load_ps(x.data()+8), _mm_load_ps(y.data()+8) )); \
				if constexpr (z.actual_size() > 12) \
					_mm_store_ps(z.data()+12, PACK_FN( _mm_load_ps(x.data()+12), _mm_load_ps(y.data()+12) )); \
				return( z ); \
			} \
		} \
	\
		size_t       n  = z.actual_size(); \
		float       *zi = z.data(); \
		float const *xi = x.data(); \
		float const *yi = y.data(); \
		for ( ; n >= 16; n -= 16, zi += 16, xi += 16, yi += 16) { \
			_mm_store_ps(zi, PACK_FN( _mm_load_ps(xi), _mm_load_ps(yi) )); \
			_mm_store_ps(zi+4, PACK_FN( _mm_load_ps(xi+4), _mm_load_ps(yi+4) )); \
			_mm_store_ps(zi+8, PACK_FN( _mm_load_ps(xi+8), _mm_load_ps(yi+8) )); \
			_mm_store_ps(zi+12, PACK_FN( _mm_load_ps(xi+12), _mm_load_ps(yi+12) )); \
		} \
		for ( ; n >= 4; n -= 4, zi += 4, xi += 4, yi += 4) { \
			_mm_store_ps(zi, PACK_FN( _mm_load_ps(xi), _mm_load_ps(yi) )); \
		} \
		return( std::move(z) ); \
	}

	// z[i] = x[i] + y[i].
	template<size_t M, size_t N> 
	_array_m128_fn_x_y(M, N, operator+, _mm_add_ps)

	// z[i] = x[i] - y[i].
	template<size_t M, size_t N> 
	_array_m128_fn_x_y(M, N, operator-, _mm_sub_ps)

	// z[i] = x[i] * y[i].
	template<size_t M, size_t N> 
	_array_m128_fn_x_y(M, N, operator*, _mm_mul_ps)

	// z[i] = x[i] / y[i].
	template<size_t M, size_t N>
	_array_m128_fn_x_y(M, N, operator/, _mm_div_ps)

	#define _array_m128_fn_x_yval(M, N, T2, FUNC, PACK_FN) \
	multi_array_m128<M,N> FUNC(const multi_array_m128<M,N> &x, const T2 yval) { \
		const __m128 xmm2 = _mm_set1_ps( static_cast<float>(yval) ); \
		multi_array_m128<M,N> z; \
		if constexpr (M != 0 && N != 0) { \
			if constexpr (z.actual_size() <= 16) { \
				if constexpr (z.actual_size() > 0) \
					_mm_store_ps(z.data(), PACK_FN( _mm_load_ps(x.data()), xmm2 )); \
				if constexpr (z.actual_size() > 4) \
					_mm_store_ps(z.data()+4, PACK_FN( _mm_load_ps(x.data()+4), xmm2 )); \
				if constexpr (z.actual_size() > 8) \
					_mm_store_ps(z.data()+8, PACK_FN( _mm_load_ps(x.data()+8), xmm2 )); \
				if constexpr (z.actual_size() > 12) \
					_mm_store_ps(z.data()+12, PACK_FN( _mm_load_ps(x.data()+12), xmm2 )); \
				return( z ); \
			} \
		} \
	\
		size_t       n  = z.actual_size(); \
		float       *zi = z.data(); \
		float const *xi = x.data(); \
		for ( ; n >= 16; n -= 16, zi += 16, xi += 16) { \
			_mm_store_ps(zi, PACK_FN( _mm_load_ps(xi), xmm2 )); \
			_mm_store_ps(zi+4, PACK_FN( _mm_load_ps(xi+4), xmm2 )); \
			_mm_store_ps(zi+8, PACK_FN( _mm_load_ps(xi+8), xmm2 )); \
			_mm_store_ps(zi+12, PACK_FN( _mm_load_ps(xi+12), xmm2 )); \
		} \
		for ( ; n >= 4; n -= 4, zi += 4, xi += 4) { \
			_mm_store_ps(zi, PACK_FN( _mm_load_ps(xi), xmm2 )); \
		} \
		return( std::move(z) ); \
	}

	// z[i] = x[i] + yval.
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, float>
	_array_m128_fn_x_yval(M, N, T2, operator+, _mm_add_ps)

	// z[i] = x[i] - yval.
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, float>
	_array_m128_fn_x_yval(M, N, T2, operator-, _mm_sub_ps)

	// z[i] = x[i] * yval.
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, float>
	_array_m128_fn_x_yval(M, N, T2, operator*, _mm_mul_ps)

	// z[i] = x[i] / yval.
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, float>
	_array_m128_fn_x_yval(M, N, T2, operator/, _mm_div_ps)

	#define _array_m128_fn_xval_y(T2, M, N, FUNC, PACK_FN) \
	multi_array_m128<M,N> FUNC(const T2 xval, const multi_array_m128<M,N> &y) { \
		const __m128 xmm1 = _mm_set1_ps( static_cast<float>(xval) ); \
		multi_array_m128<M,N> z; \
		if constexpr (M != 0 && N != 0) { \
			if constexpr (z.actual_size() <= 16) { \
				if constexpr (z.actual_size() > 0) \
					_mm_store_ps(z.data(), PACK_FN( xmm1, _mm_load_ps(y.data()) )); \
				if constexpr (z.actual_size() > 4) \
					_mm_store_ps(z.data()+4, PACK_FN( xmm1, _mm_load_ps(y.data()+4) )); \
				if constexpr (z.actual_size() > 8) \
					_mm_store_ps(z.data()+8, PACK_FN( xmm1, _mm_load_ps(y.data()+8) )); \
				if constexpr (z.actual_size() > 12) \
					_mm_store_ps(z.data()+12, PACK_FN( xmm1, _mm_load_ps(y.data()+12) )); \
				return( z ); \
			} \
		} \
	\
		size_t       n  = z.actual_size(); \
		float       *zi = z.data(); \
		float const *yi = y.data(); \
		for ( ; n >= 16; n -= 16, zi += 16, yi += 16) { \
			_mm_store_ps(zi, PACK_FN( xmm1, _mm_load_ps(yi) )); \
			_mm_store_ps(zi+4, PACK_FN( xmm1, _mm_load_ps(yi+4) )); \
			_mm_store_ps(zi+8, PACK_FN( xmm1, _mm_load_ps(yi+8) )); \
			_mm_store_ps(zi+12, PACK_FN( xmm1, _mm_load_ps(yi+12) )); \
		} \
		for ( ; n >= 4; n -= 4, zi += 4, yi += 4) { \
			_mm_store_ps(zi, PACK_FN( xmm1, _mm_load_ps(yi) )); \
		} \
		return( std::move(z) ); \
	}

	// z[i] = xval + y[i].
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, float>
	_array_m128_fn_xval_y(T2, M, N, operator+, _mm_add_ps)

	// z[i] = xval - y[i].
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, float>
	_array_m128_fn_xval_y(T2, M, N, operator-, _mm_sub_ps)

	// z[i] = xval * y[i].
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, float>
	_array_m128_fn_xval_y(T2, M, N, operator*, _mm_mul_ps)

	// z[i] = xval / y[i].
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, float>
	_array_m128_fn_xval_y(T2, M, N, operator/, _mm_div_ps)

	#define _array_m128_assignfn_x_y(M, N, AFUNC, PACK_FN) \
	multi_array_m128<M,N>& AFUNC(multi_array_m128<M,N> &x, const multi_array_m128<M,N> &y) { \
		if constexpr (M != 0 && N != 0) { \
			if constexpr (x.actual_size() <= 16) { \
				if constexpr (x.actual_size() > 0) \
					_mm_store_ps(x.data(), PACK_FN( _mm_load_ps(x.data()), _mm_load_ps(y.data()) )); \
				if constexpr (x.actual_size() > 4) \
					_mm_store_ps(x.data()+4, PACK_FN( _mm_load_ps(x.data()+4), _mm_load_ps(y.data()+4) )); \
				if constexpr (x.actual_size() > 8) \
					_mm_store_ps(x.data()+8, PACK_FN( _mm_load_ps(x.data()+8), _mm_load_ps(y.data()+8) )); \
				if constexpr (x.actual_size() > 12) \
					_mm_store_ps(x.data()+12, PACK_FN( _mm_load_ps(x.data()+12), _mm_load_ps(y.data()+12) )); \
				return( x ); \
			} \
		} \
	\
		size_t       n  = x.actual_size(); \
		float       *xi = x.data(); \
		float const *yi = y.data(); \
		for ( ; n >= 16; n -= 16, xi += 16, yi += 16) { \
			_mm_store_ps(xi, PACK_FN( _mm_load_ps(xi), _mm_load_ps(yi) )); \
			_mm_store_ps(xi+4, PACK_FN( _mm_load_ps(xi+4), _mm_load_ps(yi+4) )); \
			_mm_store_ps(xi+8, PACK_FN( _mm_load_ps(xi+8), _mm_load_ps(yi+8) )); \
			_mm_store_ps(xi+12, PACK_FN( _mm_load_ps(xi+12), _mm_load_ps(yi+12) )); \
		} \
		for ( ; n >= 4; n -= 4, xi += 4, yi += 4) { \
			_mm_store_ps(xi, PACK_FN( _mm_load_ps(xi), _mm_load_ps(yi) )); \
		} \
		return( x ); \
	}

	// x[i] += y[i].
	template<size_t M, size_t N>
	_array_m128_assignfn_x_y(M, N, operator+=, _mm_add_ps)

	// x[i] -= y[i].
	template<size_t M, size_t N>
	_array_m128_assignfn_x_y(M, N, operator-=, _mm_sub_ps)

	// x[i] *= y[i].
	template<size_t M, size_t N>
	_array_m128_assignfn_x_y(M, N, operator*=, _mm_mul_ps)

	// x[i] /= y[i].
	template<size_t M, size_t N>
	_array_m128_assignfn_x_y(M, N, operator/=, _mm_div_ps)

	#define _array_m128_assignfn_x_yval(M, N, T2, AFUNC, PACK_FN) \
	multi_array_m128<M,N>& AFUNC(multi_array_m128<M,N> &x, const T2 yval) { \
		const __m128 xmm2 = _mm_set1_ps( static_cast<float>(yval) ); \
	\
		if constexpr (M != 0 && N != 0) { \
			if constexpr (x.actual_size() <= 16) { \
				if constexpr (x.actual_size() > 0) \
					_mm_store_ps(x.data(), PACK_FN( _mm_load_ps(x.data()), xmm2 )); \
				if constexpr (x.actual_size() > 4) \
					_mm_store_ps(x.data()+4, PACK_FN( _mm_load_ps(x.data()+4), xmm2 )); \
				if constexpr (x.actual_size() > 8) \
					_mm_store_ps(x.data()+8, PACK_FN( _mm_load_ps(x.data()+8), xmm2 )); \
				if constexpr (x.actual_size() > 12) \
					_mm_store_ps(x.data()+12, PACK_FN( _mm_load_ps(x.data()+12), xmm2 )); \
				return( x ); \
			} \
		} \
	\
		size_t       n  = x.actual_size(); \
		float       *xi = x.data(); \
		for ( ; n >= 16; n -= 16, xi += 16) { \
			_mm_store_ps(xi, PACK_FN( _mm_load_ps(xi), xmm2 )); \
			_mm_store_ps(xi+4, PACK_FN( _mm_load_ps(xi+4), xmm2 )); \
			_mm_store_ps(xi+8, PACK_FN( _mm_load_ps(xi+8), xmm2 )); \
			_mm_store_ps(xi+12, PACK_FN( _mm_load_ps(xi+12), xmm2 )); \
		} \
		for ( ; n >= 4; n -= 4, xi += 4) { \
			_mm_store_ps(xi, PACK_FN( _mm_load_ps(xi), xmm2 )); \
		} \
		return( x ); \
	}

	// x[i] += yval.
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, float>
	_array_m128_assignfn_x_yval(M, N, T2, operator+=, _mm_add_ps)

	// x[i] -= yval.
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, float>
	_array_m128_assignfn_x_yval(M, N, T2, operator-=, _mm_sub_ps)

	// x[i] *= yval.
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, float>
	_array_m128_assignfn_x_yval(M, N, T2, operator*=, _mm_mul_ps)

	// x[i] /= yval.
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, float>
	_array_m128_assignfn_x_yval(M, N, T2, operator/=, _mm_div_ps)

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
	_array_m128_fn_x_y(M, N, fmod, _mm_fmod_ps)

	/// z[i] = fmod(x[i], yval).
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, float>
	_array_m128_fn_x_yval(M, N, T2, fmod, _mm_fmod_ps)

	/// z[i] = fmod(xval, y[i]).
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, float>
	_array_m128_fn_xval_y(T2, M, N, fmod, _mm_fmod_ps)

	/// z[i] = min(x[i], y[i]).
	template<size_t M, size_t N>
	_array_m128_fn_x_y(M, N, min, _mm_min_ps)

	/// z[i] = min(x[i], yval).
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, float>
	_array_m128_fn_x_yval(M, N, T2, min, _mm_min_ps)

	/// z[i] = min(xval, y[i]).
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, float>
	_array_m128_fn_xval_y(T2, M, N, min, _mm_min_ps)

	/// z[i] = max(x[i], y[i]).
	template<size_t M, size_t N>
	_array_m128_fn_x_y(M, N, max, _mm_max_ps)

	/// z[i] = max(x[i], yval).
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, float>
	_array_m128_fn_x_yval(M, N, T2, max, _mm_max_ps)

	/// z[i] = max(xval, y[i]).
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, float>
	_array_m128_fn_xval_y(T2, M, N, max, _mm_max_ps)

	/// y[i] = abs(x[i]).
	template<size_t M, size_t N>
	_array_m128_fn_x(M, N, abs, _mm_abs_ps)

	/// y[i] = trunc(x[i]).
	template<size_t M, size_t N>
	_array_m128_fn_x(M, N, trunc, _mm_trunc_ps)

	/// y[i] = floor(x[i]).
	template<size_t M, size_t N>
	_array_m128_fn_x(M, N, floor, _mm_floor_ps)

	/// y[i] = ceil(x[i]).
	template<size_t M, size_t N>
	_array_m128_fn_x(M, N, ceil, _mm_ceil_ps)

	/// y[i] = round(x[i], mode = _MM_ROUND_MODE_NEAREST).
	template<size_t M, size_t N, int mode = _MM_ROUND_MODE_NEAREST>
	multi_array_m128<M,N> round(const multi_array_m128<M,N> &x, std::integral_constant<int,mode> _Unused = {}) {
		multi_array_m128<M,N> y;
		if constexpr (M != 0 && N != 0) {
			if constexpr (y.actual_size() <= 16) {
				if constexpr (y.actual_size() > 0)
					_mm_store_ps(y.data(), _mm_round_ps( _mm_load_ps(x.data()), mode ));
				if constexpr (y.actual_size() > 4)
					_mm_store_ps(y.data()+4, _mm_round_ps( _mm_load_ps(x.data()+4), mode ));
				if constexpr (y.actual_size() > 8)
					_mm_store_ps(y.data()+8, _mm_round_ps( _mm_load_ps(x.data()+8), mode ));
				if constexpr (y.actual_size() > 12)
					_mm_store_ps(y.data()+12, _mm_round_ps( _mm_load_ps(x.data()+12), mode ));
				return( y );
			}
		}

		size_t       n  = y.actual_size();
		float       *yi = y.data();
		float const *xi = x.data();
		for ( ; n >= 16; n -= 16, yi += 16, xi += 16) {
			_mm_store_ps(yi, _mm_round_ps( _mm_load_ps(xi), mode ));
			_mm_store_ps(yi+4, _mm_round_ps( _mm_load_ps(xi+4), mode ));
			_mm_store_ps(yi+8, _mm_round_ps( _mm_load_ps(xi+8), mode ));
			_mm_store_ps(yi+12, _mm_round_ps( _mm_load_ps(xi+12), mode ));
		}
		for ( ; n >= 4; n -= 4, yi += 4, xi += 4) {
			_mm_store_ps(yi, _mm_round_ps( _mm_load_ps(xi), mode ));
		}
		return( std::move(y) );
	}

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
	_array_m128_fn_x(M, N, sqrt, _mm_sqrt_ps)

	/// y[i] = cbrt(x[i]).
	template<size_t M, size_t N>
	_array_m128_fn_x(M, N, cbrt, _mm_cbrt_ps)

	/// y[i] = exp(x[i]).
	template<size_t M, size_t N>
	_array_m128_fn_x(M, N, exp, _mm_exp_ps)

	/// y[i] = exp2(x[i]).
	template<size_t M, size_t N>
	_array_m128_fn_x(M, N, exp2, _mm_exp2_ps)

	/// y[i] = log(x[i]).
	template<size_t M, size_t N>
	_array_m128_fn_x(M, N, log, _mm_log_ps)

	/// y[i] = log2(x[i]).
	template<size_t M, size_t N>
	_array_m128_fn_x(M, N, log2, _mm_log2_ps)

	/// z[i] = pow(x[i], y[i]).
	template<size_t M, size_t N>
	_array_m128_fn_x_y(M, N, pow, _mm_pow_ps)

	/// z[i] = pow(x[i], yval).
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, float>
	_array_m128_fn_x_yval(M, N, T2, pow, _mm_pow_ps)

	/// z[i] = pow(xval, y[i]).
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, float>
	_array_m128_fn_xval_y(T2, M, N, pow, _mm_pow_ps)

	/// y[i] = sin(x[i]).
	template<size_t M, size_t N>
	_array_m128_fn_x(M, N, sin, _mm_sin_ps)

	/// y[i] = cos(x[i]).
	template<size_t M, size_t N>
	_array_m128_fn_x(M, N, cos, _mm_cos_ps)

	/// y[i] = tan(x[i]).
	template<size_t M, size_t N>
	_array_m128_fn_x(M, N, tan, _mm_tan_ps)

	/// y[i] = asin(x[i]).
	template<size_t M, size_t N>
	_array_m128_fn_x(M, N, asin, _mm_asin_ps)

	/// y[i] = acos(x[i]).
	template<size_t M, size_t N>
	_array_m128_fn_x(M, N, acos, _mm_acos_ps)

	/// y[i] = atan(x[i]).
	template<size_t M, size_t N>
	_array_m128_fn_x(M, N, atan, _mm_atan_ps)

	/// z[i] = atan2(x[i], y[i]).
	template<size_t M, size_t N>
	_array_m128_fn_x_y(M, N, atan2, _mm_atan2_ps)

	/// z[i] = atan2(x[i], yval).
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, float>
	_array_m128_fn_x_yval(M, N, T2, atan2, _mm_atan2_ps)

	/// z[i] = atan2(xval, y[i]).
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, float>
	_array_m128_fn_xval_y(T2, M, N, atan2, _mm_atan2_ps)

	/// <summary>
	/// __m128d
	/// </sumary>

	#define _array_m128d_fn_x(M, N, FUNC, PACK_FN) \
	multi_array_m128d<M,N> FUNC(const multi_array_m128d<M,N> &x) { \
		multi_array_m128d<M,N> y; \
		if constexpr (M != 0 && N != 0) { \
			if constexpr (y.actual_size() <= 16) { \
				if constexpr (y.actual_size() > 0) \
					_mm_store_pd(y.data(), PACK_FN( _mm_load_pd(x.data()) )); \
				if constexpr (y.actual_size() > 2) \
					_mm_store_pd(y.data()+2, PACK_FN( _mm_load_pd(x.data()+2) )); \
				if constexpr (y.actual_size() > 4) \
					_mm_store_pd(y.data()+4, PACK_FN( _mm_load_pd(x.data()+4) )); \
				if constexpr (y.actual_size() > 6) \
					_mm_store_pd(y.data()+6, PACK_FN( _mm_load_pd(x.data()+6) )); \
				if constexpr (y.actual_size() > 8) \
					_mm_store_pd(y.data()+8, PACK_FN( _mm_load_pd(x.data()+8) )); \
				if constexpr (y.actual_size() > 10) \
					_mm_store_pd(y.data()+10, PACK_FN( _mm_load_pd(x.data()+10) )); \
				if constexpr (y.actual_size() > 12) \
					_mm_store_pd(y.data()+12, PACK_FN( _mm_load_pd(x.data()+12) )); \
				if constexpr (y.actual_size() > 14) \
					_mm_store_pd(y.data()+14, PACK_FN( _mm_load_pd(x.data()+14) )); \
				return( y ); \
			} \
		} \
	\
		size_t        n  = y.actual_size(); \
		double       *yi = y.data(); \
		double const *xi = x.data(); \
		for ( ; n >= 16; n -= 16, yi += 16, xi += 16) { \
			_mm_store_pd(yi, PACK_FN( _mm_load_pd(xi) )); \
			_mm_store_pd(yi+2, PACK_FN( _mm_load_pd(xi+2) )); \
			_mm_store_pd(yi+4, PACK_FN( _mm_load_pd(xi+4) )); \
			_mm_store_pd(yi+6, PACK_FN( _mm_load_pd(xi+6) )); \
			_mm_store_pd(yi+8, PACK_FN( _mm_load_pd(xi+8) )); \
			_mm_store_pd(yi+10, PACK_FN( _mm_load_pd(xi+10) )); \
			_mm_store_pd(yi+12, PACK_FN( _mm_load_pd(xi+12) )); \
			_mm_store_pd(yi+14, PACK_FN( _mm_load_pd(xi+14) )); \
		} \
		for ( ; n >= 4; n -= 4, yi += 4, xi += 4) { \
			_mm_store_pd(yi, PACK_FN( _mm_load_pd(xi) )); \
			_mm_store_pd(yi+2, PACK_FN( _mm_load_pd(xi+2) )); \
		} \
		for ( ; n >= 2; n -= 2, yi += 2, xi += 2) { \
			_mm_store_pd(yi, PACK_FN( _mm_load_pd(xi) )); \
		} \
		return( std::move(y) ); \
	}

	// y[i] = x[i].
	template<size_t M, size_t N>
	_array_m128d_fn_x(M, N, operator+, )

	// y[i] = -x[i].
	template<size_t M, size_t N>
	_array_m128d_fn_x(M, N, operator-, _mm_neg_pd)

	#define _array_m128d_fn_x_y(M, N, FUNC, PACK_FN) \
	multi_array_m128d<M,N> FUNC(const multi_array_m128d<M,N> &x, const multi_array_m128d<M,N> &y) { \
		multi_array_m128d<M,N> z; \
		if constexpr (M != 0 && N != 0) { \
			if constexpr (z.actual_size() <= 16) { \
				if constexpr (z.actual_size() > 0) \
					_mm_store_pd(z.data(), PACK_FN( _mm_load_pd(x.data()), _mm_load_pd(y.data()) )); \
				if constexpr (z.actual_size() > 2) \
					_mm_store_pd(z.data()+2, PACK_FN( _mm_load_pd(x.data()+2), _mm_load_pd(y.data()+2) )); \
				if constexpr (z.actual_size() > 4) \
					_mm_store_pd(z.data()+4, PACK_FN( _mm_load_pd(x.data()+4), _mm_load_pd(y.data()+4) )); \
				if constexpr (z.actual_size() > 6) \
					_mm_store_pd(z.data()+6, PACK_FN( _mm_load_pd(x.data()+6), _mm_load_pd(y.data()+6) )); \
				if constexpr (z.actual_size() > 8) \
					_mm_store_pd(z.data()+8, PACK_FN( _mm_load_pd(x.data()+8), _mm_load_pd(y.data()+8) )); \
				if constexpr (z.actual_size() > 10) \
					_mm_store_pd(z.data()+10, PACK_FN( _mm_load_pd(x.data()+10), _mm_load_pd(y.data()+10) )); \
				if constexpr (z.actual_size() > 12) \
					_mm_store_pd(z.data()+12, PACK_FN( _mm_load_pd(x.data()+12), _mm_load_pd(y.data()+12) )); \
				if constexpr (z.actual_size() > 14) \
					_mm_store_pd(z.data()+14, PACK_FN( _mm_load_pd(x.data()+14), _mm_load_pd(y.data()+14) )); \
				return( z ); \
			} \
		} \
	\
		size_t        n  = z.actual_size(); \
		double       *zi = z.data(); \
		double const *xi = x.data(); \
		double const *yi = y.data(); \
		for ( ; n >= 16; n -= 16, zi += 16, xi += 16, yi += 16) { \
			_mm_store_pd(zi, PACK_FN( _mm_load_pd(xi), _mm_load_pd(yi) )); \
			_mm_store_pd(zi+2, PACK_FN( _mm_load_pd(xi+2), _mm_load_pd(yi+2) )); \
			_mm_store_pd(zi+4, PACK_FN( _mm_load_pd(xi+4), _mm_load_pd(yi+4) )); \
			_mm_store_pd(zi+6, PACK_FN( _mm_load_pd(xi+6), _mm_load_pd(yi+6) )); \
			_mm_store_pd(zi+8, PACK_FN( _mm_load_pd(xi+8), _mm_load_pd(yi+8) )); \
			_mm_store_pd(zi+10, PACK_FN( _mm_load_pd(xi+10), _mm_load_pd(yi+10) )); \
			_mm_store_pd(zi+12, PACK_FN( _mm_load_pd(xi+12), _mm_load_pd(yi+12) )); \
			_mm_store_pd(zi+14, PACK_FN( _mm_load_pd(xi+14), _mm_load_pd(yi+14) )); \
		} \
		for ( ; n >= 4; n -= 4, zi += 4, xi += 4, yi += 4) { \
			_mm_store_pd(zi, PACK_FN( _mm_load_pd(xi), _mm_load_pd(yi) )); \
			_mm_store_pd(zi+2, PACK_FN( _mm_load_pd(xi+2), _mm_load_pd(yi+2) )); \
		} \
		for ( ; n >= 2; n -= 2, zi += 2, xi += 2, yi += 2) { \
			_mm_store_pd(zi, PACK_FN( _mm_load_pd(xi), _mm_load_pd(yi) )); \
		} \
		return( std::move(z) ); \
	}

	// z[i] = x[i] + y[i].
	template<size_t M, size_t N>
	_array_m128d_fn_x_y(M, N, operator+, _mm_add_pd)

	// z[i] = x[i] - y[i].
	template<size_t M, size_t N>
	_array_m128d_fn_x_y(M, N, operator-, _mm_sub_pd)

	// z[i] = x[i] * y[i].
	template<size_t M, size_t N>
	_array_m128d_fn_x_y(M, N, operator*, _mm_mul_pd)

	// z[i] = x[i] / y[i].
	template<size_t M, size_t N>
	_array_m128d_fn_x_y(M, N, operator/, _mm_div_pd)

	#define _array_m128d_fn_x_yval(M, N, T2, FUNC, PACK_FN) \
	multi_array_m128d<M,N> FUNC(const multi_array_m128d<M,N> &x, const T2 yval) { \
		const __m128d xmm2 = _mm_set1_pd( static_cast<double>(yval) ); \
		multi_array_m128d<M,N> z; \
		if constexpr (M != 0 && N != 0) { \
			if constexpr (z.actual_size() <= 16) { \
				if constexpr (z.actual_size() > 0) \
					_mm_store_pd(z.data(), PACK_FN( _mm_load_pd(x.data()), xmm2 )); \
				if constexpr (z.actual_size() > 2) \
					_mm_store_pd(z.data()+2, PACK_FN( _mm_load_pd(x.data()+2), xmm2 )); \
				if constexpr (z.actual_size() > 4) \
					_mm_store_pd(z.data()+4, PACK_FN( _mm_load_pd(x.data()+4), xmm2 )); \
				if constexpr (z.actual_size() > 6) \
					_mm_store_pd(z.data()+6, PACK_FN( _mm_load_pd(x.data()+6), xmm2 )); \
				if constexpr (z.actual_size() > 8) \
					_mm_store_pd(z.data()+8, PACK_FN( _mm_load_pd(x.data()+8), xmm2 )); \
				if constexpr (z.actual_size() > 10) \
					_mm_store_pd(z.data()+10, PACK_FN( _mm_load_pd(x.data()+10), xmm2 )); \
				if constexpr (z.actual_size() > 12) \
					_mm_store_pd(z.data()+12, PACK_FN( _mm_load_pd(x.data()+12), xmm2 )); \
				if constexpr (z.actual_size() > 14) \
					_mm_store_pd(z.data()+14, PACK_FN( _mm_load_pd(x.data()+14), xmm2 )); \
				return( z ); \
			} \
		} \
	\
		size_t        n  = z.actual_size(); \
		double       *zi = z.data(); \
		double const *xi = x.data(); \
		for ( ; n >= 16; n -= 16, zi += 16, xi += 16) { \
			_mm_store_pd(zi, PACK_FN( _mm_load_pd(xi), xmm2 )); \
			_mm_store_pd(zi+2, PACK_FN( _mm_load_pd(xi+2), xmm2 )); \
			_mm_store_pd(zi+4, PACK_FN( _mm_load_pd(xi+4), xmm2 )); \
			_mm_store_pd(zi+6, PACK_FN( _mm_load_pd(xi+6), xmm2 )); \
			_mm_store_pd(zi+8, PACK_FN( _mm_load_pd(xi+8), xmm2 )); \
			_mm_store_pd(zi+10, PACK_FN( _mm_load_pd(xi+10), xmm2 )); \
			_mm_store_pd(zi+12, PACK_FN( _mm_load_pd(xi+12), xmm2 )); \
			_mm_store_pd(zi+14, PACK_FN( _mm_load_pd(xi+14), xmm2 )); \
		} \
		for ( ; n >= 4; n -= 4, zi += 4, xi += 4) { \
			_mm_store_pd(zi, PACK_FN( _mm_load_pd(xi), xmm2 )); \
			_mm_store_pd(zi+2, PACK_FN( _mm_load_pd(xi+2), xmm2 )); \
		} \
		for ( ; n >= 2; n -= 2, zi += 2, xi += 2) { \
			_mm_store_pd(zi, PACK_FN( _mm_load_pd(xi), xmm2 )); \
		} \
		return( std::move(z) ); \
	}

	// z[i] = x[i] + yval.
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, double>
	_array_m128d_fn_x_yval(M, N, T2, operator+, _mm_add_pd)

	// z[i] = x[i] - yval.
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, double>
	_array_m128d_fn_x_yval(M, N, T2, operator-, _mm_sub_pd)

	// z[i] = x[i] * yval.
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, double>
	_array_m128d_fn_x_yval(M, N, T2, operator*, _mm_mul_pd)

	// z[i] = x[i] / yval.
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, double>
	_array_m128d_fn_x_yval(M, N, T2, operator/, _mm_div_pd)

	#define _array_m128d_fn_xval_y(T2, M, N, FUNC, PACK_FN) \
	multi_array_m128d<M,N> FUNC(const T2 xval, const multi_array_m128d<M,N> &y) { \
		const __m128d xmm1 = _mm_set1_pd( static_cast<double>(xval) ); \
		multi_array_m128d<M,N> z; \
		if constexpr (M != 0 && N != 0) { \
			if constexpr (z.actual_size() <= 16) { \
				if constexpr (z.actual_size() > 0) \
					_mm_store_pd(z.data(), PACK_FN( xmm1, _mm_load_pd(y.data()) )); \
				if constexpr (z.actual_size() > 2) \
					_mm_store_pd(z.data()+2, PACK_FN( xmm1, _mm_load_pd(y.data()+2) )); \
				if constexpr (z.actual_size() > 4) \
					_mm_store_pd(z.data()+4, PACK_FN( xmm1, _mm_load_pd(y.data()+4) )); \
				if constexpr (z.actual_size() > 6) \
					_mm_store_pd(z.data()+6, PACK_FN( xmm1, _mm_load_pd(y.data()+6) )); \
				if constexpr (z.actual_size() > 8) \
					_mm_store_pd(z.data()+8, PACK_FN( xmm1, _mm_load_pd(y.data()+8) )); \
				if constexpr (z.actual_size() > 10) \
					_mm_store_pd(z.data()+10, PACK_FN( xmm1, _mm_load_pd(y.data()+10) )); \
				if constexpr (z.actual_size() > 12) \
					_mm_store_pd(z.data()+12, PACK_FN( xmm1, _mm_load_pd(y.data()+12) )); \
				if constexpr (z.actual_size() > 14) \
					_mm_store_pd(z.data()+14, PACK_FN( xmm1, _mm_load_pd(y.data()+14) )); \
				return( z ); \
			} \
		} \
	\
		size_t        n  = z.actual_size(); \
		double       *zi = z.data(); \
		double const *yi = y.data(); \
		for ( ; n >= 16; n -= 16, zi += 16, yi += 16) { \
			_mm_store_pd(zi, PACK_FN( xmm1, _mm_load_pd(yi) )); \
			_mm_store_pd(zi+2, PACK_FN( xmm1, _mm_load_pd(yi+2) )); \
			_mm_store_pd(zi+4, PACK_FN( xmm1, _mm_load_pd(yi+4) )); \
			_mm_store_pd(zi+6, PACK_FN( xmm1, _mm_load_pd(yi+6) )); \
			_mm_store_pd(zi+8, PACK_FN( xmm1, _mm_load_pd(yi+8) )); \
			_mm_store_pd(zi+10, PACK_FN( xmm1, _mm_load_pd(yi+10) )); \
			_mm_store_pd(zi+12, PACK_FN( xmm1, _mm_load_pd(yi+12) )); \
			_mm_store_pd(zi+14, PACK_FN( xmm1, _mm_load_pd(yi+14) )); \
		} \
		for ( ; n >= 4; n -= 4, zi += 4, yi += 4) { \
			_mm_store_pd(zi, PACK_FN( xmm1, _mm_load_pd(yi) )); \
			_mm_store_pd(zi+2, PACK_FN( xmm1, _mm_load_pd(yi+2) )); \
		} \
		for ( ; n >= 2; n -= 2, zi += 2, yi += 2) { \
			_mm_store_pd(zi, PACK_FN( xmm1, _mm_load_pd(yi) )); \
		} \
		return( std::move(z) ); \
	}

	// z[i] = xval + y[i].
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, double>
	_array_m128d_fn_xval_y(T2, M, N, operator+, _mm_add_pd)

	// z[i] = xval - y[i].
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, double>
	_array_m128d_fn_xval_y(T2, M, N, operator-, _mm_sub_pd)

	// z[i] = xval * y[i].
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, double>
	_array_m128d_fn_xval_y(T2, M, N, operator*, _mm_mul_pd)

	// z[i] = xval / y[i].
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, double>
	_array_m128d_fn_xval_y(T2, M, N, operator/, _mm_div_pd)

	#define _array_m128d_assignfn_x_y(M, N, AFUNC, PACK_FN) \
	multi_array_m128d<M,N>& AFUNC(multi_array_m128d<M,N> &x, const multi_array_m128d<M,N> &y) { \
		if constexpr (M != 0 && N != 0) { \
			if constexpr (x.actual_size() <= 16) { \
				if constexpr (x.actual_size() > 0) \
					_mm_store_pd(x.data(), PACK_FN( _mm_load_pd(x.data()), _mm_load_pd(y.data()) )); \
				if constexpr (x.actual_size() > 2) \
					_mm_store_pd(x.data()+2, PACK_FN( _mm_load_pd(x.data()+2), _mm_load_pd(y.data()+2) )); \
				if constexpr (x.actual_size() > 4) \
					_mm_store_pd(x.data()+4, PACK_FN( _mm_load_pd(x.data()+4), _mm_load_pd(y.data()+4) )); \
				if constexpr (x.actual_size() > 6) \
					_mm_store_pd(x.data()+6, PACK_FN( _mm_load_pd(x.data()+6), _mm_load_pd(y.data()+6) )); \
				if constexpr (x.actual_size() > 8) \
					_mm_store_pd(x.data()+8, PACK_FN( _mm_load_pd(x.data()+8), _mm_load_pd(y.data()+8) )); \
				if constexpr (x.actual_size() > 10) \
					_mm_store_pd(x.data()+10, PACK_FN( _mm_load_pd(x.data()+10), _mm_load_pd(y.data()+10) )); \
				if constexpr (x.actual_size() > 12) \
					_mm_store_pd(x.data()+12, PACK_FN( _mm_load_pd(x.data()+12), _mm_load_pd(y.data()+12) )); \
				if constexpr (x.actual_size() > 14) \
					_mm_store_pd(x.data()+14, PACK_FN( _mm_load_pd(x.data()+14), _mm_load_pd(y.data()+14) )); \
				return( x ); \
			} \
		} \
	\
		size_t        n  = x.actual_size(); \
		double       *xi = x.data(); \
		double const *yi = y.data(); \
		for ( ; n >= 16; n -= 16, xi += 16, yi += 16) { \
			_mm_store_pd(xi, PACK_FN( _mm_load_pd(xi), _mm_load_pd(yi) )); \
			_mm_store_pd(xi+2, PACK_FN( _mm_load_pd(xi+2), _mm_load_pd(yi+2) )); \
			_mm_store_pd(xi+4, PACK_FN( _mm_load_pd(xi+4), _mm_load_pd(yi+4) )); \
			_mm_store_pd(xi+6, PACK_FN( _mm_load_pd(xi+6), _mm_load_pd(yi+6) )); \
			_mm_store_pd(xi+8, PACK_FN( _mm_load_pd(xi+8), _mm_load_pd(yi+8) )); \
			_mm_store_pd(xi+10, PACK_FN( _mm_load_pd(xi+10), _mm_load_pd(yi+10) )); \
			_mm_store_pd(xi+12, PACK_FN( _mm_load_pd(xi+12), _mm_load_pd(yi+12) )); \
			_mm_store_pd(xi+14, PACK_FN( _mm_load_pd(xi+14), _mm_load_pd(yi+14) )); \
		} \
		for ( ; n >= 4; n -= 4, xi += 4, yi += 4) { \
			_mm_store_ps(xi, PACK_FN( _mm_load_pd(xi), _mm_load_pd(yi) )); \
			_mm_store_pd(xi+2, PACK_FN( _mm_load_pd(xi+2), _mm_load_pd(yi+2) )); \
		} \
		for ( ; n >= 2; n -= 2, xi += 2, yi += 2) { \
			_mm_store_ps(xi, PACK_FN( _mm_load_pd(xi), _mm_load_pd(yi) )); \
		} \
		return( x ); \
	}

	// x[i] += y[i].
	template<size_t M, size_t N>
	_array_m128d_assignfn_x_y(M, N, operator+=, _mm_add_pd)

	// x[i] -= y[i].
	template<size_t M, size_t N>
	_array_m128d_assignfn_x_y(M, N, operator-=, _mm_sub_pd)

	// x[i] *= y[i].
	template<size_t M, size_t N>
	_array_m128d_assignfn_x_y(M, N, operator*=, _mm_mul_pd)

	// x[i] /= y[i].
	template<size_t M, size_t N>
	_array_m128d_assignfn_x_y(M, N, operator/=, _mm_div_pd)

	#define _array_m128d_assignfn_x_yval(M, N, T2, AFUNC, PACK_FN) \
	multi_array_m128d<M,N>& AFUNC(multi_array_m128d<M,N> &x, const T2 yval) { \
		const __m128d xmm2 = _mm_set1_ps( static_cast<double>(yval) ); \
		if constexpr (M != 0 && N != 0) { \
			if constexpr (x.actual_size() <= 16) { \
				if constexpr (x.actual_size() > 0) \
					_mm_store_pd(x.data(), PACK_FN( _mm_load_pd(x.data()), xmm2 )); \
				if constexpr (x.actual_size() > 2) \
					_mm_store_pd(x.data()+2, PACK_FN( _mm_load_pd(x.data()+2), xmm2 )); \
				if constexpr (x.actual_size() > 4) \
					_mm_store_pd(x.data()+4, PACK_FN( _mm_load_pd(x.data()+4), xmm2 )); \
				if constexpr (x.actual_size() > 6) \
					_mm_store_pd(x.data()+6, PACK_FN( _mm_load_pd(x.data()+6), xmm2 )); \
				if constexpr (x.actual_size() > 8) \
					_mm_store_pd(x.data()+8, PACK_FN( _mm_load_pd(x.data()+8), xmm2 )); \
				if constexpr (x.actual_size() > 10) \
					_mm_store_pd(x.data()+10, PACK_FN( _mm_load_pd(x.data()+10), xmm2 )); \
				if constexpr (x.actual_size() > 12) \
					_mm_store_pd(x.data()+12, PACK_FN( _mm_load_pd(x.data()+12), xmm2 )); \
				if constexpr (x.actual_size() > 14) \
					_mm_store_pd(x.data()+14, PACK_FN( _mm_load_pd(x.data()+14), xmm2 )); \
				return( x ); \
			} \
		} \
	\
		size_t        n  = x.actual_size(); \
		double       *xi = x.data(); \
		for ( ; n >= 16; n -= 16, xi += 16) { \
			_mm_store_pd(xi, PACK_FN( _mm_load_pd(xi), xmm2 )); \
			_mm_store_pd(xi+2, PACK_FN( _mm_load_pd(xi+2), xmm2 )); \
			_mm_store_pd(xi+4, PACK_FN( _mm_load_pd(xi+4), xmm2 )); \
			_mm_store_pd(xi+6, PACK_FN( _mm_load_pd(xi+6), xmm2 )); \
			_mm_store_pd(xi+8, PACK_FN( _mm_load_pd(xi+8), xmm2 )); \
			_mm_store_pd(xi+10, PACK_FN( _mm_load_pd(xi+10), xmm2 )); \
			_mm_store_pd(xi+12, PACK_FN( _mm_load_pd(xi+12), xmm2 )); \
			_mm_store_pd(xi+14, PACK_FN( _mm_load_pd(xi+14), xmm2 )); \
		} \
		for ( ; n >= 4; n -= 4, xi += 4) { \
			_mm_store_pd(xi, PACK_FN( _mm_load_pd(xi), xmm2 )); \
			_mm_store_pd(xi+2, PACK_FN( _mm_load_pd(xi+2), xmm2 )); \
		} \
		for ( ; n >= 2; n -= 2, xi += 2) { \
			_mm_store_pd(xi, PACK_FN( _mm_load_pd(xi), xmm2 )); \
		} \
		return( x ); \
	}

	// x[i] += yval.
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, double>
	_array_m128d_assignfn_x_yval(M, N, T2, operator+=, _mm_add_pd)

	// x[i] -= yval.
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, double>
	_array_m128d_assignfn_x_yval(M, N, T2, operator-=, _mm_sub_pd)

	// x[i] *= yval.
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, double>
	_array_m128d_assignfn_x_yval(M, N, T2, operator*=, _mm_mul_pd)

	// x[i] /= yval.
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, double>
	_array_m128d_assignfn_x_yval(M, N, T2, operator/=, _mm_div_pd)

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
	_array_m128d_fn_x_y(M, N, fmod, _mm_fmod_pd)

	/// z[i] = fmod(x[i], yval).
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, double>
	_array_m128d_fn_x_yval(M, N, T2, fmod, _mm_fmod_pd)

	/// z[i] = fmod(xval, y[i]).
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, double>
	_array_m128d_fn_xval_y(T2, M, N, fmod, _mm_fmod_pd)

	/// z[i] = min(x[i], y[i]).
	template<size_t M, size_t N>
	_array_m128d_fn_x_y(M, N, min, _mm_min_pd)

	/// z[i] = min(x[i], yval).
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, double>
	_array_m128d_fn_x_yval(M, N, T2, min, _mm_min_pd)

	/// z[i] = min(xval, y[i]).
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, double>
	_array_m128d_fn_xval_y(T2, M, N, min, _mm_min_pd)

	/// z[i] = max(x[i], y[i]).
	template<size_t M, size_t N>
	_array_m128d_fn_x_y(M, N, max, _mm_max_pd)

	/// z[i] = max(x[i], yval).
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, double>
	_array_m128d_fn_x_yval(M, N, T2, max, _mm_max_pd)

	/// z[i] = max(xval, y[i]).
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, double>
	_array_m128d_fn_xval_y(T2, M, N, max, _mm_max_pd)

	/// y[i] = abs(x[i]).
	template<size_t M, size_t N>
	_array_m128d_fn_x(M, N, abs, _mm_abs_pd)

	/// y[i] = trunc(x[i]).
	template<size_t M, size_t N>
	_array_m128d_fn_x(M, N, trunc, _mm_trunc_pd)

	/// y[i] = floor(x[i]).
	template<size_t M, size_t N>
	_array_m128d_fn_x(M, N, floor, _mm_floor_pd)

	/// y[i] = ceil(x[i]).
	template<size_t M, size_t N>
	_array_m128d_fn_x(M, N, ceil, _mm_ceil_pd)

	/// y[i] = round(x[i], mode).
	template<size_t M, size_t N, int mode = _MM_ROUND_MODE_NEAREST>
	multi_array_m128d<M,N> round(const multi_array_m128d<M,N> &x, std::integral_constant<int,mode> _Unused = {}) {
		multi_array_m128d<M,N> y;
		if constexpr (M != 0 && N != 0) {
			if constexpr (y.actual_size() <= 16) {
				if constexpr (y.actual_size() > 0)
					_mm_store_pd(y.data(), _mm_round_pd( _mm_load_pd(x.data()), mode ));
				if constexpr (y.actual_size() > 2)
					_mm_store_pd(y.data()+2, _mm_round_pd( _mm_load_pd(x.data()+2), mode ));
				if constexpr (y.actual_size() > 4)
					_mm_store_pd(y.data()+4, _mm_round_pd( _mm_load_pd(x.data()+4), mode ));
				if constexpr (y.actual_size() > 6)
					_mm_store_pd(y.data()+6, _mm_round_pd( _mm_load_pd(x.data()+6), mode ));
				if constexpr (y.actual_size() > 8)
					_mm_store_pd(y.data()+8, _mm_round_pd( _mm_load_pd(x.data()+8), mode ));
				if constexpr (y.actual_size() > 10)
					_mm_store_pd(y.data()+10, _mm_round_pd( _mm_load_pd(x.data()+10), mode ));
				if constexpr (y.actual_size() > 12)
					_mm_store_pd(y.data()+12, _mm_round_pd( _mm_load_pd(x.data()+12), mode ));
				if constexpr (y.actual_size() > 14)
					_mm_store_pd(y.data()+14, _mm_round_pd( _mm_load_pd(x.data()+14), mode ));
				return( y );
			}
		}

		size_t        n  = y.actual_size();
		double       *yi = y.data();
		double const *xi = x.data();
		for ( ; n >= 16; n -= 16, yi += 16, xi += 16) {
			_mm_store_pd(yi, _mm_round_pd( _mm_load_pd(xi), mode ));
			_mm_store_pd(yi+2, _mm_round_pd( _mm_load_pd(xi+2), mode ));
			_mm_store_pd(yi+4, _mm_round_pd( _mm_load_pd(xi+4), mode ));
			_mm_store_pd(yi+6, _mm_round_pd( _mm_load_pd(xi+6), mode ));
			_mm_store_pd(yi+8, _mm_round_pd( _mm_load_pd(xi+8), mode ));
			_mm_store_pd(yi+10, _mm_round_pd( _mm_load_pd(xi+10), mode ));
			_mm_store_pd(yi+12, _mm_round_pd( _mm_load_pd(xi+12), mode ));
			_mm_store_pd(yi+14, _mm_round_pd( _mm_load_pd(xi+14), mode ));
		}
		for ( ; n >= 4; n -= 4, yi += 4, xi += 4) {
			_mm_store_pd(yi, _mm_round_pd( _mm_load_pd(xi), mode ));
			_mm_store_pd(yi+2, _mm_round_pd( _mm_load_pd(xi+2), mode ));
		}
		for ( ; n >= 2; n -= 2, yi += 2, xi += 2) {
			_mm_store_pd(yi, _mm_round_pd( _mm_load_pd(xi), mode ));
		}
		return( std::move(y) );
	}

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
	_array_m128d_fn_x(M, N, sqrt, _mm_sqrt_pd)

	/// y[i] = cbrt(x[i]).
	template<size_t M, size_t N>
	_array_m128d_fn_x(M, N, cbrt, _mm_cbrt_pd)

	/// y[i] = exp(x[i]).
	template<size_t M, size_t N>
	_array_m128d_fn_x(M, N, exp, _mm_exp_pd)

	/// y[i] = exp2(x[i]).
	template<size_t M, size_t N>
	_array_m128d_fn_x(M, N, exp2, _mm_exp2_pd)

	/// y[i] = log(x[i]).
	template<size_t M, size_t N>
	_array_m128d_fn_x(M, N, log, _mm_log_pd)

	/// y[i] = log2(x[i]).
	template<size_t M, size_t N>
	_array_m128d_fn_x(M, N, log2, _mm_log2_pd)

	/// z[i] = pow(x[i], y[i]).
	template<size_t M, size_t N>
	_array_m128d_fn_x_y(M, N, pow, _mm_pow_pd)

	/// z[i] = pow(x[i], yval).
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, double>
	_array_m128d_fn_x_yval(M, N, T2, pow, _mm_pow_pd)

	/// z[i] = pow(xval, y[i]).
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, double>
	_array_m128d_fn_xval_y(T2, M, N, pow, _mm_pow_pd)

	/// y[i] = sin(x[i]).
	template<size_t M, size_t N>
	_array_m128d_fn_x(M, N, sin, _mm_sin_pd)

	/// y[i] = cos(x[i]).
	template<size_t M, size_t N>
	_array_m128d_fn_x(M, N, cos, _mm_cos_pd)

	/// y[i] = tan(x[i]).
	template<size_t M, size_t N>
	_array_m128d_fn_x(M, N, tan, _mm_tan_pd)

	/// y[i] = asin(x[i]).
	template<size_t M, size_t N>
	_array_m128d_fn_x(M, N, asin, _mm_asin_pd)

	/// y[i] = acos(x[i]).
	template<size_t M, size_t N>
	_array_m128d_fn_x(M, N, acos, _mm_acos_pd)

	/// y[i] = atan(x[i]).
	template<size_t M, size_t N>
	_array_m128d_fn_x(M, N, atan, _mm_atan_pd)

	/// z[i] = atan2(x[i], y[i]).
	template<size_t M, size_t N>
	_array_m128d_fn_x_y(M, N, atan2, _mm_atan2_pd)

	/// z[i] = atan2(x[i], yval).
	template<size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, double>
	_array_m128d_fn_x_yval(M, N, T2, atan2, _mm_atan2_pd)

	/// z[i] = atan2(xval, y[i]).
	template<typename T2, size_t M, size_t N>
		requires std::convertible_to<T2, double>
	_array_m128d_fn_xval_y(T2, M, N, atan2, _mm_atan2_pd)
#endif
}