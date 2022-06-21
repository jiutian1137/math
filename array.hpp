#pragma once

#include "multi_array.hpp"

#include <cassert>
#include <concepts>// std::convertible_to<..,..>

#include <cmath>// math
#include <utility>// std::min, std::max


/// Operations of Array, optimized with unroll(16 times & 4 times).
///@license Free 
///@review 2022-5-14 
///@author LongJiangnan, Jiang1998Nan@outlook.com 
#define _MATH_ARRAY_

namespace math {
#ifndef unroll_16x_4x_for
	/// For-Loop unrooled by 16 times and 4 times.
	#define unroll_16x_4x_for(init_statement, condition16x, iteration16x, condition4x, iteration4x, condition, iteration, statement) \
		init_statement; \
		for ( ; condition16x; iteration16x) { \
			{ constexpr int i = 0; statement; } \
			{ constexpr int i = 1; statement; } \
			{ constexpr int i = 2; statement; } \
			{ constexpr int i = 3; statement; } \
			{ constexpr int i = 4; statement; } \
			{ constexpr int i = 5; statement; } \
			{ constexpr int i = 6; statement; } \
			{ constexpr int i = 7; statement; } \
			{ constexpr int i = 8; statement; } \
			{ constexpr int i = 9; statement; } \
			{ constexpr int i = 10; statement; } \
			{ constexpr int i = 11; statement; } \
			{ constexpr int i = 12; statement; } \
			{ constexpr int i = 13; statement; } \
			{ constexpr int i = 14; statement; } \
			{ constexpr int i = 15; statement; } \
		} \
		for ( ; condition4x; iteration4x) { \
			{ constexpr int i = 0; statement; } \
			{ constexpr int i = 1; statement; } \
			{ constexpr int i = 2; statement; } \
			{ constexpr int i = 3; statement; } \
		} \
		for ( ; condition; iteration) { \
			{ constexpr int i = 0; statement; } \
		}
#endif

#ifndef epconj
	/// Expression Conjunction.
	#define epconj(...) __VA_ARGS__
#endif

	/// y[i] = +x[i].
	template<typename T, size_t M, size_t N>
	multi_array<T,M,N> operator+(const multi_array<T,M,N>& x) {
		constexpr size_t MN = M * N;
		if constexpr (MN == 4) { return { +x[0], +x[1], +x[2], +x[3] }; }
		else if constexpr (MN == 3) { return { +x[0], +x[1], +x[2] }; }
		else if constexpr (MN == 2) { return { +x[0], +x[1] }; }
		else if constexpr (MN == 1) { return { +x[0] }; }
		multi_array<T,M,N> y;
		multi_array_alloc(y, x);
		unroll_16x_4x_for(
			epconj(size_t n = y.size(); T* yi = y.data(); const T* xi = x.data()),
			epconj(n >= 16), epconj(n-=16, yi+=16, xi+=16),
			epconj(n >= 4), epconj(n-=4, yi+=4, xi+=4),
			epconj(n != 0), epconj(--n, ++yi, ++xi),
			yi[i] = +xi[i]
		);
		return std::move(y);
	}

	/// y[i] = -x[i].
	template<typename T, size_t M, size_t N>
	multi_array<T,M,N> operator-(const multi_array<T,M,N>& x) {
		constexpr size_t MN = M * N;
		if constexpr (MN == 4) { return { -x[0], -x[1], -x[2], -x[3] }; }
		else if constexpr (MN == 3) { return { -x[0], -x[1], -x[2] }; }
		else if constexpr (MN == 2) { return { -x[0], -x[1] }; }
		else if constexpr (MN == 1) { return { -x[0] }; }
		multi_array<T,M,N> y;
		multi_array_alloc(y, x);
		unroll_16x_4x_for(
			epconj(size_t n = y.size(); T* yi = y.data(); const T* xi = x.data()),
			epconj(n >= 16), epconj(n-=16, yi+=16, xi+=16),
			epconj(n >= 4), epconj(n-=4, yi+=4, xi+=4),
			epconj(n != 0), epconj(--n, ++yi, ++xi),
			yi[i] = -xi[i]
		);
		return std::move(y);
	}

	#define _array_x_op_y(T, M, N, OP) \
	math::multi_array<T,M,N> operator##OP(const ::math::multi_array<T,M,N> &x, const ::math::multi_array<T,M,N> &y) { \
		constexpr size_t MN = M * N; \
		if constexpr (MN == 4) { return { x[0] OP y[0], x[1] OP y[1], x[2] OP y[2], x[3] OP y[3] }; } \
		else if constexpr (MN == 3) { return { x[0] OP y[0], x[1] OP y[1], x[2] OP y[2] }; } \
		else if constexpr (MN == 2) { return { x[0] OP y[0], x[1] OP y[1] }; } \
		else if constexpr (MN == 1) { return { x[0] OP y[0] }; } \
		::math::multi_array<T,M,N> z; \
		multi_array_alloc(z, x); \
		unroll_16x_4x_for( \
			epconj(size_t n = z.size(); T* zi = z.data(); const T* xi = x.data(); const T* yi = y.data()), \
			epconj(n >= 16), epconj(n-=16, zi+=16, xi+=16, yi+=16), \
			epconj(n >= 4), epconj(n-=4, zi+=4, xi+=4, yi+=4), \
			epconj(n != 0), epconj(--n, ++zi, ++xi, ++yi), \
			zi[i] = xi[i] OP yi[i] \
		); \
		return std::move(z); \
	}

	/// z[i] = x[i] & y[i].
	template<typename T, size_t M, size_t N>
		requires requires(T __x0, T __y0){ __x0 & __y0; }
	_array_x_op_y(T, M, N, &)

	/// z[i] = x[i] | y[i].
	template<typename T, size_t M, size_t N>
		requires requires(T __x0, T __y0){ __x0 | __y0; }
	_array_x_op_y(T, M, N, |)

	/// z[i] = x[i] ^ y[i].
	template<typename T, size_t M, size_t N>
		requires requires(T __x0, T __y0){ __x0 ^ __y0; }
	_array_x_op_y(T, M, N, ^)

	/// z[i] = x[i] + y[i].
	template<typename T, size_t M, size_t N>
		requires requires(T __x0, T __y0){ __x0 + __y0; }
	_array_x_op_y(T, M, N, +)

	/// z[i] = x[i] - y[i].
	template<typename T, size_t M, size_t N>
		requires requires(T __x0, T __y0){ __x0 - __y0; }
	_array_x_op_y(T, M, N, -)

	/// z[i] = x[i] * y[i].
	template<typename T, size_t M, size_t N>
		requires requires(T __x0, T __y0){ __x0 * __y0; }
	_array_x_op_y(T, M, N, *)

	/// z[i] = x[i] / y[i].
	template<typename T, size_t M, size_t N>
		requires requires(T __x0, T __y0){ __x0 / __y0; }
	_array_x_op_y(T, M, N, /)

	/// z[i] = x[i] % y[i].
	template<typename T, size_t M, size_t N>
		requires requires(T __x0, T __y0){ __x0 % __y0; }
	_array_x_op_y(T, M, N, %)

	#define _array_x_op_yval(T, M, N, T2, OP) \
	math::multi_array<T,M,N> operator##OP(const ::math::multi_array<T,M,N> &x, const T2 &yval) { \
		constexpr size_t MN = M * N; \
		if constexpr (MN == 4) { \
			return { static_cast<T>(x[0] OP yval), static_cast<T>(x[1] OP yval), static_cast<T>(x[2] OP yval), static_cast<T>(x[3] OP yval) }; } \
		else if constexpr (MN == 3) { \
			return { static_cast<T>(x[0] OP yval), static_cast<T>(x[1] OP yval), static_cast<T>(x[2] OP yval) }; } \
		else if constexpr (MN == 2) { \
			return { static_cast<T>(x[0] OP yval), static_cast<T>(x[1] OP yval) }; } \
		else if constexpr (MN == 1) { \
			return { static_cast<T>(x[0] OP yval) }; } \
		::math::multi_array<T,M,N> z; \
		multi_array_alloc(z, x); \
		unroll_16x_4x_for( \
			epconj(size_t n = z.size(); T* zi = z.data(); const T* xi = x.data()), \
			epconj(n >= 16), epconj(n-=16, zi+=16, xi+=16), \
			epconj(n >= 4), epconj(n-=4, zi+=4, xi+=4), \
			epconj(n != 0), epconj(--n, ++zi, ++xi), \
			zi[i] = static_cast<T>(xi[i] OP yval) \
		); \
		return std::move(z); \
	}

	/// z[i] = x[i] & yval.
	template<typename T, size_t M, size_t N, typename T2>
		requires requires(T __x0, T2 __y0){ static_cast<T>(__x0 & __y0); }
	_array_x_op_yval(T, M, N, T2, &)

	/// z[i] = x[i] | yval.
	template<typename T, size_t M, size_t N, typename T2>
		requires requires(T __x0, T2 __y0){ static_cast<T>(__x0 | __y0); }
	_array_x_op_yval(T, M, N, T2, |)

	/// z[i] = x[i] ^ yval.
	template<typename T, size_t M, size_t N, typename T2>
		requires requires(T __x0, T2 __y0){ static_cast<T>(__x0 ^ __y0); }
	_array_x_op_yval(T, M, N, T2, ^)

	/// z[i] = x[i] + yval.
	template<typename T, size_t M, size_t N, typename T2>
		requires requires(T __x0, T2 __y0){ static_cast<T>(__x0 + __y0); }
	_array_x_op_yval(T, M, N, T2, +)

	/// z[i] = x[i] - yval.
	template<typename T, size_t M, size_t N, typename T2>
		requires requires(T __x0, T2 __y0){ static_cast<T>(__x0 - __y0); }
	_array_x_op_yval(T, M, N, T2, -)

	/// z[i] = x[i] * yval.
	template<typename T, size_t M, size_t N, typename T2>
		requires requires(T __x0, T2 __y0){ static_cast<T>(__x0 * __y0); }
	_array_x_op_yval(T, M, N, T2, *)

	/// z[i] = x[i] / yval.
	template<typename T, size_t M, size_t N, typename T2>
		requires requires(T __x0, T2 __y0){ static_cast<T>(__x0 / __y0); }
	_array_x_op_yval(T, M, N, T2, /)

	/// z[i] = x[i] % yval.
	template<typename T, size_t M, size_t N, typename T2>
		requires requires(T __x0, T2 __y0){ static_cast<T>(__x0 % __y0); }
	_array_x_op_yval(T, M, N, T2, %)

	#define _array_xval_op_y(T2, T, M, N, OP) \
	math::multi_array<T,M,N> operator##OP(const T2 &xval, const ::math::multi_array<T,M,N> &y) { \
		constexpr size_t MN = M * N; \
		if constexpr (MN == 4) { \
			return { static_cast<T>(xval OP y[0]), static_cast<T>(xval OP y[1]), static_cast<T>(xval OP y[2]), static_cast<T>(xval OP y[3]) }; } \
		else if constexpr (MN == 3) { \
			return { static_cast<T>(xval OP y[0]), static_cast<T>(xval OP y[1]), static_cast<T>(xval OP y[2]) }; } \
		else if constexpr (MN == 2) { \
			return { static_cast<T>(xval OP y[0]), static_cast<T>(xval OP y[1]) }; } \
		else if constexpr (MN == 1) { \
			return { static_cast<T>(xval OP y[0]) }; } \
		::math::multi_array<T,M,N> z; \
		multi_array_alloc(z, y); \
		unroll_16x_4x_for( \
			epconj(size_t n = z.size(); T* zi = z.data(); const T* yi = y.data()), \
			epconj(n >= 16), epconj(n-=16, zi+=16, yi+=16), \
			epconj(n >= 4), epconj(n-=4, zi+=4, yi+=4), \
			epconj(n != 0), epconj(--n, ++zi, ++yi), \
			zi[i] = static_cast<T>(xval OP yi[i]) \
		); \
		return std::move(z); \
	}

	/// z[i] = xval & y[i].
	template<typename T2, typename T, size_t M, size_t N>
		requires requires(T2 __x0, T __y0){ static_cast<T>(__x0 & __y0); }
	_array_xval_op_y(T2, T, M, N, &)

	/// z[i] = xval | y[i].
	template<typename T2, typename T, size_t M, size_t N>
		requires requires(T2 __x0, T __y0){ static_cast<T>(__x0 | __y0); }
	_array_xval_op_y(T2, T, M, N, |)

	/// z[i] = xval ^ y[i].
	template<typename T2, typename T, size_t M, size_t N>
		requires requires(T2 __x0, T __y0){ static_cast<T>(__x0 ^ __y0); }
	_array_xval_op_y(T2, T, M, N, ^)

	/// z[i] = xval + y[i].
	template<typename T2, typename T, size_t M, size_t N>
		requires requires(T2 __x0, T __y0){ static_cast<T>(__x0 + __y0); }
	_array_xval_op_y(T2, T, M, N, +)

	/// z[i] = xval - y[i].
	template<typename T2, typename T, size_t M, size_t N>
		requires requires(T2 __x0, T __y0){ static_cast<T>(__x0 - __y0); }
	_array_xval_op_y(T2, T, M, N, -)

	/// z[i] = xval * y[i].
	template<typename T2, typename T, size_t M, size_t N>
		requires requires(T2 __x0, T __y0){ static_cast<T>(__x0 * __y0); }
	_array_xval_op_y(T2, T, M, N, *)

	/// z[i] = xval / y[i].
	template<typename T2, typename T, size_t M, size_t N>
		requires requires(T2 __x0, T __y0){ static_cast<T>(__x0 / __y0); }
	_array_xval_op_y(T2, T, M, N, /)

	/// z[i] = xval % y[i].
	template<typename T2, typename T, size_t M, size_t N>
		requires requires(T2 __x0, T __y0){ static_cast<T>(__x0 % __y0); }
	_array_xval_op_y(T2, T, M, N, %)


	#define _array_x_assignop_y(T, M, N, AOP) \
	math::multi_array<T,M,N>& operator##AOP(::math::multi_array<T,M,N> &x, const ::math::multi_array<T,M,N> &y) { \
		unroll_16x_4x_for( \
			epconj(size_t n = x.size(); T* xi = x.data(); const T* yi = y.data()), \
			epconj(n >= 16), epconj(n-=16, xi+=16, yi+=16), \
			epconj(n >= 4), epconj(n-=4, xi+=4, yi+=4), \
			epconj(n != 0), epconj(--n, ++xi, ++yi), \
			xi[i] AOP yi[i] \
		); \
		return x; \
	}

	/// x[i] &= y[i].
	template<typename T, size_t M, size_t N>
		requires requires(T __x0, T __y0){ __x0 &= __y0; }
	_array_x_assignop_y(T, M, N, &=)

	/// x[i] &= y[i].
	template<typename T, size_t M, size_t N>
		requires requires(T __x0, T __y0){ __x0 |= __y0; }
	_array_x_assignop_y(T, M, N, |=)

	/// x[i] ^= y[i].
	template<typename T, size_t M, size_t N>
		requires requires(T __x0, T __y0){ __x0 ^= __y0; }
	_array_x_assignop_y(T, M, N, ^=)

	/// x[i] += y[i].
	template<typename T, size_t M, size_t N>
		requires requires(T __x0, T __y0){ __x0 += __y0; }
	_array_x_assignop_y(T, M, N, +=)

	/// x[i] -= y[i].
	template<typename T, size_t M, size_t N>
		requires requires(T __x0, T __y0){ __x0 -= __y0; }
	_array_x_assignop_y(T, M, N, -=)

	/// x[i] *= y[i].
	template<typename T, size_t M, size_t N>
		requires requires(T __x0, T __y0){ __x0 *= __y0; }
	_array_x_assignop_y(T, M, N, *=)

	/// x[i] /= y[i].
	template<typename T, size_t M, size_t N>
		requires requires(T __x0, T __y0){ __x0 /= __y0; }
	_array_x_assignop_y(T, M, N, /=)

	/// x[i] %= y[i].
	template<typename T, size_t M, size_t N>
		requires requires(T __x0, T __y0){ __x0 %= __y0; }
	_array_x_assignop_y(T, M, N, %=)

	#define _array_x_assignop_yval(T, M, N, T2, AOP) \
	math::multi_array<T,M,N>& operator##AOP(::math::multi_array<T,M,N> &x, const T2 &yval) { \
		unroll_16x_4x_for( \
			epconj(size_t n = x.size(); T* xi = x.data()), \
			epconj(n >= 16), epconj(n-=16, xi+=16), \
			epconj(n >= 4), epconj(n-=4, xi+=4), \
			epconj(n != 0), epconj(--n, ++xi), \
			xi[i] AOP yval \
		); \
		return x; \
	}

	/// x[i] &= yval.
	template<typename T, size_t M, size_t N, typename T2>
		requires requires(T __x0, T2 __y0) { __x0 &= __y0; }
	_array_x_assignop_yval(T, M, N, T2, &=)

	/// x[i] |= yval.
	template<typename T, size_t M, size_t N, typename T2>
		requires requires(T __x0, T2 __y0) { __x0 |= __y0; }
	_array_x_assignop_yval(T, M, N, T2, |=)

	/// x[i] ^= yval.
	template<typename T, size_t M, size_t N, typename T2>
		requires requires(T __x0, T2 __y0) { __x0 ^= __y0; }
	_array_x_assignop_yval(T, M, N, T2, ^=)

	/// x[i] += yval.
	template<typename T, size_t M, size_t N, typename T2>
		requires requires(T __x0, T2 __y0) { __x0 += __y0; }
	_array_x_assignop_yval(T, M, N, T2, +=)

	/// x[i] -= yval.
	template<typename T, size_t M, size_t N, typename T2>
		requires requires(T __x0, T2 __y0) { __x0 -= __y0; }
	_array_x_assignop_yval(T, M, N, T2, -=)

	/// x[i] *= yval.
	template<typename T, size_t M, size_t N, typename T2>
		requires requires(T __x0, T2 __y0) { __x0 *= __y0; }
	_array_x_assignop_yval(T, M, N, T2, *=)

	/// x[i] /= yval.
	template<typename T, size_t M, size_t N, typename T2>
		requires requires(T __x0, T2 __y0) { __x0 /= __y0; }
	_array_x_assignop_yval(T, M, N, T2, /=)

	/// x[i] %= yval.
	template<typename T, size_t M, size_t N, typename T2>
		requires requires(T __x0, T2 __y0) { __x0 %= __y0; }
	_array_x_assignop_yval(T, M, N, T2, %=)


	#define _array_fn_x(T, M, N, FN) \
	math::multi_array<T,M,N> FN(const ::math::multi_array<T,M,N> &x) { \
		constexpr size_t MN = M * N; \
		if constexpr (MN == 4) { return { FN( x[0] ), FN( x[1] ), FN( x[2] ), FN( x[3] ) }; } \
		else if constexpr (MN == 3) { return { FN( x[0] ), FN( x[1] ), FN( x[2] ) }; } \
		else if constexpr (MN == 2) { return { FN( x[0] ), FN( x[1] ) }; } \
		else if constexpr (MN == 1) { return { FN( x[0] ) }; } \
		::math::multi_array<T,M,N> y; \
		multi_array_alloc(y, x); \
		unroll_16x_4x_for( \
			epconj(size_t n = y.size(); T* yi = y.data(); const T* xi = x.data()), \
			epconj(n >= 16), epconj(n-=16, yi+=16, xi+=16), \
			epconj(n >= 4), epconj(n-=4, yi+=4, xi+=4), \
			epconj(n != 0), epconj(--n, ++yi, ++xi), \
			yi[i] = FN( xi[i] ) \
		); \
		return std::move(y); \
	}

	#define _array_fn_x_y(T, M, N, FN) \
	math::multi_array<T,M,N> FN(const ::math::multi_array<T,M,N> &x, const ::math::multi_array<T,M,N> &y) { \
		constexpr size_t MN = M * N; \
		if constexpr (MN == 4) { return { FN( x[0],y[0] ), FN( x[1],y[1] ), FN( x[2],y[2] ), FN( x[3],y[3] ) }; } \
		else if constexpr (MN == 3) { return { FN( x[0],y[0] ), FN( x[1],y[1] ), FN( x[2],y[2] ) }; } \
		else if constexpr (MN == 2) { return { FN( x[0],y[0] ), FN( x[1],y[1] ) }; } \
		else if constexpr (MN == 1) { return { FN( x[0],y[0] ) }; } \
		::math::multi_array<T,M,N> z; \
		multi_array_alloc(z, x); \
		unroll_16x_4x_for( \
			epconj(size_t n = z.size(); T* zi = z.data(); const T* xi = x.data(); const T* yi = y.data()), \
			epconj(n >= 16), epconj(n-=16, zi+=16, xi+=16, yi+=16), \
			epconj(n >= 4), epconj(n-=4, zi+=4, xi+=4, yi+=4), \
			epconj(n != 0), epconj(--n, ++zi, ++xi, ++yi), \
			zi[i] = FN( xi[i], yi[i] ) \
		); \
		return std::move(z); \
	}

	#define _array_fn_x_yval_strict(T, M, N, T2, FN) \
	math::multi_array<T,M,N> FN(const ::math::multi_array<T,M,N> &x, const T2 &yval_) { \
		const T yval = static_cast<T>(yval_); \
		constexpr size_t MN = M * N; \
		if constexpr(MN == 4) { return { FN( x[0],yval ), FN( x[1],yval ), FN( x[2],yval ), FN( x[3],yval ) }; } \
		else if constexpr(MN == 3) { return { FN( x[0],yval ), FN( x[1],yval ), FN( x[2],yval ) }; } \
		else if constexpr(MN == 2) { return { FN( x[0],yval ), FN( x[1],yval ) }; } \
		else if constexpr(MN == 1) { return { FN( x[0],yval ) }; } \
		::math::multi_array<T,M,N> z; \
		multi_array_alloc(z, x); \
		unroll_16x_4x_for( \
			epconj(size_t n = z.size(); T* zi = z.data(); const T* xi = x.data()), \
			epconj(n >= 16), epconj(n-=16, zi+=16, xi+=16), \
			epconj(n >= 4), epconj(n-=4, zi+=4, xi+=4), \
			epconj(n != 0), epconj(--n, ++zi, ++xi), \
			zi[i] = FN( xi[i], yval ) \
		); \
		return std::move(z); \
	}

	#define _array_fn_xval_y_strict(T2, T, M, N, FN) \
	math::multi_array<T,M,N> FN(const T2 &xval_, const ::math::multi_array<T,M,N> &y) { \
		const T xval = static_cast<T>(xval_); \
		constexpr size_t MN = M * N; \
		if constexpr (MN == 4) { return { FN( xval,y[0] ), FN( xval,y[1] ), FN( xval,y[2] ), FN( xval,y[3] ) }; } \
		else if constexpr (MN == 3) { return { FN( xval,y[0] ), FN( xval,y[1] ), FN( xval,y[2] ) }; } \
		else if constexpr (MN == 2) { return { FN( xval,y[0] ), FN( xval,y[1] ) }; } \
		else if constexpr (MN == 1) { return { FN( xval,y[0] ) }; } \
		::math::multi_array<T,M,N> z; \
		multi_array_alloc(z, y); \
		unroll_16x_4x_for( \
			epconj(size_t n = z.size(); T* zi = z.data(); const T* yi = y.data()), \
			epconj(n >= 16), epconj(n-=16, zi+=16, yi+=16), \
			epconj(n >= 4), epconj(n-=4, zi+=4, yi+=4), \
			epconj(n != 0), epconj(--n, ++zi, ++yi), \
			zi[i] = FN( xval, yi[i] ) \
		); \
		return std::move(z); \
	}

	using std::fmod;
	using std::min;
	using std::max;
	using std::abs;
	using std::trunc;
	using std::floor;
	using std::ceil;
	using std::round;

	/// z[i] = fmod(x[i], y[i]).
	template<typename T, size_t M, size_t N>
	_array_fn_x_y(T, M, N, fmod)

	/// z[i] = fmod(x[i], yval).
	template<typename T, size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, T>
	_array_fn_x_yval_strict(T, M, N, T2, fmod)

	/// z[i] = fmod(xval, y[i]).
	template<typename T2, typename T, size_t M, size_t N>
		requires std::convertible_to<T2, T>
	_array_fn_xval_y_strict(T2, T, M, N, fmod)

	/// z[i] = min(x[i], y[i]).
	template<typename T, size_t M, size_t N>
	_array_fn_x_y(T, M, N, min)

	/// z[i] = min(x[i], yval).
	template<typename T, size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, T>
	_array_fn_x_yval_strict(T, M, N, T2, min)

	/// z[i] = min(xval, y[i]).
	template<typename T2, typename T, size_t M, size_t N>
		requires std::convertible_to<T2, T>
	_array_fn_xval_y_strict(T2, T, M, N, min)

	/// z[i] = max(x[i], y[i]).
	template<typename T, size_t M, size_t N>
	_array_fn_x_y(T, M, N, max)

	/// z[i] = max(x[i], yval).
	template<typename T, size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, T>
	_array_fn_x_yval_strict(T, M, N, T2, max)

	/// z[i] = max(xval, y[i]).
	template<typename T2, typename T, size_t M, size_t N>
		requires std::convertible_to<T2, T>
	_array_fn_xval_y_strict(T2, T, M, N, max)

	/// y[i] = abs(x[i]).
	template<typename T, size_t M, size_t N> 
	_array_fn_x(T, M, N, abs)

	/// y[i] = trunc(x[i]).
	template<typename T, size_t M, size_t N>
	_array_fn_x(T, M, N, trunc)

	/// y[i] = floor(x[i]).
	template<typename T, size_t M, size_t N>
	_array_fn_x(T, M, N, floor)

	/// y[i] = ceil(x[i]).
	template<typename T, size_t M, size_t N>
	_array_fn_x(T, M, N, ceil)

	/// y[i] = round(x[i]).
	template<typename T, size_t M, size_t N>
	_array_fn_x(T, M, N, round)

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
	template<typename T, size_t M, size_t N>
	_array_fn_x(T, M, N, sqrt)

	/// y[i] = cbrt(x[i]).
	template<typename T, size_t M, size_t N>
	_array_fn_x(T, M, N, cbrt)

	/// y[i] = exp(x[i]).
	template<typename T, size_t M, size_t N>
	_array_fn_x(T, M, N, exp)

	/// y[i] = exp2(x[i]).
	template<typename T, size_t M, size_t N>
	_array_fn_x(T, M, N, exp2)

	/// y[i] = log(x[i]).
	template<typename T, size_t M, size_t N>
	_array_fn_x(T, M, N, log)

	/// y[i] = log2(x[i]).
	template<typename T, size_t M, size_t N>
	_array_fn_x(T, M, N, log2)

	/// z[i] = pow(x[i], y[i]).
	template<typename T, size_t M, size_t N>
	_array_fn_x_y(T, M, N, pow)

	/// z[i] = pow(x[i], yval).
	template<typename T, size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, T>
	_array_fn_x_yval_strict(T, M, N, T2, pow)

	//_array_fn_xval_y(pow)

	/// y[i] = sin(x[i]).
	template<typename T, size_t M, size_t N>
	_array_fn_x(T, M, N, sin)

	/// y[i] = cos(x[i]).
	template<typename T, size_t M, size_t N>
	_array_fn_x(T, M, N, cos)

	/// y[i] = tan(x[i]).
	template<typename T, size_t M, size_t N>
	_array_fn_x(T, M, N, tan)

	/// y[i] = asin(x[i]).
	template<typename T, size_t M, size_t N>
	_array_fn_x(T, M, N, asin)

	/// y[i] = acos(x[i]).
	template<typename T, size_t M, size_t N>
	_array_fn_x(T, M, N, acos)

	/// y[i] = atan(x[i]).
	template<typename T, size_t M, size_t N>
	_array_fn_x(T, M, N, atan)

	/// z[i] = atan2(x[i], y[i]).
	template<typename T, size_t M, size_t N>
	_array_fn_x_y(T, M, N, atan2)

	/// z[i] = atan2(x[i], yval).
	template<typename T, size_t M, size_t N, typename T2>
		requires std::convertible_to<T2, T>
	_array_fn_x_yval_strict(T, M, N, T2, atan2)

	/// z[i] = atan2(xval, y[i]).
	template<typename T2, typename T, size_t M, size_t N>
		requires std::convertible_to<T2, T>
	_array_fn_xval_y_strict(T2, T, M, N, atan2)


	using std::modf;
	using std::frexp;
	using std::ldexp;

	/// z[i] = frac(x[i]), 
	/// (*y)[i] = trunc(x[i]). 
	template<typename T, size_t M, size_t N>
	multi_array<T,M,N> modf(const multi_array<T,M,N> &x, multi_array<T,M,N> *y) {
		constexpr size_t MN = M * N;
		if constexpr (MN == 4) { return { modf( x[0], &((*y)[0]) ), modf( x[1], &((*y)[1]) ), modf( x[2], &((*y)[2]) ), modf( x[3], &((*y)[3]) ) }; }
		else if constexpr (MN == 3) { return { modf( x[0], &((*y)[0]) ), modf( x[1], &((*y)[1]) ), modf( x[2], &((*y)[2]) ) }; }
		else if constexpr (MN == 2) { return { modf( x[0], &((*y)[0]) ), modf( x[1], &((*y)[1]) ) }; }
		else if constexpr (MN == 1) { return { modf( x[0], &((*y)[0]) ) }; }
		multi_array<T,M,N> z;
		multi_array_alloc(z, x);
		unroll_16x_4x_for(
			epconj(size_t n = z.size(); T* zi = z.data(); const T* xi = x.data(); T* yi = y->data()),
			epconj(n >= 16), epconj(n-=16, zi+=16, xi+=16, yi+=16),
			epconj(n >= 4), epconj(n-=4, zi+=4, xi+=4, yi+=4),
			epconj(n != 0), epconj(--n, ++zi, ++xi, ++yi),
			zi[i] = modf( xi[i], yi+i )
		);
		return std::move(z);
	}

	/// z[i] = x[i].signed_significant() = x[i]/exp2(int(log2(x[i]))), 
	/// (*y)[i] = x[i].exponent() = int(log2(x[i])). 
	template<typename T, size_t M, size_t N>
	multi_array<T,M,N> frexp(const multi_array<T,M,N> &x, multi_array<int,M,N> *y) {
		constexpr size_t MN = M * N;
		if constexpr (MN == 4) { return { frexp( x[0], &((*y)[0]) ), frexp( x[1], &((*y)[1]) ), frexp( x[2], &((*y)[2]) ), frexp( x[3], &((*y)[3]) ) }; }
		else if constexpr (MN == 3) { return { frexp( x[0], &((*y)[0]) ), frexp( x[1], &((*y)[1]) ), frexp( x[2], &((*y)[2]) ) }; }
		else if constexpr (MN == 2) { return { frexp( x[0], &((*y)[0]) ), frexp( x[1], &((*y)[1]) ) }; }
		else if constexpr (MN == 1) { return { frexp( x[0], &((*y)[0]) ) }; }
		multi_array<T,M,N> z;
		multi_array_alloc(z, x);
		unroll_16x_4x_for(
			epconj(size_t n = z.size(); T* zi = z.data(); const T* xi = x.data(); int* yi = y->data()),
			epconj(n >= 16), epconj(n-=16, zi+=16, xi+=16, yi+=16),
			epconj(n >= 4), epconj(n-=4, zi+=4, xi+=4, yi+=4),
			epconj(n != 0), epconj(--n, ++zi, ++xi, ++yi),
			zi[i] = frexp( xi[i], yi+i )
		);
		return std::move(z);
	}

	/// z[i] = x[i] * exp2(y[i]).
	template<typename T, size_t M, size_t N>
	multi_array<T,M,N> ldexp(const multi_array<T,M,N> &x, const multi_array<int,M,N> &y) {
		constexpr size_t MN = M * N;
		if constexpr (MN == 4) { return { ldexp(x[0],y[0]), ldexp(x[1],y[1]), ldexp(x[2],y[2]), ldexp(x[3],y[3]) }; }
		else if constexpr (MN == 3) { return { ldexp(x[0],y[0]), ldexp(x[1],y[1]), ldexp(x[2],y[2]) }; }
		else if constexpr (MN == 2) { return { ldexp(x[0],y[0]), ldexp(x[1],y[1]) }; }
		else if constexpr (MN == 1) { return { ldexp(x[0],y[0]) }; } 
		multi_array<T,M,N> z;
		multi_array_alloc(z, x);
		unroll_16x_4x_for(
			epconj(size_t n = z.size(); T* zi = z.data(); const T* xi = x.data(); const int* yi = y.data()),
			epconj(n >= 16), epconj(n-=16, zi+=16, xi+=16, yi+=16),
			epconj(n >= 4), epconj(n-=4, zi+=4, xi+=4, yi+=4),
			epconj(n != 0), epconj(--n, ++zi, ++xi, ++yi),
			zi[i] = ldexp( xi[i], yi[i] )
		);
		return std::move(z);
	}


	/// y = x.
	template<typename from_scalar, size_t from_size, typename to_scalar, size_t to_size>
	void array_cast(const multi_array<from_scalar,from_size,1>& x, multi_array<to_scalar,to_size,1>& y) {
		assert(x.size() >= y.size());
		if constexpr (to_size == 0) {
			for (size_t i = 0; i != y.size(); ++i)
				y[i] = static_cast<to_scalar>(x[i]);
		} else {
			if constexpr (to_size >= 1)
				y[0] = static_cast<to_scalar>(x[0]);
			if constexpr (to_size >= 2)
				y[1] = static_cast<to_scalar>(x[1]);
			if constexpr (to_size >= 3)
				y[2] = static_cast<to_scalar>(x[2]);
			if constexpr (to_size >= 4)
				y[3] = static_cast<to_scalar>(x[3]);
			if constexpr (to_size >= 5)
				for (size_t i = 4; i != to_size; ++i)
					y[i] = static_cast<to_scalar>(x[i]);
		}
	}

	/// y = {x,s0}.
	template<typename from_scalar, size_t from_size, typename to_scalar, size_t to_size, typename type0>
	void array_cast(const multi_array<from_scalar,from_size,1>& x, multi_array<to_scalar,to_size,1>& y, type0&& s0) {
		assert(x.size() + 1 == y.size());
		if constexpr (to_size == 0) {
			for (size_t i = 0; i != x.size(); ++i)
				y[i] = static_cast<to_scalar>(x[i]);
			y[x.size()] = static_cast<to_scalar>(s0);
		} else {
			array_cast(x, reinterpret_cast<multi_array<to_scalar,to_size-1,1>&>(y));
			y[to_size-1] = static_cast<to_scalar>(s0);
		}
	}

	/// y = {x,s0,s1}.
	template<typename from_scalar, size_t from_size, typename to_scalar, size_t to_size, typename type0, typename type1>
	void array_cast(const multi_array<from_scalar,from_size,1>& x, multi_array<to_scalar,to_size,1>& y, type0&& s0, type1&& s1) {
		assert(x.size() + 2 == y.size());
		if constexpr (to_size == 0) {
			for (size_t i = 0; i != x.size(); ++i)
				y[i] = static_cast<to_scalar>(x[i]);
			y[x.size()]   = static_cast<to_scalar>(s0);
			y[x.size()+1] = static_cast<to_scalar>(s1);
		} else {
			array_cast(x, reinterpret_cast<multi_array<to_scalar,to_size-2,1>&>(y));
			y[to_size-2] = static_cast<to_scalar>(s0);
			y[to_size-1] = static_cast<to_scalar>(s1);
		}
	}

	/// y = {x,s0,s1,s2}.
	template<typename from_scalar, size_t from_size, typename to_scalar, size_t to_size, typename type0, typename type1, typename type2>
	void array_cast(const multi_array<from_scalar,from_size,1>& x, multi_array<to_scalar,to_size,1>& y, type0&& s0, type1&& s1, type2&& s2) {
		assert(x.size() + 3 == y.size());
		if constexpr (to_size == 0) {
			for (size_t i = 0; i != x.size(); ++i)
				y[i] = static_cast<to_scalar>(x[i]);
			y[x.size()]   = static_cast<to_scalar>(s0);
			y[x.size()+1] = static_cast<to_scalar>(s1);
			y[x.size()+2] = static_cast<to_scalar>(s2);
		} else {
			array_cast(x, reinterpret_cast<multi_array<to_scalar,to_size-3,1>&>(y));
			y[to_size-3] = static_cast<to_scalar>(s0);
			y[to_size-2] = static_cast<to_scalar>(s1);
			y[to_size-1] = static_cast<to_scalar>(s2);
		}
	}

	/// y = {x,s0,s1,s2,s3}.
	template<typename from_scalar, size_t from_size, typename to_scalar, size_t to_size, typename type0, typename type1, typename type2, typename type3>
	void array_cast(const multi_array<from_scalar,from_size,1>& x, multi_array<to_scalar,to_size,1>& y, type0&& s0, type1&& s1, type2&& s2, type3&& s3) {
		assert(x.size() + 4 == y.size());
		if constexpr (to_size == 0) {
			for (size_t i = 0; i != x.size(); ++i)
				y[i] = static_cast<to_scalar>(x[i]);
			y[x.size()]   = static_cast<to_scalar>(s0);
			y[x.size()+1] = static_cast<to_scalar>(s1);
			y[x.size()+2] = static_cast<to_scalar>(s2);
			y[x.size()+3] = static_cast<to_scalar>(s3);
		} else {
			array_cast(x, reinterpret_cast<multi_array<to_scalar,to_size-4,1>&>(y));
			y[to_size-4] = static_cast<to_scalar>(s0);
			y[to_size-3] = static_cast<to_scalar>(s1);
			y[to_size-2] = static_cast<to_scalar>(s2);
			y[to_size-1] = static_cast<to_scalar>(s3);
		}
	}

	/// y = {x,s0}.
	template<typename to_scalar, typename from_scalar, size_t size, typename type0>
	multi_array<to_scalar,size+1,1> array_cast(const multi_array<from_scalar,size,1>& x, type0&& s0) {
		static_assert(size != 0);
		if constexpr (size == 1) {
			return multi_array<to_scalar,size+1,1>{ static_cast<to_scalar>(x[0]), static_cast<to_scalar>(s0) };
		} else if constexpr (size == 2) {
			return multi_array<to_scalar,size+1,1>{ static_cast<to_scalar>(x[0]), static_cast<to_scalar>(x[1]), static_cast<to_scalar>(s0) };
		} else if constexpr (size == 3) {
			return multi_array<to_scalar,size+1,1>{ static_cast<to_scalar>(x[0]), static_cast<to_scalar>(x[1]), static_cast<to_scalar>(x[2]), static_cast<to_scalar>(s0) };
		} else {
			multi_array<to_scalar,size+1,1> y;
			array_cast(x, y, std::move(s0));
			return y;
		}
	}

	/// y = {x,s0,s1}.
	template<typename to_scalar, typename from_scalar, size_t size, typename type0, typename type1>
	multi_array<to_scalar,size+2,1> array_cast(const multi_array<from_scalar,size,1>& x, type0&& s0, type1&& s1) {
		static_assert(size != 0); 
		if constexpr (size == 1) {
			return multi_array<to_scalar,size+2,1>{ static_cast<to_scalar>(x[0]), static_cast<to_scalar>(s0), static_cast<to_scalar>(s1) };
		} else if constexpr (size == 2) {
			return multi_array<to_scalar,size+2,1>{ static_cast<to_scalar>(x[0]), static_cast<to_scalar>(x[1]), static_cast<to_scalar>(s0), static_cast<to_scalar>(s1) };
		} else {
			multi_array<to_scalar,size+2,1> y;
			array_cast(x, y, std::move(s0), std::move(s1));
			return y;
		}
	}

	/// y = {x,s0,s1,s2}.
	template<typename to_scalar, typename from_scalar, size_t size, typename type0, typename type1, typename type2>
	multi_array<to_scalar,size+3,1> array_cast(const multi_array<from_scalar,size,1>& x, type0&& s0, type1&& s1, type2&& s2) {
		static_assert(size != 0);
		if constexpr (size == 1) {
			return multi_array<to_scalar,size+3,1>{ static_cast<to_scalar>(x[0]), static_cast<to_scalar>(s0), static_cast<to_scalar>(s1), static_cast<to_scalar>(s2) };
		} else {
			multi_array<to_scalar,size+3,1> y;
			array_cast(x, y, std::move(s0), std::move(s1), std::move(s2));
			return y;
		}
	}

	/// y = {x,s0,s1,s2,s3}.
	template<typename to_scalar, typename from_scalar, size_t size, typename type0, typename type1, typename type2, typename type3>
	multi_array<to_scalar,size+4,1> array_cast(const multi_array<from_scalar,size,1>& x, type0&& s0, type1&& s1, type2&& s2, type3&& s3) {
		static_assert(size != 0);
		multi_array<to_scalar,size+4,1> y;
		array_cast(x, y, std::move(s0), std::move(s1), std::move(s2), std::move(s3));
		return y;
}

///
/// Each function defined for this form,
/// 
///		multi_array operator+(const multi_array& x, const multi_array& y) {
///			+ -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- +
///			|	constexpr size_t MN = M * N;                                                        |
///			|	if constexpr (MN == 4) { return { x[0]+y[0], x[1]+y[1], x[2]+y[2], x[3]+y[3] }; }   |
///			|	else if constexpr (MN == 3) { return { x[0]+y[0], x[1]+y[1], x[2]+y[2] }; }         |
///			|	else if constexpr (MN == 2) { return { x[0]+y[0], x[1]+y[1] }; }                    |
///			|	else if constexpr (MN == 1) { return { x[0]+y[0] }; }                               | optimization for inline avoid copy.
///			+ -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- +
/// 
///			+ -- -- -- -- -- -- -- -- -- -- +
///			|	::math::multi_array<T,M,N> z; |
///			|	multi_array_alloc(z, x);      | memory_error.
///			+ -- -- -- -- -- -- -- -- -- -- +
/// 
///			+ -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- +
///			|	unroll_16x_4x_for(                                                                                 |
///			|		epconj(size_t n = z.size(); T* zi = z.data(); const T* xi = x.data(); const T* yi = y.data()),   |
///			|		epconj(n >= 16), epconj(n-=16, zi+=16, xi+=16, yi+=16),                                          |
///			|		epconj(n >= 4), epconj(n-=4, zi+=4, xi+=4, yi+=4),                                               |
///			|		epconj(n != 0), epconj(--n, ++zi, ++xi, ++yi),                                                   |
///			|		zi[i] = xi[i] OP yi[i]                                                                           |
///			|	);                                                                                                 | noexcept.
///			+ -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- +
/// 
///			return std::move(z);
///		}
/// 
/// Because { something } = something, so { ignore, memory_error, noexcept } = { memory_error } = memory_error.
/// We can simplify these functions to a list,
/// 
///		_array_x_op_y(T, M, N, &)
///		_array_x_op_y(T, M, N, |)
///		_array_x_op_y(T, M, N, ^)
///		_array_x_op_y(T, M, N, +)
///		_array_x_op_y(T, M, N, -)
///		_array_x_op_y(T, M, N, *)
///		_array_x_op_y(T, M, N, /)
///		_array_x_op_y(T, M, N, %)
/// 
/// Because here are only one exception. Must be it if throw,
/// so we not need debug in this functions.
/// 
///@example
///#include "multi_array.hpp"
///#include "array.hpp"
///int main() {
///	srand(unsigned(time(0)));
///	math::multi_array<float> A;
///	math::multi_array<float> B;
///	A.reshape(rand()%10 + 1, rand()%2 + 1);
///	B.resize(A.dims(), A.extents());
///	for (size_t i = 0; i != A.size(); ++i) {
///		A[i] = (float)rand() / float(RAND_MAX);
///		B[i] = (float)rand();
///	}
///	float c = (float)rand() / float(RAND_MAX);
///	float d = (float)rand();
///	std::cout << "A:{\n" << A << "\n}" << std::endl;
///	std::cout << "B:{\n" << B << "\n}" << std::endl;
///
///	std::cout << "=========================Arithmetic=========================\n";
///	std::cout << "A + B = { " << A + B << " }\n";
///	std::cout << "A - B = { " << A - B << " }\n";
///	std::cout << "A * B = { " << A * B << " }\n";
///	std::cout << "A / B = { " << A / B << " }\n";
///	std::cout << "A + "<<c<<" = { " << A + c << " }\n";
///	std::cout << "A - "<<c<<" = { " << A - c << " }\n";
///	std::cout << "A * "<<c<<" = { " << A * c << " }\n";
///	std::cout << "A / "<<c<<" = { " << A / c << " }\n";
///	std::cout << "=========================Numerical=========================\n";
///	std::cout << "min("<<c<<",A) = { " << math::min(c,A) << " }\n";
///	std::cout << "min(A,"<<c<<") = { " << math::min(A,c) << " }\n";
///	std::cout << "max("<<c<<",A) = { " << math::max(c,A) << " }\n";
///	std::cout << "max(A,"<<c<<") = { " << math::max(A,c) << " }\n";
///	std::cout << "abs(A-B) = { " << math::abs(A-B) << " }\n";
///	std::cout << "trunc(A+B) = { " << math::trunc(A+B) << " }\n";
///	std::cout << "floor(A+B) = { " << math::floor(A+B) << " }\n";
///	std::cout << "ceil(A+B) = { " << math::ceil(A+B) << " }\n";
///	std::cout << "round(A+B) = { " << math::round(A+B) << " }\n";
///	std::cout << "=========================Transcendental=========================\n";
///	std::cout << "sqrt(A+B) = { " << math::sqrt(A+B) << " }\n";
///	std::cout << "cbrt(A+B) = { " << math::cbrt(A+B) << " }\n";
///	std::cout << "exp(A+B) = { " << math::exp(A+B) << " }\n";
///	std::cout << "log(A+B) = { " << math::log(A+B) << " }\n";
///	std::cout << "pow(A,3) = { " << math::pow(A,3) << " }\n";
///	std::cout << "sin(A) = { " << math::sin(A) << " }\n";
///	std::cout << "cos(A) = { " << math::cos(A) << " }\n";
///	std::cout << "tan(A) = { " << math::tan(A) << " }\n";
///	std::cout << "asin(A) = { " << math::asin(A) << " }\n";
///	std::cout << "acos(A) = { " << math::acos(A) << " }\n";
///	std::cout << "atan(A) = { " << math::atan(A) << " }\n";
///	std::cout << "atan2(A,B) = { " << math::atan2(A,B) << " }\n";
/// 	math::multi_array<int> e2; e2.resize(B.dims(), B.extents());
///	std::cout << "modf(A) = {" << math::modf(A, &B) << "}\n";
///	std::cout << "{" << B << "}\n";
///	std::cout << "frexp(A) = {" << math::frexp(A, &e2) << "}\n";
///	std::cout << "{" << e2 << "}\n";
///	std::cout << "ldexp(frexp(A),..) = {" << math::ldexp(math::frexp(A, &e2), e2) << "}\n";
///	return 0;
///}
}