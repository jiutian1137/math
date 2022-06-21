#pragma once

#if __has_include("multi_array.hpp")
#include "multi_array.hpp"
#endif

#if __has_include("simd_multi_array.hpp")
#include "simd_multi_array.hpp"
#endif


/// SIMD Core of Array
///@license Free
///@review 2022-5-11
///@author LongJiangnan, Jiang1998Nan@outlook.com
#define _MATH_ARRAY_SIMDCORE_

namespace math {
#ifdef _MATH_MULTI_ARRAY_
	#define _array_simdlen4_op_x(scalar, s_rows, s_columns, op, p4fn, p4loadu, p4storeu) \
	multi_array<scalar, s_rows, s_columns, 4> operator##op(const multi_array<scalar, s_rows, s_columns, 4> &x) { \
		constexpr size_t s_size = s_rows * s_columns; \
		if constexpr (s_size == 1) { return { op x[0] }; } \
		else if constexpr (s_size == 2) { return { op x[0], op x[1] }; } \
		else if constexpr (s_size == 3) { return { op x[0], op x[1], op x[2] }; } \
		else if constexpr (s_size == 4) { \
			multi_array<scalar, s_rows, s_columns, 4> y; \
			p4storeu(y.data(), p4fn( p4loadu(x.data()) )); \
			return y; \
		} \
		else if constexpr (s_size == 16) { \
			multi_array<scalar, s_rows, s_columns, 4> y; \
			p4storeu(y.data(), p4fn( p4loadu(x.data()) )); \
			p4storeu(y.data()+4, p4fn( p4loadu(x.data()+4) )); \
			p4storeu(y.data()+8, p4fn( p4loadu(x.data()+8) )); \
			p4storeu(y.data()+12, p4fn( p4loadu(x.data()+12) )); \
			return y; \
		} \
	\
		multi_array<scalar, s_rows, s_columns, 4> y; \
		multi_array_alloc(y, x); \
		size_t        n  = y.size(); \
		scalar       *yi = y.data(); \
		const scalar *xi = x.data(); \
		for ( ; n >= 16; n-=16, yi+=16, xi+=16) { \
			p4storeu(yi, p4fn( p4loadu(xi) )); \
			p4storeu(yi+4, p4fn( p4loadu(xi+4) )); \
			p4storeu(yi+8, p4fn( p4loadu(xi+8) )); \
			p4storeu(yi+12, p4fn( p4loadu(xi+12) )); \
		} \
		for ( ; n >= 4; n-=4, yi+=4, xi+=4) { \
			p4storeu(yi, p4fn( p4loadu(xi) )); \
		} \
		for ( ; n != 0; --n, ++yi, ++xi) { \
			(*yi) = op (*xi); \
		} \
		return std::move(y); \
	}

	#define _array_simdlen4_x_op_y(scalar, s_rows, s_columns, op, p4fn, p4loadu, p4storeu) \
	multi_array<scalar, s_rows, s_columns, 4> operator##op(const multi_array<scalar, s_rows, s_columns, 4> &x, const multi_array<scalar, s_rows, s_columns, 4> &y) { \
		constexpr size_t s_size = s_rows * s_columns; \
		if constexpr (s_size == 1) { return { x[0] op y[0] }; } \
		else if constexpr (s_size == 2) { return { x[0] op y[0], x[1] op y[1] }; } \
		else if constexpr (s_size == 3) { return { x[0] op y[0], x[1] op y[1], x[2] op y[2] }; } \
		else if constexpr (s_size == 4) { \
			multi_array<scalar, s_rows, s_columns, 4> z; \
			p4storeu(z.data(), p4fn( p4loadu(x.data()), p4loadu(y.data()) )); \
			return z; \
		} \
		else if constexpr (s_size == 16) { \
			multi_array<scalar, s_rows, s_columns, 4> z; \
			p4storeu(z.data(), p4fn( p4loadu(x.data()), p4loadu(y.data()) )); \
			p4storeu(z.data()+4, p4fn( p4loadu(x.data()+4), p4loadu(y.data()+4) )); \
			p4storeu(z.data()+8, p4fn( p4loadu(x.data()+8), p4loadu(y.data()+8) )); \
			p4storeu(z.data()+12, p4fn( p4loadu(x.data()+12), p4loadu(y.data()+12) )); \
			return z; \
		} \
	\
		multi_array<scalar, s_rows, s_columns, 4> z; \
		multi_array_alloc(z, x); \
		size_t        n  = z.size(); \
		scalar       *zi = z.data(); \
		const scalar *xi = x.data(); \
		const scalar *yi = y.data(); \
		for ( ; n >= 16; n-=16, zi+=16, xi+=16, yi+=16) { \
			p4storeu(zi, p4fn( p4loadu(xi), p4loadu(yi) )); \
			p4storeu(zi+4, p4fn( p4loadu(xi+4), p4loadu(yi+4) )); \
			p4storeu(zi+8, p4fn( p4loadu(xi+8), p4loadu(yi+8) )); \
			p4storeu(zi+12, p4fn( p4loadu(xi+12), p4loadu(yi+12) )); \
		} \
		for ( ; n >= 4; n-=4, zi+=4, xi+=4, yi+=4) { \
			p4storeu(zi, p4fn( p4loadu(xi), p4loadu(yi) )); \
		} \
		for ( ; n != 0; --n, ++zi, ++xi, ++yi) { \
			(*zi) = (*xi) op (*yi); \
		} \
		return std::move(z); \
	}

	#define _array_simdlen4_x_op_yval(scalar, s_rows, s_columns, scalar2, op, p4fn, p4loadu, p4storeu, p4set1) \
	multi_array<scalar, s_rows, s_columns, 4> operator##op(const multi_array<scalar, s_rows, s_columns, 4> &x, const scalar2 yval_) { \
		const scalar yval = static_cast<scalar>(yval_); \
		constexpr size_t s_size = s_rows * s_columns; \
		if constexpr (s_size == 1) { return { x[0] op yval }; } \
		else if constexpr (s_size == 2) { return { x[0] op yval, x[1] op yval }; } \
		else if constexpr (s_size == 3) { return { x[0] op yval, x[1] op yval, x[2] op yval }; } \
		else if constexpr (s_size == 4) { \
			multi_array<scalar, s_rows, s_columns, 4> z; \
			const auto ypackage = p4set1(yval); \
			p4storeu(z.data(), p4fn( p4loadu(x.data()), ypackage )); \
			return z; \
		} \
		else if constexpr (s_size == 16) { \
			multi_array<scalar, s_rows, s_columns, 4> z; \
			const auto ypackage = p4set1(yval); \
			p4storeu(z.data(), p4fn( p4loadu(x.data()), ypackage )); \
			p4storeu(z.data()+4, p4fn( p4loadu(x.data()+4), ypackage )); \
			p4storeu(z.data()+8, p4fn( p4loadu(x.data()+8), ypackage )); \
			p4storeu(z.data()+12, p4fn( p4loadu(x.data()+12), ypackage )); \
			return z; \
		} \
	\
		multi_array<scalar, s_rows, s_columns, 4> z; \
		multi_array_alloc(z, x); \
		size_t        n  = z.size(); \
		scalar       *zi = z.data(); \
		const scalar *xi = x.data(); \
		const auto ypackage = p4set1(yval); \
		for ( ; n >= 16; n-=16, zi+=16, xi+=16) { \
			p4storeu(zi, p4fn( p4loadu(xi), ypackage )); \
			p4storeu(zi+4, p4fn( p4loadu(xi+4), ypackage )); \
			p4storeu(zi+8, p4fn( p4loadu(xi+8), ypackage )); \
			p4storeu(zi+12, p4fn( p4loadu(xi+12), ypackage )); \
		} \
		for ( ; n >= 4; n-=4, zi+=4, xi+=4) { \
			p4storeu(zi, p4fn( p4loadu(xi), ypackage )); \
		} \
		for ( ; n != 0; --n, ++zi, ++xi) { \
			(*zi) = (*xi) op yval; \
		} \
		return std::move(z); \
	}

	#define _array_simdlen4_xval_op_y(scalar2, scalar, s_rows, s_columns, op, p4fn, p4loadu, p4storeu, p4set1) \
	multi_array<scalar, s_rows, s_columns, 4> operator##op(const scalar2 xval_, const multi_array<scalar, s_rows, s_columns, 4> &y) { \
		const scalar xval = static_cast<scalar>(xval_); \
		constexpr size_t s_size = s_rows * s_columns; \
		if constexpr (s_size == 1) { return { xval op y[0] }; } \
		else if constexpr (s_size == 2) { return { xval op y[0], xval op y[1] }; } \
		else if constexpr (s_size == 3) { return { xval op y[0], xval op y[1], xval op y[2] }; } \
		else if constexpr (s_size == 4) { \
			multi_array<scalar, s_rows, s_columns, 4> z; \
			const auto xpackage = p4set1(xval); \
			p4storeu(z.data(), p4fn( xpackage, p4loadu(y.data()) )); \
			return z; \
		} \
		else if constexpr (s_size == 16) { \
			multi_array<scalar, s_rows, s_columns, 4> z; \
			const auto xpackage = p4set1(xval); \
			p4storeu(z.data(), p4fn( xpackage, p4loadu(y.data()) )); \
			p4storeu(z.data()+4, p4fn( xpackage, p4loadu(y.data()+4) )); \
			p4storeu(z.data()+8, p4fn( xpackage, p4loadu(y.data()+8) )); \
			p4storeu(z.data()+12, p4fn( xpackage, p4loadu(y.data()+12) )); \
			return z; \
		} \
	\
		multi_array<scalar, s_rows, s_columns, 4> z; \
		multi_array_alloc(z, y); \
		size_t        n  = z.size(); \
		scalar       *zi = z.data(); \
		const scalar *yi = y.data(); \
		const auto xpackage = p4set1(xval); \
		for ( ; n >= 16; n-=16, zi+=16, yi+=16) { \
			p4storeu(zi, p4fn( xpackage, p4loadu(yi) )); \
			p4storeu(zi+4, p4fn( xpackage, p4loadu(yi+4) )); \
			p4storeu(zi+8, p4fn( xpackage, p4loadu(yi+8) )); \
			p4storeu(zi+12, p4fn( xpackage, p4loadu(yi+12) )); \
		} \
		for ( ; n >= 4; n-=4, zi+=4, yi+=4) { \
			p4storeu(zi, p4fn( xpackage, p4loadu(yi) )); \
		} \
		for ( ; n != 0; --n, ++zi, ++yi) { \
			(*zi) = xval op (*yi); \
		} \
		return std::move(z); \
	}


	#define _array_simdlen4_x_assginop_y(scalar, s_rows, s_columns, aop, p4fn, p4loadu, p4storeu) \
	multi_array<scalar, s_rows, s_columns, 4>& operator##aop(multi_array<scalar, s_rows, s_columns, 4> &x, const multi_array<scalar, s_rows, s_columns, 4> &y) { \
		constexpr size_t s_size = s_rows * s_columns; \
		if constexpr (s_size == 1) { x[0] aop y[0]; return x; } \
		else if constexpr (s_size == 2) { x[0] aop y[0]; x[1] aop y[1]; return x; } \
		else if constexpr (s_size == 3) { x[0] aop y[0]; x[1] aop y[1]; x[2] aop y[2]; return x; } \
		else if constexpr (s_size == 4) { \
			p4storeu(x.data(), p4fn( p4loadu(x.data()), p4loadu(y.data()) )); \
			return x; \
		} \
		else if constexpr (s_size == 16) { \
			p4storeu(x.data(), p4fn( p4loadu(x.data()), p4loadu(y.data()) )); \
			p4storeu(x.data()+4, p4fn( p4loadu(x.data()+4), p4loadu(y.data()+4) )); \
			p4storeu(x.data()+8, p4fn( p4loadu(x.data()+8), p4loadu(y.data()+8) )); \
			p4storeu(x.data()+12, p4fn( p4loadu(x.data()+12), p4loadu(y.data()+12) )); \
			return x; \
		} \
	\
		size_t        n  = x.size(); \
		scalar       *xi = x.data(); \
		const scalar *yi = y.data(); \
		for ( ; n >= 16; n-=16, xi+=16, yi+=16) { \
			p4storeu(xi, p4fn( p4loadu(xi), p4loadu(yi) )); \
			p4storeu(xi+4, p4fn( p4loadu(xi+4), p4loadu(yi+4) )); \
			p4storeu(xi+8, p4fn( p4loadu(xi+8), p4loadu(yi+8) )); \
			p4storeu(xi+12, p4fn( p4loadu(xi+12), p4loadu(yi+12) )); \
		} \
		for ( ; n >= 4; n-=4, xi+=4, yi+=4) { \
			p4storeu(xi, p4fn( p4loadu(xi), p4loadu(yi) )); \
		} \
		for ( ; n != 0; --n, ++xi, ++yi) { \
			(*xi) aop (*yi); \
		} \
		return x; \
	}

	#define _array_simdlen4_x_assginop_yval(scalar, s_rows, s_columns, scalar2, aop, p4fn, p4loadu, p4storeu, p4set1) \
	multi_array<scalar, s_rows, s_columns, 4>& operator##aop(multi_array<scalar, s_rows, s_columns, 4> &x, const scalar2 yval_) { \
		const scalar yval = static_cast<scalar>(yval_); \
		constexpr size_t s_size = s_rows * s_columns; \
		if constexpr (s_size == 1) { x[0] aop yval; return x; } \
		else if constexpr (s_size == 2) { x[0] aop yval; x[1] aop yval; return x; } \
		else if constexpr (s_size == 3) { x[0] aop yval; x[1] aop yval; x[2] aop yval; return x; } \
		else if constexpr (s_size == 4) { \
			const auto ypackage = p4set1(yval); \
			p4storeu(x.data(), p4fn( p4loadu(x.data()), ypackage )); \
			return x; \
		} \
		else if constexpr (s_size == 16) { \
			const auto ypackage = p4set1(yval); \
			p4storeu(x.data(), p4fn( p4loadu(x.data()), ypackage )); \
			p4storeu(x.data()+4, p4fn( p4loadu(x.data()+4), ypackage )); \
			p4storeu(x.data()+8, p4fn( p4loadu(x.data()+8), ypackage )); \
			p4storeu(x.data()+12, p4fn( p4loadu(x.data()+12), ypackage )); \
			return x; \
		} \
	\
		size_t        n  = x.size(); \
		scalar       *xi = x.data(); \
		const auto ypackage = p4set1(yval); \
		for ( ; n >= 16; n-=16, xi+=16) { \
			p4storeu(xi, p4fn( p4loadu(xi), ypackage )); \
			p4storeu(xi+4, p4fn( p4loadu(xi+4), ypackage )); \
			p4storeu(xi+8, p4fn( p4loadu(xi+8), ypackage )); \
			p4storeu(xi+12, p4fn( p4loadu(xi+12), ypackage )); \
		} \
		for ( ; n >= 4; n-=4, xi+=4) { \
			p4storeu(xi, p4fn( p4loadu(xi), ypackage )); \
		} \
		for ( ; n != 0; --n, ++xi) { \
			(*xi) aop yval; \
		} \
		return x; \
	}


	#define _array_simdlen4_fn_x(scalar, s_rows, s_columns, fn, p4fn, p4loadu, p4storeu) \
	multi_array<scalar, s_rows, s_columns, 4> fn(const multi_array<scalar, s_rows, s_columns, 4> &x) { \
		constexpr size_t s_size = s_rows * s_columns; \
		if constexpr (s_size == 1) { return { fn(x[0]) }; } \
		else if constexpr (s_size == 2) { return { fn(x[0]), fn(x[1]) }; } \
		else if constexpr (s_size == 3) { return { fn(x[0]), fn(x[1]), fn(x[2]) }; } \
		else if constexpr (s_size == 4) { \
			multi_array<scalar, s_rows, s_columns, 4> y; \
			p4storeu(y.data(), p4fn( p4loadu(x.data()) )); \
			return y; \
		} \
		else if constexpr (s_size == 16) { \
			multi_array<scalar, s_rows, s_columns, 4> y; \
			p4storeu(y.data(), p4fn( p4loadu(x.data()) )); \
			p4storeu(y.data()+4, p4fn( p4loadu(x.data()+4) )); \
			p4storeu(y.data()+8, p4fn( p4loadu(x.data()+8) )); \
			p4storeu(y.data()+12, p4fn( p4loadu(x.data()+12) )); \
			return y; \
		} \
	\
		multi_array<scalar, s_rows, s_columns, 4> y; \
		multi_array_alloc(y, x); \
		size_t        n  = y.size(); \
		scalar       *yi = y.data(); \
		const scalar *xi = x.data(); \
		for ( ; n >= 16; n-=16, yi+=16, xi+=16) { \
			p4storeu(yi, p4fn( p4loadu(xi) )); \
			p4storeu(yi+4, p4fn( p4loadu(xi+4) )); \
			p4storeu(yi+8, p4fn( p4loadu(xi+8) )); \
			p4storeu(yi+12, p4fn( p4loadu(xi+12) )); \
		} \
		for ( ; n >= 4; n-=4, yi+=4, xi+=4) { \
			p4storeu(yi, p4fn( p4loadu(xi) )); \
		} \
		for ( ; n != 0; --n, ++yi, ++xi) { \
			(*yi) = fn(*xi); \
		} \
		return std::move(y); \
	}

	#define _array_simdlen4_fn_x_y(scalar, s_rows, s_columns, fn, p4fn, p4loadu, p4storeu) \
	multi_array<scalar, s_rows, s_columns, 4> fn(const multi_array<scalar, s_rows, s_columns, 4> &x, const multi_array<scalar, s_rows, s_columns, 4> &y) { \
		constexpr size_t s_size = s_rows * s_columns; \
		if constexpr (s_size == 1) { return { fn(x[0],y[0]) }; } \
		else if constexpr (s_size == 2) { return { fn(x[0],y[0]), fn(x[1],y[1]) }; } \
		else if constexpr (s_size == 3) { return { fn(x[0],y[0]), fn(x[1],y[1]), fn(x[2],y[2]) }; } \
		else if constexpr (s_size == 4) { \
			multi_array<scalar, s_rows, s_columns, 4> z; \
			p4storeu(z.data(), p4fn( p4loadu(x.data()), p4loadu(y.data()) )); \
			return z; \
		} \
		else if constexpr (s_size == 16) { \
			multi_array<scalar, s_rows, s_columns, 4> z; \
			p4storeu(z.data(), p4fn( p4loadu(x.data()), p4loadu(y.data()) )); \
			p4storeu(z.data()+4, p4fn( p4loadu(x.data()+4), p4loadu(y.data()+4) )); \
			p4storeu(z.data()+8, p4fn( p4loadu(x.data()+8), p4loadu(y.data()+8) )); \
			p4storeu(z.data()+12, p4fn( p4loadu(x.data()+12), p4loadu(y.data()+12) )); \
			return z; \
		} \
	\
		multi_array<scalar, s_rows, s_columns, 4> z; \
		multi_array_alloc(z, x); \
		size_t        n  = z.size(); \
		scalar       *zi = z.data(); \
		const scalar *xi = x.data(); \
		const scalar *yi = y.data(); \
		for ( ; n >= 16; n-=16, zi+=16, xi+=16, yi+=16) { \
			p4storeu(zi, p4fn( p4loadu(xi), p4loadu(yi) )); \
			p4storeu(zi+4, p4fn( p4loadu(xi+4), p4loadu(yi+4) )); \
			p4storeu(zi+8, p4fn( p4loadu(xi+8), p4loadu(yi+8) )); \
			p4storeu(zi+12, p4fn( p4loadu(xi+12), p4loadu(yi+12) )); \
		} \
		for ( ; n >= 4; n-=4, zi+=4, xi+=4, yi+=4) { \
			p4storeu(zi, p4fn( p4loadu(xi), p4loadu(yi) )); \
		} \
		for ( ; n != 0; --n, ++zi, ++xi, ++yi) { \
			(*zi) = fn(*xi,*yi); \
		} \
		return std::move(z); \
	}

	#define _array_simdlen4_fn_x_yval(scalar, s_rows, s_columns, scalar2, fn, p4fn, p4loadu, p4storeu, p4set1) \
	multi_array<scalar, s_rows, s_columns, 4> fn(const multi_array<scalar, s_rows, s_columns, 4> &x, const scalar2 yval_) { \
		const scalar yval = static_cast<scalar>(yval_); \
		constexpr size_t s_size = s_rows * s_columns; \
		if constexpr (s_size == 1) { return { fn(x[0],yval) }; } \
		else if constexpr (s_size == 2) { return { fn(x[0],yval), fn(x[1],yval) }; } \
		else if constexpr (s_size == 3) { return { fn(x[0],yval), fn(x[1],yval), fn(x[2],yval) }; } \
		else if constexpr (s_size == 4) { \
			multi_array<scalar, s_rows, s_columns, 4> z; \
			const auto ypackage = p4set1(yval); \
			p4storeu(z.data(), p4fn( p4loadu(x.data()), ypackage )); \
			return z; \
		} \
		else if constexpr (s_size == 16) { \
			multi_array<scalar, s_rows, s_columns, 4> z; \
			const auto ypackage = p4set1(yval); \
			p4storeu(z.data(), p4fn( p4loadu(x.data()), ypackage )); \
			p4storeu(z.data()+4, p4fn( p4loadu(x.data()+4), ypackage )); \
			p4storeu(z.data()+8, p4fn( p4loadu(x.data()+8), ypackage )); \
			p4storeu(z.data()+12, p4fn( p4loadu(x.data()+12), ypackage )); \
			return z; \
		} \
	\
		multi_array<scalar, s_rows, s_columns, 4> z; \
		multi_array_alloc(z, x); \
		size_t        n  = z.size(); \
		scalar       *zi = z.data(); \
		const scalar *xi = x.data(); \
		const auto ypackage = p4set1(yval); \
		for ( ; n >= 16; n-=16, zi+=16, xi+=16) { \
			p4storeu(zi, p4fn( p4loadu(xi), ypackage )); \
			p4storeu(zi+4, p4fn( p4loadu(xi+4), ypackage )); \
			p4storeu(zi+8, p4fn( p4loadu(xi+8), ypackage )); \
			p4storeu(zi+12, p4fn( p4loadu(xi+12), ypackage )); \
		} \
		for ( ; n >= 4; n-=4, zi+=4, xi+=4) { \
			p4storeu(zi, p4fn( p4loadu(xi), ypackage )); \
		} \
		for ( ; n != 0; --n, ++zi, ++xi) { \
			(*zi) = fn((*xi),yval); \
		} \
		return std::move(z); \
	}

	#define _array_simdlen4_fn_xval_y(scalar2, scalar, s_rows, s_columns, fn, p4fn, p4loadu, p4storeu, p4set1) \
	multi_array<scalar, s_rows, s_columns, 4> fn(const scalar2 xval_, const multi_array<scalar, s_rows, s_columns, 4> &y) { \
		const scalar xval = static_cast<scalar>(xval_); \
		constexpr size_t s_size = s_rows * s_columns; \
		if constexpr (s_size == 1) { return { fn(xval,y[0]) }; } \
		else if constexpr (s_size == 2) { return { fn(xval,y[0]), fn(xval,y[1]) }; } \
		else if constexpr (s_size == 3) { return { fn(xval,y[0]), fn(xval,y[1]), fn(xval,y[2]) }; } \
		else if constexpr (s_size == 4) { \
			multi_array<scalar, s_rows, s_columns, 4> z; \
			const auto xpackage = p4set1(xval); \
			p4storeu(z.data(), p4fn( xpackage, p4loadu(y.data()) )); \
			return z; \
		} \
		else if constexpr (s_size == 16) { \
			multi_array<scalar, s_rows, s_columns, 4> z; \
			const auto xpackage = p4set1(xval); \
			p4storeu(z.data(), p4fn( xpackage, p4loadu(y.data()) )); \
			p4storeu(z.data()+4, p4fn( xpackage, p4loadu(y.data()+4) )); \
			p4storeu(z.data()+8, p4fn( xpackage, p4loadu(y.data()+8) )); \
			p4storeu(z.data()+12, p4fn( xpackage, p4loadu(y.data()+12) )); \
			return z; \
		} \
	\
		multi_array<scalar, s_rows, s_columns, 4> z; \
		multi_array_alloc(z, y); \
		size_t        n  = z.size(); \
		scalar       *zi = z.data(); \
		const scalar *yi = y.data(); \
		const auto xpackage = p4set1(xval); \
		for ( ; n >= 16; n-=16, zi+=16, yi+=16) { \
			p4storeu(zi, p4fn( xpackage, p4loadu(yi) )); \
			p4storeu(zi+4, p4fn( xpackage, p4loadu(yi+4) )); \
			p4storeu(zi+8, p4fn( xpackage, p4loadu(yi+8) )); \
			p4storeu(zi+12, p4fn( xpackage, p4loadu(yi+12) )); \
		} \
		for ( ; n >= 4; n-=4, zi+=4, yi+=4) { \
			p4storeu(zi, p4fn( xpackage, p4loadu(yi) )); \
		} \
		for ( ; n != 0; --n, ++zi, ++yi) { \
			(*zi) = fn(xval,*yi); \
		} \
		return std::move(z); \
	}


	#define _array_simdlen2_op_x(scalar, s_rows, s_columns, op, p2fn, p2loadu, p2storeu) \
	multi_array<scalar, s_rows, s_columns, 2> operator##op(const multi_array<scalar, s_rows, s_columns, 2> &x) { \
		constexpr size_t s_size = s_rows * s_columns; \
		if constexpr (s_size == 1) { return { op x[0] }; } \
		else if constexpr (s_size == 2) { \
			multi_array<scalar, s_rows, s_columns, 2> y; \
			p2storeu(y.data(), p2fn( p2loadu(x.data()) )); \
			return y; \
		} \
		else if constexpr (s_size == 3) { \
			multi_array<scalar, s_rows, s_columns, 2> y; \
			p2storeu(y.data(), p2fn( p2loadu(x.data()) )); \
			y[2] = op x[2]; \
			return y; \
		} \
		else if constexpr (s_size == 4) { \
			multi_array<scalar, s_rows, s_columns, 2> y; \
			p2storeu(y.data(), p2fn( p2loadu(x.data()) )); \
			p2storeu(y.data()+2, p2fn( p2loadu(x.data()+2) )); \
			return y; \
		} \
		else if constexpr (s_size == 16) { \
			multi_array<scalar, s_rows, s_columns, 2> y; \
			p2storeu(y.data(), p2fn( p2loadu(x.data()) )); \
			p2storeu(y.data()+2, p2fn( p2loadu(x.data()+2) )); \
			p2storeu(y.data()+4, p2fn( p2loadu(x.data()+4) )); \
			p2storeu(y.data()+6, p2fn( p2loadu(x.data()+6) )); \
			p2storeu(y.data()+8, p2fn( p2loadu(x.data()+8) )); \
			p2storeu(y.data()+10, p2fn( p2loadu(x.data()+10) )); \
			p2storeu(y.data()+12, p2fn( p2loadu(x.data()+12) )); \
			p2storeu(y.data()+14, p2fn( p2loadu(x.data()+14) )); \
			return y; \
		} \
	\
		multi_array<scalar, s_rows, s_columns, 2> y; \
		multi_array_alloc(y, x); \
		size_t        n  = y.size(); \
		scalar       *yi = y.data(); \
		const scalar *xi = x.data(); \
		for ( ; n >= 16; n-=16, yi+=16, xi+=16) { \
			p2storeu(yi, p2fn( p2loadu(xi) )); \
			p2storeu(yi+2, p2fn( p2loadu(xi+2) )); \
			p2storeu(yi+4, p2fn( p2loadu(xi+4) )); \
			p2storeu(yi+6, p2fn( p2loadu(xi+6) )); \
			p2storeu(yi+8, p2fn( p2loadu(xi+8) )); \
			p2storeu(yi+10, p2fn( p2loadu(xi+10) )); \
			p2storeu(yi+12, p2fn( p2loadu(xi+12) )); \
			p2storeu(yi+14, p2fn( p2loadu(xi+14) )); \
		} \
		for ( ; n >= 2; n-=2, yi+=2, xi+=2) { \
			p2storeu(yi, p2fn( p2loadu(xi) )); \
		} \
		for ( ; n != 0; --n, ++yi, ++xi) { \
			(*yi) = op (*xi); \
		} \
		return std::move(y); \
	}

	#define _array_simdlen2_x_op_y(scalar, s_rows, s_columns, op, p2fn, p2loadu, p2storeu) \
	multi_array<scalar, s_rows, s_columns, 2> operator##op(const multi_array<scalar, s_rows, s_columns, 2> &x, const multi_array<scalar, s_rows, s_columns, 2> &y) { \
		constexpr size_t s_size = s_rows * s_columns; \
		if constexpr (s_size == 1) { return { x[0] op y[0] }; } \
		else if constexpr (s_size == 2) { \
			multi_array<scalar, s_rows, s_columns, 2> z; \
			p2storeu(z.data(), p2fn( p2loadu(x.data()), p2loadu(y.data()) )); \
			return z; \
		} \
		else if constexpr (s_size == 3) { \
			multi_array<scalar, s_rows, s_columns, 2> z; \
			p2storeu(z.data(), p2fn( p2loadu(x.data()), p2loadu(y.data()) )); \
			z[2] = x[2] op y[2]; \
			return z; \
		} \
		else if constexpr (s_size == 4) { \
			multi_array<scalar, s_rows, s_columns, 2> z; \
			p2storeu(z.data(), p2fn( p2loadu(x.data()), p2loadu(y.data()) )); \
			p2storeu(z.data()+2, p2fn( p2loadu(x.data()+2), p2loadu(y.data()+2) )); \
			return z; \
		} \
		else if constexpr (s_size == 16) { \
			multi_array<scalar, s_rows, s_columns, 2> z; \
			p2storeu(z.data(), p2fn( p2loadu(x.data()), p2loadu(y.data()) )); \
			p2storeu(z.data()+2, p2fn( p2loadu(x.data()+2), p2loadu(y.data()+2) )); \
			p2storeu(z.data()+4, p2fn( p2loadu(x.data()+4), p2loadu(y.data()+4) )); \
			p2storeu(z.data()+6, p2fn( p2loadu(x.data()+6), p2loadu(y.data()+6) )); \
			p2storeu(z.data()+8, p2fn( p2loadu(x.data()+8), p2loadu(y.data()+8) )); \
			p2storeu(z.data()+10, p2fn( p2loadu(x.data()+10), p2loadu(y.data()+10) )); \
			p2storeu(z.data()+12, p2fn( p2loadu(x.data()+12), p2loadu(y.data()+12) )); \
			p2storeu(z.data()+14, p2fn( p2loadu(x.data()+14), p2loadu(y.data()+14) )); \
			return z; \
		} \
	\
		multi_array<scalar, s_rows, s_columns, 2> z; \
		multi_array_alloc(z, x); \
		size_t        n  = z.size(); \
		scalar       *zi = z.data(); \
		const scalar *xi = x.data(); \
		const scalar *yi = y.data(); \
		for ( ; n >= 16; n-=16, zi+=16, xi+=16, yi+=16) { \
			p2storeu(zi, p2fn( p2loadu(xi), p2loadu(yi) )); \
			p2storeu(zi+2, p2fn( p2loadu(xi+2), p2loadu(yi+2) )); \
			p2storeu(zi+4, p2fn( p2loadu(xi+4), p2loadu(yi+4) )); \
			p2storeu(zi+6, p2fn( p2loadu(xi+6), p2loadu(yi+6) )); \
			p2storeu(zi+8, p2fn( p2loadu(xi+8), p2loadu(yi+8) )); \
			p2storeu(zi+10, p2fn( p2loadu(xi+10), p2loadu(yi+10) )); \
			p2storeu(zi+12, p2fn( p2loadu(xi+12), p2loadu(yi+12) )); \
			p2storeu(zi+14, p2fn( p2loadu(xi+14), p2loadu(yi+14) )); \
		} \
		for ( ; n >= 2; n-=2, zi+=2, xi+=2, yi+=2) { \
			p2storeu(zi, p2fn( p2loadu(xi), p2loadu(yi) )); \
		} \
		for ( ; n != 0; --n, ++zi, ++xi, ++yi) { \
			(*zi) = (*xi) op (*yi); \
		} \
		return std::move(z); \
	}

	#define _array_simdlen2_x_op_yval(scalar, s_rows, s_columns, scalar2, op, p2fn, p2loadu, p2storeu, p2set1) \
	multi_array<scalar, s_rows, s_columns, 2> operator##op(const multi_array<scalar, s_rows, s_columns, 2> &x, const scalar2 yval_) { \
		const scalar yval = static_cast<scalar>(yval_); \
		constexpr size_t s_size = s_rows * s_columns; \
		if constexpr (s_size == 1) { return { x[0] op yval }; } \
		else if constexpr (s_size == 2) { \
			multi_array<scalar, s_rows, s_columns, 2> z; \
			const auto ypackage = p2set1(yval); \
			p2storeu(z.data(), p2fn( p2loadu(x.data()), ypackage )); \
			return z; \
		} \
		else if constexpr (s_size == 3) { \
			multi_array<scalar, s_rows, s_columns, 2> z; \
			const auto ypackage = p2set1(yval); \
			p2storeu(z.data(), p2fn( p2loadu(x.data()), ypackage )); \
			z[2] = x[2] op yval; \
			return z; \
		} \
		else if constexpr (s_size == 4) { \
			multi_array<scalar, s_rows, s_columns, 2> z; \
			const auto ypackage = p2set1(yval); \
			p2storeu(z.data(), p2fn( p2loadu(x.data()), ypackage )); \
			p2storeu(z.data()+2, p2fn( p2loadu(x.data()+2), ypackage )); \
			return z; \
		} \
		else if constexpr (s_size == 16) { \
			multi_array<scalar, s_rows, s_columns, 2> z; \
			const auto ypackage = p2set1(yval); \
			p2storeu(z.data(), p2fn( p2loadu(x.data()), ypackage )); \
			p2storeu(z.data()+2, p2fn( p2loadu(x.data()+2), ypackage )); \
			p2storeu(z.data()+4, p2fn( p2loadu(x.data()+4), ypackage )); \
			p2storeu(z.data()+6, p2fn( p2loadu(x.data()+6), ypackage )); \
			p2storeu(z.data()+8, p2fn( p2loadu(x.data()+8), ypackage )); \
			p2storeu(z.data()+10, p2fn( p2loadu(x.data()+10), ypackage )); \
			p2storeu(z.data()+12, p2fn( p2loadu(x.data()+12), ypackage )); \
			p2storeu(z.data()+14, p2fn( p2loadu(x.data()+14), ypackage )); \
			return z; \
		} \
	\
		multi_array<scalar, s_rows, s_columns, 2> z; \
		multi_array_alloc(z, x); \
		size_t        n  = z.size(); \
		scalar       *zi = z.data(); \
		const scalar *xi = x.data(); \
		const auto ypackage = p2set1(yval); \
		for ( ; n >= 16; n-=16, zi+=16, xi+=16) { \
			p2storeu(zi, p2fn( p2loadu(xi), ypackage )); \
			p2storeu(zi+2, p2fn( p2loadu(xi+2), ypackage )); \
			p2storeu(zi+4, p2fn( p2loadu(xi+4), ypackage )); \
			p2storeu(zi+6, p2fn( p2loadu(xi+6), ypackage )); \
			p2storeu(zi+8, p2fn( p2loadu(xi+8), ypackage )); \
			p2storeu(zi+10, p2fn( p2loadu(xi+10), ypackage )); \
			p2storeu(zi+12, p2fn( p2loadu(xi+12), ypackage )); \
			p2storeu(zi+14, p2fn( p2loadu(xi+14), ypackage )); \
		} \
		for ( ; n >= 2; n-=2, zi+=2, xi+=2) { \
			p2storeu(zi, p2fn( p2loadu(xi), ypackage )); \
		} \
		for ( ; n != 0; --n, ++zi, ++xi) { \
			(*zi) = (*xi) op yval; \
		} \
		return std::move(z); \
	}

	#define _array_simdlen2_xval_op_y(scalar2, scalar, s_rows, s_columns, op, p2fn, p2loadu, p2storeu, p2set1) \
	multi_array<scalar, s_rows, s_columns, 2> operator##op(const scalar2 xval_, const multi_array<scalar, s_rows, s_columns, 2> &y) { \
		const scalar xval = static_cast<scalar>(xval_); \
		constexpr size_t s_size = s_rows * s_columns; \
		if constexpr (s_size == 1) { return { xval op y[0] }; } \
		else if constexpr (s_size == 2) { \
			multi_array<scalar, s_rows, s_columns, 2> z; \
			const auto xpackage = p2set1(xval); \
			p2storeu(z.data(), p2fn( xpackage, p2loadu(y.data()) )); \
			return z; \
		} \
		else if constexpr (s_size == 3) { \
			multi_array<scalar, s_rows, s_columns, 2> z; \
			const auto xpackage = p2set1(xval); \
			p2storeu(z.data(), p2fn( xpackage, p2loadu(y.data()) )); \
			z[2] = xval op y[2]; \
			return z; \
		} \
		else if constexpr (s_size == 4) { \
			multi_array<scalar, s_rows, s_columns, 2> z; \
			const auto xpackage = p2set1(xval); \
			p2storeu(z.data(), p2fn( xpackage, p2loadu(y.data()) )); \
			p2storeu(z.data()+2, p2fn( xpackage, p2loadu(y.data()+2) )); \
			return z; \
		} \
		else if constexpr (s_size == 16) { \
			multi_array<scalar, s_rows, s_columns, 2> z; \
			const auto xpackage = p2set1(xval); \
			p2storeu(z.data(), p2fn( xpackage, p2loadu(y.data()) )); \
			p2storeu(z.data()+2, p2fn( xpackage, p2loadu(y.data()+2) )); \
			p2storeu(z.data()+4, p2fn( xpackage, p2loadu(y.data()+4) )); \
			p2storeu(z.data()+6, p2fn( xpackage, p2loadu(y.data()+6) )); \
			p2storeu(z.data()+8, p2fn( xpackage, p2loadu(y.data()+8) )); \
			p2storeu(z.data()+10, p2fn( xpackage, p2loadu(y.data()+10) )); \
			p2storeu(z.data()+12, p2fn( xpackage, p2loadu(y.data()+12) )); \
			p2storeu(z.data()+14, p2fn( xpackage, p2loadu(y.data()+14) )); \
			return z; \
		} \
	\
		multi_array<scalar, s_rows, s_columns, 2> z; \
		multi_array_alloc(z, y); \
		size_t        n  = z.size(); \
		scalar       *zi = z.data(); \
		const scalar *yi = y.data(); \
		const auto xpackage = p2set1(xval); \
		for ( ; n >= 16; n-=16, zi+=16, yi+=16) { \
			p2storeu(zi, p2fn( xpackage, p2loadu(yi) )); \
			p2storeu(zi+2, p2fn( xpackage, p2loadu(yi+2) )); \
			p2storeu(zi+4, p2fn( xpackage, p2loadu(yi+4) )); \
			p2storeu(zi+6, p2fn( xpackage, p2loadu(yi+6) )); \
			p2storeu(zi+8, p2fn( xpackage, p2loadu(yi+8) )); \
			p2storeu(zi+10, p2fn( xpackage, p2loadu(yi+10) )); \
			p2storeu(zi+12, p2fn( xpackage, p2loadu(yi+12) )); \
			p2storeu(zi+14, p2fn( xpackage, p2loadu(yi+14) )); \
		} \
		for ( ; n >= 2; n-=2, zi+=2, yi+=2) { \
			p2storeu(zi, p2fn( xpackage, p2loadu(yi) )); \
		} \
		for ( ; n != 0; --n, ++zi, ++yi) { \
			(*zi) = xval op (*yi); \
		} \
		return std::move(z); \
	}


	#define _array_simdlen2_x_assginop_y(scalar, s_rows, s_columns, aop, p2fn, p2loadu, p2storeu) \
	multi_array<scalar, s_rows, s_columns, 2>& operator##aop(multi_array<scalar, s_rows, s_columns, 2> &x, const multi_array<scalar, s_rows, s_columns, 2> &y) { \
		constexpr size_t s_size = s_rows * s_columns; \
		if constexpr (s_size == 1) { x[0] aop y[0]; return x; } \
		else if constexpr (s_size == 2) { \
			p2storeu(x.data(), p2fn( p2loadu(x.data()), p2loadu(y.data()) )); \
			return x; \
		} \
		else if constexpr (s_size == 2) { \
			p2storeu(x.data(), p2fn( p2loadu(x.data()), p2loadu(y.data()) )); \
			x[2] aop y[2]; \
			return x; \
		} \
		else if constexpr (s_size == 4) { \
			p2storeu(x.data(), p2fn( p2loadu(x.data()), p2loadu(y.data()) )); \
			p2storeu(x.data()+2, p2fn( p2loadu(x.data()+2), p2loadu(y.data()+2) )); \
			return x; \
		} \
		else if constexpr (s_size == 16) { \
			p2storeu(x.data(), p2fn( p2loadu(x.data()), p2loadu(y.data()) )); \
			p2storeu(x.data()+2, p2fn( p2loadu(x.data()+2), p2loadu(y.data()+2) )); \
			p2storeu(x.data()+4, p2fn( p2loadu(x.data()+4), p2loadu(y.data()+4) )); \
			p2storeu(x.data()+6, p2fn( p2loadu(x.data()+6), p2loadu(y.data()+6) )); \
			p2storeu(x.data()+8, p2fn( p2loadu(x.data()+8), p2loadu(y.data()+8) )); \
			p2storeu(x.data()+10, p2fn( p2loadu(x.data()+10), p2loadu(y.data()+10) )); \
			p2storeu(x.data()+12, p2fn( p2loadu(x.data()+12), p2loadu(y.data()+12) )); \
			p2storeu(x.data()+14, p2fn( p2loadu(x.data()+14), p2loadu(y.data()+14) )); \
			return x; \
		} \
	\
		size_t        n  = x.size(); \
		scalar       *xi = x.data(); \
		const scalar *yi = y.data(); \
		for ( ; n >= 16; n-=16, xi+=16, yi+=16) { \
			p2storeu(xi, p2fn( p2loadu(xi), p2loadu(yi) )); \
			p2storeu(xi+2, p2fn( p2loadu(xi+2), p2loadu(yi+2) )); \
			p2storeu(xi+4, p2fn( p2loadu(xi+4), p2loadu(yi+4) )); \
			p2storeu(xi+6, p2fn( p2loadu(xi+6), p2loadu(yi+6) )); \
			p2storeu(xi+8, p2fn( p2loadu(xi+8), p2loadu(yi+8) )); \
			p2storeu(xi+10, p2fn( p2loadu(xi+10), p2loadu(yi+10) )); \
			p2storeu(xi+12, p2fn( p2loadu(xi+12), p2loadu(yi+12) )); \
			p2storeu(xi+14, p2fn( p2loadu(xi+14), p2loadu(yi+14) )); \
		} \
		for ( ; n >= 2; n-=2, xi+=2, yi+=2) { \
			p2storeu(xi, p2fn( p2loadu(xi), p2loadu(yi) )); \
		} \
		for ( ; n != 0; --n, ++xi, ++yi) { \
			(*xi) aop (*yi); \
		} \
		return x; \
	}

	#define _array_simdlen2_x_assginop_yval(scalar, s_rows, s_columns, scalar2, aop, p2fn, p2loadu, p2storeu, p2set1) \
	multi_array<scalar, s_rows, s_columns, 2>& operator##aop(multi_array<scalar, s_rows, s_columns, 2> &x, const scalar2 yval_) { \
		const scalar yval = static_cast<scalar>(yval_); \
		constexpr size_t s_size = s_rows * s_columns; \
		if constexpr (s_size == 1) { x[0] aop yval; return x; } \
		else if constexpr (s_size == 2) { \
			const auto ypackage = p2set1(yval); \
			p2storeu(x.data(), p2fn( p2loadu(x.data()), ypackage )); \
			return x; \
		} \
		else if constexpr (s_size == 2) { \
			const auto ypackage = p2set1(yval); \
			p2storeu(x.data(), p2fn( p2loadu(x.data()), ypackage )); \
			x[2] aop yval; \
			return x; \
		} \
		else if constexpr (s_size == 4) { \
			const auto ypackage = p2set1(yval); \
			p2storeu(x.data(), p2fn( p2loadu(x.data()), ypackage )); \
			p2storeu(x.data()+2, p2fn( p2loadu(x.data()+2), ypackage )); \
			return x; \
		} \
		else if constexpr (s_size == 16) { \
			const auto ypackage = p2set1(yval); \
			p2storeu(x.data(), p2fn( p2loadu(x.data()), ypackage )); \
			p2storeu(x.data()+2, p2fn( p2loadu(x.data()+2), ypackage )); \
			p2storeu(x.data()+4, p2fn( p2loadu(x.data()+4), ypackage )); \
			p2storeu(x.data()+6, p2fn( p2loadu(x.data()+6), ypackage )); \
			p2storeu(x.data()+8, p2fn( p2loadu(x.data()+8), ypackage )); \
			p2storeu(x.data()+10, p2fn( p2loadu(x.data()+10), ypackage )); \
			p2storeu(x.data()+12, p2fn( p2loadu(x.data()+12), ypackage )); \
			p2storeu(x.data()+14, p2fn( p2loadu(x.data()+14), ypackage )); \
			return x; \
		} \
	\
		size_t        n  = x.size(); \
		scalar       *xi = x.data(); \
		const auto ypackage = p2set1(yval); \
		for ( ; n >= 16; n-=16, xi+=16) { \
			p2storeu(xi, p2fn( p2loadu(xi), ypackage )); \
			p2storeu(xi+2, p2fn( p2loadu(xi+2), ypackage )); \
			p2storeu(xi+4, p2fn( p2loadu(xi+4), ypackage )); \
			p2storeu(xi+6, p2fn( p2loadu(xi+6), ypackage )); \
			p2storeu(xi+8, p2fn( p2loadu(xi+8), ypackage )); \
			p2storeu(xi+10, p2fn( p2loadu(xi+10), ypackage )); \
			p2storeu(xi+12, p2fn( p2loadu(xi+12), ypackage )); \
			p2storeu(xi+14, p2fn( p2loadu(xi+14), ypackage )); \
		} \
		for ( ; n >= 2; n-=2, xi+=2) { \
			p2storeu(xi, p2fn( p2loadu(xi), ypackage )); \
		} \
		for ( ; n != 0; --n, ++xi) { \
			(*xi) aop yval; \
		} \
		return x; \
	}


	#define _array_simdlen2_fn_x(scalar, s_rows, s_columns, fn, p2fn, p2loadu, p2storeu) \
	multi_array<scalar, s_rows, s_columns, 2> fn(const multi_array<scalar, s_rows, s_columns, 2> &x) { \
		constexpr size_t s_size = s_rows * s_columns; \
		if constexpr (s_size == 1) { return { fn(x[0]) }; } \
		else if constexpr (s_size == 2) { \
			multi_array<scalar, s_rows, s_columns, 2> y; \
			p2storeu(y.data(), p2fn( p2loadu(x.data()) )); \
			return y; \
		} \
		else if constexpr (s_size == 3) { \
			multi_array<scalar, s_rows, s_columns, 2> y; \
			p2storeu(y.data(), p2fn( p2loadu(x.data()) )); \
			y[2] = fn(x[2]); \
			return y; \
		} \
		else if constexpr (s_size == 4) { \
			multi_array<scalar, s_rows, s_columns, 2> y; \
			p2storeu(y.data(), p2fn( p2loadu(x.data()) )); \
			p2storeu(y.data()+2, p2fn( p2loadu(x.data()+2) )); \
			return y; \
		} \
		else if constexpr (s_size == 16) { \
			multi_array<scalar, s_rows, s_columns, 2> y; \
			p2storeu(y.data(), p2fn( p2loadu(x.data()) )); \
			p2storeu(y.data()+2, p2fn( p2loadu(x.data()+2) )); \
			p2storeu(y.data()+4, p2fn( p2loadu(x.data()+4) )); \
			p2storeu(y.data()+6, p2fn( p2loadu(x.data()+6) )); \
			p2storeu(y.data()+8, p2fn( p2loadu(x.data()+8) )); \
			p2storeu(y.data()+10, p2fn( p2loadu(x.data()+10) )); \
			p2storeu(y.data()+12, p2fn( p2loadu(x.data()+12) )); \
			p2storeu(y.data()+14, p2fn( p2loadu(x.data()+14) )); \
			return y; \
		} \
	\
		multi_array<scalar, s_rows, s_columns, 2> y; \
		multi_array_alloc(y, x); \
		size_t        n  = y.size(); \
		scalar       *yi = y.data(); \
		const scalar *xi = x.data(); \
		for ( ; n >= 16; n-=16, yi+=16, xi+=16) { \
			p2storeu(yi, p2fn( p2loadu(xi) )); \
			p2storeu(yi+2, p2fn( p2loadu(xi+2) )); \
			p2storeu(yi+4, p2fn( p2loadu(xi+4) )); \
			p2storeu(yi+6, p2fn( p2loadu(xi+6) )); \
			p2storeu(yi+8, p2fn( p2loadu(xi+8) )); \
			p2storeu(yi+10, p2fn( p2loadu(xi+10) )); \
			p2storeu(yi+12, p2fn( p2loadu(xi+12) )); \
			p2storeu(yi+14, p2fn( p2loadu(xi+14) )); \
		} \
		for ( ; n >= 2; n-=2, yi+=2, xi+=2) { \
			p2storeu(yi, p2fn( p2loadu(xi) )); \
		} \
		for ( ; n != 0; --n, ++yi, ++xi) { \
			(*yi) = fn(*xi); \
		} \
		return std::move(y); \
	}

	#define _array_simdlen2_fn_x_y(scalar, s_rows, s_columns, fn, p2fn, p2loadu, p2storeu) \
	multi_array<scalar, s_rows, s_columns, 2> fn(const multi_array<scalar, s_rows, s_columns, 2> &x, const multi_array<scalar, s_rows, s_columns, 2> &y) { \
		constexpr size_t s_size = s_rows * s_columns; \
		if constexpr (s_size == 1) { return { fn(x[0],y[0]) }; } \
		else if constexpr (s_size == 2) { \
			multi_array<scalar, s_rows, s_columns, 2> z; \
			p2storeu(z.data(), p2fn( p2loadu(x.data()), p2loadu(y.data()) )); \
			return z; \
		} \
		else if constexpr (s_size == 3) { \
			multi_array<scalar, s_rows, s_columns, 2> z; \
			p2storeu(z.data(), p2fn( p2loadu(x.data()), p2loadu(y.data()) )); \
			z[2] = fn(x[2],y[2]); \
			return z; \
		} \
		else if constexpr (s_size == 4) { \
			multi_array<scalar, s_rows, s_columns, 2> z; \
			p2storeu(z.data(), p2fn( p2loadu(x.data()), p2loadu(y.data()) )); \
			p2storeu(z.data()+2, p2fn( p2loadu(x.data()+2), p2loadu(y.data()+2) )); \
			return z; \
		} \
		else if constexpr (s_size == 16) { \
			multi_array<scalar, s_rows, s_columns, 2> z; \
			p2storeu(z.data(), p2fn( p2loadu(x.data()), p2loadu(y.data()) )); \
			p2storeu(z.data()+2, p2fn( p2loadu(x.data()+2), p2loadu(y.data()+2) )); \
			p2storeu(z.data()+4, p2fn( p2loadu(x.data()+4), p2loadu(y.data()+4) )); \
			p2storeu(z.data()+6, p2fn( p2loadu(x.data()+6), p2loadu(y.data()+6) )); \
			p2storeu(z.data()+8, p2fn( p2loadu(x.data()+8), p2loadu(y.data()+8) )); \
			p2storeu(z.data()+10, p2fn( p2loadu(x.data()+10), p2loadu(y.data()+10) )); \
			p2storeu(z.data()+12, p2fn( p2loadu(x.data()+12), p2loadu(y.data()+12) )); \
			p2storeu(z.data()+14, p2fn( p2loadu(x.data()+14), p2loadu(y.data()+14) )); \
			return z; \
		} \
	\
		multi_array<scalar, s_rows, s_columns, 2> z; \
		multi_array_alloc(z, x); \
		size_t        n  = z.size(); \
		scalar       *zi = z.data(); \
		const scalar *xi = x.data(); \
		const scalar *yi = y.data(); \
		for ( ; n >= 16; n-=16, zi+=16, xi+=16, yi+=16) { \
			p2storeu(zi, p2fn( p2loadu(xi), p2loadu(yi) )); \
			p2storeu(zi+2, p2fn( p2loadu(xi+2), p2loadu(yi+2) )); \
			p2storeu(zi+4, p2fn( p2loadu(xi+4), p2loadu(yi+4) )); \
			p2storeu(zi+6, p2fn( p2loadu(xi+6), p2loadu(yi+6) )); \
			p2storeu(zi+8, p2fn( p2loadu(xi+8), p2loadu(yi+8) )); \
			p2storeu(zi+10, p2fn( p2loadu(xi+10), p2loadu(yi+10) )); \
			p2storeu(zi+12, p2fn( p2loadu(xi+12), p2loadu(yi+12) )); \
			p2storeu(zi+14, p2fn( p2loadu(xi+14), p2loadu(yi+14) )); \
		} \
		for ( ; n >= 2; n-=2, zi+=2, xi+=2, yi+=2) { \
			p2storeu(zi, p2fn( p2loadu(xi), p2loadu(yi) )); \
		} \
		for ( ; n != 0; --n, ++zi, ++xi, ++yi) { \
			(*zi) = fn(*xi,*yi); \
		} \
		return std::move(z); \
	}

	#define _array_simdlen2_fn_x_yval(scalar, s_rows, s_columns, scalar2, fn, p2fn, p2loadu, p2storeu, p2set1) \
	multi_array<scalar, s_rows, s_columns, 2> fn(const multi_array<scalar, s_rows, s_columns, 2> &x, const scalar2 yval_) { \
		const scalar yval = static_cast<scalar>(yval_); \
		constexpr size_t s_size = s_rows * s_columns; \
		if constexpr (s_size == 1) { return { fn(x[0],yval) }; } \
		else if constexpr (s_size == 2) { \
			multi_array<scalar, s_rows, s_columns, 2> z; \
			const auto ypackage = p2set1(yval); \
			p2storeu(z.data(), p2fn( p2loadu(x.data()), ypackage )); \
			return z; \
		} \
		else if constexpr (s_size == 3) { \
			multi_array<scalar, s_rows, s_columns, 2> z; \
			const auto ypackage = p2set1(yval); \
			p2storeu(z.data(), p2fn( p2loadu(x.data()), ypackage )); \
			z[2] = fn(x[2],yval); \
			return z; \
		} \
		else if constexpr (s_size == 4) { \
			multi_array<scalar, s_rows, s_columns, 2> z; \
			const auto ypackage = p2set1(yval); \
			p2storeu(z.data(), p2fn( p2loadu(x.data()), ypackage )); \
			p2storeu(z.data()+2, p2fn( p2loadu(x.data()+2), ypackage )); \
			return z; \
		} \
		else if constexpr (s_size == 16) { \
			multi_array<scalar, s_rows, s_columns, 2> z; \
			const auto ypackage = p2set1(yval); \
			p2storeu(z.data(), p2fn( p2loadu(x.data()), ypackage )); \
			p2storeu(z.data()+2, p2fn( p2loadu(x.data()+2), ypackage )); \
			p2storeu(z.data()+4, p2fn( p2loadu(x.data()+4), ypackage )); \
			p2storeu(z.data()+6, p2fn( p2loadu(x.data()+6), ypackage )); \
			p2storeu(z.data()+8, p2fn( p2loadu(x.data()+8), ypackage )); \
			p2storeu(z.data()+10, p2fn( p2loadu(x.data()+10), ypackage )); \
			p2storeu(z.data()+12, p2fn( p2loadu(x.data()+12), ypackage )); \
			p2storeu(z.data()+14, p2fn( p2loadu(x.data()+14), ypackage )); \
			return z; \
		} \
	\
		multi_array<scalar, s_rows, s_columns, 2> z; \
		multi_array_alloc(z, x); \
		size_t        n  = z.size(); \
		scalar       *zi = z.data(); \
		const scalar *xi = x.data(); \
		const auto ypackage = p2set1(yval); \
		for ( ; n >= 16; n-=16, zi+=16, xi+=16) { \
			p2storeu(zi, p2fn( p2loadu(xi), ypackage )); \
			p2storeu(zi+2, p2fn( p2loadu(xi+2), ypackage )); \
			p2storeu(zi+4, p2fn( p2loadu(xi+4), ypackage )); \
			p2storeu(zi+6, p2fn( p2loadu(xi+6), ypackage )); \
			p2storeu(zi+8, p2fn( p2loadu(xi+8), ypackage )); \
			p2storeu(zi+10, p2fn( p2loadu(xi+10), ypackage )); \
			p2storeu(zi+12, p2fn( p2loadu(xi+12), ypackage )); \
			p2storeu(zi+14, p2fn( p2loadu(xi+14), ypackage )); \
		} \
		for ( ; n >= 2; n-=2, zi+=2, xi+=2) { \
			p2storeu(zi, p2fn( p2loadu(xi), ypackage )); \
		} \
		for ( ; n != 0; --n, ++zi, ++xi) { \
			(*zi) = fn(*xi,yval); \
		} \
		return std::move(z); \
	}

	#define _array_simdlen2_fn_xval_y(scalar2, scalar, s_rows, s_columns, fn, p2fn, p2loadu, p2storeu, p2set1) \
	multi_array<scalar, s_rows, s_columns, 2> fn(const scalar2 xval_, const multi_array<scalar, s_rows, s_columns, 2> &y) { \
		const scalar xval = static_cast<scalar>(xval_); \
		constexpr size_t s_size = s_rows * s_columns; \
		if constexpr (s_size == 1) { return { fn(xval,y[0]) }; } \
		else if constexpr (s_size == 2) { \
			multi_array<scalar, s_rows, s_columns, 2> z; \
			const auto xpackage = p2set1(xval); \
			p2storeu(z.data(), p2fn( xpackage, p2loadu(y.data()) )); \
			return z; \
		} \
		else if constexpr (s_size == 3) { \
			multi_array<scalar, s_rows, s_columns, 2> z; \
			const auto xpackage = p2set1(xval); \
			p2storeu(z.data(), p2fn( xpackage, p2loadu(y.data()) )); \
			z[2] = fn(xval,y[2]); \
			return z; \
		} \
		else if constexpr (s_size == 4) { \
			multi_array<scalar, s_rows, s_columns, 2> z; \
			const auto xpackage = p2set1(xval); \
			p2storeu(z.data(), p2fn( xpackage, p2loadu(y.data()) )); \
			p2storeu(z.data()+2, p2fn( xpackage, p2loadu(y.data()+2) )); \
			return z; \
		} \
		else if constexpr (s_size == 16) { \
			multi_array<scalar, s_rows, s_columns, 2> z; \
			const auto xpackage = p2set1(xval); \
			p2storeu(z.data(), p2fn( xpackage, p2loadu(y.data()) )); \
			p2storeu(z.data()+2, p2fn( xpackage, p2loadu(y.data()+2) )); \
			p2storeu(z.data()+4, p2fn( xpackage, p2loadu(y.data()+4) )); \
			p2storeu(z.data()+6, p2fn( xpackage, p2loadu(y.data()+6) )); \
			p2storeu(z.data()+8, p2fn( xpackage, p2loadu(y.data()+8) )); \
			p2storeu(z.data()+10, p2fn( xpackage, p2loadu(y.data()+10) )); \
			p2storeu(z.data()+12, p2fn( xpackage, p2loadu(y.data()+12) )); \
			p2storeu(z.data()+14, p2fn( xpackage, p2loadu(y.data()+14) )); \
			return z; \
		} \
	\
		multi_array<scalar, s_rows, s_columns, 2> z; \
		multi_array_alloc(z, y); \
		size_t        n  = z.size(); \
		scalar       *zi = z.data(); \
		const scalar *yi = y.data(); \
		const auto xpackage = p2set1(xval); \
		for ( ; n >= 16; n-=16, zi+=16, yi+=16) { \
			p2storeu(zi, p2fn( xpackage, p2loadu(yi) )); \
			p2storeu(zi+2, p2fn( xpackage, p2loadu(yi+2) )); \
			p2storeu(zi+4, p2fn( xpackage, p2loadu(yi+4) )); \
			p2storeu(zi+6, p2fn( xpackage, p2loadu(yi+6) )); \
			p2storeu(zi+8, p2fn( xpackage, p2loadu(yi+8) )); \
			p2storeu(zi+10, p2fn( xpackage, p2loadu(yi+10) )); \
			p2storeu(zi+12, p2fn( xpackage, p2loadu(yi+12) )); \
			p2storeu(zi+14, p2fn( xpackage, p2loadu(yi+14) )); \
		} \
		for ( ; n >= 2; n-=2, zi+=2, yi+=2) { \
			p2storeu(zi, p2fn( xpackage, p2loadu(yi) )); \
		} \
		for ( ; n != 0; --n, ++zi, ++yi) { \
			(*zi) = fn(xval,*yi); \
		} \
		return std::move(z); \
	}
#endif

#ifndef _MATH_SIMD_MULTI_ARRAY_
#endif
}