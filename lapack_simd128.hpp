#pragma once

#if __has_include("multi_array.hpp")
#include "multi_array.hpp"
#endif
#if __has_include("simd_multi_array.hpp")
#include "simd_multi_array.hpp"
#endif

#include <cmath>

#include <intrin.h>
#ifndef _mm_abs_ps
#define _mm_abs_ps(x) _mm_andnot_ps(_mm_set1_ps(-0.0f), x)
#endif


/// Linear Algebra Package for SIMD128. 
///		math::multi_array<float,rows,columns,4> 
///		math::multi_array<double,rows,columns,2> 
///		math::simd_multi_array<float,rows,columns,__m128> 
///		math::simd_multi_array<double,rows,columns,__m128d> 
///@license Free 
///@review 2022-5-11 
///@author LongJiangnan, Jiang1998Nan@outlook.com 
#define _MATH_LAPACK_SIMD128_

namespace math {
#ifdef _MATH_MULTI_ARRAY_
	template<size_t static_rows, size_t static_columns>
	using __m128matrixu = multi_array<float, static_rows, static_columns, 4>;

	template<size_t static_size>
	using __m128vectoru = __m128matrixu<static_size, 1>;

	template<size_t M>
	float dot(const __m128vectoru<M> &v1, const __m128vectoru<M> &v2) {
		if constexpr (M == 4) { return _mm_cvtss_f32( _mm_dp_ps(_mm_loadu_ps(v1.data()), _mm_loadu_ps(v2.data()), 0b1111'0001) ); }
		else if constexpr (M == 3) { return _mm_cvtss_f32( _mm_dp_ps(_mm_loadu_ps(v1.data()), _mm_loadu_ps(v2.data()), 0b0111'0001) ); }
		else if constexpr (M == 2) { return _mm_cvtss_f32( _mm_dp_ps(_mm_loadu_ps(v1.data()), _mm_loadu_ps(v2.data()), 0b0011'0001) ); }
		else if constexpr (M == 1) { return _mm_cvtss_f32( _mm_mul_ss(_mm_loadu_ps(v1.data()), _mm_loadu_ps(v2.data())) ); }

		__m128 sum = _mm_setzero_ps();
		size_t       n  = v1.size();
		const float *i1 = v1.data();
		const float *i2 = v2.data();
		if (n > 4) {
			for ( ; n >= 16; n-=16, i1+=16, i2+=16) {
				sum = _mm_fmadd_ps( _mm_loadu_ps(i1), _mm_loadu_ps(i2), sum );
				sum = _mm_fmadd_ps( _mm_loadu_ps(i1+4), _mm_loadu_ps(i2+4), sum );
				sum = _mm_fmadd_ps( _mm_loadu_ps(i1+8), _mm_loadu_ps(i2+8), sum );
				sum = _mm_fmadd_ps( _mm_loadu_ps(i1+12), _mm_loadu_ps(i2+12), sum );
			}
			for ( ; n >= 4; n-=4, i1+=4, i2+=4) {
				sum = _mm_fmadd_ps( _mm_loadu_ps(i1), _mm_loadu_ps(i2), sum );
			}
			sum = _mm_hadd_ps(sum, sum);
			sum = _mm_hadd_ps(sum, sum);
		}
		switch (n) {
		case 4: sum = _mm_add_ss( _mm_dp_ps(_mm_loadu_ps(i1), _mm_loadu_ps(i2), 0b1111'0001), sum ); break;
		case 3: sum = _mm_add_ss( _mm_dp_ps(_mm_loadu_ps(i1), _mm_loadu_ps(i2), 0b0111'0001), sum ); break;
		case 2: sum = _mm_add_ss( _mm_dp_ps(_mm_loadu_ps(i1), _mm_loadu_ps(i2), 0b0011'0001), sum ); break;
		case 1: sum = _mm_add_ss( _mm_mul_ss(_mm_loadu_ps(i1), _mm_loadu_ps(i2)), sum ); break;
		default: break;
		}
		return _mm_cvtss_f32(sum);
	}

	template<size_t M>
	float square(const __m128vectoru<M> &v) {
		if constexpr (M == 4) { 
			__m128 xmm1 = _mm_loadu_ps(v.data());
			return _mm_cvtss_f32( _mm_dp_ps(xmm1,xmm1,0b1111'0001) );
		} else if constexpr (M == 3) { 
			__m128 xmm1 = _mm_loadu_ps(v.data());
			return _mm_cvtss_f32( _mm_dp_ps(xmm1,xmm1,0b0111'0001) );
		} else if constexpr (M == 2) { 
			__m128 xmm1 = _mm_loadu_ps(v.data());
			return _mm_cvtss_f32( _mm_dp_ps(xmm1,xmm1,0b0011'0001) );
		} else if constexpr (M == 1) { 
			__m128 xmm1 = _mm_loadu_ps(v.data());
			return _mm_cvtss_f32( _mm_mul_ss(xmm1,xmm1) );
		}

		__m128 sum = _mm_setzero_ps();
		size_t       n  = v.size();
		const float *i1 = v.data();
		if (n > 4) {
			for ( ; n >= 16; n-=16, i1+=16, i2+=16) {
				__m128 xmm1 = _mm_loadu_ps(i1);
				sum = _mm_fmadd_ps( xmm1, xmm1, sum );
				xmm1 = _mm_loadu_ps(i1+4);
				sum = _mm_fmadd_ps( xmm1, xmm1, sum );
				xmm1 = _mm_loadu_ps(i1+8);
				sum = _mm_fmadd_ps( xmm1, xmm1, sum );
				xmm1 = _mm_loadu_ps(i1+12);
				sum = _mm_fmadd_ps( xmm1, xmm1, sum );
			}
			for ( ; n >= 4; n-=4, i1+=4, i2+=4) {
				__m128 xmm1 = _mm_loadu_ps(i1);
				sum = _mm_fmadd_ps( xmm1, xmm1, sum );
			}
			sum = _mm_hadd_ps(sum, sum);
			sum = _mm_hadd_ps(sum, sum);
		}
		if (n == 4) {
			__m128 xmm1 = _mm_loadu_ps(i1);
			sum = _mm_add_ss( _mm_dp_ps(xmm1,xmm1,0b1111'0001), sum );
		} else if (n == 3) {
			__m128 xmm1 = _mm_loadu_ps(i1);
			sum = _mm_add_ss( _mm_dp_ps(xmm1,xmm1,0b0111'0001), sum );
		} else if (n == 2) {
			__m128 xmm1 = _mm_loadu_ps(i1);
			sum = _mm_add_ss( _mm_dp_ps(xmm1,xmm1,0b0011'0001), sum );
		} else if (n == 2) {
			__m128 xmm1 = _mm_loadu_ps(i1);
			sum = _mm_add_ss( _mm_mul_ss(xmm1,xmm1), sum );
		} 
		return _mm_cvtss_f32(sum);
	}

	template<size_t M>
	float norm(const __m128vectoru<M> &v, int p) {
		__m128 xmmp = _mm_set1_ps(p);
		if constexpr (M == 4) { 
			__m128 xmm1 = _mm_pow_ps(_mm_abs_ps(_mm_loadu_ps(v.data())), xmmp);
			xmm1 = _mm_hadd_ps(xmm1, xmm1);
			return pow( _mm_cvtss_f32(_mm_hadd_ps(xmm1, xmm1)), 1.0f/p );
		} else if constexpr (M == 3) { 
			__m128 xmm1 = _mm_pow_ps(_mm_abs_ps(_mm_loadu_ps(v.data())), xmmp);
			__m128 xmm2 = _mm_hadd_ps(xmm1, xmm1);
			return pow( _mm_cvtss_f32(_mm_add_ss(_mm_movehl_ps(xmm1,xmm1), xmm2)), 1.0f/p );
		} else if constexpr (M == 2) { 
			__m128 xmm1 = _mm_pow_ps(_mm_abs_ps(_mm_loadu_ps(v.data())), xmmp);
			return pow( _mm_cvtss_f32(_mm_hadd_ps(xmm1,xmm1)), 1.0f/p );
		} else if constexpr (M == 1) { 
			return _mm_cvtss_f32(_mm_abs_ps(_mm_loadu_ps(v.data())));
		}

		__m128 sum = _mm_setzero_ps();
		size_t       n  = v.size();
		const float *i1 = v.data();
		if (n > 4) {
			for ( ; n >= 16; n-=16, i1+=16, i2+=16) {
				__m128 xmm1 = _mm_pow_ps(_mm_abs_ps(_mm_loadu_ps(i1)), xmmp);
				sum = _mm_add_ps( xmm1, sum );
				xmm1 = _mm_pow_ps(_mm_abs_ps(_mm_loadu_ps(i1+4)), xmmp);
				sum = _mm_add_ps( xmm1, sum );
				xmm1 = _mm_pow_ps(_mm_abs_ps(_mm_loadu_ps(i1+8)), xmmp);
				sum = _mm_add_ps( xmm1, sum );
				xmm1 = _mm_pow_ps(_mm_abs_ps(_mm_loadu_ps(i1+12)), xmmp);
				sum = _mm_add_ps( xmm1, sum );
			}
			for ( ; n >= 4; n-=4, i1+=4, i2+=4) {
				__m128 xmm1 = _mm_pow_ps(_mm_abs_ps(_mm_loadu_ps(i1)), xmmp);
				sum = _mm_add_ps( xmm1, sum );
			}
			sum = _mm_hadd_ps(sum, sum);
			sum = _mm_hadd_ps(sum, sum);
		}
		if (n == 4) { 
			__m128 xmm1 = _mm_pow_ps(_mm_abs_ps(_mm_loadu_ps(i1)), xmmp);
			xmm1 = _mm_hadd_ps(xmm1, xmm1);
			sum = _mm_add_ss( _mm_hadd_ps(xmm1, xmm1), sum );
		} else if (n == 3) { 
			__m128 xmm1 = _mm_pow_ps(_mm_abs_ps(_mm_loadu_ps(i1)), xmmp);
			__m128 xmm2 = _mm_hadd_ps(xmm1, xmm1);
			sum = _mm_add_ss( _mm_add_ss(_mm_movehl_ps(xmm1,xmm1), xmm2), sum );
		} else if (n == 2) { 
			__m128 xmm1 = _mm_pow_ps(_mm_abs_ps(_mm_loadu_ps(i1)), xmmp);
			sum = _mm_add_ss( _mm_hadd_ps(xmm1, xmm1), sum );
		} else if (n == 1) { 
			__m128 xmm1 = _mm_pow_ps(_mm_abs_ps(_mm_loadu_ps(i1)), xmmp);
			sum = _mm_add_ss( xmm1, sum );
		}
		return pow(_mm_cvtss_f32(sum), 1.0f/p);
	}

	template<size_t M> inline
	float length(const __m128vectoru<M> &v) {
		return sqrt(square(v));
	}

#ifndef matrix_alloc
	#define matrix_alloc(destination, rows, columns) \
	if constexpr (::math:: multi_array_traits<decltype(destination)>::static_rows == 0 \
		&& ::math:: multi_array_traits<decltype(destination)>::static_columns == 0) { \
		destination.reshape(rows, columns); \
	} else if constexpr (::math:: multi_array_traits<decltype(destination)>::static_rows == 0) { \
		destination.rerow(rows); \
	} else if constexpr (::math:: multi_array_traits<decltype(destination)>::static_columns == 0) { \
		destination.recolumn(columns); \
	}
#endif

	template<size_t M> inline
	__m128vectoru<M> normalize(const __m128vectoru<M> &v) {
		float sqlen = square(v);
		if (sqlen == 0/* || sqlen == 1*/) {
			return v;
		} else {
			return v/sqrt(sqlen);
		}
	}

	template<size_t M> inline
	__m128vectoru<M> cross(const __m128vectoru<M> &v1, const __m128vectoru<M> &v2) {
		//return { v1[1]*v2[2]-v1[2]*v2[1], v1[2]*v2[0]-v1[0]*v2[2], v1[0]*v2[1]-v1[1]*v2[0] };
		static_assert( M >= 3 );
		__m128vectoru<M> v3;
		__m128 xmm1 = _mm_loadu_ps(v1.data());
		__m128 xmm2 = _mm_loadu_ps(v2.data());
		_mm_storeu_ps(v3.data(), _mm_sub_ps(
			_mm_mul_ps( _mm_permute_ps(xmm1, _MM_PERM_AACB), _mm_permute_ps(xmm2, _MM_PERM_ABAC) ),
			_mm_mul_ps( _mm_permute_ps(xmm1, _MM_PERM_ABAC), _mm_permute_ps(xmm2, _MM_PERM_AACB) ) ));
		return v3;
	}

	template<size_t M, size_t N>
	__m128matrixu<N,M> transpose(const __m128matrixu<M,N> &A) {
		if constexpr (M == 2 && N == 2) {
			__m128matrixu<N,M> tA;
			_mm_storeu_ps(tA.data(), 
				_mm_permute_ps( _mm_loadu_ps(A.data()), _MM_PERM_DBCA ) );
			return tA;
		}
		else if constexpr (M == 4 && N == 4) {
			__m128matrixu<N,M> tA;
			__m128 row0 = _mm_loadu_ps(A.data());
			__m128 row1 = _mm_loadu_ps(A.data()+4);
			__m128 row2 = _mm_loadu_ps(A.data()+8);
			__m128 row3 = _mm_loadu_ps(A.data()+12);
			_MM_TRANSPOSE4_PS(row0, row1, row2, row3);
			_mm_storeu_ps(tA.data(), row0);
			_mm_storeu_ps(tA.data()+4, row1);
			_mm_storeu_ps(tA.data()+8, row2);
			_mm_storeu_ps(tA.data()+12, row3);
			return tA;
		}

		__m128matrixu<N,M> tA;
		matrix_alloc(tA, A.columns(), A.rows())
		for (size_t i = 0; i != A.rows(); ++i)
			for (size_t j = 0; j != A.columns(); ++j)
				tA.at(j,i) = A.at(i,j);
		return std::move(tA);
	}

	

	template<size_t M, size_t N, size_t K>
	__m128matrixu<M,N> gemm(const __m128matrixu<M,K> &A, const __m128matrixu<K,N> &B) {
		if constexpr (M == 1 && K == 1 && N == 1) { return { A[0]*B[0] }; }
		else if constexpr (M == 2 && K == 2 && N == 1) { return { A[0]*B[0]+A[1]*B[1], A[2]*B[0]+A[3]*B[1] }; }
		else if constexpr (M == 3 && K == 3 && N == 1) { 
			// mul(matrix3x3,matrix3*1).
			return { _mm_cvtss_f32( _mm_dp_ps(_mm_loadu_ps(A.data()), _mm_loadu_ps(B.data()), 0b0111'0001) ),
				_mm_cvtss_f32( _mm_dp_ps(_mm_loadu_ps(A.data()+3), _mm_loadu_ps(B.data()), 0b0111'0001) ),
				_mm_cvtss_f32( _mm_dp_ps(_mm_loadu_ps(A.data()+6), _mm_loadu_ps(B.data()), 0b0111'0001) ) };
		}
		else if constexpr (M == 4 && K == 4 && N == 1) {
			// mul(matrix4x4,matrix4*1).
			return { _mm_cvtss_f32( _mm_dp_ps(_mm_loadu_ps(A.data()), _mm_loadu_ps(B.data()), 0b1111'0001) ),
				_mm_cvtss_f32( _mm_dp_ps(_mm_loadu_ps(A.data()+4), _mm_loadu_ps(B.data()), 0b1111'0001) ),
				_mm_cvtss_f32( _mm_dp_ps(_mm_loadu_ps(A.data()+8), _mm_loadu_ps(B.data()), 0b1111'0001) ),
				_mm_cvtss_f32( _mm_dp_ps(_mm_loadu_ps(A.data()+12), _mm_loadu_ps(B.data()), 0b1111'0001) ) };
		}
		else if constexpr (M == 2 && K == 2 && N == 2) {  
			//return { A[0]*B[0]+A[1]*B[2], A[0]*B[1]+A[1]*B[3], A[2]*B[0]+A[3]*B[2], A[2]*B[1]+A[3]*B[3] };
			__m128matrixu<M,N> C;
			__m128 xmm1 = _mm_loadu_ps(A.data());
			__m128 xmm2 = _mm_loadu_ps(B.data());
			_mm_storeu_ps(C.data(), _mm_add_ps(
				_mm_mul_ps( _mm_permute_ps(xmm1, _MM_PERM_CCAA), _mm_permute_ps(xmm2, _MM_PERM_BABA) ),
				_mm_mul_ps( _mm_permute_ps(xmm1, _MM_PERM_DDBB), _mm_permute_ps(xmm2, _MM_PERM_DCDC) ) ));
			return C;
		}
		else if constexpr (M == 4 && K == 4 && N == 4) {
			// mul(matrix4x4,matrix4*4).
			__m128matrixu<M,N> C;
			auto temp = _mm_mul_ps( _mm_set1_ps(A[0]), _mm_loadu_ps(B.data()) );
			temp = _mm_fmadd_ps( _mm_set1_ps(A[1]), _mm_loadu_ps(B.data()+4), temp );
			temp = _mm_fmadd_ps( _mm_set1_ps(A[2]), _mm_loadu_ps(B.data()+8), temp );
			temp = _mm_fmadd_ps( _mm_set1_ps(A[3]), _mm_loadu_ps(B.data()+12), temp );
			_mm_storeu_ps(C.data(), temp);
			temp = _mm_mul_ps( _mm_set1_ps(A[4]), _mm_loadu_ps(B.data()) );
			temp = _mm_fmadd_ps( _mm_set1_ps(A[5]), _mm_loadu_ps(B.data()+4), temp );
			temp = _mm_fmadd_ps( _mm_set1_ps(A[6]), _mm_loadu_ps(B.data()+8), temp );
			temp = _mm_fmadd_ps( _mm_set1_ps(A[7]), _mm_loadu_ps(B.data()+12), temp );
			_mm_storeu_ps(C.data()+4, temp);
			temp = _mm_mul_ps( _mm_set1_ps(A[8]), _mm_loadu_ps(B.data()) );
			temp = _mm_fmadd_ps( _mm_set1_ps(A[9]), _mm_loadu_ps(B.data()+4), temp );
			temp = _mm_fmadd_ps( _mm_set1_ps(A[10]), _mm_loadu_ps(B.data()+8), temp );
			temp = _mm_fmadd_ps( _mm_set1_ps(A[11]), _mm_loadu_ps(B.data()+12), temp );
			_mm_storeu_ps(C.data()+8, temp);
			temp = _mm_mul_ps( _mm_set1_ps(A[12]), _mm_loadu_ps(B.data()) );
			temp = _mm_fmadd_ps( _mm_set1_ps(A[13]), _mm_loadu_ps(B.data()+4), temp );
			temp = _mm_fmadd_ps( _mm_set1_ps(A[14]), _mm_loadu_ps(B.data()+8), temp );
			temp = _mm_fmadd_ps( _mm_set1_ps(A[15]), _mm_loadu_ps(B.data()+12), temp );
			_mm_storeu_ps(C.data()+12, temp);
			return C;
		}

		__m128matrixu<M,N> C;
		matrix_alloc(C, A.rows(), B.columns());
		const size_t m  = C.rows();
		const size_t n  = C.columns();
		float       *Ci = C.data();
		const float *Ai = A.data();
		const float *Bi = B.data();
		size_t       ct;
		__m128       xmm1;
		for (size_t i = 0; i != m; ++i, Bi=B.data()) {
			// row(C,i) = A[i][0] * row(B,0).
			Ci = C.data()+i*n;
			ct = B.columns();
			xmm1 = _mm_set1_ps(*Ai);
			for ( ; ct >= 16; ct-=16, Ci+=16, Bi+=16) {
				_mm_storeu_ps(Ci, _mm_mul_ps( xmm1, _mm_loadu_ps(Bi) ));
				_mm_storeu_ps(Ci+4, _mm_mul_ps( xmm1, _mm_loadu_ps(Bi+4) ));
				_mm_storeu_ps(Ci+8, _mm_mul_ps( xmm1, _mm_loadu_ps(Bi+8) ));
				_mm_storeu_ps(Ci+12, _mm_mul_ps( xmm1, _mm_loadu_ps(Bi+12) ));
			}
			for ( ; ct >= 4; ct-=4, Ci+=4, Bi+=4) {
				_mm_storeu_ps(Ci, _mm_mul_ps( xmm1, _mm_loadu_ps(Bi) ));
			}
			for ( ; ct != 0; --ct, ++Ci, ++Bi) {
				(*Ci) = (*Ai) * (*Bi);
			}
			++Ai;

			// row(C,i) += A[i][j] * row(B,j).
			for (size_t j = 1; j != B.rows(); ++j, ++Ai) {
				Ci = C.data()+i*n;
				ct = B.columns();
				xmm1 = _mm_set1_ps(*Ai);
				for ( ; ct >= 16; ct-=16, Ci+=16, Bi+=16) {
					_mm_storeu_ps(Ci, _mm_fmadd_ps( xmm1, _mm_loadu_ps(Bi), _mm_loadu_ps(Ci) ));
					_mm_storeu_ps(Ci+4, _mm_fmadd_ps( xmm1, _mm_loadu_ps(Bi+4), _mm_loadu_ps(Ci+4) ));
					_mm_storeu_ps(Ci+8, _mm_fmadd_ps( xmm1, _mm_loadu_ps(Bi+8), _mm_loadu_ps(Ci+8) ));
					_mm_storeu_ps(Ci+12, _mm_fmadd_ps( xmm1, _mm_loadu_ps(Bi+12), _mm_loadu_ps(Ci+12) ));
				}
				for ( ; ct >= 4; ct-=4, Ci+=4, Bi+=4) {
					_mm_storeu_ps(Ci, _mm_fmadd_ps( xmm1, _mm_loadu_ps(Bi), _mm_loadu_ps(Ci) ));
				}
				for ( ; ct != 0; --ct, ++Ci, ++Bi) {
					(*Ci) += (*Ai) * (*Bi);
				}
			}
		}

		return std::move(C);
	}
#endif


#ifdef _MATH_SIMD_MULTI_ARRAY_
	template<size_t static_rows = 0, size_t static_columns = 0>
	using __m128mat = simd_multi_array<float, static_rows, static_columns, __m128>;

	template<size_t static_size = 0>
	using __m128vec = __m128mat<static_size, 1>;

#pragma region Equation
	/// solve A*x = b.
	template<size_t M, size_t N, size_t K>
	void gesv(__m128mat<M,N>& A, __m128mat<M,K>& b) {
		
	}

	/// P*A = P*L*U, factorization.
	template<size_t M, size_t N>
	void getrf(__m128mat<M,N>& A) {
		
	}
#pragma endregion

#pragma region Basic-Level1
	/// x[i] = y[i].
	template<size_t M, size_t N>
	void copy(const __m128mat<M,N>& x, __m128mat<M,N>& y) {
		if constexpr (M != 0 && N != 0) {
			if constexpr (y.actual_size() <= 16) {
				if constexpr (y.actual_size() > 0)
					_mm_store_ps(y.data(), _mm_load_ps(x.data()));
				if constexpr (y.actual_size() > 4)
					_mm_store_ps(y.data()+4, _mm_load_ps(x.data()+4));
				if constexpr (y.actual_size() > 8)
					_mm_store_ps(y.data()+8, _mm_load_ps(x.data()+8));
				if constexpr (y.actual_size() > 12)
					_mm_store_ps(y.data()+12, _mm_load_ps(x.data()+12));
				return ;
			}
		}
		size_t       n  = y.actual_size();
		float       *yi = y.data();
		const float *xi = x.data();
		for ( ; n >= 16; n-=16, yi+=16, xi+=16) {
			_mm_store_ps(yi, _mm_load_ps(xi));
			_mm_store_ps(yi+4, _mm_load_ps(xi+4));
			_mm_store_ps(yi+8, _mm_load_ps(xi+8));
			_mm_store_ps(yi+12, _mm_load_ps(xi+12));
		}
		for ( ; n >= 4; n-=4, yi+=4, xi+=4) {
			_mm_store_ps(yi, _mm_load_ps(xi));
		}
	}

	/// x[i] = y[i], y[i] = x[i].
	template<size_t M, size_t N>
	void swap(__m128mat<M,N>& x, __m128mat<M,N>& y) {
		__m128 xmm1;
		size_t       n  = y.actual_size();
		float       *yi = y.data();
		float       *xi = x.data();
		for ( ; n >= 16; n-=16, yi+=16, xi+=16) {
			xmm1 = _mm_load_ps(xi);
			_mm_store_ps(xi, _mm_load_ps(yi));
			_mm_store_ps(yi, xmm1);
			xmm1 = _mm_load_ps(xi+4);
			_mm_store_ps(xi+4, _mm_load_ps(yi+4));
			_mm_store_ps(yi+4, xmm1);
			xmm1 = _mm_load_ps(xi+8);
			_mm_store_ps(xi+8, _mm_load_ps(yi+8));
			_mm_store_ps(yi+8, xmm1);
			xmm1 = _mm_load_ps(xi+12);
			_mm_store_ps(xi+12, _mm_load_ps(yi+12));
			_mm_store_ps(yi+12, xmm1);
		}
		for ( ; n >= 4; n-=4, yi+=4, xi+=4) {
			xmm1 = _mm_load_ps(xi);
			_mm_store_ps(xi, _mm_load_ps(yi));
			_mm_store_ps(yi, xmm1);
		}
	}

	/// x[i] = alpha*x[i].
	template<size_t M, size_t N>
	void scale(float alpha, __m128mat<M,N>& x) {
		__m128 xmm1 = _mm_set1_ps(alpha);
		if constexpr (M != 0 && N != 0) {
			if constexpr (x.actual_size() <= 16) {
				if constexpr (x.actual_size() > 0)
					_mm_store_ps(x.data(), _mm_mul_ps( xmm1, _mm_load_ps(x.data()) ));
				if constexpr (y.actual_size() > 4)
					_mm_store_ps(x.data()+4, _mm_mul_ps( xmm1, _mm_load_ps(x.data()+4) ));
				if constexpr (x.actual_size() > 8)
					_mm_store_ps(x.data()+8, _mm_mul_ps( xmm1, _mm_load_ps(x.data()+8) ));
				if constexpr (x.actual_size() > 12)
					_mm_store_ps(x.data()+12, _mm_mul_ps( xmm1, _mm_load_ps(x.data()+12) ));
				return ;
			}
		}
		if (alpha != 1) {
			size_t       n  = x.actual_size();
			float       *xi = x.data();
			for ( ; n >= 16; n-=16, xi+=16) {
				_mm_store_ps(yi, _mm_mul_ps( xmm1, _mm_load_ps(xi) ));
				_mm_store_ps(yi+4, _mm_mul_ps( xmm1, _mm_load_ps(xi+4) ));
				_mm_store_ps(yi+8, _mm_mul_ps( xmm1, _mm_load_ps(xi+8) ));
				_mm_store_ps(yi+12, _mm_mul_ps( xmm1, _mm_load_ps(xi+12) ));
			}
			for ( ; n >= 4; n-=4, xi+=4) {
				_mm_store_ps(yi, _mm_mul_ps( xmm1, _mm_load_ps(xi) ));
			}
		}
	}

	/// y[i] = alpha*x[i] + y[i].
	template<size_t M, size_t N>
	void axpy(float alpha, const __m128mat<M,N>& x, __m128mat<M,N>& y) {
		__m128 xmm1 = _mm_set1_ps(alpha);
		if constexpr (M != 0 && N != 0) {
			if constexpr (y.actual_size() <= 16) {
				if constexpr (y.actual_size() > 0)
					_mm_store_ps(y.data(), _mm_fmadd_ps( xmm1, _mm_load_ps(x.data()), _mm_load_ps(y.data()) ));
				if constexpr (y.actual_size() > 4)
					_mm_store_ps(y.data()+4, _mm_fmadd_ps( xmm1, _mm_load_ps(x.data()+4), _mm_load_ps(y.data()+4) ));
				if constexpr (y.actual_size() > 8)
					_mm_store_ps(y.data()+8, _mm_fmadd_ps( xmm1, _mm_load_ps(x.data()+8), _mm_load_ps(y.data()+8) ));
				if constexpr (y.actual_size() > 12)
					_mm_store_ps(y.data()+12, _mm_fmadd_ps( xmm1, _mm_load_ps(x.data()+12), _mm_load_ps(y.data()+12) ));
				return ;
			}
		}
		if (alpha != 0) {
			size_t       n  = y.actual_size();
			float       *yi = y.data();
			const float *xi = x.data();
			for ( ; n >= 16; n-=16, yi+=16, xi+=16) {
				_mm_store_ps(yi, _mm_fmadd_ps( xmm1, _mm_load_ps(xi), _mm_load_ps(yi) ));
				_mm_store_ps(yi+4, _mm_fmadd_ps( xmm1, _mm_load_ps(xi+4), _mm_load_ps(yi+4) ));
				_mm_store_ps(yi+8, _mm_fmadd_ps( xmm1, _mm_load_ps(xi+8), _mm_load_ps(yi+8) ));
				_mm_store_ps(yi+12, _mm_fmadd_ps( xmm1, _mm_load_ps(xi+12), _mm_load_ps(yi+12) ));
			}
			for ( ; n >= 4; n-=4, yi+=4, xi+=4) {
				_mm_store_ps(yi, _mm_fmadd_ps( xmm1, _mm_load_ps(xi), _mm_load_ps(yi) ));
			}
		}
	}

	/// sum(x[i] * y[i]).
	template<size_t M>
	float dot(const __m128vec<M>& x, const __m128vec<M>& y) {
		if constexpr (M == 4) { 
			return _mm_cvtss_f32( _mm_dp_ps(_mm_load_ps(x.data()), _mm_load_ps(y.data()), 0b1111'0001) ); 
		} else if constexpr (M == 3) { 
			return _mm_cvtss_f32( _mm_dp_ps(_mm_load_ps(x.data()), _mm_load_ps(y.data()), 0b0111'0001) ); 
		} else if constexpr (M == 2) { 
			return _mm_cvtss_f32( _mm_dp_ps(_mm_load_ps(x.data()), _mm_load_ps(y.data()), 0b0011'0001) ); 
		} else if constexpr (M == 1) { 
			return _mm_cvtss_f32( _mm_mul_ss(_mm_load_ps(x.data()), _mm_load_ps(y.data())) ); 
		}
		__m128 sum = _mm_setzero_ps();
		size_t       n  = x.size();
		const float *xi = x.data();
		const float *yi = y.data();
		if (n > 4) {
			for ( ; n >= 16; n-=16, xi+=16, yi+=16) {
				sum = _mm_fmadd_ps( _mm_load_ps(xi), _mm_load_ps(yi), sum );
				sum = _mm_fmadd_ps( _mm_load_ps(xi+4), _mm_load_ps(yi+4), sum );
				sum = _mm_fmadd_ps( _mm_load_ps(xi+8), _mm_load_ps(yi+8), sum );
				sum = _mm_fmadd_ps( _mm_load_ps(xi+12), _mm_load_ps(yi+12), sum );
			}
			for ( ; n >= 4; n-=4, xi+=4, yi+=4) {
				sum = _mm_fmadd_ps( _mm_load_ps(xi), _mm_load_ps(yi), sum );
			}
			sum = _mm_hadd_ps(sum, sum);
			sum = _mm_hadd_ps(sum, sum);
		}
		switch (n) {
			case 4: sum = _mm_add_ss( _mm_dp_ps(_mm_load_ps(xi), _mm_load_ps(yi), 0b1111'0001), sum ); break;
			case 3: sum = _mm_add_ss( _mm_dp_ps(_mm_load_ps(xi), _mm_load_ps(yi), 0b0111'0001), sum ); break;
			case 2: sum = _mm_add_ss( _mm_dp_ps(_mm_load_ps(xi), _mm_load_ps(yi), 0b0011'0001), sum ); break;
			case 1: sum = _mm_add_ss( _mm_mul_ss(_mm_load_ps(xi), _mm_load_ps(yi)), sum ); break;
			default: break;
		}
		return _mm_cvtss_f32(sum);
	}

	/// sum(x[i] * x[i]).
	template<size_t M>
	float square(const __m128vec<M>& x) {
		if constexpr (M == 4) { 
			__m128 xmm1 = _mm_load_ps(x.data());
			return _mm_cvtss_f32( _mm_dp_ps(xmm1,xmm1,0b1111'0001) );
		} else if constexpr (M == 3) { 
			__m128 xmm1 = _mm_load_ps(x.data());
			return _mm_cvtss_f32( _mm_dp_ps(xmm1,xmm1,0b0111'0001) );
		} else if constexpr (M == 2) { 
			__m128 xmm1 = _mm_load_ps(x.data());
			return _mm_cvtss_f32( _mm_dp_ps(xmm1,xmm1,0b0011'0001) );
		} else if constexpr (M == 1) { 
			__m128 xmm1 = _mm_load_ps(x.data());
			return _mm_cvtss_f32( _mm_mul_ss(xmm1,xmm1) );
		}
		__m128 sum = _mm_setzero_ps();
		size_t       n  = x.size();
		const float *xi = x.data();
		if (n > 4) {
			for ( ; n >= 16; n-=16, xi+=16, i2+=16) {
				__m128 xmm1 = _mm_load_ps(xi);
				sum = _mm_fmadd_ps( xmm1, xmm1, sum );
				xmm1 = _mm_load_ps(xi+4);
				sum = _mm_fmadd_ps( xmm1, xmm1, sum );
				xmm1 = _mm_load_ps(xi+8);
				sum = _mm_fmadd_ps( xmm1, xmm1, sum );
				xmm1 = _mm_load_ps(xi+12);
				sum = _mm_fmadd_ps( xmm1, xmm1, sum );
			}
			for ( ; n >= 4; n-=4, xi+=4, i2+=4) {
				__m128 xmm1 = _mm_load_ps(xi);
				sum = _mm_fmadd_ps( xmm1, xmm1, sum );
			}
			sum = _mm_hadd_ps(sum, sum);
			sum = _mm_hadd_ps(sum, sum);
		}
		if (n == 4) {
			__m128 xmm1 = _mm_load_ps(xi);
			sum = _mm_add_ss( _mm_dp_ps(xmm1,xmm1,0b1111'0001), sum );
		} else if (n == 3) {
			__m128 xmm1 = _mm_load_ps(xi);
			sum = _mm_add_ss( _mm_dp_ps(xmm1,xmm1,0b0111'0001), sum );
		} else if (n == 2) {
			__m128 xmm1 = _mm_load_ps(xi);
			sum = _mm_add_ss( _mm_dp_ps(xmm1,xmm1,0b0011'0001), sum );
		} else if (n == 2) {
			__m128 xmm1 = _mm_load_ps(xi);
			sum = _mm_add_ss( _mm_mul_ss(xmm1,xmm1), sum );
		} 
		return _mm_cvtss_f32(sum);
	}

	/// sum(|x[i]|^p)^(1/p).
	template<size_t M>
	float norm(const __m128vec<M>& x, int p) {
		__m128 xmmp = _mm_set1_ps(p);
		if constexpr (M == 4) { 
			__m128 xmm1 = _mm_pow_ps(_mm_abs_ps(_mm_load_ps(x.data())), xmmp);
			xmm1 = _mm_hadd_ps(xmm1, xmm1);
			return pow( _mm_cvtss_f32(_mm_hadd_ps(xmm1, xmm1)), 1.0f/p );
		} else if constexpr (M == 3) { 
			__m128 xmm1 = _mm_pow_ps(_mm_abs_ps(_mm_load_ps(x.data())), xmmp);
			__m128 xmm2 = _mm_hadd_ps(xmm1, xmm1);
			return pow( _mm_cvtss_f32(_mm_add_ss(_mm_movehl_ps(xmm1,xmm1), xmm2)), 1.0f/p );
		} else if constexpr (M == 2) { 
			__m128 xmm1 = _mm_pow_ps(_mm_abs_ps(_mm_load_ps(x.data())), xmmp);
			return pow( _mm_cvtss_f32(_mm_hadd_ps(xmm1,xmm1)), 1.0f/p );
		} else if constexpr (M == 1) { 
			return _mm_cvtss_f32(_mm_abs_ps(_mm_load_ps(x.data())));
		}
		__m128 sum = _mm_setzero_ps();
		size_t       n  = x.size();
		const float *xi = x.data();
		if (n > 4) {
			for ( ; n >= 16; n-=16, xi+=16, i2+=16) {
				__m128 xmm1 = _mm_pow_ps(_mm_abs_ps(_mm_load_ps(xi)), xmmp);
				sum = _mm_add_ps( xmm1, sum );
				xmm1 = _mm_pow_ps(_mm_abs_ps(_mm_load_ps(xi+4)), xmmp);
				sum = _mm_add_ps( xmm1, sum );
				xmm1 = _mm_pow_ps(_mm_abs_ps(_mm_load_ps(xi+8)), xmmp);
				sum = _mm_add_ps( xmm1, sum );
				xmm1 = _mm_pow_ps(_mm_abs_ps(_mm_load_ps(xi+12)), xmmp);
				sum = _mm_add_ps( xmm1, sum );
			}
			for ( ; n >= 4; n-=4, xi+=4, i2+=4) {
				__m128 xmm1 = _mm_pow_ps(_mm_abs_ps(_mm_load_ps(xi)), xmmp);
				sum = _mm_add_ps( xmm1, sum );
			}
			sum = _mm_hadd_ps(sum, sum);
			sum = _mm_hadd_ps(sum, sum);
		}
		if (n == 4) { 
			__m128 xmm1 = _mm_pow_ps(_mm_abs_ps(_mm_load_ps(xi)), xmmp);
			xmm1 = _mm_hadd_ps(xmm1, xmm1);
			sum = _mm_add_ss( _mm_hadd_ps(xmm1, xmm1), sum );
		} else if (n == 3) { 
			__m128 xmm1 = _mm_pow_ps(_mm_abs_ps(_mm_load_ps(xi)), xmmp);
			__m128 xmm2 = _mm_hadd_ps(xmm1, xmm1);
			sum = _mm_add_ss( _mm_add_ss(_mm_movehl_ps(xmm1,xmm1), xmm2), sum );
		} else if (n == 2) { 
			__m128 xmm1 = _mm_pow_ps(_mm_abs_ps(_mm_load_ps(xi)), xmmp);
			sum = _mm_add_ss( _mm_hadd_ps(xmm1, xmm1), sum );
		} else if (n == 1) { 
			__m128 xmm1 = _mm_pow_ps(_mm_abs_ps(_mm_load_ps(xi)), xmmp);
			sum = _mm_add_ss( xmm1, sum );
		}
		return pow(_mm_cvtss_f32(sum), 1.0f/p);
	}

	/// sqrt(square(x)).
	template<size_t M> inline
	float length(const __m128vec<M>& x) {
		return sqrt(square(x));
	}

	/// x/lenght(x).
	template<size_t M> inline
	__m128vec<M> normalize(const __m128vec<M>& x) {
		float sqlen = square(x);
		if (sqlen == 0/* || sqlen == 1*/) {
			return x;
		} else {
			return x/sqrt(sqlen);
		}
	}
	
	template<size_t M> inline
	__m128vec<M> cross(const __m128vec<M> &v1, const __m128vec<M> &v2) {
		//return { v1[1]*v2[2]-v1[2]*v2[1], v1[2]*v2[0]-v1[0]*v2[2], v1[0]*v2[1]-v1[1]*v2[0] };
		static_assert( M >= 3 );
		__m128vec<M> v3;
		__m128 xmm1 = _mm_load_ps(v1.data());
		__m128 xmm2 = _mm_load_ps(v2.data());
		_mm_store_ps(v3.data(), _mm_sub_ps(
			_mm_mul_ps( _mm_permute_ps(xmm1, _MM_PERM_AACB), _mm_permute_ps(xmm2, _MM_PERM_ABAC) ),
			_mm_mul_ps( _mm_permute_ps(xmm1, _MM_PERM_ABAC), _mm_permute_ps(xmm2, _MM_PERM_AACB) ) ));
		return v3;
	}

	template<size_t M> inline
	__m128vec<M> cross(const __m128vec<M> &v1, const __m128vec<M> &v2, const __m128vec<M>& v3){

	}
#pragma endregion

#pragma region Basic-Level2

#pragma endregion

#pragma region Basic-Level3
	/// B = trans(A).
	template<size_t M, size_t N>
	__m128mat<N,M> trans(const __m128mat<M,N>& A) {
		
	}

	/// A = alpha*op(A).
	void imatcopy(char op, float alpha, __m128mat<>& A);

	/// B = alpha*op(A).
	void omatcopy(char op, float alpha, const __m128mat<>& A, __m128mat<>& B);

	/// C = A*B.
	template<size_t M, size_t N, size_t K>
	__m128mat<M,N> gemm(const __m128mat<M,K>& A, const __m128mat<K,N>& B) {

	}

	/// C = alpha*A*B + beta*C.
	template<size_t M, size_t N, size_t K>
	void gemm(float alpha, const __m128mat<M,K>& A, const __m128mat<K,N>& B, float beta, __m128mat<M,N>& C) {

	}
	
	/// C = alpha*op(A)*op(B) + beta*C.
	void gemm(char opA, char opB, float alpha, const __m128mat<>& A, const __m128mat<>& B, float beta, __m128mat<>& C);
#pragma endregion
#endif
}