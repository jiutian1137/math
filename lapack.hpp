#pragma once

#include "multi_array.hpp"
#include <cmath>


/// Linear Algebra Package. 
///@license Free 
///@review 2022-5-22 
///@author LongJiangnan, Jiang1998Nan@outlook.com 
#define _MATH_LAPACK_

namespace math {
	template<typename T, size_t N = 0>
	using vector = multi_array<T, N, 1>;

	template<typename T, size_t M = 0, size_t N = 0>
	using matrix = multi_array<T, M, N>;

	template<typename T, size_t N>
	using affinet = multi_array<T, N, N + 1>; // affine transform.
	
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

	template<typename T, size_t M, size_t N>
	void swap(matrix<T,M,N>& x, matrix<T,M,N>& y) {
		unroll_16x_4x_for(
			epconj(size_t n = x.size(); T* xi = x.data(); T* yi = y.data()),
			epconj(n >= 16), epconj(n-=16, yi+=16, xi+=16),
			epconj(n >= 4), epconj(n-=4, yi+=4, xi+=4),
			epconj(n != 0), epconj(--n, ++yi, ++xi),
			std::swap(xi[i], yi[i])
		);
	}

	template<typename T, size_t M, size_t N>
	void copy(const matrix<T,M,N>& x, matrix<T,M,N>& y) {
		unroll_16x_4x_for(
			epconj(size_t n = x.size(); const T* xi = x.data(); T* yi = y.data()),
			epconj(n >= 16), epconj(n-=16, yi+=16, xi+=16),
			epconj(n >= 4), epconj(n-=4, yi+=4, xi+=4),
			epconj(n != 0), epconj(--n, ++yi, ++xi),
			yi[i] = xi[i]
		);
	}

	/// x[i] = alpha * x[i].
	template<typename T, size_t M, size_t N>
	void scale(T alpha, matrix<T,M,N>& x) {
		unroll_16x_4x_for(
			epconj(size_t n = x.size(); T* xi = x.data()),
			epconj(n >= 16), epconj(n-=16, xi+=16),
			epconj(n >= 4), epconj(n-=4, xi+=4),
			epconj(n != 0), epconj(--n, ++xi),
			xi[i] *= alpha
		);
	}

	/// y[i] = alpha * x[i] + y[i].
	template<typename T, size_t M, size_t N>
	void axpy(T alpha, const matrix<T,M,N>& x, matrix<T,M,N>& y) {
		unroll_16x_4x_for(
			epconj(size_t n = x.size(); const T* xi = x.data(); T* yi = y.data()),
			epconj(n >= 16), epconj(n-=16, xi+=16, yi+=16),
			epconj(n >= 4), epconj(n-=4, xi+=4, yi+=4),
			epconj(n != 0), epconj(--n, ++xi, ++yi),
			yi[i] += xi[i]*alpha
		);
	}


	/// sum(x[i] * y[i]).
	template<typename T, size_t M>
	T dot(const vector<T,M>& v1, const vector<T,M>& v2) {
		assert(v1.size() != 0 && v2.size() != 0);
		T s2 = v1[0] * v2[0];
		unroll_16x_4x_for(
			epconj(size_t n = v1.size()-1; const T* v1i = v1.data()+1; const T* v2i = v2.data()+1),
			epconj(n >= 16), epconj(n-=16, v1i+=16, v2i+=16),
			epconj(n >= 4), epconj(n-=4, v1i+=4, v2i+=4),
			epconj(n != 0), epconj(--n, ++v1i, ++v2i),
			s2 += v1i[i] * v2i[i]
		);
		return s2;
	}

	/// sum(x[i] * x[i]).
	template<typename T, size_t M>
	T sqr(const vector<T,M>& v) {
		assert(v.size() != 0);
		T s2 = v[0] * v[0];
		unroll_16x_4x_for(
			epconj(size_t n = v.size()-1; const T* vi = v.data()+1),
			epconj(n >= 16), epconj(n-=16, vi+=16),
			epconj(n >= 4), epconj(n-=4, vi+=4),
			epconj(n != 0), epconj(--n, ++vi),
			s2 += vi[i] * vi[i]
		);
		return s2;
	}

	/// sqrt(sum(v[i] * v[i])).
	template<typename T, size_t M> inline
	T length(const vector<T,M>& v) {
		return sqrt(sqr(v));
	}

	/// v/length(v).
	template<typename T, size_t M> inline
	vector<T,M> normalize(const vector<T,M>& v) {
		T s2 = sqr(v);
		if (s2 == 0/* || s2 == 1*/) {
			return v;
		} else {
			return v/sqrt(s2);
		}
	}

	template<typename T, size_t M>
	vector<T,M> cross(const vector<T,M> &v1, const vector<T,M> &v2) {
		///@thory
		/// solve [ dot([i,j,k],v1] = 0 ]
		///       [ dot([i,j,k],v2) = 0 ]
		/// 
		///    | v1.x, v1.y, v1.z |   | i |   | 0 |
		/// => | v2.x, v2.y, v2.z | * | j | = | 0 |
		///    |   0 ,   0 ,   0  |   | k |   | 0 |
		/// 
		///    | v1.x,          v1.y,                 v1.z          |   | i |   | 0 |
		/// => |  0  , v2.y - v2.x/v1.x*v1.y, v2.z - v2.x/v1.x*v1.z | * | j | = | 0 |
		///    |  0  ,            0 ,                   0           |   | k |   | 0 |
		/// 
		/// => | v1.x,          v1.y,                 v1.z          |   | i |   | 0 |
		///    |  0  , v2.y*v1.x - v2.x*v1.y, v2.z*v1.x - v2.x*v1.z | * | j | = | 0 |
		///    |  0  ,            0 ,                   0           |   | k |   | 0 |
		/// 
		///  j =   v2.z*v1.x - v2.x*v1.z     : (v2.y*v1.x - v2.x*v1.y)*j + (v2.z*v1.x - v2.x*v1.z)*k = 0
		///  k = -(v2.y*v1.x - v2.x*v1.y)
		///  i = -(v1.y*j + v1.z*k)/v1.x
		///    = -(v1.y*v2.z*v1.x - v1.y*v2.x*v1.z - v1.z*v2.y*v1.x + v1.z*v2.x*v1.y)/v1.x
		///    = -(v1.y*v2.z - v1.y*v2.x*v1.z/v1.x - v1.z*v2.y + v1.z*v2.x*v1.y/v1.x)
		///    = -(v1.y*v2.z - v1.z*v2.y)
		/// 
		///  j = -(v2.z*v1.x - v2.x*v1.z)    : (v2.y*v1.x - v2.x*v1.y)*j + (v2.z*v1.x - v2.x*v1.z)*k = 0
		///  k =   v2.y*v1.x - v2.x*v1.y
		///  i = ...
		/// 
		///@thory
		///            |  i  ,  j  ,  k   |
		/// solve det( | v1.x, v1.y, v1.z | ) = +-1    :error, v1 and v2 are not necessarily orthogonal .
		///            | v2.x, v2.y, v2.z |
		/// 
		/// magnitude can be any nonzero
		///            |  i    j    k   |
		/// solve det( | v1.x v1.y v1.z | ) = +-?
		///            | v2.x v2.y v2.z |
		///   i*det(minor(0,0))         - j*det(minor(0,1))         + k*det(minor(0,2))         = +-?
		///   i*(v1.y*v2.z - v1.z*v2.y) - j*(v1.x*v2.z - v1.z*v2.x) + k*(v1.x*v2.y - v1.y*v2.x) = +-?
		///   i*(v1.y*v2.z - v1.z*v2.y) + j*(v1.z*v2.x - v1.x*v2.z) + k*(v1.x*v2.y - v1.y*v2.x) = +-?
		/// 
		///    i = v1.y*v2.z - v1.z*v2.y
		///     j = v1.z*v2.x - v1.x*v2.z  for positive determinant
		///      k = v1.x*v2.y - v1.y*v2.x
		/// 
		///    i = -(v1.y*v2.z - v1.z*v2.y)
		///     j = -(v1.z*v2.x - v1.x*v2.z)  for negative determinant
		///      k = -(v1.x*v2.y - v1.y*v2.x)
		/// 
		///@summary
		/// We cannot say which is good, but we like positive.
		/// So usually cross product between 'v1' and 'v2' is not meaning orthogonal bewteen 'v1' and 'v2'
		///                                                is meaning positive orthogonal between 'v1' and 'v2'.
		return { 
			v1[1]*v2[2] - v1[2]*v2[1], 
			v1[2]*v2[0] - v1[0]*v2[2], 
			v1[0]*v2[1] - v1[1]*v2[0] 
		};
	}

	template<typename T, size_t M>
	vector<T,M> cross(const vector<T,M> &v1, const vector<T,M> &v2, const vector<T,M> &v3) {
		///@diagram
		/// |   i ,     j ,    k ,   u  |
		/// | v1.x,   v1.y,  v1.z, v1.w |
		/// | v2.x,   v2.y,  v2.z, v2.w | = i*1*det(minor(0,0)) + j*-1*det(minor(0,1)) + k*1*det(minor(0,2)) + u*-1*det(minor(0,3)), 1.determinat expand
		/// | v3.x,   v3.y,  v3.z, v3.w |
		///     |      | |    |      |    = vector{ +(v1.y*detC - v1.z*detE + v1.w*detB),
		///     +-detA-+-detB-+-detC-+              -(v1.x*detC - v1.z*detF + v1.w*detD),
		///     |        |    |      |              +(v1.x*detE - v1.y*detF + v1.w*detA),
		///     +---detD-+----+      |              -(v1.x*detB - v1.y*detD + v1.z*detA) }
		///     |        |           |
		///     |        +----detE---+
		///     |                    |
		///     +-----detF-----------+
		T detA = v2[0] * v3[1] - v2[1] * v3[0];
		T detB = v2[1] * v3[2] - v2[2] * v3[1];
		T detC = v2[2] * v3[3] - v2[3] * v3[2];
		T detD = v2[0] * v3[2] - v2[2] * v3[0];
		T detE = v2[1] * v3[3] - v2[3] * v3[1];
		T detF = v2[0] * v3[3] - v2[3] * v3[0];
		return {
				v1[1]*detC - v1[2]*detE + v1[3]*detB,
			-(v1[0]*detC - v1[2]*detF + v1[3]*detD),
				v1[0]*detE - v1[1]*detF + v1[3]*detA,
			-(v1[0]*detB - v1[1]*detD + v1[2]*detA) 
		};
	}


	/// op(A) = value.
	template<typename T, size_t M, size_t N>
	void matset(char op, matrix<T,M,N>& A, T value) {
		if (op == 'd' || op == 'D') {
			for (size_t i = 0; i != A.rows(); ++i) {
				if (i >= A.columns())
					break;
				A.at(i,i) = value;
			}
		} else if (op == 'u' || op == 'U') {
			for (size_t i = 0; i != A.rows(); ++i) {
				if (i >= A.columns())
					break;
				unroll_16x_4x_for(
					epconj(size_t n = A.columns() - i; T* aij = A.data()+A.rowstep()*i + i),
					epconj(n >= 16), epconj(n-=16, aij+=16),
					epconj(n >= 4), epconj(n-=4, aij+=4),
					epconj(n != 0), epconj(--n, ++aij),
					aij[i] = value
				);
			}
		} else if (op == 'l' || op == 'L') {
			for (size_t i = 0; i != A.rows(); ++i) {
				unroll_16x_4x_for(
					epconj(size_t n = std::min(i+1,A.columns()); T * aij = A.data()+A.rowstep()*i),
					epconj(n >= 16), epconj(n-=16, aij+=16),
					epconj(n >= 4), epconj(n-=4, aij+=4),
					epconj(n != 0), epconj(--n, ++aij),
					aij[i] = value
				);
			}
		} else {
			unroll_16x_4x_for(
				epconj(size_t n = A.size(); T* ai = A.data()),
				epconj(n >= 16), epconj(n-=16, ai+=16),
				epconj(n >= 4), epconj(n-=4, ai+=4),
				epconj(n != 0), epconj(--n, ++ai),
				ai[i] = value
			);
		}
	}

	/// B = alpha*op(A).
	template<typename T>
	void matcopy(char op, const T& alpha, const matrix<T>& A, matrix<T>& B) {
		if (op == 't' || op == 'T') {
			if (B.rows() != A.columns() || B.columns() != A.rows())
				matrix_alloc(B, A.columns(), A.rows());
			for (size_t i = 0; i != A.rows(); ++i)
				for (size_t j = 0; j != A.columns(); ++j)
					B.at(j,i) = alpha*A.at(i,j);
		} else {
			if (B.rows() != B.rows() || B.columns() != A.columns())
				matrix_alloc(B, A.columns(), A.rows());
			for (size_t i = 0; i != A.rows(); ++i)
				for (size_t j = 0; j != A.columns(); ++j)
					B.at(i,j) = alpha*A.at(i,j);
		}
	}

	/// B = alpha*op(A), inplace.
	template<typename T>
	void matcopy(char op, const T& alpha, matrix<T>& AB) {
		abort();
	}

	/// B[i][j] = A[j][i].
	template<typename T, size_t M, size_t N>
	matrix<T,N,M> geT(const matrix<T,M,N>& A) {
		matrix<T,N,M> tA;
		matrix_alloc(tA, A.columns(), A.rows())
		for (size_t i = 0; i != A.rows(); ++i) {
			for (size_t j = 0; j != A.columns(); ++j) {
				tA.at(j,i) = A.at(i,j);
			}
		}

		return std::move(tA);
	}

	/// B[i][j] = A[j][i].
	template<typename T, size_t M> inline
	matrix<T,1,M>& geT(matrix<T,M,1>& A) { return reinterpret_cast<matrix<T,1,M>&>(A); }
	
	/// B[i][j] = A[j][i].
	template<typename T, size_t M> inline
	const matrix<T,1,M>& geT(const matrix<T,M,1>& A) { return reinterpret_cast<const matrix<T,1,M>&>(A); }


	/// Computes the LU factorization of a general M-by-N matrix.
	///		A = permute(ipiv,L,U)
	/// 
	///@param A is M-by-N matrix.
	///	Will overwritten by L and U. The unit diagonal elements of L are not stored.
	/// 
	///@param ipiv is permutation.
	///	Contains the pivot indices; for 1<=i<=min(M,N), row i was interchanged with row ipiv[i].
	/// 
	///@return a value "info".
	///	If info = 0, the execution is successful.
	///	If info =-i, parameter i had an illegal value.
	///	If info = i, U.at(i,i) is 0. The factorization has been completed, 
	///	 but U is exactly singular. Division by 0 will occur if you use the factor U for solving a system of linear equations.
	/// 
	///@reference Intel® oneAPI Math Kernel Library
	template<typename scalar, size_t M, size_t N>
	size_t getrf(matrix<scalar,M,N>& A, size_t* ipiv) {
		size_t count = std::min(A.rows(), A.columns());
		for (size_t k = 0; k != count; ++k) {
			// Find max pivot, max pivot may be optimize some problems.
			size_t k2 = k;
			for (size_t i = k2; i != A.rows(); ++i)
				if (abs(A.at(i,k)) > abs(A.at(k2,k)))
					k2 = i;

			// Break if singular.
			if (A.at(k2,k) == 0) {
				return k;
				///@note
				/// abs(A[k2][k]) >= abs(A[k][k]), 
				/// A[k2][k] == 0, then A[k][k] == 0.
			}

			// Interchange Row, between current(k) and max pivot(k2).
			ipiv[k] = k2;
			if (k2 != k) {
				for (size_t j = k; j != A.columns(); ++j) {
					std::swap(A.at(k2,j), A.at(k,j));
				}
			}

			// Eliminate rows, those row lower than pivot (>k).
			for (size_t i = k + 1; i != A.rows(); ++i) {
				scalar neg_multiplier = -A.at(i,k) / A.at(k,k);
				for (size_t j = k; j != A.columns(); ++j) {
					A.at(i,j) += A.at(k,j) * neg_multiplier;
				}
				// And saved multiplier into L.
				A.at(i,k) = -neg_multiplier;
			}
		}

		return 0;
	}

	/// Solve permute(ipiv)*upper(A)*lower(A) * x = b.
	template<typename T, size_t M, size_t N, size_t J>
	void getrs(const matrix<T,M,N>& A, const size_t* ipiv, matrix<T,M,J>& b) {
		size_t count = std::min(A.rows(), A.columns());
		if (count == 0) {
			return;
		}

		// Solve L*x = b.
		for (size_t k = 0; k != count; ++k) {
			size_t k2 = ipiv[k];
			if (k2 != k) {
				for (size_t j = 0; j != b.columns(); ++j) {
					std::swap(b.at(k2,j), b.at(k,j));
				}
			}
			for(size_t i = k+1; i != b.rows(); ++i) {
				T neg_multiplier = -A.at(i,k)/* /1.0 */;
				for (size_t j = 0; j != b.columns(); ++j) {
					b.at(i,j) += b.at(k,j) * neg_multiplier;
				}
			}
		}

		// Solve U*x = b.
		for (size_t k = count-1; k != size_t(-1); --k) {
			T divisor = A.at(k,k);
			for (size_t j = 0; j != b.columns(); ++j) {
				b.at(k,j) /= divisor;
			}
			for (size_t i = k-1; i != size_t(-1); --i) {
				T neg_multiplier = -A.at(i,k)/* /1.0 */;
				for (size_t j = 0; j != b.columns(); ++j) {
					b.at(i,j) += b.at(k,j) * neg_multiplier;
				}
			}
		}
	}

	/// Solve A * x = b.
	template<typename T, size_t M, size_t N, size_t J> inline
	size_t gesv(matrix<T,M,N>& A, size_t* ipiv, matrix<T,M,J>& b) {
		size_t info = getrf(A, ipiv);
		if (info == 0) {
			getrs(A, ipiv, b);
		}

		return info;
	}

	template<typename T, size_t N>
	matrix<T,N,N> inv(const matrix<T,N,N> &A) {
		// Construct [A * Ainv = Identity].
		matrix<T,N,N> Acopy;
		matrix_alloc(Acopy,A.rows(),A.columns());
		for (size_t i = 0; i != Acopy.size(); ++i) {
			Acopy[i] = A[i];
		}
		matrix<T,N,N> id;
		matrix_alloc(id,A.rows(),A.columns());
		for (size_t i = 0; i != id.rows(); ++i) {
			for (size_t j = 0; j != id.columns(); ++j)
				id.at(i,j) = 0;
			id.at(i,i) = 1;
		}
		matrix<size_t,N,1> ipiv;
		matrix_alloc(ipiv,A.rows(),1);

		// Solve [A * Ainv = Identity].
		if ( gesv(Acopy, ipiv.data(), id) == 0 ) {
			return id;
		} else {
			return matrix<T,N,N>{};
		}
	}

	template<typename T, size_t N>
	T det(const matrix<T,N,N> &A) {
		matrix<T,N,N> LU;
		matrix_alloc(LU,A.rows(),A.columns());
		for (size_t i = 0; i != LU.size(); ++i) {
			LU[i] = A[i];
		}
		matrix<size_t,N,1> ipiv;
		matrix_alloc(ipiv,A.rows(),1);

		if ( getrf(LU, ipiv.data()) == 0 ) {
			bool sign = 0;
			for (size_t k = 0; k != LU.rows(); ++k)
				if (ipiv[k] != k)
					sign ^= 1;
			T _det = static_cast<T>(sign?-1: 1);
			for (size_t k = 0; k != LU.rows(); ++k)
				_det *= LU.at(k,k);
			return _det;
		} else {
			return 0;
		}
	}


	/// C = A*B.
	template<typename T, size_t M, size_t N, size_t K>
	matrix<T,M,N> gemm(const matrix<T,M,K>& A, const matrix<T,K,N>& B) {
		///@theory 
		/// If 'A' is an m*n matrix, and if 'B' is an n*p matrix with columns b1,...,bp,
		/// then the product A*B is the m*p matrix whose columns are A*b1,...,A*bp. That is,
		///         A*B = A*[b1 b2 ... bp] = [A*b1 A*b2 ... A*bp].
		/// verify 
		///         [row(A,1)*column(B,1) row(A,1)*column(B,2) ... row(A,1)*column(B,p)]
		///         [row(A,2)*column(B,1) row(A,2)*column(B,2) ... row(A,2)*column(B,p)]
		///         [ ...                                      ...                     ]
		///         [row(A,m)*column(B,1) row(A,m)*column(B,2) ... row(A,m)*column(B,p)]
		///
		///         [[row(A,1)]             [row(A,1)]             ... [row(A,1)]            ]
		///       = [[row(A,2)]*column(B,1) [row(A,2)]*column(B,2) ... [row(A,2)]*column(B,p)]
		///         [[ ...    ]             [ ...    ]                        ...            ]
		///         [[row(A,m)]             [row(A,m)]             ... [row(A,m)]            ]
		///
		///       = [A*b1 A*b2 ... A*bp].
		///
		/// Each column of A*B is a linear combination of the columns of A using weights
		/// from the corresponding column of B.
		///                                                       MatrixMultiplication in 
		///                                        <<Linear Algebra and Its Application>> 
		///
		/// Opposite, each row of A*B is a linear combination of the rows of B using weights
		/// from the corresponding row of A, That is,
		///                                                  [t(B*a1)]
		///         A*B = t(B*A) = t([B*a1 B*a2 ... B*am]) = [t(B*a2)], t is transpose,
		///                                                  [ ...   ]
		///                                                  [t(B*am)]
		///         (linear combination is columns of A using weights from column b, 
		///          transposed linear combination is rows of A using weights from row b).
		///
		/// This is a method of partition, useful for no-parallel case.
		/// 
		///@figure
		///      [0] [1] [.] [N]  [----row[0]----]     [0] [1] [.] [N]
		/// dot(                , [----row[1]----] ) = 
		///                       [----row[.]----]
		///                       [----row[N]----]
		matrix<T,M,N> C;
		matrix_alloc(C, A.rows(), B.columns());
		for (size_t i = 0; i != C.rows(); ++i) {
			T ail = A.at(i,0);
			unroll_16x_4x_for(
				epconj(size_t n = C.columns(); T* ci = C.data() + C.rowstep()*i; const T* bi = B.data()),
				epconj(n >= 16), epconj(n-=16, ci+=16, bi+=16),
				epconj(n >= 4), epconj(n-=4, ci+=4, bi+=4),
				epconj(n != 0), epconj(--n, ++ci, ++bi),
				ci[i] = bi[i]*ail;
			);
			for (size_t l = 1; l != B.rows(); ++l) {
				ail = A.at(i,l);
				unroll_16x_4x_for(
					epconj(size_t n = C.columns(); T* ci = C.data()+C.rowstep()*i; const T* bi = B.data()+B.rowstep()*l),
					epconj(n >= 16), epconj(n-=16, ci+=16, bi+=16),
					epconj(n >= 4), epconj(n-=4, ci+=4, bi+=4),
					epconj(n != 0), epconj(--n, ++ci, ++bi),
					ci[i] += bi[i]*ail;
				);
			}
		}

		return std::move(C);
	}

	/// C = alpha*A*B + beta*C.
	template<typename T, size_t M, size_t N, size_t K>
	void gemm(const T& alpha, const matrix<T,M,K>& A, const matrix<T,K,N>& B, 
		const T& beta, matrix<T,M,N>& C) {
		for (size_t i = 0; i != C.rows(); ++i) {
			unroll_16x_4x_for(
				epconj(size_t n = C.columns(); T* ci = C.data()+C.rowstep()*i),
				epconj(n >= 16), epconj(n-=16, ci+=16),
				epconj(n >= 4), epconj(n-=4, ci+=4),
				epconj(n != 0), epconj(--n, ++ci),
				ci[i] *= beta;
			);
			for (size_t l = 0; l != B.rows(); ++l) {
				T ail = A.at(i,l);
				unroll_16x_4x_for(
					epconj(size_t n = C.columns(); T* ci = C.data()+C.rowstep()*i; const T* bi = B.data()+B.rowstep()*l),
					epconj(n >= 16), epconj(n-=16, ci+=16, bi+=16),
					epconj(n >= 4), epconj(n-=4, ci+=4, bi+=4),
					epconj(n != 0), epconj(--n, ++ci, ++bi),
					ci[i] += alpha*bi[i]*ail;
				);
			}
		}
	}

	/// C = alpha*op(A)*op(B) + beta*C.
	template<typename T>
	void gemm(char opA, char opB, const T& alpha, const matrix<T>& A, const matrix<T>& B, 
		const T& beta, matrix<T>& C) {
		abort();
	}


	template<typename OutMatrix, typename InMatrix>
	OutMatrix matrix_cast(const InMatrix& src) {
		// wait improve...
		using OutScalar = std::remove_cvref_t<decltype(OutMatrix{}.at(0,0))>;
		OutMatrix dst;
		size_t common_rows = std::min(dst.rows(), src.rows());
		size_t common_columns = std::min(dst.columns(), src.columns());
		for (size_t i = 0; i != common_rows; ++i) {
			for (size_t j = 0; j != common_columns; ++j) {
				dst.at(i,j) = static_cast<OutScalar>(src.at(i,j));
			}
			for (size_t j = common_columns; j != dst.columns(); ++j) {
				dst.at(i,j) = 0;
			}
		}
		for (size_t i = common_rows; i != dst.rows(); ++i) {
			for (size_t j = 0; j != dst.columns(); ++j) {
				dst.at(i,j) = 0;
			}
		}
		return dst;
	}
}

///@architecture 
/// 
/// I have a new idea:
///   data is shared. and can be switch various operation, may be runtime.
///
/// example: We have a subcolumn DoubleArray, which is discontinuous. So, 
///   cannot get the correct result by operations in this file, should use opreations in "Mutility.hpp".
///
/// Can operations be shared, while switching data?
///   Maybe some use.
///
/// Can operations and data be switch or composition?
///   It's a little complicated, so it's not used.
///
/// Final, this idea is still a little complicated.
///   May be not much use. 
///   Because we can do this implicitly, use address or darray<..>::reref(..).
///
///@algorithm 
///
/// Use algorithms of ATLAS style. These are results of 100000000*2 times
/// operator+=(Dynamic4x4, Dynamic4x4).
///   1. ATLAS style, axpy: 1315[ms]
///   2. single forloop, A[i] += B[j]: 2866[ms]
///   3. double forloop, A.at(i,j) += B.at(i,j): 3100[ms]
///   4. std::transform, iterator based of A[i] += B[i] : 2200[ms]
/// we don't want to rely too much on compiler optimization(except static case).
/// 
/// Use algorithm of OpenMP style. OpenMP very fast, and just in line with my goal.
/// (Note: my goal is that, there is no optimization, just axioms and theorems.
///	 Because there are countless kinds of optimization, axioms and theorems are
///  unique. If we lose this unique, we may lose our way in countless kinds of 
///  optimization. Although this can be commented, rather than programed, the
///  axioms and theorems of the Matrix are basic enough to be used directly.)
/// 
/// Final we use algorithm of "unroll(16 times & 4 times & 3 times & ..) with simd".
/// (Note: 16 times, because matrix4x4 is very useful. 4 times 3 times, because small vectors are very useful.)
///
///@diagram
/// 
/// +-----------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
/// | matrix<*,M,N> /                                                                                                                                                                           |
/// +---------------+                                                                                                                                                                           |
/// |                                                                                                                                                                                           |
/// | +-----------------------------------+  +-----------------------------------------+  +--------------------------------------------+  +---------------------------------------------------+ |
/// | | matrix<scalar_type,rows,columns>  |  | matrix<scalar_type,rows,dynamic_columns>|  | matrix<scalar_type,dynamic_rows,columns>   |  | matrix<scalar_type,dynamic_rows,dynamic_columns>  | |
/// | +-----------------------------------+  +-----------------------------------------+  +--------------------------------------------+  +---------------------------------------------------+ |
/// | | +size(): size_t                   |  | +size(): size_t                         |  | +size(): size_t                            |  | +size(): size_t                                   | |
/// | | +rows(): size_t                   |  | +rows(): size_t                         |  | +rows(): size_t                            |  | +rows(): size_t                                   | |
/// | | +columns(): size_t                |  | +columns(): size_t                      |  | +columns(): size_t                         |  | +columns(): size_t                                | |
/// | | +rowstep(): size_t                |  | +rowstep(): size_t                      |  | +rowstep(): size_t                         |  | +rowstep(): size_t                                | |
/// | | +empty(): bool                    |  | +empty(): bool                          |  | +empty(): bool                             |  | +empty(): bool                                    | |
/// | | +data(): pointer                  |  | +data(): pointer                        |  | +data(): pointer                           |  | +data(): pointer                                  | |
/// | | +begin(): iterator                |  | +begin(): iterator                      |  | +begin(): iterator                         |  | +begin(): iterator                                | |
/// | | +end(): iterator                  |  | +end(): iterator                        |  | +end(): iterator                           |  | +end(): iterator                                  | |
/// | | +[](idx:size_t): reference        |  | +[](idx:size_t): reference              |  | +[](idx:size_t): reference                 |  | +[](i:size_t): reference                          | |
/// | | +at(i:size_t,j:size_t): reference |  | +at(i:size_t,j:size_t): reference       |  | +at(i:size_t,j:size_t): reference          |  | +at(i:size_t): reference                          | |
/// | | +row(i:size_t): row_reference     |  | +row(i:size_t): row_reference           |  | +row(i:size_t): row_reference              |  | +row(i:size_t): row_reference                     | |
/// | +-----------------------------------+  +-----------------------------------------+  +--------------------------------------------+  +---------------------------------------------------+ |
/// |                                        | +recolumn(columns:size_t): void         |  | +rerow(rows:size_t): void                  |  | +resize(rows:size_t, columns:size_t): void        | |
/// |                                        | +reref(data:pointer, rows:size_t): void |  | +reref(data:pointer, columns:size_t): void |  | +reref(data:pointer, rows:size_t, columns:size_t) | |
/// |                                        +-----------------------------------------+  +--------------------------------------------+  +---------------------------------------------------+ |
/// |                                                                                                                                                                                           |
/// +---------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------+
///                                         |
///                                         |
///    +------------------------------------+----------------------------------------+
///    | Algorithm                                                                   |
///    +-----------------------------------------------------------------------------+
///    | <<Matrix-Vector Operation of Compare>>                                      |
///    | operator==(Xc:matrix<*,M,N>, Yc:matrix<*,M,N>): bool                        |
///    | operator!=(Xc:matrix<*,M,N>, Yc:matrix<*,M,N>): bool                        |
///    +-----------------------------------------------------------------------------+
///    | <<Matrix-Vector Operation Unary>>                                           |
///    | operator-(Xc:matrix<*,M,N>): matrix<*,M,N>                                  |
///    +-----------------------------------------------------------------------------+
///    | <<Matrix-Vector Operation with Scalar>>                                     |
///    | operator+(Xc:matrix<*,M,N>, alpha:scalar): matrix<*,M,N>                    |
///    | operator-(Xc:matrix<*,M,N>, alpha:scalar): matrix<*,M,N>                    |
///    | operator*(Xc:matrix<*,M,N>, alpha:scalar): matrix<*,M,N>                    |
///    | operator/(Xc:matrix<*,M,N>, alpha:scalar): matrix<*,M,N>                    |
///    +-----------------------------------------------------------------------------+
///    | <<Matrix Operation>>                                                        |
///    | operator+(Ac:matrix<*,M,N>, Bc:matrix<*,M,N>): matrix<*,M,N>                |
///    | operator-(Ac:matrix<*,M,N>, Bc:matrix<*,M,N>): matrix<*,M,N>                |
///    | transpose(Ac:matrix<*,M,N>): matrix<*,N,M>                                  |
///    | multiply(Ac:matrix<*,M,K>, Bc:matrix<*,K,N>): matrix<*,M,N>                 |
///    | inverse(src:matrix<*,N,N>): matrix<*,N,N>                                   |
///    | determinant(Ac:matrix<*,N,N>): scalar                                       |
///    |                                ...                                          |
///    +-----------------------------------------------------------------------------+
///    | <<Vector Operation>>                                                        |
///    | operator*(Xc:matrix<*,N,0>, Yc:matrix<*,N,0>): matrix<*,N,0>                |
///    | operator/(Xc:matrix<*,N,0>, Yc:matrix<*,N,0>): matrix<*,N,0>                |
///    | dot(Xc:matrix<*,N,0>, Yc:matrix<*,N,0>): scalar                             |
///    | length(Xc:matrix<*,N,0>): scalar                                            |
///    | normalize(Xc:matrix<*,N,0>): matrix<*,N,0>                                  |
///    | cross(Xc:matrix<*,N,0>, Yc:matrix<*,N,0>): matrix<*,N,0>                    |
///    | cross(Xc:matrix<*,N,0>, Yc:matrix<*,N,0>, Zc:matrix<*,N,0>): matrix<*,N,0>  |
///    +-----------------------------------------------------------------------------+
///    | <<Blas>>                                                                    |
///    |                                ...                                          |
///    +-----------------------------------------------------------------------------+
///    | <<Lapack>>                                                                  |
///    |                                                                             |
///    |
/// 

/* @test
	using namespace::calculation;

	try {
		darray<float,2,2> A;
		darray<float,2,1> x;
		A = {
			1, -3,
			-2, 4
		};
		x = {  
			5,
			3
		};

		std::cout << "1.\n"
			<<"(A*x)t={"<<transpose((A*x))<<"}\n"
			<<"(x)t*(A)t={"<<transpose(x)*transpose(A)<<"}\n"
			<<"x*(x)t={"<<x*transpose(x)<<"}\n"
			<<"(x)t*x={"<<transpose(x)*x<<"}\n"
			<<"(A)t*(x)t is undefined, because (A)t is 2x2 (x)t is 1x2, left columns not equal right rows.\n";
	} catch (const std::exception& e) {
		std::cout << e.what() << std::endl;
	}

	try {
		darray<float, 2, 3> A = {
			2, 0, -1,
			4, -5, 2
		};
		darray<float, 2, 3> B = {
			7, -5, 1,
			1, -4, -3
		};
		darray<float, 2, 2> C = {
			1, 2,
			-2, 1
		};
		darray<float, 2, 2> D = {
			3, 5,
			-1, 4
		};
		darray<float, 2, 1> E = {
			-5,
			3
		};
		
		std::cout << "1.\n"
			<< "-2*A = {"<<(-2.0f*A)<<"}\n"
			<< "B-2*A = {"<<(B-2.0f*A)<<"}\n"
			<< "A*C = undefined\n"
			<< "C*D = "<<(C*D)<<"\n";
		std::cout << "2.\n"
			<< "A+2*B = {"<<(A+2.0f*B)<<"}\n"
			<< "3*C-E = undefined\n"
			<< "C*B = {"<<(C*B)<<"}\n"
			<< "E*B = undefined\n";
	} catch (const std::exception& e) {
		std::cout << e.what() << std::endl;
	}

	try {
		darray<float> A;
		A.resize(2, 2);
		A = {
			4, -1,
			5, -2
		};

		darray<float> I;
		I.resize(A.rows(), A.columns());
		for (size_t i = 0; i != I.size(); ++i) {
			I[i] = 0;
		}
		for (size_t k = 0; k != std::min(A.rows(), A.columns()); ++k) {
			I.at(k,k) = 1;
		}
		
		std::cout << "3.\n"
			<< "3*I-A = {"<<(3.0f*I-A)<<"}\n"
			<< "(3*I)*A = {"<<((3.0f*I)*A)<<"}\n";
	} catch (const std::exception& e) {
		std::cout << e.what() << std::endl;
	}
	

	try {
		darray<float,2,2> A = {
			2, -3,
			-4, 6
		};
		darray<float,2,2> B = {
			8, 4,
			5, 5
		};
		darray<float,2,2> C = { 
			5, -2,
			3, 1
		};

		std::cout << "10.\n"
			<<"A*B = {"<<(A*B)<<"}\n"
			<<"A*C = {"<<(A*C)<<"}\n"
			<<"but B != C\n";
	} catch (const std::exception& e) {
		std::cout << e.what() << std::endl;
	}
*/

/**
 * | w0  w0  w0  w0  w0  w0  w0  w0  |   | x0 |
 * | w0  w1  w2  w3  w4  w5  w6  w7  |   | x1 |
 * | w0  w2  w4  w6  w8  w10 w12 w14 |   | x2 |
 * | w0  w3  w6  w9  ...             | * | x3 |
 * | w0  w4  w8  w12                 |   | x4 |
 * | w0  w5  w10 w15                 |   | x5 |
 * | w0  w6  w12 w18 ...             |   | x6 |
 * | w0  w7  w14 w21 ...             |   | x7 |
 *
 *   w0*x0 + w0*x1 + w0*x2 + w0*x3 + w0*x4 + w0*x5 + w0*x6 + w0*x7
 *   w0*x0 + w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5 + w6*x6 + w7*x7
 *   w0*x0 + w2*x1 + w4*x2 + w6*x3 + w8*x4 + w10*x5 + w12*x6 + w14*x7
 * = w0*x0 + w3*x1 + w6*x2 + w9*x3 + w12*x4 + w15*x5 + w18*x6 + w21*x7
 *   w0*x0 + w4*x1 + w8*x2 + w12*x3 + w16*x4 + w20*x5 + w24*x6 + w28*x7
 *   w0*x0 + w5*x1 + w10*x2 + w15*x3 + w20*x4 + w25*x5 + w30*x6 + w35*x7
 *   w0*x0 + w6*x1 + w12*x2 + w18*x3 + w24*x4 + w30*x5 + w36*x6 + w42*x7
 *   w0*x0 + w7*x1 + w14*x2 + w21*x3 + w28*x4 + w35*x5 + w42*x6 + w49*x7
 *
 * rearrange the order of the terms so that the evennumbered terms come first
 *   w0*x0 + w0*x2 + w0*x4 + w0*x6 + w0*(w0*x1 + w0*x3 + w0*x5 + w0*x7)
 *   w0*x0 + w2*x2 + w4*x4 + w6*x6 + w1*(w0*x1 + w2*x3 + w4*x5 + w6*x7)
 *   w0*x0 + w4*x2 + w8*x4 + w12*x6 + w2*(w0*x1 + w4*x3 + w8*x5 + w12*x7)
 * = w0*x0 + w6*x2 + w12*x4 + w18*x6 + w3*(w0*x1 + w6*x3 + w12*x5 + w18*x7)
 *   w0*x0 + w8*x2 + w16*x4 + w24*x6 + w4*(w0*x1 + w8*x3 + w16*x5 + w24*x7)
 *   w0*x0 + w10*x2 + w20*x4 + w30*x6 + w5*(w0*x1 + w10*x3 + w20*x5 + w30*x7)
 *   w0*x0 + w12*x2 + w24*x4 + w36*x6 + w6*(w0*x1 + w12*x3 + w24*x5 + w36*x7)
 *   w0*x0 + w14*x2 + w28*x4 + w42*x6 + w7*(w0*x1 + w14*x3 + w28*x5 + w42*x7)
 *
 * w8 = 1
 *   w0*x0 + w0*x2 + w0*x4 + w0*x6 + w0*(w0*x1 + w0*x3 + w0*x5 + w0*x7)
 *   w0*x0 + w2*x2 + w4*x4 + w6*x6 + w1*(w0*x1 + w2*x3 + w4*x5 + w6*x7)
 *   w0*x0 + w4*x2 + w8*x4 + w12*x6 + w2*(w0*x1 + w4*x3 + w8*x5 + w12*x7)
 * = w0*x0 + w6*x2 + w12*x4 + w18*x6 + w3*(w0*x1 + w6*x3 + w12*x5 + w18*x7)
 *   w0*x0 + w0*x2 + w0*x4 + w0*x6 + w4*(w0*x1 + w0*x3 + w0*x5 + w0*x7)
 *   w0*x0 + w2*x2 + w4*x4 + w6*x6 + w5*(w0*x1 + w2*x3 + w4*x5 + w6*x7)
 *   w0*x0 + w4*x2 + w8*x4 + w12*x6 + w6*(w0*x1 + w4*x3 + w8*x5 + w12*x7)
 *   w0*x0 + w6*x2 + w12*x4 + w18*x6 + w7*(w0*x1 + w6*x3 + w12*x5 + w18*x7)
 *
 * u0 = w0*x0 + w0*x2 + w0*x4 + w0*x6, v0 = w0*x1 + w0*x3 + w0*x5 + w0*x7
 * u1 = w0*x0 + w2*x2 + w4*x4 + w6*x6, v1 = w0*x1 + w2*x3 + w4*x5 + w6*x7
 * u2 = w0*x0 + w4*x2 + w8*x4 + w12*x6, v2 = w0*x1 + w4*x3 + w8*x5 + w12*x7
 * u3 = w0*x0 + w6*x2 + w12*x4 + w18*x6, v3 = w0*x1 + w6*x3 + w12*x5 + w18*x7
 * u = M4*[x0,x2,x4,x6], v = M4*[x1,x3,x5,x7]
 * = [u + [w^0,w^1,w^2,w^3] X v]
 *   [u + [w^4,w^5,w^6,w^7] X v]
*/

//math::multi_array<float> U;
//math::multi_array<float> T;
//math::multi_array<float> L;
//U.reshape(rand()%6+1,rand()%4+1);
//T.reshape(U.rows(), U.columns());
//L.reshape(U.columns(), U.rows());
///*math::multi_array<double,6,6> U;
//math::multi_array<double,6,6> T;
//math::multi_array<double,6,6> L;*/
//for (size_t i = 0; i != U.size(); ++i) {
//	U[i] = (float)rand();
//	T[i] = (float)rand()/ float(RAND_MAX);
//	L[i] = (float)rand()/ float(RAND_MAX);
//}
//std::cout << "U:{\n" << U << "\n}" << std::endl;
//std::cout << "L:{\n" << L << "\n}" << std::endl;
//std::cout << "U+T = {\n" << U + T << "\n}" << std::endl;
//std::cout << "U-T = {\n" << U + T << "\n}" << std::endl;
//std::cout << "U*L = {\n" << U * L << "\n}" << std::endl;
//if (U.rows() == L.columns()) {
//	std::cout << "inverse(U*L) = {\n" << math::inverse(U * L) << "\n}" << std::endl;
//	std::cout << "inverse(U*L)*(U*L) = {\n" << math::inverse(U*L)*(U*L) << "\n}" << std::endl;
//	std::cout << "(U*L)*inverse(U*L) = {\n" << (U*L)*math::inverse(U*L) << "\n}" << std::endl;
//	//std::cout << "determinant(U*L) = {\n" << math::determinant(U * L) << "\n}" << std::endl;
//}