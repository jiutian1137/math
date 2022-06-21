#pragma once
#include <cassert>
#include <type_traits>
#include <string>
#include <iosfwd>

#define _MATH_SIMD_MULTI_ARRAY_

namespace math {
	template<typename scalar, size_t static_rows, size_t static_columns, typename package>
	class __declspec(align(std::alignment_of_v<package>)) simd_multi_array {
	public:
		using scalar_type = scalar;

		using value_type      = scalar;
		using pointer         = scalar *;
		using const_pointer   = const scalar *;
		using reference       = scalar &;
		using const_reference = const scalar &;
		using iterator        = scalar *;
		using const_iterator  = const scalar *;

		using row_reference = simd_multi_array<scalar, 1, static_columns, package>&;
		using const_row_reference = const simd_multi_array<scalar, 1, static_columns, package>&;

		static constexpr size_t static_package_stride = sizeof(package) / sizeof(scalar);
		static constexpr size_t static_pachage_rows = static_columns == 1 ? 
			( sizeof(scalar)*static_rows / sizeof(package) + ( sizeof(scalar)*static_rows % sizeof(package) ? 1 : 0 ) ) : 
			static_rows;
		static constexpr size_t static_package_columns = static_columns == 1 ? 
			static_columns : 
			( sizeof(scalar)*static_columns / sizeof(package) + ( sizeof(scalar)*static_columns % sizeof(package) ? 1 : 0 ) );
		scalar _My_data[static_package_stride * static_pachage_rows * static_package_columns];

		static constexpr size_t package_stride() { return static_package_stride; }
		static constexpr size_t package_rows() { return static_pachage_rows; }
		static constexpr size_t package_columns() { return static_package_columns; }
		static constexpr size_t package_size() { return static_pachage_rows * static_package_columns; }

		static constexpr size_t rows() { return static_rows; }
		static constexpr size_t columns() { return static_columns; }
		static constexpr size_t rowstep() { return static_columns == 1 ? 1 : static_package_columns*static_package_stride; }
		static constexpr size_t size() { return static_rows * static_columns; }
		static constexpr size_t actual_size() { return sizeof(_My_data)/sizeof(scalar); }
		static constexpr bool empty() { return false; }

		pointer data() { return static_cast<pointer>(_My_data); }
		const_pointer data() const { return static_cast<const_pointer>(_My_data); }

		iterator begin() { return static_cast<iterator>(_My_data); }
		const_iterator begin() const { return static_cast<const_iterator>(_My_data); }
	
		iterator end() { return static_cast<iterator>(_My_data) + size(); }
		const_iterator end() const { return static_cast<const_iterator>(_My_data) + size(); }
 
		reference operator[](size_t i) { return _My_data[i]; }
		const_reference operator[](size_t i) const { return _My_data[i]; }

		reference at(size_t i, size_t j) { return _My_data[i * rowstep() + j]; }
		const_reference at(size_t i, size_t j) const { return _My_data[i * rowstep() + j]; }

		/*row_reference row(size_t i) { return reinterpret_cast<row_reference>(_My_data[i * rowstep()]); }
		const_row_reference row(size_t i) const { return reinterpret_cast<const_row_reference>(_My_data[i * rowstep()]); }*/
	};


	using std::to_string;

	template<typename T, size_t M, size_t N, typename package>
	std::string to_string(const simd_multi_array<T,M,N,package>& matrix, char newelement = ' ', char newline = '\n') {
		std::string str;
		if (matrix.rows() == 1 || matrix.columns() == 1) {/* to vector_string */
			for (size_t i = 0; i != matrix.rows() * matrix.columns(); ++i) {
				str += to_string( matrix[i] );
				if (!(i == matrix.rows() * matrix.columns() - 1)) {
					str += newelement;
				}
			}
		} else {
			for (size_t i = 0; i != matrix.rows(); ++i) {
				for (size_t j = 0; j != matrix.columns(); ++j) {
					str += to_string( matrix.at(i,j) );
					if (!(j == matrix.columns() - 1)) {
						str += newelement;
					}
				}

				if (!(i == matrix.rows() - 1)) {
					str += newline;
				}
			}
		}

		return str;
	}

	template<typename T, size_t M, size_t N, typename package,
		typename _Elem, typename _Traits>
	std::basic_ostream<_Elem,_Traits>& operator<<(std::basic_ostream<_Elem,_Traits>& ostr, const simd_multi_array<T,M,N,package>& matrix) {
		if (matrix.rows() == 1 || matrix.columns() == 1) {/* to vector_string */
			for (size_t i = 0; i != matrix.rows() * matrix.columns(); ++i) {
				ostr << matrix[i];
				if (!(i == matrix.rows() * matrix.columns() - 1)) {
					ostr << _Elem(' ');
				}
			}
		} else {
			for (size_t i = 0; i != matrix.rows(); ++i) {
				for (size_t j = 0; j != matrix.columns(); ++j) {
					ostr << matrix.at(i, j);
					if (!(j == matrix.columns() - 1)) {
						ostr << _Elem(' ');
					}
				}

				if (!(i == matrix.rows() - 1)) {
					ostr << _Elem('\n');
				}
			}
		}
	
		return ostr;
	}
}