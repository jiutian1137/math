#pragma once

#include <cassert>

#include <initializer_list>

#include <string>
#include <iosfwd>


/// Structure of Multi-Dimension Array. 
///@license Free 
///@review 2022-5-22 
///@author LongJiangnan, Jiang1998Nan@outlook.com 
#define _MATH_MULTI_ARRAY_

namespace math {
	///@tparam scalar any.
	///@tparam static_rows rows = (static_rows == 0 ? dynamic : static_rows). 
	///@tparam static_columns columns = (static_columns == 0 ? dynamic : static_columns). 
	///@tparam simdlen case 0: no simd optimize, case 4: operation use simdpX4, ...
	template<typename scalar, size_t static_rows = 0, size_t static_columns = 0, size_t simdlen = 0>
	class multi_array {
	public:
		using scalar_type = scalar;

		using value_type      = scalar;
		using pointer         = scalar *;
		using const_pointer   = const scalar *;
		using reference       = scalar &;
		using const_reference = const scalar &;
		using iterator        = scalar *;
		using const_iterator  = const scalar *;

		scalar _My_data[static_rows * static_columns];

		static constexpr
		size_t          size()        { return static_rows * static_columns; }
		static constexpr
		bool            empty()       { return static_columns == 0; }

		pointer         data()        { return static_cast<pointer>(_My_data); }
		const_pointer   data()  const { return static_cast<const_pointer>(_My_data); }

		iterator        begin()       { return static_cast<iterator>(_My_data); }
		const_iterator  begin() const { return static_cast<const_iterator>(_My_data); }
	
		iterator        end()         { return static_cast<iterator>(_My_data) + size(); }
		const_iterator  end()   const { return static_cast<const_iterator>(_My_data) + size(); }
	
		reference       operator[](size_t i)       { return _My_data[i]; }
		const_reference operator[](size_t i) const { return _My_data[i]; }

		multi_array<scalar,1,0,simdlen> subarr(size_t off, size_t count = -1) {
			assert( off <= size() );
			return std::move(multi_array<scalar,1,0,simdlen>( &_My_data[off], std::min(count, size() - off) ));
		}
		multi_array<const scalar,1,0,simdlen> subarr(size_t off, size_t count = -1) const {
			assert( off <= size() );
			return std::move(multi_array<const scalar,1,0,simdlen>( &_My_data[off], std::min(count, size() - off) ));
		}

	public:
		using row_reference = multi_array<scalar, 1, static_columns, simdlen>&;
		using const_row_reference = const multi_array<scalar, 1, static_columns, simdlen>&;

		static constexpr 
		size_t              rows()    { return static_rows; }
		static constexpr 
		size_t              columns() { return static_columns; }
		static constexpr 
		size_t              rowstep() { return static_columns; }

		reference           at(size_t i, size_t j)       { return _My_data[i * rowstep() + j]; }
		const_reference     at(size_t i, size_t j) const { return _My_data[i * rowstep() + j]; }

		row_reference       row(size_t i)       { return reinterpret_cast<row_reference>(_My_data[i * columns()]); }
		const_row_reference row(size_t i) const { return reinterpret_cast<const_row_reference>(_My_data[i * columns()]); }

		multi_array<scalar,1,0,simdlen> subrow(size_t i, size_t off, size_t count = -1) {
			assert( off <= columns() );
			return std::move(multi_array<scalar,1,0,simdlen>( &_My_data[i * columns() + off], std::min(count, columns() - off) ));
		}
		multi_array<const scalar,1,0,simdlen> subrow(size_t i, size_t off, size_t count = -1) const {
			assert( off <= columns() );
			return std::move(multi_array<const scalar,1,0,simdlen>( &_My_data[i * columns() + off], std::min(count, columns() - off) ));
		}

	public:
		// future...
	};

	template<typename scalar, size_t static_rows, size_t simdlen>
	class multi_array<scalar, static_rows, 0, simdlen> {
	public:
		using scalar_type = scalar;

		using value_type      = scalar;
		using pointer         = scalar *;
		using const_pointer   = const scalar *;
		using reference       = scalar &;
		using const_reference = const scalar &;
		using iterator        = scalar *;
		using const_iterator  = const scalar *;

		scalar *_My_data = nullptr;
		size_t _My_columns = 0;
		bool _I_am_ref = false;

		size_t          size()  const { return static_rows * _My_columns; }
		bool            empty() const { return _My_columns == 0; }
		bool            isref() const { return _I_am_ref; }

		pointer         data()        { return static_cast<pointer>(_My_data); }
		const_pointer   data()  const { return static_cast<const_pointer>(_My_data); }

		iterator        begin()       { return static_cast<iterator>(_My_data); }
		const_iterator  begin() const { return static_cast<const_iterator>(_My_data); }
	
		iterator        end()         { return static_cast<iterator>(_My_data) + size(); }
		const_iterator  end()   const { return static_cast<const_iterator>(_My_data) + size(); }
	
		reference       operator[](size_t i)       { return _My_data[i]; }
		const_reference operator[](size_t i) const { return _My_data[i]; }

		multi_array<scalar,1,0,simdlen> subarr(size_t off, size_t count = -1) {
			assert( off <= size() );
			return std::move(multi_array<scalar,1,0,simdlen>( &_My_data[off], std::min(count, size() - off) ));
		}
		multi_array<const scalar,1,0,simdlen> subarr(size_t off, size_t count = -1) const {
			assert( off <= size() );
			return std::move(multi_array<const scalar,1,0,simdlen>( &_My_data[off], std::min(count, size() - off) ));
		}

	public:
		using row_reference = multi_array<scalar, 1, 0, simdlen>;
		using const_row_reference = multi_array<const scalar, 1, 0, simdlen>;

		static constexpr 
		size_t              rows()          { return static_rows; }
		size_t              columns() const { return _My_columns; }
		size_t              rowstep() const { return _My_columns; }

		reference           at(size_t i, size_t j)       { return _My_data[i * rowstep() + j]; }
		const_reference     at(size_t i, size_t j) const { return _My_data[i * rowstep() + j]; }
	
		row_reference       row(size_t i)       { return std::move(row_reference( &_My_data[i * rowstep()], columns() )); }
		const_row_reference row(size_t i) const { return std::move(const_row_reference( &_My_data[i * rowstep()], columns() )); }

		multi_array<scalar,1,0,simdlen> subrow(size_t i, size_t off, size_t count = -1) {
			assert( off <= columns() );
			return std::move(multi_array<scalar,1,0,simdlen>( &_My_data[i * columns() + off], std::min(count, columns() - off) ));
		}
		multi_array<const scalar,1,0,simdlen> subrow(size_t i, size_t off, size_t count = -1) const {
			assert( off <= columns() );
			return std::move(multi_array<const scalar,1,0,simdlen>( &_My_data[i * columns() + off], std::min(count, columns() - off) ));
		}

	public:
		multi_array& operator=(const multi_array& right) {
			if (&right != this) {
				if (!_I_am_ref && _My_data != nullptr)
					delete[] _My_data;
				_My_data    = new scalar_type[right.size()];
				_My_columns = right.columns();
				_I_am_ref   = false;
				std::copy(right.begin(), right.end(), _My_data);
			}

			return *this;
		}
		multi_array& operator=(multi_array&& right) noexcept {
			if (&right != this) {
				if (!_I_am_ref && _My_data != nullptr)
					delete[] _My_data;
				_My_data    = right._My_data;
				_My_columns = right._My_columns;
				_I_am_ref   = right._I_am_ref;
				right._My_data    = nullptr;
				right._My_columns = 0;
				right._I_am_ref   = false;
			}

			return *this;
		}
		void clear() {
			if (!_I_am_ref && _My_data != nullptr)
				delete[] _My_data;
			_My_data    = nullptr;
			_My_columns = 0;
			_I_am_ref   = false;
		}
	
		multi_array& recolumn(size_t new_columns) {
			if (!_I_am_ref && _My_data != nullptr)
				delete[] _My_data;
			_My_data    = new scalar_type[rows() * new_columns];
			_My_columns = new_columns;
			_I_am_ref   = false;
			return *this;
		}
		multi_array& reref(pointer data, size_t data_size) {
			assert( data_size % rows() == 0 );
			if (!_I_am_ref && _My_data != nullptr)
				delete[] _My_data;
			_My_data    = data;
			_My_columns = data_size / rows();
			_I_am_ref   = true;
			return *this;
		}
		multi_array& operator=(std::initializer_list<scalar_type> scalar_list) {
			if (this->isref()) { assert( scalar_list.size() == this->size() ); }
			else { assert( scalar_list.size() % this->rows() == 0 ); this->recolumn(scalar_list.size()/this->rows()); }
			std::copy(scalar_list.begin(), scalar_list.end(), begin());
			return *this;
		}
	
		multi_array() = default;
		multi_array(const multi_array& right) { (*this) = right; }
		multi_array(multi_array&& right) noexcept { (*this) = std::move(right); }
		~multi_array() {
			if (!_I_am_ref && _My_data != nullptr)
				delete[] _My_data;
			_My_data = nullptr;
		}
	
		multi_array(pointer data, size_t data_size) { reref(data, data_size); }
		multi_array(std::initializer_list<scalar_type> scalar_list) { (*this) = scalar_list; }
	
		// future...
	};

	template<typename scalar, size_t static_columns, size_t simdlen>
	class multi_array<scalar, 0, static_columns, simdlen> {
	public:
		using scalar_type = scalar;

		using value_type      = scalar;
		using pointer         = scalar *;
		using const_pointer   = const scalar *;
		using reference       = scalar &;
		using const_reference = const scalar &;
		using iterator        = scalar *;
		using const_iterator  = const scalar *;

		scalar *_My_data = nullptr;
		size_t _My_rows = 0;
		bool _I_am_ref = false;

		size_t          size()  const { return _My_rows * static_columns; }
		bool            empty() const { return _My_rows == 0; }
		bool            isref() const { return _I_am_ref; }

		pointer         data()        { return static_cast<pointer>(_My_data); }
		const_pointer   data()  const { return static_cast<const_pointer>(_My_data); }

		iterator        begin()       { return static_cast<iterator>(_My_data); }
		const_iterator  begin() const { return static_cast<const_iterator>(_My_data); }
	
		iterator        end()         { return static_cast<iterator>(_My_data) + size(); }
		const_iterator  end()   const { return static_cast<const_iterator>(_My_data) + size(); }
	
		reference       operator[](size_t i)       { return _My_data[i]; }
		const_reference operator[](size_t i) const { return _My_data[i]; }

		multi_array<scalar,1,0,simdlen> subarr(size_t off, size_t count = -1) {
			assert( off <= size() );
			return std::move(multi_array<scalar,1,0,simdlen>( &_My_data[off], std::min(count, size() - off) ));
		}
		multi_array<const scalar,1,0,simdlen> subarr(size_t off, size_t count = -1) const {
			assert( off <= size() );
			return std::move(multi_array<const scalar,1,0,simdlen>( &_My_data[off], std::min(count, size() - off) ));
		}

	public:
		using row_reference = multi_array<scalar, 1, static_columns, simdlen>&;
		using const_row_reference = const multi_array<scalar, 1, static_columns, simdlen>&;

		size_t              rows() const { return _My_rows; }
		static constexpr 
		size_t              columns() { return static_columns; }
		static constexpr 
		size_t              rowstep() { return static_columns; }

		reference           at(size_t i, size_t j)       { return _My_data[i*rowstep() + j]; }
		const_reference     at(size_t i, size_t j) const { return _My_data[i*rowstep() + j]; }
	
		row_reference       row(size_t i)       { return reinterpret_cast<row_reference>(_My_data[i*rowstep()]); }
		const_row_reference row(size_t i) const { return reinterpret_cast<const_row_reference>(_My_data[i*rowstep()]); }

		multi_array<scalar,1,0,simdlen> subrow(size_t i, size_t off, size_t count = -1) {
			assert( off <= columns() );
			return std::move(multi_array<scalar,1,0,simdlen>( &_My_data[i*rowstep()+off], std::min(count, columns() - off) ));
		}
		multi_array<const scalar,1,0,simdlen> subrow(size_t i, size_t off, size_t count = -1) const {
			assert( off <= columns() );
			return std::move(multi_array<const scalar,1,0,simdlen>( &_My_data[i*rowstep()+off], std::min(count, columns() - off) ));
		}

	public:
		multi_array& operator=(const multi_array& right) {
			if (&right != this) {
				if (!_I_am_ref && _My_data != nullptr)
					delete[] _My_data;
				_My_data  = new scalar_type[right.size()];
				_My_rows  = right.rows();
				_I_am_ref = false;
				std::copy(right.begin(), right.end(), _My_data);
			}
			return *this;
		}
		multi_array& operator=(multi_array&& right) noexcept {
			if (&right != this) {
				if (!_I_am_ref && _My_data != nullptr)
					delete[] _My_data;
				_My_data  = right._My_data;
				_My_rows  = right._My_rows;
				_I_am_ref = right._I_am_ref;
				right._My_data  = nullptr;
				right._My_rows  = 0;
				right._I_am_ref = false;
			}
			return *this;
		}
		void clear() {
			if (!_I_am_ref && _My_data != nullptr)
				delete[] _My_data;
			_My_data  = nullptr;
			_My_rows  = 0;
			_I_am_ref = false;
		}

		multi_array& rerow(size_t new_rows) {
			if (!_I_am_ref && _My_data != nullptr)
				delete[] _My_data;
			_My_data  = new scalar_type[new_rows * columns()];
			_My_rows  = new_rows;
			_I_am_ref = false;
			return *this;
		}
		multi_array& reref(pointer data, size_t data_size) {
			assert( data_size % columns() == 0 );
			if (!_I_am_ref && _My_data != nullptr)
				delete[] _My_data;
			_My_data  = data;
			_My_rows  = data_size / columns();
			_I_am_ref = true;
			return *this;
		}
		multi_array& operator=(std::initializer_list<scalar_type> scalar_list) {
			if (this->isref()) { assert( scalar_list.size() == this->size() ); }
			else { assert( scalar_list.size() % this->columns() == 0 ); this->rerow(scalar_list.size()/this->columns()); }
			std::copy(scalar_list.begin(), scalar_list.end(), this->begin());
			return *this;
		}

		multi_array() = default;
		multi_array(const multi_array& right) { (*this) = right; }
		multi_array(multi_array&& right) noexcept { (*this) = std::move(right); }
		~multi_array() {
			if (!_I_am_ref && _My_data != nullptr) {
				delete[] _My_data;
				_My_data = nullptr;
			}
		}

		multi_array(pointer data, size_t data_size) { reref(data, data_size); }
		multi_array(std::initializer_list<scalar_type> scalar_list) { (*this) = scalar_list; }

		/// future...
	};

	template<typename scalar, size_t simdlen>
	class multi_array<scalar, 0, 0, simdlen> {
	public:
		using scalar_type = scalar;

		using value_type      = scalar;
		using pointer         = scalar*;
		using const_pointer   = const scalar*;
		using reference       = scalar&;
		using const_reference = const scalar&;
		using iterator        = scalar*;
		using const_iterator  = const scalar*;
	
		scalar* _My_data = nullptr;
		bool _I_am_ref = false;
		size_t* _My_extents = nullptr;
		size_t* _My_steps = nullptr;
		size_t _My_dims = 0;
	
		size_t dims() const { return _My_dims; }
		const size_t* extents() const { return _My_extents; }
		size_t extent(size_t dim) const { return _My_extents[dim]; }

		size_t size() const {
			if (_My_dims == 0) {
				return 0;
			} else {
				size_t _My_size = _My_extents[0];
				for (size_t i = 1; i != _My_dims; ++i)
					_My_size *= _My_extents[i];
				return _My_size;
			}
		}
	
		bool empty() const { return size() == 0; }
		bool isref() const { return _I_am_ref; }

		pointer data() { return static_cast<pointer>(_My_data); }
		const_pointer data() const { return static_cast<const_pointer>(_My_data); }

		/// iterator.
		iterator begin() { return static_cast<iterator>(_My_data); }
		const_iterator begin() const { return static_cast<const_iterator>(_My_data); }
		iterator end() { return static_cast<iterator>(_My_data) + size(); }
		const_iterator end() const { return static_cast<const_iterator>(_My_data) + size(); }
 
		/// (*this)[i].
		reference operator[](size_t i) { return _My_data[i]; }
		const_reference operator[](size_t i) const { return _My_data[i]; }

		/// f(x).
		reference operator()(size_t x) { return _My_data[x]; }
		const_reference operator()(size_t x) const { return _My_data[x]; }

		/// f(x,y).
		reference operator()(size_t x, size_t y) { return _My_data[y*_My_steps[0] + x]; }
		const_reference operator()(size_t x, size_t y) const { return _My_data[y*_My_steps[0] + x]; }

		/// f(x,y,z).
		reference operator()(size_t x, size_t y, size_t z) { return _My_data[x + _My_steps[0]*y + _My_steps[1]*z]; }
		const_reference operator()(size_t x, size_t y, size_t z) const { return _My_data[x + _My_steps[0]*y + _My_steps[1]*z]; }

		/// f(x,y,z,w).
		reference operator()(size_t x, size_t y, size_t z, size_t w) { return _My_data[x + _My_steps[0]*y + _My_steps[1]*z + _My_steps[2]*w]; }
		const_reference operator()(size_t x, size_t y, size_t z, size_t w) const { return _My_data[x + _My_steps[0]*y + _My_steps[1]*z + _My_steps[2]*w]; }
	
		/// f({...}).
		template<typename integer_array>
		reference operator()(const integer_array& p) {
			assert( p.size() == dims() );
			if (p.size() == 1) {
				assert( 0 <= p[0] && p[0] < extent(0) );
				return _My_data[static_cast<size_t>(p[0])];
			} else if (p.size() == 2) {
				assert( 0 <= p[0] && p[0] < extent(0) );
				assert( 0 <= p[1] && p[1] < extent(1) );
				return _My_data[static_cast<size_t>(p[0]) + static_cast<size_t>(p[1])*_My_steps[0]];
			} else if (p.size() == 3) {
				assert( 0 <= p[0] && p[0] < extent(0) );
				assert( 0 <= p[1] && p[1] < extent(1) );
				assert( 0 <= p[2] && p[2] < extent(2) );
				return _My_data[static_cast<size_t>(p[0]) + static_cast<size_t>(p[1])*_My_steps[0] + static_cast<size_t>(p[2])*_My_steps[1]];
			} else if (p.size() == 4) {
				assert( 0 <= p[0] && p[0] < extent(0) );
				assert( 0 <= p[1] && p[1] < extent(1) );
				assert( 0 <= p[2] && p[2] < extent(2) );
				assert( 0 <= p[3] && p[3] < extent(3) );
				return _My_data[static_cast<size_t>(p[0]) + static_cast<size_t>(p[1])*_My_steps[0] + static_cast<size_t>(p[2])*_My_steps[1] + static_cast<size_t>(p[3])*_My_steps[2]];
			} else {
				assert( 0 <= p[0] && p[0] < extent(0) );
				size_t idx = static_cast<size_t>(p[0]);
				for (size_t i = 1; i != _My_dims; ++i) {
					assert( 0 <= p[i] && p[i] < extent(i) );
					idx += static_cast<size_t>(p[i])*_My_steps[i-1];
				}
				return _My_data[idx];
			}
		}
		template<typename integer_array>
		const_reference operator()(const integer_array& p) const {
			return static_cast<const_reference>( const_cast<multi_array&>(*this)(p) );
		}

		void clear() {
			if (!_I_am_ref && _My_data != nullptr)
				delete[] _My_data;
			if (_My_extents != nullptr)
				delete[] _My_extents;
			if (_My_steps != nullptr)
				delete[] _My_steps;
			_My_data = nullptr;
			_I_am_ref = false;
			_My_extents = nullptr;
			_My_steps = nullptr;
			_My_dims = 0;
		}

		void swap(multi_array& right) {
			std::swap(_My_data, right._My_data);
			std::swap(_I_am_ref, right._I_am_ref);
			std::swap(_My_extents, right._My_extents);
			std::swap(_My_steps, right._My_steps);
			std::swap(_My_dims, right._My_dims);
		}

		multi_array& resize(size_t dims, const size_t* extents) {
			if (dims != 0) {
				// Realloc _My_extents and _My_steps.
				if (_My_dims != dims) {
					if (_My_extents != nullptr)
						delete[] _My_extents;
					if (_My_steps != nullptr)
						delete[] _My_steps;

					_My_dims = dims;
					_My_extents = new size_t[dims];
					if (dims != 1) {
						_My_steps = new size_t[dims - 1];
					} else {
						_My_steps = nullptr;
					}
				}

				// Calculate _My_size and _My_steps, copy _My_extents.
				size_t _My_size = extents[0];
				_My_extents[0] = extents[0];
				for (size_t i = 1; i != dims; ++i) {
					_My_extents[i] = extents[i];
					_My_steps[i-1] = _My_size;
					_My_size *= extents[i];
				}

				// Resize _My_data by _My_size.
				if (!_I_am_ref && _My_data != nullptr)
					delete[] _My_data;
				if (_My_size != 0) {
					_My_data  = new scalar_type[_My_size];
					_I_am_ref = false;
				} else {
					_My_data = nullptr;
				}
			} else {
				this->clear();
			}

			return *this;
		}

		multi_array& resize(size_t width) { 
			size_t extents[1] = {width};
			return resize(1, extents);
		}

		multi_array& resize(size_t width, size_t height) { 
			size_t extents[2] = {width,height};
			return resize(2, extents);
		}

		multi_array& resize(size_t width, size_t height, size_t depth) { 
			size_t extents[3] = {width,height,depth};
			return resize(3, extents);
		}

		multi_array& reref(pointer data, size_t dims, const size_t* extents) {
			_My_data = data;
			_I_am_ref = true;
			if (dims != 0) {
				// Realloc _My_extents and _My_steps.
				if (_My_dims != dims) {
					if (_My_extents != nullptr)
						delete[] _My_extents;
					if (_My_steps != nullptr)
						delete[] _My_steps;

					_My_dims = dims;
					_My_extents = new size_t[dims];
					if (dims != 1) {
						_My_steps = new size_t[dims - 1];
					} else {
						_My_steps = nullptr;
					}
				}

				// Calculate _My_size and _My_steps, copy _My_extents.
				size_t _My_size = extents[0];
				_My_extents[0] = extents[0];
				for (size_t i = 1; i != dims; ++i) {
					_My_extents[i] = extents[i];
					_My_steps[i-1] = _My_size;
					_My_size *= extents[i];
				}

				// Reref _My_data by _My_size.
				if (!_I_am_ref && _My_data != nullptr)
					delete[] _My_data;
				if (_My_size != 0) {
					_My_data  = data;
					_I_am_ref = true;
				} else {
					_My_data = nullptr;
				}
			} else {
				this->clear();
			}

			return *this;
		}

		multi_array& reref(pointer data, size_t width, size_t height) {
			size_t extents[2] = {width,height};
			return reref(data, 2, extents);
		}

		multi_array& reref(pointer data, size_t width, size_t height, size_t depth) {
			size_t extents[3] = {width,height,depth};
			return reref(data, 3, extents);
		}

		multi_array& operator=(multi_array&& right) noexcept {
			if (&right != this) {
				right.swap(*this);
				right.clear();
			}
			return *this;
		}

		multi_array& operator=(const multi_array& right) {
			if (&right != this) {
				this->resize(right.dims(), right.extents());
				std::copy(right.begin(), right.end(), this->begin());
			}
			return *this;
		}

		multi_array& operator=(std::initializer_list<scalar> scalar_list) {
			if (this->isref()) { assert(scalar_list.size() == this->size()); } 
			else { this->resize(scalar_list.size()); }
			std::copy(scalar_list.begin(), scalar_list.end(), this->begin());
			return *this;
		}

		multi_array() = default;
		multi_array(multi_array&& right){ (*this) = std::move(right); }
		multi_array(const multi_array& right){ (*this) = right; }
		multi_array(std::initializer_list<scalar> scalar_list){ (*this) = scalar_list; }
		multi_array(size_t rows, size_t columns){ this->resize(columns, rows); }
		multi_array(size_t dims, const size_t* extents){ this->resize(dims, extents); }
		~multi_array() {
			if (!_I_am_ref && _My_data != nullptr)
				delete[] _My_data;
			if (_My_extents != nullptr)
				delete[] _My_extents;
			if(_My_steps != nullptr)
				delete[] _My_steps;
		}

	public:

		using row_reference = multi_array<scalar, 1, 0, simdlen>;
		using const_row_reference = multi_array<const scalar, 1, 0, simdlen>;

		size_t columns() const { return _My_dims == 0 ? 0 : _My_extents[0]; }
		size_t rows() const { return _My_dims < 2 ? 0 : _My_extents[1]; }
		size_t rowstep() const { return columns(); }
		reference at(size_t i, size_t j) { return (*this)(j, i); }
		const_reference at(size_t i, size_t j) const { return (*this)(j, i); }
		row_reference row(size_t i) { return std::move(row_reference( &_My_data[i * rowstep()], columns() )); }
		const_row_reference row(size_t i) const { return std::move(const_row_reference( &_My_data[i * rowstep()], columns() )); }
		row_reference subrow(size_t i, size_t off, size_t count = -1) { return std::move(row_reference( &_My_data[i*rowstep()+off], std::min(count, columns() - off) )); }
		const_row_reference subrow(size_t i, size_t off, size_t count = -1) const { return std::move(const_row_reference( &_My_data[i*rowstep()+off], std::min(count, columns() - off) )); }
		multi_array& reshape(size_t rows, size_t columns) { return resize(columns, rows); }
	};

	template<typename scalar, size_t static_rows, size_t static_columns, size_t simdlen>
	bool operator==(const multi_array<scalar, static_rows, static_columns, simdlen>& x, const multi_array<scalar, static_rows, static_columns, simdlen>& y) {
		for (size_t i = 0, iend = x.size(); i < iend; ++i) {
			if (x[i] != y[i]) {
				return false;
			}
		}

		return true;
	}

	template<typename scalar, size_t static_rows, size_t static_columns, size_t simdlen>
	bool operator!=(const multi_array<scalar, static_rows, static_columns, simdlen>& x, const multi_array<scalar, static_rows, static_columns, simdlen>& y) {
		for (size_t i = 0, iend = x.size(); i < iend; ++i) {
			if (x[i] == y[i]) {
				return false;
			}
		}

		return true;
	}
	
	template<size_t to_simdlen, typename scalar, size_t static_rows, size_t static_columns, size_t simdlen> inline
	multi_array<scalar, static_rows, static_columns, to_simdlen>& simdlen_cast(multi_array<scalar, static_rows, static_columns, simdlen>& x) {
		return reinterpret_cast<multi_array<scalar, static_rows, static_columns, to_simdlen>&>(x);
	}

	template<size_t to_simdlen, typename scalar, size_t static_rows, size_t static_columns, size_t simdlen> inline
	const multi_array<scalar, static_rows, static_columns, to_simdlen>& simdlen_cast(const multi_array<scalar, static_rows, static_columns, simdlen>& x) {
		return reinterpret_cast<const multi_array<scalar, static_rows, static_columns, to_simdlen>&>(x);
	}
	
	using std::to_string;

	template<typename T, size_t M, size_t N, size_t X>
	std::string to_string(const multi_array<T,M,N,X>& matrix, char newelement = ' ', char newline = '\n') {
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

	template<typename T, size_t M, size_t N, size_t X,
		typename _Elem, typename _Traits>
	std::basic_ostream<_Elem,_Traits>& operator<<(std::basic_ostream<_Elem,_Traits>& ostr, const multi_array<T,M,N,X>& matrix) {
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


	template<typename multi_array_type>
	struct multi_array_traits {};

	template<typename scalar, size_t static_rows, size_t static_columns, size_t simdlen>
	struct multi_array_traits<multi_array<scalar, static_rows, static_columns, simdlen>> {
		using scalar_type = scalar;
		static constexpr size_t static_rows = static_rows;
		static constexpr size_t static_columns = static_columns;
		static constexpr size_t simdlen = simdlen;
	};

	template<typename scalar, size_t s_rows, size_t s_columns, size_t simdlen>
	struct multi_array_traits<const multi_array<scalar, s_rows, s_columns, simdlen>>
		: public multi_array_traits<multi_array<scalar, s_rows, s_columns, simdlen>> {};

	template<typename multi_array_type>
	struct multi_array_traits<multi_array_type&> : public multi_array_traits<multi_array_type> {};

	#define multi_array_alloc(destination, source) \
	if constexpr (::math:: multi_array_traits<decltype(destination)>::static_rows == 0 \
		&& ::math:: multi_array_traits<decltype(destination)>::static_columns == 0) { \
		destination.resize(source.dims(), source.extents()); \
	} else if constexpr (::math:: multi_array_traits<decltype(destination)>::static_rows == 0) { \
		destination.rerow(source.rows()); \
	} else if constexpr (::math:: multi_array_traits<decltype(destination)>::static_columns == 0) { \
		destination.recolumn(source.columns()); \
	}

	#define matrix_alloc(destination, rows, columns) \
	if constexpr (::math:: multi_array_traits<decltype(destination)>::static_rows == 0 \
		&& ::math:: multi_array_traits<decltype(destination)>::static_columns == 0) { \
		destination.reshape(rows, columns); \
	} else if constexpr (::math:: multi_array_traits<decltype(destination)>::static_rows == 0) { \
		destination.rerow(rows); \
	} else if constexpr (::math:: multi_array_traits<decltype(destination)>::static_columns == 0) { \
		destination.recolumn(columns); \
	}


	template<typename array_, intptr_t s_rowoffset = 0, intptr_t s_columnoffset = 0>
	struct multi_array_slice {
		array_ *_My_array = nullptr;
	
		size_t rows() const {
			/// return std::min(_My_array->rows(),
			/// 	_My_array->rows() - static_cast<size_t>(s_rowoffset));
			if (s_rowoffset >= 0) {
				return _My_array->rows() - s_rowoffset;
			} else {
				return _My_array->rows();
			}
		}

		size_t columns() const {
			if (s_columnoffset >= 0) {
				return _My_array->columns() - s_columnoffset;
			} else {
				return _My_array->columns();
			}
		}

		size_t size() const {
			return rows() * columns();
		}

		decltype((*_My_array)[0]) operator[](size_t i) const {
			if (s_rowoffset < 0) { assert(i >= static_cast<size_t>(-s_rowoffset)); }
			return (*_My_array)[i + s_rowoffset];
		}

		decltype((*_My_array).at(0,0)) at(size_t i, size_t j) const {
			if (s_rowoffset < 0) { assert(i >= static_cast<size_t>(-s_rowoffset)); }
			if (s_columnoffset < 0) { assert(j >= static_cast<size_t>(-s_columnoffset)); }
			return (*_My_array).at(i + s_rowoffset, j + s_columnoffset);
		}

		multi_array_slice<array_,0,s_columnoffset> row(size_t i) const {
			return { _My_array };
		}

		using scalar_type = std::remove_cvref_t< decltype(array_{}[0]) > ;

		multi_array_slice& operator=(std::initializer_list<scalar_type> scalar_list) {
			if (s_rowoffset < 0) {
				std::copy_n(scalar_list.begin(), std::min(scalar_list.size(), 
					(*_My_array).size()), (*_My_array).begin());
			} else {
				std::copy_n(scalar_list.begin(), std::min(scalar_list.size(), 
					(*_My_array).size() - s_rowoffset), (*_My_array).begin());
			}
			return *this;
		}
	};
}