///
/// Quaternion Number
///
///@license
/// (C) Copyright Hubert Holin 2001.
/// Distributed under the Boost Software License, Version 1.0. (See
/// accompanying file LICENSE_1_0.txt or copy at
/// http://www.boost.org/LICENSE_1_0.txt)
///
///@see http://www.boost.org for updates, documentation, and revision history.
/// 
///@diagram
/// 
/// +-------------------------------------------------------+
/// |                    quaternion<T>                      |
/// +-------------------------------------------------------+
/// | -a: Real                                              |
/// | -b: Real                                              |
/// | -c: Real                                              |
/// | -d: Real                                              |
/// +-------------------------------------------------------+
/// |                    <<accessors>>                      |
/// | +real(): Real                                         |
/// | +unreal(): quaternion                                 |
/// | +R_component_1(): Real                                |
/// | +R_component_2(): Real                                |
/// | +R_component_3(): Real                                |
/// | +R_component_4(): Real                                |
/// | +C_component_1(): Complex                             |
/// | +C_component_2(): Complex                             |
/// +-------------------------------------------------------+
/// |                    <<assignment>>                     |
/// | +operator=(r:quaternion): (*this)                     |
/// | +operator=(r:Real): (*this)                           |
/// | +operator=(r:Complex): (*this)                        |
/// +-------------------------------------------------------+
/// |                    <<arithmetic>>                     |
/// | +operator+(Real|Complex|quaternion): quaternion       |
/// | +operator-(Real|Complex|quaternion): quaternion       |
/// | +operator*(Real|Complex|quaternion): quaternion       |
/// | +operator/(Real|Complex|quaternion): quaternion       |
/// | \operator+(Real|Complex,quaternion): quaternion       |
/// | \operator-(Real|Complex,quaternion): quaternion       |
/// | \operator*(Real|Complex,quaternion): quaternion       |
/// | \operator/(Real|Complex,quaternion): quaternion       |
/// | +operator+=(Real|Complex|quaternion): (*this)         |
/// | +operator-=(Real|Complex|quaternion): (*this)         |
/// | +operator*=(Real|Complex|quaternion): (*this)         |
/// | +operator/=(Real|Complex|quaternion): (*this)         |
/// +------------------------*------------------------------+
///                          |
///                          |
/// +------------------------+-------------------------------------+
/// |                         Algorithm                            |
/// +--------------------------------------------------------------+
/// | real(q:quaternion): Real                                     |
/// | unreal(q:quaternion): quaternion                             |
/// | polar(axis:Direction, arg:Real): quaternion                  |
/// | sup(q:quaternion): Real                                      |
/// | l1(q:quaternion): Real                                       |
/// | abs(q:quaternion): Real                                      |
/// | conj(q:quaternion): quaternion                               |
/// | norm(q:quaternion): Real                                     |
/// | slerp(q1:quaternion,q2:quaternion,t:Real): quaternion        |
/// | spherical(...): quaternion                                   |
/// | semipolar(...): quaternion                                   |
/// | multipolar(...): quaternion                                  |
/// | cylindrospherical(...): quaternion                           |
/// | cylindrical(...): quaternion                                 |
/// +--------------------------------------------------------------+
/// |                     <<Transcendentals>>                      |
/// | exp(q:quaternion): quaternion                                |
/// | cos(q:quaternion): quaternion                                |
/// | sin(q:quaternion): quaternion                                |
/// | tan(q:quaternion): quaternion                                |
/// | cosh(q:quaternion): quaternion                               |
/// | sinh(q:quaternion): quaternion                               |
/// | tanh(q:quaternion): quaternion                               |
/// | pow(q:quaternion,int): quaternion                            |
/// +--------------------------------------------------------------+
/// 
/// About the class name, there is a short story.
/// I know 'quaternion' all lowercases class name is very prominent in my library
/// (I define classes as a higher level of semantics than variables, that is, the
///  semantic changes caused by first letter uppercase), but they seem to be very
/// coordinated. So I looked and found that the variable names here are very short
/// (q,p,s,c,..) you can easily distinguish between type and variable. This is no
/// problem in the mathematical module, because the class name caries enough information,
/// since the variable itself only needs to be distinguished from other variables.
/// But it should not be in the physics module, because the variables in physics
/// have practical physical meaning in addition to the mathematical meaning(type).
/// Final I still decided to change 'quaternion' to 'Quaternion', because it needs
/// to work with other modules, and the template class cannot be regarded as 'size_t'
/// something like this.
#pragma once
#ifndef _MATH_BEGIN
#define _MATH_BEGIN namespace math {
#define _MATH_END }
#endif

#include <concepts>

#include <locale>  // for the "<<" operator

#include <complex>
#include <iosfwd>  // for the "<<" and ">>" operators
#include <string> // for the "to_string"

_MATH_BEGIN
template<typename R>
class quaternion {
	R a, b, c, d;/// a + b*i + c*j + d*k
public:

	typedef R value_type;
	typedef R scalar_type;
	typedef R real_type;
	typedef std::complex<R> complex_type;

	/// constructor for H seen as R^4
	constexpr 
	explicit quaternion(
		const real_type& a  = real_type(), 
		const real_type& bi = real_type(), 
		const real_type& cj = real_type(), 
		const real_type& dk = real_type()
	) : a(a), b(bi), c(cj), d(dk) {}

	/// constructor for H seen as C^2
	constexpr 
	explicit quaternion(
		const complex_type& z0, 
		const complex_type& z1 = complex_type()
	) : a(z0.real()), b(z0.imag()), c(z1.real()), d(z1.imag()) {}

/// accessors
///
///@note    Like complex number, quaternions do have a meaningful notion of "real part",
///         but unlike them there is no meaningful notion of "imaginary part".
///         Instead there is an "unreal part" which itself is a quaternion, and usually
///         nothing simpler (as opposed to the complex number case).
///         However, for practicality, there are accessors for the other components
///         (these are necessary for the templated copy constructor, for instance).

	constexpr 
	real_type real() const {
		return(a);
	}

	constexpr 
	quaternion unreal() const {
		return(quaternion(static_cast<real_type>(0), b, c, d));
	}

	constexpr 
	real_type R_component_1() const {
		return(a);
	}

	constexpr 
	real_type R_component_2() const {
		return(b);
	}

	constexpr 
	real_type R_component_3() const {
		return(c);
	}

	constexpr 
	real_type R_component_4() const {
		return(d);
	}

	constexpr 
	complex_type C_component_1() const {
		return(complex_type(a, b));
	}

	constexpr 
	complex_type C_component_2() const {
		return(complex_type(c, d));
	}

	constexpr
	real_type* data() {
		return &a;
	}

	constexpr
	const real_type* data() const {
		return &a;
	}

	constexpr
	real_type& operator[](size_t i) {
		return i == 0 ? a
			: i == 1 ? b
			: i == 2 ? c
			: d;
	}

	constexpr
	const real_type& operator[](size_t i) const {
		return i == 0 ? a
			: i == 1 ? b
			: i == 2 ? c
			: d;
	}

	constexpr
	bool operator==(const quaternion& rhs) const {
		return this->a == rhs.a && this->b == rhs.b && this->c == rhs.c && this->d == rhs.d;
	}

	constexpr
	bool operator!=(const quaternion& rhs) const {
		return !(*this == rhs);
	}

// assignment operators

	constexpr
	quaternion& operator=(const quaternion & a_affecter) {
		a = a_affecter.a;
		b = a_affecter.b;
		c = a_affecter.c;
		d = a_affecter.d;

		return(*this);
	}

	template<typename real_type2>
		requires std::convertible_to<real_type2, real_type> constexpr
	quaternion& operator=(const real_type2& a_affecter) {
		a = static_cast<real_type>(a_affecter);

		b = c = d = static_cast<real_type>(0);

		return(*this);
	}

	constexpr
	quaternion& operator=(const complex_type & a_affecter) {
		a = a_affecter.real();
		b = a_affecter.imag();

		c = d = static_cast<real_type>(0);

		return(*this);
	}

/// arithmetic operators
///
///@note    quaternion multiplication is *NOT* commutative;
///         symbolically, "q *= rhs;" means "q = q * rhs;"
///         and "q /= rhs;" means "q = q * inverse_of(rhs);"
///
///@note    Each operator comes in 2 forms - one for the simple case where
///         type T throws no exceptions, and one exception-safe version
///         for the case where it might.

// arithmetic with real(without divide at left)

	template<typename real_type2> 
		requires std::convertible_to<real_type2, real_type> constexpr
	quaternion operator+(const real_type2& rhs) const {
		return quaternion{ a + static_cast<real_type>(rhs), b, c, d };
	}
	
	template<typename real_type2> 
		requires std::convertible_to<real_type2, real_type> constexpr
	quaternion operator-(const real_type2& rhs) const {
		return quaternion{ a - static_cast<real_type>(rhs), b, c, d };
	}
	
	template<typename real_type2> 
		requires std::convertible_to<real_type2, real_type> constexpr
	quaternion operator*(const real_type2& rhs) const {
		auto ar = static_cast<real_type>(rhs);
		return quaternion{ a*ar, b*ar, c*ar, d*ar };
	}

	template<typename real_type2> 
		requires std::convertible_to<real_type2, real_type> constexpr
	quaternion operator/(const real_type2& rhs) const {
		auto ar = static_cast<real_type>(rhs);
		return quaternion{ a/ar, b/ar, c/ar, d/ar };
	}

	template<typename real_type2> 
		requires std::convertible_to<real_type2, real_type> constexpr
	friend quaternion operator+(const real_type2& lhs, const quaternion& rhs) {
		return quaternion{ static_cast<real_type>(lhs) + rhs.R_component_1(), rhs.R_component_2(), rhs.R_component_3(), rhs.R_component_4() };
	}
	
	template<typename real_type2>
		requires std::convertible_to<real_type2, real_type> constexpr
	friend quaternion operator-(const real_type2& lhs, const quaternion& rhs) {
		return quaternion{ static_cast<real_type>(lhs) - rhs.R_component_1(), rhs.R_component_2(), rhs.R_component_3(), rhs.R_component_4() };
	}

	template<typename real_type2> 
		requires std::convertible_to<real_type2, real_type> constexpr
	friend quaternion operator*(const real_type2& lhs, const quaternion& rhs) {
		auto a = static_cast<real_type>(lhs);
		return quaternion{ a*rhs.R_component_1(), a*rhs.R_component_2(), a*rhs.R_component_3(), a*rhs.R_component_4() };
	}

	template<typename real_type2> 
		requires std::convertible_to<real_type2, real_type> constexpr
	quaternion& operator+=(const real_type2& rhs) {
		a += static_cast<real_type>(rhs);
		return *this;
	}

	template<typename real_type2> 
		requires std::convertible_to<real_type2, real_type> constexpr
	quaternion& operator-=(const real_type2& rhs) {
		a -= static_cast<real_type>(rhs);
		return *this;
	}
	
	template<typename real_type2> 
		requires std::convertible_to<real_type2, real_type> constexpr
	quaternion& operator*=(const real_type2& rhs) {
		auto ar = static_cast<real_type>(rhs);
		a *= ar; b *= ar; c *= ar; d *= ar;
		return *this;
	}

	template<typename real_type2> 
		requires std::convertible_to<real_type2, real_type> constexpr
	quaternion& operator/=(const real_type2& rhs) {
		auto ar = static_cast<real_type>(rhs);
		a /= ar; b /= ar; c /= ar; d /= ar;
		return *this;
	}

// arithmetic with complex(without mul and div)

	constexpr 
	quaternion operator+(const complex_type& rhs) const {
		return quaternion{ a + rhs.real(), b + rhs.imag(), c, d };
	}

	constexpr 
	quaternion operator-(const complex_type& rhs) const {
		return quaternion{ a - rhs.real(), b - rhs.imag(), c, d };
	}

	constexpr 
	quaternion operator*(const complex_type& rhs) const {
		real_type ar = rhs.real();
		real_type br = rhs.imag();
		///@theory
		/// (a + b*i + c*j + d*k)*(ar + br*i)
		/// 
		/// = a*(ar + br*i) + b*i*(ar + br*i) + c*j*(ar + br*i) + d*k*(ar + br*i)
		/// 
		/// = a*ar + a*br*i + b*ar*i + b*br*(i*i=-1) + c*ar*j + c*br*(j*i=-k) + d*ar*k + d*br*(k*i=j)
		/// 
		/// = (a*ar - b*br) + (a*br + b*ar)*i + (c*ar + d*br)*j + (-c*br + d*ar)*k
		return quaternion{ (a*ar - b*br), (a*br + b*ar), (c*ar - d*br), (-c*br + d*ar) };
	}

	constexpr 
	quaternion operator/(const complex_type& rhs) const {
		real_type ar = rhs.real();
		real_type br = rhs.imag();
		real_type denominator = ar*ar+br*br;
		return quaternion{ (+a*ar + b*br)/denominator, (-a*br + b*ar)/denominator, (+c*ar - d*br)/denominator, (+c*br + d*ar)/denominator };
	}
	
	constexpr 
	friend quaternion operator+(const complex_type& lhs, const quaternion& rhs) {
		return quaternion{ lhs.real() + rhs.R_component_1(), lhs.imag() + rhs.R_component_2(), rhs.R_component_3(), rhs.R_component_4() };
	}

	constexpr 
	friend quaternion operator-(const complex_type& lhs, const quaternion& rhs) {
		return quaternion{ lhs.real() - rhs.R_component_1(), lhs.imag() - rhs.R_component_2(), rhs.R_component_3(), rhs.R_component_4() };
	}

	constexpr 
	friend quaternion operator*(const complex_type& lhs, const quaternion& rhs) {
		quaternion qlhs(lhs);
		return qlhs * rhs;
	}

	constexpr 
	friend quaternion operator/(const complex_type& lhs, const quaternion& rhs) {
		quaternion qlhs(lhs);
		return qlhs / rhs;
	}
	
	constexpr 
	quaternion& operator+=(const complex_type& rhs) {
		a += rhs.real(); b += rhs.imag();
		return *this;
	}

	constexpr 
	quaternion& operator-=(const complex_type& rhs) {
		a -= rhs.real(); b -= rhs.imag();
		return *this;
	}

	constexpr 
	quaternion& operator*=(const complex_type& rhs) {
		return *this = (*this) * rhs;
	}

	constexpr 
	quaternion& operator/=(const complex_type& rhs) {
		return *this = (*this) / rhs;
	}
	
// arithmetic with quaternion

	constexpr
	quaternion operator+(const quaternion& rhs) const {
		return quaternion{ a + rhs.R_component_1(), b + rhs.R_component_2(), c + rhs.R_component_3(), d + rhs.R_component_4() };
	}

	constexpr
	quaternion operator-(const quaternion& rhs) const {
		return quaternion{ a - rhs.R_component_1(), b - rhs.R_component_2(), c - rhs.R_component_3(), d - rhs.R_component_4() };
	}

	constexpr
	quaternion operator*(const quaternion& rhs) const {
		real_type ar = rhs.R_component_1();
		real_type br = rhs.R_component_2();
		real_type cr = rhs.R_component_3();
		real_type dr = rhs.R_component_4();
		///@theory
		/// (a + b*i + c*j + d*k)*(ar + br*i + cr*j + dr*k)
		/// 
		/// = a*(ar + br*i + cr*j + dr*k) + b*i*(ar + br*i + cr*j + dr*k) + c*j*(ar + br*i + cr*j + dr*k) + d*k*(ar + br*i + cr*j + dr*k)
		/// 
		/// = a*ar + a*br*i + a*cr*j + a*dr*k
		///   + b*ar*i + b*br*(i*i=-1) + b*cr*(i*j=k) + b*dr*(i*k=-j)
		///   + c*ar*j + c*br*(j*i=-k) + c*cr*(j*j=-1) + c*dr*(j*k=i)
		///   + d*ar*k + d*br*(k*i=j) + d*cr*(k*j=-i) + d*dr*(k*k=-1)
		/// 
		/// = (a*ar - b*br - c*cr - d*dr)
		///   + (a*br + b*ar + c*dr - d*cr)*i
		///   + (a*cr - b*dr + c*ar + d*br)*j
		///   + (a*dr + b*cr - c*br + d*ar)*k
		return quaternion{ (a*ar - b*br - c*cr - d*dr), (a*br + b*ar + c*dr - d*cr), (a*cr - b*dr + c*ar + d*br), (a*dr + b*cr - c*br + d*ar) };
	}

	constexpr
	quaternion operator/(const quaternion& rhs) const {
		real_type ar = rhs.R_component_1();
		real_type br = rhs.R_component_2();
		real_type cr = rhs.R_component_3();
		real_type dr = rhs.R_component_4();
		real_type denominator = ar*ar+br*br+cr*cr+dr*dr;
		return quaternion{ (+a*ar+b*br+c*cr+d*dr)/denominator, (-a*br+b*ar-c*dr+d*cr)/denominator, (-a*cr+b*dr+c*ar-d*br)/denominator, (-a*dr-b*cr+c*br+d*ar)/denominator };
	}

	constexpr
	quaternion& operator+=(const quaternion& rhs) {
		a += rhs.R_component_1();
		b += rhs.R_component_2();
		c += rhs.R_component_3();
		d += rhs.R_component_4();
		return *this;
	}

	constexpr
	quaternion& operator-=(const quaternion& rhs) {
		a -= rhs.R_component_1();
		b -= rhs.R_component_2();
		c -= rhs.R_component_3();
		d -= rhs.R_component_4();
		return *this;
	}

	constexpr
	quaternion& operator*=(const quaternion& rhs) {
		return *this = (*this) * rhs;
	}

	constexpr
	quaternion& operator/=(const quaternion& rhs) {
		return *this = (*this) / rhs;
	}
	
	template<typename real_type2> 
		requires std::convertible_to<real_type2, real_type> constexpr
	friend quaternion operator/(const real_type2& lhs, const quaternion& rhs) {
		quaternion qlhs(static_cast<real_type>(lhs));
		return qlhs / rhs;
	}
};

using quaternion_t = quaternion<float>;

// values

template<typename R> constexpr 
R real(const quaternion<R>& q) {
	return(q.real());
}

template<typename R> constexpr 
quaternion<R> unreal(const quaternion<R>& q) {
	return(q.unreal());
}

template<typename R, typename Vector3> 
quaternion<R> polar(const Vector3& axis, const R& angle) {
	R       angle2 = angle / 2;
	R       real   = cos(angle2);
	Vector3 unreal = axis * sin(angle2);
	return quaternion<R>{ real, unreal[0], unreal[1], unreal[2] };
}

template<typename R> constexpr
R sup(quaternion<R> const & q) {
	using    ::std::abs;
	using    ::std::max;
	return max(max(abs(q.R_component_1()), abs(q.R_component_2())), max(abs(q.R_component_3()), abs(q.R_component_4())));
}

template<typename R> constexpr
R l1(quaternion<R> const & q) {
	using    ::std::abs;
	return abs(q.R_component_1()) + abs(q.R_component_2()) + abs(q.R_component_3()) + abs(q.R_component_4());
}

template<typename T> 
T abs(quaternion<T> const & q) {
	using    ::std::abs;
	using    ::std::sqrt;
						
	T  maxim = sup(q);    // overflow protection
	if (maxim == 0) {
		return maxim;
	}

	T inv_mixam = static_cast<T>(1)/maxim;    // prefer multiplications over divisions
								
	T a = q.R_component_1() * inv_mixam;
	T b = q.R_component_2() * inv_mixam;
	T c = q.R_component_3() * inv_mixam;
	T d = q.R_component_4() * inv_mixam;
	return(inv_mixam * sqrt(a*a + b*b + c*c + d*d));

	//return(sqrt(norm(q)));
}

template<typename T> constexpr 
quaternion<T> conj(quaternion<T> const & q) {
	return(quaternion<T>( +q.R_component_1(),
		-q.R_component_2(),
		-q.R_component_3(),
		-q.R_component_4()));
}

///@note This is the Cayley norm, not the Euclidean norm...
template<typename T> constexpr 
T norm(quaternion<T> const & q) {
	return(real(q*conj(q)));
}

template<typename T> constexpr 
T dot(quaternion<T> const & q1, quaternion<T> const & q2) {
	return(q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3]);
}

template<typename T>
quaternion<T> normalize(quaternion<T> const& q) {
	T sqlen = dot(q,q);
	if (sqlen == 1 || sqlen == 0) {
		return q;
	} else {
		return q/sqrt(sqlen);
	}
}

template<typename T>
quaternion<T> slerp(quaternion<T> const & q1, quaternion<T> const & q2, T t) {
	using std::clamp;
	using std::acos;
	using std::sin;
	using std::abs;
	T cosTheta = dot(q1,q2);
	//T theta  = acos( clamp(cos_theta, static_cast<T>(-1), static_cast<T>(1)) );
	//T sin_theta = sin(theta);
	//return( q1*(sin((1 - t)*theta)/sin_theta) + q2*(sin(t*theta)/sin(theta)) );
	if (abs(1 - cosTheta) <= std::numeric_limits<T>::epsilon()) {
		return normalize((1 - t) * q1+t* q2);
	} else {
		T theta = acos(clamp(cosTheta, T(-1), T(1)));
		T thetap = theta * t;
		quaternion<T> qperp = normalize(q2 - q1 * cosTheta);
		return q1 * cos(thetap) + qperp * sin(thetap);
	}
}

template<typename T>
quaternion<T> spherical(T const & rho, T const & theta, T const & phi1, T const & phi2) {
	using ::std::cos;
	using ::std::sin;
						
	//T    a = cos(theta)*cos(phi1)*cos(phi2);
	//T    b = sin(theta)*cos(phi1)*cos(phi2);
	//T    c = sin(phi1)*cos(phi2);
	//T    d = sin(phi2);
						
	T    courrant = static_cast<T>(1);
						
	T    d = sin(phi2);
						
	courrant *= cos(phi2);
						
	T    c = sin(phi1)*courrant;
						
	courrant *= cos(phi1);
						
	T    b = sin(theta)*courrant;
	T    a = cos(theta)*courrant;
						
	return(rho*quaternion<T>(a,b,c,d));
}

template<typename T>
quaternion<T> semipolar(T const & rho, T const & alpha, T const & theta1, T const & theta2) {
	using ::std::cos;
	using ::std::sin;
						
	T    a = cos(alpha)*cos(theta1);
	T    b = cos(alpha)*sin(theta1);
	T    c = sin(alpha)*cos(theta2);
	T    d = sin(alpha)*sin(theta2);
						
	return(rho*quaternion<T>(a,b,c,d));
}

template<typename T> 
quaternion<T> multipolar(T const & rho1, T const & theta1, T const & rho2, T const & theta2) {
	using ::std::cos;
	using ::std::sin;
						
	T    a = rho1*cos(theta1);
	T    b = rho1*sin(theta1);
	T    c = rho2*cos(theta2);
	T    d = rho2*sin(theta2);
						
	return(quaternion<T>(a,b,c,d));
}

template<typename T> 
quaternion<T> cylindrospherical(T const & t, T const & radius, T const & longitude, T const & latitude) {
	using ::std::cos;
	using ::std::sin;
						
						
						
	T    b = radius*cos(longitude)*cos(latitude);
	T    c = radius*sin(longitude)*cos(latitude);
	T    d = radius*sin(latitude);
						
	return(quaternion<T>(t,b,c,d));
}

template<typename T>
quaternion<T> cylindrical(T const & r, T const & angle, T const & h1, T const & h2) {
	using ::std::cos;
	using ::std::sin;
						
	T    a = r*cos(angle);
	T    b = r*sin(angle);
						
	return(quaternion<T>(a,b,h1,h2));
}

// transcendentals(please see the documentation(boost::quaternion))

template<typename T>
quaternion<T> exp(quaternion<T> const & q) {
	using    ::std::exp;
	using    ::std::cos;
						
	using    ::boost::math::sinc_pi;
						
	T    u = exp(real(q));
						
	T    z = abs(unreal(q));
						
	T    w = sinc_pi(z);
						
	return(u*quaternion<T>(cos(z),
		w*q.R_component_2(), w*q.R_component_3(),
		w*q.R_component_4()));
}

template<typename T>
quaternion<T> cos(quaternion<T> const & q) {
	using    ::std::sin;
	using    ::std::cos;
	using    ::std::cosh;
						
	using    ::boost::math::sinhc_pi;
						
	T    z = abs(unreal(q));
						
	T    w = -sin(q.real())*sinhc_pi(z);
						
	return(quaternion<T>(cos(q.real())*cosh(z),
		w*q.R_component_2(), w*q.R_component_3(),
		w*q.R_component_4()));
}

template<typename T>
quaternion<T> sin(quaternion<T> const & q) {
	using    ::std::sin;
	using    ::std::cos;
	using    ::std::cosh;
						
	using    ::boost::math::sinhc_pi;
						
	T    z = abs(unreal(q));
						
	T    w = +cos(q.real())*sinhc_pi(z);
						
	return(quaternion<T>(sin(q.real())*cosh(z),
		w*q.R_component_2(), w*q.R_component_3(),
		w*q.R_component_4()));
}

template<typename T>
quaternion<T> tan(quaternion<T> const & q) {
	return(sin(q)/cos(q));
}

template<typename T>
quaternion<T> cosh(quaternion<T> const & q) {
	return((exp(+q)+exp(-q))/static_cast<T>(2));
}

template<typename T>
quaternion<T> sinh(quaternion<T> const & q) {
	return((exp(+q)-exp(-q))/static_cast<T>(2));
}

template<typename T>
quaternion<T> tanh(quaternion<T> const & q) {
	return(sinh(q)/cosh(q));
}

template<typename T>
quaternion<T> pow(quaternion<T> const & q, int n) {
		if        (n > 1)
		{
				int    m = n>>1;
								
				quaternion<T>    result = pow(q, m);
								
				result *= result;
								
				if    (n != (m<<1))
				{
						result *= q; // n odd
				}
								
				return(result);
		}
		else if    (n == 1)
		{
				return(q);
		}
		else if    (n == 0)
		{
				return(quaternion<T>(static_cast<T>(1)));
		}
		else    /* n < 0 */
		{
				return(pow(quaternion<T>(static_cast<T>(1))/q,-n));
		}
}

using std::to_string;

template<typename T>
std::string to_string(const quaternion<T>& q) {
	return to_string(q.R_component_1()) + ' ' + to_string(q.R_component_2()) + ' ' + to_string(q.R_component_3()) + ' ' + to_string(q.R_component_4());
}

template<typename _Elem, typename _Traits, typename T>
std::basic_ostream<_Elem,_Traits>& operator<<(std::basic_ostream<_Elem,_Traits>& ostr, const quaternion<T>& q) {
	return (ostr << q.R_component_1() << ' ' << q.R_component_2() << ' ' << q.R_component_3() << ' ' << q.R_component_4());
}
_MATH_END


//template<>
//struct __declspec(align(16)) quaternion<float, calculation::BlockTraits<__m128>> {
//	using traits_type = calculation::BlockTraits<__m128>;
//	using Scalar = typename traits_type::Scalar;
//	using Block  = typename traits_type::Block;

//	quaternion() = default;
//	
//	explicit quaternion(__m128 _a_b_c_d) : a_b_c_d(_a_b_c_d) {}

//	explicit quaternion(float _Re)
//		: a_b_c_d( _mm_setr_ps(_Re, 0.0f, 0.0f, 0.0f) ) {}
//	
//	quaternion(float _Re, float _Im0, float _Im1, float _Im2)
//		: a_b_c_d( _mm_setr_ps(_Re, _Im0, _Im1, _Im2) ) {}

//	template<typename VecT>
//	quaternion(float _Re, VecT _Im)
//		: a_b_c_d( _mm_setr_ps(_Re, _Im[0], _Im[1], _Im[2]) ) {}


//	friend quaternion operator+(quaternion q, quaternion r) {
//		return quaternion(_mm_add_ps(q.a_b_c_d, r.a_b_c_d));
//	}
//	friend quaternion operator-(quaternion q, quaternion r) {
//		return quaternion(_mm_sub_ps(q.a_b_c_d, r.a_b_c_d));
//	}
//	friend quaternion operator*(quaternion q, quaternion r) {
//		//__m128 qa = _mm_permute_ps(q.a_b_c_d, _MM_SHUFFLER(0, 0, 0, 0));
//		//__m128 ra = _mm_permute_ps(r.a_b_c_d, _MM_SHUFFLER(0, 0, 0, 0));

//		//__m128 qcdb = _mm_permute_ps(q.a_b_c_d, _MM_SHUFFLER(1, 2, 3, 1));
//		//__m128 qdbc = _mm_permute_ps(q.a_b_c_d, _MM_SHUFFLER(1, 3, 1, 2));
//		//__m128 rcdb = _mm_permute_ps(r.a_b_c_d, _MM_SHUFFLER(1, 2, 3, 1));
//		//__m128 rdbc = _mm_permute_ps(r.a_b_c_d, _MM_SHUFFLER(1, 3, 1, 2));
//		//__m128 cross_imag = _mm_sub_ps(_mm_mul_ps(qcdb, rdbc), _mm_mul_ps(qdbc, rcdb));
//		//
//		//__m128 dot_imag = _mm_mul_ps(qcdb, rcdb);
//		//dot_imag = _mm_add_ss(dot_imag, _mm_permute_ps(dot_imag, _MM_SHUFFLER(1,1,1,1)));
//		//dot_imag = _mm_add_ss(dot_imag, _mm_permute_ps(dot_imag, _MM_SHUFFLER(2,2,2,2)));
//		//
//		//__m128 result_real = _mm_sub_ps(_mm_mul_ss(qa, ra), dot_imag);
//		//__m128 result_imag = _mm_add_ps(_mm_add_ps(_mm_mul_ps(qa, r.a_b_c_d), _mm_mul_ps(ra, q.a_b_c_d)), cross_imag);
//		//return quaternion(_mm_shuffler_ps(
//		//	_mm_shuffler_ps(result_real, result_imag, _MM_SHUFFLER(0, 0, 1, 1)),// { real, real, imagX, imagX }
//		//	result_imag,// { x, imagX, imagY, imagZ }
//		//	_MM_SHUFFLER(0, 2, 2, 3) ));
//	}
//	
//	//friend quaternion operator-(quaternion q, quaternion r) {
//	//	return quaternion{ q.a - r.a, q.b - r.b, q.c - r.c, q.d - r.d };
//	//}
//	//
//	//friend quaternion operator*(quaternion q, quaternion r) {
//	//	// { q.real * r.real - dot(q.imag, r.imag), cross3(q.imag, r.imag) + q.real * r.imag + q.imag * r.real }
//	//	return quaternion{ q.a*r.a - (q.b*r.b + q.c*r.c + q.d*r.d),
//	//					   q.a*r.b + q.b*r.a + (q.c*r.d - q.d*r.c),
//	//					   q.a*r.c + q.c*r.a + (q.d*r.b - q.b*r.d),
//	//					   q.a*r.d + q.d*r.a + (q.b*r.c - q.c*r.b) };
//	//}
//	//
//	//friend quaternion operator/(quaternion q, quaternion r) {
//	//	// { dot(q, r), cross3(r, q) - q.real * r.imag + q.imag * r.real } / dot(r, r)
//	//	Scalar denominator = r.a*r.a + r.b*r.b + r.c*r.c + r.d*r.d;
//	//	return quaternion{ (+q.a*r.a + q.b*r.b + q.c*r.c + q.d*r.d) / denominator, 
//	//					   (-q.a*r.b + (q.d*r.c - q.c*r.d) + q.b*r.a) / denominator,
//	//					   (-q.a*r.c + (q.b*r.d - q.d*r.b) + q.c*r.a) / denominator,
//	//					   (-q.a*r.d + (q.c*r.b - q.b*r.c) + q.d*r.a) / denominator };
//	//}

//	// a + b*i + c*j + d*k
//	__m128 a_b_c_d;
//};
//
//using m128quaternion = quaternion<float, BlockTraits<__m128>>;