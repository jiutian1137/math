///
/// Infinite Precision Calculation for Rational Number
/// 
///@license Free, 2019-2022
///@author LongJiangnan, Jiang1998Nan@outlook.com
///@readme next work is infinite precision for dynamic rational number
#pragma once
#ifndef _MATH_BEGIN
#define _MATH_BEGIN namespace math {
#define _MATH_END }
#endif

#include <numeric>// std::gcd
#include <cmath>// std::abs

#include <iosfwd>// std::basic_ostream

_MATH_BEGIN
using std::abs;
using std::gcd;
using std::lcm;
template<typename Integer> constexpr
Integer gcd(const Integer sa, const Integer sb) {
	Integer a = std::_Abs_u(sa);
	Integer b = std::_Abs_u(sb);
	if (b > a) {
		std::swap(a, b);
	}

	while (b != 0) {
		Integer tmp = b;
		b = a % b;
		a = tmp;
	}

	return a != 0 ? a : 1;
}

template<typename Z>
class rational {
	Z p, q;/// p/q
public:

	typedef Z value_type;
	typedef Z integer_type;

	constexpr 
	explicit rational(
		integer_type numerator = static_cast<integer_type>(0), 
		integer_type denominator = static_cast<integer_type>(1)
	) : p(numerator), q(denominator) {}

	constexpr 
	rational& reduce() {
		integer_type d = gcd(p, q);
		p /= d;
		q /= d;
		return *this;
	}

	constexpr
	integer_type& numerator() {
		return p;
	}

	constexpr
	integer_type& denominator() {
		return q;
	}

	constexpr 
	rational copy_reduce() const {
		integer_type d = gcd(p, q);
		return rational(p/d, q/d);
	}

	constexpr
	const integer_type& numerator() const {
		return p;
	}

	constexpr
	const integer_type& denominator() const {
		return q;
	}

	constexpr
	rational reciprocal() const {
		return rational(denominator(), numerator());
	}

	constexpr 
	rational operator-() const {
		static_assert(std::is_signed_v<integer_type>);
		return rational(-numerator(), denominator());
	}

	constexpr 
	bool operator==(const rational& rhs) const {
		// assert( simplest(this) && simplest(right) )
		return this->numerator() == rhs.numerator()
			&& rhs.denominator() == this->denominator();
	}

	constexpr 
	bool operator!=(const rational& rhs) const {
		return !((*this) == rhs);
	}

	constexpr 
	bool operator<(const rational& rhs) const {
		return this->numerator() * rhs.denominator()
			< rhs.numerator() * this->denominator();
	}

	constexpr 
	bool operator>(const rational& rhs) const {
		return rhs < (*this);
	}

	constexpr 
	bool operator>=(const rational& rhs) const {
		return !(*this < rhs);
	}

	constexpr 
	bool operator<=(const rational& rhs) const {
		return !(*this > rhs);
	}

	constexpr
	rational& operator=(const rational&) = default;

	template<typename integer_type2>
		requires std::convertible_to<integer_type2, integer_type> constexpr
	rational& operator=(const integer_type2& number) {
		p = static_cast<integer_type>(number);
		q = 1;
		return *this;
	}

// arithmetic operators

	constexpr
	rational& operator+=(const rational& rhs) {
		integer_type rp = rhs.numerator();
		integer_type rq = rhs.denominator();

		if (rp == 0) {
			// (this + 0)
			return *this;
		}
		else if (p == 0) {
			// (0 + rhs)
			p = rp;
			q = rq;
			return *this;
		}
		else {
			///  p     rp     p*rq + rp*q
			/// --- + ---- = -------------
			///  q     rq         q*rq
			/// 
			///@optimization
			///	 p*rq + rp*q   d     (p*rq + rp*q)/d
			/// -------------/--- = -----------------
			///	     q*rq      d         q*rq/d
			/// 
			/// must a common divisor for numerator and denominator, p and rp not in denominator, so only q can be used.
			/// 	 (p*rq + rp*q)/gcd(q,rq)
			///		-------------------------
			///		      q*rq/gcd(q,rq)
			integer_type d = gcd(q,rq);
			rq /= d;
			p = p*rq + rp*(q/d);
			q = q*rq;
			return reduce();
		}
	}
	
	constexpr
	rational& operator-=(const rational& rhs) {
		integer_type rp = rhs.numerator();
		integer_type rq = rhs.denominator();

		if (rp == 0) {
			// (this - 0)
			return *this;
		}
		else if (p == 0) {
			// (0 - rhs)
			p = static_cast<integer_type>(0) - rp;
			q = rq;
			return *this;
		}
		else {
			///  p     rp     p*rq - rp*q
			/// --- - ---- = -------------
			///  q     rq         q*rq
			integer_type d = gcd(q,rq);
			rq /= d;
			p = p*rq - rp*(q/d);
			q = q*rq;
			return reduce();
		}
	}

	constexpr
	rational& operator*=(const rational& rhs) {
		integer_type rp = rhs.numerator();
		integer_type rq = rhs.denominator();

		if (p == 0 || rp == 0) {
			p = 0;
			q = 1;
			return *this;
		} 
		else {
			///  p     rp     p*rp
			/// --- * ---- = ------
			///  q     rq     q*rq
			/// 
			///@optimization
			///	 p*rp   d
			///	------/---
			///	 q*rq   d
			/// 
			/// gcd(p,q) = 1, gcd(rp,rq) = 1, so only cross gcd not equal 1
			///		 p*rp   gcd(p,rq)*gcd(rp,q)
			///		------/---------------------
			///		 q*rq   gcd(p,rq)*gcd(rp,q)
			integer_type d = gcd(p, rq);
			integer_type rd = gcd(rp, q);
			p = (p / d) * (rp / rd);
			q = (q / rd) * (rq / d);
			return reduce();
		}
	}
	
	constexpr
	rational& operator/=(const rational& rhs) {
		return *this *= rhs.reciprocal();
	}

	template<typename integer_type2> 
		requires std::convertible_to<integer_type2, integer_type> constexpr
	rational& operator+=(const integer_type2& rhs) {
		integer_type rp = static_cast<integer_type>(rhs);

		if (rp == 0) {
			// (this + 0)
			return *this;
		} 
		else if (p == 0) {
			// (0 + rhs)
			p = rp;
			q = 1;
			return *this;
		}
		else {
			///  p     rp     p + rp*q
			/// --- + ---- = ----------
			///  q     1        q
			p += rp*q;
			//q = q;
			return reduce();
		}
	}

	template<typename integer_type2> 
		requires std::convertible_to<integer_type2, integer_type> constexpr
	rational& operator-=(const integer_type2& rhs) {
		integer_type rp = static_cast<integer_type>(rhs);

		if (rp == 0) {
			// (this - 0)
			return *this;
		} 
		else if (p == 0) {
			// (0 - rhs)
			p = static_cast<integer_type>(0) - rp;
			q = 1;
			return *this;
		}
		else {
			///  p     rp     p - rp*q
			/// --- - ---- = ----------
			///  q     1        q
			p -= rp*q;
			//q = q;
			return reduce();
		}
	}

	template<typename integer_type2> 
		requires std::convertible_to<integer_type2, integer_type> constexpr
	rational& operator*=(const integer_type2& rhs) {
		integer_type rp = static_cast<integer_type>(rhs);

		if (p == 0 || rp == 0) {
			p = 0;
			q = 1;
			return *this;
		} 
		else {
			///  p     rp     p*rp
			/// --- * ---- = ------
			///  q     1       q
			integer_type d = gcd(rp, q);
			p *= (rp/d);
			q /= d;
			return reduce();
		}
	}

	template<typename integer_type2>
		requires std::convertible_to<integer_type2, integer_type> constexpr
	rational& operator/=(const integer_type2& rhs) {
		integer_type rq = static_cast<integer_type>(rhs);

		if (p == 0) {
			p = 0;
			q = 1;
			return *this;
		} 
		else {
			///  p     1       p
			/// --- * ---- = ------
			///  q     rq     q*rq
			integer_type d = gcd(p, rq);
			p /= d;
			q *= (rq/d);
			return reduce();
		}
	}

	constexpr
	rational operator+(const rational& rhs) const {
		integer_type rp = rhs.numerator();
		integer_type rq = rhs.denominator();

		if (rp == 0) {
			/// (this + 0)
			return *this;
		}
		else if (p == 0) {
			/// (0 + rhs)
			return rhs;
		}
		else {
			///  p     rp     p*rq + rp*q
			/// --- + ---- = -------------
			///  q     rq         q*rq
			/// 
			///@optimization
			///	 p*rq + rp*q   d     (p*rq + rp*q)/d
			///	-------------/--- = -----------------
			///	     q*rq      d         q*rq/d
			/// 
			/// must a common divisor for numerator and denominator, p and rp not in denominator, so only q can be used.
			/// 	 (p*rq + rp*q)/gcd(q,rq)
			///		-------------------------
			///		      q*rq/gcd(q,rq)
			integer_type d = gcd(q, rq);
			rq /= d;
			return rational(
				p*rq + rp*(q/d), 
				q*rq
				).copy_reduce();
		}
	}

	constexpr
	rational operator-(const rational& rhs) const {
		integer_type rp = rhs.numerator();
		integer_type rq = rhs.denominator();

		if (rp == 0) {
			/// (this - 0)
			return *this;
		}
		else if (p == 0) {
			/// (0 - rhs)
			return rational(
				static_cast<integer_type>(0) - rp, 
				rq
				);
		}
		else {
			///  p     rp     p*rq + rp*q
			/// --- + ---- = -------------
			///  q     rq         q*rq
			/// 
			///@optimization
			///	 p*rq + rp*q   d     (p*rq + rp*q)/d
			///	-------------/--- = -----------------
			///	     q*rq      d         q*rq/d
			/// 
			/// must a common divisor for numerator and denominator, p and rp not in denominator, so only q can be used.
			/// 	 (p*rq + rp*q)/gcd(q,rq)
			///		-------------------------
			///		      q*rq/gcd(q,rq)
			integer_type d = gcd(q, rq);
			rq /= d;
			return rational(
				p*rq - rp*(q/d), 
				q*rq
				).copy_reduce();
		}
	}
	
	constexpr
	rational operator*(const rational& rhs) const {
		integer_type rp = rhs.numerator();
		integer_type rq = rhs.denominator();

		if (p == 0 || rp == 0) {
			p = 0;
			q = 1;
			return *this;
		} 
		else {
			///  p     rp     p*rp
			/// --- * ---- = ------
			///  q     rq     q*rq
			/// 
			///@optimization
			///	 p*rp   d
			///	------/---
			///	 q*rq   d
			/// 
			/// gcd(p,q) = 1, gcd(rp,rq) = 1, so only cross gcd not equal 1
			///		 p*rp   gcd(p,rq)*gcd(rp,q)
			///		------/---------------------
			///		 q*rq   gcd(p,rq)*gcd(rp,q)
			integer_type d = gcd(p, rq);
			integer_type rd = gcd(rp, q);
			return rational(
				(p / d) * (rp / rd),
				(q / rd) * (rq / d)
				).copy_reduce();
		}
	}

	constexpr
	rational operator/(const rational& rhs) const {
		return (*this) * rhs.reciprocal();
	}
	
	template<typename integer_type2>
		requires std::convertible_to<integer_type2, integer_type> constexpr
	rational operator+(const integer_type2& rhs) const {
		integer_type rp = static_cast<integer_type>(rhs);

		if (rp == 0) {
			/// (this + 0)
			return *this;
		} 
		else if (p == 0) {
			/// (0 + rhs)
			return rational(rp, 1);
		}
		else {
			///  p     rp     p + rp*q
			/// --- + ---- = ----------
			///  q     1        q
			return rational(p+rp*q, q).copy_reduce();
		}
	}

	template<typename integer_type2>
		requires std::convertible_to<integer_type2, integer_type> constexpr
	rational operator-(const integer_type2& rhs) const {
		integer_type rp = static_cast<integer_type>(rhs);

		if (rp == 0) {
			/// (this - 0)
			return *this;
		} 
		else if (p == 0) {
			/// (0 - rhs)
			return rational(static_cast<integer_type>(0) - rp, 1);
		}
		else {
			///  p     rp     p + rp*q
			/// --- + ---- = ----------
			///  q     1        q
			return rational(p-rp*q, q).copy_reduce();
		}
	}

	template<typename integer_type2>
		requires std::convertible_to<integer_type2, integer_type> constexpr
	rational operator*(const integer_type2& rhs) const {
		integer_type rp = static_cast<integer_type>(rhs);

		if (p == 0 || rp == 0) {
			return rational();
		} 
		else {
			///  p     rp     p*rp
			/// --- * ---- = ------
			///  q     1       q
			integer_type d = gcd(rp, q);
			return rational(p*(rp/d), q/d).copy_reduce();
		}
	}

	template<typename integer_type2>
		requires std::convertible_to<integer_type2, integer_type> constexpr
	rational operator/(const integer_type2& rhs) const {
		integer_type rq = static_cast<integer_type>(rhs);

		if (p == 0) {
			return rational();
		} 
		else {
			///  p     1       p
			/// --- * ---- = ------
			///  q     rq     q*rq
			integer_type d = gcd(p, rq);
			return rational(p/d, q*(rq/d)).copy_reduce();
		}
	}

	constexpr
	rational& operator++() {
		p += q;
		return *this;
	}

	constexpr
	rational operator++(int) {
		rational copied = *this;
		++(*this);
		return copied;
	}

	constexpr
	rational& operator--() {
		p -= q;
		return *this;
	}

	constexpr
	rational operator--(int) {
		rational copied = *this;
		--(*this);
		return copied;
	}

	constexpr
	explicit rational(float value) {
		/**
		 * <source>
		 *   0 00000001 10101101011010110101000
		 * </source>
		 * <first>
		 *   <tips> IEEE754-floating-formula: (-1)^S * (1+0.Fraction) * 2^(Exponent-Bias) </tips>
		 *
		 *   (-1)^0 * (1 + 0.10101101011010110101000) * 2^(00000001 - Bias)
		 *     = 1 * 1.10101101011010110101000 * pow(2, _Exp)
		 *     = 1 * 0.110101101011010110101000 * pow(2, _Exp)
		 *     = 1 * 110101101011010110101000/pow(2,_Mn) * pow(2, _Exp)
		 * </first>
		 * <second>
		 *   <tips> pow(2, X) = (1 << X) </tips>
		 *
		 *   _Nx     110101101011010110101000
		 *   ---- = -------------------------- * ( 1 << _Exp )
		 *   _Dx           1 << _Mn
		 *
		 *           110101101011010110101000 << 1
		 *          ------------------------------- * ( 1 << (_Exp - 1) )
		 *                 1 << _Mn
		 *
		 *           110101101011010110101000
		 *          ------------------------------- * ( 1 << (_Exp - 1) )
		 *                 1 << (_Mn-1)
		 *
		 *           110101101011010110101000
		 *          ------------------------------- * (1 << 0)
		 *                 1 << (_Mn - _Exp)
		 * </second>
		*/
		constexpr unsigned int sign_mask
			= 0b10000000000000000000000000000000;
		constexpr unsigned int exponent_mask
			= 0b01111111100000000000000000000000;
		constexpr unsigned int mantissa_mask
			= 0b00000000011111111111111111111111;
		constexpr unsigned int hidden_significant
			= 0b00000000100000000000000000000000;
		constexpr char exp2_bias = 127;

		unsigned int value_bits = reinterpret_cast<uint32_t&>(value);
		unsigned int exp2_bits = (value_bits & exponent_mask) >> 23;
		unsigned int signifi_bits = value_bits & mantissa_mask | hidden_significant;
		char exp2 = reinterpret_cast<char&>(exp2_bits) - exp2_bias;
		exp2 -= 23;

		// *this = significant * pow(2,exp2)
		if (exp2 > 0) {
			p = static_cast<integer_type>(signifi_bits << exp2);
			q = static_cast<integer_type>(1);
		}
		else if (exp2 < 0) {
			static_assert(sizeof(integer_type) >= sizeof(unsigned int), "ratianal(float)");
			p = static_cast<integer_type>(signifi_bits);
			q = static_cast<integer_type>(1) << (-exp2);
		}
		else {
			p = static_cast<integer_type>(signifi_bits);
			q = static_cast<integer_type>(1);
		}

		// *this *= ~sign
		if ((value_bits & sign_mask) != 0) {
			p = static_cast<integer_type>(0) - p;
		}

		// divide greater_common_divisor
		integer_type d = gcd(p, q);
		p /= d;
		q /= d;
	}

	constexpr
	explicit rational(double value) {
		constexpr unsigned long long sign_mask
			= 0b1000000000000000000000000000000000000000000000000000000000000000;
		constexpr unsigned long long exponent_mask
			= 0b0111111111110000000000000000000000000000000000000000000000000000;
		constexpr unsigned long long mantissa_mask
			= 0b0000000000001111111111111111111111111111111111111111111111111111;
		constexpr unsigned long long hidden_significant
			= 0b0000000000010000000000000000000000000000000000000000000000000000;
		constexpr short exp2_bias = 1023;

		// seperate bits
		unsigned long long value_bits = reinterpret_cast<unsigned long long&>(value);
		unsigned long long exp2_bits = (value_bits & exponent_mask) >> 52;
		unsigned long long signifi_bits = value_bits & mantissa_mask | hidden_significant;
		short exp2 = reinterpret_cast<short&>(exp2_bits) - exp2_bias;
		exp2 -= 52;

		// *this = significant * pow(2,exp2)
		if (exp2 > 0) {
			p = static_cast<integer_type>(signifi_bits << exp2);
			q = static_cast<integer_type>(1);
		}
		else if (exp2 < 0) {
			static_assert(sizeof(integer_type) >= sizeof(unsigned long long), "ratianal(double)");
			p = static_cast<integer_type>(signifi_bits);
			q = static_cast<integer_type>(1) << (-exp2);
		}
		else {
			p = static_cast<integer_type>(signifi_bits);
			q = static_cast<integer_type>(1);
		}

		// *this *= ~sign
		if ((value_bits & sign_mask) != 0) {
			p = static_cast<integer_type>(0) - p;
		}

		// divide greater_common_divisor
		integer_type d = gcd(p, q);
		p /= d;
		q /= d;
	}

	template<typename floating_type>
	floating_type to_floating() const {
		// @see floating.hpp do_divide().
		// wait implement...
		abort();
	}

	template<typename _Elem, typename _Traits>
	friend std::basic_ostream<_Elem, _Traits>& operator<<(std::basic_ostream<_Elem, _Traits>& _Ostr, const rational& _R) {
		return _Ostr << _R.numerator() << '/' << _R.denominator();
	}
};

template<typename Integer> inline
rational<Integer> abs(const rational<Integer>& x) {
	if constexpr ( std::is_signed_v<Integer> ) {
		return rational<Integer>{ abs(x.numerator()), x.denominator() };
	} else {
		return x;
	}
}

///
/// f(x) = b[0]
///   + a[1]/(b[1]
///     + a[2]/(b[2]
///       + a[3]/(b[3]
///         + a[4]/(b[4]
///           + ... 
///              + a[n]/b[n] ))))
/// 
///@alternative 
/// f(x) = A[j]/B[j], when limit j->inf   :forward_recurrence
/// 
/// f(x) = b[0] + sum<j=1,inf>( term[j-1]*(b[j]/(B[j]/B[j-1]) - 1) )  :sum_series
/// 
/// f(x) = b[0]*product<j=1,inf>( (A[j]/A[j-1])/(B[j]/B[j-1]) )  :product_series
/// 
///@reference
/// @article A.R.Barnett
/// @article Lentz.
/// @tutorial "http://www.maths.surrey.ac.uk/hosted-sites/R.Knott/Fibonacci/cfCALC.html"
/// 
///@example
/// double x = 1.57;
/// 
/// std::vector<double> a(100);
/// std::vector<double> b(100);
/// // cosine function
/// b[0] = a[0] = 0;
/// b[1] = 1; a[1] = 1;
/// b[2] = 2-x*x; a[2] = x*x;
/// b[3] = 3*4-x*x; a[3] = 2*x*x;
/// for (size_t i = 4; i != a.size(); ++i) {
///		b[i] = ((i-1)*2-1)*((i-1)*2)-x*x;
///		a[i] = ((i-1)*2-3)*((i-1)*2-2)*x*x;
/// }
/// 
/// // tangent function
/// b[0] = a[0] = 0;
/// b[1] = 1/x; a[1] = 1;
/// for (size_t i = 2; i != a.size(); ++i) {
///		b[i] = (2*i-1)/x;
///		a[i] = -1;
/// }
/// 
template<typename Iterator>
auto evaluate_continued_fraction(Iterator num_it, Iterator den_it) {
/**
 * @continued_fraction
 *   f(x)
 *   = b[0]
 *     + a[1]/(b[1]
 *       + a[2]/(b[2]
 *         + a[3]/(b[3]
 *           + a[4]/(b[4]
 *             + ... 
 *                + a[n]/b[n] ))))
 * 
 * @continued_fraction forward recurrence
 *   limit j->inf
 *                  A[j]     A[j-1]*b[j] + A[j-2]*a[j]
 *   f(x) = Y[j] = ------ = ---------------------------
 *                  B[j]     B[j-1]*b[j] + B[j-2]*a[j]
 * 
 *                   1     A[-1]
 *   Y[-1] =  0  =  --- = -------
 *                   0     B[-1]  
 * 
 *                  b[0]     A[0]
 *   Y[0] = b[0] = ------ = ------
 *                   1       B[0]
 * 
 *           A[0]*b[1] + A[-1]*a[1]     b[0]*b[1] + a[1]
 *   Y[1] = ------------------------ = ------------------
 *           B[0]*b[1] + B[-1]*a[1]     b[1]
 * 
 * @continued_fraction to sum_series(Steed.) or product_series(Lentz.)
 * 
 *   First, we should find some properties of the variables 
 * 
 *                A[j]       A[j-1]*b[j] + A[j-2]*a[j]            A[j-2]
 *     Arel[j] = -------- = --------------------------- = b[j] + --------*a[j] = b[j] + a[j]/Arel[j-1]
 *                A[j-1]              A[j-1]                      A[j-1]
 * 
 *                B[j]       B[j-1]*b[j] + B[j-2]*a[j]            B[j-2]
 *     Brel[j] = -------- = --------------------------- = b[j] + --------*a[j] = b[j] + a[j]/Brel[j-1]
 *                B[j-1]              B[j-1]                      B[j-1]
 *            
 *   Then, we want to get the recurrence relation of Y[..]
 * 
 *      Y[j]       A[j]   A[j-1]     A[j]     B[j-1]
 *     -------- = ------/-------- = --------*--------  :This is derivation of Lentz's method
 *      Y[j-1]     B[j]   B[j-1]     A[j-1]   B[j]
 * 
 *     And
 * 
 *             A[j-1]*b[j] + A[j-2]*a[j]
 *     Y[j] = ---------------------------
 *             B[j-1]*b[j] + B[j-2]*a[j]
 * 
 *          Here has two kind division, The one                    Another
 * 
 *             ( A[j-1]*b[j] + A[j-2]*a[j] )/A[j-1]                   ( A[j-1]*b[j] + A[j-2]*a[j] )/B[j-1]
 *          = --------------------------------------               = --------------------------------------
 *             ( B[j-1]*b[j] + B[j-2]*a[j] )/A[j-1]                   ( B[j-1]*b[j] + B[j-2]*a[j] )/B[j-1]
 * 
 *                          Arel[j]                                   Y[j-1]*b[j] + A[j-2]/B[j-1]*a[j]
 *          = ------------------------------------                 = ----------------------------------
 *             1/Y[j-1]*b[j] + B[j-2]/A[j-1]*a[j]                                Brel[j]
 * 
 *                          Arel[j]                                   Y[j-1]*b[j] + A[j-2]/B[j-2]*B[j-2]/B[j-1]*a[j]
 *          = --------------------------------------------------   = ------------------------------------------------
 *             1/Y[j-1]*b[j] + B[j-2]/A[j-2]*A[j-2]/A[j-1]*a[j]                  Brel[j]
 * 
 *                          Arel[j]                                   Y[j-1]*b[j] + Y[j-2]/Brel[j-1]*a[j]
 *          = ---------------------------------------              = -------------------------------------
 *             b[j]/Y[j-1] + a[j]/(Y[j-2]*Arel[j-1])                             Brel[j]
 * 
 *   Now, we derivate the difference between Y[j] and Y[j-1]
 * 
 *                    Y[j-1]*b[j] + Y[j-2]/Brel[j-1]*a[j]
 *   Y[j] - Y[j-1] = ------------------------------------- - Y[j-1]
 *                             Brel[j]
 * 
 *                    Y[j-1]*b[j] - Y[j-1]*Brel[j] + Y[j-2]*(Brel[j]-b[j])
 *                 = ------------------------------------------------------   :a[j]/Brel[j-1] = Brel[j]-b[j]
 *                             Brel[j]
 * 
 *                    Y[j-1](b[j] - Brel[j]) + Y[j-2]*(Brel[j] - b[j])
 *                 = --------------------------------------------------
 *                             Brel[j]
 * 
 *                    (Y[j-1] - Y[j-2])*(b[j] - Brel[j])
 *                 = ------------------------------------                     :Y[j-2]*(Brel[j] - b[j]) = -Y[j-2]*(b[j] - Brel[j])
 *                             Brel[j]
 * 
 *                 = (Y[j-1] - Y[j-2])*(b[j]/Brel[j] - 1)                     :This is derivation of Steed's mathod
*/
	using Number = decltype(*num_it/(*den_it));
	const Number eps = std::numeric_limits<Number>::epsilon();
#if 0
	Number num = *num_it++;
	Number den = *den_it++;
	Number series = den;

	num = *num_it++;
	den = *den_it++;
	Number Brel = den;
	Number term = num/den;
	do {
		series += term;
		num = *num_it++;
		den = *den_it++;
		Brel = den + num/Brel;
		term = term*(den/Brel - 1);
	} while ( abs(term) >= eps*abs(series) );
	
	return series;
#else
	Number num = *num_it++;
	Number den = *den_it++;
	Number series = den; if (series == 0) series = 10e-50;
	
	num = *num_it++;
	den = *den_it++;
	Number Arel = den + num/series; if(Arel == 0) Arel = 1e-50;
	Number Brel = den;              if(Brel == 0) Brel = 1e-50;
	Number term = Arel/Brel;
	do {
		series *= term;
		num = *num_it++;
		den = *den_it++;
		Arel = den + num/Arel; if(Arel == 0) Arel = 1e-50;
		Brel = den + num/Brel; if(Brel == 0) Brel = 1e-50;
		term = Arel/Brel;
	} while ( abs(1 - term) >= eps );

	return series;
#endif
};
_MATH_END