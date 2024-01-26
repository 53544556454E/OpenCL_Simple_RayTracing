#ifndef FLOAT3_H_
#define FLOAT3_H_

#include <cmath>

class Float3
{
public:
	float x_;
	float y_;
	float z_;

	Float3(float x, float y, float z):
		x_(x),
		y_(y),
		z_(z)
	{
	}

	Float3(float xyz):
		x_(xyz),
		y_(xyz),
		z_(xyz)
	{
	}

	Float3():
		x_(0),
		y_(0),
		z_(0)
	{
	}

	Float3(cl_float3 f):
		x_(f.s0),
		y_(f.s1),
		z_(f.s2)
	{
	}

	float operator[](size_t i) const
	{
		float value = std::nanf("");

		switch (i)
		{
		case 0:
			value = x_;
			break;

		case 1:
			value = y_;
			break;

		case 2:
			value = z_;
			break;
		}

		return value;
	}

	explicit operator cl_float3() const
	{
		cl_float3 f {x_, y_, z_};

		return f;
	}

	float length_squared() const
	{
		return (x_ * x_) + (y_ * y_) + (z_ * z_);
	}

	float length() const
	{
		return sqrtf(length_squared());
	}

	Float3& normalize()
	{
		float rcp = 1.0f / length();

		x_ *= rcp;
		y_ *= rcp;
		z_ *= rcp;

		return *this;
	}
}; 

inline Float3 operator +(const Float3& f1, const Float3& f2)
{
	return Float3(f1.x_ + f2.x_, f1.y_ + f2.y_, f1.z_ + f2.z_);
}

inline Float3 operator -(const Float3& f1, const Float3& f2)
{
	return Float3(f1.x_ - f2.x_, f1.y_ - f2.y_, f1.z_ - f2.z_);
}

inline Float3 operator *(const Float3& f1, const Float3& f2)
{
	return Float3(f1.x_ * f2.x_, f1.y_ * f2.y_, f1.z_ * f2.z_);
}

inline Float3 operator *(const Float3& f1, float f)
{
	return Float3(f1.x_ * f, f1.y_ * f, f1.z_ * f);
}

inline Float3 operator *(float f, const Float3& f1)
{
	return f1 * f;
}

inline Float3 operator /(const Float3& f1, const Float3& f2)
{
	return Float3(f1.x_ / f2.x_, f1.y_ / f2.y_, f1.z_ / f2.z_);
}

inline Float3 operator /(const Float3& f1, float f)
{
	return f1 * (1.0f / f);
}

inline Float3 operator -(const Float3& f1)
{
	return Float3(-f1.x_, -f1.y_, -f1.z_);
}

inline Float3& operator +=(Float3& f1, Float3 const& f2)
{
	return f1 = f1 + f2;
}

inline Float3& operator -=(Float3& f1, Float3 const& f2)
{
	return f1 = f1 - f2;
}

inline Float3& operator *=(Float3& f1, Float3 const& f2)
{
	return f1 = f1 * f2;
}

inline Float3& operator *=(Float3& f1, float f)
{
	return f1 = f1 * f;
}

inline Float3& operator /=(Float3& f1, Float3 const& f2)
{
	return f1 = f1 / f2;
}

inline Float3& operator /=(Float3& f1, float f)
{
	return f1 = f1 / f;
}

inline float dot(const Float3& f1, const Float3& f2)
{
	return (f1.x_ * f2.x_) + (f1.y_ * f2.y_) + (f1.z_ * f2.z_);
}

inline Float3 cross(const Float3& f1, const Float3& f2)
{
	return Float3(
		(f1[1] * f2[2]) - (f1[2] * f2[1]),
		(f1[2] * f2[0]) - (f1[0] * f2[2]),
		(f1[0] * f2[1]) - (f1[1] * f2[0])
	);
}

#endif
