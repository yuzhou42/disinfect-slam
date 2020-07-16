#pragma once

template<typename T>
class Vector2;

template<typename T>
class Vector3;

template<typename T>
class Vector4;

template<typename T>
class Vector2 {
 public:
  T x, y;
 public:
  __device__ __host__ Vector2<T>() {}

  __device__ __host__ Vector2<T>(const T &x_, const T &y_) 
      : x(x_), y(y_) {}

  __device__ __host__ explicit Vector2<T>(const T &scalar)
      : Vector2<T>(scalar, scalar) {}

  __device__ __host__ explicit Vector2<T>(const Vector3<T> &vec3)
      : Vector2<T>(vec3.x, vec3.y) {}

  __device__ __host__ inline Vector2<T>& operator+=(const Vector2<T> &rhs) {
    x += rhs.x; y += rhs.y;
    return *this;
  }

  __device__ __host__ inline Vector2<T>& operator+=(const T &rhs) {
    x += rhs; y += rhs;
    return *this;
  }

  __device__ __host__ inline Vector2<T>& operator-=(const Vector2<T> &rhs) {
    x -= rhs.x; y -= rhs.y;
    return *this;
  }

  __device__ __host__ inline Vector2<T>& operator-=(const T &rhs) {
    x -= rhs; y -= rhs;
    return *this;
  }

  __device__ __host__ inline Vector2<T>& operator*=(const T &rhs) {
    x *= rhs; y *= rhs;
    return *this;
  }

  __device__ __host__ inline Vector2<T>& operator/=(const T &rhs) {
    x /= rhs; y /= rhs;
    return *this;
  }

  __device__ __host__ inline Vector2<T>& operator&=(const T &rhs) {
    x &= rhs; y &= rhs;
    return *this;
  }

  __device__ __host__ inline Vector2<T>& operator<<=(const T &rhs) {
    x <<= rhs; y <<= rhs;
    return *this;
  }

  __device__ __host__ inline Vector2<T>& operator>>=(const T &rhs) {
    x >>= rhs; y >>= rhs;
    return *this;
  }

  __device__ __host__ inline Vector2<T> operator+(const Vector2<T> &rhs) const {
    Vector2<T> ret(*this); 
    return ret += rhs;
  }

  __device__ __host__ inline Vector2<T> operator+(const T &rhs) const {
    Vector2<T> ret(*this); 
    return ret += rhs;
  }

  __device__ __host__ inline Vector2<T> operator-(const Vector2<T> &rhs) const {
    Vector2<T> ret(*this); 
    return ret -= rhs;
  }

  __device__ __host__ inline Vector2<T> operator-(const T &rhs) const {
    Vector2<T> ret(*this); 
    return ret -= rhs;
  }

  __device__ __host__ inline Vector2<T> operator-() const {
    return (T)0 - *this;
  }

  __device__ __host__ inline Vector2<T> operator*(const T &rhs) const {
    Vector2<T> ret(*this);
    return ret *= rhs;
  }

  __device__ __host__ inline Vector2<T> operator/(const T &rhs) const {
    Vector2<T> ret(*this);
    return ret /= rhs;
  }

  __device__ __host__ inline Vector2<T> operator&(const T &rhs) const {
    Vector2<T> ret(*this);
    return ret &= rhs;
  }

  __device__ __host__ inline Vector2<T> operator<<(const T &rhs) const {
    Vector2<T> ret(*this);
    return ret <<= rhs;
  }

  __device__ __host__ inline Vector2<T> operator>>(const T &rhs) const {
    Vector2<T> ret(*this);
    return ret >>= rhs;
  }

  __device__ __host__ inline bool operator==(const Vector2<T> &rhs) const {
    return x == rhs.x && y == rhs.y;
  }

  __device__ __host__ inline bool operator!=(const Vector2<T> &rhs) const {
    return !operator==(rhs);
  }

  __device__ __host__ inline T dot(const Vector2<T> &rhs) const {
    return x * rhs.x + y * rhs.y;
  }

  template<typename Tout>
  __device__ __host__ inline Vector2<Tout> cast() const {
    return Vector2<Tout>(static_cast<Tout>(x), static_cast<Tout>(y));
  }
};

template<typename T>
__device__ __host__ inline Vector2<T> operator+(const T &lhs, const Vector2<T> &rhs) {
  Vector2<T> ret(lhs); 
  return ret += rhs;
}

template<typename T>
__device__ __host__ inline Vector2<T> operator-(const T &lhs, const Vector2<T> &rhs) {
  Vector2<T> ret(lhs); 
  return ret -= rhs;
}

template<typename T>
__device__ __host__ inline Vector2<T> operator*(const T &lhs, const Vector2<T> &rhs) {
  Vector2<T> ret(rhs); 
  return ret *= lhs;
}

template<typename T>
class Vector3 {
 public:
  T x, y, z;
 public:
  __device__ __host__ Vector3<T>() {}

  __device__ __host__ Vector3<T>(const T &x_, const T &y_, const T &z_) 
      : x(x_), y(y_), z(z_) {}

  __device__ __host__ explicit Vector3<T>(const Vector2<T> &vec2)
      : Vector3<T>(vec2.x, vec2.y, 1) {}

  __device__ __host__ explicit Vector3<T>(const Vector4<T> &vec4)
      : Vector3<T>(vec4.x, vec4.y, vec4.z) {}

  __device__ __host__ explicit Vector3<T>(const T &scalar)
      : Vector3<T>(scalar, scalar, scalar) {}

  __device__ __host__ inline Vector3<T>& operator+=(const Vector3<T> &rhs) {
    x += rhs.x; y += rhs.y; z += rhs.z;
    return *this;
  }

  __device__ __host__ inline Vector3<T>& operator+=(const T &rhs) {
    x += rhs; y += rhs; z += rhs;
    return *this;
  }

  __device__ __host__ inline Vector3<T>& operator-=(const Vector3<T> &rhs) {
    x -= rhs.x; y -= rhs.y; z -= rhs.z;
    return *this;
  }

  __device__ __host__ inline Vector3<T>& operator-=(const T &rhs) {
    x -= rhs; y -= rhs; z -= rhs;
    return *this;
  }

  __device__ __host__ inline Vector3<T>& operator*=(const T &rhs) {
    x *= rhs; y *= rhs; z *= rhs;
    return *this;
  }

  __device__ __host__ inline Vector3<T>& operator/=(const T &rhs) {
    x /= rhs; y /= rhs; z /= rhs;
    return *this;
  }

  __device__ __host__ inline Vector3<T>& operator&=(const T &rhs) {
    x &= rhs; y &= rhs; z &=rhs;
    return *this;
  }

  __device__ __host__ inline Vector3<T>& operator<<=(const T &rhs) {
    x <<= rhs; y <<= rhs; z <<= rhs;
    return *this;
  }

  __device__ __host__ inline Vector3<T>& operator>>=(const T &rhs) {
    x >>= rhs; y >>= rhs; z >>= rhs;
    return *this;
  }

  __device__ __host__ inline Vector3<T> operator+(const Vector3<T> &rhs) const {
    Vector3<T> ret(*this); 
    return ret += rhs;
  }

  __device__ __host__ inline Vector3<T> operator+(const T &rhs) const {
    Vector3<T> ret(*this); 
    return ret += rhs;
  }

  __device__ __host__ inline Vector3<T> operator-(const Vector3<T> &rhs) const {
    Vector3<T> ret(*this); 
    return ret -= rhs;
  }

  __device__ __host__ inline Vector3<T> operator-(const T &rhs) const {
    Vector3<T> ret(*this); 
    return ret -= rhs;
  }

  __device__ __host__ inline Vector3<T> operator-() const {
    return (T)0 - *this;
  }

  __device__ __host__ inline Vector3<T> operator*(const T &rhs) const {
    Vector3<T> ret(*this);
    return ret *= rhs;
  }

  __device__ __host__ inline Vector3<T> operator/(const T &rhs) const {
    Vector3<T> ret(*this);
    return ret /= rhs;
  }

  __device__ __host__ inline Vector3<T> operator&(const T &rhs) const {
    Vector3<T> ret(*this);
    return ret &= rhs;
  }

  __device__ __host__ inline Vector3<T> operator<<(const T &rhs) const {
    Vector3<T> ret(*this);
    return ret <<= rhs;
  }

  __device__ __host__ inline Vector3<T> operator>>(const T &rhs) const {
    Vector3<T> ret(*this);
    return ret >>= rhs;
  }

  __device__ __host__ inline bool operator==(const Vector3<T> &rhs) const {
    return x == rhs.x && y == rhs.y && z == rhs.z;
  }

  __device__ __host__ inline bool operator!=(const Vector3<T> &rhs) const {
    return !operator==(rhs);
  }

  __device__ __host__ inline T dot(const Vector3<T> &rhs) const {
    return x * rhs.x + y * rhs.y + z * rhs.z;
  }

  template<typename Tout>
  __device__ __host__ inline Vector3<Tout> cast() const {
    return Vector3<Tout>(static_cast<Tout>(x), static_cast<Tout>(y), static_cast<Tout>(z));
  }
};

template<typename T>
__device__ __host__ inline Vector3<T> operator+(const T &lhs, const Vector3<T> &rhs) {
  Vector3<T> ret(lhs); 
  return ret += rhs;
}

template<typename T>
__device__ __host__ inline Vector3<T> operator-(const T &lhs, const Vector3<T> &rhs) {
  Vector3<T> ret(lhs); 
  return ret -= rhs;
}

template<typename T>
__device__ __host__ inline Vector3<T> operator*(const T &lhs, const Vector3<T> &rhs) {
  Vector3<T> ret(rhs); 
  return ret *= lhs;
}

template<typename T>
class Vector4 {
 public:
  T x, y, z, w;
 public:
  __device__ __host__ Vector4<T>() {}

  __device__ __host__ Vector4<T>(const T &x_, const T &y_, const T &z_, const T &w_) 
      : x(x_), y(y_), z(z_), w(w_) {}

  __device__ __host__ explicit Vector4<T>(const T &scalar)
      : Vector4<T>(scalar, scalar, scalar, scalar) {}

  __device__ __host__ explicit Vector4<T>(const Vector3<T> &vec3) 
      : Vector4<T>(vec3.x, vec3.y, vec3.z, 1) {}

  __device__ __host__ inline Vector4<T>& operator+=(const Vector4<T> &rhs) {
    x += rhs.x; y += rhs.y; z += rhs.z; w += rhs.w;
    return *this;
  }

  __device__ __host__ inline Vector4<T>& operator+=(const T &rhs) {
    x += rhs; y += rhs; z += rhs; w += rhs;
    return *this;
  }

  __device__ __host__ inline Vector4<T>& operator-=(const Vector4<T> &rhs) {
    x -= rhs.x; y -= rhs.y; z -= rhs.z; w -= rhs.w;
    return *this;
  }

  __device__ __host__ inline Vector4<T>& operator-=(const T &rhs) {
    x -= rhs; y -= rhs; z -= rhs; w -= rhs;
    return *this;
  }

  __device__ __host__ inline Vector4<T>& operator*=(const T &rhs) {
    x *= rhs; y *= rhs; z *= rhs; w *= rhs;
    return *this;
  }

  __device__ __host__ inline Vector4<T> operator+(const Vector4<T> &rhs) const {
    Vector4<T> ret(*this); 
    return ret += rhs;
  }

  __device__ __host__ inline Vector4<T> operator+(const T &rhs) const {
    Vector4<T> ret(*this); 
    return ret += rhs;
  }

  __device__ __host__ inline Vector4<T> operator-(const Vector4<T> &rhs) const {
    Vector4<T> ret(*this); 
    return ret -= rhs;
  }

  __device__ __host__ inline Vector4<T> operator-(const T &rhs) const {
    Vector4<T> ret(*this); 
    return ret -= rhs;
  }

  __device__ __host__ inline Vector4<T> operator-() const {
    return (T)0 - *this;
  }

  __device__ __host__ inline Vector4<T> operator*(const T &rhs) const {
    Vector4<T> ret(*this);
    return ret *= rhs;
  }

  __device__ __host__ inline bool operator==(const Vector4<T> &rhs) const {
    return x == rhs.x && y == rhs.y && z == rhs.z && w == rhs.w;
  }

  __device__ __host__ inline bool operator!=(const Vector4<T> &rhs) const {
    return !operator==(rhs);
  }

  __device__ __host__ inline T dot(const Vector4<T> &rhs) const {
    return x * rhs.x + y * rhs.y + z * rhs.z + w * rhs.w;
  }

  template<typename Tout>
  __device__ __host__ inline Vector4<Tout> cast() const {
    return Vector4<Tout>(static_cast<Tout>(x), static_cast<Tout>(y), 
                         static_cast<Tout>(z), static_cast<Tout>(w));
  }
};

template<typename T>
__device__ __host__ inline Vector4<T> operator+(const T &lhs, const Vector4<T> &rhs) {
  Vector4<T> ret(lhs); 
  return ret += rhs;
}

template<typename T>
__device__ __host__ inline Vector4<T> operator-(const T &lhs, const Vector4<T> &rhs) {
  Vector4<T> ret(lhs); 
  return ret -= rhs;
}

template<typename T>
__device__ __host__ inline Vector4<T> operator*(const T &lhs, const Vector4<T> &rhs) {
  Vector4<T> ret(rhs); 
  return ret *= lhs;
}

