#pragma once

#include "utils/cuda/matrix.cuh"

template<typename T>
class SO3 : public Matrix3<T> {
 public:
  using Matrix3<T>::Matrix3;

  __device__ __host__ inline SO3<T> Inverse() const {
    return SO3<T>(
      this->m00, this->m10, this->m20,
      this->m01, this->m11, this->m21,
      this->m02, this->m12, this->m22
    );
  }
};

template<typename T>
class SE3 : public Matrix4<T> {
 public:
  using Matrix4<T>::Matrix4;

  __device__ __host__ SE3<T>(const SO3<T> &rot, const Vector3<T> &trans)
    : Matrix4<T>(
      rot.m00, rot.m01, rot.m02, trans.x,
      rot.m10, rot.m11, rot.m12, trans.y,
      rot.m20, rot.m21, rot.m22, trans.z,
      0,       0,       0,       1
    ) {}

  __device__ __host__ inline SE3<T> Inverse() const {
    const SO3<T> r_inv(
      this->m00, this->m10, this->m20,
      this->m01, this->m11, this->m21,
      this->m02, this->m12, this->m22
    );
    const Vector3<T> t(this->m03, this->m13, this->m23);
    return SE3<T>(r_inv, -r_inv * t);
  }

  __device__ __host__ inline SO3<T> GetR() const {
    return SO3<T>(
      this->m00, this->m01, this->m02,
      this->m10, this->m11, this->m12,
      this->m20, this->m21, this->m22
    );
  }

  __device__ __host__ inline Vector3<T> GetT() const {
    return Vector3<T>(this->m03, this->m13, this->m23);
  }

  __device__ __host__ inline Vector3<T> Apply(const Vector3<T> &vec3) const {
    return Vector3<T>(
      this->m00 * vec3.x + this->m01 * vec3.y + this->m02 * vec3.z + this->m03,
      this->m10 * vec3.x + this->m11 * vec3.y + this->m12 * vec3.z + this->m13,
      this->m20 * vec3.x + this->m21 * vec3.y + this->m22 * vec3.z + this->m23
    );
  }
};