#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <cstdio>

struct vec3_t
{
	union
	{
		struct
		{
			float x;
			float y;
			float z;
		};
		float arr[3];
	};
};

__device__ __host__ inline void init_vec3(float *vec)
{
	vec[0] = 0.0f;
	vec[1] = 0.0f;
	vec[2] = 0.0f;
}

__device__ __host__ inline void init_vec3(float *vec, float *src)
{
	vec[0] = src[0];
	vec[1] = src[1];
	vec[2] = src[2];
}

__device__ __host__ inline void init_vec3(float *vec, float x, float y, float z)
{
	vec[0] = x;
	vec[1] = y;
	vec[2] = z;
}

__device__ __host__ inline void set_vec3(float *res, float *src)
{
	res[0] = src[0];
	res[1] = src[1];
	res[2] = src[2];
}

__device__ __host__ inline float dot(float *vecA, float *vecB)
{
	return vecA[0] * vecB[0] + vecA[1] * vecB[1] + vecA[2] * vecB[2];
}

__device__ __host__ inline void add(float *vecA, float *vecB)
{
	vecA[0] += vecB[0];
	vecA[1] += vecB[1];
	vecA[2] += vecB[2];
}

__device__ __host__ inline void add(float *res, float *vecA, float *vecB)
{
	res[0] = vecA[0] + vecB[0];
	res[1] = vecA[1] + vecB[1];
	res[2] = vecA[2] + vecB[2];
}

__device__ __host__ inline void sub(float *vecA, float *vecB)
{
	vecA[0] -= vecB[0];
	vecA[1] -= vecB[1];
	vecA[2] -= vecB[2];
}

__device__ __host__ inline void sub(float *res, float *vecA, float *vecB)
{
	res[0] = vecA[0] - vecB[0];
	res[1] = vecA[1] - vecB[1];
	res[2] = vecA[2] - vecB[2];
}

__device__ __host__ inline void mul(float *vecA, float c)
{
	vecA[0] *= c;
	vecA[1] *= c;
	vecA[2] *= c;
}

__device__ __host__ inline void mul(float *vecA, float *vecB)
{
	vecA[0] *= vecB[0];
	vecA[1] *= vecB[1];
	vecA[2] *= vecB[2];
}

__device__ __host__ inline void mul(float *res, float *vecA, float *vecB)
{
	res[0] = vecA[0] * vecB[0];
	res[1] = vecA[1] * vecB[1];
	res[2] = vecA[2] * vecB[2];
}

__device__ __host__ inline void mul(float *res, float *vecA, float c)
{
	res[0] = vecA[0] * c;
	res[1] = vecA[1] * c;
	res[2] = vecA[2] * c;
}

__device__ __host__ inline void cross(float *res, float *vecA, float *vecB)
{
	res[0] = vecA[1] * vecB[2] - vecA[2] * vecB[1];
	res[1] = vecA[2] * vecB[0] - vecA[0] * vecB[2];
	res[2] = vecA[0] * vecB[1] - vecA[1] * vecB[0];
}

__device__ __host__ inline float length(float *vec)
{
	return sqrtf(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
}

__device__ __host__ inline void normalize(float *vec)
{
	float len = length(vec);
	mul(vec, 1.0f / len);
}

__device__ __host__ inline void clamp(float *vec)
{
	if (vec[0] > 1.0f)
	{
		vec[0] = 1.0f;
	}
	if (vec[1] > 1.0f)
	{
		vec[1] = 1.0f;
	}
	if (vec[2] > 1.0f)
	{
		vec[2] = 1.0f;
	}
}

__device__ __host__ inline void normalize(float *res, float *vec)
{
	float len = length(vec);
	mul(res, vec, 1.0f / len);
}

__device__ __host__ inline void reflection(float *res, float *dir, float *normal)
{
	float dotp = dot(dir, normal);
	float temp[3];
	mul(temp, normal, dotp * 2.0f);
	sub(res, dir, temp);
	normalize(res);
}

__device__ __host__ inline void print_vec(float *vec)
{
	printf("[%f, %f, %f]", vec[0], vec[1], vec[2]);
}

#endif
