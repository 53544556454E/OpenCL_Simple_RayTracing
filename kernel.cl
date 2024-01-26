typedef struct
{
	uint2 image_size_;
	uint max_reflections_;
	float3 center_;
	float3 pixel00_loc_;
	float3 pixel_delta_u_;
	float3 pixel_delta_v_;
	float defocus_angle_;
	float3 defocus_disk_u_;
	float3 defocus_disk_v_;
}
Camera;

typedef enum
{
	MaterialType_Lambertian = 0,
	MaterialType_Metal,
	MaterialType_Dielectric
}
MaterialType;

typedef struct
{
	float3 albedo_;
}
Lambertian;

typedef struct
{
	float3 albedo_;
	float fuzziness_;
}
Metal;

typedef struct
{
	float refractive_index_;
}
Dielectric;

typedef struct
{
	float3 center_;
	float radius_;
	MaterialType material_type_;
	union
	{
		Lambertian lambertian_;
		Metal metal_;
		Dielectric dielectric_;
	};
}
Sphere;

typedef struct
{
	float3 origin_;
	float3 direction_;
}
Ray;

typedef struct
{
	float3 p_;
	float t_;
	float3 normal_;
	bool front_face_;
	uint object_index_;
}
HitRecord;


kernel void AddRayColor(global float *rgb_plane, global uint *random_array, constant Camera *cam, constant Sphere *objs, uint objs_num);
kernel void DivColor(global float *rgb_plane, uint width, uint height, float divisor);
kernel void GammaCorrect(global float *rgb_plane, uint width, uint height, float gamma);

float3 ReadColor(const global float *rgb_plane, size_t plane_size, size_t index);
void WriteColor(global float *rgb_plane, float3 color, size_t plane_size, size_t index);

Ray GetRay(constant Camera *cam, uint x, uint y, uint *random_state);
float3 PixelSampleSquare(constant Camera *cam, uint *random_state);
float3 DefocusDiskSample(constant Camera *cam, uint *random_state);

float3 GetRayColor(const Ray *r, constant Sphere *objs, uint objs_num, uint reflections_remain, uint *random_state);
bool FindHitObject(HitRecord *rec, constant Sphere *objs, uint objs_num, const Ray *r, float near, float far);
bool HitObject(HitRecord *rec, constant Sphere *obj, const Ray *r, float near, float far);

bool LambertianScatter(Ray *scat_ray, float3 *attenuation, const HitRecord *rec, constant Sphere *obj, uint *random_state);
bool MetalScatter(Ray *scat_ray, float3 *attenuation, const HitRecord *rec, constant Sphere *obj, uint *random_state, const Ray *in_ray);
bool DielectricScatter(Ray *scat_ray, float3 *attenuation, const HitRecord *rec, constant Sphere *obj, uint *random_state, const Ray *in_ray);
float schlick(float cos_theta, float refraction_ratio);
float3 reflect(const float3 v, const float3 normal_uv);
float3 refract(const float3 in_ray_uv, const float3 normal_uv, float refraction_ratio);

uint GetRandomUint(uint *random_state);
float GetRandomUnitFloat(uint *random_state);
float GetRangedFloat(uint *value, float min, float max);
float3 GetRandomUnitVector(uint *random_state);
float3 GetRandomVectorInUnitSphere(uint *random_state);
float2 GetRandamVectorinUnitDisk(uint *random_state);


kernel void AddRayColor(global float *rgb_plane, global uint *random_array, constant Camera *cam, constant Sphere *objs, uint objs_num)
{
	size_t gid = get_global_id(0);
	size_t plane_size = (size_t)cam->image_size_.x * cam->image_size_.y;
	uint random_state;
	Ray ray;
	float3 ray_color;

	random_state = random_array[gid];

	{
		uint x = gid % cam->image_size_.x;
		uint y = gid / cam->image_size_.x;

		ray = GetRay(cam, x, y, &random_state);
	}

	ray_color = GetRayColor(&ray, objs, objs_num, cam->max_reflections_, &random_state);

	{
		float3 dst_color = ReadColor(rgb_plane, plane_size, gid);

		WriteColor(rgb_plane, dst_color + ray_color, plane_size, gid);
	}

	random_array[gid] = random_state;
}

kernel void DivColor(global float *rgb_plane, uint width, uint height, float divisor)
{
	size_t gid = get_global_id(0);
	size_t plane_size = (size_t)width * height;
	float3 color;

	color = ReadColor(rgb_plane, plane_size, gid);
	color /= divisor;
	WriteColor(rgb_plane, color, plane_size, gid);
}

kernel void GammaCorrect(global float *rgb_plane, uint width, uint height, float gamma)
{
	size_t gid = get_global_id(0);
	size_t plane_size = (size_t)width * height;
	float3 color;
	float inv_gamma;

	color = ReadColor(rgb_plane, plane_size, gid);
	inv_gamma = 1.0f / gamma;
	color = pow(color, (float3)(inv_gamma, inv_gamma, inv_gamma));
	WriteColor(rgb_plane, color, plane_size, gid);
}

float3 ReadColor(const global float *rgb_plane, size_t plane_size, size_t index)
{
	float3 result;

	result.s0 = rgb_plane[index];
	result.s1 = rgb_plane[plane_size + index];
	result.s2 = rgb_plane[plane_size * 2 + index];

	return result;
}

void WriteColor(global float *rgb_plane, float3 color, size_t plane_size, size_t index)
{
	rgb_plane[index] = color.s0;
	rgb_plane[plane_size + index] = color.s1;
	rgb_plane[plane_size * 2 + index] = color.s2;
}

Ray GetRay(constant Camera *cam, uint x, uint y, uint *random_state)
{
	Ray ray;
	float3 pixel_center = cam->pixel00_loc_ + ((float)x * cam->pixel_delta_u_) + ((float)y * cam->pixel_delta_v_);
	float3 pixel_sample = pixel_center + PixelSampleSquare(cam, random_state);

	ray.origin_ = (0 >= cam->defocus_angle_)? cam->center_: DefocusDiskSample(cam, random_state);
	ray.direction_ = pixel_sample - ray.origin_;

	return ray;
}

float3 PixelSampleSquare(constant Camera *cam, uint *random_state)
{
	float px = -0.5f + GetRandomUnitFloat(random_state);
	float py = -0.5f + GetRandomUnitFloat(random_state);

	return (px * cam->pixel_delta_u_) + (py * cam->pixel_delta_v_);
}

float3 DefocusDiskSample(constant Camera *cam, uint *random_state)
{
	float2 p = GetRandamVectorinUnitDisk(random_state);

	return cam->center_ + (p[0] * cam->defocus_disk_u_) + (p[1] * cam->defocus_disk_v_);
}

float3 GetRayColor(const Ray *ray, constant Sphere *objs, uint objs_num, uint reflections_remain, uint *random_state)
{
	float3 ray_color = (float3)(0.0f, 0.0f, 0.0f);
	float3 accumulated_attenuation = (float3)(1.0f, 1.0f, 1.0f);
	Ray ray_in = *ray;
	HitRecord rec;

	for (uint reflections=0; reflections<reflections_remain; ++reflections)
	{
		if (FindHitObject(&rec, objs, objs_num, &ray_in, 0.001f, INFINITY))
		{
			Ray ray_out;
			float3 attenuation;
			bool scattered = false;
	
			switch (objs[rec.object_index_].material_type_)
			{
			case MaterialType_Lambertian:
				scattered = LambertianScatter(&ray_out, &attenuation, &rec, &objs[rec.object_index_], random_state);
				break;
	
			case MaterialType_Metal:
				scattered = MetalScatter(&ray_out, &attenuation, &rec, &objs[rec.object_index_], random_state, &ray_in);
				break;
	
			case MaterialType_Dielectric:
				scattered = DielectricScatter(&ray_out, &attenuation, &rec, &objs[rec.object_index_], random_state, &ray_in);
				break;
			}
	
			if (scattered)
			{
				ray_in = ray_out;
				accumulated_attenuation *= attenuation;
			}
			else
				break;
		}
		else
		{
			float3 unit_direction = normalize(ray_in.direction_);
			float t = 0.5f * (unit_direction.s1 + 1.0f);

			ray_color = accumulated_attenuation * mix((float3)(1.0f, 1.0f, 1.0f), (float3)(0.5f, 0.7f, 1.0f), t);
			break;
		}
	}

	return ray_color;
}

bool FindHitObject(HitRecord *rec, constant Sphere *objs, uint objs_num, const Ray *r, float near, float far)
{
	HitRecord tmp_rec;
	bool hit_anything = false;
	float closest_so_far = far;

	for (uint i=0; i<objs_num; ++i)
	{
		if (HitObject(&tmp_rec, &objs[i], r, near, closest_so_far))
		{
			tmp_rec.object_index_ = i;
			hit_anything = true;
			closest_so_far = tmp_rec.t_;
			*rec = tmp_rec;
		}
	}

	return hit_anything;
}

bool HitObject(HitRecord *rec, constant Sphere *obj, const Ray *r, float near, float far)
{
	float3 oc = r->origin_ - obj->center_;
	float a = dot(r->direction_, r->direction_);
	float half_b = dot(oc, r->direction_);
	float c = dot(oc, oc) - obj->radius_ * obj->radius_;
	float d = half_b * half_b - a * c;

	if (0.0f < d)
	{
		float root = sqrt(d);

		float tmp = (-half_b - root) / a;
		if (near<tmp && far>tmp)
		{
			rec->t_ = tmp;
			rec->p_ = r->origin_ + tmp * r->direction_;
			float3 outward_normal = (rec->p_ - obj->center_) / obj->radius_;
			rec->front_face_ = 0.0f > dot(r->direction_, outward_normal);

			if (rec->front_face_)
				rec->normal_ = outward_normal;
			else
				rec->normal_ = -outward_normal;

			return true;
		}

		tmp = (-half_b + root) / a;
		if (near<tmp && far>tmp)
		{
			rec->t_ = tmp;
			rec->p_ = r->origin_ + tmp * r->direction_;
			float3 outward_normal = (rec->p_ - obj->center_) / obj->radius_;
			rec->front_face_ = 0.0f > dot(r->direction_, outward_normal);

			if (rec->front_face_)
				rec->normal_ = outward_normal;
			else
				rec->normal_ = -outward_normal;

			return true;
		}
	}

	return false;
}

bool LambertianScatter(Ray *scat_ray, float3 *attenuation, const HitRecord *rec, constant Sphere *obj, uint *random_state)
{
	scat_ray->origin_ = rec->p_;
	scat_ray->direction_ = rec->normal_ + GetRandomUnitVector(random_state);
	*attenuation = obj->lambertian_.albedo_;

	return true;
}

bool MetalScatter(Ray *scat_ray, float3 *attenuation, const HitRecord *rec, constant Sphere *obj, uint *random_state, const Ray *in_ray)
{
	float3 reflected = reflect(normalize(in_ray->direction_), rec->normal_);

	scat_ray->origin_ = rec->p_;
	scat_ray->direction_ = reflected + obj->metal_.fuzziness_ * GetRandomVectorInUnitSphere(random_state);
	*attenuation = obj->metal_.albedo_;

	if (0 < dot(scat_ray->direction_, rec->normal_))
		return true;

	return false;
}

bool DielectricScatter(Ray *scat_ray, float3 *attenuation, const HitRecord *rec, constant Sphere *obj, uint *random_state, const Ray *in_ray)
{
	float ref_idx = obj->dielectric_.refractive_index_;
	float refraction_ratio = rec->front_face_? (1.0f / ref_idx): ref_idx;
	float3 in_ray_unit_dir = normalize(in_ray->direction_);
	float cos_theta = min(dot(-in_ray_unit_dir, rec->normal_), 1.0f);
	float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
	bool cannot_refract = 1.0f < refraction_ratio * sin_theta;
	float reflectance = schlick(cos_theta, refraction_ratio);
	float3 reflected = reflect(in_ray_unit_dir, rec->normal_);
	float3 refracted = refract(in_ray_unit_dir, rec->normal_, refraction_ratio);

	*attenuation = (float3)(1.0f, 1.0f, 1.0f);
	scat_ray->origin_ = rec->p_;

	if ((1.0f < refraction_ratio * sin_theta) || (GetRandomUnitFloat(random_state) < reflectance))
		scat_ray->direction_ = reflected;
	else
		scat_ray->direction_ = refracted;

	return true;
}

float schlick(float cos_theta, float refraction_ratio)
{
	float coeff;
	float r0 = (1.0f - refraction_ratio) / (1.0f + refraction_ratio);

	r0 *= r0;
	coeff =  r0 + (1.0f - r0) * pown(1.0f - cos_theta, 5);

	return coeff;
}

float3 reflect(const float3 v, const float3 normal_uv)
{
	return v + 2.0f * dot(-v, normal_uv) * normal_uv;
}

float3 refract(const float3 in_ray_uv, const float3 normal_uv, float inrefi_div_outrefi)
{
	float cos_theta = min(dot(-in_ray_uv, normal_uv), 0.0f);
	float3 refract_parallel = inrefi_div_outrefi * (in_ray_uv + cos_theta * normal_uv);
	float3 refract_perpendicular = -sqrt(fabs(1.0f - dot(refract_parallel, refract_parallel))) * normal_uv;

	return refract_parallel + refract_perpendicular;
}

uint GetRandomUint(uint *random_state)
{
	uint x = *random_state;

	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;

	return *random_state = x;
}

float GetRandomUnitFloat(uint *random_state)
{
	uint x = GetRandomUint(random_state);

	return (float)(x & 0x7fff) / ((float)(0x7fff) + 1.0f);
}

float GetRangedFloat(uint *random_state, float min, float max)
{
	return mix(min, max, GetRandomUnitFloat(random_state));
}

float3 GetRandomUnitVector(uint *random_state)
{
	float phi = GetRangedFloat(random_state, 0.0f, 2 * M_PI_F);
	float z = GetRangedFloat(random_state, -1.0f, 1.0f);
	float r = sqrt(1.0f - z*z);
	float x = r * cos(phi);
	float y = r * sin(phi);

	return (float3)(x, y, z);
}

float3 GetRandomVectorInUnitSphere(uint *random_state)
{
	float phi = GetRangedFloat(random_state, 0.0f, 2 * M_PI_F);
	float z = GetRangedFloat(random_state, -1.0f, 1.0f);
	float r = GetRandomUnitFloat(random_state);

	float x = cbrt(r) * sqrt(1.0f - z*z) * cos(phi);
	float y = cbrt(r) * sqrt(1.0f - z*z) * sin(phi);
	z = cbrt(r) * z;

	return (float3)(x, y, z);
}

float2 GetRandamVectorinUnitDisk(uint *random_state)
{
	float theta = GetRangedFloat(random_state, -M_PI_F, M_PI_F);
	float r = sqrt(GetRandomUnitFloat(random_state));

	return (float2)(r*cos(theta), r*sin(theta));
}
