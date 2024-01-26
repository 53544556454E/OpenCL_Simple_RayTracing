#define _USE_MATH_DEFINES
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#include <CL/opencl.hpp>

#include "float3.h"

const cl_uint kWidth = 1200;
const cl_uint kHeight = 675;
const cl_uint kPlaneSize = sizeof (float) * kWidth * kHeight;
const size_t kSamples = 500;

struct Camera
{
	cl_uint2 image_size_;
	cl_float vertical_fov_;
	cl_uint max_reflections_;
	Float3 look_from_;
	Float3 look_at_;
	Float3 up_;
	cl_float defocus_angle_;
	cl_float focus_distance_;
};

enum class MaterialType: cl_uint
{
	Lambertian = 0,
	Metal,
	Dielectric
};

struct alignas(16) Lambertian
{
	cl_float3 albedo_;
};

struct alignas(16) Metal
{
	cl_float3 albedo_;
	float fuzziness_;
};

struct alignas(16) Dielectric
{
	float refractive_index_;
};

struct alignas(16) Sphere
{
	cl_float3 center_;
	cl_float radius_;
	MaterialType material_type_;
	union
	{
		Lambertian lambertian_;
		Metal metal_;
		Dielectric dielectric_;
	};
};

void PrintDevices();
void OutputImage(int width, int height, const float *red, const float *green, const float *blue);
cl::Buffer SetupRandom(cl::Context& ctx, cl::CommandQueue& cmd_queue, size_t elements);
cl::Buffer SetupCamera(cl::Context& ctx, cl::CommandQueue& cmd_queue, const Camera *cam);
cl::Buffer SetupObjects(cl::Context& ctx, cl::CommandQueue& cmd_queue, const std::vector<Sphere> objs);
float DegreesToRadians(float deg);

int main(int argc, char *argv[])
{
	if (argc < 3)
	{
		PrintDevices();
		std::cerr << argv[0] << " Platform-Index Device-Index" << std::endl;

		return 0;
	}

	std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();
	
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	std::vector<cl::Device> devices;
	platforms.at(std::atoi(argv[1])).getDevices(CL_DEVICE_TYPE_GPU, &devices);

	cl::Context ctx(devices.at(std::atoi(argv[2])));
	cl::CommandQueue command_queue(ctx);

	std::string program_string;
	{
		std::ifstream ifs("kernel.cl");
		program_string = std::string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
	}

	cl::Program program(ctx, program_string, true);
	cl::Kernel add_ray_color_kernel(program, "AddRayColor");
	cl::Kernel div_color_kernel(program, "DivColor");
	cl::Kernel gamma_correct_kernel(program, "GammaCorrect");

	cl::Buffer output_buffer(ctx, CL_MEM_READ_WRITE, kPlaneSize * 3);
	cl::Buffer random_buffer = SetupRandom(ctx, command_queue, kWidth * kHeight);

	Camera cam;
	cam.image_size_.x = kWidth;
	cam.image_size_.y = kHeight;
	cam.vertical_fov_ = 20.f;
	cam.max_reflections_ = 50;
	cam.look_from_ = Float3(13.f, 2.0f, 3.0f);
	cam.look_at_ = Float3(0.0f, 0.0f, 0.0f);
	cam.up_ = Float3(0.0f, 1.0f, 0.0f);
	cam.defocus_angle_ = 0.6f;
	cam.focus_distance_ = 10.f;
	cl::Buffer camera_buffer = SetupCamera(ctx, command_queue, &cam);

	std::vector<Sphere> objects;
	Sphere obj;

	obj.center_ = cl_float3 {0.0f, -1000.f, 0.0f};
	obj.radius_ = 1000.f;
	obj.material_type_ = MaterialType::Lambertian;
	obj.lambertian_.albedo_ = cl_float3 {0.5f, 0.5f, 0.5f};
	objects.push_back(obj);

	std::random_device seed_gen;
	std::mt19937 engine(seed_gen());
	std::uniform_real_distribution<float> dist1(0.0f, 1.0f);
	std::uniform_real_distribution<float> dist2(0.5f, 1.0f);
	std::uniform_real_distribution<float> dist3(0.0f, 0.2f);

	for (int a = -11; a < 11; a++)
	{
		for (int b = -11; b < 11; b++)
		{
			float choose_mat = dist1(engine);
			Float3 center(a + 0.9f * dist1(engine), 0.2f, b + 0.9f * dist1(engine));

			if ((center - Float3(4.0f, 0.2f, 0.0f)).length() > 0.9f)
			{
				if (choose_mat < 0.8f)
				{
					// diffuse
					Float3 albedo = Float3(dist1(engine), dist1(engine), dist1(engine)) * Float3(dist1(engine), dist1(engine), dist1(engine));
					obj.center_ = (cl_float3)center;
					obj.radius_ = 0.2f;
					obj.material_type_ = MaterialType::Lambertian;
					obj.lambertian_.albedo_ = (cl_float3)albedo;
					objects.push_back(obj);
				}
				else
				{
					if (choose_mat < 0.95f)
					{
						// metal
						Float3 albedo = Float3(dist2(engine), dist2(engine), dist2(engine));
						obj.center_ = (cl_float3)center;
						obj.radius_ = 0.2f;
						obj.material_type_ = MaterialType::Metal;
						obj.metal_.albedo_ = (cl_float3)albedo;
						obj.metal_.fuzziness_ = dist3(engine);
						objects.push_back(obj);
					}
					else
					{
						// glass
						obj.center_ = (cl_float3)center;
						obj.radius_ = 0.2f;
						obj.material_type_ = MaterialType::Dielectric;
						obj.dielectric_.refractive_index_ = 1.5f;
						objects.push_back(obj);
					}
				}
			}
		}
	}

	obj.center_ = cl_float3 {0.0f, 1.0f, 0.0f};
	obj.radius_ = 1.0f;
	obj.material_type_ = MaterialType::Dielectric;
	obj.dielectric_.refractive_index_ = 1.5f;
	objects.push_back(obj);

	obj.center_ = cl_float3 {-4.0f, 1.0f, 0.0f};
	obj.radius_ = 1.0f;
	obj.material_type_ = MaterialType::Lambertian;
	obj.lambertian_.albedo_ = cl_float3 {0.4f, 0.2f, 0.1f};
	objects.push_back(obj);

	obj.center_ = cl_float3 {4.0f, 1.0f, 0.0f};
	obj.radius_ = 1.0f;
	obj.material_type_ = MaterialType::Metal;
	obj.metal_.albedo_ = cl_float3 {0.7f, 0.6f, 0.5f};
	obj.metal_.fuzziness_ = 0.0f;
	objects.push_back(obj);

	cl::Buffer spheres_buffer = SetupObjects(ctx, command_queue, objects);

	// 出力先をクリアしておく
	command_queue.enqueueFillBuffer(output_buffer, cl_float(0.0f), 0, kPlaneSize * 3);
	command_queue.finish();

	// ray tracing
	{
		//kernel void AddRayColor(global float *rgb_plane, global uint *random_array, constant struct Camera *cam, constant struct Sphere *objs, uint objs_num);
		add_ray_color_kernel.setArg(0, output_buffer);
		add_ray_color_kernel.setArg(1, random_buffer);
		add_ray_color_kernel.setArg(2, camera_buffer);
		add_ray_color_kernel.setArg(3, spheres_buffer);
		add_ray_color_kernel.setArg(4, cl_uint(objects.size()));

		for (int i=0; i<kSamples; ++i)
		{
			command_queue.enqueueNDRangeKernel(add_ray_color_kernel, cl::NDRange(0), cl::NDRange(kWidth * kHeight));
			command_queue.finish();
		}
	}

	// 足し合わせた色の平均を求める
	{
		//kernel void DivColor(global float *rgb_plane, uint width, uint height, float divisor);
		div_color_kernel.setArg(0, output_buffer);
		div_color_kernel.setArg(1, cl_uint(kWidth));
		div_color_kernel.setArg(2, cl_uint(kHeight));
		div_color_kernel.setArg(3, cl_float(kSamples));
		command_queue.enqueueNDRangeKernel(div_color_kernel, cl::NDRange(0), cl::NDRange(kWidth * kHeight));
		command_queue.finish();
	}

	// ガンマ補正
	{
		//kernel void GammaCorrect(global float *rgb_plane, uint width, uint height, float gamma);
		gamma_correct_kernel.setArg(0, output_buffer);
		gamma_correct_kernel.setArg(1, cl_uint(kWidth));
		gamma_correct_kernel.setArg(2, cl_uint(kHeight));
		gamma_correct_kernel.setArg(3, cl_float(2.2f));
		command_queue.enqueueNDRangeKernel(gamma_correct_kernel, cl::NDRange(0), cl::NDRange(kWidth * kHeight));
		command_queue.finish();
	}
	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	std::chrono::milliseconds elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
	std::cerr << elapsed_time.count() << "[ms]" << std::endl;

	// 出力
	{
		void *p = command_queue.enqueueMapBuffer(output_buffer, CL_TRUE, CL_MAP_READ, 0, kPlaneSize * 3);
		float *r = static_cast<float *>(p);
		float *g = r + kWidth * kHeight;
		float *b = g + kWidth * kHeight;

		OutputImage(kWidth, kHeight, r, g, b);
		command_queue.enqueueUnmapMemObject(output_buffer, p);
	}
}

void PrintDevices()
{
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	std::cerr << "Platform-Index, Device-Index: Device-Name" << std::endl;

	for (int plat_index=0; plat_index<platforms.size(); ++plat_index)
	{
		std::vector<cl::Device> devices;

		platforms[plat_index].getDevices(CL_DEVICE_TYPE_ALL, &devices);
		for (int dev_index=0; dev_index<devices.size(); ++dev_index)
		{
			std::string device_name = devices[dev_index].getInfo<CL_DEVICE_NAME>();
			std::cerr << std::format("{}, {}: {}", plat_index, dev_index, device_name) << std::endl;
		}
	}
}

void OutputImage(int width, int height, const float *red, const float *green, const float *blue)
{
	std::cout << std::format("P3 {} {} 255\n", width, height);

	for (int y=0; y<height; ++y)
	{
		for (int x=0; x<width; ++x)
		{
			int r = std::clamp(static_cast<int>(255 * *red++), 0, 255);
			int g = std::clamp(static_cast<int>(255 * *green++), 0, 255);
			int b = std::clamp(static_cast<int>(255 * *blue++), 0, 255);

			std::cout << std::format("{} {} {}\n", r, g, b);
		}
	}

	std::cout.flush();
}

cl::Buffer SetupRandom(cl::Context& ctx, cl::CommandQueue& cmd_queue, size_t elements)
{
	void *ptr;
	const size_t random_bytes = sizeof (cl_uint) * elements;
	std::random_device seed_gen;
	std::mt19937 engine(seed_gen());
	cl::Buffer random_buffer(ctx, CL_MEM_READ_WRITE| CL_MEM_HOST_WRITE_ONLY, random_bytes);

	ptr = cmd_queue.enqueueMapBuffer(random_buffer, CL_TRUE, CL_MAP_WRITE, 0, random_bytes);
	
	for (size_t i=0; i<elements; ++i)
	{
		cl_uint v = 0;

		while (!v)
			v = engine();

		static_cast<cl_uint *>(ptr)[i] = v;
	}

	cmd_queue.enqueueUnmapMemObject(random_buffer, ptr);

	return random_buffer;
}

cl::Buffer SetupCamera(cl::Context& ctx, cl::CommandQueue& cmd_queue, const Camera *cam)
{
	struct alignas(16) KernelCamera
	{
		cl_uint2 image_size_;
		cl_uint max_reflections_;
		alignas(16) cl_float3 center_;
		cl_float3 pixel00_loc_;
		cl_float3 pixel_delta_u_;
		cl_float3 pixel_delta_v_;
		cl_float defocus_angle_;
		alignas(16) cl_float3 defocus_disk_u_;
		cl_float3 defocus_disk_v_;
	};

	KernelCamera kcam;
	cl::Buffer camera_buffer(ctx, CL_MEM_READ_ONLY| CL_MEM_HOST_WRITE_ONLY, sizeof (kcam));

	kcam.image_size_ = cam->image_size_;
	kcam.max_reflections_ = cam->max_reflections_;
	kcam.center_ = (cl_float3)cam->look_from_;

	Float3 w = (cam->look_from_ - cam->look_at_).normalize();
	Float3 u = cross(cam->up_, w).normalize();
	Float3 v = cross(w, u);
	float theta = DegreesToRadians(cam->vertical_fov_);
	float h = std::tanf(theta / 2.0f);
	float viewport_height = 2.0f * h * cam->focus_distance_;
	float viewport_width = viewport_height * (static_cast<float>(cam->image_size_.x)/cam->image_size_.y);
	Float3 viewport_u = viewport_width * u;
	Float3 viewport_v = viewport_height * -v;

	Float3 pixel_delta_u = viewport_u / cam->image_size_.x;
	Float3 pixel_delta_v = viewport_v / cam->image_size_.y;

	kcam.pixel_delta_u_ = (cl_float3)pixel_delta_u;
	kcam.pixel_delta_v_ = (cl_float3)pixel_delta_v;

	Float3 viewport_upper_left = cam->look_from_ - (cam->focus_distance_ * w) - viewport_u/2.0f - viewport_v/2.0f;

	kcam.pixel00_loc_ = (cl_float3)(viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v));

	float defocus_radius = cam->focus_distance_ * std::tanf(DegreesToRadians(cam->defocus_angle_ / 2.0f));
	kcam.defocus_angle_ = cam->defocus_angle_;
	kcam.defocus_disk_u_ = (cl_float3)(u * defocus_radius);
	kcam.defocus_disk_v_ = (cl_float3)(v * defocus_radius);

	void *ptr = cmd_queue.enqueueMapBuffer(camera_buffer, CL_TRUE, CL_MAP_WRITE, 0, sizeof (kcam));
	*static_cast<KernelCamera *>(ptr) = kcam;
	cmd_queue.enqueueUnmapMemObject(camera_buffer, ptr);

	return camera_buffer;
}

cl::Buffer SetupObjects(cl::Context& ctx, cl::CommandQueue& cmd_queue, const std::vector<Sphere> objs)
{
	const size_t obj_bytes = sizeof (Sphere) * objs.size();
	cl::Buffer objects_buffer(ctx, CL_MEM_READ_ONLY| CL_MEM_HOST_WRITE_ONLY, obj_bytes);

	void *ptr = cmd_queue.enqueueMapBuffer(objects_buffer, CL_TRUE, CL_MAP_WRITE, 0, obj_bytes);
	std::memcpy(ptr, objs.data(), obj_bytes);
	cmd_queue.enqueueUnmapMemObject(objects_buffer, ptr);

	return objects_buffer;
}

float DegreesToRadians(float deg)
{
	return deg * (float(M_PI) / 180.0f);
}
