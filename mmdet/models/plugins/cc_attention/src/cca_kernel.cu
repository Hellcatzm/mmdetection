#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>

template <typename scalar_t>
__global__ void ca_forward_kernel(const scalar_t*  t,
                                  const scalar_t*  f,
                                  scalar_t*  weight,
                                  int num, int chn, int height, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int sp = height * width;
    int len = height + width - 1;
    int z = blockIdx.z;

    if (x < width && y < height && z < height+width-1) {
		for (int batch = 0; batch < num; ++batch) {
			for (int plane = 0; plane < chn; ++plane) {
				float _t = t[(batch * chn + plane) * sp + y*width + x];

				if (z < width) {
					int i = z;
					float _f = f[(batch * chn + plane) * sp + y*width + i];
					weight[(batch * len + i) * sp + y*width + x] += _t*_f;
				} else {
					int i = z - width;
					int j = i<y ? i : i+1;

					float _f = f[(batch * chn + plane) * sp + j*width + x];
					weight[(batch * len + width + i) * sp + y*width + x] += _t*_f;
				}
			}
		}
    }
}

template <typename scalar_t>
__global__ void ca_backward_kernel_t(const scalar_t* dw,
                                     const scalar_t* t,
                                     const scalar_t* f,
                                     scalar_t* dt,
                                     int num, int chn, int height, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int sp = height * width;
    int len = height + width - 1;
    int plane = blockIdx.z;

    if (x < width && y < height && plane < chn) {
        for (int batch = 0; batch < num; ++batch) {

			for (int i = 0; i < width; ++i) {
				float _dw = dw[(batch * len + i) * sp + y*width + x];
				float _f = f[(batch * chn + plane) * sp + y*width + i];
				dt[(batch * chn + plane) * sp + y*width + x] += _dw * _f;
			}
			for (int i = 0; i < height; ++i)  {
				if (i == y) continue;
				int j = i<y ? i : i-1;

				float _dw = dw[(batch * len + width + j) * sp + y*width + x];
				float _f = f[(batch * chn + plane) * sp + i*width + x];
				dt[(batch * chn + plane) * sp + y*width + x] += _dw * _f;
			}
		}
    }
}

template <typename scalar_t>
__global__ void ca_backward_kernel_f(const scalar_t* dw,
                                     const scalar_t* t,
                                     const scalar_t* f,
                                     scalar_t* df,
                                     int num, int chn, int height, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int sp = height * width;
    int len = height + width - 1;
    int plane = blockIdx.z;

    if (x < width && y < height && plane < chn) {
        for (int batch = 0; batch < num; ++batch) {

			for (int i = 0; i < width; ++i) {
				float _dw = dw[(batch * len + x) * sp + y*width + i];
				float _t = t[(batch * chn + plane) * sp + y*width + i];
				df[(batch * chn + plane) * sp + y*width + x] += _dw * _t;
		    }
		    for (int i = 0; i < height; ++i) {
			    if (i == y) continue;
			    int j = i>y ? y : y-1;

  	   		    float _dw = dw[(batch * len + width + j) * sp + i*width + x];
			    float _t = t[(batch * chn + plane) * sp + i*width + x];
			    df[(batch * chn + plane) * sp + y*width + x] += _dw * _t;
		    }
		}
    }
}

template <typename scalar_t>
__global__ void ca_map_forward_kernel(const scalar_t* weight,
									  const scalar_t* g,
									  scalar_t* out,
									  int num, int chn, int height, int width) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int sp = height * width;
	int len = height + width - 1;
	int plane = blockIdx.z;

	if (x < width && y < height && plane < chn) {
		for (int batch = 0; batch < num; ++batch) {

			for (int i = 0; i < width; ++i) {
				float _g = g[(batch * chn + plane) * sp + y*width + i];
				float _w = weight[(batch * len + i) * sp + y*width + x];
				out[(batch * chn + plane) * sp + y*width + x] += _g * _w;
			}
			for (int i = 0; i < height; ++i) {
				if (i == y) continue;

				int j = i<y ? i : i-1;

				float _g = g[(batch * chn + plane) * sp + i*width + x];
				float _w = weight[(batch * len + width + j) * sp + y*width + x];
				out[(batch * chn + plane) * sp + y*width + x] += _g * _w;
			}
		}
	}
}

template <typename scalar_t>
__global__ void ca_map_backward_kernel_w(const scalar_t* dout,
									     const scalar_t* weight,
										 const scalar_t* g,
										 scalar_t* dw,
										 int num, int chn, int height, int width) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int sp = height * width;
	int len = height + width - 1;
	int z = blockIdx.z;

	if (x < width && y < height && z < height+width-1) {

		for (int batch = 0; batch < num; ++batch) {
			for (int plane = 0; plane < chn; ++plane) {
				float _dout = dout[(batch * chn + plane) * sp + y*width + x];

				if (z < width) {
					int i = z;
					float _g = g[(batch * chn + plane) * sp + y*width + i];
					dw[(batch * len + i) * sp + y*width + x] += _dout * _g;
				}
				else {
					int i = z - width;
					int j = i<y ? i : i+1;

					float _g = g[(batch * chn + plane) * sp + j*width + x];
					dw[(batch * len + width + i) * sp + y*width + x] += _dout * _g;
				}
			}
		}
	}
}

template <typename scalar_t>
__global__ void ca_map_backward_kernel_g(const scalar_t* dout,
										 const scalar_t* weight,
										 const scalar_t* g,
										 scalar_t* dg,
										 int num, int chn, int height, int width) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int sp = height * width;
	int len = height + width - 1;
	int plane = blockIdx.z;

	if (x < width && y < height && plane < chn) {

		for (int batch = 0; batch < num; ++batch) {
			for (int i = 0; i < width; ++i) {
				float _dout = dout[(batch * chn + plane) * sp + y*width + i];
				float _w = weight[(batch * len + x) * sp + y*width + i];
				dg[(batch * chn + plane) * sp + y*width + x] += _dout * _w;
			}

			for (int i = 0; i < height; ++i) {
				if (i == y) continue;
				int j = i>y ? y : y-1;

				float _dout = dout[(batch * chn + plane) * sp + i*width + x];
				float _w = weight[(batch * len + width + j) * sp + i*width + x];
				dg[(batch * chn + plane) * sp + y*width + x] += _dout * _w;
			}
		}
	}
}

int CAForwardLaucher(const int N, const int C, const int H, const int W,
                     const at::Tensor t, const at::Tensor f,
					 at::Tensor weight){
    // Run kernel
    dim3 threads(32, 32);
    int d1 = (W+threads.x-1)/threads.x;
    int d2 = (H+threads.y-1)/threads.y;
    int d3 = H+W;
    dim3 blocks(d1, d2, d3);
    // ca_forward_kernel<<<blocks, threads, 0, stream>>>(t, f, weight, N, C, H, W);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
		t.type(), "CALaucherForward", ([&] {
		ca_forward_kernel<<<blocks, threads>>>(
			t.data<scalar_t>(), f.data<scalar_t>(), weight.data<scalar_t>(), N, C, H, W);
    }));

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
		return 0;
    else
		return 1;
}

int CABackwardLaucher(const int N, const int C, const int H, const int W,
                      const at::Tensor dw, const at::Tensor t, const at::Tensor f,
                      at::Tensor dt, at::Tensor df){
    // Run kernel
    dim3 threads(32, 32);
    int d1 = (W+threads.x-1)/threads.x;
    int d2 = (H+threads.y-1)/threads.y;
    int d3 = C;
    dim3 blocks(d1, d2, d3);
    // printf("%f\n", dw[0]);
    // ca_backward_kernel_t<<<blocks, threads>>>(dw, t, f, dt, N, C, H, W);
    // ca_backward_kernel_f<<<blocks, threads>>>(dw, t, f, df, N, C, H, W);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
		t.type(), "CALaucherBackward", ([&] {
		ca_backward_kernel_t<<<blocks, threads>>>(
			dw.data<scalar_t>(), t.data<scalar_t>(), f.data<scalar_t>(), dt.data<scalar_t>(), N, C, H, W);
		ca_backward_kernel_f<<<blocks, threads>>>(
			dw.data<scalar_t>(), t.data<scalar_t>(), f.data<scalar_t>(), df.data<scalar_t>(), N, C, H, W);
    }));

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
		return 0;
    else
		return 1;
}

int CAMapForwardLaucher(const int N, const int C, const int H, const int W,
						const at::Tensor weight, const at::Tensor g,
						at::Tensor out) {
    // Run kernel
    dim3 threads(32, 32);
    dim3 blocks((W+threads.x-1)/threads.x, (H+threads.y-1)/threads.y, C);
	AT_DISPATCH_FLOATING_TYPES_AND_HALF(
		weight.type(), "CAMapLaucherForward", ([&] {
		ca_map_forward_kernel<<<blocks, threads>>>(
			weight.data<scalar_t>(), g.data<scalar_t>(), out.data<scalar_t>(), N, C, H, W);
    }));

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return 0;
    else
        return 1;
}

int CAMapBackwardLaucher(const int N, const int C, const int H, const int W,
						 const at::Tensor dout, const at::Tensor weight, const at::Tensor g,
						 at::Tensor dw, at::Tensor dg) {
    // Run kernel
    dim3 threads(32, 32);
    int d1 = (W+threads.x-1)/threads.x;
    int d2 = (H+threads.y-1)/threads.y;
    int d3 = H+W;
    dim3 blocks(d1, d2, d3);
  	AT_DISPATCH_FLOATING_TYPES_AND_HALF(
		dout.type(), "CAMapLaucherBackward", ([&] {
		ca_map_backward_kernel_w<<<blocks, threads>>>(
			dout.data<scalar_t>(), weight.data<scalar_t>(), g.data<scalar_t>(), dw.data<scalar_t>(), N, C, H, W);
    }));

    d3 = C;
    blocks = dim3(d1, d2, d3);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
		dout.type(), "CAMapLaucherBackward", ([&] {
		ca_map_backward_kernel_g<<<blocks, threads>>>(
			dout.data<scalar_t>(), weight.data<scalar_t>(), g.data<scalar_t>(), dg.data<scalar_t>(), N, C, H, W);
  }));

  // Check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return 0;
  else
    return 1;
}