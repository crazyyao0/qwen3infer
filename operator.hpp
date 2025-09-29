#pragma once
#include <sys/syscall.h>      /* Definition of SYS_* constants */
#include <unistd.h>
#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <math.h>

void *avx512_memcpy(void *dest, const void *src, size_t n) {
    for (size_t i = 0; i < n; i += 64) {
        __m512i data = _mm512_load_si512((__m512i *)((unsigned char*)src + i));
        _mm512_store_si512((__m512i *)((unsigned char*)dest + i), data);
    }
    return dest;
}
void* avx512_sum_and_copy_inplace(void *dest, const void *src, size_t n) {
    for (size_t i = 0; i < n; i += 32) {
		__m512 src_0 = _mm512_cvtpbh_ps((__m256bh)_mm256_load_si256((__m256i*)((unsigned char*)src + i)));
		__m512 dst_0 = _mm512_cvtpbh_ps((__m256bh)_mm256_load_si256((__m256i*)((unsigned char*)dest + i)));
		__m256bh fp32_0 = _mm512_cvtneps_pbh(_mm512_add_ps(src_0, dst_0));
		_mm256_store_si256((__m256i*)((unsigned char*)src + i), (__m256i)fp32_0);
		_mm256_store_si256((__m256i*)((unsigned char*)dest + i), (__m256i)fp32_0);
    }
    return dest;
}

void *avx512_memset(void *dest, int c, size_t n) {
    __m512i data = _mm512_set1_epi32(c);
    for (size_t i = 0; i < n; i += 64) {
        _mm512_store_si512((__m512i *)((unsigned char*)dest + i), data);
    }
    return dest;
}

struct __tile_config
{
  uint8_t palette_id;
  uint8_t start_row;
  uint16_t reserved_0[7];
  uint16_t colsb[8];
  uint16_t reserved_1[8];
  uint8_t rows[8];
  uint8_t reserved_2[8];
};

alignas(64) unsigned char tmp_float_result_buf[6*1024*1024];
alignas(64) float tmp_tile_buf[256];
alignas(64) float silu_lookup_table[65536];
alignas(64) float exp_lookup_table[65536];
alignas(64) struct __tile_config tileconfigs[] = {{ 1, 0, {0,0,0,0,0,0,0}, {64,64,64,64,64,64,64,64}, {0,0,0,0,0,0,0,0}, {16,16,16,16,16,16,16,16}, {0,0,0,0,0,0,0,0}}};

void initamx_tiles()
{
	#define ARCH_REQ_XCOMP_PERM 0x1023
	#define XFEATURE_XTILEDATA 18
	syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
	_tile_loadconfig (tileconfigs);
	for (int i=0 ;i<65536; i++)
	{
		float x = bf16tofloat(i);
		silu_lookup_table[i] = x / (1.0f + expf(-x));
		exp_lookup_table[i] = expf(x);
	}
}

void amx_store_fp32_as_bf16(bf16* c, int c_stride, float* tmpbuf)
{
	for (int i=0;i<16;i++)
	{
		__m512 fp32_0 = _mm512_load_ps((__m512*)(tmpbuf + 16*i));
		_mm256_store_si256((__m256i*)(c + c_stride * i), (__m256i)_mm512_cvtneps_pbh(fp32_0));
	}
}

void amx_store_fp32_add_bf16(bf16* c, int c_stride, float* tmpbuf)
{
	for (int i=0;i<16;i++)
	{
		bf16* dst = c + c_stride * i;
		__m512 fp32_0 = _mm512_load_ps((__m512*)(tmpbuf + 16*i));
		__m256bh bf16_0 = (__m256bh)_mm256_load_si256((__m256i*)(dst));
		__m512 fp32_1 = _mm512_add_ps(fp32_0, _mm512_cvtpbh_ps(bf16_0));
		_mm256_store_si256((__m256i*)(dst), (__m256i)_mm512_cvtneps_pbh(fp32_1));
	}
}

// a = (M, K), b = (K, N), C = (M, N), N, M and K should be divisible by 32, c_stride = b_stride
// first and last K block
void amx_gemm_bf16(bf16* a, bf16* b, bf16* c, int M, int N, int K, int a_stride, int b_stride) 
{
	int c_stride = b_stride;
	for(int m=0; m<M; m+=32)
	for(int n=0; n<N; n+=32)
	{
		_tile_zero(0);
		_tile_zero(1);
		_tile_zero(2);
		_tile_zero(3);
		for (int k=0; k<K; k+=32)
		{
			_tile_loadd(4, a + a_stride*(m) + k, a_stride*2);
			_tile_loadd(5, b + b_stride*k + n*2, b_stride*4);
			_tile_loadd(6, a + a_stride*(m+16) + k, a_stride*2);
			_tile_loadd(7, b + b_stride*k + (n)*2 + 32, b_stride*4);
			_tile_dpbf16ps(0, 4, 5);
			_tile_dpbf16ps(1, 4, 7);
			_tile_dpbf16ps(2, 6, 5);
			_tile_dpbf16ps(3, 6, 7);
		}
		_tile_stored(0, tmp_tile_buf, 64);
		amx_store_fp32_as_bf16(c + c_stride * (m) + (n), c_stride, tmp_tile_buf);
		_tile_stored(1, tmp_tile_buf, 64);
		amx_store_fp32_as_bf16(c + c_stride * (m) + (n+16), c_stride, tmp_tile_buf);
		_tile_stored(2, tmp_tile_buf, 64);
		amx_store_fp32_as_bf16(c + c_stride * (m+16) + (n), c_stride, tmp_tile_buf);
		_tile_stored(3, tmp_tile_buf, 64);
		amx_store_fp32_as_bf16(c + c_stride * (m+16) + (n+16), c_stride, tmp_tile_buf);
	}
}

// a = (M, K), b = (K, N), C = (M, N), N, M and K should be divisible by 32, c_stride = b_stride
// first K block. store to temp buffer
void amx_gemm_bf16_fma1(bf16* a, bf16* b, float* cc, int M, int N, int K, int a_stride, int b_stride) 
{
	int c_stride = b_stride;
	for(int m=0; m<M; m+=32)
	for(int n=0; n<N; n+=32)
	{
		_tile_zero(0);
		_tile_zero(1);
		_tile_zero(2);
		_tile_zero(3);
		for (int k=0; k<K; k+=32)
		{
			_tile_loadd(4, a + a_stride*(m) + k, a_stride*2);
			_tile_loadd(5, b + b_stride*k + n*2, b_stride*4);
			_tile_loadd(6, a + a_stride*(m+16) + k, a_stride*2);
			_tile_loadd(7, b + b_stride*k + (n)*2 + 32, b_stride*4);
			_tile_dpbf16ps(0, 4, 5);
			_tile_dpbf16ps(1, 4, 7);
			_tile_dpbf16ps(2, 6, 5);
			_tile_dpbf16ps(3, 6, 7);
		}
		_tile_stored(0, cc + c_stride * (m) + (n), c_stride*4);
		_tile_stored(1, cc + c_stride * (m) + (n+16), c_stride*4);
		_tile_stored(2, cc + c_stride * (m+16) + (n), c_stride*4);
		_tile_stored(3, cc + c_stride * (m+16) + (n+16), c_stride*4);
	}
}

// a = (M, K), b = (K, N), C = (M, N), N, M and K should be divisible by 32, c_stride = b_stride, 
// following K block. load from tempbuf and store to tempbuf
void amx_gemm_bf16_fma2(bf16* a, bf16* b, float* cc, int M, int N, int K, int a_stride, int b_stride)
{
	int c_stride = b_stride;
	for(int m=0; m<M; m+=32)
	for(int n=0; n<N; n+=32)
	{
		int oc0 = c_stride * (m) + (n);
		int oc1 = c_stride * (m) + (n+16);
		int oc2 = c_stride * (m+16) + (n);
		int oc3 = c_stride * (m+16) + (n+16);

		_tile_loadd(0, cc + oc0, c_stride*4);
		_tile_loadd(1, cc + oc1, c_stride*4);
		_tile_loadd(2, cc + oc2, c_stride*4);
		_tile_loadd(3, cc + oc3, c_stride*4);
		for (int k=0; k<K; k+=32)
		{
			_tile_loadd(4, a + a_stride*(m) + k, a_stride*2);
			_tile_loadd(5, b + b_stride*k + n*2, b_stride*4);
			_tile_loadd(6, a + a_stride*(m+16) + k, a_stride*2);
			_tile_loadd(7, b + b_stride*k + (n)*2 + 32, b_stride*4);
			_tile_dpbf16ps(0, 4, 5);
			_tile_dpbf16ps(1, 4, 7);
			_tile_dpbf16ps(2, 6, 5);
			_tile_dpbf16ps(3, 6, 7);
		}
		_tile_stored(0, cc + oc0, c_stride*4);
		_tile_stored(1, cc + oc1, c_stride*4);
		_tile_stored(2, cc + oc2, c_stride*4);
		_tile_stored(3, cc + oc3, c_stride*4);
	}
}

// a = (M, K), b = (K, N), C = (M, N), N, M and K should be divisible by 32, c_stride = b_stride, 
// last K block. load from tempbuf and store to c
void amx_gemm_bf16_fma3(bf16* a, bf16* b, bf16* c, float* cc, int M, int N, int K, int a_stride, int b_stride)
{
	int c_stride = b_stride;
	for(int m=0; m<M; m+=32)
	for(int n=0; n<N; n+=32)
	{
		int oc0 = c_stride * (m) + (n);
		int oc1 = c_stride * (m) + (n+16);
		int oc2 = c_stride * (m+16) + (n);
		int oc3 = c_stride * (m+16) + (n+16);

		_tile_loadd(0, cc + oc0, c_stride*4);
		_tile_loadd(1, cc + oc1, c_stride*4);
		_tile_loadd(2, cc + oc2, c_stride*4);
		_tile_loadd(3, cc + oc3, c_stride*4);
		for (int k=0; k<K; k+=32)
		{
			_tile_loadd(4, a + a_stride*(m) + k, a_stride*2);
			_tile_loadd(5, b + b_stride*k + n*2, b_stride*4);
			_tile_loadd(6, a + a_stride*(m+16) + k, a_stride*2);
			_tile_loadd(7, b + b_stride*k + (n)*2 + 32, b_stride*4);
			_tile_dpbf16ps(0, 4, 5);
			_tile_dpbf16ps(1, 4, 7);
			_tile_dpbf16ps(2, 6, 5);
			_tile_dpbf16ps(3, 6, 7);
		}

		_tile_stored(0, tmp_tile_buf, 64);
		amx_store_fp32_as_bf16(c + oc0, c_stride, tmp_tile_buf);
		_tile_stored(1, tmp_tile_buf, 64);
		amx_store_fp32_as_bf16(c + oc1, c_stride, tmp_tile_buf);
		_tile_stored(2, tmp_tile_buf, 64);
		amx_store_fp32_as_bf16(c + oc2, c_stride, tmp_tile_buf);
		_tile_stored(3, tmp_tile_buf, 64);
		amx_store_fp32_as_bf16(c + oc3, c_stride, tmp_tile_buf);
	}
}

void amx_gemm_bf16_split(bf16* a, bf16* b, bf16* c, int M, int N, int K, int a_stride, int b_stride, int splitM, int splitN)
{
	int c_stride = b_stride;
	for (int n=0; n<N; n+=splitN)
	for (int m=0; m<M; m+=splitM)
	{
		amx_gemm_bf16(	a + m*a_stride,
						b + n*2,
						c + m*c_stride + n,
						splitM, splitN, K, a_stride, b_stride);
	}
}

void amx_gemm_bf16_fma_split(bf16* a, bf16* b, bf16* c, int M, int N, int K, int a_stride, int b_stride, int splitM, int splitN, int splitK)
{
	int m,n,k;
	float* cc = (float*)tmp_float_result_buf;
	//int c_stride = b_stride
	if(splitK>=K) {
		for (n=0; n<N; n+=splitN)
		for (m=0; m<M; m+=splitM)
		{
			amx_gemm_bf16(	a + m*a_stride,
								b + n*2,
								c + m*b_stride + n,
								splitM, splitN, K, a_stride, b_stride);
		}
	}
	else{
		for (n=0; n<N; n+=splitN)
		for (m=0; m<M; m+=splitM)
		{
			amx_gemm_bf16_fma1(	a + m*a_stride,
								b + n*2,
								cc + m*b_stride + n,
								splitM, splitN, splitK, a_stride, b_stride);
		}
		
		for (k=splitK; k + splitK < K; k+=splitK)
		for (n=0; n<N; n+=splitN)
		for (m=0; m<M; m+=splitM)
		{
			amx_gemm_bf16_fma2(	a + m*a_stride + k,
								b + k*b_stride + n*2,
								cc + m*b_stride + n,
								splitM, splitN, splitK, a_stride, b_stride);
		}

		for (n=0; n<N; n+=splitN)
		for (m=0; m<M; m+=splitM)
		{
			amx_gemm_bf16_fma3(	a + m*a_stride + k,
								b + k*b_stride + n*2,
								c + m*b_stride + n,
								cc + m*b_stride + n,
								splitM, splitN, splitK, a_stride, b_stride);
		}
	}
}

void amx_gemv_bf16_fp32(bf16* a, bf16* b, float* c, int N, int K, int b_stride)
{
	for(int n=0;n<N;n++)
	{
		float linesum = 0.0f;
		for(int k=0;k<K;k+=32)
		{
			__m512 v0 = _mm512_cvtpbh_ps((__m256bh)_mm256_load_si256((__m256i*)(a + k)));
			__m512 v1 = _mm512_cvtpbh_ps((__m256bh)_mm256_load_si256((__m256i*)(b + b_stride*n + k)));
			linesum += _mm512_reduce_add_ps(_mm512_mul_ps(v0, v1));
		}
		c[n] = linesum;
	}
}

void amx_matrix_tranpose(bf16* in, int in_width, int in_height, int in_stride, bf16* out, int out_stride)
{
	__m512i vindex = _mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
	vindex = _mm512_mullo_epi32(vindex, _mm512_set1_epi32(in_stride*2));

	for (int y=0; y<in_height; y+=16)
	{
		for (int x=0; x<in_width; x+=2)
		{
			__m512i a = _mm512_i32gather_epi32(vindex, in + x + y*in_stride, 1);
			_mm512_store_epi32((out + x*out_stride + y*2), a);
		}
	}
}

void amx_matrix_convert(bf16* in, int in_width, int in_height, int in_stride, bf16* out, int out_stride)
{
	for(int y=0;y<in_height;y+=2)
	{
		for(int x=0; x<in_width; x+=32)
		{
			__m512bh a = (__m512bh)_mm512_load_si512(in + y*in_stride + x);
			__m512bh b = (__m512bh)_mm512_load_si512(in + (y+1)*in_stride + x);
			__m512bh ab_1357 = (__m512bh)_mm512_unpacklo_epi16((__m512i)a, (__m512i)b);
			__m512bh ab_2468 = (__m512bh)_mm512_unpackhi_epi16((__m512i)a, (__m512i)b);
			__m512bh ab_1324 = (__m512bh)_mm512_shuffle_i64x2((__m512i)ab_1357, (__m512i)ab_2468, _MM_PERM_BABA);
			__m512bh ab_1234 = (__m512bh)_mm512_shuffle_i64x2((__m512i)ab_1324, (__m512i)ab_1324, _MM_PERM_DBCA);
			_mm512_store_si512(out+y*out_stride+x*2, (__m512i)ab_1234);
			__m512bh ab_5768 = (__m512bh)_mm512_shuffle_i64x2((__m512i)ab_1357, (__m512i)ab_2468, _MM_PERM_DCDC);
			__m512bh ab_5678 = (__m512bh)_mm512_shuffle_i64x2((__m512i)ab_5768, (__m512i)ab_5768, _MM_PERM_DBCA);
			_mm512_store_si512(out+y*out_stride+x*2+32, (__m512i)ab_5678);
		}
	}
}

void convert_bf16_to_fp32(bf16* in, float* out, int in_stride, int out_stride, int width, int height)
{
	for(int y=0;y<height;y+=1)
	{
		for(int x=0; x<width; x+=16)
		{
			__m512 fp32_0 = _mm512_cvtpbh_ps((__m256bh)_mm256_load_si256((__m256i*)(in + y*in_stride + x)));
			_mm512_store_ps(out+y*out_stride + x, fp32_0);
		}
	}
}

void save_bf16_to_file(const char* filename, bf16* in, int in_stride, int width, int height)
{
	float *out = (float*)tmp_float_result_buf;
	convert_bf16_to_fp32(in, out, in_stride, in_stride, width, height);
	FILE* fp = fopen(filename, "wb");
	for(int y=0;y<height;y+=1)
		fwrite(out+y*in_stride, width, 4, fp);
	fclose(fp);
}