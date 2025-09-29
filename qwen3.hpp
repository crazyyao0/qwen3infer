#pragma once
#include "common.hpp"
#include "operator.hpp"
#include "model_weights.hpp"
#include <chrono>

struct layer_weight
{
    bf16* input_layernorm;
    bf16* mlp_down_proj;
    bf16* mlp_gate_proj;
    bf16* mlp_up_proj;
    bf16* post_attention_layernorm;
    bf16* attn_k_norm;
    bf16* attn_k_proj;
    bf16* attn_o_proj;
    bf16* attn_q_norm;
    bf16* attn_q_proj;
    bf16* attn_v_proj;
};

struct model_weight
{
    bf16* embed_tokens;
    struct layer_weight layers[28];
    bf16* norm;
    bf16* rope_cos;
    bf16* rope_sin;
    bf16* score;
};

struct infer_ctx
{
    int seqlen;
    int input_ids[512];
	float output_logits[69];
	bf16* q;    // s*2048
	bf16* k;    // s*1024
	bf16* v;    // s*1024
	bf16* kt;   // s*128
	bf16* vv;   // s*128
	bf16* qkt;  // s*s
	bf16* qktv; // s*2048
	bf16* gate; // s*3072
	bf16* up;   // s*3072
};

struct model_weight mw;
void init_model_weight()
{
    mw.embed_tokens = (bf16*)model_weights_g.get("model.embed_tokens.weight", WEIGHT_SCHEMA_NORMAL);
    mw.norm = (bf16*)model_weights_g.get("model.norm.weight", WEIGHT_SCHEMA_NORMAL);
    mw.rope_cos = (bf16*)model_weights_g.get("position_embeddings.cos", WEIGHT_SCHEMA_NORMAL);
    mw.rope_sin = (bf16*)model_weights_g.get("position_embeddings.sin", WEIGHT_SCHEMA_NORMAL);
    mw.score = (bf16*)model_weights_g.get("score.weight", WEIGHT_SCHEMA_NORMAL);
    for(int i=0;i<28;i++)
    {
        char id[64];
        sprintf(id, "model.layers.%d.input_layernorm.weight", i);
		mw.layers[i].input_layernorm = (bf16*)model_weights_g.get(id, WEIGHT_SCHEMA_NORMAL);
        sprintf(id, "model.layers.%d.post_attention_layernorm.weight", i);
		mw.layers[i].post_attention_layernorm = (bf16*)model_weights_g.get(id, WEIGHT_SCHEMA_NORMAL);
        sprintf(id, "model.layers.%d.mlp.down_proj.weight", i);
        mw.layers[i].mlp_down_proj = (bf16*)model_weights_g.get(id, WEIGHT_SCHEMA_LINEAR);
        sprintf(id, "model.layers.%d.mlp.gate_proj.weight", i);
        mw.layers[i].mlp_gate_proj = (bf16*)model_weights_g.get(id, WEIGHT_SCHEMA_LINEAR);
        sprintf(id, "model.layers.%d.mlp.up_proj.weight", i);
        mw.layers[i].mlp_up_proj = (bf16*)model_weights_g.get(id, WEIGHT_SCHEMA_LINEAR);
        sprintf(id, "model.layers.%d.self_attn.k_norm.weight", i);
        mw.layers[i].attn_k_norm = (bf16*)model_weights_g.get(id, WEIGHT_SCHEMA_NORMAL);
        sprintf(id, "model.layers.%d.self_attn.k_proj.weight", i);
        mw.layers[i].attn_k_proj = (bf16*)model_weights_g.get(id, WEIGHT_SCHEMA_LINEAR);
        sprintf(id, "model.layers.%d.self_attn.o_proj.weight", i);
        mw.layers[i].attn_o_proj = (bf16*)model_weights_g.get(id, WEIGHT_SCHEMA_LINEAR);
        sprintf(id, "model.layers.%d.self_attn.q_norm.weight", i);
        mw.layers[i].attn_q_norm = (bf16*)model_weights_g.get(id, WEIGHT_SCHEMA_NORMAL);
        sprintf(id, "model.layers.%d.self_attn.q_proj.weight", i);
        mw.layers[i].attn_q_proj = (bf16*)model_weights_g.get(id, WEIGHT_SCHEMA_LINEAR);
        sprintf(id, "model.layers.%d.self_attn.v_proj.weight", i);
        mw.layers[i].attn_v_proj = (bf16*)model_weights_g.get(id, WEIGHT_SCHEMA_LINEAR);
    }
}

void rms_layer_normal_inplace(bf16* x, bf16* w, int width, int height, int x_stride, float eps = 1e-6f)
{
    for(int j = 0; j<height; j++)
    {
		bf16* p_line = x + j * x_stride;
		float linesum = 0.0f;
        for(int i = 0; i<width; i+= 16)
        {
            __m512 float32_0 = _mm512_cvtpbh_ps((__m256bh)_mm256_load_si256((__m256i*)(p_line + i)));
			float32_0 = _mm512_mul_ps(float32_0, float32_0);
			linesum += _mm512_reduce_add_ps(float32_0);
        }
		__m512 rms_0 = _mm512_set1_ps(1.0f / sqrtf(linesum/width + eps));
		for(int i = 0; i<width; i+= 16)
        {
            __m512 float32_0 = _mm512_cvtpbh_ps((__m256bh)_mm256_load_si256((__m256i*)(p_line + i)));
			float32_0 = _mm512_mul_ps(float32_0, rms_0);
			__m512 float32_1 = _mm512_cvtpbh_ps((__m256bh)_mm256_load_si256((__m256i*)(w + i)));
			float32_0 = _mm512_mul_ps(float32_0, float32_1);
			_mm256_store_si256((__m256i*)(p_line + i), (__m256i)_mm512_cvtneps_pbh(float32_0));
        }
    }
}

void token_embedding(bf16* out, struct infer_ctx& ctx)
{
    for(int i=0;i<ctx.seqlen;i++)
        avx512_memcpy(out+i*1024, mw.embed_tokens + ctx.input_ids[i]*1024, 2048);
}

void layernormal_and_posemb_inplace(bf16* x, int height, int x_stride, bf16* layer_normal, bf16* cos, bf16* sin)
{
    for(int j = 0; j<height; j++)
    {
		bf16* p_line = x + j * x_stride;
		float linesum = 0.0f;
        for(int i = 0; i<128; i+= 16)
        {
            __m512 float32_0 = _mm512_cvtpbh_ps((__m256bh)_mm256_load_si256((__m256i*)(p_line + i)));
			float32_0 = _mm512_mul_ps(float32_0, float32_0);
			linesum += _mm512_reduce_add_ps(float32_0);
        }
		__m512 rms_0 = _mm512_set1_ps(1.0f / sqrtf(linesum/128 + 1e-6f));
		for(int i = 0; i<128; i+= 16)
        {
            __m512 float32_0 = _mm512_cvtpbh_ps((__m256bh)_mm256_load_si256((__m256i*)(p_line + i)));
			float32_0 = _mm512_mul_ps(float32_0, rms_0);
			__m512 float32_1 = _mm512_cvtpbh_ps((__m256bh)_mm256_load_si256((__m256i*)(layer_normal + i)));
			float32_0 = _mm512_mul_ps(float32_0, float32_1);
			_mm256_store_si256((__m256i*)(p_line + i), (__m256i)_mm512_cvtneps_pbh(float32_0));
        }
		
		for(int i = 0; i<64; i+= 16)
        {
            __m512 a = _mm512_cvtpbh_ps((__m256bh)_mm256_load_si256((__m256i*)(p_line + i)));
            __m512 b = _mm512_cvtpbh_ps((__m256bh)_mm256_load_si256((__m256i*)(p_line + 64 + i)));
            __m512 cos0 = _mm512_cvtpbh_ps((__m256bh)_mm256_load_si256((__m256i*)(cos + j*128 + i)));
            __m512 sin0 = _mm512_cvtpbh_ps((__m256bh)_mm256_load_si256((__m256i*)(sin + j*128 + i)));
			__m512 out = _mm512_sub_ps(_mm512_mul_ps(a, cos0), _mm512_mul_ps(b, sin0));
			_mm256_store_si256((__m256i*)(p_line + i), (__m256i)_mm512_cvtneps_pbh(out));

            cos0 = _mm512_cvtpbh_ps((__m256bh)_mm256_load_si256((__m256i*)(cos + j*128 + 64 + i)));
            sin0 = _mm512_cvtpbh_ps((__m256bh)_mm256_load_si256((__m256i*)(sin + j*128 + 64 + i)));
			out = _mm512_add_ps(_mm512_mul_ps(b, cos0), _mm512_mul_ps(a, sin0));
			_mm256_store_si256((__m256i*)(p_line + 64 + i), (__m256i)_mm512_cvtneps_pbh(out));
        }
    }	
}

void silu_and_mul_inplace(bf16* gate, bf16* up, int width, int height)
{
	for(int y=0; y<height; y++)
	{
		for(int x=0;x<width;x+=32)
		{
			__m256i bf16_0 = _mm256_load_si256((__m256i*)(gate+y*width+x));
			__m512i vindex = _mm512_cvtepu16_epi32 (bf16_0);
			__m512 fp32_0 = _mm512_i32gather_ps (vindex, silu_lookup_table, 4);
			__m512 fp32_1 = _mm512_cvtpbh_ps((__m256bh)_mm256_load_si256((__m256i*)(up+y*width+x)));
			fp32_1 = _mm512_mul_ps(fp32_0, fp32_1);
			_mm256_store_si256((__m256i*)(up+y*width+x), (__m256i)_mm512_cvtneps_pbh(fp32_1));
		}
	}
}

void amx_exp_and_store(bf16* c, int c_stride, float* tmpbuf, __m512 dk)
{
	for (int i=0;i<16;i++)
	{
		__m512 fp32_0 = _mm512_load_ps((__m512*)(tmpbuf + 16*i));
		fp32_0 = _mm512_mul_ps(fp32_0, dk);
		__m512i vindex = _mm512_cvtepu16_epi32((__m256i)_mm512_cvtneps_pbh(fp32_0));
        fp32_0 = _mm512_i32gather_ps (vindex, exp_lookup_table, 4);
		_mm256_store_si256((__m256i*)(c + c_stride * i), (__m256i)_mm512_cvtneps_pbh(fp32_0));
	}
}

void amx_exp_and_store_ltm(bf16* c, int c_stride, float* tmpbuf, __m512 dk)
{
	for (int i=0;i<16;i++)
	{
		__m512 fp32_0 = _mm512_load_ps((__m512*)(tmpbuf + 16*i));
		fp32_0 = _mm512_mul_ps(fp32_0, dk);
		__m512i vindex = _mm512_cvtepu16_epi32((__m256i)_mm512_cvtneps_pbh(fp32_0));
        fp32_0 = _mm512_i32gather_ps (vindex, exp_lookup_table, 4);
		int mask = 0xffff << (i+1);
		fp32_0 = (__m512)_mm512_mask_set1_epi32((__m512i)fp32_0, mask, 0);
		_mm256_store_si256((__m256i*)(c + c_stride * i), (__m256i)_mm512_cvtneps_pbh(fp32_0));
	}
}

// a = (M, K), b = (K, N), C = (M, N), N, M and K should be divisible by 32, c_stride = b_stride
void qwen3_attn_header(struct infer_ctx& ctx, int header_id) 
{
	bf16* a = ctx.q + header_id * 128;
	bf16* b = ctx.kt;
	bf16* c = ctx.qkt;
	int M = ctx.seqlen;
	int N = ctx.seqlen;
	int K = 128;
	int a_stride = 2048;
	int b_stride = ctx.seqlen;
	int c_stride = ctx.seqlen;
	__m512 dk_0 = _mm512_set1_ps(1.0f / sqrtf(128.0f));
	__m512 zero_0 = _mm512_set1_ps(0.0f);

	for(int m=0; m<M; m+=32)
	{
		int n=0;
		// n<m
		for(; n<m; n+=32)
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
			amx_exp_and_store(c + c_stride * (m) + (n), c_stride, tmp_tile_buf, dk_0);
			_tile_stored(1, tmp_tile_buf, 64);
			amx_exp_and_store(c + c_stride * (m) + (n+16), c_stride, tmp_tile_buf, dk_0);
			_tile_stored(2, tmp_tile_buf, 64);
			amx_exp_and_store(c + c_stride * (m+16) + (n), c_stride, tmp_tile_buf, dk_0);
			_tile_stored(3, tmp_tile_buf, 64);
			amx_exp_and_store(c + c_stride * (m+16) + (n+16), c_stride, tmp_tile_buf, dk_0);
		}
		// n=m
		_tile_zero(0);
		_tile_zero(2);
		_tile_zero(3);
		for (int k=0; k<K; k+=32)
		{
			_tile_loadd(4, a + a_stride*(m) + k, a_stride*2);
			_tile_loadd(5, b + b_stride*k + n*2, b_stride*4);
			_tile_loadd(6, a + a_stride*(m+16) + k, a_stride*2);
			_tile_loadd(7, b + b_stride*k + (n)*2 + 32, b_stride*4);
			_tile_dpbf16ps(0, 4, 5);
			_tile_dpbf16ps(2, 6, 5);
			_tile_dpbf16ps(3, 6, 7);
		}
		_tile_stored(0, tmp_tile_buf, 64);
		amx_exp_and_store_ltm(c + c_stride * (m) + (n), c_stride, tmp_tile_buf, dk_0);
		_tile_stored(2, tmp_tile_buf, 64);
		amx_exp_and_store(c + c_stride * (m+16) + (n), c_stride, tmp_tile_buf, dk_0);
		_tile_stored(3, tmp_tile_buf, 64);
		amx_exp_and_store_ltm(c + c_stride * (m+16) + (n+16), c_stride, tmp_tile_buf, dk_0);

		// n>m not necessary
		//for(n=m+32;n<N;n+=32)
		//	for (int i=0;i<32;i++)
		//		_mm512_store_ps((__m256i*)(c + c_stride*(m+i) + n), zero_0);

		// for each line from line m to m+32, calculate xi/sum(x)
		for (int y=m; y<m+32; y++)
		{
			float sum = 0.0f;
			bf16* line = c + c_stride*y;
			for (int x=0; x<=y; x+=16)
			{
				__m512 fp32_0 = _mm512_cvtpbh_ps((__m256bh)_mm256_load_si256((__m256i*)(line+x)));
				sum+=_mm512_reduce_add_ps(fp32_0);
			}
			__m512 rsum = _mm512_set1_ps(1.0f/sum);
			for (int x=0; x<=y; x+=16)
			{
				__m512 fp32_0 = _mm512_cvtpbh_ps((__m256bh)_mm256_load_si256((__m256i*)(line+x)));
				fp32_0 = _mm512_mul_ps(fp32_0, rsum);
				_mm256_store_si256((__m256i*)(line+x), (__m256i)_mm512_cvtneps_pbh(fp32_0));
			}
		}
		
		{
			bf16* a = ctx.qkt;
			int a_stride = ctx.seqlen;
			bf16* b = ctx.vv;
			int b_stride = 128;
			bf16* c = ctx.qktv + header_id * 128;
			int c_stride = 2048;
			for (int n=0;n<128;n+=32)
			{
				_tile_zero(0);
				_tile_zero(1);
				_tile_zero(2);
				_tile_zero(3);
				for (int k=0; k<=m; k+=32)
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
			//dumpmat<bf16>("qkt", ctx.qkt, ctx.seqlen, 0, 16, 0, 16, 16);
			//dumpmat<bf16>("vv", ctx.vv, 256, 0, 16, 0, 16, 16);
			//dumpmat<bf16>("qktv", ctx.qktv, 2048, 0, 16, 0, 16, 16);
		}

	}
}

void qwen3_attn(bf16* x, struct infer_ctx& ctx, struct layer_weight* lw)
{
	//save_bf16_to_file("x.bin", x, 1024, 1024, ctx.seqlen);

	amx_gemm_bf16_fma_split(x, lw->attn_q_proj, ctx.q, ctx.seqlen, 2048, 1024, 1024, 2048, 512, 1024, 512);
	//save_bf16_to_file("q.bin", ctx.q, 2048, 2048, ctx.seqlen);
	//dumpmat<bf16>("q", ctx.q, 2048, 24, 32, 48, 56, 8);
	for (int i=0; i<2048; i+=128)
		layernormal_and_posemb_inplace(ctx.q+i, ctx.seqlen, 2048, lw->attn_q_norm, mw.rope_cos, mw.rope_sin);
	//save_bf16_to_file("q_emb.bin", ctx.q, 2048, 2048, ctx.seqlen);

	amx_gemm_bf16_fma_split(x, lw->attn_k_proj, ctx.k, ctx.seqlen, 1024, 1024, 1024, 1024, 512, 1024, 512);
	//save_bf16_to_file("k.bin", ctx.k, 1024, 1024, ctx.seqlen);
	for (int i=0; i<1024; i+=128)
		layernormal_and_posemb_inplace(ctx.k+i, ctx.seqlen, 1024, lw->attn_k_norm, mw.rope_cos, mw.rope_sin);
	//save_bf16_to_file("k_emb.bin", ctx.k, 1024, 1024, ctx.seqlen);
	//dumpmat<bf16>("k_emb", ctx.k, 1024, 264, 272, 16, 24, 8);
	
	amx_gemm_bf16_fma_split(x, lw->attn_v_proj, ctx.v, ctx.seqlen, 1024, 1024, 1024, 1024, 512, 1024, 512);
	//save_bf16_to_file("v.bin", ctx.v, 1024, 1024, ctx.seqlen);

    //auto t1 = std::chrono::high_resolution_clock::now();
	for (int i=0; i<8; i++)
	{
		amx_matrix_tranpose(ctx.k+i*128, 128, ctx.seqlen, 1024, ctx.kt, ctx.seqlen);
		amx_matrix_convert(ctx.v+i*128, 128, ctx.seqlen, 1024, ctx.vv, 128);
		qwen3_attn_header(ctx, i*2);
		qwen3_attn_header(ctx, i*2+1);
	}
    //auto t2 = std::chrono::high_resolution_clock::now() - t1;
    //printf("attn timespan: %fms\n", t2.count()/1000000.0f);

	//save_bf16_to_file("qktv.bin", ctx.qktv, 2048, 2048, ctx.seqlen);
	//dumpmat<bf16>("qktv", ctx.qktv, 2048, 0, 8, 0, 8, 8);
	//dumpmat<bf16>("attn_o_proj", lw->attn_o_proj, 2048, 0, 1024, 0, 2048, 8);
	amx_gemm_bf16_fma_split(ctx.qktv, lw->attn_o_proj, x, ctx.seqlen, 1024, 2048, 2048, 1024, 512, 1024, 512);
	//dumpmat<bf16>("x_attn_output", x, 1024, 0, ctx.seqlen, 0, 1024, 8);
}

/*
// 
	test1(x, mw.layers[0].mlp_up_proj);
	test2(x, mw.layers[0].mlp_up_proj);
	test3(x, mw.layers[0].mlp_up_proj, 512, 1024, 512);

	test3 splitMNK: 512 1024 512 timespan: 1.550136ms

	(512,1024)(1024,3072) -> best MNK = 512,1024,512 (2.249047ms)
	(512,1024)(1024,2048) -> best MNK = 512,1024,512 (1.550136ms)
	(512,1024)(1024,1024) -> best MNK = 512,1024,512 (0.728590ms)
*/

int qwen_model_infer(struct infer_ctx& ctx)
{
    bf16* x =  (bf16*)aligned_alloc(64, ctx.seqlen*2048);
	bf16* xr = (bf16*)aligned_alloc(64, ctx.seqlen*2048);
	ctx.q = (bf16*)aligned_alloc(64, ctx.seqlen*4096);
	ctx.k = (bf16*)aligned_alloc(64, ctx.seqlen*2048);
	ctx.v = (bf16*)aligned_alloc(64, ctx.seqlen*2048);
	ctx.kt = (bf16*)aligned_alloc(64, ctx.seqlen*256);  
	ctx.vv = (bf16*)aligned_alloc(64, ctx.seqlen*256);  
	ctx.qkt = (bf16*)aligned_alloc(64, ctx.seqlen*ctx.seqlen*2);
	ctx.qktv = (bf16*)aligned_alloc(64, ctx.seqlen*4096);
	ctx.gate = (bf16*)aligned_alloc(64, ctx.seqlen*6144);
	ctx.up = (bf16*)aligned_alloc(64, ctx.seqlen*6144);
	avx512_memset(ctx.qkt, 0, ctx.seqlen*ctx.seqlen*2);

    token_embedding(x, ctx);
	avx512_memcpy(xr, x, ctx.seqlen*2048);  // xr = x
	//save_bf16_to_file("embedding.bin", x, 1024, 1024, ctx.seqlen);
	for(int i=0;i<28;i++)
    {
		struct layer_weight* lw = mw.layers + i;
		rms_layer_normal_inplace(x, lw->input_layernorm, 1024, ctx.seqlen, 1024);
		qwen3_attn(x, ctx, lw);

		avx512_sum_and_copy_inplace(xr, x, ctx.seqlen*2048);  // xr, x = xr + x
		rms_layer_normal_inplace(x, lw->post_attention_layernorm, 1024, ctx.seqlen, 1024);
		
		amx_gemm_bf16_fma_split(x, lw->mlp_gate_proj, ctx.gate, ctx.seqlen, 3072, 1024, 1024, 3072, 512, 1024, 512);
		amx_gemm_bf16_fma_split(x, lw->mlp_up_proj, ctx.up, ctx.seqlen, 3072, 1024, 1024, 3072, 512, 1024, 512);
		silu_and_mul_inplace(ctx.gate, ctx.up, 3072, ctx.seqlen);
		amx_gemm_bf16_fma_split(ctx.up, lw->mlp_down_proj, x, ctx.seqlen, 1024, 3072, 3072, 1024, 512, 1024, 512);
		
		avx512_sum_and_copy_inplace(xr, x, ctx.seqlen*2048);  // xr, x = xr + x
    }
	rms_layer_normal_inplace(x, mw.norm, 1024, ctx.seqlen, 1024);
	amx_gemv_bf16_fp32(x, mw.score, ctx.output_logits, 69, 1024, 1024);

	free(x);
	free(xr);
	free(ctx.q);
	free(ctx.k);
	free(ctx.v);
	free(ctx.qkt);
	free(ctx.qktv);
	free(ctx.gate);
	free(ctx.up);
    return 0;
}
