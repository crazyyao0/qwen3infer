#pragma once
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned short bf16;
float tofloat(bf16 a)
{
	uint32_t bits = ((uint32_t)a) << 16;  // Shift to upper 16 bits
    return *(float*)&bits;
}
float tofloat(float a)
{
	return a;
}
template <typename T> void dumpmat(const char* tblname, T *a, int cols, int startrow, int endrow, int startcol, int endcol, int maxdisplay=6)
{
	auto dumpmatline = [](T* linestart, int startcol, int endcol, int maxdisplay)
	{
		if (endcol-startcol > maxdisplay)
		{
			for(int i=startcol; i<startcol+maxdisplay/2; i++)
				printf("% .6f, ", tofloat(linestart[i]));
			printf(" ... ");
			for(int i=endcol-maxdisplay/2; i<endcol; i++)
				printf("% .6f, ", tofloat(linestart[i]));
		}else{
			for(int i=startcol; i<endcol; i++)
				printf("% .6f, ", tofloat(linestart[i]));
		}
		printf("\n");	
	};

	printf("%s[%d-%d,%d-%d]=\n", tblname, startrow, endrow, startcol, endcol);
	if (endrow-startrow > maxdisplay)
	{
		for(int j=startrow; j<startrow+maxdisplay/2; j++)
			dumpmatline(a+j*cols, startcol, endcol, maxdisplay);
		printf("...,\n");
		for(int j=endrow-maxdisplay/2; j<endrow; j++)
			dumpmatline(a+j*cols, startcol, endcol, maxdisplay);
	}else
	{
		for(int j=startrow; j<endrow; j++)
			dumpmatline(a+j*cols, startcol, endcol, maxdisplay);
	}
	printf("\n");	
}

static inline bf16 floattobf16(float a)
{
	int j = (*(unsigned int*)&a) >> 16;
	if ((*(unsigned int*)&a) & 0x8000)
		j += 1;
	return (bf16)j;
}
static inline float bf16tofloat(bf16 a)
{
	unsigned int j = a << 16;
	return *(float*)&j;
}

void amx_convert_linear_weights(bf16* out, bf16* in, int width, int height)
{
    float* src = (float*)in;    
    float* dst = (float*)out;
    int w = width/2;
    int h = height;
    for(int j=0; j<h; j++)
        for(int i=0; i<w; i++)
            dst[i*h+j] = src[j*w+i];
}

void amx_convert_matmul_weights(bf16* out, bf16* in, int width, int height)
{
	for(int y=0; y<height; y+=2)
	{
		for(int x=0; x<width; x++)
		{
			out[y*width + 2*x] = in[y*width+x];
			out[y*width + 2*x+1] = in[y*width+width+x];
		}
	}
}
