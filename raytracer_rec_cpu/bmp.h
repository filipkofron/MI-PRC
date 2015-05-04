#ifndef BMP_H
#define BMP_H

#include <inttypes.h>
#include <stdio.h>

#ifdef _MSC_VER
#pragma pack(push,1)

/* strictly byte aligned header of the bitmap file */
/* Little Endian compatible only */
struct bmp_header_t
{
	uint16_t bm;
	uint32_t size;
	uint32_t reserved;
	uint32_t pixel_offset;
	uint32_t bitmap_info_header_size;
	uint32_t pixel_width;
	uint32_t pixel_height;
	uint16_t color_planes_num;
	uint16_t bits_per_pixel;
	uint32_t compression_enabled;
	uint32_t pixel_data_raw_size;
	uint32_t horiz_res;
	uint32_t vert_res;
	uint32_t colors_num;
	uint32_t important_colors;
};

#pragma pack(pop)
#else

/* strictly byte aligned header of the bitmap file */
/* Little Endian compatible only */
struct bmp_header_t
{
	uint16_t bm;
	uint32_t size;
	uint32_t reserved;
	uint32_t pixel_offset;
	uint32_t bitmap_info_header_size;
	uint32_t pixel_width;
	uint32_t pixel_height;
	uint16_t color_planes_num;
	uint16_t bits_per_pixel;
	uint32_t compression_enabled;
	uint32_t pixel_data_raw_size;
	uint32_t horiz_res;
	uint32_t vert_res;
	uint32_t colors_num;
	uint32_t important_colors;
} __attribute__((packed));

#endif

/* initialize properties of given Bitmap header with provided size*/
void init_bmp_header(bmp_header_t *header, uint32_t width, uint32_t height);

/* save given RGB color array into file given by handle fout */
void write_bmp(FILE *fout, float *colors, uint32_t width, uint32_t height);

#endif
