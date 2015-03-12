#include "bmp.h"

/* initialize properties of given Bitmap header with provided size*/
void init_bmp_header(bmp_header_t *header, uint32_t width, uint32_t height)
{
    uint32_t bmp_size = 54;
    uint32_t raw_row_size = width * 3;
    uint32_t raw_pixel_data_size;
    uint32_t rest = raw_row_size % 4;
    if(rest)
    {
        raw_row_size += 4 - rest;
    }
    raw_pixel_data_size = raw_row_size * height;

    header->bm = 'B' + ('M' << 8);
    header->size = bmp_size + raw_pixel_data_size;
    header->reserved = 0;
    header->pixel_offset = bmp_size;
    header->bitmap_info_header_size = 40;
    header->pixel_width = width;
    header->pixel_height = height;
    header->color_planes_num = 1;
    header->bits_per_pixel = 24;
    header->compression_enabled = 0;
    header->pixel_data_raw_size = raw_pixel_data_size;
    header->horiz_res = 2835;
    header->vert_res = 2835;
    header->colors_num = 0;
    header->important_colors = 0;
}

/* save given RGB color array into file given by handle fout */
void write_bmp(FILE *fout, float *colors, uint32_t width, uint32_t height)
{
    int padding = (width * 3) % 4;
    if(padding)
        padding = 4 - padding;

    uint8_t header[54];

    init_bmp_header((bmp_header_t *) header, width, height);
    fwrite(header, 1, 54, fout);

    // each line
    for(uint32_t y = 0; y < height; y++)
    {
        for(uint32_t x = 0; x < width; x++)
        {
            float *elem = &colors[(y * width + x) * 3];
            int c = (int) (elem[2] * 255.0); // B
            fputc(c, fout);
            c = (int) (elem[1] * 255.0); // G
            fputc(c, fout);
            c = (int) (elem[0] * 255.0); // R
            fputc(c, fout);
        }

        for(uint32_t x = 0; x < padding; x++)
        {
            fputc(0, fout);
        }
    }
}
