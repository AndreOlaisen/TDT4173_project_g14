#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <stdarg.h>
#include <errno.h>
#include <png.h>

#include "arm_math.h"
#include "nn_util.h"

#define IMG_DIM         32
#define IMG_CH          3
#define IMG_CATEGORIES  10

const char usage[] = "Usage: ./<program> <image.png> [<activation output>]\n";

uint8_t img[IMG_DIM * IMG_DIM * IMG_CH];

/* Category index to names based on torchvision.datasets.CIFAR */
const char *cat_names[] = {
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
};


int nn_forward_pass(uint8_t *img, q7_t *out);

static void fatal_error(const char * message, ...)
{
    va_list args;
    va_start(args, message);
    vfprintf(stderr, message, args);
    va_end(args);
    fprintf(stderr, usage);
    exit(1);
}

static int img_init(const char *filename)
{
    png_structp	png_ptr;
    png_infop info_ptr;
    FILE * fp;
    png_uint_32 width;
    png_uint_32 height;
    int bit_depth;
    int color_type;
    int interlace_method;
    int compression_method;
    int filter_method;
    
    fp = fopen(filename, "rb");
    if (!fp) {
	    fatal_error("Cannot open '%s': %s\n", filename, strerror(errno));
    }

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
	    fatal_error("Cannot create PNG read structure");
    }

    info_ptr = png_create_info_struct(png_ptr);
    if (!png_ptr) {
	    fatal_error("Cannot create PNG info structure");
    }

    png_init_io(png_ptr, fp);
    png_read_png(png_ptr, info_ptr, 0, 0);
    png_get_IHDR(png_ptr, info_ptr, &width, &height, &bit_depth,
	             &color_type, &interlace_method, &compression_method,
	             &filter_method);

    if (width > IMG_DIM || height > IMG_DIM) {
        fatal_error("Image dimensions invalid: %d x %d\n", width, height);
    }
    if (bit_depth != 8) {
        fatal_error("Invalid bit depth: %d\n", bit_depth);
    }
    /* TODO: maybe validate some of the other parameters. */

    png_bytepp rows = png_get_rows(png_ptr, info_ptr);
    int rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    for (int j = 0; j < height; j++) {
	    png_bytep row = rows[j];
	    for (int i = 0; i < rowbytes; i++) {
            img[j * rowbytes + i] = row[i];
        }
    }

    png_destroy_info_struct(png_ptr, &info_ptr);
    png_destroy_read_struct(&png_ptr, NULL, NULL);
    fclose(fp);

    return 0;
}


int main(int argc, char **argv)
{
    q7_t out[IMG_CATEGORIES];

    /* Read image file. */
    if (argc < 2) {
        fatal_error("No file path provided.\n");
    }

    img_init(argv[1]);

    if (argc > 2) {
        nn_dump_path_set(argv[2]);
    }

    /* Perform forward pass and print results. */
    nn_forward_pass(img, out);

    for (int i = 0; i < 10; i++) {
        printf("%s: %hhd\n", cat_names[i], out[i]);
    }
}