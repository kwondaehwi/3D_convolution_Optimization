#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>
#include <immintrin.h>
#include <unistd.h>
#include <ctype.h>
#include <inttypes.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

#define MASK_WIDTH 5
#define TILE_SIZE 4

typedef struct workerparam_{
    unsigned int inchannels; 
    unsigned int outchannels;
    unsigned int inwidth; 
    unsigned int outwidth; 
    unsigned int kwidth;
    unsigned int inheight;
    unsigned int outheight; 
    unsigned int kheight;
    unsigned int indepth; 
    unsigned int outdepth; 
    unsigned int kdepth;
    unsigned int stride; 
    unsigned int th;
    unsigned int tid;
    unsigned int t_num;
    float* inmap_ptr; 
    float* outmap_ptr; 
    float* kernel_ptr;
} workerparam;



void verification(const float *N, const float *M, const float *P, int Rows, int Columns, int Height, int k_dim) {
    int r, c, h, k_r, k_c, k_h;
    int row_i, col_i, height_i;
    bool equal;
    float* results;

    results = (float*)malloc(Height * Rows * Columns * sizeof(float));
    memset(results, 0, Height * Rows * Columns * sizeof(float));
    long long start = __rdtsc();
    for(h = 0; h < Height; h++){
        for (r = 0; r < Rows; r++) {
            for (c = 0; c < Columns; c++) {
                for(k_h = 0; k_h < k_dim; k_h++){
                    for (k_r = 0; k_r < k_dim; k_r++) {
                        for (k_c = 0; k_c < k_dim; k_c++) {
                            row_i = r - ((k_dim - 1) / 2) + k_r;
                            col_i = c - ((k_dim - 1) / 2) + k_c;
                            height_i = h - ((k_dim - 1) / 2) + k_h;
                            if ((row_i >= 0) && (row_i < Rows) && (col_i >= 0) && (col_i < Columns) && (height_i >= 0) && (height_i < Height)) {
                                results[h*Columns*Rows+r*Columns + c] += (M[k_h*k_dim*k_dim + k_r*k_dim + k_c] * N[height_i*Columns*Rows + row_i*Columns + col_i]);
                            }
                        }
                    }
                }
            }
        }
    }
    long long end = __rdtsc();
    equal = true;
    for (int i = 0; i < Height * Rows * Columns && equal; i++) {
        if (abs(results[i] - P[i]) >= 0.001f) {
            equal = false;
            //printf("cuda: %f cpu: %f\n", results[i], P[i]);
            printf("NOT EQUAL!\n");
        }
    }

    if (equal) {
        printf("Results are equal!\n");
        printf("Sequential execution time: %llu\n", end - start);
    }
    else {
        printf("Results are NOT equal!\n");
    }

    free(results);
    return;
}
__m256i masktable_m256(int k);
__m128i masktable_m128(int k);
__m128 _mm256d_sum(__m256d hi, __m256d lo);

void* workerThread(void* workerpara);

void convolution_3d(unsigned int inchannels, unsigned int outchannels, 
                    unsigned int inwidth, unsigned int outwidth, unsigned int kwidth,
                    unsigned int inheight, unsigned int outheight, unsigned int kheight,
                    unsigned int indepth, unsigned int outdepth, unsigned int kdepth,
                    unsigned stride, unsigned int th, 
                    float* inmap_ptr, float* outmap_ptr, float* kernel_ptr);

void zero_padding_3d(unsigned int channels,
                     unsigned int inwidth, unsigned int inheight, unsigned int indepth,
                     unsigned int outwidth, unsigned int outheight, unsigned int outdepth,
                     unsigned int th,
                     unsigned int pad_left, unsigned int pad_right,
                     unsigned int pad_top, unsigned int pad_bottom,
                     unsigned int pad_front, unsigned int pad_rear,
                     float* inmap_ptr, float* outmap_ptr);

int main(int argc,char **argv){
    int check;
    extern char *optarg;
    extern int optind;
    check = getopt(argc, argv, "abcde :");
    FILE* input_file;
    FILE* output_file;
    FILE* kernel_file;
    int i_x,i_y,i_z;
    int o_x,o_y,o_z;
    int k_dim;
    float *input,*input_padd,*output,*kernel,*result;
    switch (check)
    {
        case 'a':
            input_file = fopen("./sample/test1/input.txt", "r");
            if(input_file==NULL) printf("input파일 열기 실패\n");
            output_file = fopen("./sample/test1/output.txt","r");
            if(output_file==NULL) printf("output파일 열기 실패\n");
            kernel_file = fopen("./sample/test1/kernel.txt","r");
            if(kernel_file==NULL) printf("kernel파일 열기 실패\n");
            break;
        case 'b':
            input_file = fopen("./sample/test2/input.txt", "r");
            if(input_file==NULL) printf("input파일 열기 실패\n");
            output_file = fopen("./sample/test2/output.txt","r");
            if(output_file==NULL) printf("output파일 열기 실패\n");
            kernel_file = fopen("./sample/test2/kernel.txt","r");
            if(kernel_file==NULL) printf("kernel파일 열기 실패\n");
            break;
        case 'c':
            input_file = fopen("./sample/test3/input.txt", "r");
            if(input_file==NULL) printf("input파일 열기 실패\n");
            output_file = fopen("./sample/test3/output.txt","r");
            if(output_file==NULL) printf("output파일 열기 실패\n");
            kernel_file = fopen("./sample/test3/kernel.txt","r");
            if(kernel_file==NULL) printf("kernel파일 열기 실패\n");
            break;
        case 'd':
            input_file = fopen("./sample/test4/input.txt", "r");
            if(input_file==NULL) printf("input파일 열기 실패\n");
            output_file = fopen("./sample/test4/output.txt","r");
            if(output_file==NULL) printf("output파일 열기 실패\n");
            kernel_file = fopen("./sample/test4/kernel.txt","r");
            if(kernel_file==NULL) printf("kernel파일 열기 실패\n");
            break;
        case 'e':
            input_file = fopen("./sample/test5/input.txt", "r");
            if(input_file==NULL) printf("input파일 열기 실패\n");
            output_file = fopen("./sample/test5/output.txt","r");
            if(output_file==NULL) printf("output파일 열기 실패\n");
            kernel_file = fopen("./sample/test5/kernel.txt","r");
            if(kernel_file==NULL) printf("kernel파일 열기 실패\n");
            break;
        default:
            printf("Wrong argument ./conv3d -[a...e]");
            break;
    }
    
    fscanf(input_file,"%d %d %d",&i_z,&i_y,&i_x);
    input=(float*)malloc(sizeof(float)*i_x*i_y*i_z);
    
    for(int i=0;i<i_x*i_y*i_z;i++){
        fscanf(input_file,"%f",&input[i]);
    }

    fscanf(kernel_file,"%d",&k_dim);
    kernel=(float*)malloc(sizeof(float)*k_dim*k_dim*k_dim);
    for(int i=0;i<k_dim*k_dim*k_dim;i++){
        fscanf(kernel_file,"%f",&kernel[i]);
    }

    fscanf(output_file,"%d %d %d",&o_z,&o_y,&o_x);
    output=(float*)malloc(sizeof(float)*o_x*o_y*o_z);
    for(int i=0;i<o_x*o_y*o_z;i++){
        fscanf(output_file,"%f",&output[i]);
    }

    printf("input: %d %d %d \n", i_x, i_y, i_z);
    printf("kernel: %d\n", k_dim);
    printf("output: %d %d %d\n", o_x, o_y, o_z);

    unsigned int intput_p_x = i_x + k_dim - 1;
    unsigned int intput_p_y = i_y + k_dim - 1;
    unsigned int intput_p_z = i_z + k_dim - 1;
    
    // zero_padding_3d(unsigned int channels,
    //                  unsigned int inwidth, unsigned int inheight, unsigned int indepth,
    //                  unsigned int outwidth, unsigned int outheight, unsigned int outdepth,
    //                  unsigned int th,
    //                  unsigned int pad_left, unsigned int pad_right,
    //                  unsigned int pad_top, unsigned int pad_bottom,
    //                  unsigned int pad_front, unsigned int pad_rear,
    //                  float* inmap_ptr, float* outmap_ptr)

    input_padd=(float*)malloc(sizeof(float)*intput_p_x*intput_p_y*intput_p_z);
    zero_padding_3d(1,
                     i_x, i_y, i_z,
                     intput_p_x,intput_p_y,intput_p_z,
                     0,
                     (k_dim-1)/2, (k_dim-1)/2,
                     (k_dim-1)/2, (k_dim-1)/2,
                     (k_dim-1)/2, (k_dim-1)/2,
                     input, input_padd);

    result = (float*)malloc(sizeof(float)*o_x*o_y*o_z);

    unsigned int outwidth = (intput_p_x - k_dim ) + 1;
    unsigned int outheight =(intput_p_y - k_dim ) + 1;
    unsigned int outdepth = (intput_p_z - k_dim ) + 1;
    
    printf("output size:%d %d %d\n",outwidth,outheight,outdepth);
    // convolution_3d(unsigned int inchannels, unsigned int outchannels, 
    //                 unsigned int inwidth, unsigned int outwidth, unsigned int kwidth,
    //                 unsigned int inheight, unsigned int outheight, unsigned int kheight,
    //                 unsigned int indepth, unsigned int outdepth, unsigned int kdepth,
    //                 unsigned stride, unsigned int th, 
    //                 float* inmap_ptr, float* outmap_ptr, float* kernel_ptr);

    int thread_num = 16;
    int rc;
    pthread_t thread[thread_num];
    workerparam workerpara[thread_num];

    long long start = __rdtsc();
    for(int tid = 0; tid < thread_num; tid++){
        workerpara[tid].inchannels = 1;
        workerpara[tid].outchannels = 1;
        workerpara[tid].inwidth = intput_p_x;
        workerpara[tid].outwidth = outwidth;
        workerpara[tid].kwidth = k_dim;
        workerpara[tid].inheight = intput_p_y;
        workerpara[tid].outheight = outheight;
        workerpara[tid].kheight = k_dim;
        workerpara[tid].indepth = intput_p_z;
        workerpara[tid].outdepth = outdepth;
        workerpara[tid].kdepth = k_dim;
        workerpara[tid].stride = 1;
        workerpara[tid].th = 0;
        workerpara[tid].t_num = thread_num;
        workerpara[tid].tid = tid;
        workerpara[tid].inmap_ptr = input_padd;
        workerpara[tid].outmap_ptr = result;
        workerpara[tid].kernel_ptr = kernel;
        rc = pthread_create(&thread[tid], NULL, workerThread, (void*)&workerpara[tid]);
        if(rc)
        {
            printf("Error; return code from pthread_create() is %d\n", rc);
            exit(-1);
         }
    }
    for(int tid = 0; tid < thread_num; tid++)
    {
      void* retval;
      rc = pthread_join(thread[tid], &retval);
    }
    long long end = __rdtsc();

    printf("Thread execution time: %llu\n", end - start);

    /*
    convolution_3d(1, 1, 
                    intput_p_x, outwidth, k_dim,
                    intput_p_y, outheight, k_dim,
                    intput_p_z, outdepth, k_dim,
                    1, 0, 
                     input_padd, result, kernel);
    */

    verification(input,kernel,result, i_y, i_x, i_z,k_dim);
    int err = 0;
    int good=0;
    for(int i=0;i<o_x*o_y*o_z;i++){
        
        if(abs(result[i] - output[i])>0.001f){
            //printf("result[%d]:%f  output[%d]:%f\n",i,result[i],i,output[i]);
            err++;
        }else{
            good++;
        }
    }

    if(err == 0){
        printf("validation complete\n");
    }
    printf("err:%d good:%d\n", err,good);

    free(input);
    free(output);
    free(kernel);
    free(result);
    free(input_padd);
    fclose(input_file);
    fclose(output_file);
    fclose(kernel_file);
    
}
__m256i masktable_m256(int k) {
    int v[15] = { -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0 };
    int j = 7 - k;
    return _mm256_setr_epi32(v[j], v[j + 1], v[j + 2], v[j + 3], v[j + 4], v[j + 5], v[j + 6], v[j + 7]);
}
__m128i masktable_m128(int k) {

    int v[7] = { -1, -1, -1, 0, 0, 0, 0 };
    int j = 3 - k;

    return _mm_setr_epi32(v[j], v[j + 1], v[j + 2], v[j + 3]);
}
__m128 _mm256d_sum(__m256d hi, __m256d lo) {
    __m256d u = _mm256_hadd_pd(lo, hi);
    __m256d v = _mm256_hadd_pd(u, _mm256_setzero_pd());
    __m128d w = _mm_add_pd(_mm256_extractf128_pd(v, 1), _mm256_castpd256_pd128(v));

    return _mm_cvtpd_ps(w);
}
void zero_padding_3d(unsigned int channels,
                     unsigned int inwidth, unsigned int inheight, unsigned int indepth,
                     unsigned int outwidth, unsigned int outheight, unsigned int outdepth,
                     unsigned int th,
                     unsigned int pad_left, unsigned int pad_right,
                     unsigned int pad_top, unsigned int pad_bottom,
                     unsigned int pad_front, unsigned int pad_rear,
                     float* inmap_ptr, float* outmap_ptr) {

    const unsigned int inmap_offset = channels * inwidth * inheight * indepth * th;
    const unsigned int outmap_offset = channels * outwidth * outheight * outdepth * th;
    inmap_ptr += inmap_offset;
    outmap_ptr += outmap_offset;
    
    for (unsigned int iz = 0; iz < indepth; iz++) {
        /* xyz center */ {
            const unsigned int length = channels * inwidth;
            const unsigned int j = length & ~7u, k = length - j;
            const __m256i mask = masktable_m256(k);

            unsigned int inmap_idx = channels * inwidth * inheight * iz;
            unsigned int outmap_idx = channels * (pad_left + outwidth * (pad_top + outheight * (iz + pad_front)));

            for (unsigned int iy = 0; iy < inheight; iy++) {
                for (unsigned int i = 0; i < j; i += 8) {
                    __m256 x = _mm256_loadu_ps(inmap_ptr + inmap_idx + i);
                    _mm256_storeu_ps(outmap_ptr + outmap_idx + i, x);
                }
                if (k > 0) {
                    __m256 x = _mm256_maskload_ps(inmap_ptr + inmap_idx + j, mask);
                    _mm256_maskstore_ps(outmap_ptr + outmap_idx + j, mask, x);
                }

                inmap_idx += channels * inwidth;
                outmap_idx += channels * outwidth;
            }
        }

        /* x left */ {
            const unsigned int length = channels * pad_left;
            const unsigned int j = length & ~7u, k = length - j;
            const __m256i mask = masktable_m256(k);
            const __m256 x = _mm256_setzero_ps();

            unsigned int outmap_idx = channels * outwidth * (pad_top + outheight * (iz + pad_front));

            for (unsigned int iy = 0; iy < inheight; iy++) {
                for (unsigned int i = 0; i < j; i += 8) {
                    _mm256_storeu_ps(outmap_ptr + outmap_idx + i, x);
                }
                if (k > 0) {
                    _mm256_maskstore_ps(outmap_ptr + outmap_idx + j, mask, x);
                }

                outmap_idx += channels * outwidth;
            }
        }

        /* x right */ {
            const unsigned int length = channels * pad_right;
            const unsigned int j = length & ~7u, k = length - j;
            const __m256i mask = masktable_m256(k);
            const __m256 x = _mm256_setzero_ps();

            unsigned int outmap_idx = channels * (pad_left + inwidth + outwidth * (pad_top + outheight * (iz + pad_front)));

            for (unsigned int iy = 0; iy < inheight; iy++) {
                for (unsigned int i = 0; i < j; i += 8) {
                    _mm256_storeu_ps(outmap_ptr + outmap_idx + i, x);
                }
                if (k > 0) {
                    _mm256_maskstore_ps(outmap_ptr + outmap_idx + j, mask, x);
                }

                outmap_idx += channels * outwidth;
            }
        }

        /* y top */ {
            const unsigned int length = channels * outwidth * pad_top;
            const unsigned int j = length & ~7u, k = length - j;
            const __m256i mask = masktable_m256(k);
            const __m256 x = _mm256_setzero_ps();

            const unsigned int outmap_idx = channels * outwidth * outheight * (iz + pad_front);

            for (unsigned int i = 0; i < j; i += 8) {
                _mm256_storeu_ps(outmap_ptr + outmap_idx + i, x);
            }
            if (k > 0) {
                _mm256_maskstore_ps(outmap_ptr + outmap_idx + j, mask, x);
            }
        }

        /* y bottom */ {
            const unsigned int length = channels * outwidth * pad_bottom;
            const unsigned int j = length & ~7u, k = length - j;
            const __m256i mask = masktable_m256(k);
            const __m256 x = _mm256_setzero_ps();

            const unsigned int outmap_idx = channels * outwidth * (pad_top + inheight + outheight * (iz + pad_front));

            for (unsigned int i = 0; i < j; i += 8) {
                _mm256_storeu_ps(outmap_ptr + outmap_idx + i, x);
            }
            if (k > 0) {
                _mm256_maskstore_ps(outmap_ptr + outmap_idx + j, mask, x);
            }
        }
    }

    /*z front*/{
        const unsigned int length = channels * outwidth * outheight * pad_front;
        const unsigned int j = length & ~7u, k = length - j;
        const __m256i mask = masktable_m256(k);
        const __m256 x = _mm256_setzero_ps();

        const unsigned int outmap_idx = 0;

        for (unsigned int i = 0; i < j; i += 8) {
            _mm256_storeu_ps(outmap_ptr + outmap_idx + i, x);
        }
        if (k > 0) {
            _mm256_maskstore_ps(outmap_ptr + outmap_idx + j, mask, x);
        }
    }

    /*z rear*/ {
        const unsigned int length = channels * outwidth * outheight * pad_rear;
        const unsigned int j = length & ~7u, k = length - j;
        const __m256i mask = masktable_m256(k);
        const __m256 x = _mm256_setzero_ps();

        const unsigned int outmap_idx = channels * outwidth * outheight * (pad_front + indepth);

        for (unsigned int i = 0; i < j; i += 8) {
            _mm256_storeu_ps(outmap_ptr + outmap_idx + i, x);
        }
        if (k > 0) {
            _mm256_maskstore_ps(outmap_ptr + outmap_idx + j, mask, x);
        }
    }
}


void convolution_3d(unsigned int inchannels, unsigned int outchannels, 
                    unsigned int inwidth, unsigned int outwidth, unsigned int kwidth,
                    unsigned int inheight, unsigned int outheight, unsigned int kheight,
                    unsigned int indepth, unsigned int outdepth, unsigned int kdepth,
                    unsigned stride, unsigned int th, 
                    float* inmap_ptr, float* outmap_ptr, float* kernel_ptr) {
        
    const unsigned int inmap_offset = inchannels * inwidth * inheight * indepth * th, outmap_offset = outchannels * outwidth * outheight * outdepth * th;
    const unsigned int inch_sep = inchannels & ~7u, inch_rem = inchannels - inch_sep;
    const __m256i mask = masktable_m256(inch_rem);
    const __m128i mask1 = masktable_m128(1);

    inmap_ptr += inmap_offset;
    outmap_ptr += outmap_offset;

    long long start = __rdtsc();
    for (unsigned int oz = 0; oz < outdepth; oz++) {
        for (unsigned int oy = 0; oy < outheight; oy++) {
            for (unsigned int ox = 0; ox < outwidth; ox++) {
                for (unsigned int outch = 0; outch < outchannels; outch++) {
                    __m256d uv_hi = _mm256_setzero_pd(), uv_lo = _mm256_setzero_pd();

                    for (unsigned int kz = 0, iz = oz * stride; kz < kdepth; kz++, iz++) {
                        for (unsigned int ky = 0, iy = oy * stride; ky < kheight; ky++, iy++) {
                            for (unsigned int kx = 0, ix = ox * stride; kx < kwidth; kx++, ix++) {
                                for (unsigned int inch = 0; inch < inch_sep; inch += 8) {
                                    __m256 u = _mm256_loadu_ps(inmap_ptr + inch + inchannels * (ix + inwidth * (iy + inheight * iz)));
                                    __m256 v = _mm256_loadu_ps(kernel_ptr + inch + inchannels * (outch + outchannels * (kx + kwidth * (ky + kheight * kz))));

                                    __m256d u_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(u, 1));
                                    __m256d u_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(u));

                                    __m256d v_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1));
                                    __m256d v_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(v));

                                    uv_hi = _mm256_fmadd_pd(u_hi, v_hi, uv_hi);
                                    uv_lo = _mm256_fmadd_pd(u_lo, v_lo, uv_lo);
                                }

                                if (inch_rem > 0) {
                                    __m256 u = _mm256_maskload_ps(inmap_ptr + inch_sep + inchannels * (ix + inwidth * (iy + inheight * iz)), mask);
                                    __m256 v = _mm256_maskload_ps(kernel_ptr + inch_sep + inchannels * (outch + outchannels * (kx + kwidth * (ky + kheight * kz))), mask);

                                    __m256d u_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(u, 1));
                                    __m256d u_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(u));

                                    __m256d v_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1));
                                    __m256d v_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(v));

                                    uv_hi = _mm256_fmadd_pd(u_hi, v_hi, uv_hi);
                                    uv_lo = _mm256_fmadd_pd(u_lo, v_lo, uv_lo);
                                }
                            }
                        }
                    }

                    _mm_maskstore_ps(outmap_ptr + outch + outchannels * (ox + outwidth * (oy + outheight * oz)), mask1, _mm256d_sum(uv_hi, uv_lo));
                }
            }
        }
    }
    long long end = __rdtsc();
    printf("AVX execution time: %llu\n", end - start);
}

void* workerThread(void* workerpara) {
        
    workerparam* workerp = (workerparam*)workerpara;
    unsigned int inchannels = workerp->inchannels;
    unsigned int outchannels = workerp->outchannels;
    unsigned int inwidth = workerp->inwidth;
    unsigned int outwidth = workerp->outwidth;
    unsigned int kwidth = workerp->kwidth;
    unsigned int inheight = workerp->inheight;
    unsigned int outheight = workerp->outheight;
    unsigned int kheight = workerp->kheight;
    unsigned int indepth = workerp->indepth;
    unsigned int outdepth = workerp->outdepth;
    unsigned int kdepth = workerp->kdepth;
    unsigned int stride = workerp->stride;
    unsigned int th = workerp->th;
    unsigned int tid = workerp->tid;
    unsigned int t_num = workerp->t_num;
    float* inmap_ptr = workerp->inmap_ptr;
    float* outmap_ptr = workerp->outmap_ptr;
    float* kernel_ptr = workerp->kernel_ptr;

    int z_min, z_max;
    z_min = tid*(outdepth/t_num);
    z_max = tid*(outdepth/t_num) + (outdepth/t_num);

    const unsigned int inmap_offset = inchannels * inwidth * inheight * indepth * th, outmap_offset = outchannels * outwidth * outheight * outdepth * th;
    const unsigned int inch_sep = inchannels & ~7u, inch_rem = inchannels - inch_sep;
    const __m256i mask = masktable_m256(inch_rem);
    const __m128i mask1 = masktable_m128(1);

    inmap_ptr += inmap_offset;
    outmap_ptr += outmap_offset;

    for (unsigned int oz = z_min; oz < z_max; oz++) {
        for (unsigned int oy = 0; oy < outheight; oy++) {
            for (unsigned int ox = 0; ox < outwidth; ox++) {
                for (unsigned int outch = 0; outch < outchannels; outch++) {
                    __m256d uv_hi = _mm256_setzero_pd(), uv_lo = _mm256_setzero_pd();

                    for (unsigned int kz = 0, iz = oz * stride; kz < kdepth; kz++, iz++) {
                        for (unsigned int ky = 0, iy = oy * stride; ky < kheight; ky++, iy++) {
                            for (unsigned int kx = 0, ix = ox * stride; kx < kwidth; kx++, ix++) {
                                for (unsigned int inch = 0; inch < inch_sep; inch += 8) {
                                    __m256 u = _mm256_loadu_ps(inmap_ptr + inch + inchannels * (ix + inwidth * (iy + inheight * iz)));
                                    __m256 v = _mm256_loadu_ps(kernel_ptr + inch + inchannels * (outch + outchannels * (kx + kwidth * (ky + kheight * kz))));

                                    __m256d u_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(u, 1));
                                    __m256d u_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(u));

                                    __m256d v_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1));
                                    __m256d v_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(v));

                                    uv_hi = _mm256_fmadd_pd(u_hi, v_hi, uv_hi);
                                    uv_lo = _mm256_fmadd_pd(u_lo, v_lo, uv_lo);
                                }

                                if (inch_rem > 0) {
                                    __m256 u = _mm256_maskload_ps(inmap_ptr + inch_sep + inchannels * (ix + inwidth * (iy + inheight * iz)), mask);
                                    __m256 v = _mm256_maskload_ps(kernel_ptr + inch_sep + inchannels * (outch + outchannels * (kx + kwidth * (ky + kheight * kz))), mask);

                                    __m256d u_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(u, 1));
                                    __m256d u_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(u));

                                    __m256d v_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1));
                                    __m256d v_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(v));

                                    uv_hi = _mm256_fmadd_pd(u_hi, v_hi, uv_hi);
                                    uv_lo = _mm256_fmadd_pd(u_lo, v_lo, uv_lo);
                                }
                            }
                        }
                    }

                    _mm_maskstore_ps(outmap_ptr + outch + outchannels * (ox + outwidth * (oy + outheight * oz)), mask1, _mm256d_sum(uv_hi, uv_lo));
                }
            }
        }
    }

    return NULL;
}
