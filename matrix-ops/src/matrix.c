#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails. Remember to set the error messages in numc.c.
 * Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    if (rows <= 0 || cols <= 0 || (rows <= 0 && cols <= 0)) {
        return -1;
    }
    *mat = (struct matrix *) malloc(sizeof(struct matrix));
    if ((*mat) == NULL) {
        return -2;
    }
    (*mat)->rows = rows;
    (*mat)->cols = cols;
    (*mat)->data = (double *) calloc((rows * cols), sizeof(double));
    
    if ((*mat)->data == NULL) {
        return -2;
    }
    (*mat)->ref_cnt = 1;
    (*mat)->parent = NULL;

    return 0;
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`.
 * Returns -1 if either `rows` or `cols` or both are non-positive. Returns -2 if any
 * call to allocate memory in this function fails. Returns 0 upon success.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    if (rows <= 0 || cols <= 0 || (rows <= 0 && cols <= 0)) {
        return -1;
    }
    matrix *newSlice = (struct matrix *) malloc(sizeof(struct matrix));
    if (newSlice == NULL) {
        return -2;
    }
    newSlice->rows = rows;
    newSlice->cols = cols;
    newSlice->data = from->data + offset;
    if (newSlice->data == NULL) {
        return -2;
    }
    newSlice->ref_cnt = 1;
    from->ref_cnt = from->ref_cnt + 1;
    newSlice->parent = from;

    *mat = newSlice;

    return 0;

}

/*
 * Frees `mat->data` only if `mat` is not a slice and has no existing slices,
 * or if `mat` is the last existing slice of its parent matrix and its parent matrix has no other references
 * (including itself). You cannot assume that mat is not NULL.
 */
void deallocate_matrix(matrix *mat) {
    if (mat == NULL) {
        return;
    }
    if (mat->parent == NULL) {
        if (mat->ref_cnt == 1) {
            free(mat->data);
            free(mat);
        } else {
            mat->ref_cnt -= 1;
        }
    } else {
        mat->parent->ref_cnt -= 1;
        matrix *temp;
        temp = mat->parent;
        free(mat);
        if (temp->ref_cnt == 0) {
            free(temp->data);
            free(temp);
        }
    }
    
}

void copy_matrix(matrix *src, matrix *dst) {
    dst->rows = src->rows;
    dst->cols = src->cols;
    for (int i = 0; i < dst->rows * dst->cols; i++) {
        dst->data[i] = src->data[i];
    }
    dst->parent = src->parent;
    dst->ref_cnt = src->ref_cnt;
}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid.
 */
double get(matrix *mat, int row, int col) {
    return mat->data[(mat->cols * row) + col];
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid
 */
void set(matrix *mat, int row, int col, double val) {
    mat->data[(mat->cols * row) + col] = val;
}

/*
 * Sets all entries in mat to val
 */
void fill_matrix(matrix *mat, double val) {
    for (int i = 0; i < mat->rows * mat->cols; i++) {
        mat->data[i] = val;
    }
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    if ((mat1->rows != mat2->rows) || (mat1->cols != mat2->cols)) {
        return -1;
    }
    __m256d mat1_tmp;
    __m256d mat2_tmp;
    __m256d result_vec = _mm256_set1_pd(0.0);


    for (int i = 0; i < ((mat1->rows * mat1->cols) / 4) * 4; i += 4) {
        mat1_tmp = _mm256_loadu_pd((__m256d *) (mat1->data + i));
        mat2_tmp = _mm256_loadu_pd((__m256d *) (mat2->data + i));
        result_vec = _mm256_add_pd(mat1_tmp, mat2_tmp);
        _mm256_storeu_pd ((__m256d *) (result->data + i), result_vec);

        // result->data[i] = mat1->data[i] + mat2->data[i];
    }

    for (int j = ((mat1->rows * mat1->cols) / 4) * 4; j < mat1->rows * mat1->cols; j++) {
        result->data[j] = mat1->data[j] + mat2->data[j];
    }
    return 0;
}

/*
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    if ((mat1->rows != mat2->rows) || (mat1->cols != mat2->cols)) {
        return -1;
    }
    __m256d mat1_tmp;
    __m256d mat2_tmp;
    __m256d result_vec = _mm256_set1_pd(0.0);


    for (int i = 0; i < ((mat1->rows * mat1->cols) / 4) * 4; i += 4) {
        mat1_tmp = _mm256_loadu_pd((__m256d *) (mat1->data + i));
        mat2_tmp = _mm256_loadu_pd((__m256d *) (mat2->data + i));
        result_vec = _mm256_sub_pd(mat1_tmp, mat2_tmp);
        _mm256_storeu_pd ((__m256d *) (result->data + i), result_vec);

        // result->data[i] = mat1->data[i] - mat2->data[i];
    }

    for (int j = ((mat1->rows * mat1->cols) / 4) * 4; j < mat1->rows * mat1->cols; j++) {
        result->data[j] = mat1->data[j] - mat2->data[j];
    }
    return 0;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    if ((mat1->cols != mat2->rows)) {
        return -1;
    }
    matrix *mat2t = NULL;
    allocate_matrix(&mat2t, mat2->cols, mat2->rows);
    for (int x = 0; x < mat2->cols; x++) {
        for (int y = 0; y < mat2->rows; y++) {
            mat2t->data[y + x * mat2->rows] = mat2->data[x + y * mat2->cols];
        } 
    }


    __m256d mat1_tmp;
    __m256d mat2_tmp;
    __m256d result_tmp;
    
    int i;
    #pragma omp parallel for shared(result)
    for (i = 0; i < mat1->rows; i++) {
        double res[4];
        for (int j = 0; j < mat2t->rows; j++) {
            result_tmp = _mm256_set1_pd(0.0);
            _mm256_storeu_pd(res, result_tmp);
            for (int k = 0; k < (mat1->cols / 4) * 4; k+= 4) {
                mat1_tmp = _mm256_loadu_pd((__m256d *) ((mat1->data + (i * mat1->cols)) + k));
                mat2_tmp = _mm256_loadu_pd((__m256d *) ((mat2t->data + (j * mat2t->cols)) + k));
                result_tmp = _mm256_fmadd_pd(mat1_tmp, mat2_tmp, result_tmp);
            }
            _mm256_storeu_pd((__m256d *) res, result_tmp);
            result->data[(result->cols * i) + j] = res[0] + res[1] + res[2] + res[3];
        }
    }

    for (int i = 0; i < mat1->rows; i++) {
        for (int j = 0; j < mat2t->rows; j++) {
            for (int k = (mat1->cols / 4) * 4; k < mat1->cols; k++) {
                result->data[(result->cols * i) + j] += mat1->data[mat1->cols * i + k] 
                                                    * mat2t->data[mat2t->cols * j + k];
            }

        }
    }

    deallocate_matrix(mat2t);

     /* Naive approach 
     for (int i = 0; i < (mat1->rows / 4) * 4; i+=4) {
        for (int j = 0; j < mat2->cols; j++) {
            result->data[(result->cols * i) + j] = 0.0;
            for (int k = 0; k < mat1->cols; k++) {
                result->data[(result->cols * i) + j] += mat1->data[mat1->cols * i + k] 
                                                    * mat2->data[mat2->cols * k + j];
            }

        }
    } */

    return 0;


}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    /* TODO: YOUR CODE HERE */
    if (mat->rows <= 0 || mat->cols <= 0 || 
        result->rows <= 0 || result->cols <= 0 || 
        (mat->rows != result->rows && mat->cols != result->cols) ||
        (pow < 0)) {
            return -1;
        }
    if (pow == 0) {
        for (int i = 0; i < result->rows; i++) {
            for (int j = 0; j < result->cols; j++) {
                if (i == j) {
                    result->data[(mat->cols * i) + j] = 1.0;
                } else {
                    result->data[(mat->cols * i) + j] = 0.0;
            }
        } 
    }
        return 0;
    }
    matrix *temp1 = NULL;
    matrix *temp2 = NULL;
    matrix *temp3 = NULL;
    int t1 = allocate_matrix(&temp1, result->rows, result->cols);
    int t2 = allocate_matrix(&temp2, result->rows, result->cols);
    int t3 = allocate_matrix(&temp3, result->rows, result->cols);
    for (int i = 0; i < temp1->rows; i++) {
        for (int j = 0; j < temp1->cols; j++) {
            if (i == j) {
                temp1->data[(mat->cols * i) + j] = 1.0;
            } else {
                temp1->data[(mat->cols * i) + j] = 0.0;
            }
        } 
    }
    copy_matrix(mat, temp2);
    while (pow > 0) {
        if (pow & 1) {
            mul_matrix(result, temp1, temp2);
            copy_matrix(result, temp1);
        }
        mul_matrix(temp3, temp2, temp2);
        copy_matrix(temp3, temp2);
        pow >>= 1;
    }
    deallocate_matrix(temp1);
    deallocate_matrix(temp2);
    deallocate_matrix(temp3);
    
    return 0;
}

/*
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int neg_matrix(matrix *result, matrix *mat) {
    /* TODO: YOUR CODE HERE */
    if (mat->rows != result->rows && mat->cols != result->cols) {
        return -1;
    }

    __m256d mat1_tmp;
    __m256d zero_vec = _mm256_set1_pd(-0.0);
    __m256d result_vec;


    for (int i = 0; i < ((mat->rows * mat->cols) / 4) * 4; i += 4) {
        mat1_tmp = _mm256_loadu_pd((__m256d *) (mat->data + i));
        result_vec = _mm256_xor_pd(mat1_tmp, zero_vec);
        _mm256_storeu_pd((__m256d *) (result->data + i), result_vec);

    }

    for (int i = ((mat->rows * mat->cols) / 4) * 4; i < mat->rows * mat->cols; i++) {
        result->data[i] = -1 * mat->data[i];
        
    }
    return 0;
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int abs_matrix(matrix *result, matrix *mat) {
    /* TODO: YOUR CODE HERE */
    if (mat->rows != result->rows && mat->cols != result->cols) {
        return -1;
    }

    __m256d mat1_tmp;
    __m256d zero_vec = _mm256_set1_pd(0.0);
    __m256d neg_vec = _mm256_set1_pd(-1.0);
    __m256d comp_vec1;
    __m256d comp_vec2;
    __m256d and_vec1;
    __m256d and_vec2;
    __m256d abs_vec;
    __m256d result_vec;


    for (int i = 0; i < ((mat->rows * mat->cols) / 4) * 4; i += 4) {
        mat1_tmp = _mm256_loadu_pd((__m256d *) (mat->data + i));
        comp_vec1 = _mm256_cmp_pd(mat1_tmp, zero_vec, 1);
        comp_vec2 = _mm256_cmp_pd(mat1_tmp, zero_vec, 13);
        and_vec1 = _mm256_and_pd(mat1_tmp, comp_vec1);
        and_vec2 = _mm256_and_pd(mat1_tmp, comp_vec2);
        abs_vec = _mm256_mul_pd(and_vec1, neg_vec);
        result_vec = _mm256_add_pd(and_vec2, abs_vec);
        _mm256_storeu_pd((__m256d *) (result->data + i), result_vec);

    }

    for (int j = ((mat->rows * mat->cols) / 4) * 4; j < mat->rows * mat->cols; j++) {
        if (mat->data[j] < 0) {
            result->data[j] = -1 * mat->data[j];
        } else {
            result->data[j] = mat->data[j];
        }
    }
    /* for (int i = 0; i < mat->rows * mat->cols; i++) {
        if (mat->data[i] < 0) {
            result->data[i] = -1 * mat->data[i];
        } else {
            result->data[i] = mat->data[i];
        }
    } */
    return 0;
}
