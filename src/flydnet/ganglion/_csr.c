/* flydnet CPU 희소 행렬 커널 (ctypes로 부름 - flydnet/ganglion/csr.py)

   CSR: indptr int64 (n_rows + 1), indices int32, data float32 또는 float64. x·out은 (행, nb) 행 우선 밀집.
   행 범위 [r0, r1)만 계산 → 파이썬 스레드가 행을 나눠 부름 (ctypes는 호출 동안 GIL을 놓음). OpenMP를 쓰지 않아
   플랫폼마다 런타임(libgomp·libomp·vcomp)을 함께 배포할 필요가 없음.

   덧셈 순서: 행마다 연결 순서대로 0에서 하나씩 더함 (scipy csr_matvecs·numpy reduceat과 같은 순서) → 같은 비트.
   그러려면 곱셈-덧셈을 FMA로 합치지 않게 빌드할 것 (-ffp-contract=off, MSVC는 기본이 합치지 않음).

   파이썬 확장 모듈로 빌드할 때(FLYDNET_PYEXT)는 빈 모듈 초기화 함수도 둠 - setuptools Extension이 요구.
   import해서 쓰지는 않고 ctypes로 같은 파일을 연다. */
#include <stdint.h>
#include <string.h>

#ifdef _WIN32
#define FD_EXPORT __declspec(dllexport)
#else
#define FD_EXPORT __attribute__((visibility("default")))
#endif

/* 커널 규격 번호: 인자·동작이 바뀌면 올리고 csr.py의 _ABI도 같이 */
FD_EXPORT int64_t flydnet_csr_abi(void) { return 1; }

FD_EXPORT void csr_spmm_f32(const int64_t *indptr, const int32_t *indices, const float *data,
                            const float *x, float *out, int64_t r0, int64_t r1, int64_t nb) {
    for (int64_t r = r0; r < r1; r++) {
        float *o = out + r * nb;
        memset(o, 0, sizeof(float) * (size_t)nb);
        for (int64_t e = indptr[r]; e < indptr[r + 1]; e++) {
            const float w = data[e];
            const float *xi = x + (int64_t)indices[e] * nb;
            for (int64_t b = 0; b < nb; b++) o[b] += w * xi[b];
        }
    }
}

FD_EXPORT void csr_spmm_f64(const int64_t *indptr, const int32_t *indices, const double *data,
                            const double *x, double *out, int64_t r0, int64_t r1, int64_t nb) {
    for (int64_t r = r0; r < r1; r++) {
        double *o = out + r * nb;
        memset(o, 0, sizeof(double) * (size_t)nb);
        for (int64_t e = indptr[r]; e < indptr[r + 1]; e++) {
            const double w = data[e];
            const double *xi = x + (int64_t)indices[e] * nb;
            for (int64_t b = 0; b < nb; b++) o[b] += w * xi[b];
        }
    }
}

/* 전치 구조 (계수 정렬, 안정): 열 c의 연결을 원래 행 순서대로 모음.
   t_indptr (n_cols + 1), t_indices (nnz) = 원래 행 번호, perm (nnz) = 전치의 연결 k가 원래 몇 번째 연결인지.
   count (n_cols)는 0으로 채워진 작업 공간 */
FD_EXPORT void csr_transpose(const int64_t *indptr, const int32_t *indices, int64_t n_rows, int64_t n_cols,
                             int64_t *t_indptr, int32_t *t_indices, int64_t *perm, int64_t *count) {
    const int64_t nnz = indptr[n_rows];
    for (int64_t e = 0; e < nnz; e++) count[indices[e]]++;
    t_indptr[0] = 0;
    for (int64_t c = 0; c < n_cols; c++) {
        t_indptr[c + 1] = t_indptr[c] + count[c];
        count[c] = t_indptr[c];                       /* 이제 열마다 다음에 쓸 자리 */
    }
    for (int64_t r = 0; r < n_rows; r++) {
        for (int64_t e = indptr[r]; e < indptr[r + 1]; e++) {
            const int64_t k = count[indices[e]]++;
            t_indices[k] = (int32_t)r;
            perm[k] = e;
        }
    }
}

#ifdef FLYDNET_PYEXT
#include <Python.h>
static struct PyModuleDef fd_csr_module = {PyModuleDef_HEAD_INIT, "_csr", NULL, -1, NULL};
PyMODINIT_FUNC PyInit__csr(void) { return PyModule_Create(&fd_csr_module); }
#endif
