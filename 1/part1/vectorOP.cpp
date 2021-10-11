#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N) {
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH) {
    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll);  // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll);  // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative);  //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative);  // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i,
                    maskIsNotNegative);  //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N) {
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //

  __pp_mask maskValid;
  __pp_mask maskAll = _pp_init_ones();

  __pp_vec_int exp;
  __pp_vec_float result, vals;
  __pp_vec_int zero = _pp_vset_int(0);
  __pp_vec_int one = _pp_vset_int(1);
  __pp_vec_float clamp = _pp_vset_float(9.999999f);

  for (int i = 0; i < N + VECTOR_WIDTH; i += VECTOR_WIDTH) {
    maskValid = _pp_init_ones((N - i >= VECTOR_WIDTH ? VECTOR_WIDTH : N - i));
    _pp_vload_int(exp, exponents + i, maskValid);
    _pp_vload_float(vals, values + i, maskValid);
    _pp_vset_float(result, 1.f, maskAll);
    __pp_mask maskActive =
        _pp_init_ones((N - i >= VECTOR_WIDTH ? VECTOR_WIDTH : N - i));
    while (_pp_cntbits(maskActive) > 0) {
      _pp_vgt_int(maskActive, exp, zero, maskValid);
      _pp_vsub_int(exp, exp, one, maskActive);
      _pp_vmult_float(result, result, vals, maskActive);

      __pp_mask needClamp;
      _pp_vgt_float(needClamp, result, clamp, maskAll);
      _pp_vset_float(result, 9.999999f, needClamp);
    }
    _pp_vstore_float(output + i, result, maskValid);
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N) {
  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //
  __pp_mask first = _pp_init_ones(1);
  __pp_mask maskAll = _pp_init_ones();
  float output = 0.f;
  __pp_vec_float result;

  for (int i = 0; i < N; i += VECTOR_WIDTH) {
    __pp_vec_float temp1, temp2;
    _pp_vload_float(temp1, values + i, maskAll);

    for (int a = VECTOR_WIDTH; a > 1; a = a / 2) {
      _pp_hadd_float(temp2, temp1);
      _pp_interleave_float(temp1, temp2);
    }
    _pp_vadd_float(result, result, temp1, maskAll);
  }

  _pp_vstore_float(&output, result, first);
  return output;
}