/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) pendulum_ode_constr_h_e_fun_jac_uxt_zt_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_sq CASADI_PREFIX(sq)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

casadi_real casadi_sq(casadi_real x) { return x*x;}

static const casadi_int casadi_s0[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s1[3] = {0, 0, 0};
static const casadi_int casadi_s2[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s3[3] = {1, 0, 0};

/* pendulum_ode_constr_h_e_fun_jac_uxt_zt:(i0[2],i1[],i2[],i3[])->(o0,o1[2],o2[1x0]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a4, a5, a6, a7, a8, a9;
  a0=-5.8691993515635636e+02;
  a1=arg[0]? arg[0][0] : 0;
  a2=9.8174770424681035e-01;
  a2=(a1-a2);
  a3=casadi_sq(a2);
  a4=arg[0]? arg[0][1] : 0;
  a5=5.5555555555555536e+00;
  a5=(a4-a5);
  a6=casadi_sq(a5);
  a3=(a3+a6);
  a6=1.9748555121383529e+01;
  a3=(a3/a6);
  a3=(-a3);
  a3=exp(a3);
  a7=(a0*a3);
  a8=-1.1713755758009473e+01;
  a9=8.8357293382212931e-01;
  a9=(a1-a9);
  a10=casadi_sq(a9);
  a11=-9.2592592592592595e+00;
  a11=(a4-a11);
  a12=casadi_sq(a11);
  a10=(a10+a12);
  a10=(a10/a6);
  a10=(-a10);
  a10=exp(a10);
  a12=(a8*a10);
  a7=(a7+a12);
  a12=-1.9786298728277180e+02;
  a13=1.2762720155208536e+00;
  a13=(a1-a13);
  a14=casadi_sq(a13);
  a15=4.0740740740740744e+00;
  a15=(a4-a15);
  a16=casadi_sq(a15);
  a14=(a14+a16);
  a14=(a14/a6);
  a14=(-a14);
  a14=exp(a14);
  a16=(a12*a14);
  a7=(a7+a16);
  a16=-3.7310465009282019e+01;
  a17=2.9452431127404310e-01;
  a17=(a1-a17);
  a18=casadi_sq(a17);
  a19=-7.0370370370370372e+00;
  a19=(a4-a19);
  a20=casadi_sq(a19);
  a18=(a18+a20);
  a18=(a18/a6);
  a18=(-a18);
  a18=exp(a18);
  a20=(a16*a18);
  a7=(a7+a20);
  a20=3.7397023325470371e+01;
  a21=1.1780972450961724e+00;
  a21=(a1-a21);
  a22=casadi_sq(a21);
  a23=-7.7777777777777777e+00;
  a23=(a4-a23);
  a24=casadi_sq(a23);
  a22=(a22+a24);
  a22=(a22/a6);
  a22=(-a22);
  a22=exp(a22);
  a24=(a20*a22);
  a7=(a7+a24);
  a24=1.1143781947405346e+01;
  a25=5.8904862254808621e-01;
  a25=(a1-a25);
  a26=casadi_sq(a25);
  a27=-5.5555555555555554e+00;
  a27=(a4-a27);
  a28=casadi_sq(a27);
  a26=(a26+a28);
  a26=(a26/a6);
  a26=(-a26);
  a26=exp(a26);
  a28=(a24*a26);
  a7=(a7+a28);
  a28=2.0273390289965704e+02;
  a29=6.8722339297276724e-01;
  a29=(a1-a29);
  a30=casadi_sq(a29);
  a31=6.2962962962962941e+00;
  a31=(a4-a31);
  a32=casadi_sq(a31);
  a30=(a30+a32);
  a30=(a30/a6);
  a30=(-a30);
  a30=exp(a30);
  a32=(a28*a30);
  a7=(a7+a32);
  a32=5.8181487956225055e+02;
  a33=9.6390366394000115e-01;
  a33=(a1-a33);
  a34=casadi_sq(a33);
  a35=4.7997524393445161e+00;
  a35=(a4-a35);
  a36=casadi_sq(a35);
  a34=(a34+a36);
  a34=(a34/a6);
  a34=(-a34);
  a34=exp(a34);
  a36=(a32*a34);
  a7=(a7+a36);
  a36=7.1755547163589817e-01;
  a37=1.3858021222930859e+00;
  a1=(a1-a37);
  a37=casadi_sq(a1);
  a38=9.2091494675663410e-01;
  a4=(a4-a38);
  a38=casadi_sq(a4);
  a37=(a37+a38);
  a37=(a37/a6);
  a37=(-a37);
  a37=exp(a37);
  a6=(a36*a37);
  a7=(a7+a6);
  a6=4.0943773690093094e-01;
  a7=(a7+a6);
  if (res[0]!=0) res[0][0]=a7;
  a7=5.0636615886759759e-02;
  a2=(a2+a2);
  a2=(a7*a2);
  a2=(a3*a2);
  a2=(a0*a2);
  a9=(a9+a9);
  a9=(a7*a9);
  a9=(a10*a9);
  a9=(a8*a9);
  a2=(a2+a9);
  a13=(a13+a13);
  a13=(a7*a13);
  a13=(a14*a13);
  a13=(a12*a13);
  a2=(a2+a13);
  a17=(a17+a17);
  a17=(a7*a17);
  a17=(a18*a17);
  a17=(a16*a17);
  a2=(a2+a17);
  a21=(a21+a21);
  a21=(a7*a21);
  a21=(a22*a21);
  a21=(a20*a21);
  a2=(a2+a21);
  a25=(a25+a25);
  a25=(a7*a25);
  a25=(a26*a25);
  a25=(a24*a25);
  a2=(a2+a25);
  a29=(a29+a29);
  a29=(a7*a29);
  a29=(a30*a29);
  a29=(a28*a29);
  a2=(a2+a29);
  a33=(a33+a33);
  a33=(a7*a33);
  a33=(a34*a33);
  a33=(a32*a33);
  a2=(a2+a33);
  a1=(a1+a1);
  a1=(a7*a1);
  a1=(a37*a1);
  a1=(a36*a1);
  a2=(a2+a1);
  a2=(-a2);
  if (res[1]!=0) res[1][0]=a2;
  a5=(a5+a5);
  a5=(a7*a5);
  a3=(a3*a5);
  a0=(a0*a3);
  a11=(a11+a11);
  a11=(a7*a11);
  a10=(a10*a11);
  a8=(a8*a10);
  a0=(a0+a8);
  a15=(a15+a15);
  a15=(a7*a15);
  a14=(a14*a15);
  a12=(a12*a14);
  a0=(a0+a12);
  a19=(a19+a19);
  a19=(a7*a19);
  a18=(a18*a19);
  a16=(a16*a18);
  a0=(a0+a16);
  a23=(a23+a23);
  a23=(a7*a23);
  a22=(a22*a23);
  a20=(a20*a22);
  a0=(a0+a20);
  a27=(a27+a27);
  a27=(a7*a27);
  a26=(a26*a27);
  a24=(a24*a26);
  a0=(a0+a24);
  a31=(a31+a31);
  a31=(a7*a31);
  a30=(a30*a31);
  a28=(a28*a30);
  a0=(a0+a28);
  a35=(a35+a35);
  a35=(a7*a35);
  a34=(a34*a35);
  a32=(a32*a34);
  a0=(a0+a32);
  a4=(a4+a4);
  a7=(a7*a4);
  a37=(a37*a7);
  a36=(a36*a37);
  a0=(a0+a36);
  a0=(-a0);
  if (res[1]!=0) res[1][1]=a0;
  return 0;
}

CASADI_SYMBOL_EXPORT int pendulum_ode_constr_h_e_fun_jac_uxt_zt(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int pendulum_ode_constr_h_e_fun_jac_uxt_zt_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int pendulum_ode_constr_h_e_fun_jac_uxt_zt_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void pendulum_ode_constr_h_e_fun_jac_uxt_zt_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int pendulum_ode_constr_h_e_fun_jac_uxt_zt_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void pendulum_ode_constr_h_e_fun_jac_uxt_zt_release(int mem) {
}

CASADI_SYMBOL_EXPORT void pendulum_ode_constr_h_e_fun_jac_uxt_zt_incref(void) {
}

CASADI_SYMBOL_EXPORT void pendulum_ode_constr_h_e_fun_jac_uxt_zt_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int pendulum_ode_constr_h_e_fun_jac_uxt_zt_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int pendulum_ode_constr_h_e_fun_jac_uxt_zt_n_out(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_real pendulum_ode_constr_h_e_fun_jac_uxt_zt_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* pendulum_ode_constr_h_e_fun_jac_uxt_zt_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* pendulum_ode_constr_h_e_fun_jac_uxt_zt_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* pendulum_ode_constr_h_e_fun_jac_uxt_zt_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s1;
    case 3: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* pendulum_ode_constr_h_e_fun_jac_uxt_zt_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s2;
    case 1: return casadi_s0;
    case 2: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int pendulum_ode_constr_h_e_fun_jac_uxt_zt_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4;
  if (sz_res) *sz_res = 3;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
