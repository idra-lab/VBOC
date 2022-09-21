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
  #define CASADI_PREFIX(ID) double_pendulum_ode_expl_vde_forw_ ## ID
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
#define casadi_s4 CASADI_PREFIX(s4)

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

static const casadi_int casadi_s0[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s1[23] = {4, 4, 0, 4, 8, 12, 16, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
static const casadi_int casadi_s2[13] = {4, 2, 0, 4, 8, 0, 1, 2, 3, 0, 1, 2, 3};
static const casadi_int casadi_s3[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s4[3] = {0, 0, 0};

/* double_pendulum_ode_expl_vde_forw:(i0[4],i1[4x4],i2[4x2],i3[2],i4[])->(o0[4],o1[4x4],o2[4x2]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a5, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a6, a60, a61, a62, a7, a8, a9;
  a0=arg[0]? arg[0][2] : 0;
  if (res[0]!=0) res[0][0]=a0;
  a1=arg[0]? arg[0][3] : 0;
  if (res[0]!=0) res[0][1]=a1;
  a2=arg[3]? arg[3][0] : 0;
  a3=4.2499999999999999e-01;
  a4=arg[0]? arg[0][1] : 0;
  a5=cos(a4);
  a6=(a3*a5);
  a7=(a6*a0);
  a8=-8.7349213929892103e-03;
  a9=(a0+a1);
  a10=(a8*a9);
  a11=4.9009000000000000e+00;
  a12=sin(a4);
  a13=(a3*a12);
  a14=(a13*a0);
  a15=(a11*a14);
  a10=(a10-a15);
  a15=(a7*a10);
  a16=-1.6123691172487806e+00;
  a17=(a16*a9);
  a18=(a11*a7);
  a17=(a17-a18);
  a18=(a14*a17);
  a15=(a15-a18);
  a18=-2.2204460492503131e-16;
  a19=(a14*a1);
  a20=(a18*a19);
  a21=3.4694469519536142e-18;
  a22=(a7*a1);
  a23=(a21*a22);
  a20=(a20-a23);
  a20=(a15+a20);
  a23=arg[3]? arg[3][1] : 0;
  a23=(a23-a15);
  a20=(a20+a23);
  a15=-2.2691954159611578e-02;
  a24=(a15*a19);
  a25=4.9007770675810418e+00;
  a26=(a25*a22);
  a24=(a24-a26);
  a26=(a9*a17);
  a24=(a24-a26);
  a26=(a8*a23);
  a27=6.2065688113890405e-01;
  a26=(a26/a27);
  a24=(a24+a26);
  a26=(a13*a24);
  a20=(a20-a26);
  a26=(a9*a10);
  a28=7.1221821903415350e-01;
  a29=(a28*a19);
  a30=(a15*a22);
  a29=(a29-a30);
  a26=(a26+a29);
  a29=(a16*a23);
  a29=(a29/a27);
  a26=(a26+a29);
  a29=(a6*a26);
  a20=(a20-a29);
  a2=(a2-a20);
  a20=-1.7835125000000001e+00;
  a29=(a15*a13);
  a29=(a18-a29);
  a30=(a28*a6);
  a29=(a29-a30);
  a30=(a5*a29);
  a31=(a25*a13);
  a32=(a15*a6);
  a31=(a31+a32);
  a32=(a12*a31);
  a30=(a30-a32);
  a20=(a20+a30);
  a30=9.8100000000000005e+00;
  a32=1.1615685580334619e-16;
  a33=-2.0510342851533115e-10;
  a34=arg[0]? arg[0][0] : 0;
  a35=cos(a34);
  a36=(a33*a35);
  a36=(a32*a36);
  a37=-3.8796683833791976e-17;
  a38=sin(a34);
  a39=(a37*a38);
  a36=(a36-a39);
  a36=(a36-a35);
  a36=(a30*a36);
  a39=(a20*a36);
  a40=(a5*a31);
  a41=(a12*a29);
  a40=(a40+a41);
  a35=(a37*a35);
  a41=(a33*a38);
  a41=(a32*a41);
  a35=(a35+a41);
  a35=(a35-a38);
  a35=(a30*a35);
  a38=(a40*a35);
  a39=(a39-a38);
  a38=2.0120634942366435e-09;
  a41=-1.2206961407182189e-20;
  a42=(a41*a13);
  a43=-1.3859078580071691e-18;
  a44=(a43*a6);
  a42=(a42+a44);
  a42=(a38*a42);
  a39=(a39-a42);
  a2=(a2-a39);
  a39=3.9410380624999997e-01;
  a42=(a13*a31);
  a44=(a21*a13);
  a45=(a18*a6);
  a44=(a44+a45);
  a42=(a42-a44);
  a44=(a6*a29);
  a42=(a42-a44);
  a39=(a39+a42);
  a2=(a2/a39);
  if (res[0]!=0) res[0][2]=a2;
  a42=(a27*a2);
  a44=(a5*a35);
  a45=(a13*a2);
  a44=(a44-a45);
  a45=(a12*a36);
  a44=(a44+a45);
  a44=(a44-a22);
  a44=(a8*a44);
  a42=(a42+a44);
  a44=(a5*a36);
  a22=(a6*a2);
  a45=(a12*a35);
  a22=(a22+a45);
  a44=(a44-a22);
  a44=(a44+a19);
  a44=(a16*a44);
  a42=(a42+a44);
  a44=-1.7451868893041153e-27;
  a42=(a42+a44);
  a23=(a23-a42);
  a23=(a23/a27);
  if (res[0]!=0) res[0][3]=a23;
  a23=arg[1]? arg[1][2] : 0;
  if (res[1]!=0) res[1][0]=a23;
  a42=arg[1]? arg[1][3] : 0;
  if (res[1]!=0) res[1][1]=a42;
  a44=cos(a4);
  a19=arg[1]? arg[1][1] : 0;
  a22=(a44*a19);
  a45=(a3*a22);
  a46=(a0*a45);
  a47=(a13*a23);
  a46=(a46+a47);
  a47=(a1*a46);
  a48=(a14*a42);
  a47=(a47+a48);
  a48=(a18*a47);
  a49=(a6*a23);
  a50=sin(a4);
  a19=(a50*a19);
  a51=(a3*a19);
  a52=(a0*a51);
  a49=(a49-a52);
  a52=(a1*a49);
  a53=(a7*a42);
  a52=(a52+a53);
  a53=(a21*a52);
  a48=(a48-a53);
  a53=(a24*a45);
  a54=(a15*a47);
  a55=(a25*a52);
  a54=(a54-a55);
  a23=(a23+a42);
  a42=(a17*a23);
  a55=(a16*a23);
  a56=(a11*a49);
  a55=(a55-a56);
  a56=(a9*a55);
  a42=(a42+a56);
  a54=(a54-a42);
  a42=1.6111961864742435e+00;
  a49=(a10*a49);
  a56=(a8*a23);
  a57=(a11*a46);
  a56=(a56-a57);
  a57=(a7*a56);
  a49=(a49+a57);
  a46=(a17*a46);
  a55=(a14*a55);
  a46=(a46+a55);
  a49=(a49-a46);
  a46=(a8*a49);
  a46=(a42*a46);
  a54=(a54-a46);
  a54=(a13*a54);
  a53=(a53+a54);
  a48=(a48-a53);
  a23=(a10*a23);
  a56=(a9*a56);
  a23=(a23+a56);
  a56=(a28*a47);
  a53=(a15*a52);
  a56=(a56-a53);
  a23=(a23+a56);
  a56=(a16*a49);
  a56=(a42*a56);
  a23=(a23-a56);
  a23=(a6*a23);
  a56=(a26*a51);
  a23=(a23-a56);
  a48=(a48-a23);
  a23=(a28*a51);
  a56=(a15*a45);
  a23=(a23-a56);
  a56=(a5*a23);
  a53=(a29*a19);
  a56=(a56-a53);
  a53=(a31*a22);
  a54=(a25*a45);
  a46=(a15*a51);
  a54=(a54-a46);
  a46=(a12*a54);
  a53=(a53+a46);
  a56=(a56-a53);
  a56=(a36*a56);
  a53=sin(a34);
  a46=arg[1]? arg[1][0] : 0;
  a55=(a53*a46);
  a57=(a33*a55);
  a57=(a32*a57);
  a58=cos(a34);
  a46=(a58*a46);
  a59=(a37*a46);
  a57=(a57+a59);
  a57=(a55-a57);
  a57=(a30*a57);
  a59=(a20*a57);
  a56=(a56+a59);
  a59=(a5*a54);
  a60=(a31*a19);
  a59=(a59-a60);
  a60=(a29*a22);
  a61=(a12*a23);
  a60=(a60+a61);
  a59=(a59+a60);
  a59=(a35*a59);
  a60=(a33*a46);
  a60=(a32*a60);
  a55=(a37*a55);
  a60=(a60-a55);
  a60=(a60-a46);
  a60=(a30*a60);
  a46=(a40*a60);
  a59=(a59+a46);
  a56=(a56-a59);
  a59=(a41*a45);
  a46=(a43*a51);
  a59=(a59-a46);
  a59=(a38*a59);
  a56=(a56-a59);
  a48=(a48+a56);
  a48=(a48/a39);
  a56=(a2/a39);
  a59=(a31*a45);
  a54=(a13*a54);
  a59=(a59+a54);
  a54=(a21*a45);
  a46=(a18*a51);
  a54=(a54-a46);
  a59=(a59-a54);
  a23=(a6*a23);
  a54=(a29*a51);
  a23=(a23-a54);
  a59=(a59-a23);
  a59=(a56*a59);
  a48=(a48+a59);
  a59=(-a48);
  if (res[1]!=0) res[1][2]=a59;
  a59=(a5*a60);
  a23=(a35*a19);
  a59=(a59-a23);
  a45=(a2*a45);
  a23=(a13*a48);
  a45=(a45-a23);
  a59=(a59-a45);
  a45=(a36*a22);
  a23=(a12*a57);
  a45=(a45+a23);
  a59=(a59+a45);
  a59=(a59-a52);
  a59=(a8*a59);
  a52=(a27*a48);
  a59=(a59-a52);
  a57=(a5*a57);
  a19=(a36*a19);
  a57=(a57-a19);
  a22=(a35*a22);
  a60=(a12*a60);
  a22=(a22+a60);
  a51=(a2*a51);
  a48=(a6*a48);
  a51=(a51+a48);
  a22=(a22-a51);
  a57=(a57-a22);
  a57=(a57+a47);
  a57=(a16*a57);
  a59=(a59+a57);
  a49=(a49+a59);
  a49=(a42*a49);
  a49=(-a49);
  if (res[1]!=0) res[1][3]=a49;
  a49=arg[1]? arg[1][6] : 0;
  if (res[1]!=0) res[1][4]=a49;
  a59=arg[1]? arg[1][7] : 0;
  if (res[1]!=0) res[1][5]=a59;
  a57=arg[1]? arg[1][5] : 0;
  a47=(a44*a57);
  a22=(a3*a47);
  a51=(a0*a22);
  a48=(a13*a49);
  a51=(a51+a48);
  a48=(a1*a51);
  a60=(a14*a59);
  a48=(a48+a60);
  a60=(a18*a48);
  a19=(a6*a49);
  a57=(a50*a57);
  a52=(a3*a57);
  a45=(a0*a52);
  a19=(a19-a45);
  a45=(a1*a19);
  a23=(a7*a59);
  a45=(a45+a23);
  a23=(a21*a45);
  a60=(a60-a23);
  a23=(a24*a22);
  a54=(a15*a48);
  a46=(a25*a45);
  a54=(a54-a46);
  a49=(a49+a59);
  a59=(a17*a49);
  a46=(a16*a49);
  a55=(a11*a19);
  a46=(a46-a55);
  a55=(a9*a46);
  a59=(a59+a55);
  a54=(a54-a59);
  a19=(a10*a19);
  a59=(a8*a49);
  a55=(a11*a51);
  a59=(a59-a55);
  a55=(a7*a59);
  a19=(a19+a55);
  a51=(a17*a51);
  a46=(a14*a46);
  a51=(a51+a46);
  a19=(a19-a51);
  a51=(a8*a19);
  a51=(a42*a51);
  a54=(a54-a51);
  a54=(a13*a54);
  a23=(a23+a54);
  a60=(a60-a23);
  a49=(a10*a49);
  a59=(a9*a59);
  a49=(a49+a59);
  a59=(a28*a48);
  a23=(a15*a45);
  a59=(a59-a23);
  a49=(a49+a59);
  a59=(a16*a19);
  a59=(a42*a59);
  a49=(a49-a59);
  a49=(a6*a49);
  a59=(a26*a52);
  a49=(a49-a59);
  a60=(a60-a49);
  a49=(a28*a52);
  a59=(a15*a22);
  a49=(a49-a59);
  a59=(a5*a49);
  a23=(a29*a57);
  a59=(a59-a23);
  a23=(a31*a47);
  a54=(a25*a22);
  a51=(a15*a52);
  a54=(a54-a51);
  a51=(a12*a54);
  a23=(a23+a51);
  a59=(a59-a23);
  a59=(a36*a59);
  a23=arg[1]? arg[1][4] : 0;
  a51=(a53*a23);
  a46=(a33*a51);
  a46=(a32*a46);
  a23=(a58*a23);
  a55=(a37*a23);
  a46=(a46+a55);
  a46=(a51-a46);
  a46=(a30*a46);
  a55=(a20*a46);
  a59=(a59+a55);
  a55=(a5*a54);
  a61=(a31*a57);
  a55=(a55-a61);
  a61=(a29*a47);
  a62=(a12*a49);
  a61=(a61+a62);
  a55=(a55+a61);
  a55=(a35*a55);
  a61=(a33*a23);
  a61=(a32*a61);
  a51=(a37*a51);
  a61=(a61-a51);
  a61=(a61-a23);
  a61=(a30*a61);
  a23=(a40*a61);
  a55=(a55+a23);
  a59=(a59-a55);
  a55=(a41*a22);
  a23=(a43*a52);
  a55=(a55-a23);
  a55=(a38*a55);
  a59=(a59-a55);
  a60=(a60+a59);
  a60=(a60/a39);
  a59=(a31*a22);
  a54=(a13*a54);
  a59=(a59+a54);
  a54=(a21*a22);
  a55=(a18*a52);
  a54=(a54-a55);
  a59=(a59-a54);
  a49=(a6*a49);
  a54=(a29*a52);
  a49=(a49-a54);
  a59=(a59-a49);
  a59=(a56*a59);
  a60=(a60+a59);
  a59=(-a60);
  if (res[1]!=0) res[1][6]=a59;
  a59=(a5*a61);
  a49=(a35*a57);
  a59=(a59-a49);
  a22=(a2*a22);
  a49=(a13*a60);
  a22=(a22-a49);
  a59=(a59-a22);
  a22=(a36*a47);
  a49=(a12*a46);
  a22=(a22+a49);
  a59=(a59+a22);
  a59=(a59-a45);
  a59=(a8*a59);
  a45=(a27*a60);
  a59=(a59-a45);
  a46=(a5*a46);
  a57=(a36*a57);
  a46=(a46-a57);
  a47=(a35*a47);
  a61=(a12*a61);
  a47=(a47+a61);
  a52=(a2*a52);
  a60=(a6*a60);
  a52=(a52+a60);
  a47=(a47-a52);
  a46=(a46-a47);
  a46=(a46+a48);
  a46=(a16*a46);
  a59=(a59+a46);
  a19=(a19+a59);
  a19=(a42*a19);
  a19=(-a19);
  if (res[1]!=0) res[1][7]=a19;
  a19=arg[1]? arg[1][10] : 0;
  if (res[1]!=0) res[1][8]=a19;
  a59=arg[1]? arg[1][11] : 0;
  if (res[1]!=0) res[1][9]=a59;
  a46=arg[1]? arg[1][9] : 0;
  a48=(a44*a46);
  a47=(a3*a48);
  a52=(a0*a47);
  a60=(a13*a19);
  a52=(a52+a60);
  a60=(a1*a52);
  a61=(a14*a59);
  a60=(a60+a61);
  a61=(a18*a60);
  a57=(a6*a19);
  a46=(a50*a46);
  a45=(a3*a46);
  a22=(a0*a45);
  a57=(a57-a22);
  a22=(a1*a57);
  a49=(a7*a59);
  a22=(a22+a49);
  a49=(a21*a22);
  a61=(a61-a49);
  a49=(a24*a47);
  a54=(a15*a60);
  a55=(a25*a22);
  a54=(a54-a55);
  a19=(a19+a59);
  a59=(a17*a19);
  a55=(a16*a19);
  a23=(a11*a57);
  a55=(a55-a23);
  a23=(a9*a55);
  a59=(a59+a23);
  a54=(a54-a59);
  a57=(a10*a57);
  a59=(a8*a19);
  a23=(a11*a52);
  a59=(a59-a23);
  a23=(a7*a59);
  a57=(a57+a23);
  a52=(a17*a52);
  a55=(a14*a55);
  a52=(a52+a55);
  a57=(a57-a52);
  a52=(a8*a57);
  a52=(a42*a52);
  a54=(a54-a52);
  a54=(a13*a54);
  a49=(a49+a54);
  a61=(a61-a49);
  a19=(a10*a19);
  a59=(a9*a59);
  a19=(a19+a59);
  a59=(a28*a60);
  a49=(a15*a22);
  a59=(a59-a49);
  a19=(a19+a59);
  a59=(a16*a57);
  a59=(a42*a59);
  a19=(a19-a59);
  a19=(a6*a19);
  a59=(a26*a45);
  a19=(a19-a59);
  a61=(a61-a19);
  a19=(a28*a45);
  a59=(a15*a47);
  a19=(a19-a59);
  a59=(a5*a19);
  a49=(a29*a46);
  a59=(a59-a49);
  a49=(a31*a48);
  a54=(a25*a47);
  a52=(a15*a45);
  a54=(a54-a52);
  a52=(a12*a54);
  a49=(a49+a52);
  a59=(a59-a49);
  a59=(a36*a59);
  a49=arg[1]? arg[1][8] : 0;
  a52=(a53*a49);
  a55=(a33*a52);
  a55=(a32*a55);
  a49=(a58*a49);
  a23=(a37*a49);
  a55=(a55+a23);
  a55=(a52-a55);
  a55=(a30*a55);
  a23=(a20*a55);
  a59=(a59+a23);
  a23=(a5*a54);
  a51=(a31*a46);
  a23=(a23-a51);
  a51=(a29*a48);
  a62=(a12*a19);
  a51=(a51+a62);
  a23=(a23+a51);
  a23=(a35*a23);
  a51=(a33*a49);
  a51=(a32*a51);
  a52=(a37*a52);
  a51=(a51-a52);
  a51=(a51-a49);
  a51=(a30*a51);
  a49=(a40*a51);
  a23=(a23+a49);
  a59=(a59-a23);
  a23=(a41*a47);
  a49=(a43*a45);
  a23=(a23-a49);
  a23=(a38*a23);
  a59=(a59-a23);
  a61=(a61+a59);
  a61=(a61/a39);
  a59=(a31*a47);
  a54=(a13*a54);
  a59=(a59+a54);
  a54=(a21*a47);
  a23=(a18*a45);
  a54=(a54-a23);
  a59=(a59-a54);
  a19=(a6*a19);
  a54=(a29*a45);
  a19=(a19-a54);
  a59=(a59-a19);
  a59=(a56*a59);
  a61=(a61+a59);
  a59=(-a61);
  if (res[1]!=0) res[1][10]=a59;
  a59=(a5*a51);
  a19=(a35*a46);
  a59=(a59-a19);
  a47=(a2*a47);
  a19=(a13*a61);
  a47=(a47-a19);
  a59=(a59-a47);
  a47=(a36*a48);
  a19=(a12*a55);
  a47=(a47+a19);
  a59=(a59+a47);
  a59=(a59-a22);
  a59=(a8*a59);
  a22=(a27*a61);
  a59=(a59-a22);
  a55=(a5*a55);
  a46=(a36*a46);
  a55=(a55-a46);
  a48=(a35*a48);
  a51=(a12*a51);
  a48=(a48+a51);
  a45=(a2*a45);
  a61=(a6*a61);
  a45=(a45+a61);
  a48=(a48-a45);
  a55=(a55-a48);
  a55=(a55+a60);
  a55=(a16*a55);
  a59=(a59+a55);
  a57=(a57+a59);
  a57=(a42*a57);
  a57=(-a57);
  if (res[1]!=0) res[1][11]=a57;
  a57=arg[1]? arg[1][14] : 0;
  if (res[1]!=0) res[1][12]=a57;
  a59=arg[1]? arg[1][15] : 0;
  if (res[1]!=0) res[1][13]=a59;
  a55=arg[1]? arg[1][13] : 0;
  a44=(a44*a55);
  a60=(a3*a44);
  a48=(a0*a60);
  a45=(a13*a57);
  a48=(a48+a45);
  a45=(a1*a48);
  a61=(a14*a59);
  a45=(a45+a61);
  a61=(a18*a45);
  a51=(a6*a57);
  a50=(a50*a55);
  a55=(a3*a50);
  a46=(a0*a55);
  a51=(a51-a46);
  a46=(a1*a51);
  a22=(a7*a59);
  a46=(a46+a22);
  a22=(a21*a46);
  a61=(a61-a22);
  a22=(a24*a60);
  a47=(a15*a45);
  a19=(a25*a46);
  a47=(a47-a19);
  a57=(a57+a59);
  a59=(a17*a57);
  a19=(a16*a57);
  a54=(a11*a51);
  a19=(a19-a54);
  a54=(a9*a19);
  a59=(a59+a54);
  a47=(a47-a59);
  a51=(a10*a51);
  a59=(a8*a57);
  a54=(a11*a48);
  a59=(a59-a54);
  a54=(a7*a59);
  a51=(a51+a54);
  a48=(a17*a48);
  a19=(a14*a19);
  a48=(a48+a19);
  a51=(a51-a48);
  a48=(a8*a51);
  a48=(a42*a48);
  a47=(a47-a48);
  a47=(a13*a47);
  a22=(a22+a47);
  a61=(a61-a22);
  a57=(a10*a57);
  a59=(a9*a59);
  a57=(a57+a59);
  a59=(a28*a45);
  a22=(a15*a46);
  a59=(a59-a22);
  a57=(a57+a59);
  a59=(a16*a51);
  a59=(a42*a59);
  a57=(a57-a59);
  a57=(a6*a57);
  a59=(a26*a55);
  a57=(a57-a59);
  a61=(a61-a57);
  a57=(a28*a55);
  a59=(a15*a60);
  a57=(a57-a59);
  a59=(a5*a57);
  a22=(a29*a50);
  a59=(a59-a22);
  a22=(a31*a44);
  a47=(a25*a60);
  a48=(a15*a55);
  a47=(a47-a48);
  a48=(a12*a47);
  a22=(a22+a48);
  a59=(a59-a22);
  a59=(a36*a59);
  a22=arg[1]? arg[1][12] : 0;
  a53=(a53*a22);
  a48=(a33*a53);
  a48=(a32*a48);
  a58=(a58*a22);
  a22=(a37*a58);
  a48=(a48+a22);
  a48=(a53-a48);
  a48=(a30*a48);
  a22=(a20*a48);
  a59=(a59+a22);
  a22=(a5*a47);
  a19=(a31*a50);
  a22=(a22-a19);
  a19=(a29*a44);
  a54=(a12*a57);
  a19=(a19+a54);
  a22=(a22+a19);
  a22=(a35*a22);
  a19=(a33*a58);
  a19=(a32*a19);
  a53=(a37*a53);
  a19=(a19-a53);
  a19=(a19-a58);
  a19=(a30*a19);
  a58=(a40*a19);
  a22=(a22+a58);
  a59=(a59-a22);
  a22=(a41*a60);
  a58=(a43*a55);
  a22=(a22-a58);
  a22=(a38*a22);
  a59=(a59-a22);
  a61=(a61+a59);
  a61=(a61/a39);
  a59=(a31*a60);
  a47=(a13*a47);
  a59=(a59+a47);
  a47=(a21*a60);
  a22=(a18*a55);
  a47=(a47-a22);
  a59=(a59-a47);
  a57=(a6*a57);
  a47=(a29*a55);
  a57=(a57-a47);
  a59=(a59-a57);
  a56=(a56*a59);
  a61=(a61+a56);
  a56=(-a61);
  if (res[1]!=0) res[1][14]=a56;
  a56=(a5*a19);
  a59=(a35*a50);
  a56=(a56-a59);
  a60=(a2*a60);
  a59=(a13*a61);
  a60=(a60-a59);
  a56=(a56-a60);
  a60=(a36*a44);
  a59=(a12*a48);
  a60=(a60+a59);
  a56=(a56+a60);
  a56=(a56-a46);
  a56=(a8*a56);
  a46=(a27*a61);
  a56=(a56-a46);
  a48=(a5*a48);
  a50=(a36*a50);
  a48=(a48-a50);
  a44=(a35*a44);
  a19=(a12*a19);
  a44=(a44+a19);
  a55=(a2*a55);
  a61=(a6*a61);
  a55=(a55+a61);
  a44=(a44-a55);
  a48=(a48-a44);
  a48=(a48+a45);
  a48=(a16*a48);
  a56=(a56+a48);
  a51=(a51+a56);
  a51=(a42*a51);
  a51=(-a51);
  if (res[1]!=0) res[1][15]=a51;
  a51=arg[2]? arg[2][2] : 0;
  if (res[2]!=0) res[2][0]=a51;
  a56=arg[2]? arg[2][3] : 0;
  if (res[2]!=0) res[2][1]=a56;
  a48=(1./a39);
  a45=cos(a4);
  a44=arg[2]? arg[2][1] : 0;
  a55=(a45*a44);
  a61=(a3*a55);
  a19=(a0*a61);
  a50=(a13*a51);
  a19=(a19+a50);
  a50=(a1*a19);
  a46=(a14*a56);
  a50=(a50+a46);
  a46=(a18*a50);
  a60=(a6*a51);
  a4=sin(a4);
  a44=(a4*a44);
  a59=(a3*a44);
  a57=(a0*a59);
  a60=(a60-a57);
  a57=(a1*a60);
  a47=(a7*a56);
  a57=(a57+a47);
  a47=(a21*a57);
  a46=(a46-a47);
  a47=(a24*a61);
  a22=(a15*a50);
  a58=(a25*a57);
  a22=(a22-a58);
  a51=(a51+a56);
  a56=(a17*a51);
  a58=(a16*a51);
  a53=(a11*a60);
  a58=(a58-a53);
  a53=(a9*a58);
  a56=(a56+a53);
  a22=(a22-a56);
  a60=(a10*a60);
  a56=(a8*a51);
  a53=(a11*a19);
  a56=(a56-a53);
  a53=(a7*a56);
  a60=(a60+a53);
  a19=(a17*a19);
  a58=(a14*a58);
  a19=(a19+a58);
  a60=(a60-a19);
  a19=(a8*a60);
  a19=(a42*a19);
  a22=(a22-a19);
  a22=(a13*a22);
  a47=(a47+a22);
  a46=(a46-a47);
  a51=(a10*a51);
  a56=(a9*a56);
  a51=(a51+a56);
  a56=(a28*a50);
  a47=(a15*a57);
  a56=(a56-a47);
  a51=(a51+a56);
  a56=(a16*a60);
  a56=(a42*a56);
  a51=(a51-a56);
  a51=(a6*a51);
  a56=(a26*a59);
  a51=(a51-a56);
  a46=(a46-a51);
  a51=(a28*a59);
  a56=(a15*a61);
  a51=(a51-a56);
  a56=(a5*a51);
  a47=(a29*a44);
  a56=(a56-a47);
  a47=(a31*a55);
  a22=(a25*a61);
  a19=(a15*a59);
  a22=(a22-a19);
  a19=(a12*a22);
  a47=(a47+a19);
  a56=(a56-a47);
  a56=(a36*a56);
  a47=sin(a34);
  a19=arg[2]? arg[2][0] : 0;
  a58=(a47*a19);
  a53=(a33*a58);
  a53=(a32*a53);
  a34=cos(a34);
  a19=(a34*a19);
  a54=(a37*a19);
  a53=(a53+a54);
  a53=(a58-a53);
  a53=(a30*a53);
  a54=(a20*a53);
  a56=(a56+a54);
  a54=(a5*a22);
  a23=(a31*a44);
  a54=(a54-a23);
  a23=(a29*a55);
  a49=(a12*a51);
  a23=(a23+a49);
  a54=(a54+a23);
  a54=(a35*a54);
  a23=(a33*a19);
  a23=(a32*a23);
  a58=(a37*a58);
  a23=(a23-a58);
  a23=(a23-a19);
  a23=(a30*a23);
  a19=(a40*a23);
  a54=(a54+a19);
  a56=(a56-a54);
  a54=(a41*a61);
  a19=(a43*a59);
  a54=(a54-a19);
  a54=(a38*a54);
  a56=(a56-a54);
  a46=(a46+a56);
  a46=(a46/a39);
  a56=(a2/a39);
  a54=(a31*a61);
  a22=(a13*a22);
  a54=(a54+a22);
  a22=(a21*a61);
  a19=(a18*a59);
  a22=(a22-a19);
  a54=(a54-a22);
  a51=(a6*a51);
  a22=(a29*a59);
  a51=(a51-a22);
  a54=(a54-a51);
  a54=(a56*a54);
  a46=(a46+a54);
  a48=(a48-a46);
  if (res[2]!=0) res[2][2]=a48;
  a48=(a27/a39);
  a54=(a13/a39);
  a54=(a8*a54);
  a48=(a48-a54);
  a54=(a6/a39);
  a54=(a16*a54);
  a48=(a48-a54);
  a48=(a42*a48);
  a54=(a5*a23);
  a51=(a35*a44);
  a54=(a54-a51);
  a61=(a2*a61);
  a51=(a13*a46);
  a61=(a61-a51);
  a54=(a54-a61);
  a61=(a36*a55);
  a51=(a12*a53);
  a61=(a61+a51);
  a54=(a54+a61);
  a54=(a54-a57);
  a54=(a8*a54);
  a57=(a27*a46);
  a54=(a54-a57);
  a53=(a5*a53);
  a44=(a36*a44);
  a53=(a53-a44);
  a55=(a35*a55);
  a23=(a12*a23);
  a55=(a55+a23);
  a59=(a2*a59);
  a46=(a6*a46);
  a59=(a59+a46);
  a55=(a55-a59);
  a53=(a53-a55);
  a53=(a53+a50);
  a53=(a16*a53);
  a54=(a54+a53);
  a60=(a60+a54);
  a60=(a42*a60);
  a48=(a48+a60);
  a48=(-a48);
  if (res[2]!=0) res[2][3]=a48;
  a48=arg[2]? arg[2][6] : 0;
  if (res[2]!=0) res[2][4]=a48;
  a60=arg[2]? arg[2][7] : 0;
  if (res[2]!=0) res[2][5]=a60;
  a54=1.;
  a53=-1.4073672037536502e-02;
  a53=(a53*a13);
  a53=(a54-a53);
  a50=-2.5978429729000778e+00;
  a50=(a50*a6);
  a53=(a53-a50);
  a53=(a53/a39);
  a50=arg[2]? arg[2][5] : 0;
  a45=(a45*a50);
  a55=(a3*a45);
  a59=(a0*a55);
  a46=(a13*a48);
  a59=(a59+a46);
  a46=(a1*a59);
  a23=(a14*a60);
  a46=(a46+a23);
  a23=(a18*a46);
  a44=(a6*a48);
  a4=(a4*a50);
  a3=(a3*a4);
  a0=(a0*a3);
  a44=(a44-a0);
  a1=(a1*a44);
  a0=(a7*a60);
  a1=(a1+a0);
  a0=(a21*a1);
  a23=(a23-a0);
  a24=(a24*a55);
  a0=(a15*a46);
  a50=(a25*a1);
  a0=(a0-a50);
  a48=(a48+a60);
  a60=(a17*a48);
  a50=(a16*a48);
  a57=(a11*a44);
  a50=(a50-a57);
  a57=(a9*a50);
  a60=(a60+a57);
  a0=(a0-a60);
  a44=(a10*a44);
  a60=(a8*a48);
  a11=(a11*a59);
  a60=(a60-a11);
  a7=(a7*a60);
  a44=(a44+a7);
  a17=(a17*a59);
  a14=(a14*a50);
  a17=(a17+a14);
  a44=(a44-a17);
  a17=(a8*a44);
  a17=(a42*a17);
  a0=(a0-a17);
  a0=(a13*a0);
  a24=(a24+a0);
  a23=(a23-a24);
  a10=(a10*a48);
  a9=(a9*a60);
  a10=(a10+a9);
  a9=(a28*a46);
  a60=(a15*a1);
  a9=(a9-a60);
  a10=(a10+a9);
  a9=(a16*a44);
  a9=(a42*a9);
  a10=(a10-a9);
  a10=(a6*a10);
  a26=(a26*a3);
  a10=(a10-a26);
  a23=(a23-a10);
  a28=(a28*a3);
  a10=(a15*a55);
  a28=(a28-a10);
  a10=(a5*a28);
  a26=(a29*a4);
  a10=(a10-a26);
  a26=(a31*a45);
  a25=(a25*a55);
  a15=(a15*a3);
  a25=(a25-a15);
  a15=(a12*a25);
  a26=(a26+a15);
  a10=(a10-a26);
  a10=(a36*a10);
  a26=arg[2]? arg[2][4] : 0;
  a47=(a47*a26);
  a15=(a33*a47);
  a15=(a32*a15);
  a34=(a34*a26);
  a26=(a37*a34);
  a15=(a15+a26);
  a15=(a47-a15);
  a15=(a30*a15);
  a20=(a20*a15);
  a10=(a10+a20);
  a20=(a5*a25);
  a26=(a31*a4);
  a20=(a20-a26);
  a26=(a29*a45);
  a9=(a12*a28);
  a26=(a26+a9);
  a20=(a20+a26);
  a20=(a35*a20);
  a33=(a33*a34);
  a32=(a32*a33);
  a37=(a37*a47);
  a32=(a32-a37);
  a32=(a32-a34);
  a30=(a30*a32);
  a40=(a40*a30);
  a20=(a20+a40);
  a10=(a10-a20);
  a41=(a41*a55);
  a43=(a43*a3);
  a41=(a41-a43);
  a38=(a38*a41);
  a10=(a10-a38);
  a23=(a23+a10);
  a23=(a23/a39);
  a31=(a31*a55);
  a25=(a13*a25);
  a31=(a31+a25);
  a21=(a21*a55);
  a18=(a18*a3);
  a21=(a21-a18);
  a31=(a31-a21);
  a28=(a6*a28);
  a29=(a29*a3);
  a28=(a28-a29);
  a31=(a31-a28);
  a56=(a56*a31);
  a23=(a23+a56);
  a56=(a53+a23);
  a56=(-a56);
  if (res[2]!=0) res[2][6]=a56;
  a56=(a13*a53);
  a56=(a8*a56);
  a31=(a27*a53);
  a56=(a56-a31);
  a53=(a6*a53);
  a53=(a16*a53);
  a56=(a56+a53);
  a54=(a54-a56);
  a54=(a42*a54);
  a56=(a5*a30);
  a53=(a35*a4);
  a56=(a56-a53);
  a55=(a2*a55);
  a13=(a13*a23);
  a55=(a55-a13);
  a56=(a56-a55);
  a55=(a36*a45);
  a13=(a12*a15);
  a55=(a55+a13);
  a56=(a56+a55);
  a56=(a56-a1);
  a8=(a8*a56);
  a27=(a27*a23);
  a8=(a8-a27);
  a5=(a5*a15);
  a36=(a36*a4);
  a5=(a5-a36);
  a35=(a35*a45);
  a12=(a12*a30);
  a35=(a35+a12);
  a2=(a2*a3);
  a6=(a6*a23);
  a2=(a2+a6);
  a35=(a35-a2);
  a5=(a5-a35);
  a5=(a5+a46);
  a16=(a16*a5);
  a8=(a8+a16);
  a44=(a44+a8);
  a42=(a42*a44);
  a54=(a54-a42);
  if (res[2]!=0) res[2][7]=a54;
  return 0;
}

CASADI_SYMBOL_EXPORT int double_pendulum_ode_expl_vde_forw(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int double_pendulum_ode_expl_vde_forw_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int double_pendulum_ode_expl_vde_forw_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void double_pendulum_ode_expl_vde_forw_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int double_pendulum_ode_expl_vde_forw_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void double_pendulum_ode_expl_vde_forw_release(int mem) {
}

CASADI_SYMBOL_EXPORT void double_pendulum_ode_expl_vde_forw_incref(void) {
}

CASADI_SYMBOL_EXPORT void double_pendulum_ode_expl_vde_forw_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int double_pendulum_ode_expl_vde_forw_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int double_pendulum_ode_expl_vde_forw_n_out(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_real double_pendulum_ode_expl_vde_forw_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* double_pendulum_ode_expl_vde_forw_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* double_pendulum_ode_expl_vde_forw_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* double_pendulum_ode_expl_vde_forw_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    case 3: return casadi_s3;
    case 4: return casadi_s4;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* double_pendulum_ode_expl_vde_forw_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int double_pendulum_ode_expl_vde_forw_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5;
  if (sz_res) *sz_res = 3;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif