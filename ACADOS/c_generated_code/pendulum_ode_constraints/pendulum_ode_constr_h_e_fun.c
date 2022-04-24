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
  #define CASADI_PREFIX(ID) pendulum_ode_constr_h_e_fun_ ## ID
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

/* pendulum_ode_constr_h_e_fun:(i0[2],i1[],i2[],i3[])->(o0) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a2, a3, a4, a5, a6, a7;
  a0=-1.3019559902200488e+04;
  a1=arg[0]? arg[0][0] : 0;
  a2=1.1597853494407875e+00;
  a2=(a1-a2);
  a2=casadi_sq(a2);
  a3=arg[0]? arg[0][1] : 0;
  a4=4.4500330234212271e+00;
  a4=(a3-a4);
  a4=casadi_sq(a4);
  a2=(a2+a4);
  a4=3.2384292072485252e+01;
  a2=(a2/a4);
  a2=(-a2);
  a2=exp(a2);
  a2=(a0*a2);
  a5=-5.7229654306414968e+03;
  a6=1.1997647237250570e+00;
  a6=(a1-a6);
  a6=casadi_sq(a6);
  a7=4.2447797591830501e+00;
  a7=(a3-a7);
  a7=casadi_sq(a7);
  a6=(a6+a7);
  a6=(a6/a4);
  a6=(-a6);
  a6=exp(a6);
  a5=(a5*a6);
  a2=(a2+a5);
  a5=1.0539406750766784e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=5.0109231316364351e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.1669758843840015e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=4.3606157597927169e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.1428156869748027e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=4.4581618655692736e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.2227744355433416e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=4.0253010211857916e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=5.8540541817685776e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=7.3083371437280888e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=5.5951949238128762e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=7.4058832495046474e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=2.1715415528506107e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-4.3626479703297258e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.5243934079613558e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-3.7164050195600273e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.0323690702470365e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=5.1165980795610437e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.8695390852356253e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-4.0090433368897020e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.7372332422804887e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-3.8789818625209573e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.6797089627347772e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-3.7814357567443988e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.1207647131489466e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-3.0823553320123978e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.1435826773687454e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-9.0763603109282123e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.1581554948536590e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-3.1229995427526287e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=7.5222582885942124e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=6.5401615607376904e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=8.9066759496610040e-02;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-2.7653304882385825e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=2.0516993037970452e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=9.0336838896509661e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=5.4312507271075983e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=7.4953005131331558e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.2488521089373976e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-9.5823807346441097e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=-5.9863824594311782e+03;
  a6=1.3216203225627228e+00;
  a6=(a1-a6);
  a6=casadi_sq(a6);
  a7=-9.9319209470101111e+00;
  a7=(a3-a7);
  a7=casadi_sq(a7);
  a6=(a6+a7);
  a6=(a6/a4);
  a6=(-a6);
  a6=exp(a6);
  a5=(a5*a6);
  a2=(a2+a5);
  a5=1.1855754014371149e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-9.2490982065742013e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=7.7782413325726296e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=6.4101000863689421e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=-1.8531462535243788e+03;
  a6=1.0507768397016642e-01;
  a6=(a1-a6);
  a6=casadi_sq(a6);
  a7=9.4604481024234097e+00;
  a7=(a3-a7);
  a7=casadi_sq(a7);
  a6=(a6+a7);
  a6=(a6/a4);
  a6=(-a6);
  a6=exp(a6);
  a5=(a5*a6);
  a2=(a2+a5);
  a5=-1.2309839926071199e+04;
  a6=7.7849524985196289e-02;
  a6=(a1-a6);
  a6=casadi_sq(a6);
  a7=-2.6352690138698360e+00;
  a7=(a3-a7);
  a7=casadi_sq(a7);
  a6=(a6+a7);
  a6=(a6/a4);
  a6=(-a6);
  a6=exp(a6);
  a5=(a5*a6);
  a2=(a2+a5);
  a5=1.6662866308407778e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=9.1921963115378738e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.1376385018156887e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=4.5557079713458322e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=-1.3536180895787802e+03;
  a6=4.7093210188089185e-01;
  a6=(a1-a6);
  a6=casadi_sq(a6);
  a7=7.8692272519433004e+00;
  a7=(a3-a7);
  a7=casadi_sq(a7);
  a6=(a6+a7);
  a6=(a6/a4);
  a6=(-a6);
  a6=exp(a6);
  a5=(a5*a6);
  a2=(a2+a5);
  a5=1.0245074187091225e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-8.4992125184169076e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=4.5108622543762134e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=7.9342579891276728e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.2141457936114850e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=4.1472336534064951e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.3058011456876520e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-3.4644109129705845e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=6.1723551952548483e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-6.6285627190976992e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=5.4389206310470262e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-6.2830869278057211e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=6.8492242179093876e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=6.8084133516232264e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=8.2087146911730369e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-7.5522024081694870e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=7.3516029259419358e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-7.1894528273129099e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=7.9786175729901909e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-7.4546563023929284e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=8.4330593814013122e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-7.6741350403901851e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=-1.1430057667789102e+04;
  a6=6.3276707500282703e-01;
  a6=(a1-a6);
  a6=casadi_sq(a6);
  a7=-6.7261088248742578e+00;
  a7=(a3-a7);
  a7=casadi_sq(a7);
  a6=(a6+a7);
  a6=(a6/a4);
  a6=(-a6);
  a6=exp(a6);
  a5=(a5*a6);
  a2=(a2+a5);
  a5=8.7264332070844408e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-7.8041965147589289e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=4.6019423636569234e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-5.8847736625514404e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=-7.4184011898492981e+03;
  a6=4.4677190447169300e-02;
  a6=(a1-a6);
  a6=casadi_sq(a6);
  a7=9.7043133668648025e+00;
  a7=(a3-a7);
  a7=casadi_sq(a7);
  a6=(a6+a7);
  a6=(a6/a4);
  a6=(-a6);
  a6=exp(a6);
  a5=(a5*a6);
  a2=(a2+a5);
  a5=2.6652916189513015e-02;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-1.5216176395874630e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.9174759848570515e-04;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-1.3915561652187165e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=-3.1110453972661908e+03;
  a6=1.3134710496270803e-02;
  a6=(a1-a6);
  a6=casadi_sq(a6);
  a7=-1.5460041660316008e+00;
  a7=(a3-a7);
  a7=casadi_sq(a7);
  a6=(a6+a7);
  a6=(a6/a4);
  a6=(-a6);
  a6=exp(a6);
  a5=(a5*a6);
  a2=(a2+a5);
  a5=1.4262186375366748e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=2.3304374333180924e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=7.6699039394282058e-03;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-1.2208504801097391e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.4089613536729615e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=2.5743026977594869e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.3813496994910199e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=2.7937814357567436e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.3934297981956194e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=2.6718488035360473e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.4728133039687012e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=1.8183203779911601e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=4.7936899621426289e-03;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-1.0501447950007616e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.4636094192413873e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=1.9402530102118565e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.4581446126845448e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=2.0296702738403702e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=-2.5470650429817811e+03;
  a6=1.5366652542644410e+00;
  a6=(a1-a6);
  a6=casadi_sq(a6);
  a7=1.3793629019966502e+00;
  a7=(a3-a7);
  a7=casadi_sq(a7);
  a6=(a6+a7);
  a6=(a6/a4);
  a6=(-a6);
  a6=exp(a6);
  a5=(a5*a6);
  a2=(a2+a5);
  a5=1.4857562668664863e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=1.6638723771782775e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.5142307852416135e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=1.5012955342173466e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a0=(a0*a5);
  a2=(a2+a0);
  a0=8.1173780487804879e+03;
  a5=9.5106808848909749e-02;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-2.5377229080932784e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=7.6123796598824942e-02;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-2.4401768023167207e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=5.3114084780540326e-02;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-1.5622618503276939e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.0431069357622361e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-2.7572016460905360e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.4352307746655030e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-3.3749936493420716e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=6.9220883053339552e-02;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-2.1475384849870451e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=4.1609228871398014e-02;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-1.7085810089925317e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=6.4043697894225524e-02;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-1.8305136412132299e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=5.6853162951011574e-02;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-1.7654829040288593e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=5.8866512735111481e-02;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-1.8549001676573695e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=6.5194183485139751e-02;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-1.9524462734339281e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=8.0150496167024757e-02;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-2.3426306965401622e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.0785071655564839e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-2.8356990319928186e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=5.9222309418424261e-02;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-2.0296253904035813e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.4883448594460433e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=1.5256820606614863e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=4.6786414030512057e-02;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-1.5866483767718336e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=2.7841751300124390e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-4.7690900777320531e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.2952550277709383e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=3.6595031245236989e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=7.5970321222591774e+03;
  a6=1.2446336617707121e+00;
  a6=(a1-a6);
  a6=casadi_sq(a6);
  a7=3.7733069145963487e+00;
  a7=(a3-a7);
  a7=casadi_sq(a7);
  a6=(a6+a7);
  a6=(a6/a4);
  a6=(-a6);
  a6=exp(a6);
  a5=(a5*a6);
  a2=(a2+a5);
  a5=1.4607332052641018e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=1.8427069044352997e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.2679309949867252e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=3.6757608088197919e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.3476979959567785e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=3.0173245948280236e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.7573667401214876e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-3.8139511253365850e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.3612162016500209e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=2.8953919626073255e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.3382064898317363e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=3.2205456485291872e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=7.5514646284522023e+03;
  a6=1.3943720121096510e+00;
  a6=(a1-a6);
  a6=casadi_sq(a6);
  a7=2.3854963362405663e+00;
  a7=(a3-a7);
  a7=casadi_sq(a7);
  a6=(a6+a7);
  a6=(a6/a4);
  a6=(-a6);
  a6=exp(a6);
  a5=(a5*a6);
  a2=(a2+a5);
  a5=1.4371238579087733e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=1.8906715308852411e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.4374358720480886e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=2.1028298531727891e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=2.9164809729675750e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-4.8910227099527512e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.3021579413164237e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=3.5538281765990938e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.4411749502185598e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=2.0621856424325546e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=2.1226459152367561e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-4.1594269166285631e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.4210414523775607e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=2.3466951176141837e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.5843305834652456e+03;
  a6=3.1063110954684231e-01;
  a6=(a1-a6);
  a6=casadi_sq(a6);
  a7=-4.9154092363968900e+00;
  a7=(a3-a7);
  a7=casadi_sq(a7);
  a6=(a6+a7);
  a6=(a6/a4);
  a6=(-a6);
  a6=exp(a6);
  a5=(a5*a6);
  a2=(a2+a5);
  a5=1.2909407068050098e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=3.4562820708225352e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.3986069833547334e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=2.4767565919829302e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.4348472794685316e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=1.9077376416196721e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.3174977491952800e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=3.2611898592694200e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.4443387855935741e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=1.8101915358431135e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.0085923680348091e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-2.8547477518670927e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.3261263911271368e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=3.1229995427526287e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.2831749290663388e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=3.7814357567443952e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.4275128290461965e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=2.0101489684242759e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=8.5001710408713094e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=6.0554793476604161e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=9.4780837931484052e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=5.5108469237412976e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=9.2019672513289896e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=5.6571660824061354e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=9.4090546576935519e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=5.5840065030737147e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=8.6612390235993020e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=5.9823197683279972e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=7.8846612497321955e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=6.3237311385459485e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=8.2470642108701786e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=6.2017985063252539e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.0116603296105804e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=5.1897576588934609e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=9.8232294704226741e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=5.3604633440024365e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.0935365541639765e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-8.7918508357465832e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.1660171463915729e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-9.1007468373723519e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=4.1905405836314494e+03;
  a6=1.2942962897785097e+00;
  a6=(a1-a6);
  a6=casadi_sq(a6);
  a7=-9.7043133668648078e+00;
  a7=(a3-a7);
  a7=casadi_sq(a7);
  a6=(a6+a7);
  a6=(a6/a4);
  a6=(-a6);
  a6=exp(a6);
  a5=(a5*a6);
  a2=(a2+a5);
  a5=1.2753132775284248e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-9.6148961032362958e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.3086773596649377e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-9.8018594726413664e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.1191348585618182e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-8.8893969415231417e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=3.1465780911504215e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-5.1105014479500079e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.0607477148229210e+00;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-8.6374028349336989e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=7.4148796334422185e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-7.1731951430168159e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=9.8021372345892466e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-8.2350251486053949e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=3.3881800652424099e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-5.2161763958746130e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=9.5777925443609724e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-8.1374790428288364e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=9.3534478541326971e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-8.0399329370522779e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=9.3016760025415568e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-8.0236752527561848e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=8.8443579801531502e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-7.8285830412030686e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=7.6478529656023497e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-7.2707412487933754e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=3.5099498786183931e+03;
  a6=9.0773313123132815e-01;
  a6=(a1-a6);
  a6=casadi_sq(a6);
  a7=-7.8936137783874409e+00;
  a7=(a3-a7);
  a7=casadi_sq(a7);
  a6=(a6+a7);
  a6=(a6/a4);
  a6=(-a6);
  a6=exp(a6);
  a5=(a5*a6);
  a2=(a2+a5);
  a5=3.7333257425166794e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-5.3624955545394499e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=3.9749277166086677e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-5.4945892394452063e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=7.5701951882156393e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-7.2382258802011892e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=7.9239695074217653e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-7.3896255652085561e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=5.3785201375240288e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-6.2261850327693953e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=4.3574641755876492e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-5.6734237667022303e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=3.8886412972901002e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-5.4275262917238223e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=4.1359956993366598e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=-5.5677488187776261e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=8.9373555654187165e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=5.7628410303307387e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=8.7130108751904423e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=5.8847736625514386e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.3196245265904013e+03;
  a6=8.5576953204170203e-01;
  a6=(a1-a6);
  a6=casadi_sq(a6);
  a7=5.9498043997358110e+00;
  a7=(a3-a7);
  a7=casadi_sq(a7);
  a6=(a6+a7);
  a6=(a6/a4);
  a6=(-a6);
  a6=exp(a6);
  a5=(a5*a6);
  a2=(a2+a5);
  a5=6.3832775535891240e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=7.0035055631763434e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=5.1091147616516142e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=7.6172331453538575e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=2.8311532916414367e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=8.6597571508408251e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=3.2597091742569873e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=8.4727937814357546e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=4.5827676038083531e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=7.8448407254991608e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=4.2577554243750826e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=7.9830310420159520e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=4.0966874416470905e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=8.0561906213483709e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.1715778267476584e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=9.3385154702027116e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.7985924737959141e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=9.0702636793171756e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=6.6478892394993971e-01;
  a5=(a1-a5);
  a5=casadi_sq(a5);
  a6=6.8734440888075952e+00;
  a6=(a3-a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a5=(a5/a4);
  a5=(-a5);
  a5=exp(a5);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=1.1888351106113719e-02;
  a1=(a1-a5);
  a1=casadi_sq(a1);
  a5=9.8018594726413646e+00;
  a3=(a3-a5);
  a3=casadi_sq(a3);
  a1=(a1+a3);
  a1=(a1/a4);
  a1=(-a1);
  a1=exp(a1);
  a0=(a0*a1);
  a2=(a2+a0);
  a0=-8.3618798037120399e+01;
  a2=(a2+a0);
  if (res[0]!=0) res[0][0]=a2;
  return 0;
}

CASADI_SYMBOL_EXPORT int pendulum_ode_constr_h_e_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int pendulum_ode_constr_h_e_fun_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int pendulum_ode_constr_h_e_fun_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void pendulum_ode_constr_h_e_fun_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int pendulum_ode_constr_h_e_fun_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void pendulum_ode_constr_h_e_fun_release(int mem) {
}

CASADI_SYMBOL_EXPORT void pendulum_ode_constr_h_e_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void pendulum_ode_constr_h_e_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int pendulum_ode_constr_h_e_fun_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int pendulum_ode_constr_h_e_fun_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real pendulum_ode_constr_h_e_fun_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* pendulum_ode_constr_h_e_fun_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* pendulum_ode_constr_h_e_fun_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* pendulum_ode_constr_h_e_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s1;
    case 3: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* pendulum_ode_constr_h_e_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int pendulum_ode_constr_h_e_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
