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
#define casadi_fmax CASADI_PREFIX(fmax)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)

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

casadi_real casadi_fmax(casadi_real x, casadi_real y) {
/* Pre-c99 compatibility */
#if __STDC_VERSION__ < 199901L
  return x>y ? x : y;
#else
  return fmax(x, y);
#endif
}

static const casadi_int casadi_s0[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s1[3] = {0, 0, 0};
static const casadi_int casadi_s2[5] = {1, 1, 0, 1, 0};

/* pendulum_ode_constr_h_e_fun:(i0[2],i1[],i2[],i3[])->(o0) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a3, a4, a5, a6, a7, a8, a9;
  a0=1.;
  a1=2.4694593250751495e-01;
  a2=-1.2212357521057129e+00;
  a3=-3.6125284433364868e-01;
  a4=8.6265653371810913e-02;
  a5=0.;
  a6=-1.3720893859863281e+00;
  a7=5.6118935346603394e-01;
  a8=arg[0]? arg[0][0] : 0;
  a7=(a7*a8);
  a9=2.5249761343002319e-01;
  a10=arg[0]? arg[0][1] : 0;
  a9=(a9*a10);
  a7=(a7+a9);
  a6=(a6+a7);
  a6=casadi_fmax(a5,a6);
  a4=(a4*a6);
  a7=3.2882618904113770e-01;
  a9=7.1343946456909180e-01;
  a11=-4.8767548799514771e-01;
  a11=(a11*a8);
  a12=-1.3966478407382965e-01;
  a12=(a12*a10);
  a11=(a11+a12);
  a9=(a9+a11);
  a9=casadi_fmax(a5,a9);
  a7=(a7*a9);
  a4=(a4+a7);
  a7=2.1134893596172333e-01;
  a11=-4.7655012458562851e-02;
  a12=-1.5156761407852173e+00;
  a12=(a12*a8);
  a13=-6.8755787611007690e-01;
  a13=(a13*a10);
  a12=(a12+a13);
  a11=(a11+a12);
  a11=casadi_fmax(a5,a11);
  a7=(a7*a11);
  a4=(a4+a7);
  a7=7.4575997889041901e-02;
  a12=-3.0806565284729004e-01;
  a13=-5.2143085002899170e-01;
  a13=(a13*a8);
  a14=2.1019417792558670e-02;
  a14=(a14*a10);
  a13=(a13+a14);
  a12=(a12+a13);
  a12=casadi_fmax(a5,a12);
  a7=(a7*a12);
  a4=(a4+a7);
  a7=1.4552248716354370e+00;
  a13=-1.2192111015319824e+00;
  a14=1.6669129133224487e+00;
  a14=(a14*a8);
  a15=6.9894552230834961e-02;
  a15=(a15*a10);
  a14=(a14+a15);
  a13=(a13+a14);
  a13=casadi_fmax(a5,a13);
  a7=(a7*a13);
  a4=(a4+a7);
  a7=-3.8784545660018921e-01;
  a14=7.1210175752639771e-01;
  a15=1.1753002405166626e+00;
  a15=(a15*a8);
  a16=9.6420496702194214e-02;
  a16=(a16*a10);
  a15=(a15+a16);
  a14=(a14+a15);
  a14=casadi_fmax(a5,a14);
  a7=(a7*a14);
  a4=(a4+a7);
  a7=-1.4746811985969543e-01;
  a15=-3.8766384124755859e-01;
  a16=1.3464483618736267e-01;
  a16=(a16*a8);
  a17=-1.6693690419197083e-01;
  a17=(a17*a10);
  a16=(a16+a17);
  a15=(a15+a16);
  a15=casadi_fmax(a5,a15);
  a7=(a7*a15);
  a4=(a4+a7);
  a7=1.3712064921855927e-01;
  a16=-1.2803181409835815e+00;
  a17=7.9401040077209473e-01;
  a17=(a17*a8);
  a18=6.9307106733322144e-01;
  a18=(a18*a10);
  a17=(a17+a18);
  a16=(a16+a17);
  a16=casadi_fmax(a5,a16);
  a7=(a7*a16);
  a4=(a4+a7);
  a7=-3.4024998545646667e-01;
  a17=-3.2605579495429993e-01;
  a18=1.3646727800369263e+00;
  a18=(a18*a8);
  a19=-2.0322518050670624e-01;
  a19=(a19*a10);
  a18=(a18+a19);
  a17=(a17+a18);
  a17=casadi_fmax(a5,a17);
  a7=(a7*a17);
  a4=(a4+a7);
  a7=2.5037848949432373e-01;
  a18=-6.7530351877212524e-01;
  a19=-8.1534093618392944e-01;
  a19=(a19*a8);
  a8=6.2817841768264771e-01;
  a8=(a8*a10);
  a19=(a19+a8);
  a18=(a18+a19);
  a5=casadi_fmax(a5,a18);
  a7=(a7*a5);
  a4=(a4+a7);
  a3=(a3+a4);
  a3=tanh(a3);
  a2=(a2*a3);
  a3=1.4550824165344238e+00;
  a4=6.7737632989883423e-01;
  a7=-3.1192028522491455e-01;
  a7=(a7*a6);
  a18=2.4334500730037689e-01;
  a18=(a18*a9);
  a7=(a7+a18);
  a18=-3.6633327603340149e-01;
  a18=(a18*a11);
  a7=(a7+a18);
  a18=-2.3357148468494415e-01;
  a18=(a18*a12);
  a7=(a7+a18);
  a18=-1.5612249374389648e+00;
  a18=(a18*a13);
  a7=(a7+a18);
  a18=6.5471488237380981e-01;
  a18=(a18*a14);
  a7=(a7+a18);
  a18=1.9198699295520782e-01;
  a18=(a18*a15);
  a7=(a7+a18);
  a18=-3.7441164255142212e-01;
  a18=(a18*a16);
  a7=(a7+a18);
  a18=1.3319686055183411e-02;
  a18=(a18*a17);
  a7=(a7+a18);
  a18=-1.2278471142053604e-01;
  a18=(a18*a5);
  a7=(a7+a18);
  a4=(a4+a7);
  a4=tanh(a4);
  a3=(a3*a4);
  a2=(a2+a3);
  a3=-9.2180639505386353e-01;
  a4=-2.8827148675918579e-01;
  a7=2.3948979377746582e-01;
  a7=(a7*a6);
  a18=1.0639877617359161e-01;
  a18=(a18*a9);
  a7=(a7+a18);
  a18=3.0788600444793701e-01;
  a18=(a18*a11);
  a7=(a7+a18);
  a18=-8.6998961865901947e-02;
  a18=(a18*a12);
  a7=(a7+a18);
  a18=1.4733847379684448e+00;
  a18=(a18*a13);
  a7=(a7+a18);
  a18=-4.6692219376564026e-01;
  a18=(a18*a14);
  a7=(a7+a18);
  a18=1.3953002169728279e-02;
  a18=(a18*a15);
  a7=(a7+a18);
  a18=2.4190627038478851e-01;
  a18=(a18*a16);
  a7=(a7+a18);
  a18=-3.9548331499099731e-01;
  a18=(a18*a17);
  a7=(a7+a18);
  a18=1.7315676808357239e-01;
  a18=(a18*a5);
  a7=(a7+a18);
  a4=(a4+a7);
  a4=tanh(a4);
  a3=(a3*a4);
  a2=(a2+a3);
  a3=-8.7652456760406494e-01;
  a4=-8.7926346063613892e-01;
  a7=1.7840011417865753e-01;
  a7=(a7*a6);
  a18=-2.4265293776988983e-01;
  a18=(a18*a9);
  a7=(a7+a18);
  a18=-7.5744533538818359e-01;
  a18=(a18*a11);
  a7=(a7+a18);
  a18=1.0571417957544327e-01;
  a18=(a18*a12);
  a7=(a7+a18);
  a18=8.8992619514465332e-01;
  a18=(a18*a13);
  a7=(a7+a18);
  a18=-5.6847327947616577e-01;
  a18=(a18*a14);
  a7=(a7+a18);
  a18=3.0416255816817284e-02;
  a18=(a18*a15);
  a7=(a7+a18);
  a18=1.6486300528049469e-01;
  a18=(a18*a16);
  a7=(a7+a18);
  a18=7.3885333538055420e-01;
  a18=(a18*a17);
  a7=(a7+a18);
  a18=6.7774914205074310e-02;
  a18=(a18*a5);
  a7=(a7+a18);
  a4=(a4+a7);
  a4=tanh(a4);
  a3=(a3*a4);
  a2=(a2+a3);
  a3=-1.0742480754852295e+00;
  a4=-4.7725027799606323e-01;
  a7=5.1599234342575073e-01;
  a7=(a7*a6);
  a18=-4.6606868505477905e-02;
  a18=(a18*a9);
  a7=(a7+a18);
  a18=2.6503685116767883e-01;
  a18=(a18*a11);
  a7=(a7+a18);
  a18=-3.5308126360177994e-02;
  a18=(a18*a12);
  a7=(a7+a18);
  a18=1.3563843965530396e+00;
  a18=(a18*a13);
  a7=(a7+a18);
  a18=-3.9679184556007385e-01;
  a18=(a18*a14);
  a7=(a7+a18);
  a18=7.2736114263534546e-02;
  a18=(a18*a15);
  a7=(a7+a18);
  a18=2.0488987863063812e-01;
  a18=(a18*a16);
  a7=(a7+a18);
  a18=-2.0632320642471313e-01;
  a18=(a18*a17);
  a7=(a7+a18);
  a18=-7.5496912002563477e-02;
  a18=(a18*a5);
  a7=(a7+a18);
  a4=(a4+a7);
  a4=tanh(a4);
  a3=(a3*a4);
  a2=(a2+a3);
  a3=1.3564838171005249e+00;
  a4=5.0116586685180664e-01;
  a7=-7.5275832414627075e-01;
  a7=(a7*a6);
  a18=-2.2989940643310547e-01;
  a18=(a18*a9);
  a7=(a7+a18);
  a18=-5.8688813447952271e-01;
  a18=(a18*a11);
  a7=(a7+a18);
  a18=-2.1139082312583923e-01;
  a18=(a18*a12);
  a7=(a7+a18);
  a18=-2.9960751533508301e-01;
  a18=(a18*a13);
  a7=(a7+a18);
  a18=8.9789718389511108e-01;
  a18=(a18*a14);
  a7=(a7+a18);
  a18=1.0307281464338303e-01;
  a18=(a18*a15);
  a7=(a7+a18);
  a18=-7.5997859239578247e-01;
  a18=(a18*a16);
  a7=(a7+a18);
  a18=5.8913081884384155e-01;
  a18=(a18*a17);
  a7=(a7+a18);
  a18=3.5601262003183365e-02;
  a18=(a18*a5);
  a7=(a7+a18);
  a4=(a4+a7);
  a4=tanh(a4);
  a3=(a3*a4);
  a2=(a2+a3);
  a3=-6.6920042037963867e-01;
  a4=-1.0506165027618408e+00;
  a7=3.8842478394508362e-01;
  a7=(a7*a6);
  a18=-1.5944685041904449e-01;
  a18=(a18*a9);
  a7=(a7+a18);
  a18=-9.8714262247085571e-02;
  a18=(a18*a11);
  a7=(a7+a18);
  a18=4.2080219835042953e-02;
  a18=(a18*a12);
  a7=(a7+a18);
  a18=1.0062869787216187e+00;
  a18=(a18*a13);
  a7=(a7+a18);
  a18=-5.5451357364654541e-01;
  a18=(a18*a14);
  a7=(a7+a18);
  a18=5.3279572725296021e-01;
  a18=(a18*a15);
  a7=(a7+a18);
  a18=3.0669766664505005e-01;
  a18=(a18*a16);
  a7=(a7+a18);
  a18=5.7137513160705566e-01;
  a18=(a18*a17);
  a7=(a7+a18);
  a18=2.2528368234634399e-01;
  a18=(a18*a5);
  a7=(a7+a18);
  a4=(a4+a7);
  a4=tanh(a4);
  a3=(a3*a4);
  a2=(a2+a3);
  a3=1.9514399766921997e-01;
  a4=2.5756785273551941e-01;
  a7=-1.8730789422988892e-01;
  a7=(a7*a6);
  a18=1.4296634495258331e-01;
  a18=(a18*a9);
  a7=(a7+a18);
  a18=-1.2152805924415588e-01;
  a18=(a18*a11);
  a7=(a7+a18);
  a18=-3.6510746926069260e-02;
  a18=(a18*a12);
  a7=(a7+a18);
  a18=-8.6840897798538208e-01;
  a18=(a18*a13);
  a7=(a7+a18);
  a18=2.6751106977462769e-01;
  a18=(a18*a14);
  a7=(a7+a18);
  a18=-3.6905375123023987e-01;
  a18=(a18*a15);
  a7=(a7+a18);
  a18=-1.9492265582084656e-01;
  a18=(a18*a16);
  a7=(a7+a18);
  a18=9.8362393677234650e-02;
  a18=(a18*a17);
  a7=(a7+a18);
  a18=7.8337684273719788e-02;
  a18=(a18*a5);
  a7=(a7+a18);
  a4=(a4+a7);
  a4=tanh(a4);
  a3=(a3*a4);
  a2=(a2+a3);
  a3=-1.4613819122314453e+00;
  a4=-4.4961085915565491e-01;
  a7=4.1983151435852051e-01;
  a7=(a7*a6);
  a18=-1.8351030349731445e-01;
  a18=(a18*a9);
  a7=(a7+a18);
  a18=6.1823415756225586e-01;
  a18=(a18*a11);
  a7=(a7+a18);
  a18=-1.0692720115184784e-01;
  a18=(a18*a12);
  a7=(a7+a18);
  a18=3.5691395401954651e-01;
  a18=(a18*a13);
  a7=(a7+a18);
  a18=-9.5112311840057373e-01;
  a18=(a18*a14);
  a7=(a7+a18);
  a18=-1.0560742579400539e-02;
  a18=(a18*a15);
  a7=(a7+a18);
  a18=8.3648645877838135e-01;
  a18=(a18*a16);
  a7=(a7+a18);
  a18=-3.8328742980957031e-01;
  a18=(a18*a17);
  a7=(a7+a18);
  a18=8.1459152698516846e-01;
  a18=(a18*a5);
  a7=(a7+a18);
  a4=(a4+a7);
  a4=tanh(a4);
  a3=(a3*a4);
  a2=(a2+a3);
  a3=-1.1425738334655762e+00;
  a4=-1.7251883447170258e-01;
  a7=1.8494451045989990e-01;
  a7=(a7*a6);
  a6=4.0682452917098999e-01;
  a6=(a6*a9);
  a7=(a7+a6);
  a6=1.6486254334449768e-01;
  a6=(a6*a11);
  a7=(a7+a6);
  a6=1.0551828891038895e-01;
  a6=(a6*a12);
  a7=(a7+a6);
  a6=1.0233626365661621e+00;
  a6=(a6*a13);
  a7=(a7+a6);
  a6=-3.5425442457199097e-01;
  a6=(a6*a14);
  a7=(a7+a6);
  a6=-3.6323416233062744e-01;
  a6=(a6*a15);
  a7=(a7+a6);
  a6=1.1965297907590866e-01;
  a6=(a6*a16);
  a7=(a7+a6);
  a6=-9.7285419702529907e-02;
  a6=(a6*a17);
  a7=(a7+a6);
  a6=3.6585569381713867e-01;
  a6=(a6*a5);
  a7=(a7+a6);
  a4=(a4+a7);
  a4=tanh(a4);
  a3=(a3*a4);
  a2=(a2+a3);
  a1=(a1+a2);
  a1=(-a1);
  a1=exp(a1);
  a0=(a0+a1);
  a0=(1./a0);
  if (res[0]!=0) res[0][0]=a0;
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
