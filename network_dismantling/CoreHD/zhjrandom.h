/*
  -----------
  zhjrandom.h
  
  -----------
  DESCRIPTION
  Random number generator. 

  -----------
  DECLARATION
  Haijun Zhou, Institute of Theoretical Physics,
  the Chinese Academy of Sciences, Beijing, China
*/

#ifndef __ZHJ__
#define __ZHJ__

#include <cmath>

using namespace std;

extern "C" {
extern void dgesvd_(char *jobu, char *jobvt, int *m, int *n, double *a,
                    int *lda, double *s, double *u, int *ldu, double *vt, int *ldvt,
                    double *work, int *lwork, int *info);
}

double gammln(double xx)
/* Returns the value of ln[Gamma(xx)] for xx>0.
   Copied from "NUMERICAL RECIPES IN C: THE ART OF SCIENTIFIC COMPUTING", page 214. */
{
    static double cof[] = {
            76.18009172947146e0,
            -86.50532032941677e0,
            24.01409824083091e0,
            -1.231739572450155e0,
            0.1208650973866179e-2,
            -0.5395239384953e-5
    };

    static double stp = 2.5066282746310005e0;

    double ser = 1.000000000190015e0;
    double x = xx;
    double y = x;
    double tmp = x + 5.5E0;

    tmp -= (x + 0.5E0) * log(tmp);

    for (int j = 0; j < 6; ++j) ser += cof[j] / ++y;

    return -tmp + log(stp * ser / x);
}

class ZHJRANDOMv3 {
public:
    double rdflt();

    double poidev(double);

    double gasdev();

    explicit ZHJRANDOMv3(long i = 0);

private:
    constexpr static const long IM1 = 2147483563;
    constexpr static const long IM2 = 2147483399;
    constexpr static const double AM = (1.0 / IM1);
    constexpr static const long IMM1 = (IM1 - 1);
    constexpr static const int IA1 = 40014;
    constexpr static const int IA2 = 40692;
    constexpr static const int IQ1 = 53668;
    constexpr static const int IQ2 = 52774;
    constexpr static const int IR1 = 12211;
    constexpr static const int IR2 = 3791;
    constexpr static const int NTAB = 32;
    constexpr static const int NDIV = (1 + IMM1 / NTAB);
    constexpr static const double EPS = 1.2e-7;
    constexpr static const double RNMX = 9.9999988e-1; //=1.0-EPS;

    long iv[NTAB]{};
    long idum;
    long idum2;
    long iy;
    double gas_gset{};
    int gas_iset;
    double gas_rsq{};
    double gas_v1{};
    double gas_v2{};
    double gas_fac{};

    double GammlnVal[1000]{};
};


ZHJRANDOMv3::ZHJRANDOMv3(long i) {
    gas_iset = 0;
    idum = i;
    if (idum < 1) idum = 1;
    idum2 = idum;

    for (int j = NTAB + 7; j >= 0; --j) {
        long k = idum / IQ1;
        idum = IA1 * (idum - k * IQ1) - k * IR1;
        if (idum < 0) idum += IM1;
        if (j < NTAB) iv[j] = idum;
    }
    iy = iv[0];

    for (int j = 0; j < 1000; ++j) {
        GammlnVal[j] = gammln(static_cast<double>(j) + 1.0);
    }
}

double ZHJRANDOMv3::rdflt() {
    long k = idum / IQ1;
    idum = IA1 * (idum - k * IQ1) - k * IR1;

    if (idum < 0) idum += IM1;

    k = idum2 / IQ2;

    idum2 = IA2 * (idum2 - k * IQ2) - k * IR2;

    if (idum2 < 0) idum2 += IM2;

    long j = iy / NDIV;
    iy = iv[j] - idum2;
    iv[j] = idum;
    if (iy < 1) iy += IMM1;
    double temp = AM * iy;
    if (temp > RNMX) return RNMX;
    else return temp;
}

double ZHJRANDOMv3::poidev(double xm)
/* Returns as a floating-point number an integer value that is a random
   deviate drawn from a Poisson distribution of mean xm.
   Original code is in "NUMERICAL RECIPES IN C:
   THE ART OF SCIENTIFIC COMPUTING", page 294-295.  */
{
    static double alxm, g, oldm = (-1.0), PI = (3.14159265358979e0), sq;
    double em;
    double internal_value;

    if (xm < 12.0) {
        if (xm != oldm) {
            oldm = xm;
            g = exp(-xm);
        }
        em = -1;
        double t = 1.0;
        do {
            ++em;
            t *= rdflt();
        } while (t > g);
    } else {
        if (xm != oldm) {
            oldm = xm;
            sq = sqrt(2.0 * xm);
            alxm = log(xm);
            g = xm * alxm - gammln(xm + 1.0);
        }
        double y, t;
        do {
            do {
                y = tan(PI * rdflt());
                em = sq * y + xm;
            } while (em < 0.0);
            em = floor(em);

            internal_value = (em < 1000.0 ? GammlnVal[static_cast<int>(em)] : gammln(em + 1.0)); //=gammln(em+1.0);
            t = 0.9 * (1. + y * y) * exp(em * alxm - internal_value - g);
        } while (rdflt() > t);
    }
    return em;
}

double ZHJRANDOMv3::gasdev()
/* Returns a normally distributed deviate with zero mean and unit variance. */
{
    if (gas_iset <= 0) {
        do {
            gas_v1 = 2.0e0 * rdflt() - 1.0e0;
            gas_v2 = 2.0e0 * rdflt() - 1.0e0;
            gas_rsq = gas_v1 * gas_v1 + gas_v2 * gas_v2;
        } while (gas_rsq >= 1.0e0 || gas_rsq == 0.0e0);

        gas_fac = sqrt(-2.0e0 * log(gas_rsq) / gas_rsq);
        gas_gset = gas_v1 * gas_fac;
        gas_iset = 1;
        return gas_v2 * gas_fac;
    } else {
        gas_iset = 0;
        return gas_gset;
    }
}

#endif

