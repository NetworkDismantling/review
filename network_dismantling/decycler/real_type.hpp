
/*
 * Decycler, a reinforced Max-Sum algorithm to solve the decycling problem
 * Copyright (C) 2016 Alfredo Braunstein
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation version 2 of the License.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

#ifndef REAL_TYPE_H
#define REAL_TYPE_H

#include <cmath>

#define REAL_T_FUN(g) \
	inline long double g##_r(long double x) { return g##l(x); } \
	inline double g##_r(double x) { return g(x); } \
	inline float g##_r(float x) { return g##f(x); } \
	inline real_t g##_r(int x) { return g##_r(real_t(x)); }


#define REAL_T_FUN2(g) \
	inline long double g##_r(long double x, long double y) { return g##l(x, y); } \
	inline float g##_r(float x, float y) { return g##f(x, y); } \
	inline double g##_r(double x, double y) { return g(x, y); } \
	inline real_t g##_r(long double x, double y) { return g##_r(real_t(x), real_t(y)); } \
	inline real_t g##_r(long double x, int y) { return g##_r(real_t(x), real_t(y)); } \
	inline real_t g##_r(double x, long double y) { return g##_r(real_t(x), real_t(y)); } \
	inline real_t g##_r(double x, int y) { return g##_r(real_t(x), real_t(y)); } \
	inline real_t g##_r(int x, long double y) { return g##_r(real_t(x), real_t(y)); } \
	inline real_t g##_r(int x, double y) { return g##_r(real_t(x), real_t(y)); } \
	inline real_t g##_r(int x, int y) { return g##_r(real_t(x), real_t(y)); }


typedef float real_t;

REAL_T_FUN(exp)
REAL_T_FUN(log)
REAL_T_FUN(sin)
REAL_T_FUN(cos)
REAL_T_FUN(tan)
REAL_T_FUN(asin)
REAL_T_FUN(acos)
REAL_T_FUN(atan)
REAL_T_FUN(sinh)
REAL_T_FUN(cosh)
REAL_T_FUN(tanh)
REAL_T_FUN(asinh)
REAL_T_FUN(acosh)
REAL_T_FUN(atanh)
REAL_T_FUN(sqrt)
REAL_T_FUN(fabs)
REAL_T_FUN2(pow)

#endif // REAL_TYPE_H
