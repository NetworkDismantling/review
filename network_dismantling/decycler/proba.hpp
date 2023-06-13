
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



#ifndef PROBA_H
#define PROBA_H

#include <iostream>
#include <assert.h>
#include <math.h>
#include <cstring>

#include "real_type.hpp"

real_t const inf = 1e20;


class Proba {
public:	
	//Proba(int depth) : p_(new real_t[depth]), depth(depth) { assert(0 && depth); }
	Proba() : p_(0), depth(0) {}
	Proba(int depth, real_t val = 0) : p_(new real_t[depth]), depth(depth) {
		assert(depth >= 0);
		for (int d = depth; d--; )
			p_[d] = val;
	}
	Proba(Proba const & other) : p_(0), depth(0) {
		*this = other;
	}
	Proba & operator=(Proba const & other) {
		if (depth != other.depth) {
			delete [] p_;
			depth = other.depth;
			p_ = new real_t[other.depth];
		}
		memcpy(p_, other.p_, depth * sizeof(real_t));
		return *this;
	}
	~Proba() { delete[] p_; }
	real_t & operator[](int d) {
		return p_[d];
	}
	real_t const & operator[](int d) const{
		return p_[d];
	}
	friend std::ostream & operator<<(std::ostream & ost, Proba const & p) {
		for (int d = 0; d < p.depth; ++d)
			ost << p[d] << " ";
		return ost;
	}
	int size() const { return depth; }
	Proba & operator+=(Proba const & b) {
		for (int d = depth; d--; )
			p_[d] += b[d];
		return *this;
	}

	Proba & operator*=(real_t b) {
		for (int d = depth; d--; )
			p_[d] *= b;
		return *this;
	}

	friend real_t l8dist(Proba const & a, Proba const & b) {
		real_t n = 0;
		for (int d = a.depth; d--;)
			n = std::max(n, fabs_r(a[d] - b[d]));
		return n;
	}

	friend void swap(Proba & A, Proba & B) {
		std::swap(A.depth, B.depth);
		std::swap(A.p_, B.p_);
	}
	
private:
	real_t * p_;
	int depth;
};



real_t l8dist(Proba const & a,  Proba const & b);



#endif
