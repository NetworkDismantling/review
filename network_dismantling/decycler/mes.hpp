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




#ifndef MES_H
#define MES_H



#include "proba.hpp"

#include <iostream>

template<int n>
struct Mes_t {
	Mes_t(int depth, real_t val = 0) {
		for (int s = 0; s < n; ++s)
			H[s] = Proba(depth, val);
	}
	int size() const { return H[0].size(); }
	
	Proba H[n];

	real_t maximum() const {
		real_t m = -inf;
		int const depth = size();
		for (int s = 0; s < n; ++s)
			for (int d = 0; d < depth; ++d)
				m = std::max(m, H[s][d]);
		return m;
	}

	Mes_t & operator=(Mes_t const & other) {
		for (int s = 0; s < n; ++s)
			H[s] = other.H[s];
		return *this;
	}

	Mes_t & operator+=(Mes_t const & other) {
		assert(size() == other.size());
		for(int i = 0; i < n; ++i)
			H[i] += other.H[i];
		return *this;
	}

	Mes_t & operator*=(real_t b) {
		for(int i = 0; i < n; ++i)
			H[i] *= b;
		return *this;
	}
	void reduce() {
		real_t m = maximum();
		assert(m > -inf / 3);
		int const depth = size();
		for (int s = 0; s < n; ++s)
			for (int d = 0; d < depth; ++d)
				H[s][d] -= m;
	}

	friend void swap(Mes_t & u, Mes_t & v)
	{
		for(int i = 0; i<n; ++i)
			swap(u.H[i], v.H[i]);
	}
};


template<class R, int n>
void randomize(Mes_t<n> & m, R & mes_real01)
{
	assert("'randomize()' in mes.hpp not implemented!" == 0);
	// warning: implement!
}


template<int n>
std::ostream & operator<<(std::ostream & ost, Mes_t<n> const & m)
{
	for (int i = 0; i < n; ++i) {
		ost << "H[" << i << "]: " << m.H[i] << std::endl;
	}
	return ost;
}


template<int n>
real_t l8dist(Mes_t<n> const & a,  Mes_t<n> const & b)  
{
	real_t l8 = 0;
	for (int i = 0; i < n; ++i)
		l8 = std::max(l8, l8dist(a.H[i], b.H[i]));
	return l8;
}


template<int n>
Mes_t<n> operator+(Mes_t<n> const & a, Mes_t<n> const & b) 
{
	Mes_t<n> out = a;
	out += b;
	return out;
}

template<int n>
Mes_t<n> operator*(real_t cc, Mes_t<n> const & b) 
{
	Mes_t<n> out = b;
	for (int i = 0; i < n; ++i)
		out.H[i] *= cc;
	return out;
}


#endif
