#ifndef OMP_CONFIG_H
#define OMP_CONFIG_H


#ifdef OMP

#include <omp.h>

#else

namespace {

#define omp_init_lock(lock)
#define omp_destroy_lock(lock)
#define omp_set_lock(lock)
#define omp_unset_lock(lock)
#define omp_get_num_procs() 1
#define omp_get_thread_num() 0
#define omp_set_num_threads(num)

}

#endif

#endif

