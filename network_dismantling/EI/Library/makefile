#Ask to http://www.network-theory.co.uk/docs/gccintro/gccintro_16.html
# or to http://www.cs.colby.edu/maxwell/courses/tutorials/maketutor/
CC=gcc
#CFLAGS=-Wall -Wno-maybe-uninitialized -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unused-result -g -pg -lm -O3 -lgsl -lgslcblas
CFLAGS=-Wall -Wno-maybe-uninitialized -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unused-result -g -pg -lm -O3
SOURCES=exploimmun.c rng.c networks.c scores.c newman_ziff.c
DEPENDENCIES=$(SOURCES:.c=.h)
GLOBAL_DEP=macros.h node.h
OBJECTS=$(SOURCES:.c=.o)
EXECUTABLE=../exploimmun

	
$(EXECUTABLE): $(OBJECTS) $(SOURCES) $(DEPENDENCIES) $(GLOBAL_DEP)
	$(CC) $(OBJECTS) -o $@ $(CFLAGS)
	
build: $(SOURCES) $(DEPENDENCIES) $(GLOBAL_DEP)
	$(CC) -c $(SOURCES) $(CFLAGS)
	
clean:
	rm -f *.o
