from datetime import timedelta
from time import time
import dismantler
import sys


if len(sys.argv) < 4:
	print("usage: <net_file> <removals> <out_file> <stop_condition>")
	sys.exit(-1)

net = []
removals = []

f  = open(sys.argv[1])
for l in f:
	w = l.split()
	net.append([int(w[0]), int(w[1])])
f.close()

f  = open(sys.argv[2])
for l in f:
	removals.append(int(l))
f.close()

print ("Graph creation & loading")
g = dismantler.Graph("")
g.loadGraphFromPythonList(net)

#or, even better
#print ("Graph creation & loading")
#g = dismantler.Graph(net)

print("Clone a graph")
g1 = dismantler.Graph(g)

print("Invoking external lcc dismantler on g")
start_time = time()
result = dismantler.lccThresholdDismantler(g, removals, int(sys.argv[4]))
print("External dismantler returned in {}s".format(timedelta(seconds=(time() - start_time))))

f = open(sys.argv[3], "w")
for n,lcc,slcc in result:
	f.write("{0} {1} {2}\n".format(n, lcc, slcc))
f.close()


print("Invoking external lcc dismantler on g1")
start_time = time()
result = dismantler.lccThresholdDismantler(g1, removals, int(sys.argv[4]))
print("External dismantler returned in {}s".format(timedelta(seconds=(time() - start_time))))

f = open(sys.argv[3] + "_g1", "w")
for n,lcc,slcc in result:
	f.write("{0} {1} {2}\n".format(n, lcc, slcc))
f.close()

