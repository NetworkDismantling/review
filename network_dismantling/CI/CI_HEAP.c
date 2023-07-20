#include<stdio.h>
#include<stdlib.h>
//#include<math.h>

#define MAX_DEGREE 1000000
//#define FILENAME_LENGHT 10000

//Instructions to run the CI code are as follows:
//
//  - to compile the source code, use this command: gcc -o CI CI_HEAP.c -lm -O3
//  - inputs are the file containing the network, and your desired value of L (the radius of the ball you want to consider)
//      - network file must be an adjacency list where the first number on each line is a node ID and the following numbers on that line are its neighbors, e.g.:
//      1 3255 18210 24119 70247
//      2 9205 88665 89859
//      3 3255 328244 25046 41508
//      …
//      3255 1 3 34913 73168
//
//  - to run the code, use this command: ./CI <network_filename> <L>
//  - values of (q,G) are printed on the screen for fraction of removed nodes q, and an output file “Influencers.txt” is created listing the influencers in the network by decreasing influence


typedef enum {
    OFF, ON
} off_on;
typedef enum {
    OUT, IN
} out_in;
typedef enum {
    NO, YES
} no_yes;
typedef long int int_t;
typedef double dbl_t;

//Node struct
typedef struct {
    out_in n;
    int_t deg, compNum;
} varNode;

//Heap struct
typedef struct {
    int_t CI;
    int_t id;
} vertex;
typedef struct {
    int_t num_data;
    vertex *node;
} Heap;

//FIT struct
typedef struct {
    double fit;
    int_t id;
} FIT;

//mergesort
void merge(FIT *a, FIT *b, FIT *c, int_t m, int_t n) {
    int_t i, j, k;
    i = 0;
    j = 0;
    k = 0;
    while (i < m && j < n) {
        if (a[i].fit < b[j].fit) {
            c[k].fit = a[i].fit;
            c[k].id = a[i].id;
            k++;
            i++;
        } else {
            c[k].fit = b[j].fit;
            c[k].id = b[j].id;
            k++;
            j++;
        }
    }
    while (i < m) {
        c[k].fit = a[i].fit;
        c[k].id = a[i].id;
        k++;
        i++;
    }
    while (j < n) {
        c[k].fit = b[j].fit;
        c[k].id = b[j].id;
        k++;
        j++;
    }
}

void sort(FIT *F, int_t n) {
    int_t j, k, a = 1, inc = 0, cnt = 0, lenght = 0, temp = n;
    FIT *wf, *yf;
    while (temp != 0) {
        while (a < temp - a)
            a *= 2;
        lenght += a;
        wf = calloc(a, sizeof(FIT));
        for (k = 1; k < a; k *= 2) {
            for (j = 0; j < a - k; j += 2 * k) {
                merge(F + j + inc, F + j + k + inc, wf + j, k, k);
            }
            for (j = 0; j < a; ++j) {
                F[j + inc].fit = wf[j].fit;
                F[j + inc].id = wf[j].id;
            }
        }
        ++cnt;
        free(wf);
        inc += a;
        temp -= a;
        if (cnt >= 2) {
            yf = calloc(lenght, sizeof(FIT));
            merge(F, F + lenght - a, yf, lenght - a, a);
            for (j = 0; j < lenght; ++j) {
                F[j].fit = yf[j].fit;
                F[j].id = yf[j].id;
            }
            free(yf);
        }
        a = 1;
    }
}

//get number of nodes from file
int_t get_num_nodes(const char *NETWORK) {
    int_t node, N;
    int n;
    char line[MAX_DEGREE], *start;
    FILE *list;
    list = fopen(NETWORK, "r");
    N = 0;
    while (fgets(line, MAX_DEGREE, list) != NULL) {
        start = line;
        while (sscanf(start, "%ld%n", &node, &n) == 1) {
            start += n;
        }
        N++;
    }
    fclose(list);
    return N;
}

//get adjacency list
int_t **makeRandomGraph(const char *NETWORK, int_t N) {
    int_t i, j, node;
    int_t *deg, **adj_list;
    int n;
    char line[MAX_DEGREE], *start;
    FILE *list;

    deg = (int_t *) calloc(N + 1, sizeof(int_t));
    adj_list = (int_t **) calloc(N + 1, sizeof(int_t *));
    adj_list[0] = (int_t *) calloc(1, sizeof(int_t));
    //count number of neighbours
    list = fopen(NETWORK, "r");
    i = 1;
    while (fgets(line, MAX_DEGREE, list) != NULL) {
        start = line;
        while (sscanf(start, "%ld%n", &node, &n) == 1) {
            start += n;
            deg[i]++;
        }
        adj_list[i] = (int_t *) calloc(deg[i], sizeof(long int));
        deg[i]--;
        i++;
    }
    fclose(list);
    list = fopen(NETWORK, "r");
    i = 1;
    //fill adjacency list
    while (fgets(line, MAX_DEGREE, list) != NULL) {
        start = line;
        j = 0;
        while (sscanf(start, "%ld%n", &node, &n) == 1) {
            start += n;
            adj_list[i][j] = node;
            j++;
        }
        adj_list[i][0] = deg[i];
        i++;
    }
    fclose(list);
    free(deg);
    return adj_list;
}

//heapify
void heapify(Heap *Heap_ptr, int_t index, int_t *heap_pos) {
    int_t left, right, max_index, temp_pos;
    vertex temp_node;

    left = 2 * index;
    right = left + 1;
    //Compare sons with father
    if (left < Heap_ptr->num_data) {
        if (Heap_ptr->node[left].CI > Heap_ptr->node[index].CI) {
            max_index = left;
        } else {
            max_index = index;
        }
        if (right < Heap_ptr->num_data) {
            if (Heap_ptr->node[right].CI > Heap_ptr->node[max_index].CI) {
                max_index = right;
            }
        }
        //IF necessary swap father and son
        if (max_index != index) {
            temp_node = Heap_ptr->node[index];
            temp_pos = heap_pos[Heap_ptr->node[index].id];

            heap_pos[Heap_ptr->node[index].id] = max_index;
            heap_pos[Heap_ptr->node[max_index].id] = temp_pos;

            Heap_ptr->node[index] = Heap_ptr->node[max_index];
            Heap_ptr->node[max_index] = temp_node;

            heapify(Heap_ptr, max_index, heap_pos);
        }
    }
}

void build_heap(Heap *Heap_ptr, int_t *heap_pos) {
    int_t i;
    //Build the heap starting with the father of the last element
    for (i = (int_t) (Heap_ptr->num_data / 2); i > 0; --i)
        heapify(Heap_ptr, i, heap_pos);
}

//get CI for node i
int_t get_CI(int_t i, varNode *Node, int_t N, int_t **Graph, int L, int_t *queue, int_t *check, int_t *lenght) {
    int_t *r, *w, temp, delta, cnt, k, neigh, index, deg, CI;
    int s;

    if (Node[i].deg == 0 || Node[i].deg == 1) { return 0; }
    else {
        queue[0] = i;
        check[i] = ON;
        r = queue;
        w = queue + 1;
        temp = 1;
        delta = 1;
        lenght[0] = 1;
        s = 1;
        cnt = 0;
        while (r != w) {
            if (s <= L) {
                deg = Graph[*r][0];
                for (k = 1; k <= deg; k++) {
                    neigh = Graph[*r][k];
                    if ((Node[neigh].n == IN) && (check[neigh] == OFF)) {
                        queue[temp++] = neigh;
                        check[neigh] = ON;
                        lenght[s] += 1;
                    }
                }
            }
            r += 1;
            w += temp - delta;
            delta = temp;
            cnt += 1;
            if (cnt == lenght[s - 1]) {
                s++;
                cnt = 0;
            }
        }
        index = 0;
        for (s = 0; s < L; s++)
            index += lenght[s];

        CI = 0;
        for (k = index; k < (index + lenght[L]); k++) {
            CI += (Node[queue[k]].deg - 1);
        }
        CI *= (Node[i].deg - 1);
        for (k = 0; k < temp; k++)
            check[queue[k]] = OFF;
        for (s = 0; s <= L; s++)
            lenght[s] = 0;
        return CI;
    }
}

//get list of nodes to update and store them in queue[]
void
get_listNodeToUpdate(int_t i, varNode *Node, int_t N, int_t **Graph, int L, int_t *queue, int_t *check, int_t *lenght) {
    int_t *r, *w, temp, delta, cnt, k, neigh, deg, lenght_list;
    int s;

    queue[0] = i;
    check[i] = ON;
    r = queue;
    w = queue + 1;
    temp = 1;
    delta = 1;
    lenght[0] = 1;
    s = 1;
    cnt = 0;
    while (r != w) {
        if (s <= L) {
            deg = Graph[*r][0];
            for (k = 1; k <= deg; k++) {
                neigh = Graph[*r][k];
                if ((Node[neigh].n == IN) && (check[neigh] == OFF)) {
                    queue[temp++] = neigh;
                    check[neigh] = ON;
                    lenght[s] += 1;
                }
            }
        }
        r += 1;
        w += temp - delta;
        delta = temp;
        cnt += 1;
        if (cnt == lenght[s - 1]) {
            s++;
            cnt = 0;
        }
    }
    lenght_list = 0;
    for (s = 1; s <= L; s++)
        lenght_list += lenght[s];
    queue[0] = lenght_list;      // Number of nodes to update
    for (k = 0; k < temp; k++)
        check[queue[k]] = OFF;
    for (s = 0; s <= L; s++)
        lenght[s] = 0;
}

//label components
void label_components(varNode *Node, int_t **Graph, int_t N, int_t *queue) {
    int_t i, k, compNumber, temp, delta, neigh, *r, *w;

    for (i = 1; i <= N; i++)
        Node[i].compNum = 0;

    compNumber = 0;
    for (i = 1; i <= N; i++) {
        if (Node[i].compNum == 0 && Node[i].n == IN) {
            compNumber++;
            Node[i].compNum = compNumber;
            queue[0] = i;
            r = queue;
            w = queue + 1;
            temp = 1;
            delta = 1;
            while (r != w) {
                for (k = 1; k <= Graph[*r][0]; k++) {
                    neigh = Graph[*r][k];
                    if (Node[neigh].compNum == 0 && Node[neigh].n == IN) {
                        Node[neigh].compNum = compNumber;
                        queue[temp++] = neigh;
                    }
                }
                r += 1;
                w += temp - delta;
                delta = temp;
            }
        }
    }
}

//count how many clusters "i" would join if inserted
int_t how_many_comp_would_join(int_t i, varNode *Node, int_t N, int_t **Graph) {
    int_t j, k, nj, deg_i, *flag;
    int_t num_joint_comp;
    no_yes choose_nj;

    if (Node[i].n == OUT) {
        num_joint_comp = 1;
        deg_i = Graph[i][0];
        flag = (int_t *) calloc(deg_i + 1, sizeof(int_t));

        for (j = 1; j <= deg_i; j++)
            flag[j] = N + 1;  //A number larger than N

        for (j = 1; j <= deg_i; j++) {
            nj = Graph[i][j];
            choose_nj = YES;
            for (k = 1; k <= num_joint_comp; k++) {
                if (flag[k] == Node[nj].compNum)
                    choose_nj = NO;
            }
            if (Node[nj].n == IN && choose_nj == YES) {
                flag[num_joint_comp] = Node[nj].compNum;
                num_joint_comp++;
            }
        }
        free(flag);
        return (num_joint_comp - 1);
    } else { return N + 1; } //A number larger than N
}

//compute G
int_t bigCompNodes(varNode *Node, int_t **Graph, int_t N, int_t *queue) {
    int_t i, k, compNumber, size_largest_comp, *Size_comp, temp, delta, neigh, *r, *w;

    for (i = 1; i <= N; i++) {
        Node[i].compNum = 0;
    }

    compNumber = 0;
    for (i = 1; i <= N; i++) {
        if (Node[i].compNum <= 0 && Node[i].n == IN) {
            compNumber++;
            Node[i].compNum = compNumber;
            queue[0] = i;
            r = queue;
            w = queue + 1;
            temp = 1;
            delta = 1;
            while (r != w) {
                for (k = 1; k <= Graph[*r][0]; k++) {
                    neigh = Graph[*r][k];
                    if (Node[neigh].compNum == 0 && Node[neigh].n == IN) {
                        Node[neigh].compNum = compNumber;
                        queue[temp++] = neigh;
                    }
                }
                r += 1;
                w += temp - delta;
                delta = temp;
            }
        }
    }
    Size_comp = (int_t *) calloc(compNumber + 1, sizeof(int_t));
    for (i = 1; i <= N; i++)
        Size_comp[Node[i].compNum] = Size_comp[Node[i].compNum] + 1;
    size_largest_comp = Size_comp[1];
    for (i = 1; i <= compNumber; i++) {
        if (size_largest_comp < Size_comp[i])
            size_largest_comp = Size_comp[i];
    }
    free(Size_comp);
    return size_largest_comp;
}

//reinsert nodes
int_t *reinsert(varNode *Node, int_t NumRemoved, int_t *list_removed, int_t N, int_t **Graph) {
    int_t i, k, cnt, influenc_cnt, toBeInserted, first, *queue, *list_influencer, DECIM_STEP;
    FIT *fit;

    queue = (int_t *) calloc(N + 1, sizeof(int_t));
    list_influencer = (int_t *) calloc(NumRemoved + 1, sizeof(int_t));
    fit = (FIT *) calloc(NumRemoved, sizeof(FIT));

    if (N <= 1000) {
        DECIM_STEP = 1;
    } else {
        DECIM_STEP = (int_t) ((0.001) * N);
    }

    label_components(Node, Graph, N, queue);  //assign nodes cluster labels
    for (i = 0; i < NumRemoved; i++) {
        fit[i].fit = how_many_comp_would_join(list_removed[i], Node, N, Graph);
        fit[i].id = list_removed[i];
    }
    sort(fit, NumRemoved);

    cnt = NumRemoved;
    influenc_cnt = NumRemoved + 1;
    first = 0;
    while (cnt > 0) {
        toBeInserted = fit[first].id;
        Node[toBeInserted].n = IN;
        first++;
        cnt--;
        influenc_cnt--;
        list_influencer[influenc_cnt] = toBeInserted;

        if (!(cnt % DECIM_STEP)) {
            label_components(Node, Graph, N, queue);
            k = 0;
            for (i = 0; i < NumRemoved; i++) {
                if (Node[list_removed[i]].n == OUT) {
                    fit[k].fit = how_many_comp_would_join(list_removed[i], Node, N, Graph);
                    fit[k].id = list_removed[i];
                    k++;
                }
            }
            sort(fit, cnt);
            first = 0;
        }

        //if(!(cnt % (int)(0.005*N))) {
        /*
        if(!(cnt % 1)) {
            fprintf(stdout, "%f %f\n", (dbl_t)cnt/N, (dbl_t)bigCompNodes(Node, Graph, N, queue)/N);
            fflush(stdout);
        }
         */

    }
    list_influencer[0] = NumRemoved;
    free(queue);
    free(fit);
    return list_influencer;
}

//get influencers
int_t *get_influencers(varNode *Node, int_t N, int_t **Graph, int L, int_t STOP_CONDITION) { // int ntwk) {
    int_t i, j, cnt, toBeRemoved, currentNode, pos_currentNode, NumNodesToUpdate, NumLink, CI_ave;
    int_t *heap_pos, *queue, *check, *lenght, *lenght_plus1, *listNodeToUpdate, *listNodeRemoved;
    int_t *listInfluencers;
    Heap heap;
    int_t LCC_size;
//    dbl_t STOP_CONDITION;
    int_t STEP_CHECK;

    queue = (int_t *) calloc(N + 1, sizeof(int_t));
    check = (int_t *) calloc(N + 1, sizeof(int_t));
    lenght = (int_t *) calloc(L + 1, sizeof(int_t));
    lenght_plus1 = (int_t *) calloc(L + 2, sizeof(int_t));
    listNodeToUpdate = (int_t *) calloc(N + 1, sizeof(int_t));
    listNodeRemoved = (int_t *) calloc(N + 1, sizeof(int_t));
    heap.num_data = (N + 1);
    heap.node = (vertex *) calloc(heap.num_data, sizeof(vertex));
    heap_pos = (int_t *) calloc(heap.num_data, sizeof(int_t));

    NumLink = 0;
    //Init Node variables
    for (i = 1; i <= N; i++) {
        Node[i].n = IN;
        Node[i].deg = Graph[i][0];
        NumLink += Graph[i][0];
    }
    CI_ave = 0;
    for (i = 1; i <= N; i++) {
        heap.node[i].CI = get_CI(i, Node, N, Graph, L, queue, check, lenght);
        heap.node[i].id = i;
        heap_pos[i] = i;
        CI_ave += heap.node[i].CI;
    }
    build_heap(&heap, heap_pos);

    //LCC_size = pow( ((dbl_t)CI_ave)/NumLink, 1./(L+1.) );
    //STOP_CONDITION = 1.-1./(L+1.);

    if (N >= 1000) {
////        STOP_CONDITION = 0.01;
        STEP_CHECK = (int_t) (0.01 * N);
    } else {
////        STOP_CONDITION = 1. / ((dbl_t) N);
        STEP_CHECK = 1;
    }
//    printf("STEP CHECK %ld\n", STEP_CHECK);

//    LCC_size = (dbl_t) bigCompNodes(Node, Graph, N, queue) / N;
    LCC_size = bigCompNodes(Node, Graph, N, queue);
//    printf("LCC_size %f\n", LCC_size);

    cnt = 0;
    while ((LCC_size > STOP_CONDITION) && (cnt < N)) {
        toBeRemoved = heap.node[1].id;
        Node[toBeRemoved].n = OUT;
        CI_ave -= heap.node[1].CI;

        //Swap first and last
        heap_pos[heap.node[heap.num_data - 1].id] = 1;
        heap_pos[heap.node[1].id] = heap.num_data;
        heap.node[1] = heap.node[heap.num_data - 1];
        heap.num_data--;
        heapify(&heap, 1, heap_pos);

        //Decrease degree of neighbours
        for (j = 1; j <= Graph[toBeRemoved][0]; j++)
            Node[Graph[toBeRemoved][j]].deg--;

        //Find nodes to recalculate
        get_listNodeToUpdate(toBeRemoved, Node, N, Graph, L + 1, listNodeToUpdate, check, lenght_plus1);
        NumNodesToUpdate = listNodeToUpdate[0];

        //Recalculate CI for relevant nodes
        for (j = 1; j <= NumNodesToUpdate; j++) {
            currentNode = listNodeToUpdate[j];
            pos_currentNode = heap_pos[currentNode];
            CI_ave -= heap.node[pos_currentNode].CI;
            heap.node[pos_currentNode].CI = get_CI(currentNode, Node, N, Graph, L, queue, check, lenght);
            CI_ave += heap.node[pos_currentNode].CI;
            heapify(&heap, pos_currentNode, heap_pos);
        }
        LCC_size = bigCompNodes(Node, Graph, N, queue);
//        printf("LCC_size %d\n", LCC_size);

//        //LCC_size = pow( ((dbl_t)CI_ave)/NumLink, 1./(L+1.) );
        if (!(cnt % STEP_CHECK)) {
////            LCC_size = (dbl_t) bigCompNodes(Node, Graph, N, queue) / N;
            LCC_size = bigCompNodes(Node, Graph, N, queue);
//            //fprintf(stdout, "%f %f\n", (dbl_t)cnt/N, LCC_size);
//            //fflush(stdout);
//            //Kate's edit: write GC size etc. to file
////             char fname_g[FILENAME_LENGHT];
////             sprintf(fname_g,"GC_%d_lvl_%d.txt",ntwk,L);
////             FILE *f = fopen(fname_g, "a+");
////             fprintf(f, "%f %f\n", (dbl_t)cnt/N, LCC_size);
////             fclose(f);
//            //
//            printf("LCC_size %f\n", LCC_size);
//
        }
        listNodeRemoved[cnt] = toBeRemoved;
        cnt++;
    }

    fprintf(stdout, "\t\t\t\t### FINISHING ###\n\n");

    //with reinsertion
    listInfluencers = reinsert(Node, cnt, listNodeRemoved, N, Graph);
//    return listNodeRemoved;

    //without reinsertion
    /*
    listInfluencers = (int_t *)calloc(cnt+1, sizeof(int_t));
    for(i = 1; i <= cnt; i++)
        listInfluencers[i]  = listNodeRemoved[i-1];
    listInfluencers[0] = cnt;
     */
    ///////////////

    free(queue);
    free(check);
    free(lenght);
    free(lenght_plus1);
    free(listNodeToUpdate);
    free(listNodeRemoved);
    free(heap.node);
    free(heap_pos);

    return listInfluencers;
}


/////MAIN
int main(int argc, char *argv[]) {
    //G,
    int_t N;
    int_t **Graph;
    int_t *listInfluencers;
    int_t stop_condition;
//    dbl_t stop_condition;

    int L;
//    int ntwk;
    const char *network;
    const char *fname_infl;

//    char fname_infl[FILENAME_LENGHT];

//    if (argc != 3) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s <NETWORK_FILENAME> <L> <STOP_CONDITION> <OUTPUT_FILE>\n", argv[0]);
        exit(1);
    }
    network = argv[1];
    L = atoi(argv[2]);
//    stop_condition = atof(argv[3]);
    stop_condition = atoi(argv[3]);

//    ntwk = atoi(argv[1]);
//    sprintf(fname_infl, "INFLUENCERS_%d_lvl_%d.txt", ntwk, L);
//    sprintf(fname_infl, "INFLUENCERS_%s_lvl_%d.txt", network, L);
    fname_infl = argv[4];

    N = get_num_nodes(network);
    varNode *Node;
    Node = (varNode *) calloc(N + 1, sizeof(varNode));
    Graph = makeRandomGraph(network, N);

    fprintf(stdout, "\n\n\t\t\t### COMPUTING ###\n\n");
    fflush(stdout);

    //GET INFLUENCERS
    listInfluencers = get_influencers(Node, N, Graph, L, stop_condition); //, ntwk);

    fprintf(stdout, "\t\t\t\t### NETWORK HAS %ld NODES ###\n\n", N);
    fprintf(stdout, "\t\t\t\t### USING L %d ###\n\n", L);
    fprintf(stdout, "\t\t\t\t### USING THRESHOLD %ld ###\n\n", stop_condition);
    fprintf(stdout, "\t\t\t\t### NETWORK %s DONE ###\n\n", network);
    fprintf(stdout, "\t\t\t\t### REMOVED %ld NODES ###\n\n", listInfluencers[0]);
    fprintf(stdout, "\t\t\t\t### OUTPUT SENT TO --> %s ###\n\n", fname_infl);
    fflush(stdout);

    //WRITE INFLUENCERS ON A FILE

    FILE *list_inf = fopen(fname_infl, "w");
//    fprintf(list_inf, "# LIST OF INFLUENCERS (IN DECREASING ORDER OF INFLUENCE)\n\n");
//    fprintf(list_inf, "# NUMBER OF NODES = %ld\n", N);
//    fprintf(list_inf, "# CI LEVEL L = %d\n\n", L);
//    fprintf(list_inf, "# $1 = SCORE  $2 = NODE_ID $3 = DEGREE $4 = COMPONENT\n\n");
//    for (i = 1; i <= listInfluencers[0]; i++)
//        fprintf(list_inf, "%ld\t %ld\t %ld\t %ld\n", i, listInfluencers[i], Graph[listInfluencers[i]][0],
//                Node[i].compNum);
    for (int_t i = 1; i <= listInfluencers[0]; i++)
        fprintf(list_inf, "%ld %ld\n", i, listInfluencers[i]);

    fclose(list_inf);
    free(listInfluencers);

    /*
	fprintf(stdout, "-- NUMBER OF NODES = %ld\n", N);
	fprintf(stdout, "-- CI LEVEL L = %d\n\n", L);
	fprintf(stdout, "-- LIST OF INFLUENCERS (IN DECREASING ORDER OF INFLUENCE - TOP INFLUENCER = #1 )\n\n");
	fprintf(stdout, " INFLUENCER #\t\t NODE_ID\t DEGREE\n\n");
	for(i = 1; i <= listInfluencers[0]; i++)
		fprintf(stdout, "\t%ld\t\t %ld\t\t %ld\t\n", i, listInfluencers[i], Graph[listInfluencers[i]][0]);
	*/
    return 0;
}









