typedef enum {
    false, true
}
bool;

typedef struct node {
    //immunization parameters
    bool selected;
    int id;
    int n;
    int virtual_degree;
    int dynamic_degree;
    int effective_degree;
    double ci;
    struct node **neigh;

    //newman_ziff parameters
//   int pointer;
    struct node *pointer;
    int cluster_size;
    bool visited;
} Node;
