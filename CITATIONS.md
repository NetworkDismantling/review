# Citations
In this section, we provide the citations for the paper, for the data, for the algorithms and for the libraries used in our project.

## How to cite the paper

> Artime, O., Grassia, M., De Domenico, M. et al. Robustness and resilience of complex networks. Nat Rev Phys (2024). https://doi.org/10.1038/s42254-023-00676-y

BibTex:

```bibtex
@article{artime2024robustness,
	abstract = {Complex networks are ubiquitous: a cell, the human brain, a group of people and the Internet are all examples of interconnected many-body systems characterized by macroscopic properties that cannot be trivially deduced from those of their microscopic constituents. Such systems are exposed to both internal, localized, failures and external disturbances or perturbations. Owing to their interconnected structure, complex systems might be severely degraded, to the point of disintegration or systemic dysfunction. Examples include cascading failures, triggered by an initially localized overload in power systems, and the critical slowing downs of ecosystems which can be driven towards extinction. In recent years, this general phenomenon has been investigated by framing localized and systemic failures in terms of perturbations that can alter the function of a system. We capitalize on this mathematical framework to review theoretical and computational approaches to characterize robustness and resilience of complex networks. We discuss recent approaches to mitigate the impact of perturbations in terms of designing robustness, identifying early-warning signals and adapting responses. In terms of applications, we compare the performance of the state-of-the-art dismantling techniques, highlighting their optimal range of applicability for practical problems, and provide a repository with ready-to-use scripts, a much-needed tool set.},
	author = {Artime, Oriol and Grassia, Marco and De Domenico, Manlio and Gleeson, James P. and Makse, Hern{\'a}n A. and Mangioni, Giuseppe and Perc, Matja{\v z} and Radicchi, Filippo},
	date = {2024/01/08},
	date-added = {2024-01-09 08:22:35 +0100},
	date-modified = {2024-01-09 08:23:03 +0100},
	doi = {10.1038/s42254-023-00676-y},
	id = {Artime2024},
	isbn = {2522-5820},
	journal = {Nature Reviews Physics},
	title = {Robustness and resilience of complex networks},
	url = {https://doi.org/10.1038/s42254-023-00676-y},
	year = {2024},
	bdsk-url-1 = {https://doi.org/10.1038/s42254-023-00676-y}
}
```

RIS:
```ris
TY  - JOUR
AB  - Complex networks are ubiquitous: a cell, the human brain, a group of people and the Internet are all examples of interconnected many-body systems characterized by macroscopic properties that cannot be trivially deduced from those of their microscopic constituents. Such systems are exposed to both internal, localized, failures and external disturbances or perturbations. Owing to their interconnected structure, complex systems might be severely degraded, to the point of disintegration or systemic dysfunction. Examples include cascading failures, triggered by an initially localized overload in power systems, and the critical slowing downs of ecosystems which can be driven towards extinction. In recent years, this general phenomenon has been investigated by framing localized and systemic failures in terms of perturbations that can alter the function of a system. We capitalize on this mathematical framework to review theoretical and computational approaches to characterize robustness and resilience of complex networks. We discuss recent approaches to mitigate the impact of perturbations in terms of designing robustness, identifying early-warning signals and adapting responses. In terms of applications, we compare the performance of the state-of-the-art dismantling techniques, highlighting their optimal range of applicability for practical problems, and provide a repository with ready-to-use scripts, a much-needed tool set.
AU  - Artime, Oriol
AU  - Grassia, Marco
AU  - De Domenico, Manlio
AU  - Gleeson, James P.
AU  - Makse, Hernán A.
AU  - Mangioni, Giuseppe
AU  - Perc, Matjaž
AU  - Radicchi, Filippo
DO  - 10.1038/s42254-023-00676-y
ID  - Artime2024
SN  - 2522-5820
T2  - Nature Reviews Physics
TI  - Robustness and resilience of complex networks
UR  - https://doi.org/10.1038/s42254-023-00676-y
PY  - 2024
ER  - 
```

Also, please cite the original papers of the [algorithms](#algorithms),
of the [Dataset](#dataset), and of the libraries you use (see the [Setup](#setup) section).

## Datasets
See the [Datasets](dataset/README.md) section for more information.

## Algorithms
- Brute Force (BF): 
- [Collective Influence (CI)](#collective-influence-ci)
- [CoreGDM](#coregdm) (to be implemented)
- [CoreHD](#corehd)
- [Ensemble GND (EGND)](#ensemble-gnd-egnd)
- [Explosive Immunization (EI)](#explosive-immunization-ei)
- [FINDER](#finder)
- [Generalized Network Dismantling (GND)](#generalized-network-dismantling-gnd)
- [Graph Dismantling Machine (GDM)](#graph-dismantling-machine-gdm)
- [Min-Sum & greedy reinsertion algorithm](#min-sum--greedy-reinsertion-algorithm)
- [Node heuristics](#node-heuristics)
  - Degree
  - Betweenness
  - Eigenvector centrality
  - PageRank


### Collective Influence (CI)
DOI [10.1038/srep30062](https://doi.org/10.1038/srep30062)
> Morone, F., Min, B., Bo, L., Mari, R., Makse, H.: Collective influence algorithm to find influencers via optimal percolation in massively large social media. Scientific Reports 6 (2016). https://doi.org/10.1038/srep30062

```bibtex
@Article{Morone2016,
    author={Morone, Flaviano
    and Min, Byungjoon
    and Bo, Lin
    and Mari, Romain
    and Makse, Hern{\'a}n A.},
    title={Collective Influence Algorithm to find influencers via optimal percolation in massively large social media},
    journal={Scientific Reports},
    year={2016},
    month={Jul},
    day={26},
    volume={6},
    number={1},
    pages={30062},
    abstract={We elaborate on a linear-time implementation of Collective-Influence (CI) algorithm introduced by Morone, Makse, Nature 524, 65 (2015) to find the minimal set of influencers in networks via optimal percolation. The computational complexity of CI is O(N log N) when removing nodes one-by-one, made possible through an appropriate data structure to process CI. We introduce two Belief-Propagation (BP) variants of CI that consider global optimization via message-passing: CI propagation (CIP) and Collective-Immunization-Belief-Propagation algorithm (CIBP) based on optimal immunization. Both identify a slightly smaller fraction of influencers than CI and, remarkably, reproduce the exact analytical optimal percolation threshold obtained in Random Struct. Alg. 21, 397 (2002) for cubic random regular graphs, leaving little room for improvement for random graphs. However, the small augmented performance comes at the expense of increasing running time to O(N2), rendering BP prohibitive for modern-day big-data. For instance, for big-data social networks of 200 million users (e.g., Twitter users sending 500 million tweets/day), CI finds influencers in 2.5{\thinspace}hours on a single CPU, while all BP algorithms (CIP, CIBP and BDP) would take more than 3,000 years to accomplish the same task.},
    issn={2045-2322},
    doi={10.1038/srep30062},
    url={https://doi.org/10.1038/srep30062}
}
```

### CoreGDM
DOI: [10.1007/978-3-031-28276-8_8](https://link.springer.com/chapter/10.1007/978-3-031-28276-8_8)

> Grassia, M., Mangioni, G.: CoreGDM: Geometric Deep Learning Network Decycling and Dismantling. In: Teixeira, A.S., Botta, F., Mendes, J.F., Menezes, R., Mangioni, G. (eds.) Complex Networks XIV, pp. 86–94. Springer, Cham (2023). https://doi.org/10.1007/978-3-031-28276-8_8


```bibtex
@InProceedings{10.1007/978-3-031-28276-8_8,
  author="Grassia, Marco
  and Mangioni, Giuseppe",
  editor="Teixeira, Andreia Sofia
  and Botta, Federico
  and Mendes, Jos{\'e} Fernando
  and Menezes, Ronaldo
  and Mangioni, Giuseppe",
  title="CoreGDM: Geometric Deep Learning Network Decycling and Dismantling",
  booktitle="Complex Networks XIV",
  year="2023",
  publisher="Springer Nature Switzerland",
  address="Cham",
  pages="86--94",
  abstract="Network dismantling deals with the removal of nodes or edges to disrupt the largest connected component of a network. In this work we introduce CoreGDM, a trainable algorithm for network dismantling via node-removal. The approach is based on Geometric Deep Learning and that merges the Graph Dismantling Machine (GDM) [19] framework with the CoreHD [40] algorithm, by attacking the 2-core of the network using a learnable score function in place of the degree-based one. Extensive experiments on fifteen real-world networks show that CoreGDM outperforms the original GDM formulation and the other state-of-the-art algorithms, while also being more computationally efficient.",
  isbn="978-3-031-28276-8"
}
``` 

### CoreHD
DOI [10.1038/srep37954](https://doi.org/10.1038/srep37954)
> Zdeborová, L., Zhang, P., Zhou, H.-J.: Fast and simple decycling and dismantling of networks. Scientific Reports 6(1) (2016). https://doi.org/10.1038/srep37954

```bibtex
@Article{Zdeborová2016,
    author={Zdeborov{\'a}, Lenka
    and Zhang, Pan
    and Zhou, Hai-Jun},
    title={Fast and simple decycling and dismantling of networks},
    journal={Scientific Reports},
    year={2016},
    month={Nov},
    day={29},
    volume={6},
    number={1},
    pages={37954},
    abstract={Decycling and dismantling of complex networks are underlying many important applications in network science. Recently these two closely related problems were tackled by several heuristic algorithms, simple and considerably sub-optimal, on the one hand, and involved and accurate message-passing ones that evaluate single-node marginal probabilities, on the other hand. In this paper we propose a simple and extremely fast algorithm, CoreHD, which recursively removes nodes of the highest degree from the 2-core of the network. CoreHD performs much better than all existing simple algorithms. When applied on real-world networks, it achieves equally good solutions as those obtained by the state-of-art iterative message-passing algorithms at greatly reduced computational cost, suggesting that CoreHD should be the algorithm of choice for many practical purposes.},
    issn={2045-2322},
    doi={10.1038/srep37954},
    url={https://doi.org/10.1038/srep37954}
}
``` 


### Ensemble GND (EGND)
DOI [10.1007/978-3-030-36687-2_65](https://doi.org/10.1007/978-3-030-36687-2_65)
> Ren, X.-L., Antulov-Fantulin, N.: Ensemble approach for generalized network dismantling. In: Cherifi, H., Gaito, S., Mendes, J.F., Moro, E., Rocha, L.M. (eds.) Complex Networks and Their Applications VIII, pp. 783–793. Springer, Cham (2020)

```bibtex
@InProceedings{10.1007/978-3-030-36687-2_65,
    author="Ren, Xiao-Long
    and Antulov-Fantulin, Nino",
    editor="Cherifi, Hocine
    and Gaito, Sabrina
    and Mendes, Jos{\'e} Fernendo
    and Moro, Esteban
    and Rocha, Luis Mateus",
    title="Ensemble Approach for Generalized Network Dismantling",
    booktitle="Complex Networks and Their Applications VIII",
    year="2020",
    publisher="Springer International Publishing",
    address="Cham",
    pages="783--793",
    abstract="Finding a set of nodes in a network, whose removal fragments the network below some target size at minimal cost is called network dismantling problem and it belongs to the NP-hard computational class. In this paper, we explore the (generalized) network dismantling problem by exploring the spectral approximation with the variant of the power-iteration method. In particular, we explore the network dismantling solution landscape by creating the ensemble of possible solutions from different initial conditions and a different number of iterations of the spectral approximation.",
    isbn="978-3-030-36687-2"
}
``` 


### Explosive Immunization (EI)
DOI [10.1103/PhysRevLett.117.208301](https://doi.org/10.1103/PhysRevLett.117.208301)
> Clusella, P., Grassberger, P., Pérez-Reche, F.J., Politi, A.: Immunization and targeted destruction of networks using explosive percolation. Physical Review Letters 117, 208301 (2016). https://doi.org/10.1103/PhysRevLett.117.208301

```bibtex
@article{PhysRevLett.117.208301,
  title = {Immunization and Targeted Destruction of Networks using Explosive Percolation},
  author = {Clusella, Pau and Grassberger, Peter and P\'erez-Reche, Francisco J. and Politi, Antonio},
  journal = {Phys. Rev. Lett.},
  volume = {117},
  issue = {20},
  pages = {208301},
  numpages = {5},
  year = {2016},
  month = {Nov},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevLett.117.208301},
  url = {https://link.aps.org/doi/10.1103/PhysRevLett.117.208301}
}
```
### FINDER
DOI [10.1038/s42256-020-0177-2](https://doi.org/10.1038/s42256-020-0177-2)
> Fan, C., Zeng, L., Sun, Y., Liu, Y.-Y.: Finding key players in complex networks through deep reinforcement learning. Nature Machine Intelligence 2(6), 317–324 (2020). https://doi.org/10.1038/s42256-020-0177-2

```bibtex
@Article{Fan2020,
  author={Fan, Changjun
  and Zeng, Li
  and Sun, Yizhou
  and Liu, Yang-Yu},
  title={Finding key players in complex networks through deep reinforcement learning},
  journal={Nature Machine Intelligence},
  year={2020},
  month={Jun},
  day={01},
  volume={2},
  number={6},
  pages={317-324},
  abstract={Finding an optimal set of nodes, called key players, whose activation (or removal) would maximally enhance (or degrade) a certain network functionality, is a fundamental class of problems in network science. Potential applications include network immunization, epidemic control, drug design and viral marketing. Due to their general NP-hard nature, these problems typically cannot be solved by exact algorithms with polynomial time complexity. Many approximate and heuristic strategies have been proposed to deal with specific application scenarios. Yet, we still lack a unified framework to efficiently solve this class of problems. Here, we introduce a deep reinforcement learning framework FINDER, which can be trained purely on small synthetic networks generated by toy models and then applied to a wide spectrum of application scenarios. Extensive experiments under various problem settings demonstrate that FINDER significantly outperforms existing methods in terms of solution quality. Moreover, it is several orders of magnitude faster than existing methods for large networks. The presented framework opens up a new direction of using deep learning techniques to understand the organizing principle of complex networks, which enables us to design more robust networks against both attacks and failures.},
  issn={2522-5839},
  doi={10.1038/s42256-020-0177-2},
  url={https://doi.org/10.1038/s42256-020-0177-2}
}
```

### Generalized Network Dismantling (GND)
DOI [10.1073/pnas.1806108116](https://www.pnas.org/doi/10.1073/pnas.1806108116)

> Ren, X.-L., Gleinig, N., Helbing, D., Antulov-Fantulin, N.: Generalized network dismantling. Proceedings of the National Academy of Sciences 116(14), 6554–6559 (2019). https://doi.org/10.1073/pnas.1806108116

```bibtex
@article{doi:10.1073/pnas.1806108116,
    author = {Xiao-Long Ren  and Niels Gleinig  and Dirk Helbing  and Nino Antulov-Fantulin },
    title = {Generalized network dismantling},
    journal = {Proceedings of the National Academy of Sciences},
    volume = {116},
    number = {14},
    pages = {6554-6559},
    year = {2019},
    doi = {10.1073/pnas.1806108116},
    URL = {https://www.pnas.org/doi/abs/10.1073/pnas.1806108116},
    eprint = {https://www.pnas.org/doi/pdf/10.1073/pnas.1806108116},
    abstract = {Finding an optimal subset of nodes in a network that is able to efficiently disrupt the functioning of a corrupt or criminal organization or contain an epidemic or the spread of misinformation is a highly relevant problem of network science. In this paper, we address the generalized network-dismantling problem, which aims at finding a set of nodes whose removal from the network results in the fragmentation of the network into subcritical network components at minimal overall cost. Compared with previous formulations, we allow the costs of node removals to take arbitrary nonnegative real values, which may depend on topological properties such as node centrality or on nontopological features such as the price or protection level of a node. Interestingly, we show that nonunit costs imply a significantly different dismantling strategy. To solve this optimization problem, we propose a method which is based on the spectral properties of a node-weighted Laplacian operator and combine it with a fine-tuning mechanism related to the weighted vertex cover problem. The proposed method is applicable to large-scale networks with millions of nodes. It outperforms current state-of-the-art methods and opens more directions for understanding the vulnerability and robustness of complex systems.}
}
```

### Graph Dismantling Machine (GDM)
DOI [10.1038/s41467-021-25485-8](https://doi.org/10.1038/s41467-021-25485-8)
> Grassia, M., De Domenico, M., Mangioni, G.: Machine learning dismantling and early-warning signals of disintegration in complex systems. Nature Communications 12(1), 5190 (2021). https://doi.org/10.1038/s41467-021-25485-8

```bibtex
@Article{Grassia2021,
    author = {Grassia, Marco and De Domenico, Manlio and Mangioni, Giuseppe},
    title = {Machine learning dismantling and early-warning signals of disintegration in complex systems},
    journal = {Nature Communications},
    year = {2021},
    month = {Aug},
    day = {31},
    volume = {12},
    number = {1},
    pages = {5190},
    abstract = {From physics to engineering, biology and social science, natural and artificial systems are characterized by interconnected topologies whose features -- e.g., heterogeneous connectivity, mesoscale organization, hierarchy -- affect their robustness to external perturbations, such as targeted attacks to their units. Identifying the minimal set of units to attack to disintegrate a complex network, i.e. network dismantling, is a computationally challenging (NP-hard) problem which is usually attacked with heuristics. Here, we show that a machine trained to dismantle relatively small systems is able to identify higher-order topological patterns, allowing to disintegrate large-scale social, infrastructural and technological networks more efficiently than human-based heuristics. Remarkably, the machine assesses the probability that next attacks will disintegrate the system, providing a quantitative method to quantify systemic risk and detect early-warning signals of system's collapse. This demonstrates that machine-assisted analysis can be effectively used for policy and decision-making to better quantify the fragility of complex systems and their response to shocks.},
    issn = {2041-1723},
    doi = {10.1038/s41467-021-25485-8},
    url = {https://doi.org/10.1038/s41467-021-25485-8}
}
```

### Min-Sum (MS) & greedy reinsertion algorithm
DOI [10.1073/pnas.1605083113](https://doi.org/10.1073/pnas.1605083113)

> Braunstein, A., Dall’Asta, L., Semerjian, G., Zdeborová, L.: Network dismantling. Proceedings of the National Academy of Sciences 113(44), 12368–12373 (2016). https://doi.org/10.1073/pnas.1605083113

```bibtex
@article{doi:10.1073/pnas.1605083113,
    author = {Alfredo Braunstein  and Luca Dall’Asta  and Guilhem Semerjian  and Lenka Zdeborová },
    title = {Network dismantling},
    journal = {Proceedings of the National Academy of Sciences},
    volume = {113},
    number = {44},
    pages = {12368-12373},
    year = {2016},
    doi = {10.1073/pnas.1605083113},
    URL = {https://www.pnas.org/doi/abs/10.1073/pnas.1605083113},
    eprint = {https://www.pnas.org/doi/pdf/10.1073/pnas.1605083113},
    abstract = {We study the network dismantling problem, which consists of determining a minimal set of vertices in which removal leaves the network broken into connected components of subextensive size. For a large class of random graphs, this problem is tightly connected to the decycling problem (the removal of vertices, leaving the graph acyclic). Exploiting this connection and recent works on epidemic spreading, we present precise predictions for the minimal size of a dismantling set in a large random graph with a prescribed (light-tailed) degree distribution. Building on the statistical mechanics perspective, we propose a three-stage Min-Sum algorithm for efficiently dismantling networks, including heavy-tailed ones for which the dismantling and decycling problems are not equivalent. We also provide additional insights into the dismantling problem, concluding that it is an intrinsically collective problem and that optimal dismantling sets cannot be viewed as a collection of individually well-performing nodes.}
}
```

### Network Entanglement (NE)
Multi-scale network entanglement for network dismantling.

DOI [10.1038/s42005-021-00633-0](https://doi.org/10.1038/s42005-021-00633-0)
```bibtex

@article{Ghavasieh2021unraveling,
	abstract = {Complex systems are large collections of entities that organize themselves into non-trivial structures, represented as networks. One of their key emergent properties is robustness against random failures or targeted attacks ---i.e., the networks maintain their integrity under removal of nodes or links. Here, we introduce network entanglement to study network robustness through a multiscale lens, encoded by the time required for information diffusion through the system. Our measure's foundation lies upon a recently developed statistical field theory for information dynamics within interconnected systems. We show that at the smallest temporal scales, the node-network entanglement reduces to degree, whereas at extremely large scales, it measures the direct role played by each node in keeping the network connected. At the meso-scale, entanglement plays a more important role, measuring the importance of nodes for the transport properties of the system. We use entanglement as a centrality measure capturing the role played by nodes in keeping the overall diversity of the information flow. As an application, we study the disintegration of empirical social, biological and transportation systems, showing that the nodes central for information dynamics are also responsible for keeping the network integrated.},
	author = {Ghavasieh, Arsham and Stella, Massimo and Biamonte, Jacob and De Domenico, Manlio},
	date = {2021/06/10},
	doi = {10.1038/s42005-021-00633-0},
	id = {Ghavasieh2021},
	isbn = {2399-3650},
	journal = {Communications Physics},
	number = {1},
	pages = {129},
	title = {Unraveling the effects of multiscale network entanglement on empirical systems},
	url = {https://doi.org/10.1038/s42005-021-00633-0},
	volume = {4},
	year = {2021},
	bdsk-url-1 = {https://doi.org/10.1038/s42005-021-00633-0}}
```

### Vertex Entanglement (VE)

DOI [10.1038/s42005-023-01483-8](https://doi.org/10.1038/s42005-023-01483-8)

```bibtex

@article{Huang2024empirical,
	abstract = {Empirical networks exhibit significant heterogeneity in node connections, resulting in a few vertices playing critical roles in various scenarios, including decision-making, viral marketing, and population immunization. Thus, identifying key vertices is a fundamental research problem in Network Science. In this paper, we introduce vertex entanglement (VE), an entanglement-based metric capable of quantifying the perturbations caused by individual vertices on spectral entropy, residing at the intersection of quantum information and network science. Our analytical analysis reveals that VE is closely related to network robustness and information transmission ability. As an application, VE offers an approach to the challenging problem of optimal network dismantling, and empirical experiments demonstrate its superiority over state-of-the-art algorithms. Furthermore, VE also contributes to the diagnosis of autism spectrum disorder (ASD), with significant distinctions in hub disruption indices based on VE between ASD and typical controls, promising a diagnostic role for VE in ASD assessment.},
	author = {Huang, Yiming and Wang, Hao and Ren, Xiao-Long and L{\"u}, Linyuan},
	date = {2024/01/08},
	doi = {10.1038/s42005-023-01483-8},
	id = {Huang2024},
	isbn = {2399-3650},
	journal = {Communications Physics},
	number = {1},
	pages = {19},
	title = {Identifying key players in complex networks via network entanglement},
	url = {https://doi.org/10.1038/s42005-023-01483-8},
	volume = {7},
	year = {2024},
	bdsk-url-1 = {https://doi.org/10.1038/s42005-023-01483-8}}
```


### Node heuristics
TODO

## Libraries

### graph-tool
DOI [10.6084/m9.figshare.1164194](https://doi.org/10.6084/m9.figshare.1164194)

> P. Peixoto, Tiago (2014). The graph-tool python library. figshare. Dataset. https://doi.org/10.6084/m9.figshare.1164194

```bibtex
@article{peixoto_graph-tool_2014,
    title = {The graph-tool python library},
    url = {http://figshare.com/articles/graph_tool/1164194},
    doi = {10.6084/m9.figshare.1164194},
    urldate = {2014-09-10},
    journal = {figshare},
    author = {Peixoto, Tiago P.},
    year = {2014},
    keywords = {all, complex networks, graph, network, other}
}
```

