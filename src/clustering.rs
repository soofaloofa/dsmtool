// TODO: Standardize on anyhow::Result and thiserror
// TODO: Standardize on ref/mut/clone ... when to use which?
use anyhow::Result;
use rand::Rng;
use std::fmt;
use std::vec;

/// `DSM` is a struct holding a matrix that represents a Design Structure Matrix
/// (DSM). The DSM is a square matrix that represents the relationships or
/// dependencies between elements in they system.
///
/// The matrix layout is as follows:
///
/// * The system elements names are placed down the side of the matrix as row
///   headings and across the top as column headings in the same order.  
/// * If there exists an edge from node i to node j, then the value of element ij
///   (row i, column j) is marked with the strength of the connection.
/// * Otherwise, the value of the element is zero (or left empty).
///
/// In the binary matrix representation of a system, the diagonal elements of
/// the matrix do not have any interpretation in describing the system, so they
/// are usually either left empty or blacked out, although many find it
/// intuitive to think of these diagonal cells as representative of the nodes
/// themselves.
#[derive(Debug, Clone, PartialEq)]
pub struct Dsm {
    pub labels: Vec<String>,
    pub matrix: Vec<Vec<f64>>,
}

impl Dsm {
    // Initial cluster matrix sets each element to its own cluster
    pub fn new(labels: Vec<String>, matrix: Vec<Vec<f64>>) -> Self {
        Dsm { labels, matrix }
    }

    // Number of elements in the DSM
    pub fn len(&self) -> usize {
        self.labels.len()
    }
}

impl fmt::Display for Dsm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f)?;
        // labels
        for label in self.labels.clone() {
            write!(f, "{:>8.2} ", label)?;
        }
        writeln!(f)?;

        // data
        for row in self.matrix.clone() {
            for val in row {
                write!(f, "{:>8.2} ", val)?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}

/// `Clustering` is a struct holding a matrix that represents the assignment of
/// elements to clusters. Each row in a clustering corresponds to a cluster, and
/// each column corresponds to an element belonging to that cluster.  The value
/// at a given position in the matrix indicates whether the element belongs to
/// the cluster.
///
/// # Examples
///
/// Consider the following clustering:
///
/// ```
/// let cluster_matrix = vec![
///     vec![1.0, 0.0, 0.0, 1.0], // Cluster 1
///     vec![0.0, 1.0, 1.0, 0.0], // Cluster 2
/// ];
/// ```
///
/// This matrix can be interpreted as follows:
///
/// * Cluster 1 (first row): Contains elements 0 and 3.  
/// * Cluster 2 (second /// row): Contains elements 1 and 2.
#[derive(Clone, Debug, PartialEq)]
struct Clustering {
    matrix: Vec<Vec<f64>>,
}

impl Clustering {
    // Initial cluster matrix sets each element to its own cluster
    pub fn new(num_elements: usize) -> Self {
        // Set the cluster matrix to a diagonal matrix along the axis
        let matrix: Vec<Vec<f64>> = (0..num_elements)
            .map(|i| {
                let mut row = vec![0.0; num_elements];
                row[i] = 1.0;
                row
            })
            .collect();

        Clustering { matrix }
    }

    // Number of clusters in the clustering matrix
    pub fn cluster_count(&self) -> usize {
        self.matrix.len()
    }

    // Number of clusters in the clustering matrix
    pub fn element_count(&self) -> usize {
        self.matrix[0].len()
    }

    // Returns an array where the nth element holds the size of cluster n.
    pub fn sizes(&self) -> Vec<f64> {
        self.matrix.iter().map(|row| row.iter().sum()).collect()
    }
}

impl fmt::Display for Clustering {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f)?;

        // data
        for row in self.matrix.clone() {
            for val in row {
                write!(f, "{:>8.2} ", val)?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}

pub fn cluster(
    dsm: &Dsm,
    pow_cc: f64, // penalty assigned to cluster size during costing
    initial_temperature: f64,
    cooling_rate: f64,
) -> Result<(Dsm, Vec<f64>)> {
    let rng = &mut rand::thread_rng();

    let mut clustering = Clustering::new(dsm.len());

    // Calculate the initial starting coordination cost of the clustering
    let mut curr_coord_cost = coord_cost(dsm, &clustering, pow_cc);

    // Initialize the best solution to the current solution
    let mut best_coord_cost = curr_coord_cost;
    let mut best_clustering = clustering.clone();

    let mut cost_history: Vec<f64> = vec![];
    let mut temperature = initial_temperature;
    while temperature > 1e-3 {
        // Pick a random element from the DSM to put in a new cluster
        let element = rng.gen_range(0..dsm.len());

        // Pick a random new cluster for the chosen element
        let cluster = rng.gen_range(0..clustering.cluster_count());

        let new_clustering = move_element_to_cluster(&clustering, element, cluster);

        // Calculate the coordination cost of the new cluster assignment
        let new_coord_cost = coord_cost(dsm, &new_clustering, pow_cc);

        // Accept the new cluster assignment if it has a
        // lower coord cost.  If the cost is higher, accept it with a
        // probability determined by the annealing temperature
        // Initially, we have a high likelihood of accepting worse solutions
        let accept = if new_coord_cost <= curr_coord_cost {
            true
        } else {
            let acceptance_probability = ((curr_coord_cost - new_coord_cost) / temperature).exp();
            rng.gen_bool(acceptance_probability)
        };

        if accept {
            // Update the cluster with the accepted values
            curr_coord_cost = new_coord_cost;
            clustering = new_clustering;

            // Record the solution cost
            cost_history.push(curr_coord_cost);

            // If we have a new best, update it
            if curr_coord_cost < best_coord_cost {
                best_coord_cost = curr_coord_cost;
                best_clustering = clustering.clone();
            }
        }

        temperature *= cooling_rate;
    }

    // Delete empty or duplicate clusters
    let new_clustering = prune_clusters(best_clustering);
    let ordered_dsm = reorder_dsm(dsm, &new_clustering);

    // TODO: Keep labels with the DSM and re order them together
    Ok((ordered_dsm, cost_history))
}

// Update the cluster matrix by assigning the element at the selected index
// to the selected cluster
fn move_element_to_cluster(
    clustering: &Clustering,
    element_idx: usize,
    cluster_idx: usize,
) -> Clustering {
    let mut new_clustering = clustering.clone();
    for i in 0..clustering.cluster_count() {
        if i == cluster_idx {
            new_clustering.matrix[i][element_idx] = 1.0;
        } else {
            new_clustering.matrix[i][element_idx] = 0.0;
        }
    }

    new_clustering
}

/// Delete duplicate clusters or any clusters that are within clusters
fn prune_clusters(mut clustering: Clustering) -> Clustering {
    let n_clusters = clustering.cluster_count();
    let n_elements = clustering.element_count();

    let cluster_size = clustering.sizes();
    // If the clusters are equal or cluster j is completely contained in cluster i, delete cluster j
    for i in 0..n_clusters {
        for j in (i + 1)..n_clusters {
            if cluster_size[i] >= cluster_size[j]
                && cluster_size[j] > 0.0
                && clustering.matrix[i]
                    .iter()
                    .zip(&clustering.matrix[j])
                    .all(|(&a, &b)| (a != 0.0 && b != 0.0) == (b != 0.0))
            {
                clustering.matrix[j] = vec![0.0; n_elements];
            }
        }
    }

    // If cluster i is completely contained in cluster j, delete cluster i
    for i in 0..n_clusters {
        for j in (i + 1)..n_clusters {
            if cluster_size[i] < cluster_size[j]
                && cluster_size[i] > 0.0
                && clustering.matrix[i]
                    .iter()
                    .zip(&clustering.matrix[j])
                    .all(|(&a, &b)| (a != 0.0 && b != 0.0) == (a != 0.0))
            {
                clustering.matrix[i] = vec![0.0; n_elements];
            }
        }
    }

    // Delete empty clusters
    let new_cluster_matrix = clustering
        .matrix
        .into_iter()
        .filter(|row| row.iter().any(|&x| x != 0.0))
        .collect();

    Clustering {
        matrix: new_cluster_matrix,
    }
}

/// Function to calculate the coordination cost of the clustered matrix
///
/// This checks all DSM interactions.  If a DSM interaction is contained in
/// in one or more clusters, we add the cost of all intra-cluster interactions.
/// If the interaction is not contained within any clusters, then a higher cost is
/// assigned to the out-of-cluster interaction.
fn coord_cost(dsm: &Dsm, clustering: &Clustering, pow_cc: f64) -> f64 {
    // TODO: Calculate pow_cc as a percentage from the total DSM size
    // i.e. all in same cluster -> 100% overhead, all in different clusters -> 0% overhead, then range in between

    // Looking at all DSM interactions i and j,
    // Are DSM(i) and DSM(j) both contained in the same cluster?
    //  if yes: coordination cost is (DSM(i) + DSM(j))*cluster_size(cluster n)^pow_cc
    //  if no: coordination cost is (DSM(i) + DSM(j))*DSM_size^pow_cc
    // Total coordination cost is equal to the sum of all coordination costs
    let mut coordination_cost = vec![0.0; dsm.len()];

    let cluster_size = clustering.sizes();

    // Calculate the cost of the solution
    #[allow(clippy::needless_range_loop)]
    for col in 0..dsm.len() {
        // Plus col+1 to avoid self-interactions (skips diagonals)
        for row in (col + 1)..dsm.len() {
            if dsm.matrix[col][row] > 0.0 || dsm.matrix[row][col] > 0.0 {
                // There is an interaction between the two elements
                let mut cost_total = 0.0;

                // Check if any of the cluster assignments contain both elements
                for cluster_num in 0..clustering.cluster_count() {
                    if clustering.matrix[cluster_num][col] + clustering.matrix[cluster_num][row]
                        == 2.0
                    // both elements are in the same cluster
                    {
                        cost_total += (dsm.matrix[col][row] + dsm.matrix[row][col])
                            * cluster_size[cluster_num].powf(pow_cc);
                    }
                }

                let cost_c = if cost_total > 0.0 {
                    cost_total
                } else {
                    (dsm.matrix[col][row] + dsm.matrix[row][col]) * (dsm.len() as f64).powf(pow_cc)
                };

                coordination_cost[col] += cost_c;
            }
        }
    }

    let total_coord_cost = coordination_cost.iter().sum();
    total_coord_cost
}

/// Function to reorder the DSM Matrix according to the Cluster matrix.
///
/// Places all elements in the same cluster next to each other.  If an
/// element is a member of more than one cluster, duplicate that element
/// in the DSM for each time it appears in a cluster
///
/// The new DSM will have all elements in a cluster next to each other
fn reorder_dsm(dsm: &Dsm, clustering: &Clustering) -> Dsm {
    let mut ordered_dsm = vec![vec![0.0; dsm.len()]; dsm.len()];

    // Find the new order of elements based on the clustering
    let mut new_element_order: Vec<usize> = vec![];
    for cluster in clustering.matrix.clone() {
        for (i, &elmt) in cluster.iter().enumerate() {
            if elmt > 0.0 {
                new_element_order.push(i);
            }
        }
    }

    for i in 0..ordered_dsm.len() {
        for j in 0..ordered_dsm[i].len() {
            ordered_dsm[i][j] = dsm.matrix[new_element_order[i]][new_element_order[j]];
        }
    }

    Dsm {
        labels: new_element_order
            .iter()
            .map(|&i| dsm.labels[i].clone())
            .collect(),
        matrix: ordered_dsm,
    }
}

#[cfg(test)]

mod tests {
    use super::*;

    #[test]
    fn test_reorder_dsm_by_cluster() {
        // Sample DSM matrix
        let dsm_matrix = vec![
            vec![0.0, 1.0, 0.0, 0.0],
            vec![1.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 1.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];
        let dsm = Dsm {
            labels: vec![
                "A".to_string(),
                "B".to_string(),
                "C".to_string(),
                "D".to_string(),
            ],
            matrix: dsm_matrix,
        };

        // Sample cluster matrix
        let cluster_matrix = vec![vec![1.0, 0.0, 0.0, 1.0], vec![0.0, 1.0, 1.0, 0.0]];
        let clustering = Clustering {
            matrix: cluster_matrix,
        };

        // Expected reordered DSM matrix
        let expected_dsm_matrix = vec![
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0, 1.0, 1.0, 0.0],
        ];
        let expected_dsm = Dsm {
            labels: vec![
                "A".to_string(),
                "D".to_string(),
                "B".to_string(),
                "C".to_string(),
            ],
            matrix: expected_dsm_matrix,
        };

        // Call the function
        let new_dsm = reorder_dsm(&dsm, &clustering);

        // Check if the output matches the expected output
        assert_eq!(new_dsm, expected_dsm);
    }

    #[test]
    fn test_coord_cost() {
        // Sample DSM matrix
        let dsm_matrix = vec![
            vec![0.0, 1.0, 0.0, 0.0],
            vec![1.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 1.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];
        let dsm = Dsm {
            labels: vec![
                "A".to_string(),
                "B".to_string(),
                "C".to_string(),
                "D".to_string(),
            ],
            matrix: dsm_matrix,
        };

        // Sample cluster matrix
        let cluster_matrix = vec![vec![1.0, 0.0, 0.0, 1.0], vec![0.0, 1.0, 1.0, 0.0]];
        let clustering = Clustering {
            matrix: cluster_matrix,
        };

        // Weighting function
        let pow_cc = 1.0;

        // Expected coordination cost
        let expected_coord_cost = 20.0;

        // Call the function
        let total_coord_cost = coord_cost(&dsm, &clustering, pow_cc);

        // Check if the output matches the expected output
        assert_eq!(total_coord_cost, expected_coord_cost);
    }

    #[test]
    fn test_coord_cost_with_different_weights() {
        // Sample DSM matrix
        let dsm_matrix = vec![
            vec![0.0, 2.0, 0.0, 0.0],
            vec![2.0, 0.0, 2.0, 0.0],
            vec![0.0, 2.0, 0.0, 2.0],
            vec![0.0, 0.0, 2.0, 0.0],
        ];
        let dsm = Dsm {
            labels: vec![
                "A".to_string(),
                "B".to_string(),
                "C".to_string(),
                "D".to_string(),
            ],
            matrix: dsm_matrix,
        };

        // Sample cluster matrix
        let cluster_matrix = vec![vec![1.0, 0.0, 0.0, 1.0], vec![0.0, 1.0, 1.0, 0.0]];
        let clustering = Clustering {
            matrix: cluster_matrix,
        };

        // Weighting function
        let pow_cc = 2.0;

        // Expected coordination cost
        let expected_coord_cost = 144.0;

        // Call the function
        let total_coord_cost = coord_cost(&dsm, &clustering, pow_cc);

        // Check if the output matches the expected output
        assert_eq!(total_coord_cost, expected_coord_cost);
    }

    #[test]
    fn test_delete_clusters_basic() {
        let cluster_matrix = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let clustering = Clustering {
            matrix: cluster_matrix,
        };

        let new_clustering = prune_clusters(clustering);

        let expected_cluster_matrix = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let expected_clustering = Clustering {
            matrix: expected_cluster_matrix,
        };

        assert_eq!(new_clustering, expected_clustering);
    }

    #[test]
    fn test_delete_clusters_with_duplicates() {
        let cluster_matrix = vec![
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ];
        let clustering = Clustering {
            matrix: cluster_matrix,
        };

        let new_clustering = prune_clusters(clustering);

        let expected_cluster_matrix = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let expected_clustering = Clustering {
            matrix: expected_cluster_matrix,
        };

        assert_eq!(new_clustering, expected_clustering);
    }

    #[test]
    fn test_delete_clusters_with_contained_clusters() {
        let cluster_matrix = vec![
            vec![1.0, 1.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ];
        let clustering = Clustering {
            matrix: cluster_matrix,
        };

        let new_clustering = prune_clusters(clustering);

        let expected_cluster_matrix = vec![vec![1.0, 1.0, 0.0]];
        let expected_clustering = Clustering {
            matrix: expected_cluster_matrix,
        };

        assert_eq!(new_clustering, expected_clustering);
    }

    #[test]
    fn test_delete_clusters_with_empty_clusters() {
        let cluster_matrix = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ];
        let clustering = Clustering {
            matrix: cluster_matrix,
        };

        let new_clustering = prune_clusters(clustering);

        let expected_cluster_matrix = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let expected_clustering = Clustering {
            matrix: expected_cluster_matrix,
        };

        assert_eq!(new_clustering, expected_clustering);
    }
}
