//! This is a pretty literal translation from Matlab to Rust.  It is not idiomatic Rust.
//! Make this more idiomatic.
//!
//! # clustering
//!
//! `clustering` is a collection of utilities to execute clustering algorithms
//! on a DSM
//!
//!	This function runs a clustering algorithm then calculates the cost of
//!	the proposed solution.  The objective is to find the solution that results
//!	in the lowest cost solution.
//!
//!	There is a higher cost for interactions that occur outside of clusters
//!	and lower costs for interactions within clusters.  There are also
//!	penalties assigned to the size of clusters to avoid a solution where
//!	all elements are members of a single cluster.
//!
//!	There results are highly dependant on the parameters passed to the
//!	algorithm.
use anyhow::Result;
use std::vec;

use rand::Rng;

/// DsmMatrix
///
type DsmMatrix = Vec<Vec<f64>>;

/// ClusterMatrix is a matrix that represents the assignment of elements
/// to clusters. Each row in ClusterMatrix corresponds to a cluster, and
/// each column corresponds to an element. The value at a given position in the
/// matrix indicates whether the element belongs to the cluster.
///
/// # Examples
///
/// Consider the following ClusterMatrix:
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
type ClusterMatrix = Vec<Vec<f64>>;

/// `ClusterSize`` is an array where the nth element holds the size of cluster n.
/// It can be computed from ClusterMatrix by summing the elements in each row.
/// TODO: Make this a function or property of ClusterMatrix
type ClusterSize = Vec<f64>;

#[derive(Debug, Default, Clone)]
pub struct ClusteringConfig {
    pub pow_cc: f64, // penalty assigned to cluster size during costing
}

pub fn cluster(
    dsm_matrix: &Vec<Vec<f64>>,
    config: ClusteringConfig,
    initial_temperature: f64,
    cooling_rate: f64,
) -> Result<Vec<f64>> {
    println!("dsm_matrix: ");
    pprint_matrix(dsm_matrix);
    let rng = &mut rand::thread_rng();

    let mut cluster_matrix = init_cluster_matrix(dsm_matrix);
    // All clusters start at size 1
    // TODO: Make cluster_size part of the cluster_matrix data type or struct.
    let mut cluster_size: Vec<f64> = vec![1.0; dsm_matrix.len()];

    // Calculate the initial starting coordination cost of the clustering
    let mut curr_coord_cost = coord_cost(dsm_matrix, &cluster_matrix, &cluster_size, config.pow_cc);
    println!("starting coordination cost: {:?}", curr_coord_cost);

    // Initialize the best solution to the current solution
    let mut best_coord_cost = curr_coord_cost;
    let mut best_cluster_matrix = cluster_matrix.clone();
    let mut best_cluster_size = cluster_size.clone();

    let mut temperature = initial_temperature;

    let mut solution_costs: Vec<f64> = vec![];

    let mut num_iterations = 0;
    while temperature > 1e-3 {
        // Pick a random element from the DSM to put in a new cluster
        let elmt = rng.gen_range(0..dsm_matrix.len());

        // Pick a random cluster to assign the element to
        let cluster = rng.gen_range(0..cluster_matrix.len());

        let (new_cluster_matrix, new_cluster_size) =
            assign_element_to_cluster(&cluster_matrix, cluster, elmt);

        // Calculate the new coordination cost with the updated element
        let new_coord_cost = coord_cost(
            dsm_matrix,
            &new_cluster_matrix,
            &new_cluster_size,
            config.pow_cc,
        );

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
            cluster_matrix = new_cluster_matrix;
            cluster_size = new_cluster_size;

            // Record the solution cost
            solution_costs.push(curr_coord_cost);

            // If we have a new best, update it
            if curr_coord_cost < best_coord_cost {
                best_coord_cost = curr_coord_cost;
                best_cluster_matrix = cluster_matrix.clone();
                best_cluster_size = cluster_size.clone();
            }
        }

        temperature *= cooling_rate;

        num_iterations += 1;
    }

    // Delete empty or duplicate clusters
    let (new_cluster_matrix, new_cluster_size) =
        delete_clusters(best_cluster_matrix, best_cluster_size);

    println!("new_cluster_matrix: ");
    pprint_matrix(&new_cluster_matrix);
    println!("new_cluster_size: {:?}", new_cluster_size);

    let ordered_matrix = reorder_dsm_by_cluster(dsm_matrix, &new_cluster_matrix);

    println!("ordered_matrix: ");
    pprint_matrix(&ordered_matrix);

    println!("ending coordination cost: {:?}", best_coord_cost);

    // TODO: Keep labels with the DSM and re order them together
    println!("num_iterations: {:?}", num_iterations);
    Ok(solution_costs)
}

// Helper function for debugging purposes
fn pprint_matrix(matrix: &Vec<Vec<f64>>) {
    for row in matrix {
        for val in row {
            print!("{:>8.2} ", val);
        }
        println!();
    }
}

// Update the cluster matrix by assigning the element at the selected index
// to the selected cluster
// TODO: Write tests
fn assign_element_to_cluster(
    cluster_matrix: &Vec<Vec<f64>>,
    cluster_idx: usize,
    elmt_idx: usize,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut new_cluster_matrix = cluster_matrix.clone();
    for i in 0..cluster_matrix.len() {
        if i == cluster_idx {
            new_cluster_matrix[i][elmt_idx] = 1.0;
        } else {
            new_cluster_matrix[i][elmt_idx] = 0.0;
        }
    }
    let new_cluster_size = new_cluster_matrix
        .iter()
        .map(|row| row.iter().sum())
        .collect();

    (new_cluster_matrix, new_cluster_size)
}

// Set the initial cluster matrix to have a single cluster for each element along the diagonal
fn init_cluster_matrix(dsm_matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let cluster_matrix: Vec<Vec<f64>> = (0..dsm_matrix.len())
        .map(|i| {
            let mut row = vec![0.0; dsm_matrix.len()];
            row[i] = 1.0;
            row
        })
        .collect();
    cluster_matrix
}

/// Function to calculate the bids from clusters for the selected element. Each
/// cluster makes a bid for the selected element based on the bidding
/// parameters.
fn bid(
    elmt: usize,
    dsm_matrix: &Vec<Vec<f64>>,
    cluster_matrix: &Vec<Vec<f64>>,
    cluster_size: Vec<f64>,
    max_cluster_size: usize,
    pow_dep: f64,
    pow_bid: f64,
) -> Vec<f64> {
    // Initilize all bids to 0
    let mut cluster_bid = vec![0.0; cluster_matrix.len()];

    // For each cluster, if any element in the cluster has an interaction with
    // the selected element then add the number of interactions with the
    // selected element.  Then use the number of interactions to calculate the
    // bid.
    for i in 0..cluster_matrix.len() {
        if (cluster_size[i] < max_cluster_size as f64) && cluster_size[i] > 0.0 {
            let sum_dsm_cluster: f64 = dsm_matrix[elmt]
                .iter()
                .zip(&cluster_matrix[i])
                .map(|(dsm, cluster)| dsm * cluster)
                .sum();
            cluster_bid[i] = (sum_dsm_cluster.powf(pow_dep)) / (cluster_size[i].powf(pow_bid));
        } else {
            cluster_bid[i] = 0.0;
        }
    }

    cluster_bid
}

/// Delete duplicate clusters or any clusters that are within clusters
fn delete_clusters(
    mut cluster_matrix: Vec<Vec<f64>>,
    mut cluster_size: Vec<f64>,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n_clusters = cluster_matrix.len();
    let n_elements = cluster_matrix[0].len();

    // If the clusters are equal or cluster j is completely contained in cluster i, delete cluster j
    for i in 0..n_clusters {
        for j in (i + 1)..n_clusters {
            if cluster_size[i] >= cluster_size[j] && cluster_size[j] > 0.0 {
                if cluster_matrix[i]
                    .iter()
                    .zip(&cluster_matrix[j])
                    .all(|(&a, &b)| (a != 0.0 && b != 0.0) == (b != 0.0))
                {
                    cluster_matrix[j] = vec![0.0; n_elements];
                    cluster_size[j] = 0.0;
                }
            }
        }
    }

    // If cluster i is completely contained in cluster j, delete cluster i
    for i in 0..n_clusters {
        for j in (i + 1)..n_clusters {
            if cluster_size[i] < cluster_size[j] && cluster_size[i] > 0.0 {
                if cluster_matrix[i]
                    .iter()
                    .zip(&cluster_matrix[j])
                    .all(|(&a, &b)| (a != 0.0 && b != 0.0) == (a != 0.0))
                {
                    cluster_matrix[i] = vec![0.0; n_elements];
                    cluster_size[i] = 0.0;
                }
            }
        }
    }

    // Delete empty clusters
    let new_cluster_matrix = cluster_matrix
        .into_iter()
        .filter(|row| row.iter().any(|&x| x != 0.0))
        .collect();

    // Delete empty sizes
    cluster_size.retain(|&x| x != 0.0);

    (new_cluster_matrix, cluster_size)
}

/// Function to calculate the coordination cost of the clustered matrix
///
/// This checks all DSM interactions.  If a DSM interaction is contained in
/// in one or more clusters, we add the cost of all intra-cluster interactions.
/// If the interaction is not contained within any clusters, then a higher cost is
/// assigned to the out-of-cluster interaction.
fn coord_cost(
    dsm_matrix: &DsmMatrix,
    cluster_matrix: &ClusterMatrix,
    cluster_size: &ClusterSize,
    pow_cc: f64,
) -> f64 {
    // Looking at all DSM interactions i and j,
    // Are DSM(i) and DSM(j) both contained in the same cluster?
    //  if yes: coordination cost is (DSM(i) + DSM(j))*cluster_size(cluster n)^pow_cc
    //  if no: coordination cost is (DSM(i) + DSM(j))*DSM_size^pow_cc
    // Total coordination cost is equal to the sum of all coordination costs
    let mut coordination_cost = vec![0.0; dsm_matrix.len()];

    // Calculate the cost of the solution
    for col in 0..dsm_matrix.len() {
        // Plus col+1 to avoid self-interactions (skips diagonals)
        for row in (col + 1)..dsm_matrix.len() {
            if dsm_matrix[col][row] > 0.0 || dsm_matrix[row][col] > 0.0 {
                // There is an interaction between the two elements
                let mut cost_total = 0.0;

                // Check if any of the cluster assignments contain both elements
                for cluster_index in 0..cluster_matrix.len() {
                    if cluster_matrix[cluster_index][col] + cluster_matrix[cluster_index][row]
                        == 2.0
                    // both elements are in the same cluster
                    {
                        cost_total += (dsm_matrix[col][row] + dsm_matrix[row][col])
                            * cluster_size[cluster_index].powf(pow_cc);
                    }
                }

                let cost_c = if cost_total > 0.0 {
                    cost_total
                } else {
                    (dsm_matrix[col][row] + dsm_matrix[row][col])
                        * (dsm_matrix.len() as f64).powf(pow_cc)
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
fn reorder_dsm_by_cluster(
    dsm_matrix: &Vec<Vec<f64>>,
    cluster_matrix: &Vec<Vec<f64>>,
) -> Vec<Vec<f64>> {
    let mut ordered_dsm = vec![vec![0.0; dsm_matrix.len()]; dsm_matrix.len()];

    // Find the new order of elements based on the clustering
    let mut new_element_order: Vec<usize> = vec![];
    for cluster in cluster_matrix {
        for (i, &elmt) in cluster.iter().enumerate() {
            if elmt > 0.0 {
                new_element_order.push(i);
            }
        }
    }

    for i in 0..ordered_dsm.len() {
        for j in 0..ordered_dsm[i].len() {
            ordered_dsm[i][j] = dsm_matrix[new_element_order[i]][new_element_order[j]];
        }
    }

    ordered_dsm
}

#[cfg(test)]

mod tests {
    use super::*;

    #[test]
    fn test_init_cluster_matrix() {
        // Sample DSM matrix
        let dsm_matrix = vec![
            vec![1.0, 1.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0, 0.0],
            vec![0.0, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
        ];

        let expected_cluster_matrix = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];

        let cluster_matrix = init_cluster_matrix(&dsm_matrix);

        // Check if the output matches the expected output
        assert_eq!(cluster_matrix, expected_cluster_matrix);
    }

    #[test]
    fn test_reorder_dsm_by_cluster() {
        // Sample DSM matrix
        let dsm_matrix = vec![
            vec![0.0, 1.0, 0.0, 0.0],
            vec![1.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 1.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];

        // Sample cluster matrix
        let cluster_matrix = vec![vec![1.0, 0.0, 0.0, 1.0], vec![0.0, 1.0, 1.0, 0.0]];

        // Expected reordered DSM matrix
        let expected_new_dsm_matrix = vec![
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0, 1.0, 1.0, 0.0],
        ];

        // Call the function
        let new_dsm_matrix = reorder_dsm_by_cluster(&dsm_matrix, &cluster_matrix);

        // Check if the output matches the expected output
        assert_eq!(new_dsm_matrix, expected_new_dsm_matrix);
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

        // Sample cluster matrix
        let cluster_matrix = vec![vec![1.0, 0.0, 0.0, 1.0], vec![0.0, 1.0, 1.0, 0.0]];

        // Sample cluster size
        let cluster_size = vec![2.0, 2.0];

        // Weighting function
        let pow_cc = 1.0;

        // Expected coordination cost
        let expected_coord_cost = 20.0;

        // Call the function
        let total_coord_cost = coord_cost(&dsm_matrix, &cluster_matrix, &cluster_size, pow_cc);

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

        // Sample cluster matrix
        let cluster_matrix = vec![vec![1.0, 0.0, 0.0, 1.0], vec![0.0, 1.0, 1.0, 0.0]];

        // Sample cluster size
        let cluster_size = vec![2.0, 2.0];

        // Weighting function
        let pow_cc = 2.0;

        // Expected coordination cost
        let expected_coord_cost = 144.0;

        // Call the function
        let total_coord_cost = coord_cost(&dsm_matrix, &cluster_matrix, &cluster_size, pow_cc);

        // Check if the output matches the expected output
        assert_eq!(total_coord_cost, expected_coord_cost);
    }

    #[test]
    fn test_bid_basic() {
        // Sample DSM matrix
        let dsm_matrix = vec![
            vec![0.0, 1.0, 0.0, 0.0],
            vec![1.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 1.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];

        // Sample cluster matrix
        let cluster_matrix = vec![vec![1.0, 0.0, 0.0, 1.0], vec![0.0, 1.0, 1.0, 0.0]];

        // Sample cluster size
        let cluster_size = vec![2.0, 2.0];

        let result = bid(1, &dsm_matrix, &cluster_matrix, cluster_size, 4, 1.0, 1.0);
        let expected = vec![0.5, 0.5];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_bid_with_empty_cluster() {
        // Sample DSM matrix
        let dsm_matrix = vec![
            vec![0.0, 1.0, 0.0, 0.0],
            vec![1.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 1.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];

        // Sample cluster matrix
        let cluster_matrix = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0, 0.0],
        ];

        // Sample cluster size
        let cluster_size = vec![1.0, 2.0, 2.0, 0.0];

        let result = bid(1, &dsm_matrix, &cluster_matrix, cluster_size, 2, 1.0, 1.0);
        let expected = vec![1.0, 0.0, 0.0, 0.0];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_bid_with_max_cluster_size() {
        // Sample DSM matrix
        let dsm_matrix = vec![
            vec![0.0, 1.0, 0.0, 0.0],
            vec![1.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 1.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];

        // Sample cluster matrix
        let cluster_matrix = vec![vec![1.0, 0.0, 0.0, 1.0], vec![0.0, 1.0, 1.0, 0.0]];

        // Sample cluster size
        let cluster_size = vec![2.0, 2.0];

        let result = bid(1, &dsm_matrix, &cluster_matrix, cluster_size, 1, 1.0, 1.0);
        let expected = vec![0.0, 0.0];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_bid_with_different_powers() {
        let dsm_matrix = vec![
            vec![0.0, 1.0, 0.0],
            vec![1.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
        ];
        let cluster_matrix = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let cluster_size = vec![1.0, 1.0, 1.0];

        let result = bid(1, &dsm_matrix, &cluster_matrix, cluster_size, 2, 2.0, 0.5);
        let expected = vec![1.0, 0.0, 1.0];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_delete_clusters_basic() {
        let cluster_matrix = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let cluster_size = vec![1.0, 1.0, 1.0];

        let (new_cluster_matrix, new_cluster_size) = delete_clusters(cluster_matrix, cluster_size);

        let expected_cluster_matrix = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let expected_cluster_size = vec![1.0, 1.0, 1.0];

        assert_eq!(new_cluster_matrix, expected_cluster_matrix);
        assert_eq!(new_cluster_size, expected_cluster_size);
    }

    #[test]
    fn test_delete_clusters_with_duplicates() {
        let cluster_matrix = vec![
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ];
        let cluster_size = vec![1.0, 1.0, 1.0];

        let (new_cluster_matrix, new_cluster_size) = delete_clusters(cluster_matrix, cluster_size);

        let expected_cluster_matrix = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let expected_cluster_size = vec![1.0, 1.0];

        assert_eq!(new_cluster_matrix, expected_cluster_matrix);
        assert_eq!(new_cluster_size, expected_cluster_size);
    }

    #[test]
    fn test_delete_clusters_with_contained_clusters() {
        let cluster_matrix = vec![
            vec![1.0, 1.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ];
        let cluster_size = vec![2.0, 1.0, 1.0];

        let (new_cluster_matrix, new_cluster_size) = delete_clusters(cluster_matrix, cluster_size);

        let expected_cluster_matrix = vec![vec![1.0, 1.0, 0.0]];
        let expected_cluster_size = vec![2.0];

        assert_eq!(new_cluster_matrix, expected_cluster_matrix);
        assert_eq!(new_cluster_size, expected_cluster_size);
    }

    #[test]
    fn test_delete_clusters_with_empty_clusters() {
        let cluster_matrix = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ];
        let cluster_size = vec![1.0, 0.0, 1.0];

        let (new_cluster_matrix, new_cluster_size) = delete_clusters(cluster_matrix, cluster_size);

        let expected_cluster_matrix = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let expected_cluster_size = vec![1.0, 1.0];

        assert_eq!(new_cluster_matrix, expected_cluster_matrix);
        assert_eq!(new_cluster_size, expected_cluster_size);
    }
}
