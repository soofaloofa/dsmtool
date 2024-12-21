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
    pub pow_cc: f64,             // penalty assigned to cluster size during costing
    pub pow_bid: f64,            // penalty assigned to cluster size during bidding
    pub pow_dep: f64,            // emphasizes high interactions
    pub max_cluster_size: usize, // max size of cluster
    pub bid_prob: f64,           // probability between 0 and 1 of accepting second highest bid
    pub times: u32,              // attempt "times" changes before checking for stability
}

pub fn cluster2(
    dsm_matrix: &Vec<Vec<f64>>,
    config: ClusteringConfig,
    max_iterations: usize,
    initial_temperature: f64,
    cooling_rate: f64,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>, f64) {
    println!("dsm_matrix: {:?}", dsm_matrix);
    let rng = &mut rand::thread_rng();

    // Set the initial cluster matrix to have a single cluster for each element along the diagonal
    let mut cluster_matrix: Vec<Vec<f64>> = (0..dsm_matrix.len())
        .map(|i| {
            let mut row = vec![0.0; dsm_matrix.len()];
            row[i] = 1.0;
            row
        })
        .collect();

    // All clusters start at size 1
    let mut cluster_size: Vec<f64> = vec![1.0; dsm_matrix.len()];

    // Calculate the initial starting coordination cost of the clustering
    let mut total_coord_cost =
        coord_cost(dsm_matrix, &cluster_matrix, &cluster_size, config.pow_cc);
    let mut best_coord_cost = total_coord_cost;
    let mut best_cluster_matrix = cluster_matrix.clone();
    let mut best_cluster_size = cluster_size.clone();

    let mut temperature = initial_temperature;

    for _ in 0..max_iterations {
        // If we have sufficiently cooled down, stop
        if temperature < 1e-3 {
            break;
        }

        for _ in 0..dsm_matrix.len() * config.times as usize {
            // Pick a random element from the DSM to put in a new cluster
            let elmt = rng.gen_range(0..dsm_matrix.len());

            // Accept bids for the element from all clusters
            let cluster_bid = bid(
                elmt,
                dsm_matrix,
                &cluster_matrix,
                cluster_size.clone(),
                config.clone(),
            );

            // Extract the two best bids
            let best_cluster_bid = *cluster_bid
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
            let secondbest_cluster_bid = *cluster_bid
                .iter()
                .filter(|&&x| x != best_cluster_bid)
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(&0.0);

            // Randomly select the second best bid a percentage of the time
            let selected_bid = if rng.gen_bool(config.bid_prob) {
                secondbest_cluster_bid
            } else {
                best_cluster_bid
            };

            if selected_bid > 0.0 {
                // Determine the list of affected clusters
                // These are the clusters that have the best bid and do not contain
                // the element
                let mut affected_clusters = vec![0; cluster_matrix.len()];
                for i in 0..cluster_matrix.len() {
                    if cluster_bid[i] == selected_bid && cluster_matrix[i][elmt] == 0.0 {
                        affected_clusters[i] = 1;
                    }
                }

                // Change the cluster matrix to reflect the new cluster assignments
                let mut new_cluster_matrix = cluster_matrix.clone();
                let mut new_cluster_size = cluster_size.clone();
                for i in 0..cluster_matrix.len() {
                    if affected_clusters[i] == 1 {
                        new_cluster_matrix[i][elmt] = 1.0;
                        new_cluster_size[i] += 1.0;
                    }
                }

                // Delete duplicate or empty clusters
                let (new_cluster_matrix, new_cluster_size) =
                    delete_clusters(new_cluster_matrix, new_cluster_size);

                // Calculate the change in coordination cost of the new
                // cluster matrix
                let new_total_coord_cost = coord_cost(
                    dsm_matrix,
                    &new_cluster_matrix,
                    &new_cluster_size,
                    config.pow_cc,
                );

                // Accept the new cluster assignment automatically if it has a
                // lower cost.  If the cost is higher, accept it with a
                // probability determined by the temperature
                let accept = if new_total_coord_cost <= total_coord_cost {
                    true
                } else {
                    let acceptance_probability =
                        ((total_coord_cost - new_total_coord_cost) / temperature).exp();
                    println!("acceptance_probability: {}", acceptance_probability);
                    rng.gen_bool(acceptance_probability)
                };

                if accept {
                    // Update the cluster with the accepted values
                    total_coord_cost = new_total_coord_cost;
                    cluster_matrix = new_cluster_matrix;
                    cluster_size = new_cluster_size;

                    // If we have a new best, update the best values
                    if total_coord_cost < best_coord_cost {
                        best_coord_cost = total_coord_cost;
                        best_cluster_matrix = cluster_matrix.clone();
                        best_cluster_size = cluster_size.clone();
                    }
                }
            }
        }

        temperature *= cooling_rate;
    }

    println!("dsm_matrix: {:?}", dsm_matrix);

    let ordered_matrix = reorder_dsm_by_cluster(dsm_matrix, &best_cluster_matrix);

    println!("ordered_matrix: {:?}", ordered_matrix);

    (
        ordered_matrix,
        best_cluster_matrix,
        best_cluster_size,
        best_coord_cost,
    )
}

/// Function to cluster the elements of a matrix Algorithm based on work
/// developed by Carlos Fernandez
///
/// This function runs a clustering algorithm then calculates the cost of the
/// proposed solution.  The objective is to find the solution that in the lowest
/// cost solution.
///
/// There is a higher cost for interactions that occur outside of clusters and
/// lower costs for interactions within clusters.  There are also penalties
/// assigned to the size of clusters to avoid a solution where all elements are
/// members of a single cluster.
///
/// There results are highly dependant on the parameters passed to the
/// algorithm.
// pub fn cluster(
//     dsm_matrix: &DsmMatrix,
//     config: ClusterConfig,
//     max_repeat: usize,
// ) -> (Vec<Vec<f64>>, Vec<f64>, f64) {
//     let rng = &mut rand::thread_rng();
//     // Set the initial cluster matrix to have a single cluster for each element
//     // along the diagonal
//     let mut cluster_matrix: ClusterMatrix = vec![vec![0.0; dsm_matrix.len()]; dsm_matrix.len()];
//     for i in 0..dsm_matrix.len() {
//         cluster_matrix[i][i] = 1.0;
//     }
//     // Set all cluster sizes to start at 1
//     let mut cluster_size: ClusterSize = vec![1.0; dsm_matrix.len()];

//     // Calculate the initial starting coordination cost of the clustering
//     let mut total_coord_cost =
//         coord_cost(dsm_matrix, &cluster_matrix, &cluster_size, config.pow_cc);
//     let mut best_coord_cost = total_coord_cost;
//     let mut best_curr_cost = total_coord_cost;
//     let mut best_cluster_matrix = cluster_matrix.clone();
//     let mut best_cluster_size = cluster_size.clone();

//     let mut stable; // indicates if the algorithm has met the stability criteria
//     let mut change; // indicates if a change should be made
//     let mut accept1; // indicates if the solution should be accepted
//     let mut first_run = true; // indicates if it is the first run through the algorithm
//     let mut pass = 1; // counter for the number of passes through the algorithm

//     // Continue until the results have met the stability criteria AND the final
//     // solution is the same or better than any intermediate solution that may
//     // have been found.  Due to simulated annealing, a final solution may be
//     // worse than an intermediate solution that had been found before making the
//     // random change
//     //
//     // If the final solution is not equal to or less than any best solution,
//     // then return to the best solution and continue to search for a better
//     // solution.  Do this until a better solution is found or we have looped
//     // back max_repeat times.  If we reach max_repeat, then report the best
//     // solution that had been found.
//     while (total_coord_cost > best_coord_cost && pass <= max_repeat) || first_run {
//         if !first_run {
//             pass += 1;
//             total_coord_cost = best_curr_cost;
//             cluster_matrix = best_cluster_matrix.clone();
//             cluster_size = best_cluster_size.clone();
//         }
//         first_run = false;
//         stable = 0;
//         accept1 = 0;
//         change = 0;

//         // Loop until we've reached the limit for finding a stable solution
//         while stable <= config.stable_limit {
//             for _ in 0..dsm_matrix.len() as u32 * config.times {
//                 // Pick a random element from the DSM to put in a new cluster
//                 let elmt = rng.gen_range(0..dsm_matrix.len());

//                 // Accept bids for the elemement from all clusters
//                 let cluster_bid = bid(
//                     elmt,
//                     dsm_matrix,
//                     &cluster_matrix,
//                     cluster_size.clone(),
//                     config.clone(),
//                 );

//                 // Extract the two best bids
//                 let mut best_bid = *cluster_bid
//                     .iter()
//                     .max_by(|a, b| a.partial_cmp(b).unwrap())
//                     .unwrap_or(&0.0);
//                 let second_best_bid = *cluster_bid
//                     .iter()
//                     .filter(|&&x| x != best_bid)
//                     .max_by(|a, b| a.partial_cmp(b).unwrap())
//                     .unwrap_or(&0.0);

//                 // Simulated annealing
//                 // Randomly accept the second best bid a percentage of the time
//                 if rng.gen_bool(config.bid_prob) {
//                     best_bid = second_best_bid;
//                 }

//                 if best_bid > 0.0 {
//                     let mut cluster_list = vec![0; cluster_matrix.len()];

//                     // Determine the list of affected clusters
//                     // These are the clusters that have the best bid and do not contain
//                     // the element
//                     for i in 0..cluster_matrix.len() {
//                         if cluster_bid[i] == best_bid && cluster_matrix[i][elmt] == 0.0 {
//                             cluster_list[i] = 1;
//                         }
//                     }

//                     // Copy the cluster matrix into new matrices
//                     let mut new_cluster_matrix = cluster_matrix.clone();
//                     let mut new_cluster_size = cluster_size.clone();

//                     // Change the cluster matrix to reflect the new cluster assignments
//                     for i in 0..cluster_matrix.len() {
//                         if cluster_list[i] == 1 {
//                             new_cluster_matrix[i][elmt] = 1.0;
//                             new_cluster_size[i] += 1.0;
//                         }
//                     }

//                     // Delete duplicate or empty clusters
//                     let (new_cluster_matrix, new_cluster_size) =
//                         delete_clusters(new_cluster_matrix, new_cluster_size);

//                     // Calculate the change in coordination cost of the new
//                     // cluster matrix
//                     let new_total_coord_cost = coord_cost(
//                         dsm_matrix,
//                         &new_cluster_matrix,
//                         &new_cluster_size,
//                         config.pow_cc,
//                     );
//                     if new_total_coord_cost <= total_coord_cost {
//                         accept1 = 1;
//                     } else if rng.gen_bool(config.accept_prob) {
//                         // Still accept randomly sometimes even if not optimal cost
//                         accept1 = 1;

//                         // TODO:
//                         // if we are going to accept a total cost that is not less
//                         // than our current cost then save the current cost as the
//                         // best current cost found so far (only if the current cost
//                         // is lower than any best current cost previously saved)
//                         // because we may not find a cost that is better than the
//                         // current cost
//                         //
//                         // When we think we are finished we will check the final
//                         // cost against any best cost if the final cost is not
//                         // better than the lowest cost found, then we will move back
//                         // to that best cost
//                         if total_coord_cost < best_curr_cost {
//                             best_curr_cost = total_coord_cost;
//                             best_cluster_matrix = cluster_matrix.clone();
//                             best_cluster_size = cluster_size.clone();
//                         }
//                     } else {
//                         accept1 = 0;
//                     }

//                     if accept1 == 1 {
//                         accept1 = 0;

//                         // Update the new cluster values
//                         total_coord_cost = new_total_coord_cost;
//                         cluster_matrix = new_cluster_matrix;
//                         cluster_size = new_cluster_size;

//                         if best_coord_cost > total_coord_cost {
//                             best_coord_cost = total_coord_cost;
//                             change += 1;
//                         }
//                     }
//                 }
//             }

//             // Test for system stability
//             if change > 0 {
//                 stable = 0;
//                 change = 0;
//             } else {
//                 stable += 1;
//             }
//         }
//     }

//     (cluster_matrix, cluster_size, total_coord_cost)
// }

/// Function to calculate the bids from clusters for the selected element. Each
/// cluster makes a bid for the selected element based on the bidding
/// parameters.
fn bid(
    elmt: usize,
    dsm_matrix: &Vec<Vec<f64>>,
    cluster_matrix: &Vec<Vec<f64>>,
    cluster_size: Vec<f64>,
    config: ClusteringConfig,
) -> Vec<f64> {
    // Initilize all bids to 0
    let mut cluster_bid = vec![0.0; cluster_matrix.len()];

    // For each cluster, if any element in the cluster has an interaction with
    // the selected element then add the number of interactions with the
    // selected element.  Then use the number of interactions to calculate the
    // bid.
    for i in 0..cluster_matrix.len() {
        if (cluster_size[i] < config.max_cluster_size as f64) && cluster_size[i] > 0.0 {
            let sum_dsm_cluster: f64 = dsm_matrix[elmt]
                .iter()
                .zip(&cluster_matrix[i])
                .map(|(dsm, cluster)| dsm * cluster)
                .sum();
            cluster_bid[i] =
                (sum_dsm_cluster.powf(config.pow_dep)) / (cluster_size[i].powf(config.pow_bid));
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

/// Place zeros along the diagonal (top-left to bottom-right) of the input matrix
fn zero_diagonal(matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut matrix = matrix.clone();
    for i in 0..matrix.len() {
        matrix[i][i] = 0.0;
    }
    matrix
}

fn find(matrix: &Vec<Vec<f64>>) -> (Vec<usize>, Vec<usize>) {
    let mut cluster_indices = Vec::new();
    let mut element_indices = Vec::new();

    for (i, row) in matrix.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            if val > 0.0 {
                cluster_indices.push(i);
                element_indices.push(j);
            }
        }
    }

    (cluster_indices, element_indices)
}

/// Sort the cluster list in ascending order of clusters
///
/// Example:
///
/// B = 3x1 datetime
///    1992-01-12
///    2012-12-22
///    2063-04-05
///
/// I = 3×1
///
///      3
///      1
///      2
///
/// B lists the sorted dates and I contains the corresponding indices of A.
///
/// Access the sorted elements from the original array directly by using the index array I.
fn sort(cluster_number: Vec<usize>) -> Vec<usize> {
    let mut cluster_list_index: Vec<usize> = (0..cluster_number.len()).collect();
    cluster_list_index.sort_by_key(|&i| cluster_number[i]);
    cluster_list_index
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
    let mut element_order = Vec::new();

    for (_i, row) in cluster_matrix.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            if val != 0.0 {
                element_order.push(j);
            }
        }
    }

    let new_number_elmts = element_order.len();
    let mut new_dsm_matrix = vec![vec![0.0; new_number_elmts]; new_number_elmts];

    for (new_i, &old_i) in element_order.iter().enumerate() {
        for (new_j, &old_j) in element_order.iter().enumerate() {
            new_dsm_matrix[new_i][new_j] = dsm_matrix[old_i][old_j];
        }
    }

    new_dsm_matrix
}

fn create_new_cluster_matrix(cluster_matrix: &Vec<Vec<f64>>, new_dsm_size: usize) -> Vec<Vec<f64>> {
    let n_clusters = cluster_matrix.len();
    let mut new_cluster_matrix = vec![vec![0.0; new_dsm_size]; n_clusters];
    let num_cluster_elements = cluster_matrix
        .iter()
        .map(|row| row.iter().sum())
        .collect::<Vec<f64>>();
    let mut n = 0;

    for i in 0..n_clusters {
        for j in 0..num_cluster_elements[i] as usize {
            new_cluster_matrix[i][n + j] = 1.0;
        }
        n += num_cluster_elements[i] as usize;
    }

    new_cluster_matrix
}

#[cfg(test)]

mod tests {
    use super::*;

    #[test]
    fn test_zero_diagonal() {
        let dsm_matrix = vec![
            vec![1.0, 1.0, 0.0, 0.0],
            vec![1.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];

        // Expected
        let expected = vec![
            vec![0.0, 1.0, 0.0, 0.0],
            vec![1.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 1.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];

        let output = zero_diagonal(&dsm_matrix);

        assert_eq!(output, expected);
    }

    #[test]
    fn test_find() {
        let cluster_matrix = vec![vec![1.0, 0.0, 0.0, 1.0], vec![0.0, 1.0, 1.0, 0.0]];
        let (cluster_number, element) = find(&cluster_matrix);

        assert_eq!(cluster_number, vec![0, 0, 1, 1]);
        assert_eq!(element, vec![0, 3, 1, 2]);
    }

    #[test]
    fn test_sort() {
        let cluster_number: Vec<usize> = vec![0, 0, 1, 1];
        assert_eq!(sort(cluster_number), vec![0, 1, 2, 3]);

        let cluster_number: Vec<usize> = vec![1, 0, 3, 2];
        // index 1 is lowest value, index 0 second lowest, etc.
        assert_eq!(sort(cluster_number), vec![1, 0, 3, 2]);

        let cluster_number: Vec<usize> = vec![0, 1, 1, 2, 2, 3];
        assert_eq!(sort(cluster_number), vec![0, 1, 2, 3, 4, 5]);
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
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0, 1.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
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

        let config = ClusteringConfig {
            max_cluster_size: 4,
            pow_dep: 1.0,
            pow_bid: 1.0,
            pow_cc: 0.0,
            bid_prob: 0.25,
            times: 10,
        };

        let result = bid(1, &dsm_matrix, &cluster_matrix, cluster_size, config);
        let expected = vec![0.5, 0.5];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_bid_with_empty_cluster() {
        let config = ClusteringConfig {
            max_cluster_size: 2,
            pow_dep: 1.0,
            pow_bid: 1.0,
            pow_cc: 1.0,
            bid_prob: 10.0,
            times: 10,
        };

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

        let result = bid(1, &dsm_matrix, &cluster_matrix, cluster_size, config);
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

        // clusters are over max size so don't bid
        let config = ClusteringConfig {
            max_cluster_size: 1,
            pow_dep: 1.0,
            pow_bid: 1.0,
            pow_cc: 0.0,
            bid_prob: 0.25,
            times: 10,
        };

        let result = bid(1, &dsm_matrix, &cluster_matrix, cluster_size, config);
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
        let config = ClusteringConfig {
            max_cluster_size: 2,
            pow_dep: 2.0,
            pow_bid: 0.5,
            pow_cc: 0.0,
            bid_prob: 0.25,
            times: 10,
        };

        let result = bid(1, &dsm_matrix, &cluster_matrix, cluster_size, config);
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

    // #[test]
    // fn test_cluster_basic() {
    //     // Sample DSM matrix
    //     let dsm_matrix = vec![
    //         vec![0.0, 1.0, 0.0, 0.0],
    //         vec![1.0, 0.0, 1.0, 0.0],
    //         vec![0.0, 1.0, 0.0, 1.0],
    //         vec![0.0, 0.0, 1.0, 0.0],
    //     ];

    //     let max_repeat = 10;
    //     let config = ClusteringConfig {
    //         max_cluster_size: 2,
    //         pow_dep: 1.0,
    //         pow_bid: 1.0,
    //         pow_cc: 1.0,
    //         bid_prob: 10.0,
    //         times: 10,
    //     };

    //     let (new_cluster_matrix, new_cluster_size, new_total_coord_cost) =
    //         cluster2(&dsm_matrix, config, max_repeat);

    //     // Expected results (these are just example values, adjust as needed)
    //     let expected_cluster_matrix = vec![
    //         vec![1.0, 1.0, 0.0, 0.0],
    //         vec![0.0, 1.0, 1.0, 0.0],
    //         vec![0.0, 0.0, 1.0, 1.0],
    //     ];
    //     let expected_cluster_size = vec![2.0, 2.0, 2.0];
    //     let expected_total_coord_cost = 12.0;

    //     assert_eq!(new_cluster_matrix, expected_cluster_matrix);
    //     assert_eq!(new_cluster_size, expected_cluster_size);
    //     assert_eq!(new_total_coord_cost, expected_total_coord_cost);
    // }
}
