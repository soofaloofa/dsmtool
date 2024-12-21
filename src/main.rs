// Algorithms:
// Original hill climbing variant: https://dspace.mit.edu/handle/1721.1/29168
// Markov version:
// https://gitlab.eclipse.org/eclipse/escet/escet/-/blob/develop/common/org.eclipse.escet.common.dsm/src/org/eclipse/escet/common/dsm/DsmClustering.java
// Improved variant: https://www.researchgate.net/publication/267489785_Improved_Clustering_Algorithm_for_Design_Structure_Matrix

// https://eclipse.dev/escet/tools/dsm-clustering.html#
// implmenetation of above is on gitlab: https://gitlab.eclipse.org/eclipse/escet/escet
// Matlab macros https://dsmweb.org/matlab-macro-for-clustering-dsms/
use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;

mod clustering;
mod io;

/// Search for a pattern in a file and display the lines that contain it.
#[derive(Parser)]
struct Cli {
    /// The path to the input CSV file to read
    path: PathBuf,
}

fn main() -> Result<()> {
    let args = Cli::parse();

    let content = io::read_csv(args.path.clone())
        .with_context(|| format!("failed to read csv file {:?}", args.path))?;

    let result = clustering::cluster(&content, 1000.0, 0.99);

    let (dsm, cost) = result.ok().unwrap();

    println!("cost_history: {:?}", cost);
    println!("dsm: {}", dsm);

    io::write_csv("out.csv", dsm)?;
    // TODO:
    // - [ ] Write output to file
    // - [ ] Graph total cost history?
    // Ok(())

    Ok(())
}
