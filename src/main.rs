use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;

mod clustering;
mod dsm;
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

    println!("file content: {:?}", content);

    let config = clustering::ClusteringConfig { pow_cc: 1.0 };

    let result = clustering::cluster(&content.data, config, 1000.0, 0.99);

    let cost = result.ok().unwrap();

    println!("cost: {:?}", cost);

    // println!("cluster_matrix: {:?}", cluster_matrix);
    // println!("cluster_size: {:?}", cluster_size);
    // println!("total_coord_cost: {:?}", total_coord_cost);

    // let _ = io::write_csv("out.csv", after_dsm);
    // TODO:
    // - [ ] Write output to file
    // - [ ] Graph total cost history?
    // Ok(())

    Ok(())
}
