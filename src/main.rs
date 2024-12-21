// Algorithms:
// Original hill climbing variant: https://dspace.mit.edu/handle/1721.1/29168
// Markov version:
// https://gitlab.eclipse.org/eclipse/escet/escet/-/blob/develop/common/org.eclipse.escet.common.dsm/src/org/eclipse/escet/common/dsm/DsmClustering.java
// Improved variant: https://www.researchgate.net/publication/267489785_Improved_Clustering_Algorithm_for_Design_Structure_Matrix

// https://eclipse.dev/escet/tools/dsm-clustering.html#
// implmenetation of above is on gitlab: https://gitlab.eclipse.org/eclipse/escet/escet
// Matlab macros https://dsmweb.org/matlab-macro-for-clustering-dsms/
use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod clustering;
mod io;

/// Tools for operating on Design Structure Matrices (DSMs)
#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Cluster the input CSV file
    Cluster {
        /// The path to the input CSV file to read
        #[arg(short, long, value_name = "INPUT.csv")]
        input: PathBuf,

        /// The path to the output CSV file to write
        #[arg(short, long, value_name = "OUTPUT.csv")]
        output: PathBuf,

        /// The initial temperature of the simulated annealing algorithm
        #[arg(short, long, default_value = "1000.0")]
        temperature: f64,

        /// The cooling rate for the simulated annealing algorithm
        #[arg(short, long, default_value = "0.99")]
        cooling_rate: f64,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::Cluster {
            input,
            output,
            temperature,
            cooling_rate,
        }) => {
            let dsm = io::read_csv(input.clone())
                .with_context(|| format!("failed to read csv file {:?}", input.clone()))?;

            let result = clustering::cluster(&dsm, *temperature, *cooling_rate);

            let (dsm, cost) = result.ok().unwrap();

            println!("cost_history: {:?}", cost);
            println!("dsm: {}", dsm);

            io::write_csv(output.clone(), dsm)?;
        }
        None => {}
    }

    Ok(())
}
