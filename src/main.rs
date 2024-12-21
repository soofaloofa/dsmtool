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

        /// The path to the output to write
        #[arg(short, long, value_name = "OUTPUT.csv")]
        output: PathBuf,

        /// The initial temperature of the simulated annealing algorithm
        #[arg(short, long, default_value = "1000.0")]
        temperature: f64,

        /// The cooling rate for the simulated annealing algorithm
        #[arg(short, long, default_value = "0.99")]
        rate: f64,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::Cluster {
            input,
            output,
            temperature,
            rate,
        }) => {
            let dsm = io::read_csv(input.clone())
                .with_context(|| format!("failed to read csv file {:?}", input.clone()))?;

            let result = clustering::cluster(&dsm, *temperature, *rate);

            let (dsm, cost_history) = result.ok().unwrap();

            io::write_csv(output.with_extension("csv"), dsm.clone())?;
            io::plot_cost(output.with_extension("png"), cost_history)?;
        }
        None => {}
    }

    Ok(())
}
