use anyhow::Result;
use csv::{ReaderBuilder, WriterBuilder};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use crate::dsm::DSM;

#[derive(Debug, serde::Deserialize, Eq, PartialEq)]
struct Record {
    id: String,
    values: Vec<i32>,
}

// TODO:
// - [ ] Clustering algorithm

// Algorithms:
// Original hill climbing variant: https://dspace.mit.edu/handle/1721.1/29168
// Markov version:
// https://gitlab.eclipse.org/eclipse/escet/escet/-/blob/develop/common/org.eclipse.escet.common.dsm/src/org/eclipse/escet/common/dsm/DsmClustering.java
// Improved variant: https://www.researchgate.net/publication/267489785_Improved_Clustering_Algorithm_for_Design_Structure_Matrix

// https://eclipse.dev/escet/tools/dsm-clustering.html#
// implmenetation of above is on gitlab: https://gitlab.eclipse.org/eclipse/escet/escet
// Matlab macros https://dsmweb.org/matlab-macro-for-clustering-dsms/
pub fn read_csv<P: AsRef<Path>>(path: P) -> Result<DSM> {
    let file = File::open(path)?;
    let mut rdr = ReaderBuilder::new()
        .flexible(true)
        .has_headers(true)
        .trim(csv::Trim::All)
        .from_reader(BufReader::new(file));

    let mut data: Vec<Vec<f64>> = vec![];

    for result in rdr.records() {
        let record = result?;
        let mut row: Vec<f64> = vec![];
        record.iter().skip(1).for_each(|s| {
            row.push(s.parse().unwrap());
        });

        data.push(row);
    }

    let labels = rdr
        .headers()?
        .clone()
        .iter()
        .skip(1)
        .map(|s| s.to_string())
        .collect();

    let size = u32::try_from(data.len()).unwrap();

    Ok(DSM {
        labels: labels,
        data,
        size: size,
    })
}

pub fn write_csv(file_path: &str, data: Vec<Vec<f64>>) -> Result<()> {
    let file = File::create(file_path)?;
    let mut wtr = WriterBuilder::new().from_writer(BufWriter::new(file));

    for row in data {
        let row_as_strings: Vec<String> = row.iter().map(|&f| f.to_string()).collect();
        wtr.write_record(row_as_strings)?;
    }

    wtr.flush()?;
    Ok(())
}
