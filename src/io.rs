use anyhow::Result;
use csv::{ReaderBuilder, WriterBuilder};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use crate::clustering::Dsm;

#[derive(Debug, serde::Deserialize, Eq, PartialEq)]
struct Record {
    id: String,
    values: Vec<i32>,
}

pub fn read_csv<P: AsRef<Path>>(path: P) -> Result<Dsm> {
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

    Ok(Dsm::new(labels, data))
}

pub fn write_csv<P: AsRef<Path>>(path: P, dsm: Dsm) -> Result<()> {
    let file = File::create(path)?;
    let mut wtr = WriterBuilder::new().from_writer(BufWriter::new(file));

    // Insert a blank first element in the header
    let mut header = vec!["".to_string()];
    header.extend(dsm.labels.iter().cloned());
    wtr.write_record(&header)?;

    // Write the data with a label in the first column
    for (i, row) in dsm.matrix.iter().enumerate() {
        let mut record = vec![dsm.labels[i].clone()];
        record.extend(row.iter().map(|&f| f.to_string()));
        wtr.write_record(&record)?;
    }

    wtr.flush()?;
    Ok(())
}
