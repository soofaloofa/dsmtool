use anyhow::Result;
use csv::{ReaderBuilder, WriterBuilder};
use plotters::prelude::*;
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

pub fn plot_cost<P: AsRef<Path>>(path: P, cost_history: Vec<f64>) -> Result<()> {
    let root = BitMapBackend::new(path.as_ref(), (1024, 768)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    // get the maximum cost history value to set the y-axis range
    let max_y = cost_history
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    let mut chart = ChartBuilder::on(&root)
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .margin(10)
        .caption("Cost History", ("sans-serif", 40))
        .build_cartesian_2d(0f64..cost_history.len() as f64, 0f64..max_y.ceil())
        .unwrap();

    chart
        .configure_mesh()
        .y_desc("Coordination Cost")
        .x_desc("Iteration")
        .draw()
        .unwrap();

    chart
        .draw_series(LineSeries::new(
            cost_history.iter().enumerate().map(|(i, &y)| (i as f64, y)),
            &RED,
        ))
        .unwrap();

    Ok(())
}
