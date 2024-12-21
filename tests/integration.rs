use assert_cmd::Command;
// use assert_fs::prelude::*;
// use predicates::prelude::*;
use std::error::Error;

#[test]
/// make sure the binary runs
fn test_help() -> Result<(), Box<dyn Error>> {
    let mut cmd = Command::cargo_bin("dsmtool")?;
    let assert = cmd.arg("--help").assert();
    assert.success().stderr("");
    Ok(())
}

#[test]
fn test_read_csv_fails_when_file_missing() -> Result<(), Box<dyn Error>> {
    let mut cmd = Command::cargo_bin("dsmtool")?;
    let assert = cmd.arg("tests/unknown-file.csv").assert();
    assert.failure();
    Ok(())
}

// #[test]
// fn test_read_csv() {
//     let temp_dir = assert_fs::TempDir::new().unwrap();
// TODO: Generate test file
// https://rust-cli.github.io/book/tutorial/testing.html#generating-test-files

//     let mut cmd = Command::cargo_bin("dsm").unwrap();
//     let fake_editor_path = std::env::current_dir()
//         .expect("expect to be in a dir")
//         .join("tests")
//         .join("fake-editor.sh");
//     if !fake_editor_path.exists() {
//         panic!("fake editor shell script could not be found")
//     }

//     let assert = cmd
//         .env("EDITOR", fake_editor_path.into_os_string())
//         .env("GARDEN_PATH", temp_dir.path())
//         .arg("write")
//         .arg("-t")
//         .arg("atitle")
//         .write_stdin("N\n".as_bytes())
//         .assert();

//     assert.success();

//     temp_dir
//         .child("atitle.md")
//         .assert(predicate::path::exists());
// }
