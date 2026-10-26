//! Self-play training samples and their file format.
//!
//! A file starts with the bytes `CNDRSP` and a version byte. Each sample follows as a u8 FEN
//! length and the FEN, the i8 result, a u8 move count, then each move's u16 bits and u16 visit
//! count, little-endian. Moves are stored as moves rather than network policy indices, so the
//! data stays valid if the network's move encoding changes.

use crate::chess::moves::Move;
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::Path,
};

const MAGIC: &[u8; 6] = b"CNDRSP";
const VERSION: u8 = 1;

/// One position from a self-play game, as a training target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Sample {
    pub(crate) fen: String,
    /// Every legal move with its visit count from the position's search
    pub(crate) visits: Vec<(Move, u16)>,
    /// The game's result for the side to move: 1 for a win, 0 for a draw, -1 for a loss
    pub(crate) result: i8,
}

pub(crate) struct SampleWriter {
    file: File,
}

impl SampleWriter {
    /// Opens `path` for appending, creating it and its directory if needed.
    pub(crate) fn open(path: &Path) -> io::Result<Self> {
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)?;
        }
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;

        if file.metadata()?.len() == 0 {
            file.write_all(MAGIC)?;
            file.write_all(&[VERSION])?;
        } else {
            let mut header = [0; MAGIC.len() + 1];
            File::open(path)?.read_exact(&mut header)?;
            if header[..MAGIC.len()] != MAGIC[..] || header[MAGIC.len()] != VERSION {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("{} isn't a version {VERSION} sample file", path.display()),
                ));
            }
        }

        Ok(Self { file })
    }

    /// Appends one game's samples with a single write, so stopping the run, or another process
    /// appending to the same file, doesn't split the game. A process killed during the write can
    /// still cut it short; readers skip what's left of it.
    pub(crate) fn write_game(&mut self, samples: &[Sample]) -> io::Result<()> {
        let mut bytes = vec![];

        for sample in samples {
            let fen = sample.fen.as_bytes();
            bytes.push(fen.len() as u8);
            bytes.extend_from_slice(fen);
            bytes.extend_from_slice(&[sample.result as u8, sample.visits.len() as u8]);

            for (mv, n_visits) in &sample.visits {
                bytes.extend_from_slice(&mv.0.to_le_bytes());
                bytes.extend_from_slice(&n_visits.to_le_bytes());
            }
        }

        self.file.write_all(&bytes)
    }
}

/// The contents of a sample file.
#[cfg(test)]
pub(crate) struct SampleFile {
    pub(crate) samples: Vec<Sample>,
    /// Bytes at the end that don't form a complete sample, left by a process killed mid-write
    pub(crate) truncated_bytes: usize,
}

/// Reads every sample in a file. The training loop will read data with this; for now it checks
/// that the format round-trips.
#[cfg(test)]
pub(crate) fn read_samples(path: &Path) -> io::Result<SampleFile> {
    let data = fs::read(path)?;
    let mut bytes = data.as_slice();

    let is_sample_file = take(&mut bytes, MAGIC.len() + 1)
        .is_some_and(|header| header[..MAGIC.len()] == MAGIC[..] && header[MAGIC.len()] == VERSION);
    if !is_sample_file {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a sample file",
        ));
    }

    let mut samples = vec![];
    while !bytes.is_empty() {
        let remaining = bytes.len();
        match read_sample(&mut bytes)? {
            Some(sample) => samples.push(sample),
            None => {
                return Ok(SampleFile {
                    samples,
                    truncated_bytes: remaining,
                })
            }
        }
    }

    Ok(SampleFile {
        samples,
        truncated_bytes: 0,
    })
}

// Returns None if the bytes run out before the sample ends
#[cfg(test)]
fn read_sample(bytes: &mut &[u8]) -> io::Result<Option<Sample>> {
    let Some(&[fen_len]) = take(bytes, 1) else {
        return Ok(None);
    };
    let Some(fen) = take(bytes, fen_len as usize) else {
        return Ok(None);
    };
    let fen = String::from_utf8(fen.to_vec())
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let Some(&[result, n_moves]) = take(bytes, 2) else {
        return Ok(None);
    };
    let Some(visits) = take(bytes, n_moves as usize * 4) else {
        return Ok(None);
    };

    let visits = visits
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&[move_low, move_high, visits_low, visits_high]| {
            let mv = Move(u16::from_le_bytes([move_low, move_high]));
            (mv, u16::from_le_bytes([visits_low, visits_high]))
        })
        .collect();

    Ok(Some(Sample {
        fen,
        visits,
        result: result as i8,
    }))
}

#[cfg(test)]
fn take<'a>(bytes: &mut &'a [u8], n: usize) -> Option<&'a [u8]> {
    let (head, rest) = bytes.split_at_checked(n)?;
    *bytes = rest;
    Some(head)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::State;
    use std::path::PathBuf;

    fn temp_path(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("cinder-{}-{name}.bin", std::process::id()));
        let _ = fs::remove_file(&path);
        path
    }

    fn sample(fen: &str, result: i8) -> Sample {
        let state = State::from_fen(fen);
        let visits = state
            .generate_moves()
            .iter()
            .enumerate()
            .map(|(i, mv)| (*mv, i as u16 * 7))
            .collect();
        Sample {
            fen: fen.to_string(),
            visits,
            result,
        }
    }

    #[test]
    fn samples_round_trip_across_appends() {
        let path = temp_path("round-trip");
        let first_game = vec![
            sample(
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                1,
            ),
            sample(
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
                -1,
            ),
        ];
        let second_game = vec![sample(
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            0,
        )];

        SampleWriter::open(&path)
            .unwrap()
            .write_game(&first_game)
            .unwrap();
        // A later run appends to the same file
        SampleWriter::open(&path)
            .unwrap()
            .write_game(&second_game)
            .unwrap();

        let expected: Vec<Sample> = first_game.into_iter().chain(second_game).collect();
        let file = read_samples(&path).unwrap();
        assert_eq!(file.samples, expected);
        assert_eq!(file.truncated_bytes, 0);
        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn a_cut_off_final_sample_is_skipped() {
        let path = temp_path("cut-off");
        let complete = vec![sample(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            1,
        )];
        let mut writer = SampleWriter::open(&path).unwrap();
        writer.write_game(&complete).unwrap();
        let complete_len = fs::metadata(&path).unwrap().len();
        writer
            .write_game(&[sample("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 0)])
            .unwrap();

        // As if the process were killed partway through writing the second game
        let full_len = fs::metadata(&path).unwrap().len();
        File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(full_len - 3)
            .unwrap();

        let file = read_samples(&path).unwrap();
        assert_eq!(file.samples, complete);
        assert_eq!(file.truncated_bytes as u64, full_len - 3 - complete_len);
        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn other_files_are_not_appended_to() {
        let path = temp_path("not-samples");
        fs::write(&path, b"some other data").unwrap();

        assert!(SampleWriter::open(&path).is_err());
        assert_eq!(fs::read(&path).unwrap(), b"some other data");
        fs::remove_file(&path).unwrap();
    }
}
