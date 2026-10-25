//! Self-play training samples and their file format.
//!
//! A file starts with the bytes `CNDRSP` and a version byte. Each sample follows as a u8 FEN
//! length and the FEN, the i8 result, a u8 move count, then each move's u16 bits and u16 visit
//! count, little-endian. Moves are stored as moves rather than network policy indices, so the
//! data stays valid if the network's move encoding changes.

use crate::chess::moves::Move;
use std::{
    fs::{self, File, OpenOptions},
    io::{self, BufWriter, Read, Write},
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
    file: BufWriter<File>,
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

        Ok(Self {
            file: BufWriter::new(file),
        })
    }

    /// Writes one game's samples and flushes, so a stopped run leaves only complete games.
    pub(crate) fn write_game(&mut self, samples: &[Sample]) -> io::Result<()> {
        for sample in samples {
            let fen = sample.fen.as_bytes();
            self.file.write_all(&[fen.len() as u8])?;
            self.file.write_all(fen)?;
            self.file
                .write_all(&[sample.result as u8, sample.visits.len() as u8])?;

            for (mv, n_visits) in &sample.visits {
                self.file.write_all(&mv.0.to_le_bytes())?;
                self.file.write_all(&n_visits.to_le_bytes())?;
            }
        }

        self.file.flush()
    }
}

/// Reads every sample in a file. The training loop will read data with this; for now it checks
/// that the format round-trips.
#[cfg(test)]
pub(crate) fn read_samples(path: &Path) -> io::Result<Vec<Sample>> {
    let data = fs::read(path)?;
    let mut bytes = data.as_slice();

    let header = take(&mut bytes, MAGIC.len() + 1)?;
    if header[..MAGIC.len()] != MAGIC[..] || header[MAGIC.len()] != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a sample file",
        ));
    }

    let mut samples = vec![];
    while !bytes.is_empty() {
        let fen_len = take(&mut bytes, 1)?[0] as usize;
        let fen = String::from_utf8(take(&mut bytes, fen_len)?.to_vec())
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        let [result, n_moves] = take(&mut bytes, 2)? else {
            unreachable!()
        };

        let (visits, _) = take(&mut bytes, *n_moves as usize * 4)?.as_chunks::<4>();
        let visits = visits
            .iter()
            .map(|&[move_low, move_high, visits_low, visits_high]| {
                let mv = Move(u16::from_le_bytes([move_low, move_high]));
                (mv, u16::from_le_bytes([visits_low, visits_high]))
            })
            .collect();

        samples.push(Sample {
            fen,
            visits,
            result: *result as i8,
        });
    }

    Ok(samples)
}

#[cfg(test)]
fn take<'a>(bytes: &mut &'a [u8], n: usize) -> io::Result<&'a [u8]> {
    if bytes.len() < n {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "truncated sample file",
        ));
    }
    let (head, rest) = bytes.split_at(n);
    *bytes = rest;
    Ok(head)
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
        assert_eq!(read_samples(&path).unwrap(), expected);
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
