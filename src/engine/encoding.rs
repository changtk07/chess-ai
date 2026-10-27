//! How positions and moves are encoded for the network: the pieces on each square and the rest of
//! the position's state as input, and each legal move as an index into the policy head's output.

use crate::chess::{
    board::CastlingRights,
    moves::{Move, MoveType},
    types::Direction,
    State,
};
use burn::{
    tensor::{backend::Backend, Int, TensorData},
    Tensor,
};

/// Piece tokens: 0 for an empty square, then 1-12 for the pieces in `Piece` order
pub(crate) const N_PIECE_TOKENS: usize = 13;
/// Inputs to the CLS token: side to move, castling rights, en passant file and the halfmove clock
pub(crate) const N_STATE_FEATURES: usize = 15;
pub(crate) const N_MOVE_PLANES: usize = 73;
const MAX_RAY_MOVE_DIST: usize = 7;
const KNIGHT_MOVE_PLANE_START: usize = 56;
const KNIGHT_PROMO_PLANE_START: i8 = 64;
const BISHOP_PROMO_PLANE_START: i8 = 67;
const ROOK_PROMO_PLANE_START: i8 = 70;

impl Move {
    pub(crate) fn policy_index(&self) -> usize {
        let (from, to, mv_type) = self.unpack();
        let base = from.0 as usize * N_MOVE_PLANES;
        let (rank_diff, file_diff) = (
            to.rank() as i8 - from.rank() as i8,
            to.file() as i8 - from.file() as i8,
        );

        match mv_type {
            MoveType::PromotionKnight => {
                return base + (KNIGHT_PROMO_PLANE_START + file_diff + 1) as usize
            }
            MoveType::PromotionBishop => {
                return base + (BISHOP_PROMO_PLANE_START + file_diff + 1) as usize
            }
            MoveType::PromotionRook => {
                return base + (ROOK_PROMO_PLANE_START + file_diff + 1) as usize
            }
            _ => (),
        };

        let plane = match (rank_diff, file_diff) {
            (2, 1) => KNIGHT_MOVE_PLANE_START,
            (1, 2) => KNIGHT_MOVE_PLANE_START + 1,
            (-1, 2) => KNIGHT_MOVE_PLANE_START + 2,
            (-2, 1) => KNIGHT_MOVE_PLANE_START + 3,
            (-2, -1) => KNIGHT_MOVE_PLANE_START + 4,
            (-1, -2) => KNIGHT_MOVE_PLANE_START + 5,
            (1, -2) => KNIGHT_MOVE_PLANE_START + 6,
            (2, -1) => KNIGHT_MOVE_PLANE_START + 7,
            _ => {
                let (dir, dist) = match (rank_diff.signum(), file_diff.signum()) {
                    (1, 0) => (Direction::North, rank_diff),
                    (-1, 0) => (Direction::South, -rank_diff),
                    (0, 1) => (Direction::East, file_diff),
                    (0, -1) => (Direction::West, -file_diff),
                    (1, 1) => (Direction::NorthEast, rank_diff),
                    (1, -1) => (Direction::NorthWest, rank_diff),
                    (-1, 1) => (Direction::SouthEast, -rank_diff),
                    (-1, -1) => (Direction::SouthWest, -rank_diff),
                    _ => unreachable!(),
                };
                dir as usize * MAX_RAY_MOVE_DIST + dist as usize - 1
            }
        };

        base + plane
    }
}

/// A batch of positions with their legal moves, encoded for the network. Each tensor is built in
/// one host buffer and uploaded once.
pub(crate) struct NetworkInput<B: Backend> {
    /// Each square's piece token, [batch_size, 64]
    pub(crate) pieces: Tensor<B, 2, Int>,
    /// [batch_size, N_STATE_FEATURES]
    pub(crate) state_features: Tensor<B, 2>,
    /// Policy indices of each position's legal moves, padded with 0 to the longest row,
    /// [batch_size, max legal moves]
    pub(crate) move_indices: Tensor<B, 2, Int>,
    /// 0 for moves and -1e9 for padding, to add to the gathered logits before the softmax
    pub(crate) padding: Tensor<B, 2>,
}

impl<B: Backend> NetworkInput<B> {
    /// Encodes each position with its legal moves, whose logits the network returns in the order
    /// given.
    pub(crate) fn new(states: &[&State], moves: &[impl AsRef<[Move]>], device: &B::Device) -> Self {
        assert_eq!(states.len(), moves.len());
        let batch_size = states.len();

        let mut pieces = Vec::with_capacity(batch_size * 64);
        let mut state_features = Vec::with_capacity(batch_size * N_STATE_FEATURES);
        for state in states {
            pieces.extend(
                state
                    .board
                    .mailbox
                    .iter()
                    .map(|square| square.map_or(0i32, |piece| piece as i32 + 1)),
            );
            state_features.extend(encode_state_features(state));
        }

        let (move_indices, padding) = legal_move_indices(moves, device);
        Self {
            pieces: Tensor::from_data(TensorData::new(pieces, [batch_size, 64]), device),
            state_features: Tensor::from_data(
                TensorData::new(state_features, [batch_size, N_STATE_FEATURES]),
                device,
            ),
            move_indices,
            padding,
        }
    }
}

// Side to move (1 for white, -1 for black), castling rights, the en passant file one-hot (index 0
// for none, 1-8 for files a-h), and the halfmove clock scaled down
fn encode_state_features(state: &State) -> [f32; N_STATE_FEATURES] {
    let mut features = [0.0; N_STATE_FEATURES];
    features[0] = (state.turn as i8 * -2 + 1) as f32;

    let rights = [
        CastlingRights::WHITE_KING_SIDE,
        CastlingRights::WHITE_QUEEN_SIDE,
        CastlingRights::BLACK_KING_SIDE,
        CastlingRights::BLACK_QUEEN_SIDE,
    ];
    for (feature, right) in features[1..5].iter_mut().zip(rights) {
        *feature = state.castling_rights.has(right) as i8 as f32;
    }

    features[5 + state.en_passant.map_or(0, |ep| ep.file() as usize + 1)] = 1.0;
    features[14] = state.halfmove_clock as f32 / 100.0;
    features
}

// Each position's legal-move policy indices and padding, one row per position and padded to the
// longest row, so the network can gather just those logits
fn legal_move_indices<B: Backend>(
    moves: &[impl AsRef<[Move]>],
    device: &B::Device,
) -> (Tensor<B, 2, Int>, Tensor<B, 2>) {
    let width = moves
        .iter()
        .map(|moves| moves.as_ref().len())
        .max()
        .unwrap_or(0);
    assert!(
        width > 0,
        "positions without legal moves have nothing to evaluate"
    );

    let mut indices = vec![0i32; moves.len() * width];
    let mut padding = vec![-1e9f32; moves.len() * width];
    let rows = indices
        .chunks_exact_mut(width)
        .zip(padding.chunks_exact_mut(width));

    for ((index_row, padding_row), moves) in rows.zip(moves) {
        for ((index, pad), mv) in index_row.iter_mut().zip(padding_row).zip(moves.as_ref()) {
            *index = mv.policy_index() as i32;
            *pad = 0.0;
        }
    }

    let shape = [moves.len(), width];
    (
        Tensor::from_data(TensorData::new(indices, shape), device),
        Tensor::from_data(TensorData::new(padding, shape), device),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::{state::START_FEN, types::Piece};
    use burn::backend::NdArray;

    const N_POTENTIAL_MOVES: usize = 64 * N_MOVE_PLANES;

    const FENS: [&str; 4] = [
        START_FEN,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    ];

    fn encode(fens: &[&str]) -> (NetworkInput<NdArray>, Vec<Vec<Move>>) {
        let states: Vec<State> = fens.iter().map(|fen| State::from_fen(fen)).collect();
        let states: Vec<&State> = states.iter().collect();
        let moves: Vec<Vec<Move>> = states
            .iter()
            .map(|state| state.generate_moves().to_vec())
            .collect();
        let input = NetworkInput::new(&states, &moves, &Default::default());
        (input, moves)
    }

    fn ints(tensor: Tensor<NdArray, 2, Int>) -> Vec<i32> {
        tensor.into_data().convert::<i32>().to_vec().unwrap()
    }

    fn floats(tensor: Tensor<NdArray, 2>) -> Vec<f32> {
        tensor.into_data().to_vec().unwrap()
    }

    #[test]
    fn positions_are_encoded_square_by_square() {
        // Made up to set every kind of feature: black to move, white keeps only king-side castling
        // and black only queen-side, en passant on the e-file, and 7 halfmoves
        let made_up = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b Kq e3 7 1";
        let (input, _) = encode(&[START_FEN, made_up]);
        assert_eq!(input.pieces.dims(), [2, 64]);
        assert_eq!(input.state_features.dims(), [2, N_STATE_FEATURES]);

        let pieces = ints(input.pieces);
        let token = |piece: Piece| piece as i32 + 1;
        // a1, e1, e2, e4 and e8 of the second position
        assert_eq!(pieces[64], token(Piece::WhiteRook));
        assert_eq!(pieces[64 + 4], token(Piece::WhiteKing));
        assert_eq!(pieces[64 + 12], 0);
        assert_eq!(pieces[64 + 28], token(Piece::WhitePawn));
        assert_eq!(pieces[64 + 60], token(Piece::BlackKing));
        assert_eq!(pieces[..64].iter().filter(|&&token| token == 0).count(), 32);

        let features = floats(input.state_features);
        let start_features = [1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.];
        assert_eq!(features[..N_STATE_FEATURES], start_features);
        let made_up_features = [
            -1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.07,
        ];
        assert_eq!(features[N_STATE_FEATURES..], made_up_features);
    }

    #[test]
    fn legal_move_indices_are_padded_to_the_longest_row() {
        let (input, moves) = encode(&FENS);

        let width = moves.iter().map(|moves| moves.len()).max().unwrap();
        assert_eq!(input.move_indices.dims(), [FENS.len(), width]);
        assert_eq!(input.padding.dims(), [FENS.len(), width]);

        let indices = ints(input.move_indices);
        let padding = floats(input.padding);
        let rows = indices.chunks_exact(width).zip(padding.chunks_exact(width));
        for ((index_row, padding_row), moves) in rows.zip(&moves) {
            for (i, mv) in moves.iter().enumerate() {
                assert_eq!(index_row[i], mv.policy_index() as i32);
                assert_eq!(padding_row[i], 0.0);
            }
            assert!(index_row[moves.len()..].iter().all(|&index| index == 0));
            assert!(padding_row[moves.len()..].iter().all(|&pad| pad == -1e9));
        }
    }

    #[test]
    fn test_move_indices_are_unique_and_in_bounds() {
        for fen in FENS {
            let state = State::from_fen(fen);
            let moves = state.generate_moves();
            let mut indices = Vec::new();
            for mv in moves {
                let idx = mv.policy_index();
                assert!(
                    idx < N_POTENTIAL_MOVES,
                    "Move index {} out of bounds for move {:?}",
                    idx,
                    mv
                );
                assert!(
                    !indices.contains(&idx),
                    "Duplicate move index {} for move {:?} in position:\n{}",
                    idx,
                    mv,
                    fen
                );
                indices.push(idx);
            }
        }
    }
}
