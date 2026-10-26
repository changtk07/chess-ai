use crate::chess::{
    moves::{Move, MoveType},
    types::Direction,
};
use burn::{
    tensor::{backend::Backend, Int, TensorData},
    Tensor,
};

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

pub(crate) trait LegalMoveIndices<B: Backend> {
    fn legal_move_indices(&self, device: &B::Device) -> (Tensor<B, 2, Int>, Tensor<B, 2>);
}

/// Policy indices of each position's legal moves, one row per position and padded to the longest
/// row, so the network can gather just those logits. The second tensor is 0 for moves and -1e9
/// for padding, to add to the gathered logits before the softmax.
impl<B: Backend, M: AsRef<[Move]>> LegalMoveIndices<B> for [M] {
    fn legal_move_indices(&self, device: &B::Device) -> (Tensor<B, 2, Int>, Tensor<B, 2>) {
        let width = self
            .iter()
            .map(|moves| moves.as_ref().len())
            .max()
            .unwrap_or(0);
        assert!(
            width > 0,
            "positions without legal moves have nothing to evaluate"
        );

        let mut indices = vec![0i32; self.len() * width];
        let mut padding = vec![-1e9f32; self.len() * width];
        let rows = indices
            .chunks_exact_mut(width)
            .zip(padding.chunks_exact_mut(width));

        for ((index_row, padding_row), moves) in rows.zip(self) {
            for ((index, pad), mv) in index_row.iter_mut().zip(padding_row).zip(moves.as_ref()) {
                *index = mv.policy_index() as i32;
                *pad = 0.0;
            }
        }

        let shape = [self.len(), width];
        (
            Tensor::from_data(TensorData::new(indices, shape), device),
            Tensor::from_data(TensorData::new(padding, shape), device),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::state::START_FEN;
    use crate::chess::State;
    use burn::backend::NdArray;

    const N_POTENTIAL_MOVES: usize = 64 * N_MOVE_PLANES;

    const FENS: [&str; 4] = [
        START_FEN,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    ];

    #[test]
    fn legal_move_indices_are_padded_to_the_longest_row() {
        let moves: Vec<_> = FENS
            .iter()
            .map(|fen| State::from_fen(fen).generate_moves())
            .collect();
        let (indices, padding): (Tensor<NdArray, 2, Int>, Tensor<NdArray, 2>) =
            moves.legal_move_indices(&Default::default());

        let width = moves.iter().map(|moves| moves.len()).max().unwrap();
        assert_eq!(indices.dims(), [FENS.len(), width]);
        assert_eq!(padding.dims(), [FENS.len(), width]);

        let indices = indices.to_data().convert::<i32>().to_vec::<i32>().unwrap();
        let padding = padding.to_data().to_vec::<f32>().unwrap();
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
