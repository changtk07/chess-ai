use crate::chess::{
    moves::{Move, MoveType},
    types::Direction,
    State,
};
use arrayvec::ArrayVec;
use burn::{
    tensor::{backend::Backend, TensorData},
    Tensor,
};

pub(crate) const N_MOVE_PLANES: usize = 73;
const N_POTENTIAL_MOVES: usize = 64 * N_MOVE_PLANES;
const MAX_RAY_MOVE_DIST: usize = 7;
const KNIGHT_MOVE_PLANE_START: usize = 56;
const KNIGHT_PROMO_CENTER_OFFSET: i8 = 64;
const BISHOP_PROMO_CENTER_OFFSET: i8 = 67;
const ROOK_PROMO_CENTER_OFFSET: i8 = 70;

pub(crate) fn policy_index(mv: &Move) -> usize {
    let (from, to, mv_type) = mv.unpack();
    let base = from.0 as usize * N_MOVE_PLANES;
    let (rank_diff, file_diff) = (
        to.rank() as i8 - from.rank() as i8,
        to.file() as i8 - from.file() as i8,
    );

    match mv_type {
        MoveType::PromotionKnight => {
            return base + (KNIGHT_PROMO_CENTER_OFFSET + file_diff + 1) as usize
        }
        MoveType::PromotionBishop => {
            return base + (BISHOP_PROMO_CENTER_OFFSET + file_diff + 1) as usize
        }
        MoveType::PromotionRook => {
            return base + (ROOK_PROMO_CENTER_OFFSET + file_diff + 1) as usize
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

pub(crate) fn legal_move_mask<B: Backend>(moves: &[Move], device: &B::Device) -> Tensor<B, 1> {
    let data = moves
        .iter()
        .fold(vec![-1e9; N_POTENTIAL_MOVES], |mut acc, mv| {
            acc[policy_index(mv)] = 0.0;
            acc
        });
    Tensor::from_data(TensorData::new(data, [N_POTENTIAL_MOVES]), device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::State;

    #[test]
    fn test_move_indices_are_unique_and_in_bounds() {
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        ];

        for fen in fens {
            let state = State::from_fen(fen);
            let moves = state.generate_moves();
            let mut indices = Vec::new();
            for mv in moves {
                let idx = policy_index(&mv);
                assert!(
                    idx < 64 * N_MOVE_PLANES,
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
