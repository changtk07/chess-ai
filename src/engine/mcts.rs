use crate::{
    chess::{moves::Move, State},
    engine::{
        model::{TransformerModel, TransformerModelConfig},
        policy::LegalMoveMask,
    },
};
use burn::tensor::backend::Backend;
use std::cmp::Ordering::Equal;

struct Node {
    mv: Option<Move>,
    n_visits: usize,
    value_sum: f32,
    prior: f32,
    children: Vec<usize>,
}

impl Node {
    fn new(mv: Option<Move>, prior: f32) -> Self {
        Self {
            mv,
            n_visits: 0,
            value_sum: 0.0,
            prior,
            children: vec![],
        }
    }

    fn q_val(&self) -> f32 {
        if self.n_visits == 0 {
            0.0
        } else {
            self.value_sum / self.n_visits as f32
        }
    }

    fn u_val(&self, c_puct: f32, sqrt_total_visits: f32) -> f32 {
        c_puct * self.prior * (sqrt_total_visits / (1.0 + self.n_visits as f32))
    }

    fn score(&self, c_puct: f32, sqrt_total_visits: f32) -> f32 {
        self.q_val() + self.u_val(c_puct, sqrt_total_visits)
    }

    fn record_visit(&mut self, val: f32) {
        self.n_visits += 1;
        self.value_sum += val;
    }
}

struct Mcts<B: Backend> {
    nodes: Vec<Node>,
    c_puct: f32,
    n_sims: usize,
    device: B::Device,
    model: TransformerModel<B>,
}

impl<B: Backend> Mcts<B> {
    pub fn new(c_puct: f32, n_sims: usize, device: &B::Device) -> Self {
        let mut nodes = Vec::with_capacity(n_sims);
        nodes.push(Node::new(None, 1.0));

        Self {
            nodes,
            c_puct,
            n_sims,
            device: device.clone(),
            model: TransformerModelConfig::new().init(device),
        }
    }

    fn explore(&mut self, idx: usize, mut state: State) -> f32 {
        if self.nodes[idx].n_visits > 0 {
            if self.nodes[idx].children.is_empty() {
                return self.nodes[idx].q_val();
            }

            let sqrt_total_visits = self.nodes[idx]
                .children
                .iter()
                .copied()
                .map(|i| self.nodes[i].n_visits as f32)
                .sum::<f32>()
                .sqrt();

            let chosen = self.nodes[idx]
                .children
                .iter()
                .copied()
                .map(|i| (i, self.nodes[i].score(self.c_puct, sqrt_total_visits)))
                .max_by(|(_, score_a), (_, score_b)| score_a.partial_cmp(score_b).unwrap_or(Equal))
                .map(|(i, _)| i)
                .unwrap();

            if let Some(mv) = self.nodes[chosen].mv {
                state.make_move(mv);
            }

            let value = -self.explore(chosen, state);
            self.nodes[idx].record_visit(value);

            return value;
        }

        let moves = state.generate_moves();
        let legal_move_mask = moves.legal_move_mask(&self.device).unsqueeze();
        let (policy, value) = self.model.forward(&[&state], legal_move_mask, &self.device);

        self.nodes[idx].children = (self.nodes.len()..self.nodes.len() + moves.len()).collect();

        moves.iter().for_each(|mv| {
            let policy_index = mv.policy_index();
            self.nodes.push(Node::new(Some(*mv), policy[policy_index]));
        });

        self.nodes[idx].record_visit(value[0]);

        value[0]
    }
}
