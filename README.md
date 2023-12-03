# Hi-C

**TODO: Make sure not to include this in code we might provide to reviewers, as it breaks anonymity**

Implementation of Hi-C algorithm and baselines (https://arxiv.org/abs/2302.03438)

### Old Negotiation Text

A major conceptual issue in learning Stackelberg equilibria is determining which player should be the leader and which the follower.  In centralized training these roles can be assigned in advance, but it is unclear how two independent learners can "negotiate" these roles, given that both players may prefer to be the leader or the follower.

The basic idea is to use a higher-level, *symmetric* learning process to select roles.  We can formalize this by imagining the learners playing a repeated 2x2 game, which we will call the *negotiation meta-game*, in which each player commits to either leading or following for a sufficiently long inteval for the "inner" Stackelberg learning process to converge.  While there are multiple ways the inner process could be implemented (Hierarchical gradients, Hi-C, etc.), the assumption is that having a distinct leader and follower is always preferable.

As a simplified model, imagine that for each of the four possible joint strategies in the meta-game (lead-lead, lead-follow, follow-lead, and follow-follow), the average payoffs of the inner learning process will converge to unique payoff profiles (independent of initial conditions for now).  We will denote the corresponding profiles as $ll. lf. fl,$ and $ff$, all of which are 2D vectors.  Hierarchical learning, that is, searching for Stackelberg equilibria, only makes sense when we consider that

# Running Experiments

NOTE: SPSA is highly unstable without a variance-reducing baseline

Target 2-player environments:
- Coordination game
- Battle of the Sexes
- Cournot duopoly
- Iterated Prisoner's Dilemma (possibly, but much more expensive)

Optimization test problems:
- Quadratic function
- Gaussian function
