# Intertemporal EPEC Model — Complete Mathematical Reference

This document is the authoritative equation reference for the model implemented in
`model/model_main.py`. It is the working file for writing the paper's Method section.

---

## 1. Sets and Indices

| Symbol | Description |
|--------|-------------|
| $\mathcal{R}$ | Set of all regions; every region is a strategic player |
| $\mathcal{T} = \{2025, 2030, 2035, 2040, 2045\}$ | Set of time periods |
| $t_0 = \min\mathcal{T}$, $t^* = \max\mathcal{T}$ | First and last period |
| $e \in \mathcal{R}$ | Origin (exporter) index |
| $i \in \mathcal{R}$ | Destination (importer / market) index |
| $r \in \mathcal{R}$ | Current player index (whose MPEC is written) |
| $t \in \mathcal{T}$ | Time-period index |

---

## 2. Parameters

### 2.1 Demand
| Symbol | Unit | Description |
|--------|------|-------------|
| $a_{i,t}$ | USD/kW | Intercept of inverse demand function (true, exogenous) |
| $b_{i,t}$ | USD/(kW·GW) | Slope of inverse demand function |
| $D^{\max}_{i,t}$ | GW | Hard cap on market demand |

### 2.2 Cost and Capacity
| Symbol | Unit | Description |
|--------|------|-------------|
| $K^0_i$ | GW | Initial installed capacity at $t_0$ |
| $c^{man}_{i,t}$ | USD/kW | Manufacturing / production cost (exogenous LBD schedule) |
| $c^{ship}_{e,i}$ | USD/kW | Transport (shipping) cost on route $(e,i)$ |
| $\bar{p}_{e,i}$ | USD/kW | Upper bound on offer price on route $(e,i)$ |
| $g^{exp}_i$ | GW/yr or 1/yr | Max capacity expansion rate (absolute or proportional) |
| $g^{dec}_i$ | 1/yr | Max capacity decommissioning rate (proportional) |
| $f^{hold}_i$ | mUSD/(GW·yr) | Annual O&M holding cost per GW of capacity |
| $c^{inv}_i$ | mUSD/(GW·yr) | Annualised capital cost per GW/yr of expansion |
| $y_t$ | years | Duration of period $t$ (interval length to next period) |
| $\beta_t$ | — | NPV discount factor: $\beta_t = 1/(1+r_d)^{t - t_0}$ |

### 2.3 Regularisation and Penalty Weights
| Symbol | Description |
|--------|-------------|
| $\epsilon_x > 0$ | Quadratic regularisation on trade flows (solver stability + strict convexity) |
| $\epsilon_{comp} \geq 0$ | Complementarity tolerance (0 = exact, small value = relaxed) |
| $c^{quad}_q, c^{quad}_p, c^{quad}_a$ | Economic quadratic penalty weights (convex cost / disutility) |
| $c^{pen}_q, c^{pen}_p, c^{pen}_a, c^{pen}_{dK}$ | Proximal (Gauss–Seidel) penalty weights for algorithmic stabilisation |

---

## 3. Variables

### 3.1 LLP Market Variables (solved at equilibrium for given upper-level strategies)
| Symbol | Domain | Type | Description |
|--------|--------|------|-------------|
| $x_{e,i,t}$ | $e,i \in \mathcal{R},\, t \in \mathcal{T}$ | $\geq 0$ | Trade flow from exporter $e$ to market $i$ |
| $x^{dem}_{i,t}$ | $i \in \mathcal{R},\, t \in \mathcal{T}$ | $[0, D^{\max}_{i,t}]$ | Demand covered in market $i$ |

### 3.2 LLP Dual Variables (KKT multipliers of the LLP)
| Symbol | Domain | Sign | Associated constraint |
|--------|--------|------|-----------------------|
| $\lambda_{i,t}$ | $i \in \mathcal{R},\, t \in \mathcal{T}$ | free ($\geq 0$) | Market balance (= market-clearing price) |
| $\mu^{offer}_{e,t}$ | $e \in \mathcal{R},\, t \in \mathcal{T}$ | $\geq 0$ | Offer-capacity constraint |
| $\gamma_{e,i,t}$ | $e,i \in \mathcal{R},\, t \in \mathcal{T}$ | $\geq 0$ | Non-negativity of $x_{e,i,t}$ |
| $\beta^{dem}_{i,t}$ | $i \in \mathcal{R},\, t \in \mathcal{T}$ | $\geq 0$ | Maximum demand cap $D^{\max}_{i,t}$ |
| $\psi_{i,t}$ | $i \in \mathcal{R},\, t \in \mathcal{T}$ | $\geq 0$ | Non-negativity of $x^{dem}_{i,t}$ |

### 3.3 ULP Strategic Variables (decision variables of player $r$)

There are two strategic decisions: the **offer price** on each export route and the **net capacity change** between periods.

| Symbol | Domain | Type | Description |
|--------|--------|------|-------------|
| $K_{r,t}$ | $r \in \mathcal{R},\, t \in \mathcal{T}$ | $\geq 0$ | Installed capacity (state variable, not directly controlled) |
| $\Delta K_{r,t}$ | $r \in \mathcal{R},\, t \in \mathcal{T} \setminus \{t^*\}$ | free | Net capacity change rate: $\Delta K_{r,t} = I^+_{r,t} - D^-_{r,t}$; positive = expansion, negative = decommissioning |
| $p_{r,i,t}$ | $r,i \in \mathcal{R},\, t \in \mathcal{T}$ | $[0, \bar{p}_{r,i}]$ | Offer price on export route $(r,i)$; domestic route fixed: $p_{r,r,t} = c^{man}_{r,t}$ |

> **Note on demand bids**: The demand-bid intercept is fixed to the true demand intercept, $a^{bid}_{r,t} \equiv a_{r,t}$, in all runs (`fix_a_bid_to_true_dem = True`). There is no strategic demand suppression; demand is a linear inverse function with exogenous parameters $a_{i,t}$ and $b_{i,t}$.

> **Note on $Q_{r,t}$**: The offered quantity is always set equal to the installed capacity, $Q_{r,t} = K_{r,t}$ (no strategic withholding). It is therefore not a decision variable and is substituted directly wherever it appears.

---

## 4. Lower-Level Problem (LLP)

The LLP represents uniform-price market clearing. Given the upper-level strategies
$p_{e,i,t}$ of all players and the installed capacities $K_{e,t}$ (which
fully determine offered quantities since $Q_{e,t} = K_{e,t}$), a (conceptual) market
operator minimises total net cost — equivalently maximises aggregate surplus.
Demand is a linear inverse function with exogenous parameters; since $a^{bid}_{i,t} = a_{i,t}$
identically, the true and bid demand intercepts coincide throughout.

### 4.1 LLP Objective

$$
\min_{x,\, x^{dem}} \quad z^{LLP} = \underbrace{\sum_{t \in \mathcal{T}}\sum_{e \in \mathcal{R}}\sum_{i \in \mathcal{R}} \left(p_{e,i,t} + c^{ship}_{e,i} + \frac{\epsilon_x}{2}\, x_{e,i,t}\right) x_{e,i,t}}_{\text{supply cost (incl. regularisation)}} - \underbrace{\sum_{t \in \mathcal{T}}\sum_{i \in \mathcal{R}} \left(a_{i,t}\, x^{dem}_{i,t} - \frac{b_{i,t}}{2}\,(x^{dem}_{i,t})^2\right)}_{\text{consumer gross surplus}}
$$

### 4.2 LLP Primal Constraints

**Market balance** (`eq_bal`):
$$
\sum_{e \in \mathcal{R}} x_{e,i,t} - x^{dem}_{i,t} = 0 \qquad \forall\, i \in \mathcal{R},\, t \in \mathcal{T} \quad (\lambda_{i,t} \text{ free})
$$

**Offer-capacity constraint** (`eq_cap`):
$$
K_{e,t} - \sum_{i \in \mathcal{R}} x_{e,i,t} \geq 0 \qquad \forall\, e \in \mathcal{R},\, t \in \mathcal{T} \quad (\mu^{offer}_{e,t} \geq 0)
$$

**Maximum demand** (variable upper bound):
$$
x^{dem}_{i,t} \leq D^{\max}_{i,t} \qquad \forall\, i \in \mathcal{R},\, t \in \mathcal{T} \quad (\beta^{dem}_{i,t} \geq 0)
$$

**Non-negativity**:
$$
x_{e,i,t} \geq 0 \quad (\gamma_{e,i,t} \geq 0), \qquad x^{dem}_{i,t} \geq 0 \quad (\psi_{i,t} \geq 0)
$$

---

## 5. KKT Conditions of the LLP

These are the optimality conditions of the LLP with respect to $x_{e,i,t}$ and
$x^{dem}_{i,t}$. They are substituted as constraints into each player's ULP to form
the MPEC.

### 5.1 Stationarity

**w.r.t. $x_{e,i,t}$** (`eq_stat_x`):
$$
\boxed{p_{e,i,t} + c^{ship}_{e,i} + \epsilon_x\, x_{e,i,t} - \lambda_{i,t} + \mu^{offer}_{e,t} - \gamma_{e,i,t} = 0} \qquad \forall\, e, i \in \mathcal{R},\, t \in \mathcal{T}
$$

*Interpretation*: the marginal cost of supply (offer price + shipping + regularisation)
equals the market price $\lambda_{i,t}$, net of the offer-scarcity rent $\mu^{offer}$
and the non-negativity shadow price $\gamma$.

**w.r.t. $x^{dem}_{i,t}$** (`eq_stat_dem`):
$$
\boxed{-\left(a_{i,t} - b_{i,t}\, x^{dem}_{i,t}\right) + \lambda_{i,t} + \beta^{dem}_{i,t} - \psi_{i,t} = 0} \qquad \forall\, i \in \mathcal{R},\, t \in \mathcal{T}
$$

*Interpretation*: the marginal willingness-to-pay $(a_{i,t} - b_{i,t}\,x^{dem}_{i,t})$ from
the linear inverse demand function equals the market price $\lambda_{i,t}$, adjusted for
binding demand cap ($\beta^{dem}$) and non-negativity ($\psi$).

### 5.2 Complementarity Conditions

**Offer-capacity** (`eq_comp_mu_offer`):
$$
\boxed{0 \leq \mu^{offer}_{e,t} \;\perp\; K_{e,t} - \sum_{i \in \mathcal{R}} x_{e,i,t} \geq 0} \qquad \forall\, e \in \mathcal{R},\, t \in \mathcal{T}
$$

**Trade-flow non-negativity** (`eq_comp_gamma`):
$$
\boxed{0 \leq \gamma_{e,i,t} \;\perp\; x_{e,i,t} \geq 0} \qquad \forall\, e, i \in \mathcal{R},\, t \in \mathcal{T}
$$

**Maximum demand cap** (`eq_comp_beta_dem`):
$$
\boxed{0 \leq \beta^{dem}_{i,t} \;\perp\; D^{\max}_{i,t} - x^{dem}_{i,t} \geq 0} \qquad \forall\, i \in \mathcal{R},\, t \in \mathcal{T}
$$

**Demand non-negativity** (`eq_comp_psi_dem`):
$$
\boxed{0 \leq \psi_{i,t} \;\perp\; x^{dem}_{i,t} \geq 0} \qquad \forall\, i \in \mathcal{R},\, t \in \mathcal{T}
$$

> **Relaxed form**: when $\epsilon_{comp} > 0$ all complementarity products are
> replaced by $\leq \epsilon_{comp}$ (instead of $= 0$) to improve solver
> convergence.

---

## 6. Upper-Level Problem (ULP) per Player

Each strategic player $r \in \mathcal{R}$ maximises the present-value sum of welfare
over $\mathcal{T}$, treating all LLP equilibrium quantities as endogenous outcomes of
their strategic decisions.

### 6.1 ULP Objective

$$
\max_{K_r,\, \Delta K_r,\, p_r} \quad
\Pi_r = \sum_{t \in \mathcal{T}} \beta_t\, y_t \Bigg[
\underbrace{a_{r,t}\, x^{dem}_{r,t} - \frac{b_{r,t}}{2}(x^{dem}_{r,t})^2 - \lambda_{r,t}\, x^{dem}_{r,t}}_{\text{domestic consumer surplus (CS)}}
+ \underbrace{\sum_{i \in \mathcal{R}} \left(\lambda_{i,t} - \mu^{offer}_{r,t} - c^{man}_{r,t} - c^{ship}_{r,i}\right) x_{r,i,t}}_{\text{producer surplus (PS)}}
$$
$$
\qquad - \underbrace{f^{hold}_r K_{r,t} + c^{inv}_r (\Delta K_{r,t})^+}_{\text{capacity holding + investment cost}}
- \underbrace{\frac{c^{quad}_p}{2}\sum_{i \in \mathcal{R}} (p_{r,i,t} - c^{man}_{r,t})^2}_{\text{economic quadratic penalty on price}}
$$
$$
\qquad - \underbrace{\frac{c^{pen}_p}{2}\sum_{i \in \mathcal{R}}(p_{r,i,t} - \hat{p}_{r,i,t})^2 - \frac{c^{pen}_{dK}}{2}(\Delta K_{r,t} - \widehat{\Delta K}_{r,t})^2}_{\text{proximal (Gauss–Seidel) penalties around last iterate } \hat{\cdot}}
\Bigg]
$$

**Notation:** $(\cdot)^+ = \max(\cdot,\, 0)$ and $(\cdot)^- = \max(-\cdot,\, 0)$, so $\Delta K = (\Delta K)^+ - (\Delta K)^-$ with $(\Delta K)^+$ the expansion rate and $(\Delta K)^-$ the decommissioning rate.

**Notes:**
- The CS term uses the true demand intercept $a_{r,t}$ directly. Since $a^{bid}_{r,t} = a_{r,t}$ identically (no strategic demand suppression), the LLP objective and the welfare calculation both use the same exogenous parameter.
- The PS term uses $\lambda_{i,t} - \mu^{offer}_{r,t}$ rather than the offer price directly. By the stationarity condition (Section 5.1), the effective margin is $\lambda_{i,t} - \mu^{offer}_{r,t} - c^{man}_{r,t} - c^{ship}_{r,i}$ (scarcity-rent-adjusted profit).
- Since $Q_{r,t} = K_{r,t}$ identically, the withholding penalty and its proximal counterpart are dropped.
- Policy incentives and the $a^{bid}$ penalty are implemented in the code but inactive in all baseline runs.
- The proximal terms (with hat-variables $\hat{\cdot}$ set by the diagonalisation algorithm) stabilise the Gauss–Seidel iteration and vanish at convergence.

### 6.2 ULP Constraints

**Initial capacity** (`eq_kcap_init`):
$$
K_{r,t_0} = K^0_r \qquad \forall\, r \in \mathcal{R}
$$

**Capacity transition** (`eq_kcap_trans`, for each consecutive pair $(t, t') \in \mathcal{T}$):
$$
K_{r,t'} = K_{r,t} + y_t\, \Delta K_{r,t} \qquad \forall\, r \in \mathcal{R},\; (t,t') \in \mathcal{T}^2,\; t' = \text{next}(t)
$$

**Net capacity change bounds** (`eq_icap_ub`, `eq_dcap_ub`):
$$
-\, g^{dec}_r \cdot K_{r,t} \;\leq\; \Delta K_{r,t} \;\leq\; g^{exp}_r \qquad \forall\, r \in \mathcal{R},\; t \in \mathcal{T} \setminus \{t^*\}
$$
The lower bound prevents decommissioning faster than a fraction $g^{dec}_r$ of current capacity; the upper bound caps absolute expansion at $g^{exp}_r$ GW/yr (or proportional if $g^{exp}_r \cdot K_{r,t}$ is used).

**Offer price bounds** (`eq_p_offer_bounds`):
$$
0 \;\leq\; p_{r,i,t} \;\leq\; \bar{p}_{r,i} \qquad \forall\, r,\, i \in \mathcal{R},\; t \in \mathcal{T}
$$

**Domestic self-offer** (`eq_self_offer`):
$$
p_{r,r,t} = c^{man}_{r,t} \qquad \forall\, r \in \mathcal{R},\; t \in \mathcal{T}
$$

---

## 7. Full MPEC per Player

Substituting the LLP KKT conditions (Section 5) as constraints into the ULP (Section 6)
yields the single-level MPEC for player $r$. Written compactly with $\mathbf{x}_r$
denoting all decision and dual variables of player $r$:

$$
\boxed{
\begin{aligned}
\underset{\mathbf{x}_r}{\mathrm{maximize}} \quad & \Pi_r \quad \text{(Eq. ULP Objective, Sec. 6.1)}\\[6pt]
\text{subject to} \quad
& \text{[LLP balance]} & \sum_{e} x_{e,i,t} - x^{dem}_{i,t} &= 0 & \forall\, i,t \\
& \text{[LLP offer cap]} & K_{e,t} - \sum_i x_{e,i,t} &\geq 0 & \forall\, e,t \\
& \text{[KKT stat-}x\text{]} & p_{e,i,t} + c^{ship}_{e,i} + \epsilon_x x_{e,i,t} - \lambda_{i,t} + \mu^{offer}_{e,t} - \gamma_{e,i,t} &= 0 & \forall\, e,i,t \\
& \text{[KKT stat-dem]} & -(a_{i,t} - b_{i,t} x^{dem}_{i,t}) + \lambda_{i,t} + \beta^{dem}_{i,t} - \psi_{i,t} &= 0 & \forall\, i,t \\
& \text{[KKT comp-}\mu\text{]} & 0 \leq \mu^{offer}_{e,t} \;\perp\; K_{e,t} - \textstyle\sum_i x_{e,i,t} &\geq 0 & \forall\, e,t \\
& \text{[KKT comp-}\gamma\text{]} & 0 \leq \gamma_{e,i,t} \;\perp\; x_{e,i,t} &\geq 0 & \forall\, e,i,t \\
& \text{[KKT comp-}\beta^{dem}\text{]} & 0 \leq \beta^{dem}_{i,t} \;\perp\; D^{\max}_{i,t} - x^{dem}_{i,t} &\geq 0 & \forall\, i,t \\
& \text{[KKT comp-}\psi\text{]} & 0 \leq \psi_{i,t} \;\perp\; x^{dem}_{i,t} &\geq 0 & \forall\, i,t \\
& \text{[Cap init]} & K_{r,t_0} &= K^0_r \\
& \text{[Cap trans]} & K_{r,t'} = K_{r,t} + y_t\, \Delta K_{r,t} & & \forall\, (t,t') \\
& \text{[} \Delta K \text{ bounds]} & -g^{dec}_r K_{r,t} \leq \Delta K_{r,t} &\leq g^{exp}_r & \forall\, t \\
& \text{[Price bounds]} & 0 \leq p_{r,i,t} &\leq \bar{p}_{r,i} & \forall\, i,t \\
& \text{[Self-offer]} & p_{r,r,t} &= c^{man}_{r,t} & \forall\, t \\
& \text{[Var bounds]} & K, x, x^{dem}, \mu^{offer}, \gamma, \beta^{dem}, \psi &\geq 0
\end{aligned}
}
$$

**Variable dimensions:**
The MPEC for player $r$ has the following strategic + dual variables free:
$K_r, \Delta K_r, p_{r,i}$ (ULP) and
$x, x^{dem}, \lambda, \mu^{offer}, \gamma, \beta^{dem}, \psi$ (LLP, shared across players but
solved simultaneously with $r$'s MPEC when other players' strategies are fixed).

**Note on complementarity:** The bilinear complementarity conditions are handled
directly as NLP constraints (GAMS/IPOPT). No Big-M linearisation is applied in the
current implementation; the model is solved as an NLP (not MILP).

---

## 8. EPEC Definition and Nash Equilibrium

The full EPEC is the simultaneous system of $|\mathcal{R}|$ MPECs:

$$
\text{Find } (\mathbf{x}^*_r)_{r \in \mathcal{R}} \text{ such that } \mathbf{x}^*_r \text{ solves MPEC}_r \text{ given } \mathbf{x}^*_e,\; e \neq r, \quad \forall\, r \in \mathcal{R}.
$$

A solution is a **Nash equilibrium**: no player can unilaterally improve their
objective. Existence and uniqueness are not guaranteed in general for EPECs.

---

## 9. Solution Algorithm (Gauss–Seidel Diagonalisation)

The EPEC is solved by a **diagonalisation** (best-response iteration):

```
Initialise strategies (Q̂, p̂, â_bid, dK̂_net) for all r ∈ R
Repeat:
  For each player r ∈ P:
    1. Fix all other players' strategies at current iterate (apply_player_fixings)
    2. Update proximal reference parameters (p̂_last, Q̂_last, ...) for penalty terms
    3. Solve MPEC_r (NLP via IPOPT/GAMS)
    4. Apply damping:  θ_r^(k) ← α · θ̃_r + (1-α) · θ_r^(k-1)
    5. Extract new iterate via extract_state
Until max_i ‖θ_r^(k) - θ_r^(k-1)‖ ≤ ε_conv
```

### 9.1 Damping Strategy
The damping parameter $\alpha \in (0, 1]$ controls the convex combination between
the newly solved strategy $\tilde{\theta}_r$ and the previous iterate $\hat{\theta}_r$.
Lower $\alpha$ improves stability at the cost of convergence speed.

### 9.2 Penalty Variables (Proximal Terms)
The proximal penalty terms in the ULP objective (with weights $c^{pen}$) penalise
deviation from the previous iterate. They serve as a proximal-point regularisation,
ensuring the per-player NLP is better conditioned and the diagonalisation sequence
contracts. The weights can be annealed across outer iterations.

---

## 10. Notation Summary (cross-reference to LaTeX)

| Code variable | Math symbol | Description |
|---------------|-------------|-------------|
| `x[e,i,t]` | $x_{e,i,t}$ | Trade flow |
| `x_dem[i,t]` | $x^{dem}_{i,t}$ | Covered demand |
| `lam[i,t]` | $\lambda_{i,t}$ | Market price (dual of balance) |
| `mu_offer[e,t]` | $\mu^{offer}_{e,t}$ | Dual of offer-capacity constraint |
| `gamma[e,i,t]` | $\gamma_{e,i,t}$ | Dual of trade-flow non-negativity |
| `beta_dem[i,t]` | $\beta^{dem}_{i,t}$ | Dual of max-demand cap |
| `psi_dem[i,t]` | $\psi_{i,t}$ | Dual of demand non-negativity |
| `Kcap[r,t]` | $K_{r,t}$ | Installed capacity |
| `Icap_pos[r,t]` | $(\Delta K_{r,t})^+$ | Expansion rate (positive part of net capacity change) |
| `Dcap_neg[r,t]` | $(\Delta K_{r,t})^-$ | Decommissioning rate (negative part of net capacity change) |
| — | $\Delta K_{r,t}$ | Net capacity change: $\Delta K = (\Delta K)^+ - (\Delta K)^-$; $Q_{r,t} = K_{r,t}$ always |
| `p_offer[r,i,t]` | $p_{r,i,t}$ | Offer price ($\in [0, \bar{p}_{r,i}]$) |
| `a_dem_t[r,t]` | $a_{r,t}$ | Demand intercept (exogenous; $a^{bid} \equiv a$ always) |
| `c_man_t[r,t]` | $c^{man}_{r,t}$ | Manufacturing cost (LBD schedule) |
| `beta_t[t]` | $\beta_t$ | NPV discount factor |
| `ytn[t]` | $y_t$ | Period length (years) |
