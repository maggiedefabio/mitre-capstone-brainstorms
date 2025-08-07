#Updating model to include Game Theory
from pyomo.environ import *
from pyomo.opt import SolverFactory
from math import radians, sin, cos, sqrt, atan2
import random

# === Geographic data and parameters (same as yours) ===

launch_sites = ['L1','L2','L3','L4','L5','L6','L7','L8','L9','L10']
targets = ['T1','T2','T3','T4','T5','T6']

locations = {
    'T1': (47.588611, -122.381111), 'T2': (47.568333, -122.310278),
    'T3': (47.534722, -122.302778), 'T4': (47.549444, -122.387222),
    'T5': (47.551111, -122.345278), 'T6': (47.520000, -122.321111),
    'L1': (47.590000, -122.363611), 'L2': (47.583889, -122.360000),
    'L3': (47.574722, -122.359167), 'L4': (47.574722, -122.359167),
    'L5': (47.560833, -122.345278), 'L6': (47.549444, -122.339722),
    'L7': (47.545000, -122.336944), 'L8': (47.540278, -122.331389),
    'L9': (47.536667, -122.326389), 'L10': (47.532500, -122.320000)
}

target_impact = {'T1': 100, 'T2': 80, 'T3': 1, 'T4': 60, 'T5': 40, 'T6': 50}
target_hits_required = {'T1': 30, 'T2': 20, 'T3': 10, 'T4': 10, 'T5': 15, 'T6': 15}

max_range_meters = 8000
max_total_drones = 100
max_drones_per_site = 34
fixed_launch_sites = 3
base_success_prob = 0.8
failure_rate_per_meter = 0.00002  # 2% per km
max_fortification_budget = 1.25  # Defender total budget for fortification

def haversine_meters(coord1, coord2):
    R = 6371000
    lat1, lon1 = map(radians, coord1)
    lat2, lon2 = map(radians, coord2)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

distance_data = {
    (site, tgt): round(haversine_meters(locations[site], locations[tgt]), 1)
    for site in launch_sites for tgt in targets
}

feasible_pairs = [(site, tgt) for site in launch_sites for tgt in targets if distance_data[(site, tgt)] <= max_range_meters]

# === Attacker model builder function ===
def build_attacker_model(fortification_levels, prev_drones=None, penalty_weight=0.05):
    model = ConcreteModel()

    model.SITES = Set(initialize=launch_sites)
    model.TARGETS = Set(initialize=targets)
    model.PAIRS = Set(within=model.SITES * model.TARGETS, initialize=feasible_pairs)

    model.distance = Param(model.PAIRS, initialize={k: distance_data[k] for k in feasible_pairs})
    model.impact = Param(model.TARGETS, initialize=target_impact)
    model.required_hits = Param(model.TARGETS, initialize=target_hits_required)
    model.fortification = Param(model.TARGETS, initialize=fortification_levels)

    # Update success probabilities using current fortifications
    success_prob_data = {}
    for (site, tgt) in feasible_pairs:
        dist_fail = failure_rate_per_meter * distance_data[(site, tgt)]
        val = base_success_prob * (1 - dist_fail) * (1 - fortification_levels[tgt])
        success_prob_data[(site, tgt)] = max(0, min(1, val))
    model.success_prob = Param(model.PAIRS, initialize=success_prob_data)

    # Decision variables
    model.use_site = Var(model.SITES, within=Binary)
    model.num_drones = Var(model.PAIRS, within=NonNegativeIntegers)
    model.expected_hits = Var(model.TARGETS, within=NonNegativeReals)
    model.capped_hits = Var(model.TARGETS, within=NonNegativeReals)

    # Constraints
    def per_site_drone_limit(m, s):
        return sum(m.num_drones[s, t] for t in m.TARGETS if (s, t) in m.PAIRS) <= max_drones_per_site * m.use_site[s]
    model.site_limit = Constraint(model.SITES, rule=per_site_drone_limit)

    def total_drone_limit(m):
        return sum(m.num_drones[s, t] for (s, t) in m.PAIRS) <= max_total_drones
    model.total_limit = Constraint(rule=total_drone_limit)

    def hit_calc(m, t):
        return m.expected_hits[t] == sum(
            m.num_drones[s, t] *
            base_success_prob *
            (1 - failure_rate_per_meter * m.distance[s, t]) *
            (1 - m.fortification[t])
            for s in m.SITES if (s, t) in m.PAIRS
        )
    model.hit_calc = Constraint(model.TARGETS, rule=hit_calc)

    def cap_success(m, t):
        return m.capped_hits[t] <= m.expected_hits[t]
    model.cap_success = Constraint(model.TARGETS, rule=cap_success)

    def cap_required(m, t):
        return m.capped_hits[t] <= m.required_hits[t]
    model.cap_required = Constraint(model.TARGETS, rule=cap_required)

    def fixed_sites(m):
        return sum(m.use_site[s] for s in m.SITES) == fixed_launch_sites
    model.fixed_sites = Constraint(rule=fixed_sites)

#inertia penalty
    if prev_drones:
        model.drone_diff = Var(model.PAIRS, within=NonNegativeReals)

        def abs_diff_constraints(m, s, t):
            prev_val = prev_drones.get((s, t), 0)
            return [
                m.drone_diff[s, t] >= m.num_drones[s, t] - prev_val,
                m.drone_diff[s, t] >= prev_val - m.num_drones[s, t]
            ]
        model.abs_diff_constraints = ConstraintList()
        for (s, t) in model.PAIRS:
            for constr in abs_diff_constraints(model, s, t):
                model.abs_diff_constraints.add(constr)

    # Objective
    def total_obj(m):
        reward = sum(m.impact[t] * m.capped_hits[t] / m.required_hits[t] for t in m.TARGETS)

        if prev_drones:
            penalty = sum(penalty_weight * m.drone_diff[s, t] for (s, t) in m.PAIRS)
            return reward - penalty
        else:
            return reward

    model.objective = Objective(rule=total_obj, sense=maximize)

    return model

# === Defender update function with stochasticity ===
def update_defender_fortification(prev_fortification, attacker_hits, max_step=0.01, max_budget=1.25):
    new_fort = {}
    # Step 1: Raw update with capped delta
    for t in targets:
        hit_frac = min(attacker_hits.get(t, 0) / target_hits_required[t], 1.0)
        base = prev_fortification.get(t, 0)

        delta = 0.03 * hit_frac
        delta = max(-max_step, min(max_step, delta))

        updated = max(0, min(1, base + delta))
        new_fort[t] = updated

    # Step 2: Enforce total budget by scaling if needed
    total_fort = sum(new_fort.values())
    if total_fort > max_budget:
        scale_factor = max_budget / total_fort
        for t in targets:
            new_fort[t] *= scale_factor

    return new_fort

# === Main iterative attacker-defender game ===

# Initialize fortifications — defender guesses zero protection initially
current_fortification = {t: 0 for t in targets}
convergence_tol = .01
max_rounds = 50
prev_fortification = None
prev_drones = {}  # No previous plan in round 0
round_num = 0
solver = SolverFactory('glpk')

while round_num < max_rounds:
    print(f"\n=== Round {round_num+1} ===")
    print(f"Defender Fortification (before attack): {current_fortification}")

    model = build_attacker_model(current_fortification, prev_drones=prev_drones)
    solver.solve(model, tee=False)

    # Extract current drone assignments for convergence check
    current_drones = {
        (s, t): model.num_drones[s, t].value or 0
        for (s, t) in feasible_pairs
    }

    attacker_hits = {}
    for t in targets:
        val = model.expected_hits[t].value
        attacker_hits[t] = val if val is not None else 0

    print("\nSelected Launch Sites:")
    for s in launch_sites:
        if model.use_site[s].value is not None and model.use_site[s].value >= 0.5:
            print(f"  {s}")

    print("\nDrones Assigned (site -> target):")
    for (s, t) in feasible_pairs:
        drones = model.num_drones[s, t].value
        if drones is not None and drones > 0:
            print(f"  {s} -> {t}: {int(round(drones))} drones")

    print("\nEstimated Successful Hits per Target:")
    for t in targets:
        hits = attacker_hits[t]
        req = target_hits_required[t]
        status = "Fully destroyed" if hits >= req else f"{(hits / req)*100:.1f}% destroyed"
        print(f"  {t}: {hits:.2f} hits ({status}, Required: {req})")

    total_expected_impact = sum(
        model.impact[tgt] * model.capped_hits[tgt].value / model.required_hits[tgt]
        for tgt in model.TARGETS if model.capped_hits[tgt].value is not None
    )
    print(f"\n Total Expected Impact: {total_expected_impact:.2f}")

    # Update fortifications
    new_fortification = update_defender_fortification(current_fortification, attacker_hits)

    # Check convergence
    if prev_fortification is not None:
        diff = sum((new_fortification[t] - prev_fortification[t])**2 for t in targets) ** 0.5
        if diff < convergence_tol:
            print("\nConverged!")
            break


    prev_fortification = current_fortification
    current_fortification = new_fortification
    prev_drones = current_drones.copy()
    round_num += 1

print("\n=== Final defender fortifications ===")
for t, fort in current_fortification.items():
    print(f"  {t}: {fort:.2f}")
