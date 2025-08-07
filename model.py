from pyomo.environ import *
from pyomo.opt import SolverFactory
from math import radians, sin, cos, sqrt, atan2

# === GEOGRAPHIC DATA ===

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

# === TARGET CHARACTERISTICS ===

target_impact = {'T1': 100, 'T2': 80, 'T3': 1, 'T4': 60, 'T5': 40, 'T6': 50}
target_hits_required = {'T1': 20, 'T2': 13, 'T3': 5, 'T4': 9, 'T5': 1, 'T6': 8}
target_fortification = {'T1': 0.3, 'T2': 0.1, 'T3': 0.1, 'T4': 0.25, 'T5': 0.3, 'T6': 0.2}

# === SYSTEM LIMITS AND SETTINGS ===

max_range_meters = 8000
max_total_drones = 100
max_drones_per_site = 34
fixed_launch_sites = 3
base_success_prob = 0.8
failure_rate_per_meter = 0.00002  # 2% per km

# === DISTANCE FUNCTION ===

def haversine_meters(coord1, coord2):
    R = 6371000
    lat1, lon1 = map(radians, coord1)
    lat2, lon2 = map(radians, coord2)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# Precompute distances
distance_data = {
    (site, tgt): round(haversine_meters(locations[site], locations[tgt]), 1)
    for site in launch_sites for tgt in targets
}

# === MODEL DEFINITION ===

model = ConcreteModel()

feasible_pairs = [(site, tgt) for site in launch_sites for tgt in targets if distance_data[(site, tgt)] <= max_range_meters]

# Sets
model.SITES = Set(initialize=launch_sites)
model.TARGETS = Set(initialize=targets)
model.PAIRS = Set(within=model.SITES * model.TARGETS, initialize=feasible_pairs)

# Parameters
model.distance = Param(model.PAIRS, initialize={k: distance_data[k] for k in feasible_pairs})
model.impact = Param(model.TARGETS, initialize=target_impact)
model.required_hits = Param(model.TARGETS, initialize=target_hits_required)
model.fortification = Param(model.TARGETS, initialize=target_fortification)

# Correct success probability initialization with dictionary
success_prob_data = {}
for (site, tgt) in feasible_pairs:
    dist_fail = failure_rate_per_meter * distance_data[(site, tgt)]
    val = base_success_prob * (1 - dist_fail) * (1 - target_fortification[tgt])
    success_prob_data[(site, tgt)] = max(0, min(1, val))

model.success_prob = Param(model.PAIRS, initialize=success_prob_data)

# === DECISION VARIABLES ===

model.use_site = Var(model.SITES, within=Binary)
model.num_drones = Var(model.PAIRS, within=NonNegativeIntegers)
model.expected_hits = Var(model.TARGETS, within=NonNegativeReals)
model.capped_hits = Var(model.TARGETS, within=NonNegativeReals)

# === CONSTRAINTS ===

def per_site_drone_limit(model, site):
    return sum(model.num_drones[site, tgt] for tgt in model.TARGETS if (site, tgt) in model.PAIRS) <= max_drones_per_site * model.use_site[site]
model.drone_limit_per_site = Constraint(model.SITES, rule=per_site_drone_limit)

model.total_drone_limit = Constraint(expr=sum(model.num_drones[site, tgt] for (site, tgt) in model.PAIRS) <= max_total_drones)

def successful_hits(model, tgt):
    return model.expected_hits[tgt] == sum(
        model.num_drones[site, tgt] * model.success_prob[site, tgt]
        for (site, tgt2) in model.PAIRS if tgt2 == tgt
    )
model.hit_calculation = Constraint(model.TARGETS, rule=successful_hits)

def cap_to_successful(model, tgt):
    return model.capped_hits[tgt] <= model.expected_hits[tgt]
model.cap_by_success = Constraint(model.TARGETS, rule=cap_to_successful)

def cap_to_required(model, tgt):
    return model.capped_hits[tgt] <= model.required_hits[tgt]
model.cap_by_requirement = Constraint(model.TARGETS, rule=cap_to_required)

def enforce_fixed_sites(model):
    return sum(model.use_site[site] for site in model.SITES) == fixed_launch_sites
model.fixed_sites = Constraint(rule=enforce_fixed_sites)

# === OBJECTIVE FUNCTION ===

def total_objective(model):
    return sum(
        model.impact[tgt] * model.capped_hits[tgt] / model.required_hits[tgt]
        for tgt in model.TARGETS
    )
model.objective = Objective(rule=total_objective, sense=maximize)

# === SOLVING ===

solver = SolverFactory('glpk')
results = solver.solve(model, tee=False)

# === RESULTS ===

print("\n Selected Launch Sites:")
for site in model.SITES:
    if model.use_site[site].value is not None and model.use_site[site].value >= 0.5:
        print(f"  {site}")

print("\n Drones Assigned per Launch-Target Pair:")
for (site, tgt) in model.PAIRS:
    drones = model.num_drones[site, tgt].value
    if drones is not None and drones > 0:
        print(f"  {site} → {tgt}: {int(round(drones))} drones")

print("\n Total Drones per Target:")
for tgt in model.TARGETS:
    total = sum(model.num_drones[site, tgt].value for site in model.SITES if (site, tgt) in model.PAIRS and model.num_drones[site, tgt].value is not None)
    print(f"  {tgt}: {int(round(total))}")

print("\n Estimated Successful Hits per Target:")
for tgt in model.TARGETS:
    hits = model.expected_hits[tgt].value
    required = model.required_hits[tgt]
    if hits is None:
        hits = 0
    if hits >= required:
        print(f"  {tgt}: Fully destroyed (Hits: {hits:.2f}, Required: {required})")
    else:
        percent = (hits / required) * 100 if required > 0 else 0
        print(f"  {tgt}: {hits:.2f} hits ({percent:.1f}% destroyed, Required: {required})")

# === IMPACT REPORT ===
total_expected_impact = sum(
    model.impact[tgt] * model.capped_hits[tgt].value / model.required_hits[tgt]
    for tgt in model.TARGETS if model.capped_hits[tgt].value is not None
)

print(f"\n Total Expected Impact: {total_expected_impact:.2f}")
