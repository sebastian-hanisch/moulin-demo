"""Konstanten der Moulin-Kostenteilungs-Demo: Vehikel B "Spediteurs-Kooperation" (wie shapley-demo/kern-demo) mit privater Zahlungsbereitschaft, Regler, Experimente."""

EPS = 1e-9
TOL = 1e-7
SEED_MAX = 999999

AREA = 100.0
CLUSTER_SIGMA = 8.0
N_CLUSTERS = 2
LAYOUTS = ("uniform", "clustered")
LAYOUT_LABELS = {"uniform": "Gleichmäßig verteilt", "clustered": "Zwei Ballungszentren"}

N_MIN, N_MAX, DEFAULT_N, N_STEP = 3, 9, 7, 1
DEFAULT_SEED = 35

# Zahlungsbereitschaft: v_i = theta_i * (Kosten der Alleinfahrt von i), theta_i ~ U(0, theta_max)
THETA_MIN, THETA_MAX, DEFAULT_THETA, THETA_STEP = 0.4, 3.0, 1.5, 0.1
# Faktor auf die Folk-Anteile des Spannbaums (2 = Tour-Verdopplung, garantiert kostendeckend)
FACTOR_MIN, FACTOR_MAX, DEFAULT_FACTOR, FACTOR_STEP = 1.0, 2.0, 2.0, 0.1

MECHANISMS = ("moulin_folk", "moulin_shapley", "vcg")
MECHANISM_LABELS = {"moulin_folk": "Moulin mit Spannbaum-Anteilen (Folk)", "moulin_shapley": "Moulin mit Shapley-Anteilen", "vcg": "Grenzkosten-Mechanismus (VCG)"}

# --- Experimente (feste Seeds) ---------------------------------------------------------------------------------------------------------------

PREREQ_NS = (5, 6, 7, 8)
PREREQ_SEEDS = tuple(range(950000, 950030))
COMPARE_N = 6
COMPARE_SEEDS = tuple(range(960000, 960030))
COMPARE_PROFILES = 4
FACTOR_N = 7
FACTOR_SEEDS = tuple(range(970000, 970020))
FACTOR_LEVELS = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0)
