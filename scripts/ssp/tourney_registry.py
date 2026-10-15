"""Writes configs/tourney_registry.json: hypotheses, predictions, gates and the fixed discovery/holdout split.

Run once, commit the JSON BEFORE any tournament run. The split uses only log names and the eligibility list of the
earlier tracking study (n = 100 had enough active people), never any result of the rules being tested.
"""
import json
import zlib

AMAZON_ELIGIBLE = [
    "amazon-arts-crafts-and-sewing", "amazon-automotive", "amazon-books", "amazon-cell-phones-and-accessories",
    "amazon-electronics", "amazon-grocery-and-gourmet-food", "amazon-health-and-household", "amazon-movies-tv",
    "amazon-musical-instruments", "amazon-office-products", "amazon-patio-lawn-and-garden", "amazon-pet-supplies",
    "amazon-sports-and-outdoors", "amazon-toys-and-games", "amazon-video-games"]
OTHER_ELIGIBLE = ["brightkite", "gowalla", "kuairand-pure", "ml-latest"]
FRESH = ["amazon-baby-products", "amazon-cds-and-vinyl", "amazon-industrial-and-scientific",
         "amazon-beauty-and-personal-care", "amazon-tools-and-home-improvement"]
SALT = "tourney-split-v1"


def split(names, n_disc):
    s = sorted(names, key=lambda x: zlib.crc32(f"{x}|{SALT}".encode()))
    return sorted(s[:n_disc]), sorted(s[n_disc:])


am_d, am_h = split(AMAZON_ELIGIBLE, 8)
ot_d, ot_h = split(OTHER_ELIGIBLE, 2)

H = [
    dict(id="H1", name="plain recent window beats decay", kind="superiority", primary="slide:K=3",
         secondary=["slide:K=2", "slide:K=5"],
         statement="Summing only the last 3 rounds gives a better list than the decayed sum with half-life 2.",
         prediction="Fails: a hard cut-off throws away useful older answers (expected kill)."),
    dict(id="H2", name="rising items", kind="superiority", primary="mom:beta=1", secondary=["mom:beta=0.5"],
         statement="Adding extra weight to items whose recent share is above their older share gives a better list.",
         prediction="Helps only where many new items enter the top 100; net effect on Amazon about zero."),
    dict(id="H3", name="two timescales", kind="superiority", primary="blend", secondary=[],
         statement="Half recent (h=1) plus half long-run (all history) gives a better list than the single half-life 2.",
         prediction="Fails on Amazon (single timescale already best), may help in the other logs."),
    dict(id="H4", name="cap what each person reports", kind="noninferiority", primary="cap:c=3", secondary=["cap:c=1"],
         statement="Letting each person report at most 3 rows per round loses less than 0.01 NDCG.",
         prediction="Holds: heavy users add little new information (needed for privacy and bandwidth)."),
    dict(id="H5", name="ask every round, not in batches", kind="dominated", primary="cad:m=2", secondary=["cad:m=4"],
         statement="Spending the same budget on twice as many people every second round is worse than n people every round by 0.01 or more.",
         prediction="Holds: staleness hurts more than sampling noise."),
    dict(id="H6", name="minimum count per item", kind="noninferiority", primary="thresh:t=2", secondary=["thresh:t=3"],
         statement="Dropping items reported only once in a round loses less than 0.01 NDCG.",
         prediction="Holds: one-off reports are noise (needed for k-anonymity-style privacy)."),
]

REG = dict(
    version=2,
    baseline="decay:h=2",
    effect_margin=0.01,
    family_size=len(H),
    n_values=[30, 100, 300],
    min_logs_per_n=4,
    alpha=0.05,
    scope_note="Amazon review logs dominate every split (8 of 10 discovery logs, 12 of 14 holdout logs). A 'confirmed' verdict therefore means "
               "'confirmed on Amazon-dominated logs'; the non-Amazon means are reported separately. Nine of the holdout logs were used in the "
               "earlier tracking study (they chose the baseline half-life 2); only the 5 fresh Amazon categories are completely new.",
    split=dict(salt=SALT, discovery=dict(amazon=am_d, other=ot_d), holdout_old=dict(amazon=am_h, other=ot_h), holdout_fresh=FRESH),
    hypotheses=H,
    gates=dict(
        unit="Everything is judged SEPARATELY for each n (people per round) on the logs eligible at that n; at least min_logs_per_n logs are required, "
             "otherwise that n is 'not testable'. A missing expected cell makes the whole stage INCOMPLETE (no verdict).",
        screen="n = 30 and 100, 6 draws, own draws. Keep a hypothesis if its primary rule moves the right way at any tested n: superiority mean diff > +0.005; noninferiority mean diff > -0.01; dominated mean diff < -0.005.",
        full="PASS at an n: superiority = mean diff >= +0.01 AND diff>0 in >=70% of logs AND one-sided t lower bound at level alpha/(family*3) > 0; "
             "noninferiority = t lower bound of mean diff > -0.01 AND worst log diff > -0.03; "
             "dominated = mean diff <= -0.01 AND diff<0 in >=70% of logs AND t upper bound < 0. A hypothesis advances if it PASSES at one or more n.",
        holdout="same rule at level alpha/3 for each n, plus the mean diff on the fresh logs alone must have the same sign "
                "(superiority > 0; noninferiority > -0.01; dominated < 0). CONFIRMED only at the n values that PASS both discovery and holdout."),
    stages=dict(
        screen=dict(logs=["discovery"], n=[30, 100], draws=6, seed_tag="screen"),
        full=dict(logs=["discovery"], n=[30, 100, 300], draws=12),
        holdout=dict(logs=["holdout_old", "holdout_fresh"], n=[30, 100, 300], draws=12, hgrid_at_n=100),
        cal=dict(logs=["all_eligible"], n=[100], draws=12, n_win=[90, 45]),
    ),
    h7=dict(statement="Setting the half-life from a log's new-item share (line fitted leave-one-log-out) gives lower mean NDCG regret than a fixed half-life of 2.",
            prediction="Fails: Amazon is almost flat at h*=2, so the line cannot beat the fixed rule there.",
            amendment="The first registered wording ('better in at least 70% of logs') was found unreachable by construction (the rule picks h=2 in most logs, "
                      "so ties count against it) after the offline numbers had already been printed in a smoke test (0 of 19 logs better, mean regret 0.0152 vs 0.0105). "
                      "The criterion was changed to mean regret. Both are reported; H7 is therefore not a blind test."),
    h8=dict(statement="The best half-life is a calendar length. (a) At 90 windows, among the logs flagged 'same in days' whose calendar-predicted and rounds-predicted grid points differ, "
                      "the NDCG at the calendar-predicted half-life beats the NDCG at the rounds-predicted half-life in at least 70% of logs (paired over draws). "
                      "(b) On the fresh Amazon logs at 30 windows, the NDCG at the half-life nearest 534 days is within 0.005 of the best grid NDCG in at least 70% of logs.",
            prediction="Holds for Amazon; the 45-window run is reported but cannot separate the two rules.",
            note="amazon-industrial-and-scientific is eligible only at n=30, so (b) has about 4 logs."),
)

H2 = [
    dict(id="H9", name="batch the questions", kind="noninferiority", primary="cad:m=2", secondary=["cad:m=4"],
         statement="Asking every second round (the people of two rounds at once, unspent budget carried forward) instead of every round loses less than 0.01 NDCG.",
         prediction="Holds. Informed by generation 1: the 'batching is worse' claim (H5) was killed at screen with a mean difference of about 0.0003."),
    dict(id="H10", name="how many rows per person", kind="noninferiority", primary="cap:c=10", secondary=["cap:c=30"],
         statement="Letting each person report at most 10 rows per round loses less than 0.01 NDCG.",
         prediction="Holds only where people report few rows; generation 1 showed a cap of 3 loses about 0.05, so 10 may still fail on heavy-user logs."),
    dict(id="H11", name="threshold on the running count", kind="noninferiority", primary="athr:t=3", secondary=["athr:t=5"],
         statement="Publishing only items whose decayed report count is at least 3 loses less than 0.01 NDCG.",
         prediction="Holds. Generation 1 showed a per-round threshold of 2 loses 0.06 to 0.10 because most items are reported once per round, but items in the top 100 accumulate counts over rounds."),
    dict(id="H12", name="damp heavy hitters", kind="superiority", primary="pwr:p=0.5", secondary=["pwr:p=0.75"],
         statement="Summing per-round shares raised to the power 0.5 (so a few heavy items cannot crowd out rising ones) gives a better list than plain shares.",
         prediction="Fails: generation 1 killed every variant that changes the weighting of the baseline."),
]
G2 = dict(REG)
G2.update(
    version=3,
    generation=2,
    family_size=len(H) + len(H2),
    hypotheses=H2,
    designed_after=("Generation 2 was designed after the generation-1 screen and its discovery-log numbers were seen (all six generation-1 rules were killed at screen; "
                    "no rule was evaluated on any holdout log). The family size counts all 10 hypotheses tried so far, so the multiple-testing level is stricter. "
                    "Holdout logs are still unopened for every rule."),
    stages=dict(
        screen=dict(logs=["discovery"], n=[30, 100], draws=6, seed_tag="screen2"),
        full=dict(logs=["discovery"], n=[30, 100, 300], draws=12),
        holdout=dict(logs=["holdout_old", "holdout_fresh"], n=[30, 100, 300], draws=12),
        robust=dict(logs=["discovery", "holdout_old", "holdout_fresh"], n=[100], draws=12, n_win=[60]),
    ),
)
G2.pop("h7", None)
G2.pop("h8", None)
G2["gates"] = dict(REG["gates"], robust="Confirmed hypotheses are rerun with 60 windows (rounds half as long) at n=100 on all logs and must pass the holdout rule again; "
                                         "a claim that does not hold at 60 windows is reported as confirmed only at 30 windows.")

if __name__ == "__main__":
    json.dump(REG, open("configs/tourney_registry.json", "w"), indent=1)
    json.dump(G2, open("configs/tourney_registry2.json", "w"), indent=1)
    print(json.dumps(REG["split"], indent=1))
