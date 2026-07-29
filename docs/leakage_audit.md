# Leakage audit — 2026-07-22
commit: NOT A GIT REPOSITORY — no commit hash exists; file integrity anchors instead:
  sha256[src/features.py] = 5728fa5e2cb36b0c…
  sha256[src/test_leakage.py] = 5f204969403d95a5…
  sha256[data/features.csv] = ec075dd02e24044c…
  sha256[data/games_full.csv] = e3e531ea24ef75bd…
  sha256[data/pitcher_games.csv] = 21af59a0e26b866c…
split: train seasons <= 2023, test > 2023; chronological, asserted on game_order
model: SimpleImputer(median) + StandardScaler + LogisticRegression(max_iter=2000), all fit on train only
features: 11 columns (home_field deliberately omitted in Chunk 3, so 'twelve' is eleven on disk)
rows: 23776 modelable (post burn-in, non-null label) = 18917 train + 4859 test

## Test 1 — shuffle test (5 seeds; judged on log loss; accuracy stays near the base rate BY DESIGN since shuffling preserves the class balance)
  seed 11: log_loss=0.6903  acc=0.5388  [PASS]
  seed 22: log_loss=0.6918  acc=0.5302  [PASS]
  seed 33: log_loss=0.6913  acc=0.5304  [PASS]
  seed 44: log_loss=0.6922  acc=0.5242  [PASS]
  seed 55: log_loss=0.6911  acc=0.5293  [PASS]
Test 1: PASS

## Test 2 — real model, chronological holdout (train 2015-2023 n=18917, test 2024-2025 n=4859)
  model:               acc=0.5538  log_loss=0.6817  brier=0.2444  auc=0.5760
  always-pick-home:    acc=0.5322
  base-rate constant (0.5323): log_loss=0.6911
Test 2: PASS (within honest range; tripwire acc>0.59 / ll<0.655 not hit)

## Test 3 — prediction distribution and calibration
  prediction quantiles: min=0.194  p1=0.333  p5=0.396  p25=0.480  p50=0.533  p75=0.584  p95=0.663  p99=0.718  max=0.845
  predictions outside [0.25, 0.80]: 10 of 4859
  10 most extreme predictions:
    game_pk=778472 p=0.845 form=(1.0, 0.0) rdpg=(3.00, -2.83) fip=(-1.27, 1.92)
    game_pk=746900 p=0.835 form=(0.6, 0.1666666666666666) rdpg=(1.80, -5.50) fip=(1.15, 2.01)
    game_pk=746497 p=0.817 form=(0.8333333333333334, 0.1428571428571428) rdpg=(1.67, -4.14) fip=(-1.10, 1.85)
    game_pk=746655 p=0.816 form=(0.7272727272727273, 0.1818181818181818) rdpg=(3.09, -2.55) fip=(0.03, 2.92)
    game_pk=745678 p=0.194 form=(0.1666666666666666, 0.6666666666666666) rdpg=(-4.67, 2.33) fip=(1.14, 0.30)
    game_pk=746654 p=0.804 form=(0.7777777777777778, 0.1111111111111111) rdpg=(3.56, -2.89) fip=(0.19, 1.32)
    game_pk=746653 p=0.802 form=(0.8, 0.1) rdpg=(3.60, -3.00) fip=(1.66, 2.48)
    game_pk=746328 p=0.796 form=(0.4285714285714285, 0.1666666666666666) rdpg=(1.86, -3.33) fip=(0.62, 3.13)
    game_pk=777700 p=0.784 form=(0.5333333333333333, 0.1333333333333333) rdpg=(1.04, -3.11) fip=(0.07, 2.23)
    game_pk=777768 p=0.221 form=(0.2, 0.7333333333333333) rdpg=(-3.27, 2.08) fip=(2.60, 0.30)
  calibration by decile (pred mean vs observed rate):
    (0.193, 0.43]      n= 486  pred=0.386  obs=0.414  gap=+0.028
    (0.43, 0.466]      n= 486  pred=0.450  obs=0.490  gap=+0.040
    (0.466, 0.492]     n= 486  pred=0.479  obs=0.490  gap=+0.010
    (0.492, 0.513]     n= 486  pred=0.503  obs=0.467  gap=+0.036
    (0.513, 0.533]     n= 486  pred=0.523  obs=0.545  gap=+0.022
    (0.533, 0.553]     n= 485  pred=0.542  obs=0.540  gap=+0.002
    (0.553, 0.573]     n= 486  pred=0.563  obs=0.553  gap=+0.009
    (0.573, 0.598]     n= 486  pred=0.585  obs=0.543  gap=+0.042
    (0.598, 0.631]     n= 486  pred=0.614  obs=0.605  gap=+0.009
    (0.631, 0.845]     n= 486  pred=0.672  obs=0.675  gap=+0.003
  worst decile gap: 0.042
Test 3: PASS

## Test 4 — single-feature models (fail if any lone feature acc > 0.580; scrutiny note above 0.560)
  home_run_diff_pg     acc=0.5507  log_loss=0.6865
  away_run_diff_pg     acc=0.5371  log_loss=0.6873
  home_form_15         acc=0.5339  log_loss=0.6892
  home_sp_k_rate       acc=0.5334  log_loss=0.6898
  away_tz_delta        acc=0.5324  log_loss=0.6910
  away_rest_days       acc=0.5322  log_loss=0.6911
  away_form_15         acc=0.5320  log_loss=0.6893
  away_sp_k_rate       acc=0.5316  log_loss=0.6893
  home_rest_days       acc=0.5306  log_loss=0.6911
  home_sp_fip_60       acc=0.5267  log_loss=0.6898
  away_sp_fip_60       acc=0.5267  log_loss=0.6910
  leave-one-out (change in test log loss vs full model; scrutiny above +0.010):
  drop home_form_15         ll=0.6816  delta=-0.0001
  drop away_form_15         ll=0.6816  delta=-0.0000
  drop home_run_diff_pg     ll=0.6834  delta=+0.0018
  drop away_run_diff_pg     ll=0.6833  delta=+0.0016
  drop home_rest_days       ll=0.6817  delta=+0.0000
  drop away_rest_days       ll=0.6817  delta=+0.0000
  drop home_sp_fip_60       ll=0.6819  delta=+0.0002
  drop away_sp_fip_60       ll=0.6815  delta=-0.0002
  drop home_sp_k_rate       ll=0.6817  delta=+0.0000
  drop away_sp_k_rate       ll=0.6824  delta=+0.0007
  drop away_tz_delta        ll=0.6817  delta=+0.0001
Test 4: PASS

## Test 5 — manual trace of three hard-coded games ([446946, 633121, 663152]) — every feature, 6-decimal equality

  game_pk=446946 date=2016-04-10 (game_order=2505):
    feature                          hand        table  match
    home_games_played                   6            6  OK
    home_form_low_confidence         True         True  OK
    home_form_15                 0.666667     0.666667  OK
    home_run_diff_pg             2.500000     2.500000  OK
    home_rest_days                      1            1  OK
    home_sp_fip_60               1.341709     1.341709  OK
    home_sp_k_rate               0.147766     0.147766  OK
    home_sp_trailing_ip         66.333333    66.333333  OK
    home_sp_low_confidence          False        False  OK
    away_games_played                   6            6  OK
    away_form_low_confidence         True         True  OK
    away_form_15                 0.666667     0.666667  OK
    away_run_diff_pg             3.166667     3.166667  OK
    away_rest_days                      1            1  OK
    away_tz_delta                       0     0.000000  OK
    away_sp_fip_60               2.242268     2.242268  OK
    away_sp_k_rate               0.177083     0.177083  OK
    away_sp_trailing_ip         64.666667    64.666667  OK
    away_sp_low_confidence          False        False  OK
    contributing pks [home_form_15]: [446875, 446885, 446898, 446907, 446916, 446931]
    contributing pks [away_form_15]: [446871, 446884, 446894, 446907, 446916, 446931]
    contributing pks [home_sp_fip_60]: [415383, 415468, 415533, 415620, 415684, 415783, 415850, 415936, 415998, 416072, 446885]
    contributing pks [away_sp_fip_60]: [415257, 415336, 415422, 415484, 415565, 415640, 415716, 415798, 415868, 415954, 416012, 446884]

  game_pk=633121 date=2021-07-26 (game_order=13636):
    feature                          hand        table  match
    home_games_played                  97           97  OK
    home_form_low_confidence        False        False  OK
    home_form_15                 0.466667     0.466667  OK
    home_run_diff_pg            -0.896907    -0.896907  OK
    home_rest_days                      1            1  OK
    home_sp_fip_60               0.978947     0.978947  OK
    home_sp_k_rate               0.208178     0.208178  OK
    home_sp_trailing_ip         63.333333    63.333333  OK
    home_sp_low_confidence          False        False  OK
    away_games_played                  99           99  OK
    away_form_low_confidence        False        False  OK
    away_form_15                 0.666667     0.666667  OK
    away_run_diff_pg             1.161616     1.161616  OK
    away_rest_days                      1            1  OK
    away_tz_delta                       0     0.000000  OK
    away_sp_fip_60               1.647668     1.647668  OK
    away_sp_k_rate               0.166078     0.166078  OK
    away_sp_trailing_ip         64.333333    64.333333  OK
    away_sp_low_confidence          False        False  OK
    contributing pks [home_form_15]: [633431, 633363, 633369, 633383, 633409, 633373, 633317, 633328, 633248, 633287, 633223, 633227, 633232, 633201, 633183]
    contributing pks [away_form_15]: [633352, 633335, 633377, 633350, 633282, 633309, 633262, 633247, 633278, 633476, 633268, 633205, 633163, 633167, 633151]
    contributing pks [home_sp_fip_60]: [634064, 633900, 633799, 633770, 633755, 633558, 633556, 633467, 633363, 633317, 633223]
    contributing pks [away_sp_fip_60]: [634160, 634077, 634009, 633960, 633840, 633733, 633676, 633602, 633501, 633442, 633377, 633268]

  game_pk=663152 date=2022-06-04 (game_order=15370):
    feature                          hand        table  match
    home_games_played                  53           53  OK
    home_form_low_confidence        False        False  OK
    home_form_15                 0.533333     0.533333  OK
    home_run_diff_pg            -0.150943    -0.150943  OK
    home_rest_days                      0            0  OK
    home_sp_fip_60               1.122588     1.122588  OK
    home_sp_k_rate               0.218221     0.218221  OK
    home_sp_trailing_ip          0.000000     0.000000  OK
    home_sp_low_confidence           True         True  OK
    away_games_played                  53           53  OK
    away_form_low_confidence        False        False  OK
    away_form_15                 0.666667     0.666667  OK
    away_run_diff_pg             0.924528     0.924528  OK
    away_rest_days                      0            0  OK
    away_tz_delta                       0     0.000000  OK
    away_sp_fip_60               1.122588     1.122588  OK
    away_sp_k_rate               0.218221     0.218221  OK
    away_sp_trailing_ip         25.333333    25.333333  OK
    away_sp_low_confidence           True         True  OK
    contributing pks [home_form_15]: [663194, 663150, 663073, 663103, 663082, 663081, 661518, 661461, 663149, 663147, 663154, 663184, 663139, 663153, 663179]
    contributing pks [away_form_15]: [662506, 662478, 662479, 661999, 661998, 661997, 661992, 661991, 661990, 661989, 661977, 661982, 663139, 663153, 663179]
    contributing pks [home_sp_fip_60]: []
    contributing pks [away_sp_fip_60]: [662019, 661267, 661404, 663100, 662007, 662037, 661988, 662105, 662098, 661987, 661985, 662566, 662520, 662478, 661999, 661997, 661989]
    DOUBLEHEADER: game 1 (pk=663179) in game 2's form windows: home=True away=True OK
Test 5: PASS

## Test 6 — ordering assertion on game_order via provenance (additive path; sets are the definitional strict-< slices, proven equal to the fast path by Defense 1 + Test 5)
  sample: 3317 games = ALL 359 doubleheader game-2s + 3000 random + 3 traces
  strict-order violations: 0 (must be 0)
  doubleheader game-2 rows whose windows LACK game 1: 0 (must be 0)
Test 6: PASS

## Test 7 — re-run Chunk 3's defenses (verbatim output)
  | building fast-path features once for comparison...
  |   reference check 500/3000 rows...
  |   reference check 1000/3000 rows...
  |   reference check 1500/3000 rows...
  |   reference check 2000/3000 rows...
  |   reference check 2500/3000 rows...
  |   reference check 3000/3000 rows...
  | Defense 1 (reference implementation, 3000 rows): PASS
  |   future-deletion check 50/200 games...
  |   future-deletion check 100/200 games...
  |   future-deletion check 150/200 games...
  |   future-deletion check 200/200 games...
  | Defense 2 (future deletion, 200 games): PASS
  |   shuffled-label corr home_form_15           r=-0.0158
  |   shuffled-label corr away_form_15           r=+0.0010
  |   shuffled-label corr home_run_diff_pg       r=+0.0027
  |   shuffled-label corr away_run_diff_pg       r=-0.0056
  |   shuffled-label corr home_rest_days         r=+0.0095
  |   shuffled-label corr away_rest_days         r=+0.0077
  |   shuffled-label corr home_sp_fip_60         r=-0.0191
  |   shuffled-label corr away_sp_fip_60         r=-0.0015
  |   shuffled-label corr home_sp_k_rate         r=+0.0126
  |   shuffled-label corr away_sp_k_rate         r=-0.0019
  |   shuffled-label corr away_tz_delta          r=-0.0007
  | Defense 3 (shuffled labels): worst |r|=0.0191 â€” PASS
  | 
  | ============================================================
  |   PASS  Defense 1 â€” reference implementation
  |   PASS  Defense 2 â€” future deletion
  |   PASS  Defense 3 â€” shuffled labels
  | ============================================================
  | load_games: 24295 rows | game_type breakdown: {'R': 24295}
  confirm: Defense 1 sampled 3000 rows: YES
  confirm: Defense 2 covered 200 games: YES
  confirm: Defense 3 all |r| < 0.03: YES
Test 7: PASS (exit code 0)

============================================================
AUDIT SUMMARY
  PASS  Test 1 (shuffle)
  PASS  Test 2 (holdout)
  PASS  Test 3 (calibration)
  PASS  Test 4 (ablation)
  PASS  Test 5 (manual traces)
  PASS  Test 6 (ordering/provenance)
  PASS  Test 7 (Chunk 3 defenses)
AUDIT: PASS
============================================================
