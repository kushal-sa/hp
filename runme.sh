!python runner.py EXP2 Random -n 5 2>&1 | tee ./manual_logs/EXP2_Random_5.txt
!python runner.py EXP2 BayesOpt -n 5 2>&1 | tee ./manual_logs/EXP2_BayesOpt_5.txt
!python runner.py EXP2 HyperBand -n 5 2>&1 | tee ./manual_logs/EXP2_HyperBand_5.txt

