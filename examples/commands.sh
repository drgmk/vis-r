# run the single Gaussian fit
vis-r -v ../data/hr4796.selfcal.npy -o vis-r_1gauss -g 0.03 0.03 26 76 -t gauss -p 0.015 1.07 0.06 0.04 --walker-factor 4.5. --rew

# run the two-sided Gaussian fit
vis-r -v ../data/hr4796.selfcal.npy -o vis-r_1gauss2 -g 0.03 0.03 26 76 -t gauss2 -p 0.015 1.07 0.06 0.06 0.04 --rew

# run the same for Stan
vis-r-stan -v ../data/hr4796.selfcal.npy -o stan_gauss -g 0.03 0.03 0.464 1.335 -t gauss -p 0.015 1.07 0.06 0.04 --rew --sz 2 --sc 20 --save-chains