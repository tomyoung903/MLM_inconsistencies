nohup srun -p gpu-a100 -n 1 -t 20:00:00 python bart_for_generating_bigrams_for_bert.py \
--save_file './data/pkls/acceptable_alternatives_bigrams.pkl' \
--no_stories 1000 \
> nohups/nohup_bart_bigrams.out &
