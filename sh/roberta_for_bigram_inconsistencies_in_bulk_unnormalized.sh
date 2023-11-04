# nohup srun -p gpu-a100 -n 1 -t 20:00:00 python roberta_for_bigram_inconsistencies_in_bulk_unnormalized.py --model_name 'roberta-large' --cache_dir './roberta-large-cache' --analytics_filename 'roberta-large-on-bigrams-analytics.pkl' > nohups/roberta_large_for_bigram_inconsistencies_in_bulk_test.out &

nohup srun -p gpu-a100 -n 1 -t 20:00:00 python roberta_for_bigram_inconsistencies_in_bulk_unnormalized.py --model_name 'roberta-base' --cache_dir './roberta-base-cache' --analytics_filename './data/pkls/roberta-base-on-bigrams-analytics.pkl' > nohups/roberta_base_for_bigram_inconsistencies_in_bulk_test.out &

nohup srun -p gpu-a100 -n 1 -t 20:00:00 python roberta_for_bigram_inconsistencies_in_bulk_unnormalized.py --model_name 'roberta-large' --cache_dir './roberta-large-cache' --analytics_filename './data/pkls/roberta-large-on-bigrams-analytics.pkl' > nohups/roberta_large_for_bigram_inconsistencies_in_bulk_test.out &

nohup srun -p gpu-a100 -n 1 -t 20:00:00 python roberta_for_bigram_inconsistencies_in_bulk_unnormalized.py --model_name 'smallbenchnlp/roberta-small' --cache_dir './roberta-small-cache' --analytics_filename './data/pkls/roberta-small-on-bigrams-analytics.pkl' > nohups/roberta_small_for_bigram_inconsistencies_in_bulk_test.out &

nohup srun -p normal -n 1 -t 20:00:00 python roberta_for_bigram_inconsistencies_in_bulk_unnormalized.py --model_name 'smallbenchnlp/roberta-base' --cache_dir './roberta-base-cache-smallbenchblp' --analytics_filename './data/pkls/roberta-base-smallbenchnlp-on-bigrams-analytics.pkl' > nohups/roberta-base-smallbenchnlp_for_bigram_inconsistencies_in_bulk_test.out &
