nohup python calculate_inconsistencies_t5_c4.py \
--url_to_probs_dict_file '/work/09127/tomyoung/ls6/data/pkls/url_to_probs_c4_dict_with_labels_t5_3b_valid.pkl' \
--c4_json_file '/work/09127/tomyoung/ls6/data/jsons/c4-validation.00000-of-00001-list-of-lists.json' --acceptable_alternatives_file '/work/09127/tomyoung/ls6/data/pkls/acceptable_alternatives_1000_ignore_cws_nos_50_valid.pkl' --model_name 't5-3b' --model_parallelism --cache_dir '/work/09127/tomyoung/ls6/inconsistencies_project/t5-3b-cache' > nohups/nohup_t5_3b_valid.out &


nohup python calculate_inconsistencies_t5_c4.py \
--url_to_probs_dict_file '/work/09127/tomyoung/ls6/data/pkls/url_to_probs_c4_dict_with_labels_t5_large_valid.pkl' \
--c4_json_file '/work/09127/tomyoung/ls6/data/jsons/c4-validation.00000-of-00001-list-of-lists.json' --acceptable_alternatives_file '/work/09127/tomyoung/ls6/data/pkls/acceptable_alternatives_1000_ignore_cws_nos_50_valid.pkl' --model_name 't5-large' --model_parallelism --cache_dir '/work/09127/tomyoung/ls6/inconsistencies_project/t5-large-cache' > nohups/nohup_t5_large_valid.out &


nohup python calculate_inconsistencies_t5_c4.py \
--url_to_probs_dict_file '/work/09127/tomyoung/ls6/data/pkls/url_to_probs_c4_dict_with_labels_t5_base_valid.pkl' \
--c4_json_file '/work/09127/tomyoung/ls6/data/jsons/c4-validation.00000-of-00001-list-of-lists.json' --acceptable_alternatives_file '/work/09127/tomyoung/ls6/data/pkls/acceptable_alternatives_1000_ignore_cws_nos_50_valid.pkl' --model_name 't5-base' --model_parallelism --cache_dir '/work/09127/tomyoung/ls6/inconsistencies_project/t5-base-cache' > nohups/nohup_t5_base_valid.out &


# 626097 running
nohup srun -p gpu-a100 -n 1 -t 20:00:00 python calculate_inconsistencies_t5_c4.py \
--url_to_probs_dict_file './data/pkls/url_to_probs_t5_11b_train.pkl' \
--c4_json_file './data/c4-train.00000-of-00512-list-of-lists.json' \
--acceptable_alternatives_file './data/pkls/acceptable_alternatives_1000_ignore_cws_nos_50.pkl' \
--model_name 't5-11b' --model_parallelism \
--cache_dir './t5-11b-cache' > nohups/nohup_t5_11b_train.out &


# nohup srun -p gpu-a100 -n 1 -t 20:00:00 python calculate_inconsistencies_t5_c4.py \
# --url_to_probs_dict_file './data/pkls/url_to_probs_t5_3b_train.pkl' \
# --c4_json_file './data/c4-train.00000-of-00512-list-of-lists.json' \
# --acceptable_alternatives_file './data/pkls/acceptable_alternatives_1000_ignore_cws_nos_50.pkl' \
# --model_name 't5-3b' --model_parallelism \
# --cache_dir './t5-3b-cache' > nohups/nohup_t5_3b_train.out &

nohup srun -p gpu-a100 -n 1 -t 20:00:00 python calculate_inconsistencies_t5_c4.py \
--url_to_probs_dict_file './data/pkls/url_to_probs_t5_base_train.pkl' \
--c4_json_file './data/c4-train.00000-of-00512-list-of-lists.json' \
--acceptable_alternatives_file './data/pkls/acceptable_alternatives_1000_ignore_cws_nos_50.pkl' \
--model_name 't5-base' --model_parallelism \
--cache_dir './t5-base-cache' > nohups/nohup_t5_base_train.out &