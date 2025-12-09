python -m visual_anagrams.generate --name inner_circle.einstein_monroe --prompts "albert einstein" "marilyn monroe" --style "an oil painting of" --views identity inner_circle --num_samples 1 --num_inference_steps 30 --guidance_scale 10.0 --generate_1024 --save_dir results/test


python -m visual_anagrams.generate --name flip.einstein_monroe --prompts "albert einstein" "marilyn monroe" --style "an oil painting of" --views identity flip --num_samples 10 --num_inference_steps 30 --guidance_scale 10.0 --generate_1024




python -m visual_anagrams.generate_variant1_sdxl --name flip.einstein_monroe2 --prompts "albert einstein" "marilyn monroe" --style "an oil painting of" --views identity flip --num_samples 10 --num_inference_steps 30 --guidance_scale 10.0 --generate_1024




python -m visual_anagrams.generate_variant2_lightweight --name rotate_cw.village.horse4 --prompts "a snowy mountain village" "a horse" --style "an oil painting of" --views identity rotate_cw --num_samples 10 --num_inference_steps 30 --guidance_scale 10.0 --generate_1024

generate_variant2_lightweight