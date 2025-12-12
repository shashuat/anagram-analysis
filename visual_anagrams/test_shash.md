python -m visual_anagrams.generate --name inner_circle.einstein_monroe --prompts "albert einstein" "marilyn monroe" --style "an oil painting of" --views identity inner_circle --num_samples 1 --num_inference_steps 30 --guidance_scale 10.0 --generate_1024 --save_dir results/test


python -m visual_anagrams.generate --name flip.einstein_monroe --prompts "albert einstein" "marilyn monroe" --style "an oil painting of" --views identity flip --num_samples 10 --num_inference_steps 30 --guidance_scale 10.0 --generate_1024




python -m visual_anagrams.generate_variant1_sdxl --name flip.einstein_monroe2 --prompts "albert einstein" "marilyn monroe" --style "an oil painting of" --views identity flip --num_samples 10 --num_inference_steps 30 --guidance_scale 10.0 --generate_1024




python -m visual_anagrams.generate_variant2_lightweight --name rotate_cw.village.horse4 --prompts "a snowy mountain village" "a horse" --style "an oil painting of" --views identity rotate_cw --num_samples 10 --num_inference_steps 30 --guidance_scale 10.0 --generate_1024

generate_variant2_lightweight



name="jigsaw.houseplants.marilyn"
python generate.py --name ${name} --prompts "houseplants" "marilyn monroe" --style "an oil painting of" --views identity jigsaw --num_samples 1 --num_inference_steps 30 --guidance_scale 10.0 --generate_1024 --save_dir results/test
python animate.py --im_path results/test/${name}/0000/sample_64.png --metadata_path results/test/${name}/metadata.pkl
python animate.py --im_path results/test/${name}/0000/sample_256.png --metadata_path results/test/${name}/metadata.pkl
python animate.py --im_path results/test/${name}/0000/sample_1024.png --metadata_path results/test/${name}/metadata.pkl



python -m visual_anagrams.generate --name jigsaw.einstein_monroe --prompts "houseplants" "marilyn monroe" --style "an oil painting of" --views identity jigsaw --num_samples 1 --num_inference_steps 30 --guidance_scale 10.0 --generate_1024


python -m visual_anagrams.animate --im_path results/jigsaw.einstein_monroe/0000/sample_1024.png --metadata_path results/jigsaw.einstein_monroe/metadata.pkl



python -m flip.generate_variant1_sdxl --name jigsaw.homer_trump2 --prompts "homer simpson" "donald trump" --style "an oil painting of" --views identity flip --num_samples 1 --num_inference_steps 30 --guidance_scale 10.0 --generate_1024

python -m visual_anagrams.animate --im_path results/jigsaw.homer_trump/0000/sample_1024.png --metadata_path results/jigsaw.homer_trump/metadata.pkl




python -m visual_anagrams.generate_variant1_sdxl --name flip.homer_trump_shash --prompts "homer simpson" "donald trump" --style "an oil painting of" --views identity flip --num_samples 1 --num_inference_steps 30 --guidance_scale 10.0 --generate_1024

python -m visual_anagrams.animate --im_path results/flip.homer_trump_shash/0000/sample_1024.png --metadata_path results/flip.homer_trump_shash/metadata.pkl


python -m visual_anagrams.generate --name flip.homer_trump_auth --prompts "homer simpson" "donald trump" --style "an oil painting of" --views identity flip --num_samples 1 --num_inference_steps 30 --guidance_scale 10.0 --generate_1024

python -m visual_anagrams.animate --im_path results/flip.homer_trump_auth/0000/sample_1024.png --metadata_path results/flip.homer_trump_auth/metadata.pkl

python -m visual_anagrams.generate_variant2_adaptive --name flip.einstein_monroe_4 --prompts "albert einstein" "marilyn monroe" --style "an oil painting of" --views identity flip --num_samples 10 --num_inference_steps 30 --guidance_scale 10.0 --generate_1024




# variant 3

python -m visual_anagrams.generate_variant3_frequency_orthogonal --name freq_ortho.einstein_monroe5 --prompts "albert einstein" "marilyn monroe" --style "an oil painting of" --views identity flip --num_samples 10 --num_inference_steps 30 --guidance_scale 10.0 --generate_1024

python -m visual_anagrams.generate_variant3_frequency \
  --name filp.dog_cat.variant3 \
  --prompts "dog" "cat" \
  --style "an oil painting of" \
  --views identity flip \
  --num_samples 10 \
  --num_inference_steps 30 \
  --guidance_scale 10.0 \
  --generate_1024

# the fles are generate_variant0_author, generate_variant1_sdxl, generate_variant2_adaptive, generate_variant3_frequency


# Make them executable
chmod +x scripts/generate_all_variants.sh
chmod +x scripts/evaluate_all_variants.sh

# Run generation (this will take a long time!)
./scripts/generate_all_variants.sh

# After generation completes, run evaluation
./scripts/evaluate_all_variants.sh
```

The generation script will create this structure:
```
results/
├── flip.dog_cat.variant0/
├── flip.dog_cat.variant1/
├── flip.dog_cat.variant2/
├── flip.dog_cat.variant3/
├── flip.campfire_man.variant0/
├── flip.campfire_man.variant1/
... (40 directories total: 10 prompt pairs × 4 variants)
```

The evaluation script will create:
```
results/comparisons/
├── dog_cat/
│   ├── comparison_table.csv
│   ├── score_distributions.png
│   └── scatter_comparison.png
├── campfire_man/
... (one directory per prompt pair)
├── all_results.csv (comprehensive results)
└── summary_by_variant.csv (aggregated statistics)