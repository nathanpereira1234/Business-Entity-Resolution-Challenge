# Business Entity Resolution
 
Runs end-to-end: data -> normalisation -> blocking -> features -> LightGBM -> decoding -> outputs.
 
```bash
pip install -r requirements.txt
python src/pipeline.py --data ../../dataset --out ../../output
python3 ../../utils/validate_submission.py --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv --test-dir ../../dataset/test
```
 
Prints the blocking recall ceiling, 5-fold grouped OOF macro F0.5 (the validation score), the chosen
threshold, top features, and the predicted singleton rate per country (sanity check for France).
 
No external data, APIs or geocoding are used. TF-IDF vectorisers are fit only on the provided
records of each split. Model: LightGBM (MIT licence), far below the 8B-parameter limit.
