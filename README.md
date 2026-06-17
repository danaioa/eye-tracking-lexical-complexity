# Eye-Tracking Lexical Complexity

Estimarea complexitatii lexicale in limba romana utilizand date de eye-tracking, tehnici NLP si algoritmi de machine learning.

## Despre proiect

Acest proiect investigheaza relatia dintre complexitatea lexicala a cuvintelor si comportamentul de citire observat prin date de eye-tracking.

Au fost testate mai multe abordari:
- CatBoost cu pseudo-etichete si embeddings;
- LightGBM cu caracteristici de eye-tracking;
- LightGBM antrenat pe target generat de LLM;
- Metoda hibrida bazata pe Transformer si proxy-uri de complexitate;
- Agregarea statistica directa a metricilor de eye-tracking.

## Date utilizate

Pentru fiecare cuvant au fost utilizate urmatoarele metrici:
- FFD (First Fixation Duration)
- FPRT (First Pass Reading Time)
- TFT (Total Fixation Time)
- RRT (Re-Reading Time)
- TFC (Total Fixation Count)
- skipped

## Rezultate

Cea mai buna performanta a fost obtinuta prin agregarea directa a statisticilor de eye-tracking, cu un scor de 43.59/100.

## Tehnologii utilizate

- Python
- Pandas
- NumPy
- Scikit-Learn
- LightGBM
- CatBoost
