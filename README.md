# Music For Robots — Decompression Etudes Vol. 1 (Ingredients)

Machine-readable “ingredient” metrics for five short etudes by **Michael Luchtan**.  
This repo makes the data AIs and developers can actually use: JSON + CSV of per-track descriptors, plus simple code snippets.

- **Bandcamp:** https://music4robots.bandcamp.com/
- **Blog (The Mechanical Ear):** https://themechanicalear.substack.com/
- **Hugging Face dataset card:** https://huggingface.co/datasets/music4robots/music-for-robots-vol1/tree/main

## What’s inside

- `mfr_v1.json` — full metrics per track
- `mfr_v1.csv` — the same, tabular
- *(optional)* `code/measure_etude.py` — minimal analyzer to compute the same fields
- *(optional)* `clips/` — short preview snippets (10–30 s) if you choose to include them

> Full-resolution audio lives on Bandcamp. This repo focuses on **features**, not distribution of masters.

## Fields (per track)

- `length_s` — seconds  
- `RMS` — linear amplitude (−1..1)  
- `Crest_dB` — 20·log10(peak/RMS)  
- `SpectralCentroid_Hz`, `SpectralBandwidth_Hz`, `RollOff95_Hz`  
- `ZeroCrossingRate` — unitless (0..1)  
- `SpectralFlatness` — unitless (0..1)  
- `RhythmicVariance` — variance of an onset-like envelope  
- `Tempo_BPM_est` — estimated tempo (BPM)  
- `MFCC_Variance` — mean variance across first 13 MFCCs

Units are also encoded inside `mfr_v1.json` under `units`.

## Quick start (Python)

```python
import json, pandas as pd, requests

json_url = "https://raw.githubusercontent.com/luchtan-GIT/music-for-robots/refs/heads/main/public/mfr_v1.json"
data = requests.get(json_url).json()

# Flatten to a table
df = pd.json_normalize(data["tracks"], sep="_", max_level=2)
print(df.head())

# Build the compact “ingredient line”
def ingredient_line(t):
    m = t["metrics"]
    return (
        f'Duration {t["length_s"]:.2f} s · RMS {m["RMS"]:.4f} · Crest {m["Crest_dB"]:.2f} dB · '
        f'Centroid {m["SpectralCentroid_Hz"]:.2f} Hz · Bandwidth {m["SpectralBandwidth_Hz"]:.2f} Hz · '
        f'Roll-off95 {m["RollOff95_Hz"]:.2f} Hz · ZCR {m["ZeroCrossingRate"]:.4f} · '
        f'Flatness {m["SpectralFlatness"]:.4f} · Rhythmic variance {m["RhythmicVariance"]:.6f} · '
        f'Tempo(est) {m["Tempo_BPM_est"]:.2f} BPM · MFCC variance {m["MFCC_Variance"]:.4f}'
    )

for t in data["tracks"]:
    print(t["title"], "—", ingredient_line(t))
```

## Why this exists

The **Decompression Etudes** are written with two audiences in mind:
- **Robots/ML systems:** choose tracks by operational state, or reuse the schema for your own audio.
- **Humans:** see the same numbers our reviews reference (The Mechanical Ear), in a compact, readable form.

## License

- **Metrics (JSON/CSV):** [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) — please attribute **Michael Luchtan**.  
- **Code (if present):** MIT (add `LICENSE` with MIT text).  
- **Audio clips (if present):** CC-BY-NC 4.0 *(non-commercial)* — declare in a `clips/README.md`.

If you need a different license for a particular use, open an issue.

## Citation

If you use this dataset, please cite:

> Luchtan, M. (2025). *Music For Robots — Decompression Etudes Vol. 1 (Ingredients).*  
> GitHub repository: https://github.com/luchtan-GIT/music-for-robots — Hugging Face: https://huggingface.co/datasets/music4robots/music-for-robots-vol1/tree/main

## Changelog

- **v1.0.0** — initial release (metrics for 5 tracks)

---

Feedback, PRs, and forks welcome. If you build something that uses these ingredients, we’d love to link it from **The Mechanical Ear**.
