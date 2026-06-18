# ⚽ WK 2026 Match Predictor

Een **live, zelf-updatende voorspeller** voor het WK 2026 (FIFA World Cup, Canada/Mexico/VS). Het model combineert een zelfgebouwde Elo-rating, een getraind ML-classificatiemodel en een Monte-Carlo-simulatie van het volledige toernooi — en het schuift met elke speelronde mee, omdat gespeelde wedstrijden als vaststaand worden ingeladen.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Status](https://img.shields.io/badge/status-live%20tijdens%20WK%202026-brightgreen)

---

## Wat het doet

Het notebook beantwoordt één vraag: **wie wint het WK 2026, en hoe groot is ieders kans?** In plaats van één voorspelling geeft het een *kansverdeling*, geschat door het toernooi 50.000× te simuleren.

De kern is een pipeline van vier stappen:

1. **Elo-rating** — chronologisch opgebouwd uit ~49.000 interlands sinds 1872. Zwaardere toernooien en grotere overwinningen tellen sterker mee.
2. **ML-classificatiemodel** — een logistische regressie getraind op match-features (Elo-verschil, recente vorm, doelsaldo-trend), eerlijk geëvalueerd met een temporele train/test-split en de Ranked Probability Score (RPS).
3. **Poisson-scorelinemodel** — voor de groepsranglijst en de knockout, gekoppeld aan het Elo-verschil.
4. **Monte-Carlo-simulatie** — 50.000 toernooien, waarbij al gespeelde wedstrijden hun échte uitslag behouden en alleen de resterende worden gesimuleerd.

## Live tijdens het toernooi

De dataset wordt tijdens het WK vaak dagelijks bijgewerkt op GitHub. Bij elke rerun van het notebook:

- worden de nieuwste uitslagen opgehaald en als vaststaand ingeladen;
- worden de groepsstanden (punten, doelsaldo) doorgerekend;
- worden alleen de nog te spelen wedstrijden gesimuleerd;
- wordt een gedateerde snapshot van de inputdata opgeslagen voor reproduceerbaarheid;
- worden drie post-klare visualisaties gegenereerd.

Zo wordt elke update een echt momentopname-verhaal: *de titelkansen ná speelronde X*.

## Validatie

De zelfgebouwde Elo-engine is gevalideerd tegen de officiële [World Football Elo Ratings](https://www.eloratings.net) — **Pearson-correlatie ≈ 0,99, Spearman ≈ 0,98**. De engine reproduceert de gezaghebbende ratings dus bijna perfect, ook al is hij volledig from-scratch uit ruwe uitslagen opgebouwd.

## Output

Het notebook produceert drie visualisaties (en bijbehorende CSV's):

| Visualisatie | Inhoud |
|---|---|
| `post_1_titelkansen.png` | Titelkansen per land |
| `post_2_groepen.png` | De 12 groepen met huidige stand + doorgangskans |
| `post_3_heatmap.png` | Kans per land om elke ronde te bereiken |

## De wiskunde

**Verwachte Elo-score**

$$E_A = \frac{1}{1 + 10^{\,(R_B - R_A)/400}}$$

**Elo-update** (K = toernooigewicht, G = doelsaldo-multiplier)

$$R_A' = R_A + K \cdot G \cdot (S_A - E_A)$$

**Verwachte goals** (d = Elo-verschil)

$$\lambda = \mu \cdot e^{\,\beta d / 200}$$

**Scoreline (Poisson)**

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

**Ranked Probability Score** (evaluatie, lager = beter)

$$\text{RPS} = \frac{1}{R-1} \sum_{i=1}^{R-1} \left( \sum_{r=1}^{i} p_r - \sum_{r=1}^{i} o_r \right)^2$$

**Titelkans (Monte Carlo)**

$$P(\text{kampioen} = T) \approx \frac{1}{N} \sum_{n=1}^{N} \mathbb{1}\{\text{winnaar}_n = T\}$$

## Gebruik

```bash
# vereisten
pip install pandas numpy scikit-learn scipy matplotlib networkx

# open en draai het notebook (Run All)
jupyter notebook "wk2026_match_predictor.ipynb"
```

Het notebook haalt de data automatisch van GitHub. Voor een offline of reproduceerbare run kun je een opgeslagen snapshot inladen in plaats van de live URL.

## Databronnen & credits

- **Wedstrijduitslagen** (1872–heden, incl. WK 2026-loting): [martj42/international_results](https://github.com/martj42/international_results) — publiek domein, vaak dagelijks bijgewerkt.
- **Validatie van de ratings**: [World Football Elo Ratings](https://www.eloratings.net).

## Kanttekeningen

- De knockout-bracket wordt na de groepsfase eenmalig willekeurig geloot en daarna vast afgespeeld — net als een echt toernooi, maar niet de letterlijke officiële Ronde-van-32-structuur.
- Geen blessures, schorsingen, opstellingen of marktwaarden. Een logische volgende stap is bookmaker-odds of spelerratings als extra feature.
- Een voorspelling is een **kansverdeling**, geen voorspelde uitslag. Een favoriet met 22% titelkans wint in ruim 3 van de 4 simulaties juist *niet*. Daarom is voetbal mooi.

---

*Gemaakt door [Amusu Okamoto](https://github.com/IamusuOkamoto).*
