# ⚽ WK 2026 Match Predictor & NL Goal-analyse

Twee projecten rond het WK 2026 (FIFA World Cup, Canada/Mexico/VS):

1. **Match Predictor** — een live, zelf-updatende voorspeller die een zelfgebouwde Elo-rating, een getraind ML-classificatiemodel en een Monte-Carlo-simulatie van het volledige toernooi combineert, en met elke speelronde meeschuift.
2. **NL Goal-analyse** — een verkennende analyse van *vanaf waar* en *wanneer* Oranje scoort, over meerdere WK's heen, op basis van echte event-data.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Status](https://img.shields.io/badge/status-live%20tijdens%20WK%202026-brightgreen)

---

## 1. Match Predictor

`Notebooks/wk2026_match_predictor.ipynb`

Het notebook beantwoordt één vraag: **wie wint het WK 2026, en hoe groot is ieders kans?** In plaats van één voorspelling geeft het een *kansverdeling*, geschat door het toernooi 50.000× te simuleren.

De kern is een pipeline van vier stappen:

1. **Elo-rating** — chronologisch opgebouwd uit ~49.000 interlands sinds 1872. Zwaardere toernooien en grotere overwinningen tellen sterker mee.
2. **ML-classificatiemodel** — een logistische regressie getraind op match-features (Elo-verschil, recente vorm, doelsaldo-trend), eerlijk geëvalueerd met een temporele train/test-split en de Ranked Probability Score (RPS).
3. **Poisson-scorelinemodel** — voor de groepsranglijst en de knockout, gekoppeld aan het Elo-verschil.
4. **Monte-Carlo-simulatie** — 50.000 toernooien, waarbij al gespeelde wedstrijden hun échte uitslag behouden en alleen de resterende worden gesimuleerd.

### Individuele wedstrijdvoorspeller

Naast de toernooisimulatie bevat het notebook een `predict_match(thuis, uit)`-functie die elke afzonderlijke wedstrijd voorspelt — gekalibreerde W/D/L-kansen (logistische regressie) plus een Poisson-scoreline-heatmap, met optioneel een vergelijking tegen de bookmakermarkt en een post-match Brier-score.

### Live tijdens het toernooi

De dataset wordt tijdens het WK vaak dagelijks bijgewerkt op GitHub. Bij elke rerun:

- worden de nieuwste uitslagen opgehaald en als vaststaand ingeladen;
- worden de groepsstanden (punten, doelsaldo) doorgerekend;
- worden alleen de nog te spelen wedstrijden gesimuleerd;
- wordt een gedateerde snapshot van de inputdata opgeslagen voor reproduceerbaarheid;
- worden drie post-klare visualisaties gegenereerd.

Zo wordt elke update een echt momentopname-verhaal: *de titelkansen ná speelronde X*.

### Validatie

De zelfgebouwde Elo-engine is gevalideerd tegen de officiële [World Football Elo Ratings](https://www.eloratings.net) — **Pearson-correlatie ≈ 0,99, Spearman ≈ 0,98**. De engine reproduceert de gezaghebbende ratings dus bijna perfect, ook al is hij volledig from-scratch uit ruwe uitslagen opgebouwd.

---

## 2. NL Goal-analyse

`Notebooks/nl_heatmap_1974_2022_2026.ipynb`

Een verkennende analyse van het scoringspatroon van Oranje over verschillende WK's:

- **Schot-heatmaps** — vanaf waar Nederland scoorde (1974, 2022, en 2026 voor zover gespeeld), op basis van StatsBomb event-data. Voor recente wedstrijden zonder gepubliceerde event-data worden goalposities handmatig ingevoerd via een coördinaten-afleeskaart.
- **Goal-timing** — in welke wedstrijdfase Oranje scoort, vergeleken over meerdere WK's en afgezet tegen een eeuw interlandvoetbal.

> ⚠️ **Kanttekening:** met ~7–15 goals per toernooi is dit een beschrijvende vergelijking, geen harde statistische claim. Eén goal meer of minder verschuift het beeld zichtbaar. De analyse toont een patroon, geen bewijs.

---

## Output

| Visualisatie | Inhoud |
|---|---|
| `post_1_titelkansen.png` | Titelkansen per land |
| `post_2_groepen.png` | De 12 groepen met huidige stand + doorgangskans |
| `post_3_heatmap.png` | Kans per land om elke ronde te bereiken |
| `nl_heatmaps_1974_2022_2026.png` | Vanaf waar Oranje scoort, per WK |
| `nl_timing_per_wk.png` | Wanneer Oranje scoort, verdeeld over de wedstrijdfases |

## De wiskunde

De volledige afleiding staat in sectie 12 van het predictor-notebook; hieronder de kern.

**Verwachte Elo-score** van team $A$ tegen $B$, gegeven ratings $R_A$ en $R_B$:

$$E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}$$

**Elo-update** na de wedstrijd, met werkelijke uitkomst $S_A \in \lbrace 1, \tfrac{1}{2}, 0 \rbrace$:

$$R_A' = R_A + K \cdot G \cdot (S_A - E_A)$$

waarbij $K$ het toernooigewicht is (een WK-duel weegt zwaarder dan een vriendschappelijke) en $G$ een doelsaldo-multiplier die grotere overwinningen sterker laat meetellen. Met $\Delta$ = doelsaldo:

$$G = 1 \ \ (|\Delta| \le 1), \qquad G = \tfrac{3}{2} \ \ (|\Delta| = 2), \qquad G = \frac{11 + |\Delta|}{8} \ \ (|\Delta| \ge 3)$$

Thuisspelende teams krijgen vooraf $+H$ Elo-punten ($H = 65$), behalve op neutrale grond.

**Van Elo naar verwachte goals**, log-lineair gekoppeld aan het Elo-verschil $d = R_A - R_B$ rond het toernooigemiddelde $\mu$:

$$\lambda_A = \mu \cdot e^{\beta d / 200}, \qquad \lambda_B = \mu \cdot e^{-\beta d / 200}$$

**Scoreline (Poisson)** — kans op precies $k$ goals; een specifieke uitslag $(i,j)$ is het product $P(X_A = i)\,P(X_B = j)$:

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

**Ranked Probability Score** (evaluatie, lager = beter):

$$\text{RPS} = \frac{1}{R-1} \sum_{i=1}^{R-1} \left( \sum_{r=1}^{i} p_r - \sum_{r=1}^{i} o_r \right)^2$$

**Titelkans (Monte Carlo)** over $N$ gesimuleerde toernooien:

$$P(\text{kampioen} = T) \approx \frac{1}{N} \sum_{n=1}^{N} \mathbb{1}\lbrace \text{winnaar}_n = T \rbrace$$

## Gebruik

```bash
# vereisten
pip install pandas numpy scikit-learn scipy matplotlib networkx mplsoccer

# open en draai een notebook (Run All)
jupyter notebook "Notebooks/wk2026_match_predictor.ipynb"
jupyter notebook "Notebooks/nl_heatmap_1974_2022_2026.ipynb"
```

De notebooks halen hun data automatisch op. Voor een offline of reproduceerbare run kun je een opgeslagen snapshot inladen in plaats van de live URL.

## Databronnen & credits

- **Wedstrijduitslagen** (1872–heden, incl. WK 2026-loting): [martj42/international_results](https://github.com/martj42/international_results) — publiek domein, vaak dagelijks bijgewerkt.
- **Event- en schotdata** (schotlocaties, goal-minuten): [StatsBomb Open Data](https://github.com/statsbomb/open-data).
- **Validatie van de ratings**: [World Football Elo Ratings](https://www.eloratings.net).

## Kanttekeningen

- De knockout-bracket wordt na de groepsfase eenmalig willekeurig geloot en daarna vast afgespeeld — net als een echt toernooi, maar niet de letterlijke officiële Ronde-van-32-structuur.
- Geen blessures, schorsingen, opstellingen of marktwaarden. Een logische volgende stap is bookmaker-odds of spelerratings als extra feature.
- De goalposities voor recente, nog niet door StatsBomb gepubliceerde wedstrijden zijn handmatige schattingen; ze blijven indicatief tot de officiële event-data verschijnt.
- Een voorspelling is een **kansverdeling**, geen voorspelde uitslag. Een favoriet met 22% titelkans wint in ruim 3 van de 4 simulaties juist *niet*. Daarom is voetbal mooi.

---

*Gemaakt door [Amusu Okamoto](https://github.com/IamusuOkamoto).*
