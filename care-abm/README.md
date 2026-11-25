# care-abm: senyvo amžiaus asmenų globos ABM + agentinis tyrėjas (kursinio darbo prototipas)

Kodas, reikalingas visiems darbo eksperimentams: simuliacija (ODD modelis), kontroliniai tyrėjai
(atsitiktinė paieška, taisyklėmis grįstas, tinklelis), DKM tyrėjas (atmintis, refleksija, įrankiai),
scenarijai S1–S16, statistinė analizė ir paveikslai.

## 1. Diegimas ir greitas patikrinimas

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

Then, in the same terminal with the virtual environment activated:

```bash
pip install -r requirements.txt
python -m pytest -q                       # 11 testų, ~3 s
python -m care_abm.calibrate              # profilių kalibravimas -> configs/profiles.json (jau įtrauktas)
python -m care_abm.run_experiment --list  # visi scenarijai
```

## 2. Kokį API naudoti

Visi modeliai kviečiami per **vieną OpenAI-suderinamą sąsają** (`openai` paketas), todėl `llm_client.py` vienodai
veikia su GPT, atviraisiais modeliais ir kitais tiekėjais; keičiasi tik `configs/models.yaml` įrašas.

| Darbo žymuo | Ką naudoti | Kaip |
|---|---|---|
| **DKM-C** (komercinis) | jūsų turimas **GPT** (OpenAI API) | `export OPENAI_API_KEY=...`, `models.yaml` -> `model: <GPT id>` |
| **DKM-A** (mažas atviras, ~7–8B) | atviras instruct modelis per **vLLM** VU HPC (`slurm/serve_vllm.sh`) *arba*, jei HPC dar neturite, per **OpenRouter / Together / Groq** (tik `base_url`, `model`, raktas) arba **Ollama** lokaliai | žr. komentarus `models.yaml` |
| **DKM-B** (didesnis atviras, >=30B) | tas pats, didesnis modelis (HPC GPU arba pigiai per OpenRouter/Together) | |
| bandymams | `--model MOCK` (be API, nemokamai) | |

Praktiniai dalykai:
* **Tikslų modelio id / versiją (snapshot) užfiksuokite** `models.yaml` ir 3.2 sk. lentelėje (reikalavimas atkuriamumui).
* **Samprotaujantys modeliai (GPT-5 / o-serijos) nepriima `temperature`/`top_p` ir `max_tokens`**: `models.yaml` yra užkomentuoti
  nustatymai (`send_temperature: false`, `token_param: max_completion_tokens`, didesnis `max_tokens`).
  Kad temperatūra būtų vienoda visiems modeliams, geriau naudokite nesamprotaujantį GPT variantą arba aprašykite šį skirtumą darbe.
* Jei serveris nepalaiko `response_format` (JSON režimo), klientas automatiškai pereina prie įprasto teksto ir JSON išgaunamas
  programiškai (`tools.extract_json`).
* Raktų niekada nedėkite į kodą / git – tik aplinkos kintamieji.
* Užklausų kalba: `--lang lt|en`. Mažiems atviriems modeliams lietuvių kalba gali silpninti rezultatą; verta paleisti ir `en`
  kaip papildomą palyginimą (kitaip sumaišysite modelio dydžio ir kalbos efektą).

## 3. Paleidimo eilė (darbo 3.3 sk. tvarka)

```bash
# 0) pilotas (rezultatų NEįtraukti į darbą): formatui ir kaštams įvertinti
python -m care_abm.run_experiment --scenario S5 --model DKM-C --repeats 2 --out results_pilot

# 1) be DKM + tinklelis (greita, ~minutės)
for s in S1 S2 S3 S4 S4b; do python -m care_abm.run_experiment --scenario $s --workers 8; done
python -m care_abm.run_experiment --scenario G1 --workers 8      # tinklelis: tikrasis optimumas (žemas profilis)
python -m care_abm.run_experiment --scenario G2 --workers 8      # (aukštas profilis)

# 2) trys DKM
python -m care_abm.run_experiment --scenario S5 --model DKM-A --workers 4
python -m care_abm.run_experiment --scenario S6 --model DKM-B --workers 4
python -m care_abm.run_experiment --scenario S7 --model DKM-C --workers 4

# 3) PASIRINKITE geriausią DKM (kriterijų nustatykite IŠ ANKSTO, žr. 5 p.), tada ablacijos, kitas profilis, šokai
BEST=DKM-B
for s in S8 S9 S10 S11 S12; do python -m care_abm.run_experiment --scenario $s --model $BEST --workers 4; done
for s in S11_rnd S11_rule S11_fixed S12_rnd S12_rule S12_fixed; do python -m care_abm.run_experiment --scenario $s --workers 8; done

# 4) jautrumas ir atsparumas
for s in S13a S13b S14a S14b S15 S15b S16 S13a_rule S13b_rule S14a_rule S14b_rule; do python -m care_abm.run_experiment --scenario $s --workers 8; done

# 5) analiza -> analysis_out/ (CSV lentelės + paveikslai)
python -m care_abm.analysis.analyze --results results --policy last
```
Paleidimai atnaujinami (jau esantys `rep_XXXX.json` praleidžiami), todėl nutrūkus galima tiesiog paleisti iš naujo.
HPC: `slurm/run_array.sbatch` (užduočių masyvas), `slurm/serve_vllm.sh` (atviras modelis kaip API).

## 4. Moduliai (atitinka 3.1 sk. lentelę)

| Failas | Paskirtis |
|---|---|
| `care_abm/model.py` | `CareModel` (mėnesio žingsnių seka, rodiklių rinkimas), `run_one`, `evaluate` |
| `care_abm/agents.py` | Senolis, Šeimos narys, Darbuotojai, Savivaldybė |
| `care_abm/processes.py` | demografija, poreikio kaita, neformali globa, paslaugų priskyrimas, eilė, ASPN |
| `care_abm/indicators.py` | R1–R11, tyrėjo kokybės funkcija `score` |
| `care_abm/rng.py` | PCG64 + `SeedSequence.spawn`, bendri atsitiktiniai skaičiai |
| `care_abm/researchers/` | `simple.py` (fiksuotas, atsitiktinis, taisyklė), `llm.py` (DKM), `tools.py` (įrankiai, tikrintuvas) |
| `care_abm/llm_client.py` | OpenAI-suderinamas klientas, žetonų / laiko matavimas, `MOCK` |
| `care_abm/run_experiment.py`, `scenarios.py` | scenarijų S1–S16, G1–G2 paleidimas |
| `care_abm/calibrate.py` | profilių kalibravimas |
| `care_abm/analysis/analyze.py` | lentelės, Wilcoxon / Mann–Whitney / Kruskal–Wallis, Holm, bootstrap PI, paveikslai |

## 5. SVARBU: kodas vietomis skiriasi nuo dabartinio darbo teksto – tekstą reikia atnaujinti

1. **Ne Mesa, o NumPy masyvai.** Vienas masyvo elementas = vienas agentas (agentinis modelis, tik vektorizuotas): ~25 ms vietoj
   sekundžių vienam paleidimui (DKM scenarijams reikia dešimčių tūkstančių simuliacijų). Tekste „Mesa“ pakeiskite arba paaiškinkite.
2. **Kalibruojamas biudžetas, ne svertai.** 41,9 ir 312,7 (7,5 karto skirtumas) trimis svertais gauti neįmanoma, todėl profilius
   skiria savivaldybės biudžetas (`configs/profiles.json`), o svertai fiksuoti `DEFAULT_THETA`.
3. **Ciklas be ankstyvo stabdymo.** Visada I = 10 iteracijų; R9 (iki 61 ir 107) skaičiuojamas po fakto. Taip pašalinamas
   1 algoritmo prieštaravimas (stabdymas ties 61 neleido matuoti 107, o aukšto profilio ciklas baigtųsi po 3 iteracijų).
4. **Tyrėjo tikslas – ne vien R1.** `indicators.score`: bauda už nepasiektą 61 ir 107 + globos skurdas + šeimų perdegimas
   (svoriai `config.SCORE_WEIGHTS` – **pagrįskite darbe arba pakeiskite**).
5. **Galutinis vertinimas ant atskirų (held-out) sėklų**, ne ant ciklo metu matytų (išvengiama „geriausias iš 10 triukšmingų“ poslinkio).
   Ciklo viduje visos iteracijos naudoja tas pačias sėklas (CRN), kad tyrėjas nesivytų triukšmo (`--no-crn-iterations` išjungia).
6. **Tyrėjas gali keisti 1–3 svertus** (`pakeitimai` objektas; senas `parametras/nauja_reiksme` formatas irgi priimamas).
7. **Įrankiai**: `palyginti_su_normatyvais` ir `gauti_istorija` įvykdomi kode ir jų rezultatas įterpiamas į užklausą;
   `tikrinti_pasiulyma` taikomas po atsakymo. Tai ne „function calling“ – tekste nerašykite, kad DKM pats kviečia funkcijas.
8. **Refleksija** – laukas `refleksija` tame pačiame atsakyme (1 užklausa / iteracija). Trumpalaikė atmintis = dabartinė ir
   praėjusi iteracija; ilgalaikė = paskutinių `--memory-window` (5) iteracijų lentelė (S8 ją išjungia, S9 išjungia refleksiją).
9. **Papildomi scenarijai** plano spragoms užpildyti: `S4b` (taisyklė su 3 svertais), `G1/G2` (tinklelis = tikrasis optimumas, regret),
   `S11_*`/`S12_*` (šokai su kontroliniais tyrėjais), `S13*_rule/_llm`, `S14*_rule/_llm` (tyrėjo atsparumas), `S15` (sveikatos slenksčiai), `S16` (N = 5000).
10. **ASPN** modeliuojamas kaip nacionalinis intensyvumas (6,2 %, `aspn_capacity_share`), ne savivaldybės eilė, todėl R1b pagal konstrukciją ≈ 6,2 %.
    Laikino atokvėpio gavėjai ataskaitoje `respite_per_1000` nelyginami su SADM 35 / 65 (jis skaičiuojamas kitai grupei).
11. R1 apibrėžimas: (pagalbos į namus + stacionarios globos gavėjai) / gyvi senjorai × 1000, **paskutinių 12 mėn. vidurkis**.
12. Tyrėjo „įvertis“ ir regret: `regret_vs_tinklelis.csv` lygina su G1/G2 geriausia kombinacija (tinklelis 5 × 4 × 4 = 80 taškų).

## 6. Laikini parametrai (`# TODO-DUOMENYS` faile `config.py`) – pakeiskite šaltiniais

Gyvenimo trukmės lentelė (`life_table_csv` su stulpeliais `age,q` iš OSP; kol nėra – Gompertz placeholder), `entrant_rate_annual`
(E_t/S_t), `p_emigration_annual`, sveikatos šuolio parametrai (EHIS / EU-SILC), `beta`, `h_beta_a/b`, `cost_inst`,
`n_social_workers`. Pakeitę **perkalibruokite** (`python -m care_abm.calibrate`) ir į 3.1 sk. kalibravimo lentelę perkelkite
`configs/profiles.json` (`details`: taikinys, gauta reikšmė, sd). **Šie parametrai yra prielaidos – darbe tai turi būti aiškiai pasakyta.**
Perkalibravus pasikeičia ir tinklelio optimumas – iš naujo paleiskite G1/G2.

## 7. Ką parodė tinklelio patikra (pirmas dalykas, kurį verta pažiūrėti)

`results/G1/grid.csv` rodo tikrą kokybės funkcijos paviršių. Su dabartiniais placeholder'iais žemo profilio optimumas yra
`alpha_amb` ties viršutine riba (0,9) ir `respite` ≈ 0,15; `max_darbuotojo_kruvis` žemame profilyje beveik nieko nekeičia
(darbuotojų limitas neriboja), aukštame profilyje (G2) optimumas yra vidinis (`alpha_amb` ≈ 0,3). Tai reiškia, kad žemame profilyje
protinga taisyklė gali susitvarkyti taip pat gerai kaip DKM – tai **teisėtas rezultatas**, jį reikia raportuoti, o ne slėpti.
Jei norite jautresnio palyginimo, keiskite parametrus (`--set ...`) ir tikrinkite G1/G2 prieš leisdami DKM.

## 8. Eksperimentų rezultatai ir 3 skyriaus \TODO

* Kalibravimo lentelė – `configs/profiles.json` (`details`).
* Vykdymo suvestinė (planuota / baigta / nepavyko / trukmė / žetonai) – suskaičiuokite iš `results/*/rep_*.json`
  (failų skaičius, `time_s`, `tech.*`); T1–T7 jau apskaičiuoti `analysis_out/techniniai_rodikliai.csv`.
* Pirminė DKM atrankos metrika (kriterijus nustatomas IŠ ANKSTO!): pvz., `score` mediana, `--policy last`.
* Skaičiavimo aplinka (CPU, GPU, versijos) – `pip freeze`, `nvidia-smi`, `sinfo`; įrašykite į 3.2 lentelę.
