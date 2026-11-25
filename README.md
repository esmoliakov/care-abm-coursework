# care-abm

Senyvo amžiaus asmenų ilgalaikės globos agentais grįstas modelis (ABM) su autonominiu didžiųjų kalbos modelių (DKM) tyrėju.

Kodas yra kursiniam darbui **„Agentinis tyrimas automatizuotam socialinės politikos formavimui: senyvo amžiaus asmenų globos atvejis“**

* savivaldybės lygmens globos simuliacija (ODD modelis, vektorizuota su NumPy);
* kontroliniai tyrėjai: fiksuota politika, atsitiktinė paieška, taisyklėmis grįsti tyrėjai, tinklelio paieška;
* DKM tyrėjas su trumpalaike ir ilgalaike atmintimi, refleksija ir įrankiais;
* scenarijai S1–S16 ir G1–G2, statistinė analizė ir paveikslai.

## 1. Diegimas

Reikalavimai: Python 3.11 arba naujesnė versija.

```bash
python -m venv .venv
# .\.venv\Scripts\Activate.ps1     # Windows PowerShell
# .venv\Scripts\activate.bat       # Windows Command Prompt
pip install -r requirements.txt
```

Patikrinimas:

```bash
python -m pytest -q                       # vienetų testai
python -m care_abm.run_experiment --list  # visų scenarijų sąrašas
```

Profilių kalibravimo rezultatas jau įtrauktas (`configs/profiles.json`). Jį galima atkurti taip:

```bash
python -m care_abm.calibrate
```

## 2. DKM konfigūracija

Visi modeliai kviečiami per vieną OpenAI-suderinamą sąsają, todėl modelis keičiamas tik faile `configs/models.yaml` (`base_url`, `model`).

| Žymuo | Darbe naudotas modelis | Aptarnavimas |
|---|---|---|
| DKM-A | Qwen2.5-7B-Instruct | vLLM, VU MIF HPC (1 × V100) |
| DKM-B | Qwen2.5-32B-Instruct | vLLM, VU MIF HPC (4 × V100) |
| DKM-C | gpt-4.1 | OpenAI API |


Atvirus modelius taip pat galima pasiekti per bet kurį OpenAI-suderinamą tiekėją (pvz., OpenRouter, Together) arba lokaliai (Ollama), pakeitus `base_url` ir `model`.

Jei serveris nepalaiko JSON režimo (`response_format`), klientas automatiškai pereina prie teksto, o JSON išgaunamas programiškai.

## 3. Eksperimentų paleidimas

Paleidimai tęsiami nuo ten, kur sustojo: jau esantys `rep_XXXX.json` failai praleidžiami.

```bash
# 1) Kontroliniai scenarijai ir tinklelio paieška (be DKM)
for s in S1 S2 S3 S4 S4b; do python -m care_abm.run_experiment --scenario $s --workers 8; done
python -m care_abm.run_experiment --scenario G1 --workers 8   # žemas profilis
python -m care_abm.run_experiment --scenario G2 --workers 8   # aukštas profilis

# 2) DKM tyrėjų palyginimas
python -m care_abm.run_experiment --scenario S5 --model DKM-A --workers 4
python -m care_abm.run_experiment --scenario S6 --model DKM-B --workers 4
python -m care_abm.run_experiment --scenario S7 --model DKM-C --workers 4

# 3) Ablacijos, aukštas profilis ir šokai (DKM-A)
for s in S8 S9 S10 S11 S12; do python -m care_abm.run_experiment --scenario $s --model DKM-A --workers 4; done
for s in S11_rnd S11_rule S11_fixed S12_rnd S12_rule S12_fixed; do python -m care_abm.run_experiment --scenario $s --workers 8; done


# 5) Analizė: CSV lentelės ir paveikslai -> analysis_out/
python -m care_abm.analysis.analyze --results results --policy last
```

Windows PowerShell ciklo pavyzdys:

```powershell
foreach ($s in "S1","S2","S3","S4","S4b") { python -m care_abm.run_experiment --scenario $s --workers 8 }
```

Kitos naudingos parinktys:

| Parinktis | Paskirtis |
|---|---|
| `--repeats N` | pakartojimų skaičius (numatytasis nurodytas scenarijuje) |
| `--out DIR` | rezultatų katalogas (numatytasis `results`) |
| `--model MOCK` | paleidimas be DKM API |
| `--lang lt\|en` | užklausų kalba (darbe naudota `lt`) |
| `--memory-window K` | ilgalaikės atminties langas (numatytasis 5) |
| `--no-crn-iterations` | išjungia bendrus atsitiktinius skaičius tarp iteracijų |
| `--set ...` | modelio parametro pakeitimas |


## 4. Pagrindiniai eksperimentų nustatymai

* Simuliacijos trukmė 36 mėnesiai, tyrėjo ciklas I = 10 iteracijų, kiekvienoje iteracijoje m = 5 simuliacijos pakartojimai.
* Ciklo iteracijos naudoja bendrus atsitiktinius skaičius (CRN). Galutinė politika vertinama ant n_eval = 20 atidėtųjų sėklų, kurių tyrėjas nematė.
* Tyrėjas per iteraciją keičia 1–3 svertus: pagalbos į namus biudžeto dalį `alpha_amb`, darbuotojo krūvio ribą `max_darbuotojo_kruvis` ir laikino atokvėpio dalį `respite`.
* Tyrėjo kokybės funkcija (`indicators.score`) jungia normatyvų (61 ir 107 gavėjai 1000-iui senjorų) nepasiekimą, globos skurdą ir šeimų perdegimą. Svoriai nurodyti `config.SCORE_WEIGHTS`.
* Atotrūkis nuo optimumo skaičiuojamas lyginant su geriausia G1 / G2 tinklelio kombinacija (5 × 4 × 4 = 80 taškų).
* Modelio parametrai ir jų šaltiniai (OSP, SADM, modelio prielaidos) aprašyti `care_abm/config.py` ir darbo 11 lentelėje.

## 5. Repozitorijos struktūra

| Failas / katalogas | Paskirtis |
|---|---|
| `care_abm/model.py` | `CareModel`: mėnesio žingsnių seka, rodiklių rinkimas, `run_one`, `evaluate` |
| `care_abm/agents.py` | agentai: senjorai, šeimos nariai, darbuotojai, savivaldybė |
| `care_abm/processes.py` | demografija, poreikio kaita, neformali globa, paslaugų priskyrimas, eilė, ASPN |
| `care_abm/indicators.py` | rodikliai R1–R11, kokybės funkcija `score` |
| `care_abm/rng.py` | PCG64 ir `SeedSequence.spawn`, bendri atsitiktiniai skaičiai |
| `care_abm/researchers/` | `simple.py` (kontroliniai tyrėjai), `llm.py` (DKM tyrėjas), `tools.py` (įrankiai ir pasiūlymų tikrintuvas) |
| `care_abm/llm_client.py` | OpenAI-suderinamas klientas, žetonų ir laiko matavimas, `MOCK` režimas |
| `care_abm/run_experiment.py`, `scenarios.py` | scenarijų S1–S16, G1–G2 registras ir paleidimas |
| `care_abm/calibrate.py` | profilių kalibravimas (savivaldybės biudžetas) |
| `care_abm/analysis/analyze.py` | lentelės, Wilcoxono, Manno–Whitney ir Kruskalo–Wallis testai, Holmo korekcija, savirankos PI, paveikslai |
| `configs/` | profiliai, DKM konfigūracija, gyvenimo trukmės lentelė |
| `slurm/` | HPC skriptai |
| `results/` | paleidimų rezultatai (`rep_XXXX.json`) |
| `analysis_out/` | analizės lentelės (CSV) ir paveikslai |
