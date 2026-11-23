"""Builds notebooks/GraphAML_step_by_step.ipynb (the Colab walkthrough) from the cells below.

Edit the text here, then:  python scripts/make_notebook.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
REPO = "namiqsarxanli-creator/graph-aml"

cells: list[tuple[str, str]] = []
md = lambda s: cells.append(("md", s.strip()))      # noqa: E731
code = lambda s: cells.append(("code", s.strip()))  # noqa: E731

md(f"""
# Pul yuyan şəbəkələri qraf ML ilə tapmaq: addım-addım

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/notebooks/GraphAML_step_by_step.ipynb)

Bank hər gün yüz minlərlə ödəniş görür. Onların təxminən **0,1%-i** çirkli pulun yuyulmasıdır. Analitiklər isə gündə
yalnız bir neçə yüz xəbərdarlığa (alert) baxa bilir. Sual belədir: **hansı əməliyyatları yoxlamağa göndərək?**

Pul yuyan bir əməliyyatın özünə baxanda o, adi ödəniş kimi görünür. Onu fərqləndirən şey ətrafdakı **şəbəkədir**:
pul 10 hesaba paylanır (fan-out), yenidən bir hesabda toplanır (scatter-gather) və ya dövrə vurub geri qayıdır (cycle).
Bu notebook həmin şəbəkəni modelə öyrədir və nəticəni bankın bu gün istifadə etdiyi qaydalarla müqayisə edir.

**Nə görəcəksiniz:**
1. Real (sintetik, amma realist) bank datası: IBM AML, 5 milyon əməliyyat
2. Pul yuyma sxemlərinin şəkilləri
3. Zamana görə bölgü və **leakage yoxlaması** (gələcək məlumat modelə sızmır)
4. 4 model: qaydalar, LightGBM, LightGBM + qraf feature-ləri, **qraf neyron şəbəkəsi (GNN)**
5. Test datası **bir dəfə** qiymətləndirilir: güvən intervalları ilə
6. Hansı sxem növünü hansı model tutur
7. Pul hesabı: gündə neçə alert-ə baxmaq sərfəlidir
8. Bir alert-in "pul izi" (money trail) və səbəbləri
9. Modeli saxlamaq və API-də işə salmaq

**Başlamazdan əvvəl:** `Runtime → Change runtime type → T4 GPU` seçin. GPU olmadan da işləyir, sadəcə GNN yavaş öyrənir.
""")

md("""
## 0. Qurulum

Layihəni Colab-a gətirməyin iki yolu var:
* **GitHub:** layihəni GitHub-a yükləyin və linki `REPO_URL`-a yazın.
* **Zip:** `REPO_URL` boş qalsın. `GraphAML.zip` faylını sol paneldəki 📁 Files bölməsinə sürükləyin
  (və ya hüceyrə işə düşəndə faylı seçin).
""")
code("""
import os, sys, subprocess, pathlib, zipfile

REPO_URL = "https://github.com/namiqsarxanli-creator/graph-aml.git"   # zip ilə işləmək üçün boş qoyun

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    ROOT = pathlib.Path("/content/GraphAML")
    if not ROOT.exists():
        if pathlib.Path("/content/GraphAML.zip").exists():   # uploaded through the Files panel
            zipfile.ZipFile("/content/GraphAML.zip").extractall("/content")
        elif REPO_URL:
            subprocess.run(["git", "clone", "-q", REPO_URL, str(ROOT)], check=True)
        else:
            from google.colab import files
            print("GraphAML.zip faylını seçin")
            uploaded = files.upload()
            zipfile.ZipFile(next(iter(uploaded))).extractall("/content")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements-colab.txt")], check=True)
else:
    ROOT = pathlib.Path.cwd().parent if pathlib.Path.cwd().name == "notebooks" else pathlib.Path.cwd()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from IPython.display import display, Markdown
from graphaml import business, colab, data, explain, features, metrics, pipeline, plots, synthetic
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
print("GPU:", colab.gpu_name())
""")

md("""
## 1. Data

`DATA = "ibm"` real təcrübədir: Kaggle-dan IBM AML datasını yükləyir (HI-Small: 5 milyon əməliyyat, ~0,1% pul yuyma).
Kaggle üçün açar lazımdır: Colab-ın sol panelində **🔑 Secrets** bölməsinə `KAGGLE_USERNAME` və `KAGGLE_KEY` əlavə edin
(kaggle.com → Settings → API → Create New Token).

`DATA = "synthetic"` hər şeyi 2-3 dəqiqədə yoxlamaq üçündür (eyni format, kiçik həcm).

IBM datası sintetikdir, amma bank simulyatoru ilə yaradılıb və hər pul yuyma cəhdinin **hansı sxem olduğu** məlumdur.
Real bank datasında bu etiketlər demək olar ki, heç vaxt olmur. Ona görə bu data sahədə standart benchmark-dır.
""")
code("""
DATA = "ibm"          # "ibm" və ya "synthetic"
SIZE = "HI-Small"     # IBM: HI-Small (5M) kifayətdir; HI-Medium (32M) üçün Colab Pro RAM lazımdır
NROWS = None          # sürətli sınaq üçün məsələn 1_000_000

if DATA == "ibm":
    trans_path, patterns_path = colab.download_ibm(SIZE)
    std = data.load_ibm(trans_path, patterns_path, nrows=NROWS)
    SOURCE = f"IBM AML {SIZE}"
else:
    std = synthetic.make_dataset(seed=0)
    SOURCE = "synthetic"

std, tail = data.drop_simulation_tail(std)
print(f"Simulyasiyanın 'quyruğu' çıxarıldı: {tail['days_dropped']} gün, {tail['rows_dropped']:,} əməliyyat "
      f"(onların {tail['laundering_share_in_dropped']:.0%}-i pul yuyma; bax 3-cü addım)")
print(f"{len(std):,} əməliyyat, {pd.concat([std.src_key, std.dst_key]).nunique():,} hesab, "
      f"{std.ts.min():%Y-%m-%d} → {std.ts.max():%Y-%m-%d}")
print(f"pul yuyma: {std.is_laundering.sum():,} əməliyyat ({std.is_laundering.mean():.3%}), "
      f"{std.loc[std.attempt_id >= 0, 'attempt_id'].nunique():,} sxem")
display(std.head(3))
""")
code("""
# Sxem növləri və ödəniş formatları: pul yuyanlar hansı kanalları sevir?
display(std[std.is_laundering == 1].pattern.replace("", "(etiketsiz)").value_counts().to_frame("əməliyyat"))
display(pd.crosstab(std.payment_format, std.is_laundering, normalize="columns").round(3)
          .rename(columns={0: "normal", 1: "pul yuyma"}))
""")

md("""
## 2. Sxemlər necə görünür?

Hər nöqtə bir hesabdır, hər ox bir ödənişdir. Bu formaları bir əməliyyatın sətrinə baxaraq görmək mümkün deyil.
Onları yalnız **qraf** göstərir. GNN-in işi məhz budur.
""")
code("""
fig = plots.scheme_examples(std)
""")

md("""
## 3. Zamana görə bölgü

Model keçmişdən öyrənib gələcəyi proqnozlaşdırmalıdır, ona görə bölgü təsadüfi deyil, **günlərə görə** aparılır:
ilk ~60% train, sonrakı ~20% validation (model seçimi), son ~20% test (yalnız bir dəfə baxılır).

**Data artefaktı.** IBM faylının son ~8 günündə (HI-Small: 09-11 → 09-18) adi bank fəaliyyəti dayanır və yalnız pul
yuyma sxemlərinin son addımları qalır: həmin günlərdə əməliyyatların ~60%-i pul yuymadır. Real bankda belə gün olmur.
Onlar test hissəsinə düşsəydi, "sakit gün = pul yuyma" qaydasını tutan istənilən model yaxşı görünərdi. Ona görə
1-ci addımda bu günlər çıxarılır (`data.drop_simulation_tail`).

**Real iş axını belədir:** hər səhər dünənki əməliyyatlar o anadək yığılmış bütün qrafla birlikdə qiymətləndirilir
və analitiklər ən riskli `k` alert-ə baxır. Modellər də məhz bu informasiyanı görür, artıq heç nə görmür.
""")
code("""
exp = pipeline.build_experiment(std)
fig = plots.daily_volume(exp.df, exp.split)
display(pd.DataFrame({s: {"gün": len(exp.days(s)), "əməliyyat": len(exp.rows(s)),
                          "pul yuyma": int(exp.y[exp.rows(s)].sum())} for s in ("train", "valid", "test")}).T)
""")

md("""
## 4. Feature-lər: üç informasiya səviyyəsi

| Səviyyə | Nə bilir | Kim istifadə edir |
|---|---|---|
| **Əməliyyat** | məbləğ, format, valyuta, saat | hamı |
| **Tarixçə** | hər iki hesabın **bu dəqiqədən əvvəlki** fəaliyyəti: say, məbləğ, 24 saatlıq partlayışlar, yeni qarşı tərəf, "pul gəldi və dərhal getdi" | qaydalar, LightGBM |
| **Qraf** | günün sonunadək olan qraf: hesabların dərəcəsi, pulu ötürmə payı, qonşuların xüsusiyyətləri (2 addım) | LightGBM+qraf, GNN |

**Leakage yoxlaması.** Aşağıdakı hüceyrə datanın sonunu silir və feature-ləri yenidən hesablayır. Əgər hər hansı
feature gələcəkdən məlumat götürsəydi, qalan sətirlərin dəyərləri dəyişərdi. Onlar dəyişmirsə, sızma yoxdur.
""")
code("""
print({"əməliyyat": len(exp.feats.edge_cols), "tarixçə": len(exp.feats.history_cols), "qraf": len(exp.feats.graph_cols)})

cut_day = exp.days("valid")[0]
cut = int(np.searchsorted(exp.df["t"].to_numpy(), cut_day * features.DAY))
past = features.prepare(std.iloc[:cut])
h_past = features.history_features(past)
g_past = features.graph_features(past)
same_h = np.allclose(h_past.to_numpy(), exp.feats.history.iloc[:cut].to_numpy(), equal_nan=True)
same_g = np.allclose(g_past.to_numpy(), exp.feats.graph.iloc[:cut].to_numpy(), equal_nan=True)
print(f"Gələcək silindi: tarixçə feature-ləri eynidir: {same_h}, qraf feature-ləri eynidir: {same_g}")
assert same_h and same_g
""")

md("""
## 5. Baseline-lar: qaydalar və LightGBM

* **Qaydalar** bankların bu gün işlətdiyi sistemə bənzəyir: "böyük məbləğ", "24 saatda 5+ ödəniş", "pul gəldi və getdi" və s.
  Etiketlərlə öyrədilmir.
* **LightGBM (tabular)**: əməliyyat və tarixçə feature-ləri.
* **LightGBM + qraf**: üstəlik qraf feature-ləri. Bu güclü baseline-dır: GNN onu keçmirsə, GNN lazım deyil.

Qiymətləndirmə vahidləri: `ap` (precision-recall altında sahə), `schemes@k` (gündə `k` alert ilə tutulan sxemlərin payı).
Bir sxemin bir əməliyyatını tutmaq kifayətdir: analitik qalanını qrafda izləyir.
""")
code("""
models = pipeline.train_models(exp, which=("rules", "lgbm_tabular", "lgbm_graph"))
K = exp.default_k()
# validation və test günləri bir dəfə qiymətləndirilir; test nəticələrinə 7-ci addıma qədər baxılmır
base_scores = pipeline.predict_models(models, exp.feats, exp.snaps, exp.days("valid") + exp.days("test"))
print(f"alert büdcəsi: gündə {K} əməliyyat (tipik günün 0,5%-i)")
display(pipeline.evaluate_split(exp, base_scores, "valid", K, n_boot=0).point.round(3))
""")
code("""
imp = models["lgbm_graph"].importance().head(15)
display(pd.DataFrame({"feature": imp.index, "izah": [explain.label(f) for f in imp.index],
                      "gain payı": (imp / models["lgbm_graph"].importance().sum()).round(3).values}))
""")

md("""
## 6. Qraf neyron şəbəkəsi (GNN)

Hər hesab başlanğıcda öz göstəricilərini (dərəcə, məbləğlər, ACH payı və s.) daşıyır. Hər qatda hesab **pul göndərdiyi**
və **pul aldığı** hesablardan ayrı-ayrı məlumat toplayır (istiqamət vacibdir: toplayıcı ilə paylayıcı fərqli şeylərdir).
3 qatdan sonra hər hesab 3 addım uzaqlıqdakı şəbəkəni "görür". Sonra hər əməliyyat üçün göndərənin vektoru,
alanın vektoru və əməliyyatın öz feature-ləri birləşdirilir və risk hesablanır.

Kod sadə PyTorch-dur (sparse matris vurması), xüsusi kitabxana tələb etmir. Hər gün üçün qraf yalnız **o günün sonunadək**
olan əməliyyatlardan qurulur.
""")
code("""
GNN_EPOCHS = 40 if DATA == "ibm" else 25
models.update(pipeline.train_models(exp, which=("gnn",), gnn_kwargs={"epochs": GNN_EPOCHS}))
print(models["gnn"].model)
fig = plots.training_curve(models["gnn"].history)
""")

md("""
## 7. Model seçimi (validation) və test (bir dəfə)

Seçim qaydası: validation-da ən yaxşı modeldən **statistik olaraq fərqlənməyən ən sadə model** seçilir.
Mürəkkəb model yalnız fərq səs-küydən böyük olanda qazanır. Sonra test datasına **bir dəfə** baxılır.

İntervallar **cluster bootstrap**-dır: bir sxemin əməliyyatları bir-biri ilə bağlı olduğu üçün bütöv sxemlər birlikdə
yenidən seçilir. Bütün modellər eyni nümunələr üzərində qiymətləndirilir, ona görə də fərqlərin öz intervalı var.
""")
code("""
scores = pipeline.predict_models(models, exp.feats, exp.snaps, exp.days("valid") + exp.days("test"), reuse=base_scores)
thr = pipeline.f1_thresholds(exp, scores)
val = pipeline.evaluate_split(exp, scores, "valid", K, thr, n_boot=200)
SERVED, WHY = metrics.simplest_not_worse(val, pipeline.MODEL_ORDER, "ap")
print("Seçilən model:", WHY)

test = pipeline.evaluate_split(exp, scores, "test", K, thr, n_boot=200)   # yeganə baxış
display(test.table().rename(index=pipeline.MODEL_LABELS))
display(Markdown("**Qaydalarla fərq (paired, 95% CI):**"))
display(test.vs_reference.rename(index=pipeline.MODEL_LABELS))
fig = plots.metric_bars(test, "ap", pipeline.MODEL_LABELS)
""")

md("""
## 8. Hansı sxemləri kim tutur?

Qrafın dəyəri burada görünür: fan-in, scatter-gather və cycle kimi sxemlər əməliyyatın sətrindən çox qrafın formasında gizlənir.
""")
code("""
te = exp.rows("test"); d = exp.df.iloc[te]
by_pattern = pd.DataFrame({m: metrics.recall_by_pattern(metrics.alerts(s[te], d.day, K), d.is_laundering, d.cluster, d.pattern)
                           for m, s in scores.items()})
display(by_pattern.round(2))
fig = plots.pattern_recall(by_pattern, K)
need = pd.DataFrame({m: {f"sxemlərin {int(q*100)}%-i": business.alerts_needed(s[te], d.day, d.cluster, d.is_laundering, q)
                         for q in (0.5, 0.8)} for m, s in scores.items()}).T
display(Markdown("**Gündə neçə alert lazımdır?** (az = analitiklərin vaxtı az gedir)")); display(need)
""")

md("""
## 9. Pul hesabı

Datada cərimə və ya bərpa olunan məbləğ yoxdur, ona görə də iqtisadiyyat açıq ssenaridir:
bir alert-ə baxmaq **15** (analitikin ~20 dəqiqəsi), bir sxemi aşkarlamaq **2000** dəyərindədir. Rəqəmləri dəyişib yenidən işə salın.
""")
code("""
econ = business.Economics(cost_per_alert=15, value_per_scheme=2000)
ks = np.unique(np.linspace(1, 20 * K, 60).astype(int))
curves = {m: business.value_curve(s[te], d.day, d.cluster, d.is_laundering, d.amount_usd, ks, econ) for m, s in scores.items()}
fig = plots.value_curves(curves, f"Test günləri, {SOURCE}")
best = pd.DataFrame({m: c.loc[c.net.idxmax(), ["alerts_per_day", "scheme_recall", "net_per_day"]] for m, c in curves.items()}).T
display(best.rename(columns={"alerts_per_day": "ən yaxşı alert/gün", "scheme_recall": "tutulan sxem payı",
                             "net_per_day": "gündəlik xalis dəyər"}).round(2))
""")

md("""
## 10. Bir alert-in hekayəsi

Seçilmiş modelin test günündəki ən riskli və həqiqətən pul yuyma olan alert-ini götürürük. Analitik bunu görəcək:
pulun hardan gəlib hara getdiyini, dövrələri və modelin səbəblərini.
""")
code("""
from graphaml.serving import Scorer  # noqa: F401  (API eyni kodu istifadə edir)
risk = scores[SERVED]
alert_mask = np.zeros(len(exp.df), bool); alert_mask[te] = metrics.alerts(risk[te], d.day, K)
cand = exp.df[alert_mask & (exp.df.is_laundering == 1)]
pref = cand[cand.pattern.isin(["CYCLE", "SCATTER-GATHER", "GATHER-SCATTER", "FAN-IN", "FAN-OUT"])]
pool = pref if len(pref) else (cand if len(cand) else exp.df[alert_mask])
row = pool.index[np.argmax(risk[pool.index])]
r = exp.df.loc[row]
print(f"{r.ts:%Y-%m-%d %H:%M}  {r.src_key} → {r.dst_key}  {r.amount_paid:,.2f} {r.pay_currency} {r.payment_format}  "
      f"risk={risk[row]:.3f}  həqiqət: {r.pattern}")

day_end = (int(r.day) + 1) * features.DAY
ego = explain.ego_network(exp.df, np.nan_to_num(risk), int(r.src), hops=2, max_edges=80, per_node=8, upto_t=day_end)
cycles = explain.cycles_through(ego, int(r.src))
keys = pd.concat([exp.df.set_index("src").src_key, exp.df.set_index("dst").dst_key]).groupby(level=0).first()
fig = plots.ego(ego, row, keys, cycles)
print("dövrələr:", [[keys[v] for v in c] for c in cycles[:3]] or "yoxdur")
C = models["lgbm_graph"].contributions(exp.feats.full.loc[[row]])
display(pd.DataFrame(explain.top_reasons(C.iloc[0], exp.feats.full.loc[row], 6)))
""")

md("""
## 11. Modeli saxlamaq və API-də işə salmaq

Aşağıdakı hüceyrə bütün modelləri `models/` qovluğuna yazır və zip kimi endirir. Zip-i layihənin `models/` qovluğuna açın,
sonra lokal olaraq:

```
docker compose up --build      →  http://127.0.0.1:8080
```

Demo səhifəsində alert növbəsi, pul izi qrafı və qaydalarla müqayisə görünür.
""")
code("""
from graphaml import artifact
meta = pipeline.make_metadata(exp, models, SOURCE, SERVED, WHY, K, thr, val, test)
artifact.save(models, meta)
pathlib.Path("reports").mkdir(exist_ok=True)
pathlib.Path(f"reports/results_{DATA}.md").write_text(
    f"# Results on {SOURCE}\\n\\n" + test.table().rename(index=pipeline.MODEL_LABELS).to_markdown()
    + "\\n\\n## Schemes caught by type\\n\\n" + by_pattern.round(2).to_markdown() + f"\\n\\nServed model: {WHY}\\n", encoding="utf-8")
print(open(f"reports/results_{DATA}.md", encoding="utf-8").read())

if IN_COLAB:
    with zipfile.ZipFile("/content/graphaml_models.zip", "w") as z:
        for f in ["models/aml_bundle.pkl", "models/model_meta.json", f"reports/results_{DATA}.md"]:
            z.write(f)
    from google.colab import files
    files.download("/content/graphaml_models.zip")
""")


def build() -> Path:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s) for k, s in cells]
    nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                      "language_info": {"name": "python"}, "accelerator": "GPU",
                      "colab": {"provenance": [], "gpuType": "T4"}}
    out = ROOT / "notebooks" / "GraphAML_step_by_step.ipynb"
    out.parent.mkdir(exist_ok=True)
    nbf.write(nb, out)
    return out


if __name__ == "__main__":
    print(build())
