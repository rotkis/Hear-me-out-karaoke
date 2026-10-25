"""Exploracao dos dados: colunas, contagens por classe e por cantor, duracao."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import CLASSES, EMVD_DIR, OUT_DIR, load_metadata


def main():
    OUT_DIR.mkdir(exist_ok=True)
    raw = pd.read_csv(EMVD_DIR / "metadata_files.csv")
    print("Colunas de metadata_files.csv:", list(raw.columns))
    print("Linhas:", len(raw))
    for c in ["type", "name", "range", "vowel", "authors_rank"]:
        if c in raw.columns:
            print(f"\n{c}:", raw[c].value_counts(dropna=False).to_dict())
    sp = EMVD_DIR / "metadata_singers.csv"
    if sp.exists():
        s = pd.read_csv(sp)
        print("\nmetadata_singers.csv:", list(s.columns), "| cantores:", len(s))

    df = load_metadata(verbose=True)
    dur = "duration(s)"
    if dur not in df.columns:
        raise SystemExit("coluna duration(s) ausente; ajuste o script")

    by_class = (
        df.groupby("name")
        .agg(arquivos=("file_name", "count"), cantores=("singer_id", "nunique"), minutos=(dur, lambda x: x.sum() / 60))
        .reindex(CLASSES)
    )
    print("\n=== Por classe ===\n", by_class.round(1))
    by_class.to_csv(OUT_DIR / "eda_por_classe.csv")

    pivot = df.pivot_table(index="singer_id", columns="name", values="file_name", aggfunc="count", fill_value=0).reindex(columns=CLASSES, fill_value=0)
    pivot.to_csv(OUT_DIR / "eda_cantor_x_classe.csv")
    print("\n=== Cantores sem alguma classe ===")
    for c in CLASSES:
        faltam = pivot.index[pivot[c] == 0].tolist()
        print(f"{c}: {len(faltam)} cantores sem a classe {faltam[:10]}")
    print("\nArquivos por cantor: min/mediana/max =", pivot.sum(axis=1).min(), pivot.sum(axis=1).median(), pivot.sum(axis=1).max())
    print("Duracao por arquivo (s): min/mediana/max =", df[dur].min(), df[dur].median(), df[dur].max())

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    by_class["minutos"].plot.bar(ax=ax[0], title="Minutos por classe")
    im = ax[1].imshow(pivot.values, aspect="auto")
    ax[1].set_title("Arquivos: cantor x classe")
    ax[1].set_xticks(range(len(CLASSES)))
    ax[1].set_xticklabels(CLASSES, rotation=45, ha="right")
    fig.colorbar(im, ax=ax[1])
    fig.tight_layout()
    fig.savefig(OUT_DIR / "eda.png", dpi=120)
    print("\nSalvos em", OUT_DIR)


if __name__ == "__main__":
    main()
