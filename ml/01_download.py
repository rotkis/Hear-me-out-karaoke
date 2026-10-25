"""Baixa o EMVD do Zenodo (record 8406322) e extrai os .tar.xz em ./EMVD."""
import glob
import subprocess
import tarfile

from common import AUDIO_DIR, EMVD_DIR

DOI = "10.5281/zenodo.8406322"


def main():
    EMVD_DIR.mkdir(exist_ok=True)
    subprocess.run(["zenodo_get", DOI, "-o", str(EMVD_DIR)], check=True)
    for f in glob.glob(str(EMVD_DIR / "*.tar.xz")):
        print("extraindo", f)
        with tarfile.open(f, "r:xz") as t:
            t.extractall(EMVD_DIR)
    if not AUDIO_DIR.exists():
        print("ATENCAO: EMVD/audio nao existe. Conteudo de EMVD:")
        for p in sorted(EMVD_DIR.iterdir()):
            print(" ", p.name)
    else:
        print("ok:", len(list(AUDIO_DIR.glob("*"))), "arquivos em EMVD/audio")
    print("Confira tambem se o record mais novo (17813678, citado no README do repo) traz metadados atualizados.")


if __name__ == "__main__":
    main()
