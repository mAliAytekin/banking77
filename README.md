# BANKING77 + Jev

Jev + DeepSeek V4 Pro karsilastirmasi icin [COMPARISON.md](COMPARISON.md).
`python compare.py --pilot` model basina dengeli 154 ornek,
`python compare.py` tum test bolumunu (3080 ornek/model) calistirir.
Asagidaki eski komut yalnizca Jev icindir.

Python 3.10+ ve macOS/Linux icin beklemeli, istemci tarafinda batch
siniflandirma. Resmi Jev `POST /v1/systemone` API'si kullanilir.
Her metin 77 secenekli bir Choice sorusuyla ayri istekte siniflandirilir.
HF deposundaki eski betik modern datasets ile desteklenmedigi icin, betigin
isaret ettigi orijinal PolyAI GitHub CSV dosyalari datasets ile yuklenir.
Varsayilan: test bolumu, 20 kayitlik gruplar, istekler arasinda 0.3 saniye,
gruplar arasinda ek 2 saniye. Paralel istek yapilmaz.

## Kurulum

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export TYPESAFE_API_KEY='kendi-api-anahtarin'
```

Anahtari kaynak koda yazmayin. `JEV_API_KEY` de desteklenir.

## Calistirma

API harcamasi yapmadan veri indirme ve istek onizlemesi:

```bash
python banking77_jev.py --dry-run --limit 5
```

Once 20 orneklik deneme, sonra ayni klasorde tum test bolumune devam:

```bash
python banking77_jev.py --limit 20
python banking77_jev.py --batch-size 20 --delay 0.3 --batch-delay 2
```

`--limit 0` tum bolumu secer. Her ornek bir API istegidir; batch-size,
tek istekte gonderilen metin sayisi degildir. Hiz ayarlari hesap limitinize
gore degistirilebilir. 429, 408, 5xx ve ag hatalarinda artan bekleme ve
jitter uygulanir; Retry-After saniye veya HTTP tarihi olarak dikkate alinir.
Diger HTTP hatalarinda calisma durur. Timeout sonrasi tekrar edilen istek
sunucuda zaten islenmis olabilir; tam olarak bir kez faturalama garantisi yoktur.

## Sonuclar

- `results/test/predictions.jsonl`: metin, gercek/tahmin etiketleri, confidence,
  olasiliklar, sunucunun model adi, token kullanimi ve sure.
- `results/test/metrics.json`: accuracy, 77 sinif uzerinden macro F1,
  sinif bazinda F1 ve kaydedilen input token toplami.
- `results/test/run.json`: veri parmak izi, model ve soru yapilandirmasi.

Her basarili sonuc hemen diske yazilir. Ctrl+C veya hata sonrasi ayni komut
tamamlanmis kayitlari atlar. Ayni cikti klasorunde eszamanli calisma engellenir.
Model/veri/soru degisirse farkli `--output` kullanin. `jev-latest` zamanla
degisebilir; tekrarlanabilir deneylerde mevcut sabit model adini `--model`,
PolyAI-LDN/task-specific-datasets commit SHA'sini `--revision` ile belirtin.

Gercek etiketler API'ye gonderilmez. Bu zero-shot bir baslangic deneyidir;
egitim/fine-tuning yapmaz. Kucuk ve sirali alt kumelerdeki skorlar tum test
basarisini temsil etmeyebilir. Kayitli token toplami, cevabi kaybolan veya
yeniden denenmis isteklerin maliyetini kapsamayabilir.

```bash
python -m unittest discover -s tests -v
```

Kaynaklar: https://docs.typesafe.ai/api ve
https://huggingface.co/datasets/PolyAI/banking77
