# BANKING77: Jev ve DeepSeek V4 Pro

Iki modele ayni 77 kategori ve ayni metinler verilir. Zero-shot siniflandirma
yapilir; gercek etiketler API'ye gonderilmez. DeepSeek dusunme kapali,
temperature=0 ve JSON cevap ile calisir. Gecersiz/kesilmis cevaplar yanlis
sayilir, modele ek cevap sansi verilmez.

## Kurulum

Python 3.10+, macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export TYPESAFE_API_KEY='jev-api-anahtarin'
export DEEPSEEK_API_KEY='deepseek-api-anahtarin'
```

`JEV_API_KEY` de desteklenir. Anahtarlar dosyalara kaydedilmez.
`JEV_AI_KEY` ve `DEEPSEEK_KEY` alternatif isimleri de desteklenir.
`TYPESAFE_API_KEY`/`JEV_API_KEY` resmi TypeSafe endpoint'ini; ekran
goruntusundeki bagimsiz jev-ai.pro servisinden alinmis `JEV_AI_KEY` ise
`https://jev-ai.pro/api/v1/systemone` endpoint'ini otomatik kullanir.
Anahtarlari proje kokundeki `.env` dosyasina da yazabilirsiniz; otomatik yuklenir.
Mevcut terminal ortam degiskenleri `.env` degerlerinden once gelir.

## Pilot ve tam test

```bash
# Her siniftan rastgele 2 ornek: model basina 154, toplam 308 istek
python compare.py --pilot

# Parametresiz: test bolumunun tamami, model basina 3080 ornek
python compare.py

# Anahtarsiz, ucretsiz veri/secim kontrolu: AI istegi yapilmaz
python compare.py --pilot --dry-run
```

Pilot varsayilan `--seed 42` ile tekrarlanabilir ve siniflar dengelidir.
Pilot ve tam test sirasiyla `results/pilot/` ve `results/full-test/` klasorlerine
yazilir; birbirlerinin kayitlarini kullanmazlar. `--output` ile degistirilebilir.
"Tam" varsayilan olarak 3080 kayitlik test bolumudur; egitim bolumu karistirilmaz.
10003 kayitlik egitim bolumu icin ayri `--split train` kullanilabilir.

HF deposundaki eski Python yukleyicisi modern datasets ile desteklenmedigi
icin ayni yukleyicinin isaret ettigi orijinal PolyAI CSV dosyalari kullanilir.
Veri icerigi parmak iziyle kontrol edilir. `--revision` orijinal
PolyAI-LDN/task-specific-datasets deposunun branch veya commit SHA'sidir.

## Bekleme ve devam etme

```bash
python compare.py --pilot --batch-size 20 --delay 0.3 --batch-delay 2
```

Her metin, her model icin ayri bir HTTP istegidir. Batch istemci tarafindadir.
Varsayilan `--concurrency 3` ile her saglayicida en fazla 3 ornek paralel
islenir. Gruplar arasinda 0.3 saniye, her 20 tamamlanan istekte ek 2 saniye
beklenir. Daha temkinli seri calisma icin `--concurrency 1` kullanilabilir.
429/408/5xx ve ag hatalarinda artan bekleme, jitter ve Retry-After uygulanir.
`--retries 5`, `--timeout 60` varsayilandir. Kimlik/kredi gibi diger hatalar durdurur.

Her cevap hemen diske yazilir. Ayni komutu tekrar calistirmak tamamlanmis
model-ornek ciftlerini atlar. Ctrl+C ve hata durumunda kismi rapor uretilir.
Ayni cikti klasoru icin eszamanli calisma engellenir. Model, veri, seed, fiyat
veya soru yapilandirmasi degisirse yeni `--output` gerekir.
Model alias'lari zamanla degisebilir; donen model adi da kaydedilir.

## Maliyet

Fiyat verilmezse maliyet N/A olur; tokenlar yine kaydedilir. Birimler USD/1M token.
Hesabinizin guncel paket ve zaman dilimi tarifesini kullanin:

```bash
python compare.py --pilot \
  --jev-input-price 0.427 \
  --deepseek-input-price 0.66 \
  --deepseek-cached-price 0.022 \
  --deepseek-output-price 1.98
```

Bu fiyatlar ornektir: Jev ekran goruntusundeki kucuk paket ve DeepSeek'in
20 Eylul 2026'da listelenen off-peak tarifesi. Program fiyatlari otomatik
guncellemez veya saat dilimine gore degistirmez. DeepSeek cache-hit, cache-miss
ve output tokenlari ayri hesaplanir. Cache bilgisi yoksa maliyet N/A kalir.
Tahminler fatura degildir; cevabi kaybolan/retry edilen istekler ek maliyet
yaratabilir. Pilot ve tam test ayri deneylerdir, pilot maliyeti tekrar kullanilmaz.

## Grafikler ve raporlar

Bitiste veya kesintide otomatik uretilir; API kullanmadan yeniden uretmek icin:

```bash
python plot_comparison.py results/pilot
python plot_comparison.py results/full-test
```

- `comparison.png` / `.svg`: accuracy, macro F1, gecersiz cevap orani,
  medyan/P95 API suresi ve tahmini maliyet. Accuracy cubuklari yuzde ve
  `dogru/toplam` oranini birlikte gosterir.
- `per_class_f1.png` / `.svg`: 77 sinif icin model bazinda F1; her cubugun
  ucunda yuzde degeri, grafigin altinda genel accuracy ve macro F1 ozeti vardir.
- `summary.json`: ortak orneklerin karsilastirmasi (`paired`), tum tamamlanan
  kayitlar (`all_completed`), tokenlar, API/retry/bekleme sureleri ve modeller.
- `jev.jsonl`, `deepseek.jsonl`: her ornegin tahmini, gercek etiketi, ham
  siniflandirma cevabi, token kullanimi, sure ve maliyet tahmini.
- `comparison.json`: secilen indeksler, etiketler, model/soru/fiyat yapilandirmasi.

Grafikler sadece iki modelin de tamamladigi ayni ornekleri kullanir. Kismi
calisma INCOMPLETE olarak isaretlenir. API suresi tum HTTP denemelerini kapsar;
yapay bekleme ve retry beklemesi ayridir. Basarisiz son istegin sure/maliyeti
kayitlara dahil olmayabilir. Token sayilari farkli tokenizer'lar nedeniyle
dogrudan maliyet olcusu degildir. Sinif basina 2 ornek, nihai kalite karari
icin yetersizdir; pilot sonuclari on kontroldur.

```bash
python -m unittest discover -s tests -v
```

Kaynaklar:
- https://docs.typesafe.ai/api
- https://api-docs.deepseek.com/guides/thinking_mode/
- https://api-docs.deepseek.com/quick_start/pricing/
- https://huggingface.co/datasets/PolyAI/banking77
