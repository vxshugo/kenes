Threads: 4. WER/CER in %, normalised (lowercase, ё->е, no punctuation). RTF = processing time / audio duration (lower is better; 0.1 = 10x faster than real time).

| model | fleurs_ru WER / CER | fleurs_kk WER / CER | RTF (all sets) | peak RSS MB | size MB |
|---|---|---|---|---|---|
| gigaam-ml-ctc-pt | 2.4 / 0.4 (n=10) | 5.8 / 1.6 (n=10) | 0.0523 | 2004 | 842.3 |
| gigaam-ml-large-ctc-pt | 0.5 / 0.1 (n=10) | 5.8 / 1.7 (n=10) | 0.0964 | 4806 | 2233.2 |
| gigaam-ml-ctc-onnx-int8 | 3.3 / 0.5 (n=10) | 9.0 / 1.9 (n=10) | 0.0495 | 436 | 214.4 |
| gigaam-ml-large-ctc-onnx-int8 | 0.5 / 0.1 (n=10) | 5.8 / 1.7 (n=10) | 0.0931 | 950 | 564.2 |
| gigaam-ml-ctc-onnx-int8-ortfeat | 2.8 / 0.5 (n=10) | 8.4 / 1.8 (n=10) | 0.0256 | 459 | 214.4 |
| gigaam-ml-large-ctc-onnx-int8-ortfeat | 0.5 / 0.1 (n=10) | 6.5 / 1.8 (n=10) | 0.0536 | 958 | 564.2 |
| gigaam-v3-ru-ctc-onnx-int8 | 0.5 / 0.1 (n=10) | – | 0.0329 | 417 | 214.3 |
| whisper-kaz-rus-ct2-int8 | 7.3 / 1.1 (n=5) | 6.9 / 2.9 (n=5) | 1.5095 | 1630 | 785.9 |
| whisper-large-v3-turbo-ct2-int8 | 2.1 / 0.3 (n=5) | 15.5 / 3.6 (n=5) | 1.5842 | 2027 | 1546.5 |
| vosk-small-kz-0.42 | – | 25.2 / 11.8 (n=10) | 0.0707 | 284 | 102.0 |

| model | set | n | audio min | WER | CER | RTF | punct | upper | digits |
|---|---|---|---|---|---|---|---|---|---|
| gigaam-ml-ctc-pt | fleurs_ru | 10 | 1.9 | 2.36 | 0.41 | 0.06 | 0.0 | 0.0 | 0.0 |
| gigaam-ml-ctc-pt | fleurs_kk | 10 | 2.2 | 5.81 | 1.59 | 0.0457 | 0.0 | 0.0 | 0.0 |
| gigaam-ml-large-ctc-pt | fleurs_ru | 10 | 1.9 | 0.47 | 0.07 | 0.0931 | 0.0 | 0.0 | 0.0 |
| gigaam-ml-large-ctc-pt | fleurs_kk | 10 | 2.2 | 5.81 | 1.68 | 0.0993 | 0.0 | 0.0 | 0.0 |
| gigaam-ml-ctc-onnx-int8 | fleurs_ru | 10 | 1.9 | 3.3 | 0.54 | 0.0553 | 0.0 | 0.0 | 0.0 |
| gigaam-ml-ctc-onnx-int8 | fleurs_kk | 10 | 2.2 | 9.03 | 1.93 | 0.0445 | 0.0 | 0.0 | 0.0 |
| gigaam-ml-large-ctc-onnx-int8 | fleurs_ru | 10 | 1.9 | 0.47 | 0.07 | 0.0954 | 0.0 | 0.0 | 0.0 |
| gigaam-ml-large-ctc-onnx-int8 | fleurs_kk | 10 | 2.2 | 5.81 | 1.68 | 0.0911 | 0.0 | 0.0 | 0.0 |
| gigaam-ml-ctc-onnx-int8-ortfeat | fleurs_ru | 10 | 1.9 | 2.83 | 0.47 | 0.0272 | 0.0 | 0.0 | 0.0 |
| gigaam-ml-ctc-onnx-int8-ortfeat | fleurs_kk | 10 | 2.2 | 8.39 | 1.84 | 0.0242 | 0.0 | 0.0 | 0.0 |
| gigaam-ml-large-ctc-onnx-int8-ortfeat | fleurs_ru | 10 | 1.9 | 0.47 | 0.07 | 0.0505 | 0.0 | 0.0 | 0.0 |
| gigaam-ml-large-ctc-onnx-int8-ortfeat | fleurs_kk | 10 | 2.2 | 6.45 | 1.76 | 0.0563 | 0.0 | 0.0 | 0.0 |
| gigaam-v3-ru-ctc-onnx-int8 | fleurs_ru | 10 | 1.9 | 0.47 | 0.07 | 0.0329 | 0.0 | 0.0 | 0.0 |
| whisper-kaz-rus-ct2-int8 | fleurs_ru | 5 | 0.9 | 7.29 | 1.13 | 1.575 | 0.0 | 0.0 | 0.0 |
| whisper-kaz-rus-ct2-int8 | fleurs_kk | 5 | 0.9 | 6.9 | 2.91 | 1.4451 | 0.0 | 0.0 | 0.0 |
| whisper-large-v3-turbo-ct2-int8 | fleurs_ru | 5 | 0.9 | 2.08 | 0.28 | 0.8983 | 1.0 | 1.0 | 0.0 |
| whisper-large-v3-turbo-ct2-int8 | fleurs_kk | 5 | 0.9 | 15.52 | 3.59 | 2.2593 | 0.4 | 0.4 | 0.0 |
| vosk-small-kz-0.42 | fleurs_kk | 10 | 2.2 | 25.16 | 11.81 | 0.0707 | 0.0 | 0.0 | 0.0 |

Matched subset (first N utterances of each set, the ones every model ran): WER / CER

| model | fleurs_ru (n=5) | fleurs_kk (n=5) |
|---|---|---|
| gigaam-ml-ctc-pt | 1.0 / 0.1 | 3.5 / 2.2 |
| gigaam-ml-large-ctc-pt | 0.0 / 0.0 | 5.2 / 2.5 |
| gigaam-ml-ctc-onnx-int8 | 2.1 / 0.3 | 6.9 / 2.5 |
| gigaam-ml-large-ctc-onnx-int8 | 0.0 / 0.0 | 5.2 / 2.5 |
| gigaam-ml-ctc-onnx-int8-ortfeat | 2.1 / 0.3 | 6.9 / 2.5 |
| gigaam-ml-large-ctc-onnx-int8-ortfeat | 0.0 / 0.0 | 5.2 / 2.5 |
| gigaam-v3-ru-ctc-onnx-int8 | 0.0 / 0.0 | – |
| whisper-kaz-rus-ct2-int8 | 7.3 / 1.1 | 6.9 / 2.9 |
| whisper-large-v3-turbo-ct2-int8 | 2.1 / 0.3 | 15.5 / 3.6 |
| vosk-small-kz-0.42 | – | 32.8 / 16.4 |
