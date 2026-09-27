Threads: 4. WER/CER in %, normalised (lowercase, ё->е, no punctuation). RTF = processing time / audio duration (lower is better; 0.1 = 10x faster than real time).

| model | fleurs_ru WER / CER | fleurs_kk WER / CER | cv_kk WER / CER | codeswitch WER / CER | RTF (all sets) | peak RSS MB | size MB |
|---|---|---|---|---|---|---|---|
| gigaam-ml-ctc-pt | 3.7 / 0.7 | 5.2 / 1.6 | 10.7 / 2.3 | 5.3 / 1.0 | 0.1031 | 2067 | 842.3 |
| gigaam-ml-large-ctc-pt | 2.4 / 0.4 | 4.4 / 1.4 | 9.3 / 2.1 | 4.3 / 0.7 | 0.4561 | 4882 | 2233.2 |
| gigaam-ml-ctc-onnx-int8 | 4.0 / 0.7 | 6.1 / 1.6 | 8.8 / 2.0 | 7.5 / 2.3 | 0.0538 | 503 | 214.4 |
| gigaam-ml-large-ctc-onnx-int8 | 2.3 / 0.4 | 4.7 / 1.5 | 7.1 / 1.5 | 5.7 / 0.8 | 0.1378 | 1031 | 564.2 |
| gigaam-ml-ctc-onnx-int8-ortfeat | 4.2 / 0.7 | 5.8 / 1.6 | 10.1 / 2.4 | 5.1 / 1.0 | 0.0423 | 544 | 214.4 |
| gigaam-ml-large-ctc-onnx-int8-ortfeat | 2.2 / 0.4 | 4.5 / 1.4 | 8.8 / 2.0 | 4.4 / 0.7 | 0.0753 | 1051 | 564.2 |
| gigaam-v3-ru-ctc-onnx-int8 | 2.7 / 0.5 | – | – | – | 0.0444 | 419 | 214.3 |
| whisper-kaz-rus-ct2-int8 | 7.1 / 1.3 (n=20) | 8.2 / 3.2 (n=20) | 14.7 / 3.0 (n=20) | 25.2 / 15.7 | 1.4 | 1628 | 785.9 |
| whisper-large-v3-turbo-ct2-int8 | 1.5 / 0.6 (n=20) | 24.2 / 6.8 (n=20) | 53.5 / 25.9 (n=20) | 17.3 / 4.5 | 2.1092 | 2113 | 1546.5 |
| vosk-small-kz-0.42 | – | 20.8 / 7.0 | 26.3 / 8.6 | – | 0.1551 | 301 | 102.0 |
| gigaam-ml-ctc-onnx-fp32-self | 4.0 / 0.7 | 5.7 / 1.6 | 8.8 / 2.0 | 7.0 / 2.0 | 0.0726 | 1478 | 844.4 |

| model | set | n | audio min | WER | CER | RTF | punct | upper | digits |
|---|---|---|---|---|---|---|---|---|---|
| gigaam-ml-ctc-pt | fleurs_ru | 60 | 11.1 | 3.72 | 0.66 | 0.0501 | 0.0 | 0.0 | 0.0 |
| gigaam-ml-ctc-pt | fleurs_kk | 60 | 13.3 | 5.16 | 1.58 | 0.0606 | 0.0 | 0.0 | 0.0 |
| gigaam-ml-ctc-pt | cv_kk | 60 | 5.0 | 10.68 | 2.31 | 0.0779 | 0.0 | 0.0 | 0.0 |
| gigaam-ml-ctc-pt | codeswitch | 30 | 9.4 | 5.31 | 1.01 | 0.2389 | 0.0 | 0.0 | 0.0 |
| gigaam-ml-large-ctc-pt | fleurs_ru | 60 | 11.1 | 2.37 | 0.44 | 0.3753 | 0.0 | 0.0 | 0.0 |
| gigaam-ml-large-ctc-pt | fleurs_kk | 60 | 13.3 | 4.4 | 1.43 | 0.7929 | 0.0 | 0.0 | 0.0 |
| gigaam-ml-large-ctc-pt | cv_kk | 60 | 5.0 | 9.32 | 2.13 | 0.292 | 0.0 | 0.0 | 0.0 |
| gigaam-ml-large-ctc-pt | codeswitch | 30 | 9.4 | 4.3 | 0.74 | 0.1636 | 0.0 | 0.0 | 0.0 |
| gigaam-ml-ctc-onnx-int8 | fleurs_ru | 60 | 11.1 | 3.98 | 0.72 | 0.0473 | 0.0 | 0.0 | 0.0 |
| gigaam-ml-ctc-onnx-int8 | fleurs_kk | 60 | 13.3 | 6.12 | 1.65 | 0.0498 | 0.0 | 0.0 | 0.0 |
| gigaam-ml-ctc-onnx-int8 | cv_kk | 60 | 5.0 | 8.77 | 2.0 | 0.057 | 0.0 | 0.0 | 0.0 |
| gigaam-ml-ctc-onnx-int8 | codeswitch | 30 | 9.4 | 7.46 | 2.27 | 0.0657 | 0.0 | 0.0 | 0.0 |
| gigaam-ml-large-ctc-onnx-int8 | fleurs_ru | 60 | 11.1 | 2.28 | 0.43 | 0.1264 | 0.0 | 0.0 | 0.0 |
| gigaam-ml-large-ctc-onnx-int8 | fleurs_kk | 60 | 13.3 | 4.73 | 1.48 | 0.1386 | 0.0 | 0.0 | 0.0 |
| gigaam-ml-large-ctc-onnx-int8 | cv_kk | 60 | 5.0 | 7.12 | 1.48 | 0.1473 | 0.0 | 0.0 | 0.0 |
| gigaam-ml-large-ctc-onnx-int8 | codeswitch | 30 | 9.4 | 5.69 | 0.79 | 0.1451 | 0.0 | 0.0 | 0.0 |
| gigaam-ml-ctc-onnx-int8-ortfeat | fleurs_ru | 60 | 11.1 | 4.15 | 0.73 | 0.0373 | 0.0 | 0.0 | 0.0 |
| gigaam-ml-ctc-onnx-int8-ortfeat | fleurs_kk | 60 | 13.3 | 5.8 | 1.65 | 0.0385 | 0.0 | 0.0 | 0.0 |
| gigaam-ml-ctc-onnx-int8-ortfeat | cv_kk | 60 | 5.0 | 10.14 | 2.35 | 0.0468 | 0.0 | 0.0 | 0.0 |
| gigaam-ml-ctc-onnx-int8-ortfeat | codeswitch | 30 | 9.4 | 5.06 | 0.96 | 0.0511 | 0.0 | 0.0 | 0.0 |
| gigaam-ml-large-ctc-onnx-int8-ortfeat | fleurs_ru | 60 | 11.1 | 2.2 | 0.42 | 0.0651 | 0.0 | 0.0 | 0.0 |
| gigaam-ml-large-ctc-onnx-int8-ortfeat | fleurs_kk | 60 | 13.3 | 4.51 | 1.44 | 0.0797 | 0.0 | 0.0 | 0.0 |
| gigaam-ml-large-ctc-onnx-int8-ortfeat | cv_kk | 60 | 5.0 | 8.77 | 1.96 | 0.0894 | 0.0 | 0.0 | 0.0 |
| gigaam-ml-large-ctc-onnx-int8-ortfeat | codeswitch | 30 | 9.4 | 4.42 | 0.74 | 0.0737 | 0.0 | 0.0 | 0.0 |
| gigaam-v3-ru-ctc-onnx-int8 | fleurs_ru | 60 | 11.1 | 2.71 | 0.52 | 0.0444 | 0.0 | 0.0 | 0.0 |
| whisper-kaz-rus-ct2-int8 | fleurs_ru | 20 | 3.7 | 7.07 | 1.33 | 1.3092 | 0.0 | 0.0 | 0.0 |
| whisper-kaz-rus-ct2-int8 | fleurs_kk | 20 | 4.5 | 8.16 | 3.17 | 1.1293 | 0.0 | 0.0 | 0.0 |
| whisper-kaz-rus-ct2-int8 | cv_kk | 20 | 1.6 | 14.66 | 3.01 | 2.9613 | 0.0 | 0.0 | 0.0 |
| whisper-kaz-rus-ct2-int8 | codeswitch | 20 | 6.1 | 25.24 | 15.65 | 1.2341 | 0.0 | 0.0 | 0.0 |
| whisper-large-v3-turbo-ct2-int8 | fleurs_ru | 20 | 3.7 | 1.46 | 0.56 | 3.3359 | 1.0 | 1.0 | 0.05 |
| whisper-large-v3-turbo-ct2-int8 | fleurs_kk | 20 | 4.5 | 24.17 | 6.83 | 1.3294 | 0.4 | 0.4 | 0.0 |
| whisper-large-v3-turbo-ct2-int8 | cv_kk | 20 | 1.6 | 53.45 | 25.89 | 5.4803 | 0.55 | 0.6 | 0.0 |
| whisper-large-v3-turbo-ct2-int8 | codeswitch | 20 | 6.1 | 17.28 | 4.52 | 1.0293 | 0.15 | 0.15 | 0.0 |
| vosk-small-kz-0.42 | fleurs_kk | 60 | 13.3 | 20.84 | 7.03 | 0.133 | 0.0 | 0.0 | 0.0 |
| vosk-small-kz-0.42 | cv_kk | 60 | 5.0 | 26.3 | 8.57 | 0.2145 | 0.0 | 0.0 | 0.0 |
| gigaam-ml-ctc-onnx-fp32-self | fleurs_ru | 60 | 11.1 | 3.98 | 0.71 | 0.0725 | 0.0 | 0.0 | 0.0 |
| gigaam-ml-ctc-onnx-fp32-self | fleurs_kk | 60 | 13.3 | 5.69 | 1.61 | 0.0682 | 0.0 | 0.0 | 0.0 |
| gigaam-ml-ctc-onnx-fp32-self | cv_kk | 60 | 5.0 | 8.77 | 1.96 | 0.0679 | 0.0 | 0.0 | 0.0 |
| gigaam-ml-ctc-onnx-fp32-self | codeswitch | 30 | 9.4 | 6.95 | 2.0 | 0.0814 | 0.0 | 0.0 | 0.0 |

Matched subset (first N utterances of each set, the ones every model ran): WER / CER

| model | fleurs_ru (n=20) | fleurs_kk (n=20) | cv_kk (n=20) | codeswitch (n=20) |
|---|---|---|---|---|
| gigaam-ml-ctc-pt | 1.7 / 0.3 | 7.5 / 2.9 | 11.2 / 2.2 | 5.2 / 1.0 |
| gigaam-ml-large-ctc-pt | 0.5 / 0.1 | 5.7 / 2.5 | 10.3 / 2.5 | 3.9 / 0.8 |
| gigaam-ml-ctc-onnx-int8 | 2.2 / 0.3 | 9.1 / 3.0 | 11.2 / 2.0 | 8.3 / 3.0 |
| gigaam-ml-large-ctc-onnx-int8 | 0.5 / 0.1 | 6.0 / 2.5 | 8.6 / 1.5 | 5.2 / 0.8 |
| gigaam-ml-ctc-onnx-int8-ortfeat | 1.9 / 0.3 | 8.8 / 3.0 | 11.2 / 2.2 | 5.0 / 1.0 |
| gigaam-ml-large-ctc-onnx-int8-ortfeat | 0.5 / 0.1 | 5.7 / 2.5 | 10.3 / 2.3 | 4.5 / 0.8 |
| gigaam-v3-ru-ctc-onnx-int8 | 0.7 / 0.1 | – | – | – |
| whisper-kaz-rus-ct2-int8 | 7.1 / 1.3 | 8.2 / 3.2 | 14.7 / 3.0 | 25.2 / 15.7 |
| whisper-large-v3-turbo-ct2-int8 | 1.5 / 0.6 | 24.2 / 6.8 | 53.5 / 25.9 | 17.3 / 4.5 |
| vosk-small-kz-0.42 | – | 24.8 / 10.1 | 27.6 / 8.8 | – |
| gigaam-ml-ctc-onnx-fp32-self | 2.2 / 0.3 | 8.8 / 3.0 | 10.3 / 1.9 | 7.6 / 2.6 |

Code-switch WER split by the language of the reference half:

| model | kk half WER | ru half WER | 1st half WER | 2nd half WER |
|---|---|---|---|---|
| gigaam-ml-ctc-pt | 4.9 | 5.66 | 3.75 | 6.91 |
| gigaam-ml-large-ctc-pt | 5.45 | 3.3 | 3.75 | 4.86 |
| gigaam-ml-ctc-onnx-int8 | 5.72 | 8.96 | 4.25 | 10.74 |
| gigaam-ml-large-ctc-onnx-int8 | 6.81 | 4.72 | 5.25 | 6.14 |
| gigaam-ml-ctc-onnx-int8-ortfeat | 4.9 | 5.19 | 3.75 | 6.39 |
| gigaam-ml-large-ctc-onnx-int8-ortfeat | 5.45 | 3.54 | 3.75 | 5.12 |
| whisper-kaz-rus-ct2-int8 | 19.13 | 30.18 | 9.84 | 39.11 |
| whisper-large-v3-turbo-ct2-int8 | 16.96 | 17.54 | 16.8 | 17.71 |
| gigaam-ml-ctc-onnx-fp32-self | 5.45 | 8.25 | 4.0 | 9.97 |
