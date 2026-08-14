---
name: indextts-voice-api
description: "Use when calling IndexTTS Voice API for zero-shot TTS, voice cloning, Hermes TTS provider setup, or managing voice_id libraries. Covers /v1/health, /v1/voices, /v1/tts, and indextts-hermes-bridge."
version: 1.0.0
author: IndexTTS API
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [tts, indextts, voice-cloning, hermes, audio, speech]
    related_skills: []
prerequisites:
  commands: [curl, jq, base64]
---

# IndexTTS Voice API

## Overview

IndexTTS Voice API 是以 `voice_id` 為核心的 IndexTTS 2.5 HTTP 服務，預設監聽 `http://127.0.0.1:8001`。流程固定為：

1. 上傳 BASE64 WAV 參考音 → 取得 `voice_id`
2. 用 `voice_id` + 文字呼叫 `/v1/tts` 合成
3. （可選）透過 `indextts-hermes-bridge` 接上 Hermes command-type TTS provider

設定範例見本 skill 同 repo 的 `hermes/config.example.yaml`。

## When to Use

- 使用者要合成語音、音色克隆、註冊／列出／刪除 voice
- 要設定或除錯 Hermes Agent 的 IndexTTS TTS provider
- 提到 IndexTTS、`indextts-hermes-bridge`、`INDEXTTS_*` 環境變數

不要用本 skill 去跑原始 `indextts2` CLI 或 Gradio WebUI；那些不是這套 Voice API。

## Base URL 與環境變數

| 變數 | 預設 | 用途 |
| --- | --- | --- |
| `INDEXTTS_API_URL` | `http://127.0.0.1:8001` | API／bridge 基底 URL |
| `INDEXTTS_DEFAULT_VOICE_ID` | （空） | bridge 預設 `voice_id`（必填才能合成） |
| `INDEXTTS_USE_EMO_TEXT` | 關閉 | bridge：從文字推情感（`1`/`true`） |
| `INDEXTTS_EMO_ALPHA` | `0.6` | bridge：情感強度 `0.0–1.0` |
| `INDEXTTS_VOICES_DIR` | `./voices` | voice library 目錄 |
| `INDEXTTS_MODEL_DIR` | `./checkpoints` | 模型目錄（compose 把 `./checkpoints_2.5` 掛到這） |
| `INDEXTTS_CFG_PATH` | `./checkpoints/config.yaml` | 模型設定 |
| `INDEXTTS_USE_BF16` | `true` | BF16 推論 |
| `INDEXTTS_LANG` | `zh` | 合成語言 `zh`/`en`/`ja`/`ar`/`es` |
| `INDEXTTS_DURATION_FACTOR` | `1.0` | 語速 `0.5`（快）~ `2.0`（慢） |
| `INDEXTTS_TEXT_NORMALIZATION` | `true` | 文字正規化；關掉可保留繁體原文 |
| `INDEXTTS_USE_QWEN_EMO` | `true` | 載入 QwenEmotion；關掉省約 1.2GB VRAM 但 `use_emo_text` 失效 |
| `INDEXTTS_IDLE_UNLOAD_SEC` | `600` | 閒置這麼多秒沒推論就卸載模型、釋放 VRAM（`0` = 永不釋放） |
| `INDEXTTS_IDLE_CHECK_INTERVAL_SEC` | `30` | 閒置檢查間隔 |
| `INDEXTTS_LOG_LEVEL` | `INFO` | 服務日誌等級 |

## 模型版本

本服務只跑 **IndexTTS 2.5**（`indextts.infer_v2_5`），不再支援 2.0。與 2.0 的差異：
`lang` 為必填、多了 `duration_factor`、改用 bf16、輸出取樣率是 **22050 Hz**（2.0 為 24000 Hz）。
若下游有寫死 24000 Hz 的地方要一併改。

下載權重：`uv run python scripts/download_checkpoints.py --model-dir checkpoints_2.5`

模型是 lazy 載入的：服務啟動時不吃 VRAM，第一個 `/v1/tts` 才載入（實測 8~27 秒，
視 page cache 冷熱）。閒置超過 `INDEXTTS_IDLE_UNLOAD_SEC` 會自動釋放約 7GB VRAM，
之後的第一個請求要再等一次載入。
`/v1/health` 的 `model_ready` 與 `model_idle_sec` 可以看目前狀態；health check 本身
不會重置閒置計時。

解析 API URL：

1. `INDEXTTS_API_URL`（若有）
2. 否則 `http://127.0.0.1:8001`

Docker 內 Hermes、API 在宿主機時用 `http://host.docker.internal:8001`。

## 啟動服務

在 repo 根目錄（建議用 Compose，開機／中斷會自動重啟）：

```bash
docker compose up -d --build
```

或手動：

```bash
uv sync --extra hermes_api
uv run uvicorn indextts_api.main:app --host 0.0.0.0 --port 8001
```

先確認健康狀態（`model_ready` 在首次 TTS 前可能為 `false`，屬正常 lazy load）：

```bash
curl -s "${INDEXTTS_API_URL:-http://127.0.0.1:8001}/v1/health" | jq .
```

預期欄位：`status`、`model_ready`、`voices_count`、`indextts_available`、
`model_idle_sec`（模型未載入時為 `null`）、`idle_unload_sec`、`gpu`。

### GPU 狀態

`/v1/health` 會實際對 GPU 下一個小運算來驗證 CUDA context 還活著：

| `gpu.healthy` | 意義 | HTTP |
| --- | --- | --- |
| `null` | 尚未建立 CUDA context（模型未載入），正常 | `200` |
| `true` | GPU 正常 | `200` |
| `false` | **CUDA context 已損毀，服務必須重啟** | `503` |

`false` 時 `status` 為 `degraded`、`gpu.detail` 帶錯誤訊息，且回 **HTTP 503**——
docker healthcheck 用 `curl -fsS`，因此容器會轉為 `unhealthy`。
這是刻意設計：CUDA context 一旦中毒，模型物件還在、推論卻全數失敗，
若 health 照回 200 就會像 2026-08-12 那次連續 16.5 小時假裝健康。

**容器 `unhealthy` 不會自動重啟**（docker 的 restart policy 不看 healthcheck），
需要人工 `docker compose restart indextts-api`。

## API 速查

所有路徑前綴 `/v1`。`voice_id` 為 32 碼 hex。

### 建立 voice

參考音必須是 **WAV**，以 BASE64 送出：

```bash
API="${INDEXTTS_API_URL:-http://127.0.0.1:8001}"
AUDIO_B64=$(base64 -w0 ref.wav)

curl -s -X POST "$API/v1/voices" \
  -H "Content-Type: application/json" \
  -d "{\"audio_base64\": \"${AUDIO_B64}\", \"reference_text\": \"可選逐字稿\"}" | jq .
```

回傳：`voice_id`、`sample_rate`、`duration_sec`、`reference_text`、`created_at`。

### 列出／查詢／刪除

```bash
curl -s "$API/v1/voices" | jq .
curl -s "$API/v1/voices/<voice_id>" | jq .
curl -s -o /dev/null -w "%{http_code}\n" -X DELETE "$API/v1/voices/<voice_id>"
```

刪除成功為 `204`。

### 合成語音（audio）

```bash
curl -s -X POST "$API/v1/tts" \
  -H "Content-Type: application/json" \
  -d "{
    \"voice_id\": \"${INDEXTTS_DEFAULT_VOICE_ID}\",
    \"text\": \"你好，這是 IndexTTS 合成測試。\",
    \"lang\": \"zh\",
    \"duration_factor\": 1.0,
    \"use_emo_text\": false,
    \"emo_alpha\": 0.6,
    \"response_format\": \"audio\"
  }" \
  --output out.wav
```

成功：`Content-Type: audio/wav`，標頭含 `X-Sample-Rate`（**22050**）、`X-Duration-Sec`、`X-Request-Id`。

其他語言把 `lang` 換掉即可（`en`/`ja`/`ar`/`es`）：

```bash
curl -s -X POST "$API/v1/tts" \
  -H "Content-Type: application/json" \
  -d "{\"voice_id\": \"$VOICE_ID\", \"text\": \"Hello from IndexTTS.\", \"lang\": \"en\"}" \
  --output out_en.wav
```

保留繁體原文（關掉正規化）：

```bash
curl -s -X POST "$API/v1/tts" \
  -H "Content-Type: application/json" \
  -d "{\"voice_id\": \"$VOICE_ID\", \"text\": \"這是繁體原文。\", \"lang\": \"zh\", \"text_normalization\": false}" \
  --output out_raw.wav
```

實測差異（同一句「這是繁體字的煙霧測試，共 3 個項目。」）：

| `text_normalization` | 模型實際收到 |
| --- | --- |
| `true`（預設） | `这是繁体字的烟雾测试,共 三个项目.` |
| `false` | `這是繁體字的煙霧測試，共 3 個項目。` |

關掉後繁體保留，但**數字不再展開成唸法**（`3` 不會變「三」），阿拉伯數字、\
單位、符號都得自己寫成要唸的樣子。日常用途建議維持預設。

調語速（`0.5` 最快、`2.0` 最慢）：

```bash
curl -s -X POST "$API/v1/tts" \
  -H "Content-Type: application/json" \
  -d "{\"voice_id\": \"$VOICE_ID\", \"text\": \"慢慢說。\", \"lang\": \"zh\", \"duration_factor\": 1.5}" \
  --output out_slow.wav
```

### 合成語音（json）

`response_format: "json"` 回傳：

```json
{
  "audio_base64": "...",
  "sample_rate": 22050,
  "duration_sec": 1.23
}
```

### TTS 參數

| 欄位 | 預設 | 說明 |
| --- | --- | --- |
| `voice_id` | 必填 | 已註冊音色 |
| `text` | 必填 | 合成文字（非空） |
| `lang` | `zh` | 合成語言 `zh`/`en`/`ja`/`ar`/`es`；其他值回 `422` |
| `duration_factor` | `1.0` | 語速／時長 `0.5`（快）~ `2.0`（慢） |
| `text_normalization` | `true` | 數字轉唸法＋標點清理，中文會轉簡體；`false` 保留原文 |
| `use_emo_text` | `false` | 依文字內容推情感（需 `INDEXTTS_USE_QWEN_EMO=true`） |
| `emo_alpha` | `0.6` | 情感強度，範圍 `0.0–1.0` |
| `temperature` | `0.8` | GPT 採樣溫度 |
| `top_p` / `top_k` | `0.7` / `30` | 採樣參數 |
| `response_format` | `audio` | `audio`（WAV bytes）或 `json` |

## 標準工作流（agent 照做）

```
Task Progress:
- [ ] 1. GET /v1/health，確認 API 可連
- [ ] 2. 若尚無 voice：POST /v1/voices 上傳 WAV，記下 voice_id
- [ ] 3. export INDEXTTS_DEFAULT_VOICE_ID=<voice_id>
- [ ] 4. POST /v1/tts 合成，寫出 .wav
- [ ] 5. （可選）驗證 bridge / 更新 Hermes config
```

註冊並合成的一鍵範例：

```bash
API="${INDEXTTS_API_URL:-http://127.0.0.1:8001}"
AUDIO_B64=$(base64 -w0 ref.wav)
VOICE_ID=$(curl -s -X POST "$API/v1/voices" \
  -H "Content-Type: application/json" \
  -d "{\"audio_base64\": \"${AUDIO_B64}\"}" | jq -r .voice_id)
export INDEXTTS_DEFAULT_VOICE_ID="$VOICE_ID"

curl -s -X POST "$API/v1/tts" \
  -H "Content-Type: application/json" \
  -d "{\"voice_id\": \"$VOICE_ID\", \"text\": \"測試語音。\", \"lang\": \"zh\", \"response_format\": \"audio\"}" \
  --output /tmp/indextts-out.wav
```

## Hermes Agent TTS 整合

### 1. 設定 provider

將 repo 內 `hermes/config.example.yaml` 合併進 `~/.hermes/config.yaml`：

```yaml
tts:
  provider: indextts
  providers:
    indextts:
      type: command
      command: "indextts-hermes-bridge --text-file {input_path} --out {output_path}"
      output_format: wav
      timeout: 180
      voice_compatible: true
      max_text_length: 5000
```

Docker 內 Hermes：

```yaml
command: "indextts-hermes-bridge --api-url http://host.docker.internal:8001 --text-file {input_path} --out {output_path}"
```

### 2. 環境變數（必須）

```bash
export INDEXTTS_API_URL=http://127.0.0.1:8001
export INDEXTTS_DEFAULT_VOICE_ID=<32-hex-voice-id>
# 可選情感：
# export INDEXTTS_USE_EMO_TEXT=1
# export INDEXTTS_EMO_ALPHA=0.6
```

### 3. 驗證 bridge

```bash
echo "你好，這是 Hermes bridge 測試。" > /tmp/tts.txt
uv run indextts-hermes-bridge --text-file /tmp/tts.txt --out /tmp/bridge-out.wav
```

或在已 `uv sync --extra hermes_api` 的環境直接呼叫 entry point。缺少 `INDEXTTS_DEFAULT_VOICE_ID` 時 exit code 為 `2`。

Bridge 行為：讀 UTF-8 文字檔 → `POST /v1/tts`（`response_format=audio`）→ 寫出 WAV。逾時預設 180 秒。

## 錯誤對照

| 狀況 | 處理 |
| --- | --- |
| 連線失敗 | 確認 uvicorn 在 8001；Docker 改 `host.docker.internal` |
| `400` audio_base64 | 必須是合法 BASE64 的 WAV，不是 mp3/路徑字串 |
| `404` voice_id | 先 `GET /v1/voices`；id 須為 32 hex |
| `422` lang 不支援 | 只接受 `zh`/`en`/`ja`/`ar`/`es` |
| `422` duration_factor 超界 | 範圍 `0.5`–`2.0` |
| `500` TTS 推論失敗 | 回應會帶 `request_id`，用它撈日誌：`docker logs indextts-api \| grep <id>` |
| `503` health / 全部請求都 500 | 先看 `gpu.healthy`；為 `false` 代表 CUDA context 損毀，`docker compose restart indextts-api` |
| bridge exit `2` | 設定 `INDEXTTS_DEFAULT_VOICE_ID` 或傳 `--voice-id` |
| 首次 TTS 很慢 | lazy load 模型，屬預期 |

## Common Pitfalls

1. **把檔案路徑當成 audio_base64** — 必須先 `base64 -w0 ref.wav`。
2. **忘了註冊 voice 就合成** — bridge／API 都需要既有 `voice_id`。
3. **port 搞混** — 本 API 預設 **8001**（不是 8000）。
4. **用非 WAV 參考音** — decoder 只接受 WAV。
5. **在 Hermes 裡改 provider 後沒設環境變數** — 沒有 default voice_id 會直接失敗。
6. **以為 health 的 `model_ready: false` 代表壞掉** — 第一次成功 TTS 後才會變 `true`；
   閒置釋放後也會變回 `false`，同樣正常。
7. **假設取樣率是 24000 Hz** — IndexTTS 2.5 輸出 **22050 Hz**，別在下游寫死。
8. **關掉 `text_normalization` 又送阿拉伯數字** — 數字不會展開成唸法，要自己寫「三」。
9. **忘了帶 `lang`** — 不帶會用 `INDEXTTS_LANG`（預設 `zh`）；中文文字配 `lang=en` 會念得很怪。
10. **`use_emo_text` 沒作用或報錯** — 需要 `INDEXTTS_USE_QWEN_EMO=true`（預設開啟）。

## Verification Checklist

- [ ] `GET /v1/health` 回 `status: ok`（且 `gpu.healthy` 不是 `false`）
- [ ] `POST /v1/voices` 回 201 與 32-hex `voice_id`
- [ ] `POST /v1/tts` 產出可播放的 `.wav`（`X-Sample-Rate: 22050`）
- [ ] `lang` 換成 `en` 也能合成；`lang=de` 回 `422`
- [ ] `INDEXTTS_DEFAULT_VOICE_ID` 已設定
- [ ] （若接 Hermes）`indextts-hermes-bridge` 能寫出非空 WAV
- [ ] （若接 Hermes）`~/.hermes/config.yaml` 的 `tts.provider` 為 `indextts`

## One-Shot Recipes

### A. 快速健康檢查

```bash
curl -sf "${INDEXTTS_API_URL:-http://127.0.0.1:8001}/v1/health" | jq .
```

### B. 情感語音

```bash
curl -s -X POST "${INDEXTTS_API_URL:-http://127.0.0.1:8001}/v1/tts" \
  -H "Content-Type: application/json" \
  -d "{
    \"voice_id\": \"$INDEXTTS_DEFAULT_VOICE_ID\",
    \"text\": \"今天真的太開心了！\",
    \"use_emo_text\": true,
    \"emo_alpha\": 0.8,
    \"response_format\": \"audio\"
  }" --output emo.wav
```

### C. JSON 回傳再解碼

```bash
curl -s -X POST "${INDEXTTS_API_URL:-http://127.0.0.1:8001}/v1/tts" \
  -H "Content-Type: application/json" \
  -d "{\"voice_id\": \"$INDEXTTS_DEFAULT_VOICE_ID\", \"text\": \"json 模式\", \"response_format\": \"json\"}" \
  | jq -r .audio_base64 | base64 -d > from-json.wav
```
