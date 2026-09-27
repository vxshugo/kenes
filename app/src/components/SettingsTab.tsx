import { useEffect, useMemo, useRef, useState } from "react";
import { formatDateTime } from "../lib/format";
import { describeError } from "../llm/client";
import type { ControllerState } from "../session/controller";
import { controller } from "../session/useController";
import { defaultAutoHintMode, parseMyNames } from "../types";
import type { AnswerLanguage, AutoHintMode, DeviceInfo, HintEffort, MicMode, ModelInfo, Settings, SummaryEffort } from "../types";
import { IconEye, IconEyeOff, IconMic, IconTrash } from "./Icons";

const MODEL_SUGGESTIONS = ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"];

const HINT_EFFORTS: Array<{ v: HintEffort; label: string }> = [
  { v: "low", label: "Низкое (быстро)" },
  { v: "medium", label: "Среднее" },
  { v: "high", label: "Высокое (вдумчиво)" },
];
const SUMMARY_EFFORTS: Array<{ v: SummaryEffort; label: string }> = [
  { v: "low", label: "Низкое" },
  { v: "medium", label: "Среднее" },
  { v: "high", label: "Высокое" },
  { v: "xhigh", label: "Очень высокое" },
];
const AUTO_HINT_MODES: Array<{ v: AutoHintMode; label: string }> = [
  { v: "addressed", label: "Когда обращаются ко мне" },
  { v: "any", label: "На любой вопрос" },
  { v: "off", label: "Выключены" },
];
const MIC_MODES: Array<{ v: MicMode; label: string }> = [
  { v: "me", label: "Только меня (гарнитура, наушники)" },
  { v: "room", label: "Весь зал (спикерфон, микрофон на столе)" },
];
const LANGS: Array<{ v: AnswerLanguage; label: string }> = [
  { v: "auto", label: "Как в вопросе (казахский — с переводом)" },
  { v: "ru", label: "Всегда по-русски" },
  { v: "kk", label: "Всегда по-казахски (с переводом)" },
];

function maskKey(key: string): string {
  return key.length > 12 ? `${key.slice(0, 7)}…${key.slice(-4)}` : "••••";
}

function ApiKeySection({ state }: { state: ControllerState }) {
  const [value, setValue] = useState("");
  const [show, setShow] = useState(false);
  const [msg, setMsg] = useState<{ text: string; tone: "ok" | "error" } | null>(null);
  const [busy, setBusy] = useState(false);

  const save = async (key: string) => {
    setBusy(true);
    setMsg(null);
    try {
      await controller.setApiKey(key);
      setValue("");
      setMsg({ text: key ? "Ключ сохранён." : "Ключ удалён.", tone: "ok" });
    } catch (e) {
      setMsg({ text: describeError(e), tone: "error" });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="field">
      <span className="field-label">
        API-ключ Claude
        <span className="field-hint">
          {state.apiKey ? `сохранён: ${maskKey(state.apiKey)}` : "не задан"}
          {state.backendKind === "mock" ? " · в демо-режиме хранится в localStorage" : ""}
        </span>
      </span>
      <div className="input-row">
        <input
          type={show ? "text" : "password"}
          value={value}
          placeholder={state.apiKey ? "Ввести новый ключ" : "sk-ant-…"}
          autoComplete="off"
          spellCheck={false}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && value.trim()) {
              e.preventDefault();
              void save(value.trim());
            }
          }}
          aria-label="API-ключ Claude"
        />
        <button type="button" className="icon-btn" onClick={() => setShow((s) => !s)} title={show ? "Скрыть" : "Показать"} aria-label={show ? "Скрыть ключ" : "Показать ключ"}>
          {show ? <IconEyeOff /> : <IconEye />}
        </button>
        <button type="button" className="btn btn-sm" disabled={busy || !value.trim()} onClick={() => void save(value.trim())}>
          Сохранить ключ
        </button>
        {state.apiKey && (
          <button type="button" className="btn btn-ghost btn-sm" disabled={busy} onClick={() => void save("")}>
            Удалить
          </button>
        )}
      </div>
      {msg && <span className={msg.tone === "ok" ? "ok small" : "card-error small"}>{msg.text}</span>}
      <span className="field-hint">Ключ уходит только на api.anthropic.com. На сервер Claude отправляется текст расшифровки, звук остаётся на компьютере.</span>
    </div>
  );
}


const ENROLL_SECONDS = 20;
/** Read aloud while the voiceprint is recorded: neutral Russian plus one Kazakh sentence. */
const ENROLL_TEXT_RU =
  "Сегодня мы обсуждаем план работ на следующую неделю. Я коротко расскажу, что уже сделано, какие есть риски и что нужно согласовать. Если появятся вопросы, задавайте их сразу — так мы сэкономим время. Начнём с бюджета, потом перейдём к срокам и в конце распределим задачи между командами.";
const ENROLL_TEXT_KK = "Бүгін біз келесі аптаның жоспарын бірге талқылаймыз. Барлық сұрақтарды жиналыстың соңында қарастырамыз.";
/** Less speech than this makes a weak voiceprint. */
const MIN_GOOD_SPEECH_MS = 12_000;

/** «Мой голос»: record a 20-second voiceprint so the diarizer can label the user «Я» in a room. */
function VoiceSection({ state }: { state: ControllerState }) {
  const [left, setLeft] = useState<number | null>(null);
  const [msg, setMsg] = useState<{ text: string; tone: "ok" | "warn" | "error" } | null>(null);
  const [busy, setBusy] = useState(false);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);
  const sessionBusy = ["starting", "loading", "running", "stopping"].includes(state.phase);
  const recording = state.enrolling || left !== null;
  const vp = state.voiceprint;

  useEffect(() => () => {
    if (timer.current) clearInterval(timer.current);
  }, []);

  const record = async () => {
    setMsg(null);
    const started = Date.now();
    setLeft(ENROLL_SECONDS);
    timer.current = setInterval(() => setLeft(Math.max(0, ENROLL_SECONDS - Math.floor((Date.now() - started) / 1000))), 200);
    try {
      const r = await controller.enrollVoice(ENROLL_SECONDS);
      const sec = (r.speechMs / 1000).toLocaleString("ru-RU", { maximumFractionDigits: 1 });
      setMsg(
        r.speechMs < MIN_GOOD_SPEECH_MS
          ? { text: `Образец сохранён, но речи мало: ${sec} с. Лучше перезаписать — говорите ближе к микрофону, без пауз.`, tone: "warn" }
          : { text: `Готово: образец сохранён, речи — ${sec} с.`, tone: "ok" },
      );
    } catch (e) {
      setMsg({ text: `Не удалось записать образец: ${describeError(e)}`, tone: "error" });
    } finally {
      if (timer.current) clearInterval(timer.current);
      timer.current = null;
      setLeft(null);
    }
  };

  const remove = async () => {
    setBusy(true);
    setMsg(null);
    try {
      await controller.clearVoiceprint();
      setMsg({ text: "Образец голоса удалён.", tone: "ok" });
    } catch (e) {
      setMsg({ text: describeError(e), tone: "error" });
    } finally {
      setBusy(false);
    }
  };

  return (
    <fieldset disabled={sessionBusy} className="voice">
      <legend>Мой голос</legend>
      <p className="field-hint voice-why">
        В зале один микрофон слышит всех, и без образца приложение не знает, какой голос ваш. Запишите 20 секунд своей речи — тогда ваши
        реплики будут подписаны «Я», и авто-подсказки не будут срабатывать на ваши собственные вопросы. Образец хранится только на этом
        компьютере.
      </p>
      <p className="voice-status small">
        {vp === null ? "Статус образца неизвестен." : vp.enrolled ? `Образец записан${vp.createdAt ? ` ${formatDateTime(vp.createdAt)}` : ""}.` : "Образец не записан."}
      </p>
      {recording && (
        <div className="voice-card" role="status" aria-live="polite">
          <div className="voice-countdown">
            <span className="rec-dot" aria-hidden="true" /> Читайте вслух · {left ?? 0} с
          </div>
          <p className="voice-text">{ENROLL_TEXT_RU}</p>
          <p className="voice-text voice-text-kk">{ENROLL_TEXT_KK}</p>
          <div className="voice-progress" aria-hidden="true">
            <span style={{ transform: `scaleX(${left === null ? 1 : (ENROLL_SECONDS - left) / ENROLL_SECONDS})` }} />
          </div>
        </div>
      )}
      <div className="input-row">
        <button type="button" className="btn btn-sm" onClick={() => void record()} disabled={recording || busy || sessionBusy}>
          <IconMic /> {vp?.enrolled ? "Перезаписать 20 секунд" : "Записать 20 секунд"}
        </button>
        {vp?.enrolled && (
          <button type="button" className="btn btn-ghost btn-sm" onClick={() => void remove()} disabled={recording || busy || sessionBusy}>
            <IconTrash /> Удалить образец
          </button>
        )}
      </div>
      {!recording && !msg && <span className="field-hint">После нажатия появится текст — прочитайте его обычным голосом, как на встрече.</span>}
      {msg && <span className={msg.tone === "ok" ? "ok small" : msg.tone === "warn" ? "warn-text small" : "card-error small"}>{msg.text}</span>}
      {sessionBusy && <span className="field-hint">Недоступно во время записи встречи.</span>}
    </fieldset>
  );
}

export function SettingsTab({ state }: { state: ControllerState }) {
  const [draft, setDraft] = useState<Settings>(state.settings);
  const [devices, setDevices] = useState<DeviceInfo[] | null>(null);
  const [models, setModels] = useState<ModelInfo[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saveMsg, setSaveMsg] = useState<{ text: string; tone: "ok" | "error" } | null>(null);
  const [saving, setSaving] = useState(false);
  // Free text for myNames; parsed into the draft on every edit, re-derived only on reset.
  const [namesText, setNamesText] = useState(state.settings.myNames.join(", "));

  useEffect(() => {
    setDraft(state.settings);
    setNamesText(state.settings.myNames.join(", "));
  }, [state.settings]);

  useEffect(() => {
    if (!state.ready || state.initError) return;
    controller.api.listDevices().then(setDevices).catch((e) => setLoadError(`Устройства: ${describeError(e)}`));
    controller.api.listModels().then(setModels).catch((e) => setLoadError(`Модели: ${describeError(e)}`));
  }, [state.ready, state.initError]);

  const dirty = useMemo(() => JSON.stringify(draft) !== JSON.stringify(state.settings), [draft, state.settings]);
  const set = <K extends keyof Settings>(k: K, v: Settings[K]) => setDraft((d) => ({ ...d, [k]: v }));
  const inputs = (devices ?? []).filter((d) => d.kind === "input");
  const monitors = (devices ?? []).filter((d) => d.kind === "monitor");
  const sessionLive = ["starting", "loading", "running"].includes(state.phase);

  const save = async () => {
    setSaving(true);
    setSaveMsg(null);
    try {
      await controller.saveSettings({ ...draft, claudeModel: draft.claudeModel.trim() || "claude-opus-5" });
      setSaveMsg({ text: sessionLive ? "Сохранено. Настройки звука применятся со следующей записи." : "Сохранено.", tone: "ok" });
    } catch (e) {
      setSaveMsg({ text: describeError(e), tone: "error" });
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="tab-pane settings">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          void save();
        }}
      >
        <fieldset>
          <legend>Claude</legend>
          <ApiKeySection state={state} />
          <label className="field">
            <span className="field-label">
              Модель
              <span className="field-hint">смена модели посреди встречи сбрасывает кэш промпта</span>
            </span>
            <input type="text" list="claude-models" value={draft.claudeModel} spellCheck={false} onChange={(e) => set("claudeModel", e.target.value)} />
            <datalist id="claude-models">
              {MODEL_SUGGESTIONS.map((m) => (
                <option key={m} value={m} />
              ))}
            </datalist>
          </label>
          <div className="field-grid">
            <label className="field">
              <span className="field-label">Усилие подсказок</span>
              <select value={draft.hintEffort} onChange={(e) => set("hintEffort", e.target.value as HintEffort)}>
                {HINT_EFFORTS.map((o) => (
                  <option key={o.v} value={o.v}>
                    {o.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span className="field-label">Усилие итогов</span>
              <select value={draft.summaryEffort} onChange={(e) => set("summaryEffort", e.target.value as SummaryEffort)}>
                {SUMMARY_EFFORTS.map((o) => (
                  <option key={o.v} value={o.v}>
                    {o.label}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <label className="field">
            <span className="field-label">
              Как ко мне обращаются
              <span className="field-hint">через запятую; падежи узнаются сами («Хугоға», «Хуго, а вы…»)</span>
            </span>
            <input
              type="text"
              value={namesText}
              placeholder="Например: Хуго, Hugo"
              spellCheck={false}
              onChange={(e) => {
                setNamesText(e.target.value);
                set("myNames", parseMyNames(e.target.value));
              }}
            />
          </label>
          <label className="field">
            <span className="field-label">Авто-подсказки</span>
            <select value={draft.autoHintMode} onChange={(e) => set("autoHintMode", e.target.value as AutoHintMode)}>
              {AUTO_HINT_MODES.map((o) => (
                <option key={o.v} value={o.v}>
                  {o.label}
                </option>
              ))}
            </select>
            <span className="field-hint">
              {draft.autoHintMode === "addressed"
                ? "Когда вас назвали по имени или после вопроса все замолчали; Claude сам решает, вам ли вопрос, и молчит, если нет. Для встреч с большим залом."
                : draft.autoHintMode === "any"
                  ? "На любой вопрос других участников, не чаще раза в 20 с. Удобно для разговора один на один."
                  : "Только по кнопке «Что ответить?»."}
              {draft.autoHintMode === "addressed" && !draft.myNames.length ? " Укажите имя выше — иначе обращения по имени не распознать." : ""}
            </span>
          </label>
          <div className="field-grid">
            <label className="field">
              <span className="field-label">Резюме по ходу, мин</span>
              <input
                type="number"
                min={0}
                max={120}
                value={draft.rollingSummaryMinutes}
                onChange={(e) => set("rollingSummaryMinutes", Math.max(0, Math.min(120, Number(e.target.value) || 0)))}
              />
              <span className="field-hint">0 — выключено</span>
            </label>
            <label className="field">
              <span className="field-label">Язык ответа</span>
              <select value={draft.answerLanguage} onChange={(e) => set("answerLanguage", e.target.value as AnswerLanguage)}>
                {LANGS.map((o) => (
                  <option key={o.v} value={o.v}>
                    {o.label}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <label className="field">
            <span className="field-label">
              Обо мне
              <span className="field-hint">роль, компания, продукты, типичные темы — подсказки опираются на это</span>
            </span>
            <textarea
              rows={5}
              value={draft.profile}
              placeholder="Например: техлид бэкенда в «Дала Софт», Алматы. Отвечаю за платежи и интеграции. Говорю по-русски, понимаю казахский."
              onChange={(e) => set("profile", e.target.value)}
            />
          </label>
        </fieldset>

        <fieldset>
          <legend>Распознавание речи</legend>
          <label className="field">
            <span className="field-label">Модель распознавания</span>
            <select value={draft.sttModel} onChange={(e) => set("sttModel", e.target.value)}>
              {!models?.some((m) => m.id === draft.sttModel) && <option value={draft.sttModel}>{draft.sttModel}</option>}
              {(models ?? []).map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} · {m.languages.join("/")} · {m.sizeMb >= 1000 ? `${(m.sizeMb / 1000).toFixed(1)} ГБ` : `${Math.round(m.sizeMb)} МБ`}
                  {m.downloaded ? "" : " · будет скачана"}
                </option>
              ))}
            </select>
          </label>
          <label className="field field-narrow">
            <span className="field-label">Потоки</span>
            <input type="number" min={1} max={32} value={draft.numThreads} onChange={(e) => set("numThreads", Math.max(1, Math.min(32, Math.round(Number(e.target.value) || 1))))} />
          </label>
        </fieldset>

        <fieldset>
          <legend>Захват звука</legend>
          <label className="check">
            <input type="checkbox" checked={draft.captureMic} onChange={(e) => set("captureMic", e.target.checked)} />
            <span>Микрофон</span>
          </label>
          <label className="field">
            <span className="field-label">Микрофон</span>
            <select value={draft.micDevice ?? ""} disabled={!draft.captureMic} onChange={(e) => set("micDevice", e.target.value || null)}>
              <option value="">По умолчанию</option>
              {inputs.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name}
                  {d.isDefault ? " (по умолчанию)" : ""}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span className="field-label">
              Микрофон слышит
              <span className="field-hint">в зале голоса различаются автоматически, до 10–15 человек</span>
            </span>
            <select
              value={draft.micMode}
              disabled={!draft.captureMic}
              onChange={(e) => {
                const micMode = e.target.value as MicMode;
                // The auto-hint mode follows the mic mode's default unless the user picked another one.
                setDraft((d) => ({
                  ...d,
                  micMode,
                  autoHintMode: d.autoHintMode === defaultAutoHintMode(d.micMode) ? defaultAutoHintMode(micMode) : d.autoHintMode,
                }));
              }}
            >
              {MIC_MODES.map((o) => (
                <option key={o.v} value={o.v}>
                  {o.label}
                </option>
              ))}
            </select>
          </label>
          <label className="check">
            <input type="checkbox" checked={draft.captureSystem} onChange={(e) => set("captureSystem", e.target.checked)} />
            <span>
              Системный звук (звонок)
              <span className="field-hint">выключите для очной встречи без удалённых участников</span>
            </span>
          </label>
          <label className="field">
            <span className="field-label">Источник системного звука</span>
            <select value={draft.systemDevice ?? ""} disabled={!draft.captureSystem} onChange={(e) => set("systemDevice", e.target.value || null)}>
              <option value="">По умолчанию</option>
              {monitors.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name}
                  {d.isDefault ? " (по умолчанию)" : ""}
                </option>
              ))}
            </select>
          </label>
          {devices === null && !loadError && <span className="muted small">Загрузка устройств…</span>}
        </fieldset>

        <VoiceSection state={state} />
        {loadError && <p className="card-error small">{loadError}</p>}

        <div className="settings-footer">
          {saveMsg && <span className={saveMsg.tone === "ok" ? "ok small" : "card-error small"}>{saveMsg.text}</span>}
          <span className="spacer" />
          <button
            type="button"
            className="btn btn-ghost"
            disabled={!dirty || saving}
            onClick={() => {
              setDraft(state.settings);
              setNamesText(state.settings.myNames.join(", "));
            }}
          >
            Отменить
          </button>
          <button type="submit" className="btn btn-primary" disabled={!dirty || saving}>
            Сохранить
          </button>
        </div>
      </form>
    </div>
  );
}
