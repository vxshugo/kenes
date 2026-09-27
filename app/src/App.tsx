import { useCallback, useEffect, useRef, useState } from "react";
import { Header } from "./components/Header";
import { HistoryTab } from "./components/HistoryTab";
import { IconX } from "./components/Icons";
import { LiveTab, canAct } from "./components/LiveTab";
import { PreStart, type Draft } from "./components/PreStart";
import { SettingsTab } from "./components/SettingsTab";
import { SummaryTab } from "./components/SummaryTab";
import { hasMod, SHORTCUTS } from "./lib/keys";
import { controller, useController } from "./session/useController";
import "./App.css";

type Tab = "live" | "summary" | "history" | "settings";

const TABS: Array<{ id: Tab; label: string }> = [
  { id: "live", label: "Эфир" },
  { id: "summary", label: "Итоги" },
  { id: "history", label: "История" },
  { id: "settings", label: "Настройки" },
];

/** Remembers the last text the user selected in the transcript or a card, for "Объяснить". */
function useLastSelection() {
  const last = useRef<{ text: string; at: number } | null>(null);
  useEffect(() => {
    const onChange = () => {
      const sel = window.getSelection();
      const text = sel?.toString().trim() ?? "";
      if (!text || text.length > 300) return;
      const node = sel?.anchorNode;
      const el = node instanceof Element ? node : node?.parentElement;
      if (el?.closest("input, textarea")) return;
      last.current = { text, at: Date.now() };
    };
    document.addEventListener("selectionchange", onChange);
    return () => document.removeEventListener("selectionchange", onChange);
  }, []);
  return last;
}

export default function App() {
  const state = useController();
  const [tab, setTab] = useState<Tab>("live");
  const [draft, setDraft] = useState<Draft>({ title: "", context: "" });
  const [question, setQuestion] = useState("");
  const askRef = useRef<HTMLInputElement | null>(null);
  const lastSelection = useLastSelection();

  const start = useCallback(() => {
    setTab("live");
    void controller.start(draft.title, draft.context);
  }, [draft]);

  const stop = useCallback(() => {
    setTab("summary");
    void controller.stop();
  }, []);

  const newMeeting = useCallback(() => {
    controller.reset();
    setDraft({ title: "", context: "" });
    setTab("live");
  }, []);

  const explain = useCallback(() => {
    const current = window.getSelection()?.toString().trim() ?? "";
    const recent = lastSelection.current && Date.now() - lastSelection.current.at < 60_000 ? lastSelection.current.text : "";
    const term = current && current.length <= 300 ? current : recent;
    if (term) {
      controller.explain(term);
      lastSelection.current = null;
      return;
    }
    setTab("live");
    setQuestion("Объясни: ");
    window.setTimeout(() => {
      const el = askRef.current;
      if (el) {
        el.focus();
        el.setSelectionRange(el.value.length, el.value.length);
      }
    }, 0);
  }, [lastSelection]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.altKey && !e.ctrlKey && !e.metaKey && /^Digit[1-4]$/.test(e.code)) {
        e.preventDefault();
        setTab(TABS[Number(e.code.slice(5)) - 1].id);
        return;
      }
      if (!hasMod(e)) return;
      const live = canAct(controller.getState());
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        if (live) {
          setTab("live");
          controller.hint();
        }
      } else if (e.code === "KeyK" && !e.shiftKey) {
        e.preventDefault();
        setTab("live");
        window.setTimeout(() => askRef.current?.focus(), 0);
      } else if (e.shiftKey && e.code === "KeyE") {
        e.preventDefault();
        if (live) explain();
      } else if (e.shiftKey && e.code === "KeyU") {
        e.preventDefault();
        if (live) {
          setTab("live");
          controller.translate();
        }
      } else if (e.shiftKey && e.code === "KeyK") {
        e.preventDefault();
        if (live) {
          setTab("live");
          controller.recap(5);
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [explain]);

  return (
    <div className="app">
      <Header state={state} onStart={start} onStop={stop} onNew={newMeeting} />
      <nav className="tabs" role="tablist" aria-label="Разделы">
        {TABS.map((t, i) => (
          <button
            key={t.id}
            role="tab"
            aria-selected={tab === t.id}
            className={`tab${tab === t.id ? " is-active" : ""}`}
            onClick={() => setTab(t.id)}
            title={`${t.label} (Alt+${i + 1})`}
          >
            {t.label}
            {t.id === "live" && state.cards.some((c) => c.status === "streaming" && !c.hidden) && tab !== "live" && <span className="tab-dot" aria-label="новая подсказка" />}
            {t.id === "summary" && state.final.status === "streaming" && tab !== "summary" && <span className="tab-dot" aria-label="итоги готовятся" />}
          </button>
        ))}
      </nav>
      <main className="content" role="tabpanel">
        {state.initError ? (
          <div className="tab-pane">
            <div className="banner banner-error">Не удалось подключиться к приложению: {state.initError}</div>
          </div>
        ) : !state.ready ? (
          <div className="tab-pane muted">Загрузка…</div>
        ) : tab === "live" ? (
          state.phase === "idle" ? (
            <PreStart state={state} draft={draft} onChange={setDraft} onStart={start} onOpenSettings={() => setTab("settings")} />
          ) : (
            <LiveTab state={state} askRef={askRef} question={question} setQuestion={setQuestion} onExplain={explain} />
          )
        ) : tab === "summary" ? (
          <SummaryTab state={state} />
        ) : tab === "history" ? (
          <HistoryTab activeId={["starting", "loading", "running", "stopping"].includes(state.phase) ? state.meetingId : null} />
        ) : (
          <SettingsTab state={state} />
        )}
      </main>
      <div className="toasts" aria-live="polite">
        {state.toasts.map((t) => (
          <div key={t.id} className={`toast toast-${t.tone}`} role={t.tone === "error" ? "alert" : "status"}>
            <span>{t.text}</span>
            <button className="icon-btn" onClick={() => controller.dismissToast(t.id)} aria-label="Закрыть">
              <IconX size={14} />
            </button>
          </div>
        ))}
      </div>
      <span className="sr-only">Горячие клавиши: {Object.values(SHORTCUTS).join(", ")}</span>
    </div>
  );
}
