import { memo, useCallback, useLayoutEffect, useRef, useState } from "react";
import { formatClock } from "../llm/prompts";
import type { SpeakerDirectory } from "../llm/speakers";
import { findName, looksLikeQuestion } from "../llm/triggers";
import type { Segment } from "../types";
import { IconArrowDown } from "./Icons";
import { SpeakerName } from "./Speakers";

type LinesProps = {
  segments: Segment[];
  speakers: SpeakerDirectory;
  /** The user's names: lines addressing the user get a mark. */
  myNames: readonly string[];
  /** Renames a label; without it names are not clickable. */
  onRename?: (label: string, name: string) => void;
};

/** Header key: a new speaker header whenever this changes between lines. */
function lineKey(seg: Segment, speakers: SpeakerDirectory): string {
  const label = speakers.labelOf(seg);
  if (label) return label;
  return `${seg.isFinal ? "?" : "~"}${seg.source}`;
}

type LineProps = {
  seg: Segment;
  label: string | null;
  name: string;
  custom: boolean;
  showSpeaker: boolean;
  myNames: readonly string[];
  onRename?: (label: string, name: string) => void;
};

const Line = memo(function Line({ seg, label, name, custom, showSpeaker, myNames, onRename }: LineProps) {
  const partialHeader = !seg.isFinal && !label;
  const own = label === "me";
  const hit = seg.isFinal && !own ? findName(seg.text, myNames) : null;
  const question = seg.isFinal && !own && looksLikeQuestion(seg.text);
  // Addressed: the name in calling position or in a question; otherwise just a mention.
  const addressed = !!hit && (hit.vocative || question);
  return (
    <div
      className={`line src-${seg.source}${seg.isFinal ? "" : " is-partial"}${question ? " is-question" : ""}${addressed ? " is-addressed" : ""}${showSpeaker ? " has-speaker" : ""}`}
    >
      <span className="line-time">{formatClock(seg.startMs)}</span>
      <div className="line-main">
        {showSpeaker &&
          (partialHeader ? (
            <span className="line-speaker spk spk-pending">{seg.source === "system" ? "Звонок…" : "Зал…"}</span>
          ) : (
            <span className="line-speaker">
              <SpeakerName label={label} name={name} custom={custom} onRename={onRename && label ? (n) => onRename(label, n) : undefined} />
            </span>
          ))}
        <span className="line-text">
          {seg.text}
          {hit ? (
            <span className={`q-mark ${addressed ? "addr-mark" : "mention-mark"}`} title={addressed ? "Обращаются к вам по имени" : "Вас упомянули по имени"}>
              @
            </span>
          ) : (
            question && (
              <span className="q-mark" title="Похоже на вопрос">
                ?
              </span>
            )
          )}
        </span>
      </div>
    </div>
  );
});

/** Transcript lines with speaker headers in stable colors; names are renamable in place. */
export function TranscriptLines({ segments, speakers, myNames, onRename }: LinesProps) {
  return (
    <>
      {segments.map((seg, i) => {
        const prev = segments[i - 1];
        const label = speakers.labelOf(seg);
        const key = lineKey(seg, speakers);
        const showSpeaker = !prev || lineKey(prev, speakers) !== key;
        return (
          <Line
            key={seg.id + (seg.isFinal ? "" : "~")}
            seg={seg}
            label={label}
            name={speakers.nameOf(label)}
            custom={!!speakers.customName(label)}
            showSpeaker={showSpeaker}
            myNames={myNames}
            onRename={onRename}
          />
        );
      })}
    </>
  );
}

type Props = Omit<LinesProps, "segments"> & {
  finals: Segment[];
  partials: Segment[];
  emptyText: string;
};

/** Live transcript with auto-scroll that pauses while the user reads back, plus "jump to live". */
export function Transcript({ finals, partials, emptyText, ...lines }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [atBottom, setAtBottom] = useState(true);
  const stick = useRef(true);

  const onScroll = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    const bottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    stick.current = bottom;
    setAtBottom(bottom);
  }, []);

  useLayoutEffect(() => {
    const el = ref.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [finals, partials]);

  const jump = () => {
    const el = ref.current;
    if (!el) return;
    stick.current = true;
    setAtBottom(true);
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  };

  const all = [...finals, ...partials];
  return (
    <section className="transcript-wrap" aria-label="Расшифровка">
      <div className="transcript" ref={ref} onScroll={onScroll} role="log" aria-live="off" data-selectable="transcript">
        {all.length === 0 ? <p className="empty">{emptyText}</p> : <TranscriptLines segments={all} {...lines} />}
      </div>
      {!atBottom && all.length > 0 && (
        <button className="jump-live" onClick={jump}>
          <IconArrowDown size={14} /> К живому
        </button>
      )}
    </section>
  );
}
