import type Anthropic from "@anthropic-ai/sdk";
import type { Segment, SpeakerChange } from "../types";
import {
  cleanText,
  compareSegments,
  formatTranscript,
  fullTranscriptBlock,
  renameNote,
  speakerNotesBlock,
  transcriptUpdateBlock,
} from "./prompts";
import { compareLabels, defaultSpeakerName, type SpeakerDirectory } from "./speakers";

export type TextBlock = Anthropic.Beta.Messages.BetaTextBlockParam;
export type MessageParam = Anthropic.Beta.Messages.BetaMessageParam;

export type RequestPayload = {
  system: TextBlock[];
  messages: MessageParam[];
};

const EPHEMERAL = { type: "ephemeral" } as const;

/** label → the display name the log last used for it. */
type NameState = ReadonlyMap<string, string>;

type UserTurn = {
  role: "user";
  blocks: string[];
  /** Transcript lines this turn carries (ids into `finals`), requeued on rollback. */
  segmentIds: string[];
  /** Names the log knew before this turn, restored on rollback. */
  namesBefore: NameState;
};
type AssistantTurn = { role: "assistant"; text: string };
type Turn = UserTurn | AssistantTurn;

/**
 * Per-meeting, append-only message log laid out for prompt caching:
 *
 *   system:   [SYSTEM_PROMPT ◆]                              frozen for every meeting
 *   user:     [meeting context ◆, update₁, task₁]             context fixed for the meeting
 *   assistant:[answer₁]
 *   user:     [speaker_names?, update₂, task₂ ◆]              ◆ = cache_control breakpoint
 *
 * Each request re-sends the previous request's bytes unchanged and appends to them,
 * so every call reads the cached prefix and writes only the new tail. The only
 * permitted edit is dropping a trailing user turn whose request failed; its transcript
 * lines go back to the pending queue and ride along with the next turn.
 *
 * Speaker names: lines are rendered with the names current when they are sent and never
 * re-rendered. A rename therefore adds a note («Участник 3 теперь зовут Айдос.») to the
 * next user turn instead of rewriting earlier turns.
 */
export class Conversation {
  private readonly systemBlocks: readonly string[];
  private readonly turns: Turn[] = [];
  private readonly finals = new Map<string, Segment>();
  private pending: string[] = [];
  /** What the committed + in-flight log has told Claude each label is called. */
  private names: NameState = new Map();

  constructor(
    systemPrompt: string,
    readonly contextBlock: string,
    readonly speakers: SpeakerDirectory,
  ) {
    this.systemBlocks = [systemPrompt];
  }

  /** Records a final segment. Duplicates (same id) are ignored. */
  addFinal(seg: Segment): boolean {
    if (!seg.isFinal || this.finals.has(seg.id) || !cleanText(seg.text)) return false;
    this.finals.set(seg.id, seg);
    this.pending.push(seg.id);
    return true;
  }

  /**
   * Applies `speakersRelabeled`. Lines already in the log keep their old names (append-only);
   * lines not sent yet and the final summary's full transcript use the new labels.
   */
  relabel(changes: readonly SpeakerChange[]): number {
    let n = 0;
    for (const c of changes) {
      const seg = this.finals.get(c.segmentId);
      const speaker = c.speaker ?? null;
      if (!seg || seg.speaker === speaker) continue;
      this.finals.set(seg.id, { ...seg, speaker });
      n++;
    }
    return n;
  }

  /** All final segments seen so far, in transcript order. */
  allFinals(): Segment[] {
    return [...this.finals.values()].sort(compareSegments);
  }

  pendingCount(): number {
    return this.pending.length;
  }

  /** True while a live turn is waiting for its reply. */
  get awaitingReply(): boolean {
    return this.turns.at(-1)?.role === "user";
  }

  /** Number of completed user→assistant exchanges. */
  get exchanges(): number {
    return this.turns.filter((t) => t.role === "assistant").length;
  }

  private segmentsOf(ids: readonly string[]): Segment[] {
    return ids.map((id) => this.finals.get(id)).filter((s): s is Segment => !!s).sort(compareSegments);
  }

  /**
   * Blocks that bring the log up to date: rename notes for labels whose name changed since
   * `known`, then the new transcript lines rendered with current names. Returns the name state
   * the log will have once these blocks are in it.
   */
  private updateBlocks(ids: readonly string[], known: NameState): { blocks: string[]; names: Map<string, string> } {
    const names = new Map(known);
    const notes: string[] = [];
    for (const label of [...known.keys()].sort(compareLabels)) {
      const before = known.get(label)!;
      const now = this.speakers.nameOf(label);
      if (now === before) continue;
      notes.push(renameNote(before, now, now === defaultSpeakerName(label)));
      names.set(label, now);
    }
    const segments = this.segmentsOf(ids);
    for (const seg of segments) {
      const label = this.speakers.labelOf(seg);
      if (label) names.set(label, this.speakers.nameOf(label));
    }
    const blocks: string[] = [];
    if (notes.length) blocks.push(speakerNotesBlock(notes));
    if (segments.length) blocks.push(transcriptUpdateBlock(formatTranscript(segments, this.speakers)));
    return { blocks, names };
  }

  /**
   * Appends a user turn (rename notes, new transcript lines, the task) and returns the
   * request. Must be followed by `commit()` on success or `rollback()` on failure.
   */
  beginTurn(taskText: string): RequestPayload {
    if (this.awaitingReply) throw new Error("previous turn is still awaiting its reply");
    const segmentIds = this.pending;
    const { blocks, names } = this.updateBlocks(segmentIds, this.names);
    this.pending = [];
    this.turns.push({ role: "user", blocks: [...blocks, taskText], segmentIds, namesBefore: this.names });
    this.names = names;
    return this.render(this.turns, "last");
  }

  /** Appends the assistant reply to the trailing user turn. */
  commit(text: string): void {
    if (!this.awaitingReply) throw new Error("no turn to commit");
    const t = text.trim();
    if (!t) throw new Error("empty assistant reply");
    this.turns.push({ role: "assistant", text: t });
  }

  /** Drops the trailing user turn after a failed request; its lines and rename notes return to the queue. */
  rollback(): void {
    const last = this.turns.at(-1);
    if (!last || last.role !== "user") return;
    this.turns.pop();
    this.names = last.namesBefore;
    this.pending = [...last.segmentIds, ...this.pending];
  }

  /**
   * A one-off request that reuses the log's cached prefix without extending the log
   * (rolling summary). The breakpoint goes on the last *shared* block, not on the
   * one-off tail, so the fork never pays to cache bytes nobody will read again.
   * Pending lines and rename notes are included but stay pending for the next live turn.
   */
  fork(taskText: string): RequestPayload {
    const inflight = this.awaitingReply ? (this.turns.at(-1) as UserTurn) : null;
    const committed = inflight ? this.turns.slice(0, -1) : [...this.turns];
    // A live turn in flight holds lines that are no longer pending; include them too.
    const ids = inflight ? [...inflight.segmentIds, ...this.pending] : this.pending;
    const { blocks } = this.updateBlocks(ids, inflight ? inflight.namesBefore : this.names);
    const turns: Turn[] = [...committed, { role: "user", blocks: [...blocks, taskText], segmentIds: [], namesBefore: this.names }];
    return this.render(turns, committed.length ? committed.length - 1 : "none");
  }

  /**
   * Separate request over the full transcript (final summary), rendered with the current
   * names and labels. Shares the frozen system prompt and context block; the breakpoint
   * sits after the transcript so a regenerate of the summary reads it back.
   */
  finalRequest(taskText: string): RequestPayload {
    const transcript = fullTranscriptBlock(formatTranscript(this.allFinals(), this.speakers));
    return {
      system: this.renderSystem(),
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: this.contextBlock },
            { type: "text", text: transcript, cache_control: EPHEMERAL },
            { type: "text", text: taskText },
          ],
        },
      ],
    };
  }

  private renderSystem(): TextBlock[] {
    return this.systemBlocks.map((text, i, all) =>
      i === all.length - 1 ? { type: "text", text, cache_control: EPHEMERAL } : { type: "text", text },
    );
  }

  /**
   * Renders turns to API params. `moving` picks the turn that gets the moving breakpoint
   * on its last block: "last" (live turns), an index (fork: last committed turn), or "none".
   */
  private render(turns: readonly Turn[], moving: "last" | "none" | number): RequestPayload {
    const movingIndex = moving === "last" ? turns.length - 1 : moving === "none" ? -1 : moving;
    const messages: MessageParam[] = turns.map((turn, i) => {
      const texts = turn.role === "user" ? turn.blocks : [turn.text];
      const content: TextBlock[] = texts.map((text) => ({ type: "text", text }));
      if (i === 0) {
        content.unshift({ type: "text", text: this.contextBlock, cache_control: EPHEMERAL });
      }
      if (i === movingIndex) {
        const last = content[content.length - 1];
        content[content.length - 1] = { ...last, cache_control: EPHEMERAL };
      }
      return { role: turn.role, content };
    });
    return { system: this.renderSystem(), messages };
  }
}
