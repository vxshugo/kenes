import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { findName, looksLikeQuestion, SILENCE_WAIT_MS } from "../llm/triggers";
import { applyRelabel } from "../llm/speakers";
import { MemoryStorage, waitFor } from "../test/helpers";
import type { PipelineEvent, Segment, Settings } from "../types";
import { MOCK_DEFAULT_SETTINGS, MockBackend, MockDiarizer } from "./mock";
import { MOCK_PEOPLE, MOCK_SCRIPT, type ScriptLine } from "./mockScript";

beforeEach(() => {
  vi.stubGlobal("localStorage", new MemoryStorage());
});
afterEach(() => {
  vi.unstubAllGlobals();
});

type Played = { events: PipelineEvent[]; finals: Segment[]; meetingId: string; backend: MockBackend };

/** Plays a script to the end at high speed, then stops the session. */
async function play(settings: Partial<Settings>, script: ScriptLine[] = MOCK_SCRIPT, enroll = false): Promise<Played> {
  const backend = new MockBackend({ speed: 1000, script });
  await backend.saveSettings({ ...MOCK_DEFAULT_SETTINGS, ...settings });
  if (enroll) await backend.enrollVoice(10);
  const events: PipelineEvent[] = [];
  await backend.onEvent((e) => events.push(e));
  const { meetingId } = await backend.startSession("Синк", "");
  const finals = () => events.filter((e): e is Extract<PipelineEvent, { type: "segment" }> => e.type === "segment" && e.isFinal);
  const expected = script.length;
  await waitFor(() => finals().length >= expected, 20_000, "the script to finish");
  await backend.stopSession();
  return { events, finals: finals().map(({ type: _t, ...s }) => s), meetingId, backend };
}

describe("MockBackend: scripted multi-party meeting", () => {
  it("the default script is a 10-person meeting that addresses the user as «Хуго»", () => {
    expect(MOCK_PEOPLE.length).toBeGreaterThanOrEqual(8);
    expect(MOCK_PEOPLE.length).toBeLessThanOrEqual(10);
    expect(new Set(MOCK_SCRIPT.map((l) => l.who)).size).toBe(MOCK_PEOPLE.length);
    expect(MOCK_DEFAULT_SETTINGS.myNames).toEqual(["Хуго"]);
    const addressed = MOCK_SCRIPT.filter((l) => l.who !== "me" && findName(l.text, ["Хуго"]));
    expect(addressed.length).toBeGreaterThanOrEqual(2);
    expect(addressed.some((l) => /хугоға/.test(l.text))).toBe(true);
    // An unnamed question followed by silence (condition (b) of the "addressed" mode).
    const silent = MOCK_SCRIPT.filter(
      (l, i) => l.who !== "me" && looksLikeQuestion(l.text) && !findName(l.text, ["Хуго"]) && (MOCK_SCRIPT[i + 1]?.pause ?? 0) > SILENCE_WAIT_MS,
    );
    expect(silent.map((l) => l.who)).toEqual(["erlan"]);
  });

  it("online: the user is «me», everyone else is sys:N; labels only on finals; relabel on stop", async () => {
    const { events, finals, meetingId, backend } = await play({ captureMic: true, captureSystem: true, micMode: "me" });
    const partials = events.filter((e) => e.type === "segment" && !e.isFinal);
    expect(partials.length).toBeGreaterThan(0);
    expect(partials.every((e) => e.type === "segment" && e.speaker === null)).toBe(true);
    expect(finals.filter((s) => s.source === "mic").every((s) => s.speaker === "me")).toBe(true);
    const sysLabels = new Set(finals.filter((s) => s.source === "system").map((s) => s.speaker));
    expect([...sysLabels].every((l) => l === null || /^sys:\d+$/.test(l))).toBe(true);
    // 9 people in the call plus one spurious online cluster.
    expect([...sysLabels].filter(Boolean)).toHaveLength(10);

    // speakersRelabeled arrives before the session reports idle, for a couple of segments.
    const relabelAt = events.findIndex((e) => e.type === "speakersRelabeled");
    const idleAt = events.findIndex((e) => e.type === "status" && e.state === "idle");
    expect(relabelAt).toBeGreaterThan(-1);
    expect(relabelAt).toBeLessThan(idleAt);
    const relabel = events[relabelAt] as Extract<PipelineEvent, { type: "speakersRelabeled" }>;
    expect(relabel.changes.length).toBeGreaterThanOrEqual(2);
    for (const c of relabel.changes) {
      expect(finals.some((f) => f.id === c.segmentId)).toBe(true);
      expect(c.speaker).toMatch(/^sys:\d+$/);
    }
    // The unsure line (speaker null) gets a label; the stray label disappears.
    const fixed = applyRelabel(finals, relabel.changes);
    expect(fixed.filter((s) => s.speaker === null)).toHaveLength(0);
    const stored = await backend.getMeeting(meetingId);
    expect(stored.segments.map((s) => s.speaker)).toEqual(fixed.map((s) => s.speaker));
    expect(stored.speakers.map((s) => s.label)).not.toContain("sys:10");
    expect(stored.speakers).toHaveLength(10); // me + 9
    expect(stored.speakers[0].talkMs).toBeGreaterThanOrEqual(stored.speakers[1].talkMs);
  });

  it("room without a voiceprint: everyone, the user included, is mic:N", async () => {
    const { finals } = await play({ captureMic: true, captureSystem: false, micMode: "room" });
    expect(finals.every((s) => s.source === "mic")).toBe(true);
    expect(finals.some((s) => s.speaker === "me")).toBe(false);
    // 10 voices plus the spurious online cluster that re-clustering merges back.
    expect(new Set(finals.map((s) => s.speaker).filter(Boolean)).size).toBe(11);
  });

  it("room with a voiceprint: the user's segments are «me»", async () => {
    const { finals } = await play({ captureMic: true, captureSystem: false, micMode: "room" }, MOCK_SCRIPT, true);
    const mine = MOCK_SCRIPT.filter((l) => l.who === "me").map((l) => l.text);
    expect(finals.filter((s) => s.speaker === "me").map((s) => s.text)).toEqual(mine);
  });

  it("hybrid: people in the room are mic:N, remote people are sys:N", async () => {
    const { finals } = await play({ captureMic: true, captureSystem: true, micMode: "room" });
    const room = new Set(MOCK_PEOPLE.filter((p) => p.site === "room" && !p.me).map((p) => p.id));
    for (const [i, line] of MOCK_SCRIPT.entries()) {
      const s = finals[i];
      if (line.who === "me") expect(s.source).toBe("mic");
      else expect(s.source).toBe(room.has(line.who) ? "mic" : "system");
    }
    expect(finals.some((s) => s.speaker?.startsWith("mic:"))).toBe(true);
    expect(finals.some((s) => s.speaker?.startsWith("sys:"))).toBe(true);
  });
});

describe("MockBackend: speaker names and voiceprint", () => {
  const SHORT: ScriptLine[] = [
    { who: "aigerim", pause: 10, text: "айдос что по мобилке" },
    { who: "aidos", pause: 10, text: "всё по плану" },
  ];

  it("rename_speaker persists names in get_meeting; an empty name resets", async () => {
    const { meetingId, backend } = await play({ captureSystem: true, micMode: "me" }, SHORT);
    await backend.renameSpeaker(meetingId, "sys:2", "Айдос");
    await backend.renameSpeaker(meetingId, "sys:7", "Никто");
    let m = await backend.getMeeting(meetingId);
    expect(m.speakers.find((s) => s.label === "sys:2")?.name).toBe("Айдос");
    expect(m.speakers.find((s) => s.label === "sys:7")).toMatchObject({ name: "Никто", segmentCount: 0, talkMs: 0 });
    await backend.renameSpeaker(meetingId, "sys:7", "");
    m = await backend.getMeeting(meetingId);
    expect(m.speakers.map((s) => s.label).sort()).toEqual(["sys:1", "sys:2"]);
    await expect(backend.renameSpeaker("nope", "sys:1", "x")).rejects.toThrow();
  });

  it("enroll_voice records for the given seconds, persists, and can be cleared; not during a session", async () => {
    const backend = new MockBackend({ speed: 1000, script: SHORT });
    expect(await backend.voiceprintStatus()).toEqual({ enrolled: false, createdAt: null });
    const t0 = Date.now();
    const r = await backend.enrollVoice(10);
    expect(Date.now() - t0).toBeGreaterThanOrEqual(9); // 10 s at 1000x
    expect(r.speechMs).toBeGreaterThan(6_000);
    expect(r.speechMs).toBeLessThanOrEqual(10_000);
    const st = await backend.voiceprintStatus();
    expect(st.enrolled).toBe(true);
    expect(Date.parse(st.createdAt!)).not.toBeNaN();
    expect(localStorage.getItem("kenes.mock.voiceprint")).toContain("createdAt");
    await backend.clearVoiceprint();
    expect((await backend.voiceprintStatus()).enrolled).toBe(false);

    await backend.startSession("x", "");
    await expect(backend.enrollVoice(10)).rejects.toThrow(/во время встречи/);
    await backend.stopSession();
  });

  it("MockDiarizer numbers labels per prefix by first appearance and never reuses them", () => {
    const d = new MockDiarizer("room", false);
    const [me, a, b] = MOCK_PEOPLE;
    expect(d.assign("1", a, "system", { who: a.id, text: "x", pause: 0 })).toBe("sys:1");
    expect(d.assign("2", b, "mic", { who: b.id, text: "x", pause: 0 })).toBe("mic:1");
    expect(d.assign("3", me, "mic", { who: me.id, text: "x", pause: 0 })).toBe("mic:2");
    expect(d.assign("4", a, "system", { who: a.id, text: "x", pause: 0, stray: true })).toBe("sys:2");
    expect(d.assign("5", a, "system", { who: a.id, text: "да", pause: 0, unsure: true })).toBeNull();
    expect(d.assign("6", a, "system", { who: a.id, text: "x", pause: 0 })).toBe("sys:1");
    expect(d.recluster(new Set(["4", "5"]))).toEqual([
      { segmentId: "4", speaker: "sys:1" },
      { segmentId: "5", speaker: "sys:1" },
    ]);
  });
});
