import { describe, expect, it } from "vitest";
import type { Segment } from "../types";
import {
  AUTO_HINT_DEBOUNCE_MS,
  NAMED_HINT_DEBOUNCE_MS,
  SILENCE_WAIT_MS,
  SilenceWatch,
  analyzeQuestion,
  evaluateAutoHint,
  findName,
  foldForMatch,
  latinToCyrillic,
  looksLikeQuestion,
  previousSameSpeaker,
  rollingSummaryDue,
  suggestionsDue,
  tokenMatchesName,
  tokenize,
  type AutoHintInput,
} from "./triggers";

describe("tokenize", () => {
  it("lowercases, folds ё and splits on hyphens and punctuation", () => {
    expect(tokenize("Что-то Ещё, ПОКАЖИТЕ!")).toEqual(["что", "то", "еще", "покажите"]);
  });
  it("keeps Kazakh letters inside words", () => {
    expect(tokenize("тестілеуді қашан бастаймыз")).toEqual(["тестілеуді", "қашан", "бастаймыз"]);
  });
});

describe("looksLikeQuestion: Russian", () => {
  const questions = [
    "как у вас обстоят дела с интеграцией платежного шлюза успеваете к пятнице",
    "а по возвратам какие риски видите",
    "сколько времени нужно чтобы собрать метрики по крэшам",
    "а у вас есть оценка сколько займут автотесты",
    "это вообще реально с текущей архитектурой или нет",
    "хорошо тогда расскажите в двух словах какой бюджет на это закладывать",
    "и последний вопрос по безопасности пентест кто будет делать внешний подрядчик или своими силами",
    "почему решили переходить на кафку",
    "зачем нам второй брокер",
    "когда планируете закончить интеграцию",
    "где будет развернут стейдж",
    "кто отвечает за релиз",
    "что у нас по срокам",
    "а когда будет готово",
    "ну и как вам новая схема",
    "можно ли перенести релиз на понедельник",
    "не могли бы вы прислать оценку",
    "готовы ли вы выйти в прод на этой неделе",
    "объясните пожалуйста как работает возврат",
    "подскажите какая сейчас нагрузка на базу",
    "а вы что думаете по этому поводу",
    "как вы считаете стоит ли брать подрядчика",
    "ваше мнение по поводу нового дизайна",
    "чья это зона ответственности",
    "насколько это критично для клиента",
  ];
  it.each(questions)("detects: %s", (q) => {
    expect(analyzeQuestion(q).isQuestion).toBe(true);
  });

  const statements = [
    "я думаю что это хорошая идея",
    "так как у нас мало времени давайте быстрее",
    "мы как раз закончили тестирование",
    "когда мы запустили первую версию было много багов",
    "вряд ли мы успеем к пятнице",
    "что то пошло не так на стейдже",
    "вот почему мы решили перейти на кафку",
    "не знаю почему так получилось",
    "понятно тогда фиксируем релиз в пятницу возвраты в среду",
    "по мобилке у нас канареечный релиз на десять процентов пользователей",
    "хорошо тогда айгерим запиши задачу на оценку инфраструктуры",
    "ерлан тут подключился он по продукту",
    "что касается бюджета мы уложимся",
    "я вам скажу что это сложно",
    "в целом да осталось закрыть возвраты",
  ];
  it.each(statements)("ignores: %s", (s) => {
    expect(analyzeQuestion(s).isQuestion).toBe(false);
  });
});

describe("looksLikeQuestion: Kazakh and mixed", () => {
  const questions = [
    "жақсы түсінікті онда тестілеуді қашан бастаймыз",
    "бұл қосымша шығын ба бюджетке кіре ме",
    "бізге екі тестировщик керек пе әлде біреуі жеткілікті ме",
    "ерлан сіз келісесіз бе",
    "интеграцияны қалай тестілейміз",
    "неге релиз кешікті",
    "бұл жұмысты кім жасайды",
    "қанша уақыт керек",
    "сервер қайда орналасқан",
    "қандай тәуекелдер бар",
    "бюджет жеткілікті бола ма",
    "сіз бұл туралы білесіз бе",
    "дайынсызба бастауға",
    "айтыңызшы жоба туралы",
    "түсіндіріңізші мына схеманы",
    "ал сіз ше",
    "бұл не",
    "сіз не ойлайсыз",
    "а по срокам релиз бейсенбіде бола ма",
    "керек пе",
    "деплой бар ма",
  ];
  it.each(questions)("detects: %s", (q) => {
    expect(analyzeQuestion(q).isQuestion).toBe(true);
  });

  const statements = [
    "сәлеметсіздер ме коллеги давайте начнем синк по проекту жолдас у нас полчаса",
    "енді келесі мәселе жаңа қызметкерлер туралы",
    "иә келісемін только давайте без переносов клиент ждет",
    "отлично спасибо всем келесі синк бейсенбіде сау болыңыздар",
    "бүгін біз релизді талқылаймыз",
    "мен келісемін",
    "не знаю пока",
    "это не критично",
  ];
  it.each(statements)("ignores: %s", (s) => {
    expect(analyzeQuestion(s).isQuestion).toBe(false);
  });
});

describe("analyzeQuestion details", () => {
  it("reports the matching rule", () => {
    expect(analyzeQuestion("готовы ли вы").reason).toBe("particle:ли");
    expect(analyzeQuestion("тестілеуді қашан бастаймыз").reason).toBe("wh-kk:қашан");
    expect(analyzeQuestion("это реально или нет").reason).toBe("phrase:или нет");
  });
  it("counts words of the original text", () => {
    expect(analyzeQuestion("сәлеметсіз бе").words).toBe(2);
  });
  it("handles empty input", () => {
    expect(looksLikeQuestion("")).toBe(false);
    expect(looksLikeQuestion("   ")).toBe(false);
  });
});

function seg(partial: Partial<Segment> = {}): Segment {
  return {
    id: "system-1",
    source: "system",
    speaker: "sys:1",
    startMs: 10_000,
    endMs: 13_000,
    text: "когда планируете закончить интеграцию",
    isFinal: true,
    ...partial,
  };
}

function input(partial: Partial<AutoHintInput> = {}): AutoHintInput {
  return {
    segment: seg(),
    own: false,
    previous: null,
    mode: "any",
    myNames: ["Хуго", "Hugo"],
    nowMs: 1_000_000,
    lastAutoHintAt: null,
    hintBusy: false,
    hasApiKey: true,
    ...partial,
  };
}

describe("findName: the user's name in raw ASR text", () => {
  const names = ["Хуго", "Hugo"];
  it.each([
    ["хуго а по бэкенду что со сроками", 0],
    ["хуго, а вы что думаете", 0],
    ["хугоға сұрақ интеграция қашан дайын", 0],
    ["а хугоның ойы қандай", 1],
    ["хугоны тыңдайық", 0],
    ["хугодан сұрайық", 0],
    ["хугомен келістік пе", 0],
    ["коллеги хуго ты с нами", 1],
    ["вопрос к хуго по серверам", 2],
    ["ХУГО скажите", 0],
  ])("finds it in «%s»", (text, index) => {
    expect(findName(text, names)).toMatchObject({ name: "Хуго", index });
  });

  it.each(["хугл это поисковик", "губы", "хугоист", "мы с хугоооо"])("ignores «%s»", (text) => {
    expect(findName(text, names)).toBeNull();
  });

  it("marks calling position: the first words after fillers or the last word", () => {
    expect(findName("хуго что скажешь", names)?.vocative).toBe(true);
    expect(findName("ну а хуго что скажет", names)?.vocative).toBe(true);
    expect(findName("а что скажет хуго", names)?.vocative).toBe(true);
    expect(findName("мы с хуго вчера обсуждали", names)?.vocative).toBe(false);
  });

  it("handles Russian declension of consonant-, а- and й-final names", () => {
    expect(findName("спросим у айдоса", ["Айдос"])).not.toBeNull();
    expect(findName("передай ерлану", ["Ерлан"])).not.toBeNull();
    expect(findName("с ерланом договорились", ["Ерлан"])).not.toBeNull();
    expect(findName("вопрос к дане", ["Дана"])).not.toBeNull();
    expect(findName("у марии есть оценка", ["Мария"])).not.toBeNull();
    expect(findName("спроси андрея", ["Андрей"])).not.toBeNull();
    expect(findName("игорю отправлю", ["Игорь"])).not.toBeNull();
  });

  it("handles Kazakh endings and letters on names typed in Russian spelling", () => {
    expect(findName("айдосқа сұрақ", ["Айдос"])).not.toBeNull();
    expect(findName("айгерімге айтайық", ["Айгерим"])).not.toBeNull();
    expect(findName("нұрланның пікірі", ["Нурлан"])).not.toBeNull();
    expect(findName("ерланжан қалайсыз", ["Ерлан"])).not.toBeNull();
  });

  it("matches Latin names against the recognizer's Cyrillic, and multi-word names as a phrase", () => {
    expect(latinToCyrillic("Hugo")).toBe("хуго");
    expect(latinToCyrillic("Aidos")).toBe("айдос");
    expect(latinToCyrillic("Zhanna")).toBe("жанна");
    expect(findName("хуго а вы", ["Hugo"])).toMatchObject({ name: "Hugo" });
    expect(findName("вопрос к хуго мырзаға", ["Хуго мырза"])).toMatchObject({ index: 2 });
    expect(findName("мырза келді", ["Хуго мырза"])).toBeNull();
  });

  it("folds Kazakh letters and ё; two-letter names match only exactly", () => {
    expect(foldForMatch("Әйгерім ҚҰЁ")).toBe("айгерим куе");
    expect(tokenMatchesName("ан", "ан")).toBe(true);
    expect(tokenMatchesName("анна", "ан")).toBe(false);
    expect(findName("", names)).toBeNull();
    expect(findName("хуго", [])).toBeNull();
  });
});

describe("evaluateAutoHint: mode \"any\"", () => {
  it("fires on a final question from someone else", () => {
    expect(evaluateAutoHint(input())).toEqual({ action: "fire", kind: "any", reason: "wh-start:когда" });
  });
  it("never fires on the user's own speech", () => {
    expect(evaluateAutoHint(input({ own: true })).reason).toBe("own-speech");
    expect(evaluateAutoHint(input({ own: true, mode: "addressed", segment: seg({ text: "хуго когда релиз" }) })).reason).toBe("own-speech");
  });
  it("ignores partial segments", () => {
    expect(evaluateAutoHint(input({ segment: seg({ isFinal: false }) })).reason).toBe("partial");
  });
  it("debounces 20 s between auto hints", () => {
    const now = 1_000_000;
    expect(evaluateAutoHint(input({ nowMs: now, lastAutoHintAt: now - AUTO_HINT_DEBOUNCE_MS + 1 })).reason).toBe("debounce");
    expect(evaluateAutoHint(input({ nowMs: now, lastAutoHintAt: now - AUTO_HINT_DEBOUNCE_MS })).action).toBe("fire");
  });
  it("skips while a hint is streaming, statements, and short lines", () => {
    expect(evaluateAutoHint(input({ hintBusy: true })).reason).toBe("busy");
    expect(evaluateAutoHint(input({ segment: seg({ text: "понятно тогда фиксируем релиз в пятницу" }) })).reason).toBe("not-question");
    expect(evaluateAutoHint(input({ segment: seg({ text: "кім жасайды" }) })).reason).toBe("too-short");
  });
  it("respects the mode and the key", () => {
    expect(evaluateAutoHint(input({ mode: "off" })).reason).toBe("disabled");
    expect(evaluateAutoHint(input({ hasApiKey: false })).reason).toBe("no-key");
  });
});

describe("evaluateAutoHint: mode \"addressed\"", () => {
  const addressed = (p: Partial<AutoHintInput>) => evaluateAutoHint(input({ mode: "addressed", ...p }));

  it("(a) fires at once when the question names the user, with inflection", () => {
    expect(addressed({ segment: seg({ text: "хуго а по бэкенду что у нас со сроками" }) })).toMatchObject({ action: "fire", kind: "named" });
    expect(addressed({ segment: seg({ text: "хугоға сұрақ бюджетке қанша қосады" }) })).toMatchObject({ action: "fire", kind: "named" });
    expect(addressed({ segment: seg({ text: "хуго сіз не дейсіз" }) })).toMatchObject({ action: "fire", kind: "named" });
  });

  it("(a) fires when the same speaker's previous segment had the name", () => {
    const previous = seg({ id: "system-0", text: "так хуго", startMs: 5_000, endMs: 6_000 });
    expect(addressed({ previous, segment: seg({ text: "а сроки какие у вас" }) })).toMatchObject({ action: "fire", kind: "named" });
    // …but the previous name alone doesn't make a statement a question.
    expect(addressed({ previous, segment: seg({ text: "релиз переносим на понедельник" }) }).action).toBe("skip");
  });

  it("(a) a name in calling position counts without a question; a third-person mention needs one", () => {
    expect(addressed({ segment: seg({ text: "хуго твоя очередь рассказывать" }) })).toMatchObject({ action: "fire", kind: "named" });
    expect(addressed({ segment: seg({ text: "хуго ты" }) })).toMatchObject({ action: "fire", kind: "named" });
    expect(addressed({ segment: seg({ text: "итак фиксируем хуго и нурлан считают серверы" }) })).toEqual({ action: "skip", reason: "name-not-addressed" });
  });

  it("(a) uses a shorter debounce and still skips while busy", () => {
    const named = seg({ text: "хуго когда будет готово" });
    const now = 1_000_000;
    expect(addressed({ segment: named, nowMs: now, lastAutoHintAt: now - NAMED_HINT_DEBOUNCE_MS + 1 }).reason).toBe("debounce");
    expect(addressed({ segment: named, nowMs: now, lastAutoHintAt: now - NAMED_HINT_DEBOUNCE_MS }).action).toBe("fire");
    expect(addressed({ segment: named, hintBusy: true }).reason).toBe("busy");
  });

  it("(b) an unnamed question waits for silence instead of firing", () => {
    expect(addressed({ segment: seg({ text: "коллеги кто может взять ревью до среды" }) })).toEqual({
      action: "wait",
      kind: "silence",
      reason: "wh-start:кто",
      waitMs: SILENCE_WAIT_MS,
    });
    expect(addressed({ segment: seg({ text: "понятно тогда фиксируем релиз" }) }).reason).toBe("not-question");
    expect(addressed({ segment: seg({ text: "коллеги кто может взять ревью" }), nowMs: 100_000, lastAutoHintAt: 90_000 }).reason).toBe("debounce");
  });

  it("without configured names only (b) is possible", () => {
    expect(addressed({ myNames: [], segment: seg({ text: "хуго когда будет готово" }) }).action).toBe("wait");
  });
});

describe("previousSameSpeaker", () => {
  const labelOf = (s: Segment) => s.speaker;
  const a1 = seg({ id: "a1", speaker: "sys:1", startMs: 0, endMs: 2_000, text: "хуго" });
  const a2 = seg({ id: "a2", speaker: "sys:1", startMs: 2_500, endMs: 4_000, text: "а сроки какие" });
  const b1 = seg({ id: "b1", speaker: "sys:2", startMs: 5_000, endMs: 6_000, text: "минуту" });
  const a3 = seg({ id: "a3", speaker: "sys:1", startMs: 7_000, endMs: 9_000, text: "айдос что по мобилке" });
  it("returns the segment right before when the same speaker said it", () => {
    expect(previousSameSpeaker([a1, a2, b1, a3], a2, labelOf)?.id).toBe("a1");
  });
  it("returns nothing if someone else spoke in between (the name was about something else)", () => {
    expect(previousSameSpeaker([a1, a2, b1, a3], a3, labelOf)).toBeNull();
    expect(previousSameSpeaker([a1, a2, b1, a3], b1, labelOf)).toBeNull();
  });
  it("ignores it if it ended too long ago", () => {
    expect(previousSameSpeaker([a1, a2], { ...a2, startMs: 30_000 }, labelOf)).toBeNull();
  });
  it("unlabelled segments match by source", () => {
    const n1 = seg({ id: "n1", speaker: null, source: "mic", startMs: 0, endMs: 1_000 });
    const n2 = seg({ id: "n2", speaker: null, source: "mic", startMs: 2_000, endMs: 3_000 });
    const s1 = seg({ id: "s1", speaker: null, source: "system", startMs: 1_000, endMs: 1_500 });
    expect(previousSameSpeaker([n1, n2], n2, labelOf)?.id).toBe("n1");
    expect(previousSameSpeaker([n1, s1, n2], n2, labelOf)).toBeNull();
  });
});

describe("SilenceWatch", () => {
  it("returns the question after the wait if nobody spoke", () => {
    const w = new SilenceWatch();
    const q = seg();
    w.arm(q, 1_000);
    expect(w.take(1_000 + SILENCE_WAIT_MS - 1)).toBeNull();
    expect(w.take(1_000 + SILENCE_WAIT_MS)).toBe(q);
    expect(w.armed).toBeNull();
  });
  it("any new speech cancels it; the question itself and empty text don't", () => {
    const w = new SilenceWatch();
    const q = seg();
    w.arm(q, 0);
    expect(w.onSpeech({ id: q.id, text: q.text })).toBe(false);
    expect(w.onSpeech({ id: "mic-9", text: "  " })).toBe(false);
    expect(w.onSpeech({ id: "mic-9", text: "ну" })).toBe(true);
    expect(w.take(10_000)).toBeNull();
  });
});

describe("suggestionsDue", () => {
  const base = {
    intervalMinutes: 4,
    nowMs: 10 * 60_000,
    sessionStartedAt: 0,
    lastRunAt: null,
    newFinals: 10,
    unnamed: 2,
    busy: false,
    hasApiKey: true,
  };
  it("runs at most once per rolling-summary interval", () => {
    expect(suggestionsDue({ ...base, nowMs: 4 * 60_000 - 1 })).toBe(false);
    expect(suggestionsDue({ ...base, nowMs: 4 * 60_000 })).toBe(true);
    expect(suggestionsDue({ ...base, lastRunAt: 7 * 60_000 })).toBe(false);
    expect(suggestionsDue({ ...base, intervalMinutes: 0, nowMs: 4 * 60_000 })).toBe(true);
  });
  it("only while unnamed speakers exist and something new was said", () => {
    expect(suggestionsDue({ ...base, unnamed: 0 })).toBe(false);
    expect(suggestionsDue({ ...base, newFinals: 1 })).toBe(false);
    expect(suggestionsDue({ ...base, busy: true })).toBe(false);
    expect(suggestionsDue({ ...base, hasApiKey: false })).toBe(false);
  });
});

describe("rollingSummaryDue", () => {
  const base = {
    minutes: 4,
    nowMs: 10 * 60_000,
    sessionStartedAt: 0,
    lastRunAt: null,
    newFinals: 3,
    busy: false,
    hasApiKey: true,
  };
  it("runs once the interval has passed since session start", () => {
    expect(rollingSummaryDue({ ...base, nowMs: 4 * 60_000 })).toBe(true);
    expect(rollingSummaryDue({ ...base, nowMs: 4 * 60_000 - 1 })).toBe(false);
  });
  it("measures from the previous run", () => {
    expect(rollingSummaryDue({ ...base, lastRunAt: 7 * 60_000 })).toBe(false);
    expect(rollingSummaryDue({ ...base, lastRunAt: 6 * 60_000 })).toBe(true);
  });
  it("is off at 0 minutes, when busy, without key or without new speech", () => {
    expect(rollingSummaryDue({ ...base, minutes: 0 })).toBe(false);
    expect(rollingSummaryDue({ ...base, busy: true })).toBe(false);
    expect(rollingSummaryDue({ ...base, hasApiKey: false })).toBe(false);
    expect(rollingSummaryDue({ ...base, newFinals: 0 })).toBe(false);
  });
});
