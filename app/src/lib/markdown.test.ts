import { describe, expect, it } from "vitest";
import { parseInline, parseMarkdown, type ListBlock } from "./markdown";

describe("parseInline", () => {
  it("parses bold, italic and code", () => {
    expect(parseInline("**Главное:** срок *пятница* и `sla`")).toEqual([
      { t: "strong", c: [{ t: "text", v: "Главное:" }] },
      { t: "text", v: " срок " },
      { t: "em", c: [{ t: "text", v: "пятница" }] },
      { t: "text", v: " и " },
      { t: "code", v: "sla" },
    ]);
  });
  it("leaves snake_case and stray stars alone", () => {
    expect(parseInline("user_id и 2 * 3")).toEqual([{ t: "text", v: "user_id и 2 * 3" }]);
  });
});

describe("parseMarkdown", () => {
  it("parses a final summary", () => {
    const md = "## Итоги\nРелиз в пятницу.\n\n## Задачи\n- **Техлид** — оценка — четверг\n  - детали\n- **ПМ** — план — не назван\n\n---\n1. раз\n2. два";
    const b = parseMarkdown(md);
    expect(b.map((x) => x.t)).toEqual(["h", "p", "h", "ul", "hr", "ol"]);
    const ul = b[3] as ListBlock;
    expect(ul.items).toHaveLength(2);
    expect(ul.items[0].children?.items).toHaveLength(1);
  });
  it("parses tables and quotes", () => {
    const b = parseMarkdown("| кто | что |\n|---|---|\n| я | оценка |\n\n> цитата");
    expect(b[0]).toMatchObject({ t: "table", rows: [[[{ t: "text", v: "я" }], [{ t: "text", v: "оценка" }]]] });
    expect(b[1].t).toBe("quote");
  });
  it("keeps line breaks inside paragraphs", () => {
    const b = parseMarkdown("Сәлеметсіз бе\n*Здравствуйте*");
    expect(b).toHaveLength(1);
    expect(b[0]).toMatchObject({ t: "p" });
  });
});
