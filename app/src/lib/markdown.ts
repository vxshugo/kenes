/**
 * A small Markdown subset for Claude's hints and summaries: headings, paragraphs,
 * bullet/numbered lists (one nesting level), blockquotes, rules, fenced code, simple
 * tables; inline bold, italic, code. No HTML passthrough, so nothing needs sanitizing.
 */

export type Inline =
  | { t: "text"; v: string }
  | { t: "strong"; c: Inline[] }
  | { t: "em"; c: Inline[] }
  | { t: "code"; v: string }
  | { t: "br" };

export type ListItem = { c: Inline[]; children: ListBlock | null };
export type ListBlock = { t: "ul" | "ol"; items: ListItem[]; start: number };

export type Block =
  | { t: "h"; level: 1 | 2 | 3 | 4; c: Inline[] }
  | { t: "p"; c: Inline[] }
  | ListBlock
  | { t: "quote"; c: Block[] }
  | { t: "hr" }
  | { t: "code"; v: string }
  | { t: "table"; head: Inline[][]; rows: Inline[][][] };

const BULLET = /^(\s*)([-*+•])\s+(.*)$/;
const ORDERED = /^(\s*)(\d{1,3})[.)]\s+(.*)$/;
const HEADING = /^(#{1,6})\s+(.*?)\s*#*\s*$/;
const HR = /^\s*([-*_])(\s*\1){2,}\s*$/;
const WORD = /[\p{L}\p{N}]/u;
const TABLE_SEP = /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/;

export function parseInline(src: string): Inline[] {
  const out: Inline[] = [];
  let buf = "";
  const flush = () => {
    if (buf) out.push({ t: "text", v: buf });
    buf = "";
  };
  let i = 0;
  while (i < src.length) {
    const ch = src[i];
    if (ch === "\\" && i + 1 < src.length && /[\\`*_#>|-]/.test(src[i + 1])) {
      buf += src[i + 1];
      i += 2;
      continue;
    }
    if (ch === "`") {
      const end = src.indexOf("`", i + 1);
      if (end > i + 1) {
        flush();
        out.push({ t: "code", v: src.slice(i + 1, end) });
        i = end + 1;
        continue;
      }
    }
    if ((ch === "*" || ch === "_") && src[i + 1] === ch) {
      const end = src.indexOf(ch + ch, i + 2);
      if (end > i + 2) {
        flush();
        out.push({ t: "strong", c: parseInline(src.slice(i + 2, end)) });
        i = end + 2;
        continue;
      }
    }
    if (ch === "*" || ch === "_") {
      const prev = src[i - 1];
      const next = src[i + 1];
      // "_" inside words (snake_case) and a lone "*" before a space are literal.
      const opens = next !== undefined && next !== " " && (ch === "*" || !prev || !WORD.test(prev));
      if (opens) {
        let end = src.indexOf(ch, i + 1);
        while (end !== -1 && (src[end - 1] === " " || src[end + 1] === ch || (ch === "_" && WORD.test(src[end + 1] ?? "")))) {
          end = src.indexOf(ch, end + (src[end + 1] === ch ? 2 : 1));
        }
        if (end > i + 1) {
          flush();
          out.push({ t: "em", c: parseInline(src.slice(i + 1, end)) });
          i = end + 1;
          continue;
        }
      }
    }
    buf += ch;
    i++;
  }
  flush();
  return out;
}

function inlineLines(lines: string[]): Inline[] {
  const out: Inline[] = [];
  lines.forEach((l, i) => {
    if (i > 0) out.push({ t: "br" });
    out.push(...parseInline(l.trim()));
  });
  return out;
}

function splitRow(line: string): string[] {
  let s = line.trim();
  if (s.startsWith("|")) s = s.slice(1);
  if (s.endsWith("|")) s = s.slice(0, -1);
  return s.split("|").map((c) => c.trim());
}

function indentOf(s: string): number {
  return s.replace(/\t/g, "    ").length;
}

export function parseMarkdown(src: string): Block[] {
  const lines = src.replace(/\r\n?/g, "\n").split("\n");
  const blocks: Block[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      i++;
      continue;
    }
    if (/^\s*```/.test(line)) {
      const body: string[] = [];
      i++;
      while (i < lines.length && !/^\s*```/.test(lines[i])) body.push(lines[i++]);
      i++;
      blocks.push({ t: "code", v: body.join("\n") });
      continue;
    }
    const h = HEADING.exec(line);
    if (h) {
      blocks.push({ t: "h", level: Math.min(4, h[1].length) as 1 | 2 | 3 | 4, c: parseInline(h[2]) });
      i++;
      continue;
    }
    if (HR.test(line)) {
      blocks.push({ t: "hr" });
      i++;
      continue;
    }
    if (/^\s*>/.test(line)) {
      const body: string[] = [];
      while (i < lines.length && /^\s*>/.test(lines[i])) body.push(lines[i++].replace(/^\s*>\s?/, ""));
      blocks.push({ t: "quote", c: parseMarkdown(body.join("\n")) });
      continue;
    }
    if (line.includes("|") && i + 1 < lines.length && TABLE_SEP.test(lines[i + 1])) {
      const head = splitRow(line).map(parseInline);
      i += 2;
      const rows: Inline[][][] = [];
      while (i < lines.length && lines[i].includes("|") && lines[i].trim()) rows.push(splitRow(lines[i++]).map(parseInline));
      blocks.push({ t: "table", head, rows });
      continue;
    }
    if (BULLET.test(line) || ORDERED.test(line)) {
      const [list, next] = parseList(lines, i);
      blocks.push(list);
      i = next;
      continue;
    }
    const para: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() &&
      !HEADING.test(lines[i]) &&
      !BULLET.test(lines[i]) &&
      !ORDERED.test(lines[i]) &&
      !/^\s*>/.test(lines[i]) &&
      !/^\s*```/.test(lines[i]) &&
      !HR.test(lines[i])
    ) {
      para.push(lines[i++]);
    }
    blocks.push({ t: "p", c: inlineLines(para) });
  }
  return blocks;
}

function parseList(lines: string[], from: number): [ListBlock, number] {
  const first = BULLET.exec(lines[from]) ?? ORDERED.exec(lines[from])!;
  const baseIndent = indentOf(first[1]);
  const ordered = !BULLET.test(lines[from]);
  const list: ListBlock = { t: ordered ? "ol" : "ul", items: [], start: ordered ? Number(first[2]) : 1 };
  let i = from;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      // A blank line ends the list unless the next line continues it.
      const next = lines[i + 1];
      if (next && (BULLET.test(next) || ORDERED.test(next)) && indentOf((BULLET.exec(next) ?? ORDERED.exec(next))![1]) >= baseIndent) {
        i++;
        continue;
      }
      break;
    }
    const m = BULLET.exec(line) ?? ORDERED.exec(line);
    if (m) {
      const indent = indentOf(m[1]);
      if (indent < baseIndent) break;
      if (indent > baseIndent && list.items.length) {
        const [child, next] = parseList(lines, i);
        const parent = list.items[list.items.length - 1];
        // A second nested run under the same item (other indent or list type) extends the first.
        if (parent.children) parent.children.items.push(...child.items);
        else parent.children = child;
        i = next;
        continue;
      }
      const isOrdered = !BULLET.test(line);
      if (isOrdered !== ordered) break;
      list.items.push({ c: parseInline(m[3]), children: null });
      i++;
      continue;
    }
    // Lazy continuation of the previous item.
    if (indentOf(/^\s*/.exec(line)![0]) > baseIndent && list.items.length) {
      const item = list.items[list.items.length - 1];
      item.c.push({ t: "br" }, ...parseInline(line.trim()));
      i++;
      continue;
    }
    break;
  }
  return [list, i];
}
