import { memo, useMemo, type ReactNode } from "react";
import { parseMarkdown, type Block, type Inline, type ListBlock } from "../lib/markdown";

function renderInline(nodes: Inline[]): ReactNode[] {
  return nodes.map((n, i) => {
    switch (n.t) {
      case "text":
        return n.v;
      case "strong":
        return <strong key={i}>{renderInline(n.c)}</strong>;
      case "em":
        return <em key={i}>{renderInline(n.c)}</em>;
      case "code":
        return <code key={i}>{n.v}</code>;
      case "br":
        return <br key={i} />;
    }
  });
}

function renderList(list: ListBlock, key: number): ReactNode {
  const items = list.items.map((it, i) => (
    <li key={i}>
      {renderInline(it.c)}
      {it.children && renderList(it.children, 0)}
    </li>
  ));
  return list.t === "ol" ? (
    <ol key={key} start={list.start !== 1 ? list.start : undefined}>
      {items}
    </ol>
  ) : (
    <ul key={key}>{items}</ul>
  );
}

function renderBlocks(blocks: Block[]): ReactNode[] {
  return blocks.map((b, i) => {
    switch (b.t) {
      case "h": {
        // Card-level headings: never larger than h3 inside the panel.
        const Tag = (b.level <= 2 ? "h3" : "h4") as "h3" | "h4";
        return <Tag key={i}>{renderInline(b.c)}</Tag>;
      }
      case "p":
        return <p key={i}>{renderInline(b.c)}</p>;
      case "ul":
      case "ol":
        return renderList(b, i);
      case "quote":
        return <blockquote key={i}>{renderBlocks(b.c)}</blockquote>;
      case "hr":
        return <hr key={i} />;
      case "code":
        return (
          <pre key={i}>
            <code>{b.v}</code>
          </pre>
        );
      case "table":
        return (
          <div className="md-table" key={i}>
            <table>
              <thead>
                <tr>
                  {b.head.map((c, j) => (
                    <th key={j}>{renderInline(c)}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {b.rows.map((r, j) => (
                  <tr key={j}>
                    {r.map((c, k) => (
                      <td key={k}>{renderInline(c)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
    }
  });
}

export const Markdown = memo(function Markdown({ text, className }: { text: string; className?: string }) {
  const blocks = useMemo(() => parseMarkdown(text), [text]);
  return <div className={`md ${className ?? ""}`}>{renderBlocks(blocks)}</div>;
});
