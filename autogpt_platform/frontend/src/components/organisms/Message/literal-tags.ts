interface HastNode {
  type: string;
  value?: string;
  tagName?: string;
  properties?: Record<string, unknown>;
  children?: HastNode[];
}

interface Options {
  keep: readonly string[];
}

// Runs before rehype-raw: a `<` in raw HTML that opens no kept element becomes
// `&lt;`, so the parser reads it as text instead of a tag the sanitiser drops.
export function rehypeLiteralTags({ keep }: Options) {
  const unkept = unkeptTagStart(keep);
  return function escapeUnkeptTags(tree: HastNode) {
    escapeRawChildren(tree, unkept);
  };
}

function escapeRawChildren(parent: HastNode, unkept: RegExp) {
  if (!parent.children) return;
  parent.children = parent.children.map((child) => {
    if (child.type !== "raw") {
      escapeRawChildren(child, unkept);
      return child;
    }
    // Spread: rehype-raw drops the text of a raw node that has no position.
    const escaped = {
      ...child,
      value: (child.value ?? "").replace(unkept, "&lt;"),
    };
    // A block that is now all text reads as a paragraph, like its neighbours.
    if (parent.type === "root" && !escaped.value.includes("<"))
      return paragraph(escaped);
    return escaped;
  });
}

function unkeptTagStart(keep: readonly string[]) {
  if (keep.length === 0) return /</g;
  return new RegExp(`<(?!/?(?:${keep.join("|")})(?=[\\s/>]))`, "gi");
}

function paragraph(child: HastNode): HastNode {
  return { type: "element", tagName: "p", properties: {}, children: [child] };
}
