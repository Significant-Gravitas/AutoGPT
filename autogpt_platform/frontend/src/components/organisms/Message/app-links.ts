// The chat's link hardening (rehype-harden in Streamdown) only accepts
// absolute URLs and paths starting with "/", "./" or "../", so a bare app path
// such as "library/agents/<id>" rendered as "[blocked]". Links whose first
// segment is one of the app's own top-level routes get the missing "/".
const APP_ROUTES = new Set([
  "library",
  "marketplace",
  "copilot",
  "build",
  "settings",
  "profile",
  "artifacts",
  "home",
  "team",
]);

const SCHEME = /^[a-z][a-z\d+.-]*:/i;

interface MarkdownNode {
  type: string;
  url?: string;
  children?: MarkdownNode[];
}

export function toAppPath(url: string): string {
  if (SCHEME.test(url) || /^[/#?.\\]/.test(url)) return url;
  const firstSegment = url.split(/[/?#]/, 1)[0];
  return APP_ROUTES.has(firstSegment) ? `/${url}` : url;
}

function rewriteLinks(node: MarkdownNode) {
  if (
    (node.type === "link" || node.type === "definition") &&
    typeof node.url === "string"
  ) {
    node.url = toAppPath(node.url);
  }
  node.children?.forEach(rewriteLinks);
}

// Streamdown keys its parse cache on plugin function names, so this one keeps
// a distinct name.
export function remarkAppLinks() {
  return function transformAppLinks(tree: MarkdownNode) {
    rewriteLinks(tree);
  };
}
