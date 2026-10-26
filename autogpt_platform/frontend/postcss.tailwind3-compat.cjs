/*
  Runs after @tailwindcss/postcss and puts back Tailwind 3's output for the
  utilities whose CSS Tailwind 4 changed in a way no theme value can undo.
  Each rewrite matches the shape of Tailwind 4's rule, so variants
  (`hover:`, `sm:`, `[&_x]:`) are covered too. Only the stylesheet Tailwind
  built is touched; other CSS (CSS modules, a library's own stylesheet)
  passes through, as it did with Tailwind 3.

  - `space-x/y-*` and `divide-x/y-*`: Tailwind 4 spaces children with
    `:where(.space-y-4 > :not(:last-child))` and a bottom margin. Tailwind 3
    used `.space-y-4 > :not([hidden]) ~ :not([hidden])` and a top margin,
    which outranks a child's own margin class and skips `hidden` children.
  - `translate-*`, `rotate-*`, `scale-*` and `transform`: Tailwind 4 uses the
    separate `translate`, `rotate` and `scale` properties. Tailwind 3 wrote one
    `transform`, which a keyframe or an inline `transform` replaced instead
    of adding to (the Dialog's slide-in classes rely on that), and which
    always made a stacking context.
  - `outline`, `outline-<width>` and `outline-none`: Tailwind 4's width
    classes also turn the outline on, and its `outline-none` removes it.
    Tailwind 3's width classes only set the width, `outline` only set the
    style, and `outline-none` drew a transparent 2px outline that forced
    colours mode shows.
  - Opacity modifiers (`bg-surface/50`, `bg-zinc-500/50`): Tailwind 4 mixes
    the colour with transparent in oklab. Tailwind 3 wrote the same colour
    with an alpha, `rgb(131 131 140 / 0.5)`, which this puts back so every
    pixel blends exactly as before.
  - `bg-gradient-to-*`: Tailwind 4 blends the colours in oklab. Tailwind 3
    left the gradient in the browser's default sRGB. And Tailwind 4's `to-*`
    also writes the gradient stops, so `to-*` without a `from-*` draws a
    transparent gradient; in Tailwind 3 only `from-*` and `via-*` wrote them,
    and the element had no background image.
  - Bare `transition`: Tailwind 4 hard-codes a longer property list (outline
    colour, gradient stops, `display`, `overlay` and more), so an outline or
    gradient change that used to snap would now animate.
  - Scrollbars: Tailwind 4 has its own `scrollbar-thumb-*`,
    `scrollbar-track-*` and `scrollbar-thin` utilities, under the same names
    as the tailwind-scrollbar classes (tailwind.scrollbar.ts), and they set
    `scrollbar-color` directly. Their declarations are dropped. And as
    tailwind-scrollbar did, a `hover:` (`focus:`, `active:`, ...) scrollbar
    colour sets `--scrollbar-thumb-hover` on the element itself, which the
    thumb's own `:hover` rule reads.
  - Shadows and rings: Tailwind 4 stacks five shadows (inset shadow, inset
    ring, ring offset, ring, shadow) and a ring sets no offset shadow.
    Tailwind 3 stacked three, and a ring drew its offset too, so the resolved
    `box-shadow` reads as it did.
  - `sr-only` / `not-sr-only`: Tailwind 4 hides with `clip-path`, Tailwind 3
    with `clip`.
  - Border widths set `border-style: var(--tw-border-style)` in Tailwind 4,
    so `file:border-0` gives the file button a style. Tailwind 3 set only the
    width and left the style to the preflight.
  - Text sizes set their line height outright: Tailwind 4 writes
    `var(--tw-leading, <size's line height>)`, so a `leading-*` class beats a
    later text size; in Tailwind 3 the later rule won.
  - Rule order: Tailwind 4 sorts `group-*`/`peer-*` first and `hover:`,
    `focus:`, `focus-visible:`, `active:`, `enabled:` and `disabled:` before
    `has-*`, `aria-*` and `data-*`. In this app's Tailwind 3 build they came
    after `data-*` (tailwind-scrollbar 3 registered them again), `dark:` came
    before the breakpoints and `*:` next to `first:`, so when two such classes
    on one element set the same property, the other one won. The utilities
    are put back in that order, sorted as Tailwind 3 sorted them: by all of a
    class's variants together, so `max-md:[&>svg]:size-5` follows
    `[&>svg]:size-4`, and then arbitrary properties after the other
    utilities of the same variants (`[animation-timing-function:...]` beats
    tailwindcss-animate's `ease-*`); `outline-none` also goes back in front of the
    outline width, offset and colour classes it used to come before.
  - `transition-none` sets no timing or duration, and `justify-self-start` /
    `-end` mean `start` / `end`, as in Tailwind 3.

  Remove a rewrite once no class in src relies on Tailwind 3's behaviour.
  Next.js and Storybook cache built CSS without watching this file, so after
  editing it delete `.next/cache` and `node_modules/.cache` before building.
*/

const SPACED_CHILDREN = /^:where\((.+) > :not\(:last-child\)\)$/;

// Tailwind 4 puts the gap after each child, Tailwind 3 before it, so each
// side maps to the opposite physical one (and `--tw-*-reverse` still flips
// them back).
const SPACED_SIDES = {
  "margin-block-start": "margin-bottom",
  "margin-block-end": "margin-top",
  "margin-inline-start": "margin-right",
  "margin-inline-end": "margin-left",
  "border-top-width": "border-bottom-width",
  "border-bottom-width": "border-top-width",
  "border-inline-start-width": "border-right-width",
  "border-inline-end-width": "border-left-width",
};

const TRANSFORM =
  "translate(var(--tw-translate-x), var(--tw-translate-y)) rotate(var(--tw-rotate)) skewX(var(--tw-skew-x, 0)) skewY(var(--tw-skew-y, 0)) scaleX(var(--tw-scale-x)) scaleY(var(--tw-scale-y))";

const TAILWIND_4_TRANSFORM =
  "var(--tw-rotate-x,) var(--tw-rotate-y,) var(--tw-rotate-z,) var(--tw-skew-x,) var(--tw-skew-y,)";

const ROLE_MIX =
  /color-mix\(in (?:oklab|srgb), rgb\((?:rgb\()?var\((--[\w-]+)\) \/ 1\)(?: \/ 1\))? ([\d.]+)%, transparent\)/g;
const HEX_MIX =
  /color-mix\(in (?:oklab|srgb), #([0-9a-fA-F]{6}|[0-9a-fA-F]{3}) +([\d.]+)%, transparent\)/g;
// A shadow colour is mixed once more with its `/<alpha>` modifier, 100%
// unless the class sets one.
const SHADOW_ALPHA_MIX =
  /color-mix\(in oklab, (rgb\([^()]*(?:\([^()]*\)[^()]*)*\)|#[0-9a-fA-F]{3,8}) var\(--tw-[\w-]+-alpha\), transparent\)/g;

function alpha(percent) {
  return String(Number((Number(percent) / 100).toFixed(6)));
}

function hexChannels(hex) {
  const full =
    hex.length === 3
      ? hex
          .split("")
          .map((c) => c + c)
          .join("")
      : hex;
  return [0, 2, 4].map((i) => parseInt(full.slice(i, i + 2), 16)).join(" ");
}

function restoreOpacity(decl) {
  if (!decl.value.includes("color-mix(")) return;
  decl.value = decl.value
    .replace(
      ROLE_MIX,
      (_, name, percent) => `rgb(var(${name}) / ${alpha(percent)})`,
    )
    .replace(
      HEX_MIX,
      (_, hex, percent) => `rgb(${hexChannels(hex)} / ${alpha(percent)})`,
    )
    .replace(SHADOW_ALPHA_MIX, "$1");
}

const TAILWIND_4_TRANSITION =
  "color, background-color, border-color, outline-color, text-decoration-color, fill, stroke, --tw-gradient-from, --tw-gradient-via, --tw-gradient-to, opacity, box-shadow, transform, translate, scale, rotate, filter, -webkit-backdrop-filter, backdrop-filter, display, content-visibility, overlay, pointer-events";
const TAILWIND_3_TRANSITION =
  "color, background-color, border-color, text-decoration-color, fill, stroke, opacity, box-shadow, transform, filter, backdrop-filter";

function restoreGradient(decl) {
  if (
    decl.prop === "--tw-gradient-position" &&
    decl.value.endsWith(" in oklab")
  ) {
    decl.value = decl.value.slice(0, -" in oklab".length);
  }
}

function restoreTransitionProperty(decl) {
  if (
    decl.prop === "transition-property" &&
    decl.value === TAILWIND_4_TRANSITION
  ) {
    decl.value = TAILWIND_3_TRANSITION;
  }
}

const NATIVE_SCROLLBAR_COLOR =
  "var(--tw-scrollbar-thumb) var(--tw-scrollbar-track)";
const SCROLLBAR_STATES =
  /:(hover|focus|focus-visible|active|enabled|disabled)$/;

function restoreScrollbar(rule) {
  rule.each((decl) => {
    if (decl.type !== "decl") return;
    if (
      decl.prop === "--tw-scrollbar-thumb" ||
      decl.prop === "--tw-scrollbar-track" ||
      (decl.prop === "scrollbar-color" && decl.value === NATIVE_SCROLLBAR_COLOR)
    ) {
      decl.remove();
    }
  });
  if (rule.nodes.length === 0) {
    rule.remove();
    return;
  }
  const decls = rule.nodes.filter((n) => n.type === "decl");
  if (
    decls.length === 0 ||
    !decls.every((d) => d.prop.startsWith("--scrollbar-")) ||
    !rule.selectors.every((s) => SCROLLBAR_STATES.test(s))
  ) {
    return;
  }
  const state = rule.selectors[0].match(SCROLLBAR_STATES)[1];
  rule.selectors = rule.selectors.map((s) => s.replace(SCROLLBAR_STATES, ""));
  for (const decl of decls) {
    if (!decl.prop.endsWith(`-${state}`)) decl.prop += `-${state}`;
  }
}

function restoreScreenReaderOnly(rule) {
  const decls = rule.nodes.filter((n) => n.type === "decl");
  const has = (prop, value) =>
    decls.some((d) => d.prop === prop && d.value === value);
  const clipPath = decls.find((d) => d.prop === "clip-path");
  if (!clipPath) return;
  if (
    has("width", "1px") &&
    has("white-space", "nowrap") &&
    clipPath.value === "inset(50%)"
  ) {
    clipPath.replaceWith({
      prop: "clip",
      value: "rect(0, 0, 0, 0)",
      important: clipPath.important,
    });
  } else if (
    has("position", "static") &&
    has("overflow", "visible") &&
    clipPath.value === "none"
  ) {
    clipPath.replaceWith({
      prop: "clip",
      value: "auto",
      important: clipPath.important,
    });
  }
}

const TAILWIND_4_SHADOWS =
  "var(--tw-inset-shadow), var(--tw-inset-ring-shadow), var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow)";
const TAILWIND_3_SHADOWS =
  "var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow)";
const RING_OFFSET_SHADOW =
  "var(--tw-ring-inset,) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color)";

function restoreShadows(rule) {
  rule.each((decl) => {
    if (decl.type !== "decl") return;
    if (decl.prop === "box-shadow" && decl.value === TAILWIND_4_SHADOWS) {
      decl.value = TAILWIND_3_SHADOWS;
    } else if (
      decl.prop === "--tw-ring-shadow" &&
      decl.value.startsWith("var(--tw-ring-inset,) 0 0 0 calc(") &&
      !rule.nodes.some(
        (n) => n.type === "decl" && n.prop === "--tw-ring-offset-shadow",
      )
    ) {
      decl.cloneBefore({
        prop: "--tw-ring-offset-shadow",
        value: RING_OFFSET_SHADOW,
      });
    }
  });
}

const JUSTIFY_SELF = { "flex-start": "start", "flex-end": "end" };
const TEXT_LEADING = /^var\(--tw-leading, (.+)\)$/;

function restoreDeclarations(rule) {
  const decls = rule.nodes.filter((n) => n.type === "decl");
  const noTransition = decls.some(
    (d) => d.prop === "transition-property" && d.value === "none",
  );
  const gradientEndOnly =
    decls.some((d) => d.prop === "--tw-gradient-to") &&
    !decls.some(
      (d) => d.prop === "--tw-gradient-from" || d.prop === "--tw-gradient-via",
    );
  for (const decl of decls) {
    if (
      /^border(-[a-z]+)*-style$/.test(decl.prop) &&
      decl.value === "var(--tw-border-style)"
    ) {
      decl.remove();
    } else if (
      noTransition &&
      /^transition-(timing-function|duration)$/.test(decl.prop) &&
      /^var\(--tw-(ease|duration),/.test(decl.value)
    ) {
      decl.remove();
    } else if (decl.prop === "line-height" && TEXT_LEADING.test(decl.value)) {
      decl.value = decl.value.replace(TEXT_LEADING, "$1");
    } else if (gradientEndOnly && decl.prop === "--tw-gradient-stops") {
      decl.remove();
    } else if (decl.prop === "justify-self" && JUSTIFY_SELF[decl.value]) {
      decl.value = JUSTIFY_SELF[decl.value];
    }
  }
  if (rule.nodes.length === 0) rule.remove();
}

function restoreSpacing(rule) {
  if (!rule.selectors.every((s) => SPACED_CHILDREN.test(s))) return;
  rule.selectors = rule.selectors.map((s) =>
    s.replace(SPACED_CHILDREN, "$1 > :not([hidden]) ~ :not([hidden])"),
  );
  rule.walkDecls((decl) => {
    if (/-style$/.test(decl.prop) && decl.value === "var(--tw-border-style)") {
      decl.remove();
    } else if (SPACED_SIDES[decl.prop]) {
      decl.prop = SPACED_SIDES[decl.prop];
    }
  });
}

function restoreTransform(rule) {
  let rewrote = false;
  rule.each((decl) => {
    if (decl.type !== "decl") return;
    const isTranslate =
      decl.prop === "translate" &&
      decl.value === "var(--tw-translate-x) var(--tw-translate-y)";
    const isScale =
      decl.prop === "scale" &&
      decl.value === "var(--tw-scale-x) var(--tw-scale-y)";
    const isTransform =
      decl.prop === "transform" && decl.value === TAILWIND_4_TRANSFORM;
    if (isTranslate || isScale || isTransform) {
      decl.prop = "transform";
      decl.value = TRANSFORM;
      rewrote = true;
    } else if (decl.prop === "rotate") {
      decl.cloneBefore({ prop: "--tw-rotate" });
      decl.prop = "transform";
      decl.value = TRANSFORM;
      rewrote = true;
    } else if (decl.prop === "scale" && !/\s/.test(decl.value)) {
      // An arbitrary `scale-[0.92]` sets the property directly.
      decl.cloneBefore({ prop: "--tw-scale-x" });
      decl.cloneBefore({ prop: "--tw-scale-y" });
      decl.prop = "transform";
      decl.value = TRANSFORM;
      rewrote = true;
    } else if (decl.prop === "--tw-scale-z") {
      decl.remove();
    }
  });
  return rewrote;
}

function restoreOutline(rule) {
  const decls = rule.nodes.filter((n) => n.type === "decl");
  const props = decls.map((d) => `${d.prop}:${d.value}`);
  if (
    props.length === 2 &&
    props[0] === "--tw-outline-style:none" &&
    props[1] === "outline-style:none"
  ) {
    const important = decls[0].important;
    decls[0].replaceWith({
      prop: "outline",
      value: "2px solid transparent",
      important,
    });
    decls[1].replaceWith({ prop: "outline-offset", value: "2px", important });
    return;
  }
  if (
    props.length === 2 &&
    props[0] === "outline-style:var(--tw-outline-style)" &&
    decls[1].prop === "outline-width"
  ) {
    if (decls[1].value === "medium") {
      // Bare `outline`: Tailwind 3 set only the style.
      decls[0].value = "solid";
      decls[1].remove();
    } else {
      decls[0].remove();
    }
  }
}

// Tailwind 3's variant order in this app, by category: the rule sort is
// stable, so within a category Tailwind 4's own order stands.
const PSEUDO_ELEMENTS = new Set([
  "file",
  "placeholder",
  "backdrop",
  "before",
  "after",
  "marker",
  "selection",
  "first-letter",
  "first-line",
]);
const INTERACTIONS = new Set([
  "hover",
  "focus",
  "focus-visible",
  "active",
  "enabled",
  "disabled",
]);
const AT_RULE_VARIANT =
  /^(sm|md|lg|xl|2xl|max-.+|min-.+|supports-.+|not-supports-.+|motion-safe|motion-reduce|print|contrast-.+|forced-colors|portrait|landscape)$/;
const STATE_KINDS = ["has", "aria", "data"];

function variantRank(variant) {
  if (variant === undefined) return 0;
  if (PSEUDO_ELEMENTS.has(variant)) return 1;
  if (variant === "first") return 2;
  if (variant === "*") return 2.1;
  const scoped = variant.match(/^(group|peer)-(.+)$/);
  const kind = (scoped ? scoped[2] : variant).match(/^(has|aria|data)-/);
  if (kind) {
    const scope = scoped ? (scoped[1] === "group" ? 1 : 2) : 0;
    return 4 + (STATE_KINDS.indexOf(kind[1]) * 3 + scope) / 10;
  }
  if (scoped) return 3 + (scoped[1] === "peer" ? 0.5 : 0);
  if (INTERACTIONS.has(variant)) return 5;
  if (
    /^(supports-|not-supports-|motion-|print$|contrast-|forced-colors$)/.test(
      variant,
    )
  )
    return 6;
  if (variant === "dark") return 6.5;
  if (AT_RULE_VARIANT.test(variant)) return 7;
  if (variant === "rtl" || variant === "ltr") return 8;
  if (variant.startsWith("[")) return 9;
  return 2.2;
}

function unescapeClass(name) {
  return name.replace(/\\([0-9a-fA-F]{1,6}) ?|\\(.)/g, (_, hex, ch) =>
    hex ? String.fromCodePoint(parseInt(hex, 16)) : ch,
  );
}

// `hover:data-[x=y]:bg-red` -> ["hover", "data-[x=y]"]
function variantsOf(node) {
  return splitClass(node)?.variants ?? null;
}

// `md:[mask:none]` -> true: an arbitrary property, which Tailwind 3 put after
// the other utilities of the same variants.
function isArbitraryProperty(node) {
  const utility = splitClass(node)?.utility ?? "";
  return /^!?\[[a-zA-Z-]+:.+\]!?$/.test(utility);
}

function splitClass(node) {
  let rule = node;
  if (node.type === "atrule") {
    rule = null;
    node.walkRules((r) => {
      rule ??= r;
    });
  }
  if (!rule || rule.type !== "rule") return null;
  const match = rule.selector.match(/\.((?:\\[0-9a-fA-F]{1,6} ?|\\.|[\w-])+)/);
  if (!match) return null;
  const parts = [];
  let depth = 0;
  let current = "";
  for (const ch of unescapeClass(match[1])) {
    if (ch === "[" || ch === "(") depth++;
    if (ch === "]" || ch === ")") depth--;
    if (ch === ":" && depth === 0) {
      parts.push(current);
      current = "";
    } else current += ch;
  }
  return { variants: parts, utility: current };
}

// Tailwind 3 gave every variant a bit, in the order above, and sorted a
// class by all of its variants ORed together. So the class's highest variant
// decides first, then the next highest, and a class with one more variant
// sorts after one without it (`max-md:[&>svg]:size-5` after `[&>svg]:size-4`).
function sortKey(node) {
  return {
    variants: (variantsOf(node) ?? []).map(variantRank).sort((a, b) => b - a),
    arbitrary: isArbitraryProperty(node) ? 1 : 0,
  };
}

function compareKeys(a, b) {
  const [va, vb] = [a.variants, b.variants];
  for (let i = 0; i < Math.min(va.length, vb.length); i++) {
    if (va[i] !== vb[i]) return va[i] - vb[i];
  }
  return va.length - vb.length || a.arbitrary - b.arbitrary;
}

// One rule per at-rule block, so each sorts by its own variants.
function splitAtRules(nodes) {
  return nodes.flatMap((node) => {
    if (
      node.type !== "atrule" ||
      !/^(media|supports|container)$/.test(node.name) ||
      !node.nodes ||
      node.nodes.length < 2
    ) {
      return [node];
    }
    return node.nodes.map((child) => {
      const wrapper = node.clone({ nodes: [] });
      wrapper.append(child.clone());
      return wrapper;
    });
  });
}

function mergeAdjacentAtRules(nodes) {
  const merged = [];
  for (const node of nodes) {
    const last = merged[merged.length - 1];
    if (
      node.type === "atrule" &&
      last?.type === "atrule" &&
      last.name === node.name &&
      last.params === node.params
    ) {
      last.append(node.nodes.map((child) => child.clone()));
    } else merged.push(node);
  }
  return merged;
}

function sortNodes(nodes) {
  return nodes
    .map((node, index) => ({ node, index, key: sortKey(node) }))
    .sort((a, b) => compareKeys(a.key, b.key) || a.index - b.index)
    .map((entry) => entry.node);
}

const OUTLINE_SETTINGS = new Set([
  "outline-width",
  "outline-offset",
  "outline-color",
]);

function isOutlineNone(node) {
  return (
    node.type === "rule" &&
    node.nodes.some(
      (d) =>
        d.type === "decl" &&
        d.prop === "outline" &&
        d.value === "2px solid transparent",
    )
  );
}

function variantPrefix(node) {
  const variants = variantsOf(node);
  return variants ? variants.join(":") : null;
}

// Tailwind 3 emitted `outline-none` before the outline width, offset and
// colour utilities of the same variants. `siblings` are utility rules of one
// parent, in order; rules outside the utility block are never moved.
function moveOutlineNone(siblings) {
  const list = [...siblings];
  for (const node of [...list]) {
    if (!isOutlineNone(node)) continue;
    const prefix = variantPrefix(node);
    const at = list.indexOf(node);
    const before = list.find(
      (other, index) =>
        index < at &&
        other.type === "rule" &&
        variantPrefix(other) === prefix &&
        other.nodes.some(
          (d) => d.type === "decl" && OUTLINE_SETTINGS.has(d.prop),
        ),
    );
    if (!before) continue;
    before.before(node);
    list.splice(at, 1);
    list.splice(list.indexOf(before), 0, node);
  }
}

function restoreRuleOrder(root) {
  const start = root.nodes.findIndex(
    (n) => n.type === "comment" && n.text.includes("tailwind-utilities:start"),
  );
  const end = root.nodes.findIndex(
    (n) => n.type === "comment" && n.text.includes("tailwind-utilities:end"),
  );
  if (start < 0 || end < 0) return;
  const block = root.nodes.slice(start + 1, end);
  const sorted = mergeAdjacentAtRules(sortNodes(splitAtRules(block)));
  for (const node of block) node.remove();
  root.nodes[start].after(sorted);
  for (const node of sorted) {
    if (node.type === "atrule" && node.nodes) moveOutlineNone(node.nodes);
  }
  moveOutlineNone(sorted.filter((node) => node.type === "rule"));
  root.nodes[start].remove();
  root.nodes
    .find(
      (n) => n.type === "comment" && n.text.includes("tailwind-utilities:end"),
    )
    ?.remove();
}

module.exports = () => ({
  postcssPlugin: "tailwind3-compat",
  Once(root, { postcss }) {
    const banner = root.first;
    if (banner?.type !== "comment" || !banner.text.includes("tailwindcss v4")) {
      return;
    }
    let usesRotate = false;
    root.walkRules((rule) => {
      if (rule.parent?.type === "atrule" && rule.parent.name === "keyframes") {
        return;
      }
      restoreSpacing(rule);
      if (restoreTransform(rule)) usesRotate = true;
      restoreOutline(rule);
      restoreScrollbar(rule);
      restoreScreenReaderOnly(rule);
      restoreDeclarations(rule);
      restoreShadows(rule);
    });
    root.walkDecls((decl) => {
      restoreOpacity(decl);
      restoreTransitionProperty(decl);
      restoreGradient(decl);
    });
    restoreRuleOrder(root);
    if (usesRotate) {
      // Tailwind 3 reset these on every element; registering them keeps a
      // parent's rotation from reaching its children the same way.
      root.append(
        postcss.atRule({
          name: "property",
          params: "--tw-rotate",
          nodes: [
            postcss.decl({ prop: "syntax", value: '"*"' }),
            postcss.decl({ prop: "inherits", value: "false" }),
            postcss.decl({ prop: "initial-value", value: "0" }),
          ],
        }),
      );
    }
  },
});
module.exports.postcss = true;
