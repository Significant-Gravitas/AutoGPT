// Shared DOM serialiser: inlines computed styles, maps colours to Paper tokens.
import { readFileSync } from "node:fs";

export function buildColorMap(tokensPath) {
  const tokens = JSON.parse(readFileSync(tokensPath, "utf8"));
  const colorByHex = {};
  for (const t of tokens) {
    if (t.type !== "color") continue;
    const v = String(t.value).toUpperCase();
    if (v.startsWith("#") && !colorByHex[v]) colorByHex[v] = t.name;
  }
  return colorByHex;
}

// __ROOT_SELECTOR__ is replaced per caller; __COLORS__ with the colour map JSON.
export const SERIALIZE = `(() => {
  const PROPS = ["display","flex-direction","flex-wrap","justify-content","align-items","align-self","align-content","flex-grow","flex-shrink","flex-basis","order","gap","row-gap","column-gap","grid-template-columns","grid-template-rows","grid-auto-flow","grid-column","grid-row","width","height","min-width","min-height","max-width","max-height","padding-top","padding-right","padding-bottom","padding-left","margin-top","margin-right","margin-bottom","margin-left","border-top-width","border-right-width","border-bottom-width","border-left-width","border-top-style","border-right-style","border-bottom-style","border-left-style","border-top-color","border-right-color","border-bottom-color","border-left-color","border-top-left-radius","border-top-right-radius","border-bottom-left-radius","border-bottom-right-radius","background-color","background-image","background-size","background-position","background-repeat","color","font-family","font-size","font-weight","font-style","line-height","letter-spacing","text-align","text-transform","text-decoration-line","text-decoration-color","text-overflow","white-space","word-break","overflow-wrap","overflow-x","overflow-y","opacity","box-shadow","position","top","right","bottom","left","z-index","inset","vertical-align","list-style-type","object-fit","aspect-ratio","box-sizing","visibility","backdrop-filter","filter","mix-blend-mode","text-wrap","font-variant-numeric","-webkit-line-clamp","-webkit-box-orient","background-clip","-webkit-background-clip","-webkit-text-fill-color"];
  const iframe = document.createElement("iframe");
  iframe.style.cssText = "position:absolute;width:0;height:0;border:0;visibility:hidden";
  document.body.appendChild(iframe);
  const doc = iframe.contentDocument;
  const defaults = {};
  function defaultsFor(tag) {
    if (!defaults[tag]) {
      const el = doc.createElement(tag);
      doc.body.appendChild(el);
      const cs = iframe.contentWindow.getComputedStyle(el);
      const d = {};
      for (const p of PROPS) d[p] = cs.getPropertyValue(p);
      defaults[tag] = d;
    }
    return defaults[tag];
  }
  const COLOR_PROPS = new Set(["color","background-color","border-top-color","border-right-color","border-bottom-color","border-left-color","text-decoration-color"]);
  const colorByHex = __COLORS__;
  function rgbToToken(v) {
    const m = v.match(/^rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)(?:,\\s*([\\d.]+))?\\)$/);
    if (!m) return v;
    if (m[4] !== undefined && Number(m[4]) === 0) return "transparent";
    if (m[4] !== undefined && Number(m[4]) !== 1) return v;
    const hex = "#" + [m[1],m[2],m[3]].map(n => Number(n).toString(16).padStart(2,"0").toUpperCase()).join("");
    return colorByHex[hex] ? "var(" + colorByHex[hex] + ")" : hex;
  }
  const esc = s => s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
  const escAttr = s => s.replace(/&/g,"&amp;").replace(/"/g,"&quot;");
  const VOID = new Set(["img","br","hr"]);
  const SKIP = new Set(["script","style","noscript","template","link","meta"]);
  const TAGMAP = { button: "div", input: "div", textarea: "div", select: "div", a: "div", label: "span", ul: "div", ol: "div", li: "div", nav: "div", section: "div", article: "div", header: "div", footer: "div", main: "div", aside: "div", form: "div", fieldset: "div", legend: "span", h1: "div", h2: "div", h3: "div", h4: "div", h5: "div", h6: "div", p: "div", table: "div", thead: "div", tbody: "div", tfoot: "div", tr: "div", td: "div", th: "div", code: "span", kbd: "span", strong: "span", em: "span", b: "span", i: "span", small: "span", dl: "div", dt: "div", dd: "div", picture: "div", figure: "div" };
  const KNOWN_FONTS = ["Poppins", "Geist", "Geist Mono", "GeistMono", "Inter", "Arial", "Helvetica", "Times New Roman", "Georgia", "Courier New"];
  function mapFont(v) {
    const fams = v.split(",").map(f => f.replace(/"/g, "").trim());
    for (const f of fams) { if (f === "GeistMono") return "Geist Mono"; if (KNOWN_FONTS.includes(f)) return f; }
    const lower = v.toLowerCase();
    if (/mono|courier|menlo|consolas|cascadia/.test(lower)) return "Geist Mono";
    return "Geist";
  }
  const TABLE_DISPLAY = { table: "flex", "inline-table": "flex", "table-row-group": "flex", "table-header-group": "flex", "table-footer-group": "flex", "table-row": "flex", "table-cell": "flex", "table-caption": "block" };
  function isTextLeaf(el) {
    let hasText = false;
    for (const n of el.childNodes) {
      if (n.nodeType === 3 && n.textContent.trim()) hasText = true;
      else if (n.nodeType === 1) { const d = getComputedStyle(n).display; if (d !== "inline" && d !== "none") return false; }
    }
    return hasText;
  }
  function styleFor(el, outTag) {
    const cs = getComputedStyle(el);
    const d = defaultsFor(outTag || "div");
    const parts = [];
    let gradientText = false;
    const bgClip = cs.getPropertyValue("-webkit-background-clip") || cs.getPropertyValue("background-clip");
    if (bgClip === "text" && cs.backgroundImage.startsWith("linear-gradient")) gradientText = true;
    for (const p of PROPS) {
      let v = cs.getPropertyValue(p);
      if (!v || v === d[p]) continue;
      if (p === "display" && TABLE_DISPLAY[v]) v = TABLE_DISPLAY[v];
      if (p === "background-image") {
        if (gradientText) continue;
        if (v.startsWith("url(") && !v.includes("data:")) { parts.push("background-image:" + v); continue; }
        if (v.startsWith("linear-gradient") || v.startsWith("radial-gradient")) { parts.push("background-image:" + v.replace(/rgba?\\([^)]*\\)/g, m => rgbToToken(m))); continue; }
        continue;
      }
      if (p === "background-clip" || p === "-webkit-background-clip" || p === "-webkit-text-fill-color") continue;
      if (p === "color" && gradientText) { const first = cs.backgroundImage.match(/rgba?\\([^)]*\\)/); v = first ? first[0] : v; }
      if (COLOR_PROPS.has(p)) v = rgbToToken(v);
      if (p === "font-family") v = mapFont(v);
      if ((p === "width" || p === "height") && (v === "auto" || v.endsWith("%"))) continue;
      if (p === "position" && v === "static") continue;
      if (p === "box-shadow" && /^(rgba\\(0, 0, 0, 0\\) 0px 0px 0px 0px(, )?)+$/.test(v)) continue;
      parts.push(p + ":" + v);
    }
    const tag = el.tagName.toLowerCase();
    const disp = cs.display;
    // Table structure → flex rows and cells with their measured widths.
    if (tag === "tr") { parts.push("display:flex", "flex-direction:row", "align-items:stretch", "width:" + Math.round(el.getBoundingClientRect().width) + "px"); }
    if (tag === "td" || tag === "th") { const r = el.getBoundingClientRect(); parts.push("display:flex", "align-items:center", "flex-shrink:0", "box-sizing:border-box", "width:" + Math.round(r.width) + "px", "min-height:" + Math.round(r.height) + "px"); }
    if (tag === "table" || tag === "thead" || tag === "tbody" || tag === "tfoot") { parts.push("display:flex", "flex-direction:column"); }
    // Text leaves keep Chrome's measured box so Paper never re-wraps them.
    if (disp !== "contents" && isTextLeaf(el)) {
      const r = el.getBoundingClientRect();
      const lh = parseFloat(cs.lineHeight) || parseFloat(cs.fontSize) * 1.3;
      const singleLine = r.height < lh * 1.6 && !/^pre/.test(cs.whiteSpace);
      if (disp !== "inline" && !parts.some(x => x.startsWith("width:"))) parts.push("width:" + (Math.ceil(r.width) + 2) + "px");
      if (singleLine && !parts.some(x => x.startsWith("white-space:"))) parts.push("white-space:nowrap");
      if (disp !== "inline" && !parts.some(x => x.startsWith("flex-shrink:")) && cs.flexShrink !== "0") parts.push("flex-shrink:0");
    }
    // Paper has no viewport, so fixed/absolute boxes become plain absolute boxes
    // with explicit left/top/width/height measured by Chrome.
    if (cs.position === "fixed" || cs.position === "absolute" || cs.position === "sticky") {
      const r = el.getBoundingClientRect();
      let ox = 0, oy = 0;
      if (cs.position !== "fixed") {
        const op = el.offsetParent;
        if (op && op !== document.body && op !== document.documentElement) {
          const pr = op.getBoundingClientRect(); const pcs = getComputedStyle(op);
          ox = pr.left + (parseFloat(pcs.borderLeftWidth) || 0); oy = pr.top + (parseFloat(pcs.borderTopWidth) || 0);
        }
      }
      const kept = parts.filter(x => !/^(position|top|right|bottom|left|inset|width|height|min-width|min-height|max-width|max-height):/.test(x));
      kept.push("position:absolute", "left:" + Math.round(r.left - ox) + "px", "top:" + Math.round(r.top - oy) + "px", "width:" + Math.round(r.width) + "px", "height:" + Math.round(r.height) + "px", "box-sizing:border-box");
      return kept.join(";");
    }
    return parts.join(";");
  }
  function ser(node, depth) {
    if (node.nodeType === 3) {
      const t = node.textContent;
      if (!t.trim()) return "";
      const ws = node.parentElement ? getComputedStyle(node.parentElement).whiteSpace : "normal";
      if (/^pre/.test(ws)) return esc(t);
      return esc(t.replace(/\\s+/g, " "));
    }
    if (node.nodeType !== 1) return "";
    const el = node;
    const tag = el.tagName.toLowerCase();
    if (SKIP.has(tag)) return "";
    const cs = getComputedStyle(el);
    if (cs.display === "none" || cs.visibility === "hidden" || parseFloat(cs.opacity) === 0) return "";
    { const rr = el.getBoundingClientRect(); if ((cs.position === "fixed" || cs.position === "absolute") && (rr.width === 0 || rr.height === 0)) return ""; }
    if (el.getAttribute("aria-hidden") === "true" && tag !== "svg" && !el.querySelector("svg") && !el.textContent.trim()) return "";
    if (tag === "svg") {
      const r = el.getBoundingClientRect();
      const c = el.cloneNode(true);
      c.removeAttribute("class");
      // SVG children styled via classes (Tailwind fill-*/stroke-*) need their computed paint inlined.
      const srcEls = el.querySelectorAll("*"); const dstEls = c.querySelectorAll("*");
      for (let i = 0; i < srcEls.length && i < dstEls.length; i++) {
        const se = srcEls[i], de = dstEls[i];
        if (!se.getAttribute("class") && !se.getAttribute("style")) continue;
        const scs = getComputedStyle(se);
        const st = [];
        for (const p of ["fill", "stroke", "stroke-width", "opacity", "fill-opacity", "stroke-opacity", "stroke-dasharray", "stroke-linecap", "stroke-linejoin"]) {
          let v = scs.getPropertyValue(p); if (!v) continue;
          if ((p === "fill" || p === "stroke") && v !== "none") v = rgbToToken(v);
          st.push(p + ":" + v);
        }
        const tr = scs.transform; if (tr && tr !== "none") st.push("transform:" + tr + ";transform-origin:" + scs.transformOrigin);
        de.removeAttribute("class"); de.setAttribute("style", st.join(";"));
      }
      c.setAttribute("width", Math.round(r.width));
      c.setAttribute("height", Math.round(r.height));
      c.setAttribute("style", "color:" + rgbToToken(cs.color) + ";flex-shrink:0;display:block");
      return c.outerHTML;
    }
    if (tag === "img") {
      const r = el.getBoundingClientRect();
      return '<img src="' + escAttr(el.currentSrc || el.src) + '" data-src="' + escAttr(el.currentSrc || el.src) + '" style="' + styleFor(el, "img") + ';width:' + Math.round(r.width) + 'px;height:' + Math.round(r.height) + 'px" />';
    }
    if (tag === "video" || tag === "canvas" || tag === "iframe") {
      const r = el.getBoundingClientRect();
      return '<div style="' + styleFor(el, "div") + ';width:' + Math.round(r.width) + 'px;height:' + Math.round(r.height) + 'px;background-color:#EFEFF0"></div>';
    }
    const out = TAGMAP[tag] || (tag === "span" || tag === "div" ? tag : "div");
    let style = styleFor(el, out);
    let inner = "";
    if (tag === "input" || tag === "textarea") {
      const val = el.value || el.getAttribute("placeholder") || "";
      if (!el.value && el.getAttribute("placeholder")) {
        const pc = getComputedStyle(el, "::placeholder").color;
        inner = '<span style="color:' + rgbToToken(pc) + '">' + esc(val) + "</span>";
      } else inner = esc(val);
      if (el.type === "checkbox" || el.type === "radio") inner = "";
      style += ";display:flex;align-items:center;flex-shrink:0";
    } else if (tag === "select") {
      inner = esc(el.options[el.selectedIndex]?.text || "");
    } else if (isTextLeaf(el) && [...el.children].some(c => getComputedStyle(c).display === "inline") && el.getBoundingClientRect().height < (parseFloat(cs.lineHeight) || parseFloat(cs.fontSize) * 1.3) * 1.6) {
      // Single-line rich text: emit each run as its own span so Paper keeps per-run colour, weight and underline.
      for (const ch of el.childNodes) {
        if (ch.nodeType === 3) { if (ch.textContent) inner += '<span style="white-space:pre">' + esc(ch.textContent.replace(/\\s+/g, " ")) + "</span>"; }
        else if (ch.nodeType === 1) inner += ser(ch, depth + 1);
      }
      style += ";display:flex;flex-direction:row;align-items:baseline;white-space:pre";
      if (cs.textAlign === "center") style += ";justify-content:center"; else if (cs.textAlign === "right" || cs.textAlign === "end") style += ";justify-content:flex-end";
    } else {
      const before = getComputedStyle(el, "::before"); const after = getComputedStyle(el, "::after");
      if (before.content && before.content !== "none" && before.content !== "normal" && before.content !== '""') inner += esc(before.content.replace(/^"|"$/g, ""));
      for (const ch of el.childNodes) inner += ser(ch, depth + 1);
      if (after.content && after.content !== "none" && after.content !== "normal" && after.content !== '""') inner += esc(after.content.replace(/^"|"$/g, ""));
    }
    const name = el.getAttribute("data-paper-name") || el.getAttribute("aria-label") || "";
    const nameAttr = name ? ' data-name="' + escAttr(name) + '"' : "";
    return "<" + out + nameAttr + ' style="' + style + '">' + inner + "</" + out + ">";
  }
  // Unwrap Storybook decorators / centering wrappers: descend while there is a
  // single element child that carries no text of its own.
  let rootEl = document.querySelector(__ROOT_SELECTOR__);
  for (let i = 0; i < 4; i++) {
    const els = [...rootEl.children].filter(c => c.nodeType === 1);
    const ownText = [...rootEl.childNodes].some(n => n.nodeType === 3 && n.textContent.trim());
    if (els.length === 1 && !ownText && !["SVG","IMG","BUTTON","INPUT","A"].includes(els[0].tagName) && els[0].children.length > 0) rootEl = els[0]; else break;
  }
  const kids = [...rootEl.childNodes].filter(n => n.nodeType === 1 || (n.nodeType === 3 && n.textContent.trim()));
  let l = Infinity, t = Infinity, r = 0, b = 0;
  for (const k of kids) {
    if (k.nodeType !== 1) continue;
    const rc = k.getBoundingClientRect(); if (rc.width === 0 && rc.height === 0) continue;
    l = Math.min(l, rc.left); t = Math.min(t, rc.top); r = Math.max(r, rc.right); b = Math.max(b, rc.bottom);
  }
  if (!isFinite(l)) { const rc = rootEl.getBoundingClientRect(); l = rc.left; t = rc.top; r = rc.right; b = rc.bottom; }
  const cs0 = getComputedStyle(rootEl);
  const containerStyle = (cs0.display === "flex" || cs0.display === "grid") ? styleFor(rootEl, "div") : "";
  let html = kids.map(k => ser(k, 0)).join("");
  if (containerStyle) html = '<div style="' + containerStyle.replace(/(^|;)(width|height|min-width|min-height):[^;]*/g, "$1") + '">' + html + "</div>";
  iframe.remove();
  return { html, width: Math.ceil(r - l), height: Math.ceil(b - t), clip: { x: Math.max(0, l), y: Math.max(0, t), width: Math.max(1, Math.ceil(r - l)), height: Math.max(1, Math.ceil(b - t)) } };
})()`;


export function serializerFor(rootSelector, colorByHex) {
  return SERIALIZE.replace("__COLORS__", JSON.stringify(colorByHex)).replace("__ROOT_SELECTOR__", JSON.stringify(rootSelector));
}

export async function inlineLocalImages(html, allowedHosts = ["127.0.0.1", "localhost"]) {
  const srcs = [...new Set([...html.matchAll(/data-src="([^"]+)"/g)].map((m) => m[1]))];
  for (const src of srcs) {
    try {
      const u = new URL(src);
      if (!allowedHosts.includes(u.hostname)) continue;
      const r = await fetch(src);
      const buf = Buffer.from(await r.arrayBuffer());
      let mime = r.headers.get("content-type") || "";
      const head = buf.subarray(0, 12);
      if (head.subarray(0, 4).toString("ascii") === "RIFF" && head.subarray(8, 12).toString("ascii") === "WEBP") mime = "image/webp";
      else if (head[0] === 0x89 && head.subarray(1, 4).toString("ascii") === "PNG") mime = "image/png";
      else if (head[0] === 0xff && head[1] === 0xd8) mime = "image/jpeg";
      else if (head.subarray(0, 3).toString("ascii") === "GIF") mime = "image/gif";
      else if (buf.subarray(0, 300).toString("utf8").includes("<svg")) mime = "image/svg+xml";
      else if (!mime || mime === "application/octet-stream") mime = /\.avif(\?|$)/.test(src) ? "image/avif" : "image/png";
      if (buf.length > 1_500_000) continue;
      html = html.split(`src="${src}"`).join(`src="data:${mime};base64,${buf.toString("base64")}"`);
    } catch {}
  }
  return html.replace(/ data-src="[^"]*"/g, "");
}
