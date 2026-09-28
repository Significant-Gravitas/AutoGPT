// Runs on the Mac (where Paper Desktop's MCP server listens on 127.0.0.1:29979).
// Imports the serialised kit and assets into the "AutoGPT Design System" file.
//
//   node paper-import.mjs kit [--only <substring>] [--limit N] [--force]
//   node paper-import.mjs assets
//   node paper-import.mjs shot <nodeId> <out.png>
//   node paper-import.mjs info [pageId]
import { readFileSync, writeFileSync, readdirSync, statSync, existsSync } from "node:fs";
import { join, dirname, extname, basename } from "node:path";
import { fileURLToPath } from "node:url";

const M = "http://127.0.0.1:29979/mcp";
const FID = process.env.PAPER_FILE_ID || "01M3FAR8BZ91D6DWJ2KAEFKN50";
const here = dirname(fileURLToPath(import.meta.url));
const paperDir = join(here, "..");
const frontendDir = join(paperDir, "..", "..");
const args = process.argv.slice(2);
const cmd = args[0];
const opt = (k) => (args.includes(k) ? args[args.indexOf(k) + 1] : null);
const flag = (k) => args.includes(k);

let sid = null;
async function rpc(method, params, id = 1) {
  const r = await fetch(M, { method: "POST", headers: { "Content-Type": "application/json", Accept: "application/json, text/event-stream", ...(sid ? { "Mcp-Session-Id": sid } : {}) }, body: JSON.stringify({ jsonrpc: "2.0", id, method, params }) });
  sid = r.headers.get("mcp-session-id") || sid;
  const t = await r.text();
  const line = t.split("\n").map((l) => l.replace(/^data: /, "")).find((l) => l.startsWith("{"));
  return line ? JSON.parse(line) : null;
}
async function call(name, args_) {
  const j = await rpc("tools/call", { name, arguments: { fileId: FID, ...args_ } }, Date.now() % 100000);
  if (!j) throw new Error(`${name}: empty response`);
  if (j.error) throw new Error(`${name}: ${JSON.stringify(j.error)}`);
  const texts = (j.result?.content || []).filter((c) => c.type === "text").map((c) => c.text);
  const images = (j.result?.content || []).filter((c) => c.type === "image");
  if (j.result?.isError) throw new Error(`${name}: ${texts.join("\n").slice(0, 400)}`);
  return { texts, images, raw: j.result };
}
function parseJsonTexts(texts) {
  const out = [];
  for (const t of texts) { try { out.push(JSON.parse(t)); } catch { out.push(t); } }
  return out;
}
async function init() {
  await rpc("initialize", { protocolVersion: "2025-03-26", capabilities: {}, clientInfo: { name: "paper-import", version: "1" } });
  await fetch(M, { method: "POST", headers: { "Content-Type": "application/json", Accept: "application/json, text/event-stream", "Mcp-Session-Id": sid }, body: JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized" }) });
}
async function pages() {
  const info = parseJsonTexts((await call("get_basic_info", {})).texts).find((x) => x && x.pages);
  return Object.fromEntries(info.pages.map((p) => [p.name, p.id]));
}
async function ensurePage(name) {
  const p = await pages();
  if (p[name]) return p[name];
  const res = parseJsonTexts((await call("create_page", { name })).texts);
  const id = res.map((r) => r?.pageId || r?.id).find(Boolean) || String(res[0]).match(/p-\d+-\d+/)?.[0];
  return id;
}
// get_basic_info and get_children list at most 100 artboards; the tree summary
// at depth 1 lists every top-level frame on the page.
async function artboardsOn(pageId) {
  try {
    const res = parseJsonTexts((await call("get_tree_summary", { nodeId: `root_node_${pageId}`, depth: 1 })).texts).find((x) => x && typeof x.summary === "string");
    if (res) {
      const list = [];
      for (const m of res.summary.matchAll(/^  \S+ "((?:[^"\\]|\\.)*)" \(([A-Za-z0-9]+-\d+)\) (\d+|\?)×(\d+|\?)/gm)) list.push({ id: m[2], name: m[1], width: Number(m[3]) || 0, height: Number(m[4]) || 0 });
      if (list.length) return list;
    }
  } catch {}
  const info = parseJsonTexts((await call("get_basic_info", { pageId })).texts).find((x) => x && x.artboards);
  return info?.artboards || [];
}

const idRank = (id) => [id.length, id];
async function dedupe() {
  const pageId = args[1];
  const abs = await artboardsOn(pageId);
  const byName = {};
  for (const a of abs) (byName[a.name] ||= []).push(a);
  const toDelete = [];
  for (const [name, list] of Object.entries(byName)) {
    if (list.length < 2) continue;
    list.sort((a, b) => (idRank(a.id)[0] - idRank(b.id)[0]) || (a.id < b.id ? -1 : 1));
    toDelete.push(...list.slice(0, -1).map((a) => a.id));
    console.log(`dup ${name}: keep ${list[list.length - 1].id}, delete ${list.slice(0, -1).map((a) => a.id).join(",")}`);
  }
  for (let i = 0; i < toDelete.length; i += 25) await call("delete_nodes", { nodeIds: toDelete.slice(i, i + 25) });
  console.log(`${abs.length} artboards, deleted ${toDelete.length} duplicates`);
}
function artboardIdFrom(res) {
  const j = parseJsonTexts(res.texts).filter((x) => x && typeof x === "object" && !("file" in x));
  const isId = (v) => typeof v === "string" && /^[A-Za-z0-9]+-\d+$/.test(v);
  for (const x of j) {
    if (isId(x.nodeId)) return x.nodeId;
    if (isId(x.id)) return x.id;
    for (const k of ["createdNodes", "nodes"]) if (Array.isArray(x[k]) && x[k].length && isId(x[k][0].id)) return x[k][0].id;
    for (const k of ["nodeIds", "createdNodeIds"]) if (Array.isArray(x[k]) && x[k].length && isId(x[k][0])) return x[k][0];
  }
  return undefined;
}

// Splits an HTML string into its top-level elements (serializer output only has
// <div>, <span>, <img>, <svg> tags with attribute values that never contain '<').
function splitTopLevel(html) {
  const parts = []; let depth = 0; let start = 0; const re = /<(\/?)([a-zA-Z][a-zA-Z0-9]*)[^>]*?(\/?)>/g; let m;
  while ((m = re.exec(html))) {
    const closing = m[1] === "/"; const selfClosing = m[3] === "/" || ["img", "br", "hr", "path", "circle", "rect", "line", "polyline", "polygon", "ellipse", "use", "stop"].includes(m[2]) && m[3] === "/";
    if (closing) depth--; else if (!selfClosing && !(m[0].endsWith("/>"))) depth++;
    if (depth === 0) { parts.push(html.slice(start, re.lastIndex)); start = re.lastIndex; }
  }
  if (start < html.length) parts.push(html.slice(start));
  return parts.filter((p) => p.trim());
}

async function importKit() {
  const manifestPath = process.env.PAPER_MANIFEST ? join(paperDir, "kit", process.env.PAPER_MANIFEST) : join(paperDir, "kit", "manifest.json");
  const manifestRaw = JSON.parse(readFileSync(manifestPath, "utf8"));
  let stories = Array.isArray(manifestRaw) ? manifestRaw : manifestRaw.stories;
  const only = opt("--only"); const limit = Number(opt("--limit") || Infinity);
  if (only) stories = stories.filter((s) => s.id.includes(only) || s.artboardName.includes(only));
  stories = stories.slice(0, limit);
  const pageIds = {};
  const existing = {};
  for (const s of stories) {
    if (!pageIds[s.page]) { pageIds[s.page] = await ensurePage(s.page); existing[s.page] = new Set((await artboardsOn(pageIds[s.page])).map((a) => a.name)); }
  }
  const results = [];
  const touched = [];
  for (const s of stories) {
    const pageId = pageIds[s.page];
    if (existing[s.page].has(s.artboardName) && !flag("--force") && !flag("--replace")) { results.push(`skip ${s.artboardName} (exists)`); continue; }
    if (existing[s.page].has(s.artboardName) && flag("--replace")) {
      const olds = (await artboardsOn(pageId)).filter((a) => a.name === s.artboardName).map((a) => a.id);
      if (olds.length) await call("delete_nodes", { nodeIds: olds });
    }
    try {
      const html = readFileSync(join(paperDir, "kit", s.file), "utf8");
      const exact = Boolean(s.exact);
      const w = exact ? s.width : Math.max(200, Math.min(1440, s.width + 48)); const h = exact ? s.height : Math.max(80, s.height + 48);
      const ab = await call("create_artboard", { pageId, name: s.artboardName, styles: { width: `${w}px`, height: `${h}px`, backgroundColor: exact ? "#F6F7F8" : "#FFFFFF", display: "flex", flexDirection: "column", alignItems: "flex-start", padding: exact ? "0px" : "24px", gap: "0px", overflow: "hidden" } });
      const abId = artboardIdFrom(ab);
      if (!abId) throw new Error("no artboard id in " + ab.texts.join(" ").slice(0, 200));
      if (html.length > 700_000) {
        // Very large pages: split the top-level children so no single call is huge.
        const parts = splitTopLevel(html);
        for (const part of parts) await call("write_html", { targetNodeId: abId, mode: "insert-children", html: part });
      } else {
        await call("write_html", { targetNodeId: abId, mode: "insert-children", html });
      }
      if (!exact) await call("update_styles", { updates: [{ nodeIds: [abId], styles: { height: "fit-content" } }] });
      touched.push(abId);
      results.push(`ok   ${s.artboardName} -> ${abId} (${s.bytes}b)`);
    } catch (e) {
      results.push(`FAIL ${s.artboardName}: ${String(e).slice(0, 300)}`);
    }
    console.log(results[results.length - 1]);
  }
  if (touched.length) await call("finish_working_on_nodes", { nodeIds: touched });
  console.log(`\n${results.filter((r) => r.startsWith("ok")).length} ok, ${results.filter((r) => r.startsWith("FAIL")).length} failed, ${results.filter((r) => r.startsWith("skip")).length} skipped`);
}

const MIME = { ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".svg": "image/svg+xml", ".webp": "image/webp", ".gif": "image/gif", ".ico": "image/x-icon" };
function listFiles(dir, out = []) { for (const f of readdirSync(dir)) { const p = join(dir, f); if (statSync(p).isDirectory()) listFiles(p, out); else out.push(p); } return out; }
function dataUrl(p) { const ext = extname(p).toLowerCase(); if (!MIME[ext]) return null; return `data:${MIME[ext]};base64,${readFileSync(p).toString("base64")}`; }

async function importAssets() {
  const spec = JSON.parse(readFileSync(join(paperDir, "assets.json"), "utf8"));
  const pageId = await ensurePage(spec.page);
  const have = new Set((await artboardsOn(pageId)).map((a) => a.name));
  const touched = [];
  for (const group of spec.groups) {
    if (have.has(group.name) && !flag("--force")) { console.log(`skip ${group.name}`); continue; }
    let files = [];
    for (const src of group.sources) {
      const p = join(frontendDir, src);
      if (!existsSync(p)) continue;
      if (statSync(p).isDirectory()) files.push(...listFiles(p)); else files.push(p);
    }
    files = files.filter((f) => MIME[extname(f).toLowerCase()] && (!group.match || new RegExp(group.match).test(f))).sort();
    if (group.maxFiles) files = files.slice(0, group.maxFiles);
    const cols = group.columns || 8; const cell = group.cell || 96;
    const w = cols * (cell + 24) + 48; const rows = Math.ceil(files.length / cols);
    const h = 80 + rows * (cell + 44) + 48;
    const ab = await call("create_artboard", { pageId, name: group.name, styles: { width: `${w}px`, height: `${h}px`, backgroundColor: group.background || "#FFFFFF", display: "flex", flexDirection: "column", padding: "24px", gap: "16px" } });
    const abId = artboardIdFrom(ab);
    await call("write_html", { targetNodeId: abId, mode: "insert-children", html: `<div style="font-family:Poppins;font-size:22px;font-weight:500;line-height:24px;color:var(--color-black)">${group.name.replace(/^Assets\//, "")}</div><div style="font-family:Geist;font-size:12px;line-height:18px;color:var(--color-zinc-500)">${group.note || ""} ${files.length} files from ${group.sources.join(", ")}</div>` });
    // An empty container would be parsed as a Rectangle (no children allowed),
    // so the grid is written together with its first batch of tiles.
    let gridId = null;
    let batch = [];
    const gridOpen = `<div style="display:flex;flex-wrap:wrap;gap:24px;width:${w - 48}px">`;
    const flush = async () => {
      if (!batch.length) return;
      if (!gridId) {
        const res = await call("write_html", { targetNodeId: abId, mode: "insert-children", html: gridOpen + batch.join("") + "</div>" });
        gridId = artboardIdFrom(res);
        if (!gridId) throw new Error("grid id missing: " + res.texts.join(" ").slice(0, 200));
      } else {
        await call("write_html", { targetNodeId: gridId, mode: "insert-children", html: batch.join("") });
      }
      batch = [];
    };
    let total = 0;
    for (const f of files) {
      const d = dataUrl(f); if (!d) continue;
      const label = basename(f).replace(/\.[^.]+$/, "");
      const fit = group.fit || "contain";
      batch.push(`<div style="display:flex;flex-direction:column;align-items:center;gap:6px;width:${cell}px"><div style="display:flex;align-items:center;justify-content:center;width:${cell}px;height:${cell}px;border-radius:12px;background-color:${group.tile || "#F9F9FA"};overflow:hidden"><img src="${d}" style="width:${group.imgSize || cell - 16}px;height:${group.imgSize || cell - 16}px;object-fit:${fit}" /></div><div style="font-family:Geist;font-size:11px;line-height:14px;color:var(--color-zinc-600);text-align:center;width:${cell}px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${label}</div></div>`);
      total += d.length;
      if (batch.length >= 12 || total > 1_200_000) { await flush(); total = 0; }
    }
    await flush();
    await call("update_styles", { updates: [{ nodeIds: [abId], styles: { height: "fit-content" } }] });
    touched.push(abId);
    console.log(`ok   ${group.name}: ${files.length} files -> ${abId}`);
  }
  if (touched.length) await call("finish_working_on_nodes", { nodeIds: touched });
}

const esc = (t) => t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
function inline(md) {
  return esc(md).replace(/`([^`]+)`/g, '<span style="font-family:Geist Mono;font-size:12px;background-color:var(--color-zinc-100);border-radius:4px;padding:1px 4px">$1</span>').replace(/\*\*([^*]+)\*\*/g, '<span style="font-weight:600">$1</span>');
}
const P = 'font-family:Geist;font-size:14px;line-height:22px;color:var(--color-black)';
function mdToHtml(md) {
  const lines = md.split("\n"); const out = []; let i = 0;
  while (i < lines.length) {
    const l = lines[i];
    if (/^### /.test(l)) { out.push(`<div style="font-family:Poppins;font-size:16px;font-weight:500;line-height:24px;color:var(--color-black);margin-top:8px">${inline(l.slice(4))}</div>`); i++; continue; }
    if (/^\|/.test(l)) {
      const rows = []; while (i < lines.length && /^\|/.test(lines[i])) { rows.push(lines[i]); i++; }
      const cells = rows.filter((r) => !/^\|\s*-/.test(r)).map((r) => r.replace(/^\||\|$/g, "").split("|").map((c) => c.trim()));
      const html = cells.map((r, ri) => `<div style="display:flex;gap:16px;padding:6px 0;border-bottom-width:1px;border-bottom-style:solid;border-bottom-color:var(--color-zinc-200)">${r.map((c) => `<div style="${P};flex:1;min-width:0;${ri === 0 ? "font-weight:500;color:var(--color-zinc-600);font-size:12px;text-transform:uppercase;letter-spacing:0.06em" : ""}">${inline(c)}</div>`).join("")}</div>`).join("");
      out.push(`<div style="display:flex;flex-direction:column;width:100%">${html}</div>`); continue;
    }
    if (/^(- |\d+\. )/.test(l)) {
      const items = []; while (i < lines.length && /^(- |\d+\. |\s{2,}\S)/.test(lines[i])) { if (/^(- |\d+\. )/.test(lines[i])) items.push(lines[i].replace(/^(- |\d+\. )/, "")); else items[items.length - 1] += " " + lines[i].trim(); i++; }
      out.push(`<div style="display:flex;flex-direction:column;gap:4px">${items.map((t) => `<div style="display:flex;gap:8px"><div style="${P};width:12px;flex-shrink:0">•</div><div style="${P}">${inline(t)}</div></div>`).join("")}</div>`); continue;
    }
    if (/^```/.test(l)) { const code = []; i++; while (i < lines.length && !/^```/.test(lines[i])) { code.push(lines[i]); i++; } i++; out.push(`<div style="font-family:Geist Mono;font-size:12px;line-height:18px;background-color:var(--color-zinc-50);border-radius:8px;padding:12px;white-space:pre-wrap">${esc(code.join("\n"))}</div>`); continue; }
    if (!l.trim()) { i++; continue; }
    const para = []; while (i < lines.length && lines[i].trim() && !/^(#|\||- |\d+\. |```)/.test(lines[i])) { para.push(lines[i].trim()); i++; }
    out.push(`<div style="${P}">${inline(para.join(" "))}</div>`);
  }
  return out.join("");
}

async function notes() {
  const md = readFileSync(join(paperDir, "DESIGN.md"), "utf8");
  const pageId = await ensurePage("05 Taste");
  const have = new Set((await artboardsOn(pageId)).map((a) => a.name));
  const sections = md.split(/^## /m);
  const intro = sections.shift();
  const all = [["Notes/00 About this file", "AutoGPT Platform — DESIGN.md", intro.replace(/^# .*\n/, "")], ...sections.map((sec) => { const title = sec.split("\n")[0].trim(); return [`Notes/${title.replace(/^(\d+)\. /, "$1 ")}`, title, sec.slice(sec.indexOf("\n") + 1)]; })];
  const touched = [];
  for (const [name, title, body] of all) {
    if (have.has(name) && !flag("--force")) { console.log(`skip ${name}`); continue; }
    const ab = await call("create_artboard", { pageId, name, styles: { width: "760px", height: "400px", backgroundColor: "#FFFFFF", display: "flex", flexDirection: "column", gap: "12px", padding: "32px" } });
    const abId = artboardIdFrom(ab);
    await call("write_html", { targetNodeId: abId, mode: "insert-children", html: `<div style="font-family:Poppins;font-size:22px;font-weight:500;line-height:24px;color:var(--color-black)">${esc(title)}</div>` });
    const chunks = mdToHtml(body).match(/<div[\s\S]*?(?=<div style="font-family:Poppins;font-size:16px|<div style="display:flex;flex-direction:column;width:100%|$)/g) || [mdToHtml(body)];
    const html = mdToHtml(body);
    await call("write_html", { targetNodeId: abId, mode: "insert-children", html: `<div style="display:flex;flex-direction:column;gap:12px;width:696px">${html}</div>` });
    await call("update_styles", { updates: [{ nodeIds: [abId], styles: { height: "fit-content" } }] });
    touched.push(abId); console.log(`ok   ${name} -> ${abId}`);
  }
  if (touched.length) await call("finish_working_on_nodes", { nodeIds: touched });
}

const PATTERNS = [
  ["Pattern/Loading — card skeleton", "01 Atoms", "Kit/Skeleton/card"],
  ["Pattern/Loading — table skeleton", "01 Atoms", "Kit/Skeleton/table"],
  ["Pattern/Loading — dashboard skeleton", "01 Atoms", "Kit/Skeleton/dashboard"],
  ["Pattern/Loading — list skeleton", "01 Atoms", "Kit/Skeleton/list-items"],
  ["Pattern/Error — with retry", "02 Molecules", "Kit/ErrorCard/response-error"],
  ["Pattern/Error — no retry", "02 Molecules", "Kit/ErrorCard/no-retry-button"],
  ["Pattern/Confirm dialog", "02 Molecules", "Kit/Dialog/with-footer/open"],
  ["Pattern/Dialog — basic", "02 Molecules", "Kit/Dialog/basic/open"],
  ["Pattern/Toast — success", "02 Molecules", "Kit/Toast/success-toast/open"],
  ["Pattern/Toast — error", "02 Molecules", "Kit/Toast/error-toast/open"],
  ["Pattern/Alert — warning", "02 Molecules", "Kit/Alert/warning"],
  ["Pattern/Form — input states", "01 Atoms", "Kit/Input/input-types"],
  ["Pattern/Form — select states", "01 Atoms", "Kit/Select/all-variants"],
  ["Pattern/List — infinite list", "02 Molecules", "Kit/InfiniteList/basic"],
  ["Pattern/Table — key value", "02 Molecules", "Kit/Table/key-value-pairs"],
  ["Pattern/Tabs", "02 Molecules", "Kit/TabsLine/default"],
  ["Pattern/Search input", "02 Molecules", "Kit/SearchInput/default"],
  ["Pattern/Status badges", "01 Atoms", "Kit/Badge/all-variants"],
  ["Pattern/Buttons — every variant", "01 Atoms", "Kit/Button/all-variants"],
  ["Pattern/Typography scale", "01 Atoms", "Kit/Text/all-variants"],
  ["Pattern/Plans", "02b Organisms", "Kit/SubscriptionPlans/monthly-trial"],
  ["Pattern/Command palette", "02b Organisms", "Kit/SearchCommandModal/default/open"],
];

async function patterns() {
  const pg = await pages();
  const target = pg["04 Patterns"];
  const have = new Set((await artboardsOn(target)).map((a) => a.name));
  const cache = {};
  const touched = [];
  for (const [name, srcPage, srcName] of PATTERNS) {
    if (have.has(name) && !flag("--force")) { console.log(`skip ${name}`); continue; }
    if (!cache[srcPage]) cache[srcPage] = await artboardsOn(pg[srcPage]);
    const src = cache[srcPage].find((a) => a.name === srcName);
    if (!src) { console.log(`MISSING ${srcName}`); continue; }
    try {
      const dup = parseJsonTexts((await call("duplicate_nodes", { nodes: [{ id: src.id }] })).texts).filter((x) => x && !("file" in x));
      const txt = JSON.stringify(dup);
      const newId = (txt.match(/"(?:newNodeId|newId|duplicateId|nodeId)"\s*:\s*"([A-Za-z0-9]+-\d+)"/g) || []).map((m) => m.match(/"([A-Za-z0-9]+-\d+)"$/)[1]).find((id) => id !== src.id) || (txt.match(/"([A-Za-z0-9]+-\d+)"/g) || []).map((m) => m.slice(1, -1)).find((id) => id !== src.id);
      if (!newId) throw new Error("no duplicate id in " + txt.slice(0, 200));
      await call("move_nodes", { moves: [{ nodeId: newId, parentId: `root_node_${target}` }] });
      await call("rename_nodes", { updates: [{ nodeId: newId, name }] });
      touched.push(newId); console.log(`ok   ${name} <- ${srcName} (${newId})`);
    } catch (e) { console.log(`FAIL ${name}: ${String(e).slice(0, 200)}`); }
  }
  if (touched.length) await call("finish_working_on_nodes", { nodeIds: touched });
}

async function shot() {
  const res = await call("get_screenshot", { nodeId: args[1], scale: 1 });
  const img = res.images[0];
  if (!img) { console.log(res.texts.join("\n").slice(0, 500)); return; }
  writeFileSync(args[2], Buffer.from(img.data, "base64"));
  console.log(`wrote ${args[2]} ${img.data.length} b64 chars`);
  if (flag("--print")) console.log("BASE64:" + img.data);
}

async function del() {
  const ids = args.slice(1);
  const res = await call("delete_nodes", { nodeIds: ids });
  console.log(res.texts.join("\n").slice(0, 500));
}

async function deleteByName() {
  const pageId = args[1]; const name = args[2];
  const abs = (await artboardsOn(pageId)).filter((a) => a.name === name);
  if (!abs.length) { console.log("none"); return; }
  const res = await call("delete_nodes", { nodeIds: abs.map((a) => a.id) });
  console.log(res.texts.join(" ").slice(-200));
}

async function fixHeights() {
  const pageId = args[1];
  const abs = await artboardsOn(pageId);
  const ids = abs.filter((a) => a.name.startsWith("Kit/") || a.name.startsWith("Assets/")).map((a) => a.id);
  for (let i = 0; i < ids.length; i += 25) await call("update_styles", { updates: [{ nodeIds: ids.slice(i, i + 25), styles: { height: "fit-content" } }] });
  console.log(`fit-content applied to ${ids.length} artboards`);
}

async function info() {
  const res = await call("get_basic_info", args[1] ? { pageId: args[1] } : {});
  console.log(res.texts.join("\n").slice(0, 30000));
}


async function comments() {
  const pageId = args[1];
  const status = opt("--status") || "open";
  const res = await call("list_comment_threads", { pageId, status });
  console.log(res.texts.join("\n"));
}
async function thread() { const res = await call("get_comment_thread", { commentThreadId: args[1] }); console.log(res.texts.join("\n")); }
async function resolveThreads() { for (const id of args.slice(1)) { const res = await call("set_comment_thread_status", { commentThreadId: id, status: "resolved" }); console.log(id, res.texts.join(" ").slice(0, 200)); } }
async function nodeChain() {
  for (const id0 of args.slice(1)) {
    let id = id0; const chain = [];
    for (let i = 0; i < 12 && id; i++) {
      const res = parseJsonTexts((await call("get_node_info", { nodeId: id })).texts).find((x) => x && typeof x === "object");
      if (!res) break;
      chain.push({ id, name: res.name, type: res.type, parentId: res.parentId, x: res.x ?? res.layout?.x, y: res.y ?? res.layout?.y, w: res.width ?? res.layout?.width, h: res.height ?? res.layout?.height, text: res.text || res.textContent });
      if (chain.length === 1) console.log(JSON.stringify(res).slice(0, 1500));
      id = res.parentId;
      if (!id || String(id).startsWith("root_node")) break;
    }
    console.log(JSON.stringify(chain, null, 1));
  }
}
async function tree() { const abs = await artboardsOn(args[1]); for (const a of abs) console.log(a.id, a.name, a.width + "x" + a.height); console.log(abs.length, "artboards"); }
async function children() { const res = await call("get_children", { nodeId: args[1] }); console.log(res.texts.join("\n").slice(0, 20000)); }

(async () => {
  await init();
  if (cmd === "kit") await importKit();
  else if (cmd === "assets") await importAssets();
  else if (cmd === "shot") await shot();
  else if (cmd === "info") await info();
  else if (cmd === "delete") await del();
  else if (cmd === "dedupe") await dedupe();
  else if (cmd === "notes") await notes();
  else if (cmd === "patterns") await patterns();
  else if (cmd === "delete-by-name") await deleteByName();
  else if (cmd === "fix-heights") await fixHeights();
  else if (cmd === "comments") await comments();
  else if (cmd === "thread") await thread();
  else if (cmd === "resolve") await resolveThreads();
  else if (cmd === "node") await nodeChain();
  else if (cmd === "tree") await tree();
  else if (cmd === "children") await children();
  else console.log("usage: kit|assets|shot|info");
})().catch((e) => { console.error(e); process.exit(1); });
