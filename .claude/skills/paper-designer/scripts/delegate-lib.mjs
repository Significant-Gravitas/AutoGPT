// Small HTML component library for the delegation screens. Inline styles only,
// values from the design tokens (see DESIGN.md), Hugeicons via core-free-icons.
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import * as HI from "@hugeicons/core-free-icons";

const here = dirname(fileURLToPath(import.meta.url));
export const frontendDir = join(here, "..", "..", "..");
export const kitDir = join(here, "..", "kit");

export const C = {
  black: "var(--color-black)", z50: "var(--color-zinc-50)", z100: "var(--color-zinc-100)", z200: "var(--color-zinc-200)", z300: "var(--color-zinc-300)", z400: "var(--color-zinc-400)", z500: "var(--color-zinc-500)", z600: "var(--color-zinc-600)", z700: "var(--color-zinc-700)", z800: "var(--color-zinc-800)", z900: "var(--color-zinc-900)",
  white: "#FFFFFF", inset: "#F9F9F9", accent: "var(--color-accent)", accentSoft: "#F5F3FF", violet100: "#EDE9FE", violet700: "#6D28D9",
  successBg: "#ECFDF5", successText: "#047857", successDot: "#10B981", warnBg: "#FFFBEB", warnText: "#92400E", warnDot: "#F59E0B", errBg: "var(--color-red-50)", errText: "var(--color-red-700)", errDot: "var(--color-red-500)", infoBg: "var(--color-zinc-50)", infoText: "var(--color-zinc-600)",
};
export const F = {
  h4: "font-family:Poppins;font-size:22px;font-weight:500;line-height:24px;color:var(--color-black)",
  h5: "font-family:Poppins;font-size:16px;font-weight:500;line-height:24px;color:var(--color-black)",
  h3: "font-family:Poppins;font-size:28px;font-weight:500;line-height:40px;letter-spacing:-0.21px;color:var(--color-black)",
  lead: "font-family:Geist;font-size:20px;font-weight:400;line-height:28px;color:var(--color-black)",
  large: "font-family:Geist;font-size:16px;font-weight:400;line-height:26px;color:var(--color-black)",
  largeM: "font-family:Geist;font-size:16px;font-weight:500;line-height:26px;color:var(--color-black)",
  body: "font-family:Geist;font-size:14px;font-weight:400;line-height:22px;color:var(--color-black)",
  bodyM: "font-family:Geist;font-size:14px;font-weight:500;line-height:22px;color:var(--color-black)",
  small: "font-family:Geist;font-size:12px;font-weight:400;line-height:18px;color:var(--color-black)",
  smallM: "font-family:Geist;font-size:12px;font-weight:500;line-height:18px;color:var(--color-black)",
  eyebrow: "font-family:Geist;font-size:12px;font-weight:500;line-height:16px;letter-spacing:0.06em;text-transform:uppercase;color:var(--color-zinc-500)",
  mono: "font-family:Geist Mono;font-size:12px;font-weight:400;line-height:18px;color:var(--color-black)",
};
export const esc = (t) => String(t).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
export const div = (style, inner = "", attrs = "") => `<div ${attrs} style="${style}">${inner}</div>`;
export const span = (style, inner = "") => `<span style="${style}">${inner}</span>`;
export const row = (gap, inner, extra = "") => div(`display:flex;flex-direction:row;align-items:center;gap:${gap}px;${extra}`, inner);
export const col = (gap, inner, extra = "") => div(`display:flex;flex-direction:column;gap:${gap}px;${extra}`, inner);
export const text = (font, t, extra = "") => div(`${font};${extra}`, esc(t));
export const muted = (font, t, extra = "") => text(font.replace(/color:[^;]+/, "color:" + C.z500), t, extra);
export const sec = (font, t, extra = "") => text(font.replace(/color:[^;]+/, "color:" + C.z600), t, extra);

export function icon(name, size = 16, color = "currentColor", stroke = 2) {
  const data = HI[name]; if (!data) throw new Error("icon " + name);
  const inner = data.map(([tag, attrs]) => {
    const a = Object.entries(attrs).filter(([k]) => k !== "key").map(([k, v]) => {
      const kk = k.replace(/[A-Z]/g, (m) => "-" + m.toLowerCase());
      if (kk === "stroke-width") v = String(stroke);
      return `${kk}="${v}"`;
    }).join(" ");
    return `<${tag} ${a}></${tag}>`;
  }).join("");
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="${size}" height="${size}" fill="none" style="color:${color};flex-shrink:0;display:block">${inner}</svg>`;
}

const MIME = { png: "image/png", webp: "image/webp", svg: "image/svg+xml", jpg: "image/jpeg" };
export function dataUrl(rel) { const p = join(frontendDir, rel); const ext = rel.split(".").pop(); return `data:${MIME[ext]};base64,${readFileSync(p).toString("base64")}`; }
const avatarCache = {};
export function expertAvatar(key, size = 32) {
  const k = key.toLowerCase();
  if (!avatarCache[k]) avatarCache[k] = dataUrl(`public/autogpt-characters/v2.1/expert-${k}/neutral/128.png`);
  return `<img src="${avatarCache[k]}" style="width:${size}px;height:${size}px;border-radius:9999px;flex-shrink:0;object-fit:cover;background-color:${C.z100}" />`;
}
export function ottoAvatar(size = 32) {
  return div(`width:${size}px;height:${size}px;border-radius:9999px;flex-shrink:0;background-image:linear-gradient(135deg, #C4B5FD 0%, #7C3AED 55%, #A78BFA 100%);box-shadow:0 0 ${Math.round(size / 2)}px rgba(124,58,237,0.35)`);
}
export function avatarFor(who, size = 32) { return who === "Otto" ? ottoAvatar(size) : expertAvatar(who, size); }

export function badge(variant, label, size = "medium") {
  const v = { success: [C.successBg, C.successText, "rgba(5,150,105,0.2)"], error: [C.errBg, C.errText, "rgba(220,38,38,0.1)"], warning: [C.warnBg, C.warnText, "rgba(245,158,11,0.2)"], info: [C.infoBg, C.infoText, "rgba(113,113,122,0.1)"], accent: [C.violet100, C.violet700, "rgba(124,58,237,0.15)"] }[variant];
  const sz = size === "small" ? "padding:2px 6px;font-size:11px;line-height:16px" : "padding:2px 8px;font-size:12px;line-height:20px";
  return div(`display:inline-flex;align-items:center;gap:6px;border-radius:6px;font-family:Geist;font-weight:500;${sz};background-color:${v[0]};color:${v[1]};box-shadow:inset 0 0 0 1px ${v[2]};white-space:nowrap;flex-shrink:0`, esc(label));
}
export function dot(color, size = 8) { return div(`width:${size}px;height:${size}px;border-radius:9999px;background-color:${color};flex-shrink:0`); }

export function button(variant, label, { size = "large", leadingIcon, rightIcon, extra = "" } = {}) {
  const base = "display:inline-flex;align-items:center;justify-content:center;white-space:nowrap;font-family:Geist;font-weight:500;flex-shrink:0;box-sizing:border-box";
  const sizes = { large: "height:46px;padding:10px 16px;gap:8px;font-size:14px;line-height:20px;min-width:123px", small: "height:36px;padding:8px 12px;gap:6px;font-size:14px;line-height:20px;min-width:88px", xs: "height:28px;padding:0 10px;gap:6px;font-size:12px;line-height:16px;border-radius:6px" };
  const variants = {
    primary: `background-color:${C.z800};border:1px solid ${C.z800};color:#FEFEFE;border-radius:9999px`,
    secondary: `background-color:#FEFEFE;border:1px solid ${C.z200};color:${C.z800};border-radius:9999px;box-shadow:0 1px 2px rgba(0,0,0,0.05)`,
    destructive: `background-color:var(--color-red-500);border:1px solid var(--color-red-500);color:#FEFEFE;border-radius:9999px`,
    outline: `background-color:transparent;border:1px solid #A6A6A6;color:${C.black};border-radius:9999px`,
    ghost: `background-color:transparent;border:1px solid transparent;color:${C.black};border-radius:9999px`,
    link: `background-color:transparent;border:none;color:${C.z600};text-decoration-line:underline;padding:0;min-width:0;height:auto`,
  };
  const ic = leadingIcon ? icon(leadingIcon, size === "large" ? 18 : size === "small" ? 16 : 14) : "";
  const ri = rightIcon ? icon(rightIcon, size === "large" ? 18 : 16) : "";
  return div(`${base};${sizes[size]};${variants[variant]};${extra}`, ic + esc(label) + ri);
}
export function iconButton(name, { size = 32, extra = "" } = {}) {
  return div(`display:flex;align-items:center;justify-content:center;width:${size}px;height:${size}px;border-radius:8px;background-color:#FEFEFE;border:1px solid ${C.z200};box-shadow:0 1px 2px rgba(0,0,0,0.05);color:${C.z600};flex-shrink:0;${extra}`, icon(name, 16));
}
export function card(inner, extra = "") { return div(`display:flex;flex-direction:column;background-color:${C.white};border:1px solid ${C.z200};border-radius:16px;${extra}`, inner); }
export function divider(extra = "") { return div(`height:1px;background-color:${C.z200};width:100%;flex-shrink:0;${extra}`); }
export function chip(label, iconName) { return div(`display:inline-flex;align-items:center;gap:6px;height:28px;padding:0 10px;border-radius:9999px;border:1px solid ${C.z200};background-color:#FEFEFE;${F.small};color:${C.z700};white-space:nowrap`, (iconName ? icon(iconName, 14, C.z500) : "") + esc(label)); }

// ---------- app chrome ----------
export function sidebar1440(active = "copilot") {
  const items = [["home", "Home01Icon"], ["copilot", "AiChat02Icon"], ["library", "Store01Icon"], ["marketplace", "DashboardSquare01Icon"], ["team", "UserAdd01Icon"], ["files", "Folder01Icon"]];
  const rail = items.map(([k, ic]) => div(`display:flex;align-items:center;justify-content:center;width:32px;height:32px;border-radius:8px;background-color:${k === active ? C.z200 : "transparent"};color:${k === active ? C.z800 : C.z600}`, icon(ic, 18))).join("");
  const logo = dataUrl("public/autogpt-logo-light-bg.png");
  return div(`position:absolute;left:0px;top:0px;width:48px;height:900px;display:flex;flex-direction:column;align-items:center;justify-content:space-between;padding:12px 0;border-right:1px solid ${C.z200};background-color:${C.inset};box-sizing:border-box`,
    col(4, `<img src="${logo}" style="width:28px;height:28px;object-fit:contain;margin-bottom:8px" />` + div(`display:flex;align-items:center;justify-content:center;width:32px;height:32px;color:${C.z600}`, icon("Search01Icon", 18)) + rail, "align-items:center") +
    div(`width:28px;height:28px;border-radius:9999px;background-image:linear-gradient(135deg,#F0ABFC,#A21CAF)`));
}
export function page1440(inner, { bg = C.inset, active = "copilot", height = 900 } = {}) {
  return div(`position:relative;width:1440px;height:${height}px;background-color:${bg};overflow:hidden;font-family:Geist`, sidebar1440(active).replace("height:900px", `height:${height}px`) + div(`position:absolute;left:48px;top:0px;width:1392px;height:${height}px;display:flex;flex-direction:column`, inner));
}
export function fab() { return div(`position:absolute;left:1328px;top:836px;width:44px;height:44px;border-radius:9999px;background-color:${C.z800};display:flex;align-items:center;justify-content:center;color:#FEFEFE;box-shadow:0 8px 24px rgba(0,0,0,0.18)`, icon("Task01Icon", 20)); }
export function pageHeader(title, subtitle, actions = "") {
  return div(`display:flex;align-items:flex-start;justify-content:space-between;width:1104px;padding-top:32px`, col(2, text(F.h4, title) + sec(F.body, subtitle)) + row(8, actions));
}
export function contentColumn(inner, { width = 1104, top = 0 } = {}) { return div(`display:flex;flex-direction:column;width:${width}px;margin-left:${Math.round((1392 - width) / 2)}px;margin-top:${top}px;gap:24px`, inner); }

// mobile chrome (390)
export function page390(inner, { height = 844 } = {}) {
  const trigger = div(`position:absolute;left:24px;top:24px;width:36px;height:36px;border-radius:9999px;border:1px solid #DADADC;background-color:#FEFEFE;display:flex;align-items:center;justify-content:center;color:${C.z700}`, icon("SidebarLeftIcon", 18));
  return div(`position:relative;width:390px;height:${height}px;background-color:${C.inset};overflow:hidden;font-family:Geist`, trigger + div(`position:absolute;left:0px;top:0px;width:390px;height:${height}px;display:flex;flex-direction:column`, inner));
}

// ---------- chat pieces ----------
export function userMessage(t, width = 806) {
  return div(`display:flex;justify-content:flex-end;width:${width}px`, div(`max-width:${Math.round(width * 0.95)}px;padding:12px 16px;border-radius:8px;background-color:#F5F5F5;${F.body};color:#0A0A0A`, esc(t)));
}
// Assistant turns render as plain text in the real UI: no avatar, no name.
export function assistantText(paragraphs, width = 806, extra = "") {
  const body = paragraphs.map((p) => (typeof p === "string" && !p.trim().startsWith("<") ? div(`${F.body};color:#0A0A0A`, esc(p)) : p)).join("");
  return div(`display:flex;flex-direction:column;gap:12px;width:${width}px;${extra}`, body);
}
export function inputSmall(placeholder, width) {
  return div(`display:flex;align-items:center;height:36px;padding:0 16px;border-radius:12px;border:1px solid ${C.z200};background-color:#FEFEFE;${F.body};color:${C.z400};box-sizing:border-box;white-space:nowrap;overflow:hidden;min-width:0;${width ? "width:" + width + "px" : "flex:1"}`, esc(placeholder));
}
// Docked summary above the composer, same construction as TaskProgressBar.
export function dockBar({ icon: ic = "spinner", title, count, rows = [], expanded = false, width = 806, open = false }) {
  const w = Math.round(width * 0.95);
  const statusIcon = (k) => k === "spinner" ? icon("Loading03Icon", 14, "#A855F7") : k === "done" ? icon("Tick02Icon", 14, "#10B981") : k === "warn" ? icon("MessageQuestionIcon", 14, "#F59E0B") : k === "error" ? icon("Alert02Icon", 14, "#EF4444") : icon("CircleIcon", 14, C.z400);
  const header = div(`display:flex;align-items:center;gap:8px;padding:12px 12px`, row(8, statusIcon(ic) + div(`${F.bodyM};color:${C.z800};white-space:nowrap`, esc(title)), "flex:1;min-width:0") + (count ? div(`${F.body};color:${C.z900};font-variant-numeric:tabular-nums`, esc(count)) : "") + (open ? row(4, div(`${F.small};color:${C.z600}`, "Open work") + icon("ArrowRight01Icon", 14, C.z500)) : div(`color:${C.z400};transform:rotate(${expanded ? 180 : 0}deg)`, icon("ArrowDown01Icon", 14))));
  const list = expanded ? div(`display:flex;flex-direction:column;gap:8px;padding:0 12px 12px 12px`, rows.map(([k, t, dim]) => row(8, statusIcon(k) + div(`${F.body};color:${dim ? C.z400 : k === "spinner" ? C.z900 : C.z600};${k === "spinner" ? "font-weight:500" : ""}`, esc(t)))).join("") + row(6, icon("Task01Icon", 14, C.z500) + div(`${F.small};color:${C.z600};text-decoration-line:underline`, "Show work in the panel"), "padding-top:4px")) : "";
  return div(`display:flex;flex-direction:column;width:${w}px;margin-left:${Math.round((width - w) / 2)}px;margin-bottom:-14px;padding-bottom:14px;border-radius:24px 24px 0 0;border:1px solid ${C.z200};border-bottom:none;background-color:#F5F5F5;box-shadow:inset 0 1px 0 0 rgba(255,255,255,0.9);box-sizing:border-box;overflow:hidden`, header + list);
}
// ToolChain rows as ChainRowView renders them.
export function toolChain(rows, { width = 806, collapsed = null } = {}) {
  if (collapsed) return div(`display:flex;align-items:center;gap:6px;width:fit-content;padding:4px 8px;margin-left:-8px;border-radius:8px;${F.body};color:${C.z600}`, esc(collapsed) + icon("ArrowDown01Icon", 10, C.z300));
  const r = rows.map((x, i) => {
    const last = i === rows.length - 1;
    const circle = div(`position:relative;display:flex;align-items:center;justify-content:center;width:28px;height:28px;border-radius:9999px;background-color:${x.state === "error" ? C.errBg : C.z100};flex-shrink:0`, icon(x.icon || "ZapIcon", 16, x.state === "error" ? "#EF4444" : x.iconColor || C.z600));
    const gutter = div(`display:flex;flex-direction:column;align-items:center;width:28px;flex-shrink:0`, circle + (last ? "" : div(`width:1px;flex:1;background-color:${C.z200}`)));
    if (x.node) {
      // The wire runs straight into the top of the card, directly above the
      // expert's photo, and continues out of its bottom to the next row.
      const seg = (h) => div(`display:flex;justify-content:center;width:28px;height:${h}px;flex-shrink:0`, div(`width:1px;height:${h}px;background-color:${C.z200}`));
      // The card keeps its own 16px padding and shifts left instead, so the
      // photo inside it lands on the wire (x = 14) like the row icons do.
      return div(`display:flex;flex-direction:column;width:${width}px`, seg(12) + div("margin-left:-16px", x.node) + (last ? "" : seg(12)));
    }
    const tag = x.tag ? div(`${F.small};font-weight:500;color:${x.tagColor || C.warnText};white-space:nowrap`, esc(x.tag)) : "";
    const label = div(`display:flex;align-items:center;gap:6px;height:28px`, div(`${F.body};color:${x.state === "error" ? "#EF4444" : x.state === "running" ? C.z900 : C.z600};white-space:nowrap`, esc(x.text)) + tag + (x.content ? icon("ArrowDown01Icon", 10, C.z300) : ""));
    const body = div(`display:flex;flex-direction:column;min-width:0;flex:1;padding-bottom:${last ? 0 : 12}px`, label + (x.content ? div("padding:6px 1px 1px 1px", x.content) : ""));
    return div(`display:flex;gap:10px;align-items:stretch;width:${width}px`, gutter + body);
  }).join("");
  return div(`display:flex;flex-direction:column;width:${width}px;margin:8px 0`, r);
}
export function skeletonBlock(w, h = 12, extra = "") { return div(`width:${w};height:${h}px;border-radius:6px;background-color:${C.z100};flex-shrink:0;${extra}`); }
export function agentMessage(who, role, paragraphs, width = 806, extra = "", indent = 38) {
  const body = paragraphs.map((p) => (typeof p === "string" && !p.trim().startsWith("<") ? text(F.body, p) : p)).join("");
  return div(`display:flex;flex-direction:column;gap:10px;width:${width}px;${extra}`, row(10, avatarFor(who, 28) + text(F.bodyM, who) + (role ? sec(F.small, "· " + role) : "")) + col(10, body, `padding-left:${indent}px`));
}
export function composer(width = 806, placeholder = "Reply to Otto…", who = "Otto") {
  const pill = div(`display:inline-flex;align-items:center;gap:6px;height:36px;padding:0 12px;border-radius:9999px;border:1px solid ${C.z200};background-color:#FEFEFE;${F.body}`, avatarFor(who, 18) + esc(who) + icon("ArrowDown01Icon", 14, C.z500));
  return div(`display:flex;flex-direction:column;justify-content:space-between;width:${width}px;height:132px;padding:20px 16px 16px 16px;border-radius:24px;border:1px solid ${C.z200};background-color:#FEFEFE;box-sizing:border-box`,
    muted(F.large, placeholder) + row(8, iconButton("PlusSignIcon", { size: 36, extra: "border-radius:9999px" }) + pill + div("flex:1") + iconButton("Mic01Icon", { size: 36, extra: "border-radius:9999px" }) + div(`display:flex;align-items:center;justify-content:center;width:36px;height:36px;border-radius:9999px;background-color:${C.z100};color:${C.z400}`, icon("ArrowUp01Icon", 16))));
}
