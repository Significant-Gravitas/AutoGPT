#!/usr/bin/env node
/**
 * extract-animations.js
 *
 * Fetches a URL and extracts all CSS animations, transitions, and JS animation
 * libraries (GSAP, Framer Motion, AOS, etc.) into reusable files.
 *
 * Usage:
 *   node extract-animations.js https://example.com
 *
 * Output:
 *   ../assets/extracted-keyframes.css
 *   ../assets/extracted-transitions.css
 *   ../assets/extracted-scroll-animations.js
 *   ../assets/animation-inventory.json
 */

const https = require("https");
const http = require("http");
const fs = require("fs");
const path = require("path");
const { URL } = require("url");

const targetUrl = process.argv[2];
if (!targetUrl) {
  console.error("Usage: node extract-animations.js <URL>");
  process.exit(1);
}

const ASSETS_DIR = path.join(__dirname, "..", "assets");

function fetch(url) {
  return new Promise((resolve, reject) => {
    const client = url.startsWith("https") ? https : http;
    client.get(url, { headers: { "User-Agent": "Mozilla/5.0" } }, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        return fetch(res.headers.location).then(resolve).catch(reject);
      }
      let data = "";
      res.on("data", (chunk) => (data += chunk));
      res.on("end", () => resolve(data));
      res.on("error", reject);
    }).on("error", reject);
  });
}

function extractKeyframes(css) {
  const keyframeRegex = /@keyframes\s+[\w-]+\s*\{[^}]*(?:\{[^}]*\}[^}]*)*\}/g;
  return css.match(keyframeRegex) || [];
}

function extractTransitions(css) {
  const transitionRegex = /[^{}]*\{[^}]*transition[^}]*\}/g;
  const matches = css.match(transitionRegex) || [];
  return matches.filter((m) => m.includes("transition"));
}

function extractAnimationProperties(css) {
  const animationRegex = /[^{}]*\{[^}]*animation[^}]*\}/g;
  const matches = css.match(animationRegex) || [];
  return matches.filter((m) => m.includes("animation"));
}

function extractLinkedStylesheets(html) {
  const linkRegex = /<link[^>]+rel=["']stylesheet["'][^>]+href=["']([^"']+)["']/g;
  const urls = [];
  let match;
  while ((match = linkRegex.exec(html)) !== null) {
    urls.push(match[1]);
  }
  return urls;
}

function extractInlineStyles(html) {
  const styleRegex = /<style[^>]*>([\s\S]*?)<\/style>/g;
  const styles = [];
  let match;
  while ((match = styleRegex.exec(html)) !== null) {
    styles.push(match[1]);
  }
  return styles.join("\n");
}

function detectAnimationLibraries(html) {
  const libraries = [];
  if (html.includes("gsap") || html.includes("TweenMax") || html.includes("ScrollTrigger")) {
    libraries.push("GSAP");
  }
  if (html.includes("framer-motion") || html.includes("motion.div")) {
    libraries.push("Framer Motion");
  }
  if (html.includes("data-aos") || html.includes("aos.js")) {
    libraries.push("AOS (Animate On Scroll)");
  }
  if (html.includes("lottie") || html.includes("bodymovin")) {
    libraries.push("Lottie");
  }
  if (html.includes("anime.min.js") || html.includes("animejs")) {
    libraries.push("Anime.js");
  }
  if (html.includes("wow.min.js") || html.includes("WOW(")) {
    libraries.push("WOW.js");
  }
  if (html.includes("scrollmagic") || html.includes("ScrollMagic")) {
    libraries.push("ScrollMagic");
  }
  if (html.includes("locomotive-scroll") || html.includes("data-scroll")) {
    libraries.push("Locomotive Scroll");
  }
  return libraries;
}

function resolveUrl(base, href) {
  try {
    return new URL(href, base).toString();
  } catch {
    return null;
  }
}

async function main() {
  console.log(`Fetching: ${targetUrl}`);
  const html = await fetch(targetUrl);

  // Extract inline CSS
  const inlineCSS = extractInlineStyles(html);

  // Fetch linked stylesheets
  const stylesheetUrls = extractLinkedStylesheets(html);
  console.log(`Found ${stylesheetUrls.length} linked stylesheet(s)`);

  let allCSS = inlineCSS + "\n";
  for (const href of stylesheetUrls) {
    const fullUrl = resolveUrl(targetUrl, href);
    if (fullUrl) {
      try {
        console.log(`  Fetching stylesheet: ${fullUrl}`);
        const css = await fetch(fullUrl);
        allCSS += css + "\n";
      } catch (err) {
        console.warn(`  Failed to fetch: ${fullUrl}`);
      }
    }
  }

  // Extract animations
  const keyframes = extractKeyframes(allCSS);
  const transitions = extractTransitions(allCSS);
  const animationProps = extractAnimationProperties(allCSS);
  const libraries = detectAnimationLibraries(html);

  // Write outputs
  if (!fs.existsSync(ASSETS_DIR)) fs.mkdirSync(ASSETS_DIR, { recursive: true });

  // Keyframes
  const keyframesFile = path.join(ASSETS_DIR, "extracted-keyframes.css");
  fs.writeFileSync(
    keyframesFile,
    `/* Extracted keyframes from ${targetUrl} */\n/* ${new Date().toISOString()} */\n\n${keyframes.join("\n\n")}\n`
  );
  console.log(`\nWrote ${keyframes.length} @keyframes to ${keyframesFile}`);

  // Transitions
  const transitionsFile = path.join(ASSETS_DIR, "extracted-transitions.css");
  fs.writeFileSync(
    transitionsFile,
    `/* Extracted transition rules from ${targetUrl} */\n/* ${new Date().toISOString()} */\n\n${transitions.join("\n\n")}\n`
  );
  console.log(`Wrote ${transitions.length} transition rules to ${transitionsFile}`);

  // Animation properties
  const animPropsFile = path.join(ASSETS_DIR, "extracted-animation-properties.css");
  fs.writeFileSync(
    animPropsFile,
    `/* Extracted animation properties from ${targetUrl} */\n/* ${new Date().toISOString()} */\n\n${animationProps.join("\n\n")}\n`
  );
  console.log(`Wrote ${animationProps.length} animation rules to ${animPropsFile}`);

  // Inventory
  const inventory = {
    source: targetUrl,
    extractedAt: new Date().toISOString(),
    counts: {
      keyframes: keyframes.length,
      transitionRules: transitions.length,
      animationRules: animationProps.length,
    },
    detectedLibraries: libraries,
    stylesheetsFetched: stylesheetUrls.length,
  };
  const inventoryFile = path.join(ASSETS_DIR, "animation-inventory.json");
  fs.writeFileSync(inventoryFile, JSON.stringify(inventory, null, 2) + "\n");
  console.log(`Wrote inventory to ${inventoryFile}`);

  // Summary
  console.log("\n--- EXTRACTION SUMMARY ---");
  console.log(`Keyframes:          ${keyframes.length}`);
  console.log(`Transition rules:   ${transitions.length}`);
  console.log(`Animation rules:    ${animationProps.length}`);
  console.log(`Libraries detected: ${libraries.length > 0 ? libraries.join(", ") : "None"}`);
  console.log("--------------------------");

  if (libraries.length > 0) {
    console.log(`\nNOTE: Detected animation libraries: ${libraries.join(", ")}`);
    console.log("You may want to include these in your Shopify theme for full fidelity.");
  }
}

main().catch((err) => {
  console.error("Error:", err.message);
  process.exit(1);
});
