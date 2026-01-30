#!/usr/bin/env node
/**
 * notion-to-shopify.js
 *
 * Converts a Notion database CSV export into:
 *   1. Shopify product import CSV
 *   2. TikTok Shop product feed CSV
 *
 * Usage:
 *   node notion-to-shopify.js ../imports/notion-export.csv
 *
 * Output:
 *   ../imports/shopify-import-ready.csv
 *   ../imports/tiktok-feed-ready.csv
 *
 * The script auto-maps common Notion column names to Shopify/TikTok fields.
 * Edit the COLUMN_MAP below if your Notion columns use different names.
 */

const fs = require("fs");
const path = require("path");

const inputFile = process.argv[2];
if (!inputFile) {
  console.error("Usage: node notion-to-shopify.js <path-to-notion-csv>");
  process.exit(1);
}

// --- CONFIGURATION ---
// Map your Notion column headers to internal field names.
// Add or rename keys to match your actual Notion export.
const COLUMN_MAP = {
  // Notion column name (case-insensitive) => internal key
  "name": "title",
  "title": "title",
  "product name": "title",
  "product": "title",
  "description": "description",
  "body": "description",
  "details": "description",
  "price": "price",
  "retail price": "price",
  "compare at price": "comparePrice",
  "original price": "comparePrice",
  "msrp": "comparePrice",
  "sku": "sku",
  "variant sku": "sku",
  "type": "type",
  "product type": "type",
  "category": "type",
  "tags": "tags",
  "tag": "tags",
  "vendor": "vendor",
  "brand": "vendor",
  "image": "image",
  "image url": "image",
  "photo": "image",
  "image src": "image",
  "inventory": "inventory",
  "qty": "inventory",
  "quantity": "inventory",
  "stock": "inventory",
  "weight": "weight",
  "weight (g)": "weight",
  "color": "color",
  "colour": "color",
  "size": "size",
  "option": "option1",
  "variant": "option1",
};

const STORE_URL = "https://your-store.myshopify.com"; // Replace with your actual store URL
const BRAND = "COSMETIKA";

// --- CSV PARSING ---
function parseCSV(text) {
  const lines = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (char === '"') {
      if (inQuotes && text[i + 1] === '"') {
        current += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === "," && !inQuotes) {
      lines.push(current);
      current = "";
    } else if ((char === "\n" || char === "\r") && !inQuotes) {
      if (current || lines.length > 0) {
        lines.push(current);
        current = "";
      }
      if (lines.length > 0) {
        yield lines.splice(0);
      }
      if (char === "\r" && text[i + 1] === "\n") i++;
    } else {
      current += char;
    }
  }
  if (current || lines.length > 0) {
    lines.push(current);
    yield lines.splice(0);
  }
}

function* parseCSVGenerator(text) {
  const rows = text.split(/\r?\n/);
  for (const row of rows) {
    if (!row.trim()) continue;
    const fields = [];
    let current = "";
    let inQuotes = false;
    for (let i = 0; i < row.length; i++) {
      const c = row[i];
      if (c === '"') {
        if (inQuotes && row[i + 1] === '"') {
          current += '"';
          i++;
        } else {
          inQuotes = !inQuotes;
        }
      } else if (c === "," && !inQuotes) {
        fields.push(current.trim());
        current = "";
      } else {
        current += c;
      }
    }
    fields.push(current.trim());
    yield fields;
  }
}

function slugify(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

function escapeCSV(val) {
  if (val === undefined || val === null) return "";
  const str = String(val);
  if (str.includes(",") || str.includes('"') || str.includes("\n")) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

// --- MAIN ---
function main() {
  const raw = fs.readFileSync(path.resolve(inputFile), "utf-8");
  const gen = parseCSVGenerator(raw);
  const headerRow = gen.next().value;

  if (!headerRow) {
    console.error("Empty CSV file.");
    process.exit(1);
  }

  // Map headers
  const colIndices = {};
  headerRow.forEach((h, i) => {
    const normalized = h.toLowerCase().trim();
    if (COLUMN_MAP[normalized]) {
      colIndices[COLUMN_MAP[normalized]] = i;
    }
  });

  console.log("Detected column mapping:");
  for (const [key, idx] of Object.entries(colIndices)) {
    console.log(`  ${key} => column ${idx} ("${headerRow[idx]}")`);
  }

  const products = [];
  for (const row of gen) {
    if (!row || row.length === 0) continue;
    const get = (key) => (colIndices[key] !== undefined ? row[colIndices[key]] || "" : "");

    const title = get("title");
    if (!title) continue;

    products.push({
      title,
      description: get("description"),
      price: get("price").replace(/[^0-9.]/g, "") || "0.00",
      comparePrice: get("comparePrice").replace(/[^0-9.]/g, "") || "",
      sku: get("sku") || `COSM-${slugify(title).substring(0, 10).toUpperCase()}-${String(products.length + 1).padStart(3, "0")}`,
      type: get("type") || "Beauty",
      tags: get("tags") || "cosmetika, beauty, tiktok",
      vendor: get("vendor") || BRAND,
      image: get("image"),
      inventory: get("inventory") || "100",
      weight: get("weight") || "200",
      color: get("color"),
      size: get("size"),
    });
  }

  console.log(`\nParsed ${products.length} product(s) from Notion export.`);

  // --- SHOPIFY CSV ---
  const shopifyHeaders = [
    "Handle", "Title", "Body (HTML)", "Vendor", "Product Category", "Type", "Tags",
    "Published", "Option1 Name", "Option1 Value", "Variant SKU", "Variant Grams",
    "Variant Inventory Tracker", "Variant Inventory Qty", "Variant Inventory Policy",
    "Variant Fulfillment Service", "Variant Price", "Variant Compare At Price",
    "Variant Requires Shipping", "Variant Taxable", "Image Src", "Image Position",
    "Image Alt Text", "Gift Card", "SEO Title", "SEO Description", "Status",
  ];

  const shopifyRows = [shopifyHeaders.join(",")];
  for (const p of products) {
    const handle = slugify(p.title);
    const htmlBody = `<p>${p.description}</p>`;
    const seoTitle = `${p.title} | ${BRAND}`;
    const seoDesc = p.description.substring(0, 320);
    const optionName = p.color ? "Color" : p.size ? "Size" : "";
    const optionValue = p.color || p.size || "";

    const row = [
      handle, p.title, htmlBody, p.vendor,
      "Health & Beauty > Personal Care > Cosmetics", p.type,
      p.tags, "true", optionName, optionValue, p.sku, p.weight,
      "shopify", p.inventory, "deny", "manual", p.price, p.comparePrice,
      "true", "true", p.image, "1", p.title, "false", seoTitle, seoDesc, "active",
    ];
    shopifyRows.push(row.map(escapeCSV).join(","));
  }

  const shopifyOut = path.join(path.dirname(path.resolve(inputFile)), "shopify-import-ready.csv");
  fs.writeFileSync(shopifyOut, shopifyRows.join("\n") + "\n");
  console.log(`Wrote Shopify CSV: ${shopifyOut}`);

  // --- TIKTOK FEED CSV ---
  const tiktokHeaders = [
    "sku_id", "title", "description", "availability", "condition", "price",
    "sale_price", "link", "image_link", "brand",
    "google_product_category", "item_group_id", "color", "size",
    "shipping_weight", "age_group", "gender",
  ];

  const tiktokRows = [tiktokHeaders.join(",")];
  for (const p of products) {
    const handle = slugify(p.title);
    const row = [
      p.sku, p.title, p.description, "in stock", "new",
      `${p.comparePrice || p.price} USD`, `${p.price} USD`,
      `${STORE_URL}/products/${handle}`, p.image, BRAND,
      "Health & Beauty > Personal Care > Cosmetics",
      p.sku.split("-").slice(0, 2).join("-"),
      p.color, p.size, `${p.weight} g`, "adult", "unisex",
    ];
    tiktokRows.push(row.map(escapeCSV).join(","));
  }

  const tiktokOut = path.join(path.dirname(path.resolve(inputFile)), "tiktok-feed-ready.csv");
  fs.writeFileSync(tiktokOut, tiktokRows.join("\n") + "\n");
  console.log(`Wrote TikTok feed CSV: ${tiktokOut}`);

  console.log("\nDone. Next steps:");
  console.log("  1. Import shopify-import-ready.csv in Shopify Admin > Products > Import");
  console.log("  2. Upload tiktok-feed-ready.csv in TikTok Seller Center > Product Catalog");
}

main();
