# COSMETIKA TikTok Launch Protocol
## 6-Hour Desktop Sprint

Open this file first thing tomorrow. Follow in order. Each step tells you
exactly what to paste into Claude Code.

---

## PHASE 1: EXTRACT (Hours 0-1)

### 1A — Pull Animations from Live Reference Site

Paste this into Claude Code (replace the URL with your reference site):

```
Fetch https://YOUR-REFERENCE-SITE.com and extract all CSS keyframes,
transitions, scroll-triggered animations, and any GSAP/Framer Motion
config. Output them as clean reusable CSS and JS files into
cosmetika-launch/assets/
```

If you have **Window MCP** connected and the site is open in your browser:

```
Take a screenshot of the current browser window showing the homepage,
then fetch the URL from the address bar and extract all animation CSS/JS
from the page source.
```

### 1B — Pull Product Data from Notion

If **Notion MCP** is connected:

```
Search my Notion workspace for "COSMETIKA" or "product list". Export the
database contents and convert them into the Shopify product CSV format
using the template in cosmetika-launch/templates/shopify-products.csv
```

If Notion MCP is NOT connected:
1. Open Notion
2. Go to your product database
3. Click `...` > `Export` > `Markdown & CSV`
4. Drop the CSV file into `cosmetika-launch/imports/`
5. Then paste into Claude Code:

```
Read the CSV in cosmetika-launch/imports/ and transform it into
Shopify import format using the template in
cosmetika-launch/templates/shopify-products.csv. Also generate a
TikTok Shop feed using cosmetika-launch/templates/tiktok-product-feed.csv
```

---

## PHASE 2: BUILD (Hours 1-4)

### 2A — Generate Shopify Theme

Paste into Claude Code:

```
Using the extracted animations in cosmetika-launch/assets/ and the
product data, generate a complete Shopify Dawn-based theme with:
- Homepage hero with looping video background
- Product grid with hover animations
- Product detail page with image zoom + add-to-cart animation
- Mobile-first layout (TikTok traffic is 95% mobile)
- Fast load time (inline critical CSS, lazy load everything else)
Output the theme into cosmetika-launch/theme/
```

### 2B — TikTok Integration Assets

Paste into Claude Code:

```
Generate the following TikTok Shop integration files:
1. TikTok Pixel base code snippet for Shopify theme.liquid
2. Product feed CSV formatted for TikTok Shop catalog upload
3. A meta tag block for Open Graph optimized for TikTok sharing
Output into cosmetika-launch/tiktok/
```

---

## PHASE 3: DEPLOY (Hours 4-6)

### 3A — Package Theme
```
Zip the cosmetika-launch/theme/ directory into cosmetika-theme.zip
ready for Shopify upload
```

### 3B — Deploy Checklist (Manual Steps)

- [ ] Shopify Admin > Online Store > Themes > Upload `cosmetika-theme.zip`
- [ ] Shopify Admin > Products > Import `shopify-products.csv`
- [ ] Shopify Admin > Settings > Apps > Connect TikTok channel
- [ ] TikTok Seller Center > Upload product feed CSV
- [ ] TikTok Seller Center > Set up LIVE Shopping showcase
- [ ] Test a product purchase end-to-end on mobile
- [ ] Go live

---

## EMERGENCY SHORTCUTS

If you're behind schedule, paste this:

```
I'm behind on my COSMETIKA launch. Skip the custom theme. Instead:
1. Install Shopify Dawn theme with zero modifications
2. Just generate the product import CSVs and TikTok pixel code
3. Give me only the 5 files I absolutely need to go live tonight
```

---

## FILES IN THIS KIT

| File | Purpose |
|------|---------|
| `templates/shopify-products.csv` | Pre-formatted Shopify import template |
| `templates/tiktok-product-feed.csv` | Pre-formatted TikTok catalog template |
| `scripts/extract-animations.js` | Node script to pull CSS/JS from a URL |
| `LAUNCH-PROTOCOL.md` | This file |
