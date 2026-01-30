# COSMETIKA — 24-Hour Launch Execution Plan

## Pre-Requisites (Before You Sit Down)
- [ ] Shopify store created (even on free trial is fine)
- [ ] Shopify CLI installed (`npm install -g @shopify/cli`)
- [ ] Reference URL ready
- [ ] Figma file with `html.to.design` export ready
- [ ] Notion "Brand Config" page populated (hex codes, fonts, voice, copy)
- [ ] Product inventory ready (CSV or manual entry)
- [ ] COSMETIKA logo in SVG format
- [ ] TikTok for Business account created

---

## Phase 1: Motion Extraction (~1 hour)
**Tools:** Window MCP + Browser DevTools

1. [ ] Open reference site in browser
2. [ ] Use Window MCP to inspect the DOM — identify:
   - GSAP version / animation library used
   - Scroll-trigger class names and data attributes
   - Easing curves (check computed styles or JS source)
   - Transition durations
   - Parallax speeds
3. [ ] Update `cosmetika-animations.js` CONFIG object with extracted values
4. [ ] Update `cosmetika-animations.css` timing if CSS-driven animations found
5. [ ] Note any custom JS (e.g., text splitting, magnetic cursor) — recreate in our scaffold

## Phase 2: Layout Match (~1 hour)
**Tools:** Figma Dev Mode MCP

1. [ ] Open Figma file (html.to.design export)
2. [ ] Extract from Figma Dev Mode:
   - Grid column counts + gap values
   - Section padding / margin values
   - Font sizes (clamp values for responsive)
   - Border radius values
   - Color tokens (cross-reference with Notion)
3. [ ] Update CSS custom properties in `cosmetika-base.css`
4. [ ] Adjust section HTML structure if Figma layout differs from current scaffold

## Phase 3: Brand Injection (~1 hour)
**Tools:** Notion MCP

1. [ ] Pull from Notion Brand Config:
   - Primary / secondary / accent hex codes → `settings_schema.json` + CSS vars
   - Font names → Google Fonts link in `theme.liquid`
   - Brand voice / copy → section default text
   - Taglines → hero subtitle/heading
2. [ ] Upload COSMETIKA logo SVG to `assets/images/`
3. [ ] Update placeholder copy in all section files

## Phase 4: Product Upload (~1-2 hours)
**Tools:** Shopify Admin / CLI

1. [ ] Upload products via CSV or manual entry
   - Ensure every product meets TikTok sync requirements (see TIKTOK_SYNC_CHECKLIST.md)
2. [ ] Create collections: All, Best Sellers, New Arrivals
3. [ ] Assign products to collections
4. [ ] Set up navigation menus (main-menu, footer)

## Phase 5: Theme Deployment (~30 min)
**Tools:** Shopify CLI

1. [ ] Connect local theme to Shopify store:
   ```
   shopify theme dev --store your-store.myshopify.com
   ```
2. [ ] Preview and QA all sections
3. [ ] Fix any responsive / visual issues
4. [ ] Push theme:
   ```
   shopify theme push --live
   ```

## Phase 6: TikTok Bridge (~30 min)

1. [ ] Install TikTok sales channel on Shopify
2. [ ] Connect TikTok for Business account
3. [ ] Enable product catalog sync
4. [ ] Add TikTok Pixel ID to theme settings
5. [ ] Verify products appear in TikTok Seller Center
6. [ ] Run through TIKTOK_SYNC_CHECKLIST.md

## Phase 7: Pre-Livestream QA (~30 min)

1. [ ] Full site walkthrough on mobile + desktop
2. [ ] Test add-to-cart → checkout flow
3. [ ] Test TikTok Shop purchase flow
4. [ ] Verify animations play correctly
5. [ ] Check page speed (Lighthouse > 80 on mobile)
6. [ ] Announcement bar says "LIVE ON TIKTOK"
7. [ ] Pin products in TikTok LIVE shopping tab

---

## File Map
```
cosmetika-theme/
├── assets/
│   ├── css/
│   │   ├── cosmetika-base.css          ← Design tokens + component styles
│   │   └── cosmetika-animations.css    ← Animation initial states
│   ├── js/
│   │   └── cosmetika-animations.js     ← GSAP ScrollTrigger controller
│   └── images/                         ← Logo, brand assets
├── config/
│   ├── settings_schema.json            ← Theme settings (colors, TikTok, social)
│   └── settings_data.json              ← Default setting values
├── layout/
│   └── theme.liquid                    ← Main layout (head, GSAP CDN, TikTok Pixel)
├── sections/
│   ├── announcement-bar.liquid         ← "LIVE ON TIKTOK" banner
│   ├── header.liquid                   ← Logo + nav + cart
│   ├── home-hero.liquid                ← Full-bleed hero with video/image
│   ├── product-grid.liquid             ← Collection-powered product grid
│   └── footer.liquid                   ← Footer with social links
├── templates/
│   └── index.json                      ← Homepage section order
├── LAUNCH_PLAN.md                      ← This file
└── TIKTOK_SYNC_CHECKLIST.md            ← TikTok sync requirements
```
