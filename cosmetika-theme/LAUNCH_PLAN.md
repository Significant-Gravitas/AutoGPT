# COSMETIKA — 24-Hour Launch Execution Plan

## Reference Site Analysis (COMPLETED)
**URL:** https://fractalforest.co/
**Platform:** Shopify (Sunrise theme) — `fractal-flowers.myshopify.com`
**Animation:** No GSAP/Locomotive/Lenis on reference. CSS transitions only. Our GSAP scaffold is an upgrade.
**Palette:** `#E9E6DE` warm cream bg, `#000000` text, `#0E7A82` teal accent, `#676986` muted text
**Integrations on ref:** Klaviyo (email), Okendo (reviews), Rebuy (personalization), TikTok/FB/Google pixels

**Section flow extracted:**
1. Hero (fullscreen image + headline + CTA)
2. Product Carousel (3-across, stars + reviews + "Add to cart")
3. Brand Story / Mission (split: image + text + CTA)
4. Shop Kits Carousel (curated bundles)
5. Product Grid (4-column best sellers)
6. Collections Grid (4 category cards: image + title + description)
7. Ingredients Showcase (9 items, 3-column, circular images)
8. Journal (blog articles, 3-column)
9. Newsletter Popup (15% off, email capture)
10. Reviews Widget (Okendo)

---

## Pre-Requisites (Before You Sit Down)
- [ ] Shopify store created (even on free trial is fine)
- [ ] Shopify CLI installed (`npm install -g @shopify/cli`)
- [ ] Figma file with `html.to.design` export of fractalforest.co ready
- [ ] Notion "Brand Config" page populated (hex codes, fonts, voice, copy)
- [ ] Product inventory ready (CSV or manual entry)
- [ ] COSMETIKA logo in SVG format
- [ ] TikTok for Business account created
- [ ] MCPs configured: Figma Dev Mode, Notion, Window

---

## Phase 1: Fine-Tune Layout from Figma (~45 min)
**Tools:** Figma Dev Mode MCP
**Status:** SCAFFOLD DONE — just needs pixel adjustments

1. [ ] Open Figma file (html.to.design export of fractalforest.co)
2. [ ] Extract from Figma Dev Mode:
   - Grid column counts + gap values → cross-check with our CSS grid
   - Section padding / margin values → update `--space-*` tokens
   - Font sizes per heading level → update `clamp()` values
   - Border radius values → update `--radius-*` tokens
   - Container max-width → update `--container-max`
3. [ ] Update CSS custom properties in `cosmetika-base.css :root`
4. [ ] Adjust section HTML structure if Figma layout differs

## Phase 2: Brand Injection (~45 min)
**Tools:** Notion MCP

1. [ ] Pull from Notion Brand Config:
   - Primary / secondary / accent hex codes → `cosmetika-base.css :root` + `settings_schema.json`
   - Font names → Google Fonts link in `theme.liquid`
   - Brand voice / copy → section default text in `index.json`
   - Taglines → hero subtitle/heading
2. [ ] Upload COSMETIKA logo SVG to `assets/images/`
3. [ ] Update placeholder copy in all `.liquid` section files

## Phase 3: Motion Polish (~30 min)
**Tools:** Window MCP (optional — reference uses basic CSS only)

The reference site uses **no JS animation library** — only CSS transitions. Our GSAP scaffold already exceeds the reference motion. Optional:

1. [ ] Use Window MCP to capture exact CSS transition durations from computed styles
2. [ ] Update `cosmetika-animations.js` CONFIG if desired:
   - `fadeUpDuration`, `staggerInterval`, `heroZoomDuration`
   - Easing curves
3. [ ] The current defaults (power3.out, 1s reveal, 0.1s stagger) already produce a premium feel

## Phase 4: Product Upload (~1-2 hours)
**Tools:** Shopify Admin / CLI

1. [ ] Upload products via CSV or manual entry
   - Ensure every product meets TikTok sync requirements (see TIKTOK_SYNC_CHECKLIST.md)
2. [ ] Create collections: All, Best Sellers, New Arrivals, Skincare, Makeup, Fragrance, Wellness
3. [ ] Assign collections in theme editor:
   - "Trending Now" carousel → Best Sellers collection
   - "Shop Kits" carousel → Kits collection
   - "Best Sellers" grid → Best Sellers collection
   - Collections grid → assign each category
4. [ ] Set up navigation menus (main-menu, footer)
5. [ ] Install Okendo or Judge.me for reviews (matches reference pattern)

## Phase 5: Theme Deployment (~30 min)
**Tools:** Shopify CLI

1. [ ] Connect local theme to Shopify store:
   ```
   shopify theme dev --store your-store.myshopify.com
   ```
2. [ ] Preview and QA all 9 homepage sections
3. [ ] Fix any responsive / visual issues
4. [ ] Push theme:
   ```
   shopify theme push --live
   ```

## Phase 6: TikTok Bridge (~30 min)

1. [ ] Install TikTok sales channel on Shopify
2. [ ] Connect TikTok for Business account
3. [ ] Enable product catalog sync
4. [ ] Add TikTok Pixel ID to theme settings (Theme Editor → TikTok Shop Integration)
5. [ ] Verify products appear in TikTok Seller Center
6. [ ] Run through TIKTOK_SYNC_CHECKLIST.md

## Phase 7: Pre-Livestream QA (~30 min)

1. [ ] Full site walkthrough on mobile + desktop
2. [ ] Test add-to-cart → checkout flow
3. [ ] Test TikTok Shop purchase flow
4. [ ] Verify animations play correctly (GSAP scroll reveals, carousel, header)
5. [ ] Verify newsletter popup fires after 5 seconds
6. [ ] Check page speed (Lighthouse > 80 on mobile)
7. [ ] Announcement bar shows "LIVE ON TIKTOK — Shop the Launch"
8. [ ] Pin products in TikTok LIVE shopping tab

---

## File Map
```
cosmetika-theme/
├── assets/
│   ├── css/
│   │   ├── cosmetika-base.css            ← Design tokens + all component styles
│   │   └── cosmetika-animations.css      ← GSAP animation initial states + reduced motion
│   ├── js/
│   │   └── cosmetika-animations.js       ← GSAP ScrollTrigger + carousel + header controller
│   └── images/                           ← Logo, brand assets (add tomorrow)
├── config/
│   ├── settings_schema.json              ← Theme settings (colors, TikTok, social, typography)
│   └── settings_data.json                ← Default setting values
├── layout/
│   └── theme.liquid                      ← Main layout (GSAP CDN, TikTok Pixel, Google Fonts)
├── sections/
│   ├── announcement-bar.liquid           ← "LIVE ON TIKTOK" banner
│   ├── header.liquid                     ← Logo + nav dropdowns + cart + mobile toggle
│   ├── home-hero.liquid                  ← Full-bleed hero (video/image) with scroll animations
│   ├── featured-carousel.liquid          ← Product carousel (3-across) with arrows + drag scroll
│   ├── brand-story.liquid                ← Split image/text mission section
│   ├── product-grid.liquid               ← Collection-powered product grid (2/3/4 col)
│   ├── collections-grid.liquid           ← Category cards (image + title + desc + CTA)
│   ├── ingredients-showcase.liquid        ← 9-item ingredient grid (circular images)
│   ├── journal.liquid                    ← Blog article cards (2/3 col)
│   ├── newsletter-popup.liquid           ← Email capture modal (15% off)
│   └── footer.liquid                     ← Footer with social + menu columns + newsletter
├── templates/
│   └── index.json                        ← Homepage: 9 sections matching reference flow
├── LAUNCH_PLAN.md                        ← This file
└── TIKTOK_SYNC_CHECKLIST.md              ← TikTok product sync requirements
```

## Section-to-Reference Mapping
| # | Reference (fractalforest.co) | COSMETIKA Section File | Status |
|---|---|---|---|
| 1 | Hero ("Plant Based Tek") | `home-hero.liquid` | Built |
| 2 | Superfoods Carousel | `featured-carousel.liquid` | Built |
| 3 | Our Mission | `brand-story.liquid` | Built |
| 4 | Shop Kits Carousel | `featured-carousel.liquid` (reuse) | Built |
| 5 | Biohacking Grid | `product-grid.liquid` | Built |
| 6 | Collections Grid (4 cards) | `collections-grid.liquid` | Built |
| 7 | Ingredients (9 botanicals) | `ingredients-showcase.liquid` | Built |
| 8 | Journal | `journal.liquid` | Built |
| 9 | Newsletter Popup (15% off) | `newsletter-popup.liquid` | Built |
| 10 | Okendo Reviews | Install Okendo app (matches ref) | Tomorrow |
