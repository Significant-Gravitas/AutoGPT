# Lane I — Marketing & SEO

Runs only when the recon found marketing routes: a landing page, pricing, blog, docs, or changelog. These are the pages a stranger sees first, the pages that get shared, and the pages a crawler reads. They get one visit to make their case, so the failures here are different from product UI: the link preview with no image, the page that performs at the visitor instead of answering them, the blog that's fetched from the CMS on every request. Owning skill: `emil-marketing-pages`.

Static generation of content routes is B15. Fabricated proof is H2. Count each once.

## I1. Every public route has a title and description — BLOCKER

A page with no `<title>` shows its URL in the tab, in search results, and in every link preview. The framework default ("Create Next App") is worse.

**Hunt for:** the root layout's `metadata` / `<head>`; public routes with no `metadata` export, `generateMetadata`, or `<title>`; the same title on every route; dynamic routes (`/blog/[slug]`) with static metadata.

```tsx
export const metadata: Metadata = {
  metadataBase: new URL("https://example.com"),
  title: { default: "Product", template: "%s · Product" },
  description: "…",
};
```

```tsx
// app/blog/[slug]/page.tsx
export async function generateMetadata({ params }) {
  const post = await getPost((await params).slug);
  return { title: post.title, description: post.excerpt };
}
```

The title template and per-route wiring are a fix. The words are content: take the title from the page's `<h1>` and the description from existing copy (the hero subhead, the post excerpt). If there's nothing to take them from, flag it (Hard Rule 8). Without `metadataBase`, relative OG image URLs resolve to `localhost` in production.

## I2. Shared links have a preview — SHOULD FIX, BLOCKER for a launch that depends on sharing

Launch day *is* people pasting the link into Slack, X, iMessage, and LinkedIn. With no Open Graph image the link is a bare URL; with a broken one it's a grey box.

**Hunt for:** `openGraph` / `twitter` in metadata, or `og:image` / `twitter:card` meta tags; an `opengraph-image` file in the app directory; an image path that doesn't exist in `public/`; an image under 1200×630.

```tsx
openGraph: { images: [{ url: "/og.png", width: 1200, height: 630 }] },
twitter: { card: "summary_large_image" },
```

If no OG image exists in the repo, that's the human list: it can't be invented. Blog posts and docs pages get per-page images when the project already has a generator; otherwise the site-wide one is fine. Put on the device list: "paste the production URL into Slack and iMessage and look at the preview".

## I3. Crawlers can find the real pages and not the fake ones — SHOULD FIX

**Hunt for:** `robots.txt` / `robots.ts` (missing, or a leftover `Disallow: /` from staging, which de-indexes the whole site); `sitemap.xml` / `sitemap.ts` (missing, or listing `/proto`, `/lab`, and auth-gated routes); `<link rel="canonical">` on pages reachable at more than one URL; `noindex` left on from a staging config.

A leftover `Disallow: /` or a site-wide `noindex` is a BLOCKER. The scaffolding routes from H10 go in `disallow` as a second layer.

## I4. Favicon and app icons are the product's — SHOULD FIX

**Hunt for:** the framework's default `favicon.ico`; no `apple-touch-icon` (iOS uses a screenshot of the page when someone adds it to their home screen); no SVG icon; a manifest with default `name` / `short_name`.

Check the icon on a dark tab bar (G6). A missing icon asset is the human list.

## I5. Motion maps to user input — SHOULD FIX

Every section fading up as it scrolls into view, scroll hijacking, parallax that isn't 1:1 with scroll, an auto-advancing carousel. All of these move without the user asking, and the page feels like it's performing at the visitor. Scroll-triggered reveals also hide content from anyone who scrolls fast, prints, or arrives with JavaScript still loading.

**Hunt for:** `whileInView`, `useInView`, `IntersectionObserver` driving opacity or translate on content sections; AOS / ScrollReveal / GSAP ScrollTrigger on entrances; `wheel` listeners with `preventDefault`; `scroll-snap` on the page itself; carousels with `autoplay` / `setInterval`.

Removing these changes the design, so it's an ask, not a silent fix. Two exceptions are defects and get fixed outright: content that stays at `opacity: 0` when the observer never fires (no-JS, reduced motion, print), and an auto-advancing carousel with no pause control (also a WCAG failure).

## I6. Intro animations play once per session — POLISH

A hero entrance that replays on every navigation back to the homepage is a tax. Gate it on `sessionStorage`, not `localStorage`, so it plays again on a new visit but not on every internal navigation.

```jsx
useEffect(() => {
  if (sessionStorage.getItem("hasSeenIntro")) setSkipIntro(true);
  else sessionStorage.setItem("hasSeenIntro", "true");
}, []);
```

## I7. Heroes don't shift or overflow — SHOULD FIX

The hero is the LCP element and the first thing that can go wrong. Fonts preloaded (B2), the hero image preloaded with dimensions (B1, B3), `min-height: 100svh` not `100vh` or `100dvh` (C7), and a headline with `text-wrap: balance` (H5) checked at 375px wide, where a long word in a large size overflows the viewport (C17).

## I8. Nav submenu content lives in the DOM — SHOULD FIX

Header submenus that mount their content on hover are invisible to crawlers and to assistive tech, and unreachable on touch. Render the content always and hide it visually; hover and focus reveal it.

**Hunt for:** `{isOpen && <Submenu />}` in the marketing header; submenus opened by `onMouseEnter` only, with no click or focus path.

```html
<nav>
  <button aria-expanded="false" aria-controls="products-menu">Products</button>
  <div id="products-menu" class="submenu" hidden>…full content…</div>
</nav>
```

## I9. CTAs know who's looking — SHOULD FIX

A logged-in user told to "Sign up" is a dead end, or worse, lands on a signup form that errors. Logged-out visitors get "Get started"; logged-in users get "Go to dashboard".

**Hunt for:** hero and header CTAs with a hardcoded `/signup` href in an app that has auth.

Resolve this on the client so the marketing page stays static (B15); reading cookies in the layout to decide the CTA opts the whole page out of static rendering.

## I10. The conversion path works end to end — BLOCKER

The one flow that pays for the launch: landing → CTA → signup or checkout → first screen. Every link in it resolves, the pricing on the page matches the pricing in checkout, the plan the button names is the plan the checkout opens, and the post-signup redirect lands somewhere that exists.

**Hunt for:** CTA hrefs pointing at routes that don't exist; price strings hardcoded on the pricing page vs the values in the billing config; test-mode payment keys or links (`buy.stripe.com/test_`, sandbox checkout URLs); `mailto:` and social links with placeholder handles; waitlist and newsletter forms whose `action` or handler posts nowhere.

Mismatched prices and test-mode payment links are blockers and go at the top of the report. You can find them from code; you can't confirm the fix without the user, so state both values and ask which is right.

## I11. Docs are for copying — POLISH

Every code snippet has a copy button that confirms (E13). Pages are available as markdown (a "Copy as Markdown" button or `.md` URLs), which serves both people and the LLMs reading the docs. Concepts get a visual example, not only code.

## I12. Blogs and changelogs are feeds — POLISH

RSS at a predictable path (`/blog/rss.xml`, `/changelog/rss.xml`), linked from the `<head>` with `<link rel="alternate" type="application/rss+xml">`. Dates in `<time datetime>`.

## I13. Analytics and consent are wired, once — SHOULD FIX

Launch day traffic is the traffic you most want measured, and it's the day people discover the analytics snippet was only on the homepage, or is loaded twice, or is blocking render.

**Hunt for:** analytics / tag scripts in the root layout vs a single page; the same snippet included twice; third-party `<script>` without `async` / `defer` or the framework's script strategy; a cookie banner that pushes content down when it mounts (E8) or covers the mobile CTA (C15).

Whether the site needs a consent banner is a legal decision, not a ship-check one. Note what tracking exists and leave the decision on the human list.

## I14. Look at it as a stranger — always

Put on the device list: "open the production URL in a private window on a phone over cellular", "paste the URL into Slack, X, and iMessage and check each preview", "run the production build through Lighthouse on mobile and read the LCP and CLS numbers", "click the primary CTA and complete the whole flow with a real card in live mode, then refund it", "view source and confirm the headline and body copy are in the HTML". Report these as checks to run, never as verified.
