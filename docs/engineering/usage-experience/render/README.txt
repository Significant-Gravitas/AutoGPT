Actual component screenshots
============================

These fixtures render the source components that ship in this PR. The billing
screen imports the real billing page, not a separate arrangement of its cards.
The notice, limit dialog, plan chooser, activation review/status, Max review, and
chat composer all import the application components. Presentation text, styling,
icons, controls, and layout come from those components and the real globals.css
and Tailwind configuration. There is no screenshot-only CSS override.

Only service responses, navigation, inactive integrations, and backend actions
are replaced with deterministic fixtures. Prices, account data, dates, invoice
line items, draft text, and percentages are illustrative, not customer data.
The seeded sessionStorage draft is read by the real useChatDraft hook, remains
editable, and has sending disabled in paused states. Closed/empty ancillary
composer views are mocked; they are not visible in the captures. No payment or
other remote action is performed by this fixture.

From the repository root, after installing frontend dependencies and generating
the API client:

  node docs/engineering/usage-experience/render/build.mjs
  USAGE_SCREENSHOTS_OUTPUT=/absolute/output/path \
    node docs/engineering/usage-experience/render/capture.mjs

The build uses the installed frontend's esbuild (via Vite), PostCSS, Tailwind,
and React. The capture uses its @playwright/test installation. Install its
Chromium browser beforehand, or set CHROMIUM_EXECUTABLE to an existing executable.
No production server, account, API credential, or database is needed.

Optional environment settings:
  USAGE_RENDER_OUTPUT        Bundle directory (default /tmp/autogpt-usage-render)
  USAGE_SCREENSHOTS_OUTPUT   PNG/manifest output directory
  USAGE_SCREENSHOT_WIDTHS    Comma-separated viewport widths
  USAGE_SCREENSHOT_CASES     Comma-separated IDs from cases.mjs
  CHROMIUM_EXECUTABLE        Explicit existing Chromium executable

The default capture covers every case at 1280, 1000, 390, and 320 CSS pixels.
1280 selects the real desktop modal. 1000 selects the real tablet drawer because
the application switches at 1024. Each PNG crops to the component boundary with
no rescaling. Billing and composer screenshots omit the unchanged app shell.
For long mobile drawers only the viewport height is increased until all default
visible content fits; the exact viewport and bounds are recorded. No CSS is
modified and payment terms or controls are not clipped. Collapsed disclosures
are captured in their actual default state. Page clock, locale and timezone are
fixed to October 7, 2026 at 12:00 UTC and en-US so dates are reproducible.

capture-manifest.json records each screenshot's component text, reachable
controls, scenario data, dimensions, and SHA-256, plus the source-file hashes
from the actual build. It also records page/console errors and browser cleanup.
The capture fails on horizontal page overflow, omitted scrollable content,
unreachable visible controls, missing saved drafts, or disabled draft editing.
The browser, contexts and local server are closed in finally, including errors.

Regenerate after any component, styling, fixture or copy change. Rebuild after
formatting so source hashes correspond to the exact files committed. Publishing
must pin image links to the commit that contains these regenerated screenshots.

Before publishing, verify that no source or PNG changed since capture:

  node docs/engineering/usage-experience/render/verify.mjs \
    /absolute/output/path/capture-manifest.json

The verification checks the exact files used in the compiled component bundle,
fixtures, renderer, fonts and CSS, as well as each screenshot's hash and successful
cleanup. It fails rather than silently allowing stale renders.

To reproduce the compact before/after composites, install Python Pillow and the
DejaVu Sans system fonts, then run from the repository root:

  python docs/engineering/usage-experience/render/compose.py \
    docs/engineering/usage-experience/production \
    docs/engineering/usage-experience/implementation/captures \
    docs/engineering/usage-experience/comparisons

This only places the browser PNGs beside one another with comparison labels.
It never redraws or changes the UI. Production captures use DPR 2 and the new
captures DPR 1; both are placed at the same CSS-pixel scale before making the
compact composite. Full component PNGs remain separately downloadable.

Set USAGE_COMPARISON_FULL_OUTPUT to also export full-size comparison sheets.
The default comparison sheets stay compact for review in GitHub and chat.
