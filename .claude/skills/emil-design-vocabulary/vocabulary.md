# Design Vocabulary

*The words designers use when they know what they're looking at.*


## Typography

**Kerning** — The space between two specific characters, adjusted manually. Different from tracking, which affects all characters equally.

**Tracking** — Letter-spacing applied uniformly across a word or block of text. Uppercase labels almost always need more of it.

**Leading** — The vertical space between lines of text. Too tight and text suffocates. Too loose and it stops reading as a paragraph.

**Optical kerning** — Spacing adjusted by eye rather than the font's built-in metrics. Removes awkward gaps between certain letter pairs that default metrics leave behind.

**Tabular nums** — A font feature that gives every digit the same width so numbers in columns stay aligned as they change. Essential for prices, stats, and any data that updates.

**Type scale** — A predetermined set of font sizes that work together, usually based on a ratio. Picking sizes from the scale keeps hierarchy consistent.

**Weight** — How thick or thin a typeface's strokes are. Bold is for UI emphasis and hierarchy. Italic is for linguistic stress or citation, not UI hierarchy.

**x-height** — The height of a lowercase letter like x, relative to the cap height. A high x-height makes a font feel larger at the same point size, which is why two fonts at 16px can look very different in size.

**Cap height** — The height of a capital letter. Along with x-height, it determines how large a font feels on screen. Useful when matching fallback fonts or pairing typefaces.

**Ligature** — Two characters that merge into a single glyph to avoid awkward collisions. fi and fl are the most common. Some ligatures are decorative and belong in display use only.

**Font smoothing** — How the browser renders font edges. On Mac, antialiased renders thinner and lighter than the default. Designers often apply it globally for a crisper feel, though it reduces fidelity on non-retina screens.

**Text overflow** — What happens when text is longer than its container. Truncation with an ellipsis is the most common solution. Use the single unicode character, not three periods, and never cut mid-word.

**Hyphenation** — Breaking a word across lines with a hyphen. Helps with narrow columns and justified text. Best left to the browser automatically to avoid breaking at unexpected moments.

**Clamp** — A technique for fluid type sizing that sets a minimum, preferred, and maximum value. The size scales smoothly between breakpoints without needing separate declarations at each one.

**Widow** — A single word left alone on the last line of a paragraph. text-wrap: balance can fix it automatically.

**Orphan** — A single line of a paragraph stranded at the top of a new column or page. The CSS orphans property exists for this, though browser support is inconsistent. Often fixed manually.

**Font stack** — The ordered list of fonts the browser tries before giving up. A well-matched fallback font avoids layout shift when the primary font loads, since mismatched x-heights and weights cause content to reflow.

**Line length** — How wide a column of text runs. Around 65 characters is comfortable. Wider and the eye struggles to find the next line.

**Variable font** — A single font file containing a range of weights, widths, or other axes. Lets you animate weight without layout shift and reduces font file overhead.

**Superscript** — Raised characters for footnotes or exponents. The HTML sup element is too large by default and disrupts line spacing. Proper superscripts are manually sized and shifted.

**Subscript** — Lowered characters for formulas or notation. The HTML sub element has the same problem as sup: oversized by default and disruptive to line rhythm if left unstyled.


## Color

**sRGB** — The standard color space most screens have used for decades. All hex values live here. Its gamut is limited: there are colors a modern display can show that sRGB cannot describe.

**P3** — A wider color space supported by most modern screens. Allows more vivid colors, particularly greens and reds, that sRGB cannot reach.

**OKLCH** — A color space that matches how humans perceive brightness. Unlike HSL, two colors at the same lightness value actually look the same brightness. Useful for generating palettes and accessible color mixes.

**Alpha** — The transparency component of a color, from fully opaque to fully invisible. An alpha border recedes into the surface beneath it. A solid hex sits on top. The difference is subtle but consistently felt.

**Semantic token** — A color variable named for its purpose rather than its value. --color-border-subtle instead of #e0e0e0. When the value changes, everything using the token updates.

**Contrast ratio** — The luminance difference between a foreground and background color. WCAG requires 4.5:1 for body text, 3:1 for large text and UI components.

**Tinted neutral** — A grey with a slight hue bias. Pure #808080 looks like a placeholder. A tinted neutral feels chosen.

**Saturation** — How vivid or muted a color is. Brand colors at full saturation in dark mode can vibrate off the screen. Pulling back 20-30% settles them.

**Chroma** — The OKLCH equivalent of saturation, but perceptually accurate. Reducing chroma for a light tint keeps the color alive. Reducing opacity does the same but turns it grey and lifeless.

**Gradient** — A transition between two or more colors. In HSL or sRGB, the midpoint often goes grey. Building gradients in OKLCH keeps the colors vivid throughout.

**Opacity vs visibility** — Two ways to hide an element. Opacity zero still occupies space and receives pointer events. Visibility hidden removes interaction but keeps the space. Worth knowing which you need.

**Dark mode** — An interface built on dark surfaces. The brightest surface should sit at the top of the layering hierarchy. Light mode tokens rarely translate directly, so every color decision needs rechecking.

**Blending** — How a layer interacts with what's beneath it. Multiply darkens, Screen lightens, Overlay does both depending on the base color. Useful for integrating images with colored surfaces and adding depth to illustrations.


## Iconography

**Stroke weight** — How thick the lines of an icon are. Thin strokes disappear at small sizes. At large sizes they look frail. Weight needs to scale with size.

**Optical centre** — Where something looks centred versus where it mathematically is. A play button centred by coordinates looks left-heavy. Nudge it right and it sits.

**Filled vs outlined** — Two icon styles for communicating state. Filled often signals active or selected. Outlined signals default. Mixing both without a clear system creates noise.

**Cap style** — Whether stroke ends are square or round. Round caps feel finished in most UI contexts. Square caps can read as unrefined.

**Pixel hinting** — Adjusting icon paths to sit cleanly on a pixel grid at small sizes. A full-detail logo at 16px turns to mush without it.

**Icon library** — A set of icons drawn at the same size, weight, and radius. Mixing two libraries, even similar ones, creates subtle mismatches that compound across a product.


## Layout

**Border radius** — The rounding on a corner. An inner element inside a rounded container needs a smaller radius, calculated as outer radius minus padding. Matching both creates a visible gap.

**Gap** — Space between flex or grid children, set on the parent. Unlike margin, it leaves no trailing space after the last item.

**Negative space** — The empty area around and between elements. It defines shape, creates breathing room, and guides the eye. Crowding elements together removes clarity, it doesn't add information.

**Flexbox** — A CSS layout model for arranging elements in a row or column. Controls direction, alignment, spacing, and how items grow or shrink to fill available space. The foundation of most modern UI layout.

**Auto layout** — Figma's implementation of flexbox: elements that space themselves, resize with their content, and respond to direction and alignment rules.

**Layout shift** — When elements move unexpectedly as a page loads: images popping in, fonts swapping, content reordering. Avoided by reserving space before content loads and using size-matched font fallbacks.

**Overflow** — What happens when content is larger than its container. Hidden clips silently. Scroll adds a scrollbar. Auto only shows one when needed. Hidden also silently breaks sticky positioning on child elements.

**Sticky positioning** — An element that scrolls normally until it hits a threshold, then fixes in place. Requires a parent that is not set to overflow hidden, which silently breaks it.

**Aspect ratio** — The proportional relationship between width and height. Setting it on images and embeds lets the browser reserve the right space before the content loads, preventing layout shift.

**Viewport units** — Units relative to the browser window. vw and vh are the classics. dvh accounts for the mobile browser chrome that shows and hides on scroll. Using vh for full-screen mobile layouts often causes overflow because of this.

**Safe area** — The part of a screen not obscured by notches or home indicators. Fixed bottom elements need to account for this or they sit underneath the home indicator.

**Max-width** — A cap on how wide a container gets. Without one, text lines stretch to unreadable lengths on wide screens.

**Breakpoint** — The point at which a layout changes to suit a different screen size. Best set where the content actually breaks, not at assumed device widths.

**Responsive** — A design that adapts to different screen sizes rather than being fixed. Achieved through fluid layouts, flexible images, and breakpoints. The baseline expectation for anything on the web.

**Grid** — A system of columns that organises layout. 12 columns is conventional. 8 often works better for simpler layouts.

**Asymmetry** — A layout where columns or elements are intentionally unequal. Creates tension and visual interest. Symmetry is stable but rarely interesting.

**Baseline grid** — A horizontal rhythm built from the line-height of body text. Creates consistency in editorial layouts. Often overkill in product UI.

**Z-index** — The stacking order of elements on the same plane. Modals, tooltips, and dropdowns need explicit z-index values or they clip behind things they should sit above.


## Interaction

**Affordance** — The visual signal that tells you how something can be used. A button looks pressable. A link looks clickable. When affordance is missing, users guess.

**Hover state** — The visual change when a cursor moves over an interactive element. Should confirm interactivity through cursor change, color shift, or both. Color alone is not enough.

**Focus state** — The visible indicator that an element is keyboard-focused. Removing it is an accessibility failure. Replace it with a custom style, never just delete it.

**Active state** — The visual change when an element is pressed. Without it, buttons feel unresponsive. A small scale or color shift is enough.

**Disabled state** — A non-interactive element. Using opacity alone is unreliable since it can pass or fail contrast checks unpredictably depending on the background. A specific muted token is more predictable.

**Cursor** — The pointer icon that signals what an element does. Pointer means clickable. Default means static. Text means selectable. The wrong cursor breaks the implicit contract.

**Pointer events** — Whether an element can receive mouse or touch input. Setting to none makes an element visually present but non-interactive, useful for decorative layers that should not intercept clicks.

**Optimistic update** — Updating the UI before the server confirms the action. Feels instant. Requires a rollback if the request fails.

**Debounce** — A delay before firing a function, reset each time it is triggered. Stops a search input making a request on every keystroke. 300ms is a common threshold.

**Touch target** — The tappable area of an interactive element. The minimum is 44x44px. The visual element can be smaller than the touch target.

**Copy to clipboard** — A button that copies text. Needs visible confirmation, typically a checkmark for a second or two. Without it, users click it repeatedly thinking it failed.

**Skip link** — A visually hidden link that jumps to the main content. Appears on focus for keyboard users. Without it, every page requires tabbing through the full navigation first.


## Motion

**Easing** — The rate at which an animation speeds up or slows down. Ease-out decelerates into place and feels natural. Ease-in starts slow and makes the interface feel like it's dragging.

**Ease-out** — Starts fast, slows down. The default for most UI transitions. Use for things entering the screen.

**Ease-in** — Starts slow, ends fast. Use for things leaving the screen. On entrances, it feels reluctant.

**Stagger** — Animating list items one after another with a small delay between each. 40ms intervals create a sense of arrival. All at once feels like a flash.

**Duration** — How long an animation takes. Hover transitions around 150ms feel native. 400ms feels like the UI is thinking. Over 400ms with no feedback looks broken.

**Transition property** — Which CSS properties animate. Animating all properties can accidentally catch layout ones and cause jank. Opacity and transform are the safe pair.

**Reduced motion** — A user preference for less animation. Ignoring it can trigger vestibular symptoms. Any significant movement should check for it.

**Skeleton shimmer** — A moving gradient that sweeps across a skeleton loader to suggest activity. Should respect reduced motion preferences.


## Accessibility

**WCAG** — Web Content Accessibility Guidelines. The standard most teams use for color contrast. AA requires 4.5:1 for body text, 3:1 for large text and UI components.

**APCA** — Advanced Perceptual Contrast Algorithm. A newer contrast model that accounts for font size and weight. More nuanced than WCAG for real-world conditions. The two specs sometimes disagree.

**Screen reader** — Software that reads interface content aloud for blind or low-vision users. Navigates by DOM order, heading structure, and ARIA roles, not visual layout. VoiceOver on Mac and iOS, NVDA and JAWS on Windows are the most common.

**Tab order** — The sequence keyboard focus moves through when pressing Tab. Should follow a logical reading order. Unexpected jumps are often caused by DOM and visual order being out of sync.

**Prefers color scheme** — A media query that detects the user's system theme preference. The right way to honour dark mode without requiring a separate in-product setting.

**aria-label** — An accessible name for an element with no visible text. Should describe the action, not the element type. Search, not icon.

**Focus trap** — Keeping keyboard focus inside an open modal or dialog. Without it, focus escapes to the page behind the overlay.

**DOM order** — The sequence of elements in the HTML. Screen readers follow this, not visual order. Reordering visually with CSS while leaving the DOM unchanged creates a disconnect between what is seen and what is read.

**Semantic HTML** — Using the right element for the job. A button handles keyboard events, focus, and role for free. A div styled as a button needs all of that added back manually.

**Label association** — Linking a label to its input via matching for and id attributes. Without it, clicking the label does not focus the field.

**Color-only state** — Using color as the sole signal for a state change. An error shown only by a red border is invisible to colorblind users. Always pair color with an icon or text.


## Information Architecture

**Progressive disclosure** — Showing only what is needed at each step, revealing complexity as the user goes deeper. The opposite is putting everything on one page.

**Navigation** — The system for moving between areas of a product. Labels should match how users describe things, not internal terminology.

**Mental model** — The user's internal picture of how something works. Good IA matches it. When it does not, users get lost and blame themselves.

**Hierarchy** — The ranking of elements by importance. Without it, everything competes and nothing wins.

**Empty state** — The view when there is no content yet. Should explain why and offer a next step. No data ends the journey.

**Error state** — What the UI shows when something fails. Should say what went wrong and offer a specific way to recover.

**Onboarding** — Introducing a new user to a product. The best onboarding delivers value immediately. A user who completes one meaningful action is more likely to return than one who watched a tour.

**Confirmation dialog** — A prompt before a destructive or irreversible action. Should describe what will be lost, not just ask are you sure. Keep the destructive and safe actions visually far apart.


## Copywriting

**Microcopy** — Short UI text: labels, error messages, placeholders, tooltips. Has an outsized effect on how trustworthy a product feels.

**CTA** — Call to action. The label on the primary action. Save changes builds more trust than Submit. Own the action, do not apologise for asking.

**Error message** — Text explaining what went wrong and how to fix it. Invalid input names nothing. Your email must include an @ symbol names everything.

**Placeholder** — Hint text inside an input that disappears when the user starts typing. Cannot replace a label. By the time they need the reminder, it is gone.

**Sentence case** — Capitalising only the first word and proper nouns. The default for UI labels. Title Case On Everything reads like a legal document.

**Front-loading** — Putting the most important word first. Export ready beats Your export is complete and available for download. Users scan.

**Inline error** — An error message next to the field that caused it. A summary at the top of a form makes users hunt. Inline is unambiguous.


## Tools

**Design system** — A shared set of components, tokens, patterns, and guidelines. At its best a living product, not a static document. The gap between the system and what ships is where inconsistency lives.

**Source of truth** — The single place a team defers to for design decisions. Traditionally a Figma file, but increasingly the codebase itself as teams work closer to the real experience than a facsimile of it. Wherever it lives, it only works if everyone uses it.

**Variables** — Named containers for values like colors, spacing, or type sizes that can be reused and updated from one place. Change the variable and everything referencing it updates. The foundation of any scalable design system.

**Tokens** — Design decisions stored as named variables that both design tools and code can reference. A token like --spacing-4 means the same thing in Figma and in a stylesheet. Tokens are what make a design system portable across platforms.

**Visual language** — The cohesive set of decisions covering color, type, shape, motion, and tone that makes a product recognisable across every touchpoint. When it's working, you know the product without seeing its name.

**Artboard / Frame** — A named canvas in a design file representing a screen, view, or component. In Sketch they're artboards, in Figma they're frames. Frames also define clipping, layout grids, and auto layout behaviour, making them more structural than their name suggests.

**Prototype** — An interactive mockup used to test or communicate a flow before it's built. Can range from a few linked screens to something nearly indistinguishable from the real product. The fidelity should match the question being asked.

**Handoff** — The moment design moves to engineering. Done well, it means accurate specs, shared tokens, and a designer who stays available for questions. Done poorly, it means developers making design decisions on their own.

**Redline / annotation** — Measurements and notes added to designs to communicate intent to developers: spacing values, color tokens, interaction behaviour, edge cases. Less necessary when teams share a design system, but still useful for anything non-standard.

**Moodboard** — A collection of visual references to explore or communicate a direction before committing. Used to align stakeholders on tone and aesthetic. The best ones make an argument. The worst are just Pinterest boards.

**HiDPI / Retina** — Screens with twice or more the pixel density of a standard display. Assets not prepared for it look blurry. Icons, images, and borders all need to account for 2x and sometimes 3x densities.

**Open Graph** — The metadata controlling how a link previews when shared: the image, title, and description. Often an afterthought, but the first impression of a page for most people who don't visit it directly.

**Skeuomorphism** — A design approach that models interfaces on real-world objects: a notes app that looks like a legal pad, a calendar with stitched leather, a trash can that looks like an actual bin. The idea was to make unfamiliar software feel approachable by referencing the physical world. Not to be confused with using shadows or gradients, which are just visual depth tools.


## Analysis

**A/B test** — Two design variants running simultaneously to measure which performs better. Validates decisions with real behaviour rather than internal opinion. Results are only as good as the metric being measured.

**Heatmap** — A visualisation of where users click, tap, or scroll on a page. Shows what draws attention and what gets ignored. Useful for spotting affordance problems and finding content nobody reaches.

**Session recording** — Video of individual user sessions through a product. Shows exactly where people hesitate, get confused, or give up. More specific than a heatmap, and harder to ignore.

**Funnel** — The path users take toward a conversion goal: signing up, purchasing, completing a task. Drop-off at any step is a design problem. Most funnels lose the majority of users before the end.

**Conversion** — The moment a visitor takes the desired action. Design affects it through clarity, trust, friction reduction, and the quality of the CTA.

**Bounce rate** — The percentage of visitors who leave without taking any action. High bounce rates on key pages often point to a mismatch between what users expected and what they found.

**Retention** — How many users come back after their first visit. A product can have strong acquisition and terrible retention. Design affects both, but retention is the harder and more important problem.

**Churn** — Users who stop using the product. The inverse of retention. High churn means the product is not delivering enough value to keep people coming back.

**NPS** — Net Promoter Score. A measure of how likely users are to recommend a product, on a scale of 0 to 10. A blunt instrument, but widely used as a rough signal of overall satisfaction.

**Scroll depth** — How far down a page users actually read. Most don't get to the bottom. Content and CTAs buried below the fold may be seen by far fewer people than assumed.


## Components

**Button** — The primary action trigger. Needs distinct default, hover, active, focus, disabled, and loading states. Never more than one primary button in the same view.

**Input** — A text entry field. Needs a persistent label above it, not just a placeholder. Each state (default, hover, focus, error, disabled) should be visually distinct.

**Textarea** — A multi-line input. Needs the same state treatment as a single-line input.

**Select** — A dropdown for choosing one option from a list. The native HTML element is accessible but hard to style. Custom selects need careful keyboard and focus handling.

**Checkbox** — A toggle for a binary choice. Clicking the label should toggle it, not just clicking the box itself.

**Radio group** — A set of options where exactly one can be selected. Distinct from checkboxes, which allow multiple.

**Switch** — A toggle for an on/off state. Implies an immediate effect. Use for settings that take effect without a save action.

**Slider** — A control for selecting a value within a range. Should update the value live on drag, not only on release.

**Modal / Dialog** — An overlay that interrupts the current flow to require attention. Focus must be trapped inside it and the background should be inert to screen readers.

**Sheet** — A panel that slides in from the screen edge. Common on mobile for secondary navigation or contextual actions. Same focus and dismissal rules as a modal.

**Drawer** — A bottom sheet that pulls up from the base of the screen. A common mobile replacement for a dialog.

**Popover** — A small overlay anchored to an element, triggered by click. Can contain interactive content like links or buttons. Different from a tooltip, which cannot.

**Tooltip** — A small label on hover that explains an element. Cannot contain interactive content. If you need a link inside it, use a popover.

**Toast** — A temporary notification that auto-dismisses. Duration should reflect reading time, not a flat 2 seconds regardless of length.

**Badge** — A small label attached to another element. Implies a count when numeric. When it carries a word like New or a status label, it behaves more like a tag and needs different treatment.

**Tag** — A label that categorises content. Tags are standalone, selectable, or removable. Badges are attached and informational. They are not the same thing.

**Accordion** — Stacked sections that expand and collapse. Good for FAQs and secondary content. Not a substitute for tabs when users need to compare across sections.

**Tabs** — Switches between related views in the same space. Best for filtering the same content by time or category. Not for navigating unrelated sections.

**Stepper** — A multi-step flow that shows progress through a sequence. Showing all steps upfront sets expectations. Hiding them creates anxiety.

**Carousel** — A horizontally scrollable row of items. On desktop, often a sign the layout hasn't been solved. On mobile, a natural pattern. Either way, the first item gets most of the attention.

**Navigation menu** — A top-level or flyout structure for moving between major sections. Labels should reflect how users think about the product, not how it was built.

**Sidebar** — Persistent navigation anchored to an edge. Common in dashboards. Works well when users move between sections frequently.

**Breadcrumb** — A trail showing location within a hierarchy. Lets users navigate back at depth without using the browser's back button.

**Pagination** — Controls for moving between pages of content. Use when the full list is too long to load at once. Infinite scroll is an alternative with different tradeoffs.

**Skeleton** — A placeholder that holds the shape of content while it loads. Keeps the layout stable and signals what's coming. Better than a spinner for list and page loads.

**Spinner** — An indeterminate loading indicator. Good for short action-level waits. For page or list loads, a skeleton is almost always better.

**Avatar** — A small image representing a user or entity. Needs a fallback when the image fails, usually initials or a generic icon.

**Card** — A contained surface grouping related content. The inner radius should be smaller than the outer, calculated from the padding. Making the full surface a link makes text selection impossible.

**Data table** — A structured grid of rows and columns. Numbers should be right-aligned with tabular nums. Proper alignment does the work zebra striping is often used to compensate for.

**Combobox** — An input that filters a dropdown as you type. Combines the feel of a text field with the constraints of a select.

**Command menu** — A keyboard-triggered search interface for navigating or executing actions. Common in tools where power users want to avoid the mouse.

**Progress** — A bar showing completion through a determinate process. For unknown durations, use a spinner instead.

**Separator** — A visual divider between items or sections. Often overused when better spacing would do the same job.

**Empty state** — What a view shows when there is no content yet. Should explain why and offer a first action. No items found tells the user nothing.
