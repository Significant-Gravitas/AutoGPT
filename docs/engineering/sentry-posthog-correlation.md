# Sentry and PostHog error correlation

The browser's Sentry `beforeSend` forwards surviving error/fatal events through
PostHog's Sentry integration. PostHog records `$exception` with `$sentry_event_id`
and a `$sentry_url` pointing to that exact event in Sentry. Sentry receives
`PostHog Person URL` and, while PostHog recording is active, a timestamped
`PostHog Recording URL` tag. Anonymous visitors use their current PostHog ID;
signed-in visitors use the identified user ID.

Both SDKs deliver independently. Mirroring runs before Sentry's transport
acknowledges the event. Sentry rate limits, quota rejection, or network failure
can leave a PostHog exception with a Sentry link that has no stored event.
PostHog delivery can also fail independently. Event counts and billed usage
can therefore differ; a correlation ID does not confirm storage in both products.

Forwarding and links require analytics consent and an initialized, opted-in
PostHog client. Consent is checked for each error. Early startup errors and
errors without consent still go to Sentry, but are not queued for later mirroring.
Sentry's existing filters run first. Failed HTTP responses the HttpClient
integration reports (mechanism `auto.http.client.*`) stay in Sentry only: they
are not page bugs, and a failing PostHog relay would otherwise be mirrored
back into PostHog. Warnings, transactions, backend errors, and
historical events are outside this bridge. PostHog failures do not stop Sentry
reporting. PostHog exception autocapture stays disabled to avoid a second capture
of the same browser error.

Disabling Sentry with `DISABLE_SENTRY=true` also stops PostHog exception mirroring.
The same applies when no Sentry DSN is configured or available through the hosted
fallback. PostHog product analytics continues independently.

The Sentry link uses the configured DSN's project in AutoGPT's Sentry organization,
including preview projects. Other organizations and self-hosted DSNs still get
PostHog tags, but no guessed Sentry URL.

## What PostHog receives

`beforeSend` first strips the event the way every Sentry event is stripped
(the user reduced to id and role, IP-like span attributes dropped), runs the
filters, and only then mirrors it. PostHog's integration copies just these
fields of the Sentry event into `$exception`:

- `$exception_list` and `$sentry_exception`: the exception type, message,
  mechanism and stack frames (file URLs, functions, lines and columns).
- `$exception_message`, `$exception_type`, `$sentry_exception_message`,
  `$sentry_exception_type` and `$exception_level`.
- `$sentry_event_id`, `$sentry_url` and `$sentry_tags` (the Sentry tags,
  including the PostHog person and recording links).

It never copies the Sentry user, request (URL, headers), contexts, extra data
or breadcrumbs; `sentry-posthog-instrumentation.test.ts` checks that no email,
IP address, IP header or breadcrumb URL reaches PostHog. PostHog then adds the
standard properties it puts on every consented event (current URL, browser,
device, session and person).

## Deploy

The bridge uses the existing public PostHog key/host, Sentry DSN, and Cookiebot
configuration. Deploying the frontend enables it for future eligible events.
The Sentry warehouse connector is independent: it imports tables, and does not
install this browser bridge or synchronize issue resolution status.

Source-map upload additionally requires these **build-time** variables:

- `POSTHOG_SOURCEMAP_UPLOAD_KEY`: a private PostHog personal API key with source-map
  write access. Not the backend's `POSTHOG_API_KEY`, which is the public project key.
- `POSTHOG_SOURCEMAP_PROJECT_ID`: the destination project ID (`117338` for production).
- `NEXT_PUBLIC_POSTHOG_HOST`: the matching PostHog Cloud region.
- `NEXT_PUBLIC_SOURCEMAPS=true`: explicit opt-in for Docker builds; Vercel builds
  already enable source maps unless explicitly disabled.

Set the private key in the deployment platform's secret store, never in a public
environment variable or committed file. On Vercel, add both variables to the
Production environment only; Next.js inlines only `NEXT_PUBLIC_` variables and
`env` from `next.config.mjs` into the browser, so neither reaches the bundle.
Without them, builds still pass and PostHog shows minified stack traces. For Docker, pass it using BuildKit
`--secret id=POSTHOG_SOURCEMAP_UPLOAD_KEY,env=POSTHOG_SOURCEMAP_UPLOAD_KEY`, plus
`--build-arg POSTHOG_SOURCEMAP_PROJECT_ID=117338 --build-arg NEXT_PUBLIC_SOURCEMAPS=true`.
The Dockerfile mounts the key only for the build command. Existing Sentry upload
credentials remain necessary for Sentry's own source maps. Both plugins retain
maps, matching the existing public-source policy and avoiding deletion races.
Without PostHog upload credentials, builds continue with the existing config.

## Verify in a preview

1. Accept analytics consent, sign in, and trigger one controlled browser error
   through Sentry in the preview's project.
2. Find the `$exception` in PostHog. Follow `$sentry_url` and verify the Sentry event
   ID equals `$sentry_event_id`. Verify the stack resolves to the original source.
3. Follow the Sentry `PostHog Person URL` tag. If recording is active, follow its
   recording tag and verify the session and timestamp.
4. Repeat anonymously, then revoke analytics consent and repeat. The anonymous
   event should link to that anonymous person; the revoked event should only
   reach Sentry, without PostHog tags.
5. Verify a warning and an error matching an existing Sentry filter do not create
   PostHog exceptions. One accepted error should create at most one exception.

Mirroring adds a PostHog exception for each eligible Sentry event, subject to SDK
delivery limits. It does not reduce Sentry usage. Warehouse sync has its separate
row-based usage. The bridge reuses the existing PostHog SDK and asynchronous
transport; its synchronous work is consent checking and event/property mapping.
