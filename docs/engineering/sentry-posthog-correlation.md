# Sentry and PostHog error correlation

The browser's Sentry `beforeSend` forwards surviving error/fatal events through
PostHog's Sentry integration. PostHog records `$exception` with `$sentry_event_id`
and a `$sentry_url` pointing to that exact event in Sentry. Sentry receives
`PostHog Person URL` and, while PostHog recording is active, a timestamped
`PostHog Recording URL` tag. Anonymous visitors use their current PostHog ID;
signed-in visitors use the identified user ID.

Forwarding and links require analytics consent and an initialized, opted-in
PostHog client. Consent is checked for each error. Early startup errors and
errors without consent still go to Sentry, but are not queued for later mirroring.
Sentry's existing filters run first. Warnings, transactions, backend errors, and
historical events are outside this bridge. PostHog failures do not stop Sentry
reporting. PostHog exception autocapture stays disabled to avoid a second capture
of the same browser error.

The Sentry link uses the configured DSN's project in AutoGPT's Sentry organization,
including preview projects. Other organizations and self-hosted DSNs still get
PostHog tags, but no guessed Sentry URL.

## Deploy

The bridge uses the existing public PostHog key/host, Sentry DSN, and Cookiebot
configuration. Deploying the frontend enables it for future eligible events.
The Sentry warehouse connector is independent: it imports tables, and does not
install this browser bridge or synchronize issue resolution status.

Source-map upload additionally requires these **build-time** variables:

- `POSTHOG_API_KEY`: a private PostHog personal API key with source-map write access.
- `POSTHOG_PROJECT_ID`: the destination project ID (`117338` for production).
- `NEXT_PUBLIC_POSTHOG_HOST`: the matching PostHog Cloud region.
- `NEXT_PUBLIC_SOURCEMAPS=true`: explicit opt-in for Docker builds; Vercel builds
  already enable source maps unless explicitly disabled.

Set the private key in the deployment platform's secret store, never in a public
environment variable or committed file. For Docker, pass it using BuildKit
`--secret id=POSTHOG_API_KEY,env=POSTHOG_API_KEY`, plus
`--build-arg POSTHOG_PROJECT_ID=117338 --build-arg NEXT_PUBLIC_SOURCEMAPS=true`.
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
