// Sentry's wrapper around Expo's default config: it gives each bundle a debug
// ID, so stack traces from store builds and over-the-air updates map back to
// the source.
const { getSentryExpoConfig } = require("@sentry/react-native/metro");

module.exports = getSentryExpoConfig(__dirname);
