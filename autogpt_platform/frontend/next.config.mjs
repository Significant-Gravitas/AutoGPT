import { withSentryConfig } from "@sentry/nextjs";

// Allow Docker builds to skip source-map generation (halves memory usage).
// Defaults to true so Vercel/local builds are unaffected.
const enableSourceMaps = process.env.NEXT_PUBLIC_SOURCEMAPS !== "false";

/** @type {import('next').NextConfig} */
const nextConfig = {
  productionBrowserSourceMaps: enableSourceMaps,
  // Externalize OpenTelemetry packages to fix Turbopack HMR issues
  serverExternalPackages: [
    "@opentelemetry/instrumentation",
    "@opentelemetry/sdk-node",
    "import-in-the-middle",
    "require-in-the-middle",
  ],
  experimental: {
    serverActions: {
      bodySizeLimit: "256mb",
    },
    middlewareClientMaxBodySize: "256mb",
    // Limit parallel webpack workers to reduce peak memory during builds.
    cpus: 2,
  },
  // Work around cssnano "Invalid array length" bug in Next.js's bundled
  // cssnano-simple comment parser when processing very large CSS chunks.
  // CSS is still bundled correctly; gzip handles most of the size savings anyway.
  webpack: (config, { dev }) => {
    if (!dev) {
      // Next.js adds CssMinimizerPlugin internally (after user config), so we
      // can't filter it from config.plugins. Instead, intercept the webpack
      // compilation hooks and replace the buggy plugin's tap with a no-op.
      config.plugins.push({
        apply(compiler) {
          compiler.hooks.compilation.tap(
            "DisableCssMinimizer",
            (compilation) => {
              compilation.hooks.processAssets.intercept({
                register: (tap) => {
                  if (tap.name === "CssMinimizerPlugin") {
                    return { ...tap, fn: async () => {} };
                  }
                  return tap;
                },
              });
            },
          );
        },
      });
    }
    return config;
  },
  images: {
    domains: [
      // We dont need to maintain alphabetical order here
      // as we are doing logical grouping of domains
      "images.unsplash.com",
      "ddz4ak4pa3d19.cloudfront.net",
      "upload.wikimedia.org",
      "storage.googleapis.com",

      "ideogram.ai", // for generated images
      "picsum.photos", // for placeholder images
      "example.com", // for local test data images
    ],
    remotePatterns: [
      {
        protocol: "https",
        hostname: "storage.googleapis.com",
        pathname: "/**",
      },
      {
        protocol: "https",
        hostname: "storage.cloud.google.com",
        pathname: "/**",
      },
      {
        protocol: "https",
        hostname: "lh3.googleusercontent.com",
        pathname: "/**",
      },
    ],
  },
  // Vercel has its own deployment mechanism and doesn't need standalone mode
  ...(process.env.VERCEL ? {} : { output: "standalone" }),
  transpilePackages: ["geist"],
};

// Only run the Sentry webpack plugin when we can actually upload source maps
// (i.e. on Vercel with SENTRY_AUTH_TOKEN set). The Sentry *runtime* SDK
// (imported in app code) still captures errors without the plugin.
// Skipping the plugin saves ~1 GB of peak memory during `next build`.
const skipSentryPlugin =
  process.env.NODE_ENV !== "production" ||
  !enableSourceMaps ||
  !process.env.SENTRY_AUTH_TOKEN;

export default skipSentryPlugin
  ? nextConfig
  : withSentryConfig(nextConfig, {
      // For all available options, see:
      // https://github.com/getsentry/sentry-webpack-plugin#options

      org: "significant-gravitas",
      project: "builder",

      // Expose Vercel env to the client
      env: {
        NEXT_PUBLIC_VERCEL_ENV: process.env.VERCEL_ENV,
      },

      // Only print logs for uploading source maps in CI
      silent: !process.env.CI,

      // For all available options, see:
      // https://docs.sentry.io/platforms/javascript/guides/nextjs/manual-setup/

      // Upload a larger set of source maps for prettier stack traces (increases build time)
      widenClientFileUpload: true,

      // Automatically annotate React components to show their full name in breadcrumbs and session replay
      reactComponentAnnotation: {
        enabled: true,
      },

      // Route browser requests to Sentry through a Next.js rewrite to circumvent ad-blockers.
      // This can increase your server load as well as your hosting bill.
      // Note: Check that the configured route will not match with your Next.js middleware, otherwise reporting of client-
      // side errors will fail.
      tunnelRoute: "/store",

      // No need to hide source maps from generated client bundles
      // since the source is public anyway :)
      hideSourceMaps: false,

      // This helps Sentry with sourcemaps... https://docs.sentry.io/platforms/javascript/guides/nextjs/sourcemaps/
      sourcemaps: {
        disable: !enableSourceMaps,
        assets: [".next/**/*.js", ".next/**/*.js.map"],
        ignore: ["**/node_modules/**"],
        deleteSourcemapsAfterUpload: false, // Source is public anyway :)
      },

      // Automatically tree-shake Sentry logger statements to reduce bundle size
      disableLogger: true,

      // Enables automatic instrumentation of Vercel Cron Monitors. (Does not yet work with App Router route handlers.)
      // See the following for more information:
      // https://docs.sentry.io/product/crons/
      // https://vercel.com/docs/cron-jobs
      automaticVercelMonitors: true,

      async headers() {
        return [
          {
            source: "/:path*",
            headers: [
              {
                key: "Document-Policy",
                value: "js-profiling",
              },
            ],
          },
        ];
      },
    });
