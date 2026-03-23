import { withSentryConfig } from "@sentry/nextjs";

/**
 * Security headers applied to every response from the Next.js server.
 * These complement the backend's SecurityHeadersMiddleware.
 */
const securityHeaders = [
  // Enforce HTTPS for 1 year
  {
    key: "Strict-Transport-Security",
    value: "max-age=31536000; includeSubDomains",
  },
  // Prevent MIME-type sniffing
  {
    key: "X-Content-Type-Options",
    value: "nosniff",
  },
  // Disallow embedding in iframes (clickjacking protection)
  {
    key: "X-Frame-Options",
    value: "DENY",
  },
  // Control what data is sent in the Referer header
  {
    key: "Referrer-Policy",
    value: "strict-origin-when-cross-origin",
  },
  // Restrict access to powerful browser features
  {
    key: "Permissions-Policy",
    value: "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
  },
  // Content-Security-Policy: allow same-origin scripts/styles, trusted CDNs,
  // Sentry, PostHog analytics, and Supabase auth.
  // Adjust the connect-src list when new external services are added.
  {
    key: "Content-Security-Policy",
    value: [
      "default-src 'self'",
      // Scripts: self + inline eval needed by Next.js HMR and some third-party libs
      "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://*.sentry.io https://*.posthog.com https://*.launchdarkly.com",
      // Styles: self + inline styles used by Tailwind/shadcn
      "style-src 'self' 'unsafe-inline'",
      // Images: self + remote patterns already whitelisted in next.config images
      "img-src 'self' data: blob: https:",
      // Fonts: self + data URIs
      "font-src 'self' data:",
      // API connections: self + Supabase, Sentry, PostHog, LaunchDarkly, backend
      "connect-src 'self' https://*.supabase.co wss://*.supabase.co https://*.sentry.io https://*.posthog.com https://*.launchdarkly.com https://*.agpt.co",
      // Frames: deny embedding of this app and block all iframes
      "frame-src 'none'",
      "frame-ancestors 'none'",
      // Workers: self only
      "worker-src 'self' blob:",
    ].join("; "),
  },
];

/** @type {import('next').NextConfig} */
const nextConfig = {
  productionBrowserSourceMaps: true,
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
    // Increase body size limit for API routes (file uploads) - 256MB to match backend limit
    proxyClientMaxBodySize: "256mb",
    middlewareClientMaxBodySize: "256mb",
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
  async headers() {
    return [
      {
        // Apply security headers to all routes
        source: "/:path*",
        headers: securityHeaders,
      },
    ];
  },
};

const isDevelopmentBuild = process.env.NODE_ENV !== "production";

export default isDevelopmentBuild
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
        disable: false,
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
