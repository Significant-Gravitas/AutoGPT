import { withSentryConfig } from "@sentry/nextjs";

/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    domains: [
      "images.unsplash.com",
      "ddz4ak4pa3d19.cloudfront.net",
      "upload.wikimedia.org",
      "storage.googleapis.com",

      "ideogram.ai", // for generated images
      "picsum.photos", // for placeholder images
      "dummyimage.com", // for placeholder images
      "placekitten.com", // for placeholder images
    ],
  },
  output: "standalone",
  transpilePackages: ["geist"],
};

const isDevelopmentBuild = process.env.NODE_ENV !== "production";

export default isDevelopmentBuild
  ? nextConfig
  : withSentryConfig(nextConfig, {
      // For all available options, see:
      // https://github.com/getsentry/sentry-webpack-plugin#options

      org: "significant-gravitas",
      project: "builder",

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
