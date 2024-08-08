import dotenv from "dotenv";

// Load environment variables
dotenv.config();

/** @type {import('next').NextConfig} */
const nextConfig = {
  env: {
    NEXT_PUBLIC_AGPT_SERVER_URL: process.env.NEXT_PUBLIC_AGPT_SERVER_URL,
    NEXT_PUBLIC_AGPT_MARKETPLACE_URL:
      process.env.NEXT_PUBLIC_AGPT_MARKETPLACE_URL,
  },
  async redirects() {
    return [
      {
        source: "/monitor", // FIXME: Remove after 2024-09-01
        destination: "/",
        permanent: false,
      },
    ];
  },
};

export default nextConfig;
