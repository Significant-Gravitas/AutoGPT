const dotenv = require('dotenv');
dotenv.config();

const nextConfig = {
  output: 'export',
  env: {
    AGPT_SERVER_URL: process.env.AGPT_SERVER_URL,
  },
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
};

module.exports = nextConfig
