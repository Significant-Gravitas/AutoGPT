const dotenv = require('dotenv');
dotenv.config();

const nextConfig = {
  output: 'export',

  // Optional: Change links `/me` -> `/me/` and emit `/me.html` -> `/me/index.html`
  // trailingSlash: true,

  // Optional: Prevent automatic `/me` -> `/me/`, instead preserve `href`
  // skipTrailingSlashRedirect: true,

  // Optional: Change the output directory `out` -> `dist`
  // distDir: 'dist',
}

module.exports = {
  env: {
    AGPT_SERVER_URL: process.env.AGPT_SERVER_URL,
  },
  nextConfig,
};
