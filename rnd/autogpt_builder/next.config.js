const dotenv = require('dotenv');
dotenv.config();

module.exports = {
  env: {
    AGPT_SERVER_URL: process.env.AGPT_SERVER_URL,
  },
};
