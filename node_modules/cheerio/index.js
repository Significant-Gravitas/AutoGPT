/**
 * Export cheerio (with )
 */

exports = module.exports = require('./lib/cheerio');

/*
  Export the version
*/

exports.version = require('./package.json').version;
