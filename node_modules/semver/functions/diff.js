const parse = require('./parse.js')

const diff = (version1, version2) => {
  const v1 = parse(version1, null, true)
  const v2 = parse(version2, null, true)
  const comparison = v1.compare(v2)

  if (comparison === 0) {
    return null
  }

  const v1Higher = comparison > 0
  const highVersion = v1Higher ? v1 : v2
  const lowVersion = v1Higher ? v2 : v1
  const highHasPre = !!highVersion.prerelease.length

  // add the `pre` prefix if we are going to a prerelease version
  const prefix = highHasPre ? 'pre' : ''

  if (v1.major !== v2.major) {
    return prefix + 'major'
  }

  if (v1.minor !== v2.minor) {
    return prefix + 'minor'
  }

  if (v1.patch !== v2.patch) {
    return prefix + 'patch'
  }

  // at this point we know stable versions match but overall versions are not equal,
  // so either they are both prereleases, or the lower version is a prerelease

  if (highHasPre) {
    // high and low are preleases
    return 'prerelease'
  }

  if (lowVersion.patch) {
    // anything higher than a patch bump would result in the wrong version
    return 'patch'
  }

  if (lowVersion.minor) {
    // anything higher than a minor bump would result in the wrong version
    return 'minor'
  }

  // bumping major/minor/patch all have same result
  return 'major'
}

module.exports = diff
