// Type definitions for semver 7.5
// Project: https://github.com/npm/node-semver
// Definitions by: Bart van der Schoor <https://github.com/Bartvds>
//                 BendingBender <https://github.com/BendingBender>
//                 Lucian Buzzo <https://github.com/LucianBuzzo>
//                 Klaus Meinhardt <https://github.com/ajafff>
//                 ExE Boss <https://github.com/ExE-Boss>
//                 Piotr Błażejewicz <https://github.com/peterblazejewicz>
// Definitions: https://github.com/DefinitelyTyped/DefinitelyTyped

// re-exports for index file

// functions for working with versions
import semverParse = require('./functions/parse');
import semverValid = require('./functions/valid');
import semverClean = require('./functions/clean');
import semverInc = require('./functions/inc');
import semverDiff = require('./functions/diff');
import semverMajor = require('./functions/major');
import semverMinor = require('./functions/minor');
import semverPatch = require('./functions/patch');
import semverPrerelease = require('./functions/prerelease');
import semverCompare = require('./functions/compare');
import semverRcompare = require('./functions/rcompare');
import semverCompareLoose = require('./functions/compare-loose');
import semverCompareBuild = require('./functions/compare-build');
import semverSort = require('./functions/sort');
import semverRsort = require('./functions/rsort');

export {
    semverParse as parse,
    semverValid as valid,
    semverClean as clean,
    semverInc as inc,
    semverDiff as diff,
    semverMajor as major,
    semverMinor as minor,
    semverPatch as patch,
    semverPrerelease as prerelease,
    semverCompare as compare,
    semverRcompare as rcompare,
    semverCompareLoose as compareLoose,
    semverCompareBuild as compareBuild,
    semverSort as sort,
    semverRsort as rsort,
};

// low-level comparators between versions
import semverGt = require('./functions/gt');
import semverLt = require('./functions/lt');
import semverEq = require('./functions/eq');
import semverNeq = require('./functions/neq');
import semverGte = require('./functions/gte');
import semverLte = require('./functions/lte');
import semverCmp = require('./functions/cmp');
import semverCoerce = require('./functions/coerce');

export {
    semverGt as gt,
    semverLt as lt,
    semverEq as eq,
    semverNeq as neq,
    semverGte as gte,
    semverLte as lte,
    semverCmp as cmp,
    semverCoerce as coerce,
};

// working with ranges
import semverSatisfies = require('./functions/satisfies');
import semverMaxSatisfying = require('./ranges/max-satisfying');
import semverMinSatisfying = require('./ranges/min-satisfying');
import semverToComparators = require('./ranges/to-comparators');
import semverMinVersion = require('./ranges/min-version');
import semverValidRange = require('./ranges/valid');
import semverOutside = require('./ranges/outside');
import semverGtr = require('./ranges/gtr');
import semverLtr = require('./ranges/ltr');
import semverIntersects = require('./ranges/intersects');
import simplify = require('./ranges/simplify');
import rangeSubset = require('./ranges/subset');

export {
    semverSatisfies as satisfies,
    semverMaxSatisfying as maxSatisfying,
    semverMinSatisfying as minSatisfying,
    semverToComparators as toComparators,
    semverMinVersion as minVersion,
    semverValidRange as validRange,
    semverOutside as outside,
    semverGtr as gtr,
    semverLtr as ltr,
    semverIntersects as intersects,
    simplify as simplifyRange,
    rangeSubset as subset,
};

// classes
import SemVer = require('./classes/semver');
import Range = require('./classes/range');
import Comparator = require('./classes/comparator');

export { SemVer, Range, Comparator };

// internals
import identifiers = require('./internals/identifiers');
export import compareIdentifiers = identifiers.compareIdentifiers;
export import rcompareIdentifiers = identifiers.rcompareIdentifiers;

export const SEMVER_SPEC_VERSION: '2.0.0';

export type ReleaseType = 'major' | 'premajor' | 'minor' | 'preminor' | 'patch' | 'prepatch' | 'prerelease';

export interface Options {
    loose?: boolean | undefined;
}

export interface RangeOptions extends Options {
    includePrerelease?: boolean | undefined;
}
export interface CoerceOptions extends Options {
    /**
     * Used by `coerce()` to coerce from right to left.
     *
     * @default false
     *
     * @example
     * coerce('1.2.3.4', { rtl: true });
     * // => SemVer { version: '2.3.4', ... }
     *
     * @since 6.2.0
     */
    rtl?: boolean | undefined;
}

export type Operator = '===' | '!==' | '' | '=' | '==' | '!=' | '>' | '>=' | '<' | '<=';
