// Type definitions for grapheme-splitter

/*~ Note that ES6 modules cannot directly export class objects.
 *~ This file should be imported using the CommonJS-style:
 *~
 *~   import GraphemeSplitter = require('grapheme-splitter')
 *~
 *~ Refer to the documentation to understand common
 *~ workarounds for this limitation of ES6 modules.
 *~
 *~   https://www.typescriptlang.org/docs/handbook/declaration-files/introduction.html
 */

declare class GraphemeSplitter {
  constructor();
  /** iterate the string to an iterable iterator of grapheme clusters */
  iterateGraphemes(s: string): IterableIterator<string>;
  /** split the string to an array of grapheme clusters */
  splitGraphemes(s: string): string[];
  /** count the number of grapheme clusters in a string */
  countGraphemes(s: string): number;
}

export = GraphemeSplitter;
