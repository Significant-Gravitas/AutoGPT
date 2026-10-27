/**
 * Artwork for DrawnLineLoader. Each entry is ONE continuous stroke (a single
 * path, possibly with a pen lift via `M`) drawn on a 24x24 canvas. The loader
 * draws them one after another, so to change the art just edit or reorder
 * entries. Keep exactly 3: the keyframes in DrawnLineLoader.module.css give
 * each shape a third of the cycle.
 *
 * The paths are original, loosely hand-drawn doodles inspired by the feel of
 * Miro-style sketch shapes. They are not traced from any logo or brand mark.
 * The first entry doubles as the static shape shown under reduced motion.
 */
export interface DrawnLineShape {
  /** Stable name, used as the React key and in tests. */
  name: string;
  /** SVG path data on a 24x24 viewBox. */
  d: string;
}

export const DRAWN_LINE_VIEWBOX = 24;

export const DRAWN_LINE_SHAPES: readonly DrawnLineShape[] = [
  {
    name: "spiral",
    d: "M11.7 11.5 C11.8 11.5 12.2 11.2 12.6 11.2 C12.9 11.2 13.5 11.2 13.7 11.5 C14.0 11.8 14.2 12.3 14.2 12.8 C14.2 13.3 14.1 14.1 13.8 14.5 C13.5 15.0 12.9 15.3 12.3 15.5 C11.7 15.7 10.7 16.0 10.0 15.7 C9.4 15.4 8.7 14.6 8.3 13.8 C7.9 13.0 7.5 11.8 7.7 10.9 C7.9 10.0 8.6 9.0 9.4 8.4 C10.2 7.8 11.5 7.4 12.6 7.5 C13.8 7.5 15.3 7.8 16.2 8.5 C17.0 9.3 17.7 10.6 17.9 11.8 C18.1 13.1 18.1 14.8 17.5 16.0 C16.8 17.2 15.6 18.5 14.3 19.1 C13.0 19.6 11.1 19.8 9.6 19.4 C8.1 19.0 6.4 17.8 5.5 16.6 C4.5 15.3 4.1 13.4 4.1 11.7 C4.2 10.0 4.7 7.9 5.8 6.6 C6.9 5.2 9.0 3.9 10.9 3.6 C12.7 3.2 15.1 3.6 16.8 4.6 C18.6 5.5 20.5 7.2 21.3 9.0 C22.1 10.9 21.5 14.6 21.6 15.7",
  },
  {
    name: "star",
    d: "M12.0 2.2 Q15.4 11.5 18.0 20.5 Q10.5 14.8 2.0 9.6 Q12.0 9.7 21.4 9.4 Q14.0 15.0 6.3 20.7 Q9.6 12.2 12.0 2.4 Q12.4 3.6 12.9 4.2",
  },
  {
    // Crescent, then a pen lift and a small looped dot.
    name: "crescent-and-dot",
    d: "M15.6 3.2 C9.5 3.6 3.6 8.2 3.8 14 C4 19.2 9.8 21.6 15.2 20.4 C17.2 20 18.6 19.2 19.6 18.4 C13.4 18.8 9.6 14.8 10.3 10.4 C10.8 7.6 12.8 5.2 16.2 4.2 M20.2 7.4 C21.6 7.2 22.2 8.8 21.2 9.6 C20.2 10.4 18.8 9.4 19.2 8.2 C19.4 7.6 19.8 7.3 20.4 7.4",
  },
];
