// The motion tokens of src/app/tokens/motion.css, for Framer Motion: each
// duration in seconds and each curve as its four cubic-bezier numbers. A test
// keeps them equal to the CSS.
export const duration = {
  instant: 0.1,
  fast: 0.15,
  base: 0.2,
  slow: 0.3,
  slower: 0.4,
  slowest: 0.5,
} as const;

export const ease = {
  standard: [0.4, 0, 0.2, 1],
  out: [0, 0, 0.2, 1],
  in: [0.4, 0, 1, 1],
  emphasized: [0.16, 1, 0.3, 1],
  outQuint: [0.23, 1, 0.32, 1],
  glide: [0.22, 1, 0.36, 1],
  drawer: [0.32, 0.72, 0, 1],
} as const;

export const spring = {
  snappy: { stiffness: 520, damping: 30 },
  gentle: { stiffness: 300, damping: 25 },
  bouncy: { stiffness: 200, damping: 15 },
} as const;
