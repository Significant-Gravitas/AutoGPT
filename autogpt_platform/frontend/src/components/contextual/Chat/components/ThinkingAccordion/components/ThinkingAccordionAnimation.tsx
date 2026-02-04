export function ThinkingAccordionAnimation() {
  return (
    <span
      className="relative inline-flex shrink-0 items-center justify-center"
      style={{ width: "1.25rem", height: "1.25rem" }}
    >
      <span
        className="absolute rounded-full"
        style={{
          width: "100%",
          height: "100%",
          background: `linear-gradient(
            165deg,
            rgba(80, 80, 80, 1) 0%,
            rgb(60, 60, 60) 40%,
            rgb(40, 40, 40) 98%,
            rgb(10, 10, 10) 100%
          )`,
        }}
      />
      <span
        className="absolute rounded-full"
        style={{
          width: "100%",
          height: "100%",
          borderRadius: "50%",
          borderBottom: "0 solid transparent",
          boxShadow: `
            0 -3px 6px 6px rgba(120, 120, 120, 0.25) inset,
            0 -2px 4px 3px rgba(140, 140, 140, 0.3) inset,
            0 -1px 2px rgba(160, 160, 160, 0.5) inset,
            0 -1px 1px rgba(180, 180, 180, 0.7) inset,
            0 1px 0px rgba(200, 200, 200, 0.6),
            0 1px 2px rgba(180, 180, 180, 0.5),
            0 2px 3px rgba(160, 160, 160, 0.4),
            0 3px 5px rgba(140, 140, 140, 0.3),
            0 4px 8px rgba(120, 120, 120, 0.2)
          `,
          filter: "blur(0.5px)",
          animation: "spinLoader 2s linear infinite",
        }}
      />
      <style jsx>{`
        @keyframes spinLoader {
          100% {
            transform: rotate(360deg);
          }
        }
      `}</style>
    </span>
  );
}
