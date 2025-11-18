export const RunningBackground = () => {
  return (
    <div className="absolute inset-0 h-full w-full">
      <style jsx>{`
        @keyframes pulse {
          0%,
          100% {
            opacity: 1;
          }
          50% {
            opacity: 0.5;
          }
        }
        .animate-pulse-border {
          animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }
      `}</style>
      <div
        className="animate-pulse-border absolute inset-0 bg-transparent blur-xl"
        style={{
          borderWidth: "15px",
          borderStyle: "solid",
          borderColor: "transparent",
          borderImage: "linear-gradient(to right, #BC82F3, #BC82F3) 1",
        }}
      ></div>
      <div
        className="animate-pulse-border absolute inset-0 bg-transparent blur-lg"
        style={{
          borderWidth: "10px",
          borderStyle: "solid",
          borderColor: "transparent",
          borderImage: "linear-gradient(to right, #BC82F3, #BC82F3) 1",
        }}
      ></div>
      <div
        className="animate-pulse-border absolute inset-0 bg-transparent blur-md"
        style={{
          borderWidth: "6px",
          borderStyle: "solid",
          borderColor: "transparent",
          borderImage: "linear-gradient(to right, #BC82F3, #BC82F3) 1",
        }}
      ></div>
      <div
        className="animate-pulse-border absolute inset-0 bg-transparent blur-sm"
        style={{
          borderWidth: "6px",
          borderStyle: "solid",
          borderColor: "transparent",
          borderImage: "linear-gradient(to right, #BC82F3, #BC82F3) 1",
        }}
      ></div>
    </div>
  );
};
