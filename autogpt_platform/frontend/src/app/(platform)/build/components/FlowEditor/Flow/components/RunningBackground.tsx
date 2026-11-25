export const RunningBackground = () => {
  return (
    <div className="absolute inset-0 h-full w-full">
      {/* oxlint-disable design-system/no-arbitrary-values -- the glow's layered border widths have no border-width step */}
      <div className="absolute inset-0 animate-pulse border-[15px] border-solid border-brand/40 bg-transparent blur-xl" />
      <div className="absolute inset-0 animate-pulse border-[10px] border-solid border-brand/40 bg-transparent blur-lg" />
      <div className="absolute inset-0 animate-pulse border-[6px] border-solid border-brand/40 bg-transparent blur-md" />
      <div className="absolute inset-0 animate-pulse border-[6px] border-solid border-brand/40 bg-transparent blur-sm" />
      {/* oxlint-enable design-system/no-arbitrary-values */}
    </div>
  );
};
