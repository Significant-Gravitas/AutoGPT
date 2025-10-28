import { RunGraph } from "./components/RunGraph";

export const BuilderActions = () => {
  return (
    <div className="absolute bottom-4 left-[50%] z-[100] -translate-x-1/2">
      {/* TODO: Add Agent Output */}
      <RunGraph />
      {/* TODO: Add Schedule run button */}
    </div>
  );
};
