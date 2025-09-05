import dynamic from "next/dynamic";

// Lazy load the Flow component which includes the heavy @xyflow/react library
const FlowEditor = dynamic(() => import("./Flow"), {
  loading: () => (
    <div className="flex h-full items-center justify-center">
      <div className="flex flex-col items-center gap-4">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent"></div>
        <p className="text-sm text-muted-foreground">
          Loading workflow editor...
        </p>
      </div>
    </div>
  ),
  ssr: false, // Disable SSR for this component
});

export default FlowEditor;
