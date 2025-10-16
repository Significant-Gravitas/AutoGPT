import { Text } from "@/components/atoms/Text/Text";

export const GraphLoadingBox = () => {
  return (
    <div className="absolute left-[50%] top-[50%] z-[99] -translate-x-1/2 -translate-y-1/2">
      <div className="flex flex-col items-center gap-4 rounded-xlarge border border-gray-200 bg-white p-8 shadow-lg dark:border-gray-700 dark:bg-slate-800">
        <div className="relative h-12 w-12">
          <div className="absolute inset-0 animate-spin rounded-full border-4 border-gray-200 border-t-black dark:border-gray-700 dark:border-t-blue-400"></div>
        </div>
        <div className="flex flex-col items-center gap-2">
          <Text variant="h4">Loading Flow</Text>
          <Text variant="small">Please wait while we load your graph...</Text>
        </div>
      </div>
    </div>
  );
};
