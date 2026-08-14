import { OrbitLoader } from "../../../OrbitLoader/OrbitLoader";

interface Props {
  expertName?: string;
}

export function ExpertKickoffLoader({ expertName }: Props) {
  return (
    <div
      role="status"
      aria-live="polite"
      className="flex h-full flex-1 flex-col items-center justify-center gap-4 text-center"
    >
      <OrbitLoader size={32} />
      <div>
        <p className="text-base font-medium text-neutral-900">
          {expertName
            ? `Opening ${expertName}'s workspace`
            : "Opening workspace"}
        </p>
        <p className="mt-1 text-sm text-neutral-500">
          Your expert is getting ready to start.
        </p>
      </div>
    </div>
  );
}
