import { Button } from "@/components/atoms/Button/Button";
import type { MapLocation } from "@/lib/openui/catalog-sections";
import { useOpenUIDisabled } from "../../interactionContext";

interface Props {
  place: MapLocation;
  onDiscuss: () => void;
}

export function SelectedPlace({ place, onDiscuss }: Props) {
  const disabled = useOpenUIDisabled();
  return (
    <div
      className="space-y-3 rounded-lg border border-purple-100 bg-purple-50/50 p-3"
      aria-label="Selected place"
    >
      <h4 className="break-words text-sm font-semibold text-zinc-900">
        {place.name}
      </h4>
      <p className="break-words text-sm leading-relaxed text-zinc-600">
        {place.detail}
      </p>
      <p className="text-xs text-zinc-500">
        {place.latitude.toFixed(5)}, {place.longitude.toFixed(5)}
      </p>
      <Button
        size="small"
        variant="secondary"
        disabled={disabled}
        onClick={onDiscuss}
      >
        Discuss this place
      </Button>
    </div>
  );
}
