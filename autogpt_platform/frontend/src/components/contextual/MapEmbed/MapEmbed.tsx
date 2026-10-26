import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Text } from "@/components/atoms/Text/Text";
import { googleMapsPinUrl, type MapData } from "./helpers";
import { useMapEmbed } from "./useMapEmbed";

interface Props {
  data: MapData;
}

function PinLinks({ data }: Props) {
  return (
    <ul className="m-0 list-none space-y-1 p-3">
      {data.pins.map((pin) => (
        <li key={`${pin.lat},${pin.lng},${pin.name}`}>
          <a
            href={googleMapsPinUrl(pin)}
            target="_blank"
            rel="noreferrer"
            // TODO(tokens): no link colour role yet; matches chat links
            className="text-sm text-blue-500 hover:underline"
          >
            {pin.name}
          </a>
          {pin.address ? (
            <Text variant="small" className="text-foreground-tertiary">
              {pin.address}
            </Text>
          ) : null}
        </li>
      ))}
    </ul>
  );
}

export function MapEmbed({ data }: Props) {
  const { containerRef, status } = useMapEmbed(data.pins);

  return (
    <figure
      className="my-3 overflow-hidden rounded-xl border border-border-base"
      data-testid="map-embed"
    >
      {data.title ? (
        <figcaption className="border-b border-separator px-3 py-2 text-sm font-medium text-foreground-emphasis">
          {data.title}
        </figcaption>
      ) : null}
      {status === "failed" ? (
        <PinLinks data={data} />
      ) : (
        <div className="relative h-72 w-full bg-surface-muted">
          <div
            ref={containerRef}
            className="h-full w-full"
            aria-label={data.title ?? "Map"}
          />
          {status === "loading" ? (
            <div className="absolute inset-0 flex items-center justify-center text-foreground-tertiary">
              <LoadingSpinner />
            </div>
          ) : null}
        </div>
      )}
    </figure>
  );
}
