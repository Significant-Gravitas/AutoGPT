import type { ReactNode } from "react";
import { MapEmbed } from "./MapEmbed";
import { mapLanguageSource, parseMapSource } from "./helpers";

interface Props {
  children?: ReactNode;
}

export function MapAwarePre({ children }: Props) {
  const source = mapLanguageSource(children);
  if (source === null) return <>{children}</>;
  const data = parseMapSource(source);
  if (!data) return <>{children}</>;
  return <MapEmbed data={data} />;
}
