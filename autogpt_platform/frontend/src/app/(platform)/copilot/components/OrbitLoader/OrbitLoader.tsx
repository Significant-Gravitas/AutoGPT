import { cn } from "@/lib/utils";
import styles from "./OrbitLoader.module.css";

interface Props {
  size?: number;
  className?: string;
}

export function OrbitLoader({ size = 24, className }: Props) {
  const ballSize = Math.round(size * 0.4);
  const spacing = Math.round(size * 0.6);
  const gap = Math.round(size * 0.2);

  return (
    <div
      className={cn(styles.loader, className)}
      style={
        {
          width: size,
          height: size,
          "--ball-size": `${ballSize}px`,
          "--spacing": `${spacing}px`,
          "--gap": `${gap}px`,
        } as React.CSSProperties
      }
    />
  );
}
