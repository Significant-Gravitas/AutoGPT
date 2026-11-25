import { ActivityIndicator, View } from "react-native";
import Animated, { useReducedMotion } from "react-native-reanimated";

import { createStyles, radius, useColors } from "@/lib/theme";

// Fixed shapes scaled by the live level, so the bars move with the voice.
const BAR_WEIGHTS = [0.55, 0.9, 1, 0.7, 0.85];
const RING_WIDTH = 2;

interface Props {
  /** How loud the voice is, 0 to 1. */
  level: number;
  size?: number;
  /** A spinner stands in for the bars and the rings stop. */
  isBusy?: boolean;
}

/** The recording mark: a disc with level bars inside and rings pulsing out. */
export function VoiceOrb({ level, size = 40, isBusy }: Props) {
  const styles = useStyles();
  const colors = useColors();
  const reduceMotion = useReducedMotion();
  const halo = size / 8;
  const around = { top: -halo, right: -halo, bottom: -halo, left: -halo };
  const barWidth = Math.max(2, Math.round(size * 0.075));
  const barMin = size / 10;
  const barRange = size * 0.4;

  return (
    <View
      accessibilityElementsHidden
      importantForAccessibility="no-hide-descendants"
      style={[styles.orb, { width: size, height: size }]}
    >
      <View style={[styles.layer, styles.halo, around]} />
      {isBusy || reduceMotion ? null : (
        // A quiet room fades the rings; they never stop, so the orb reads as on.
        <View style={[styles.layer, around, { opacity: 0.2 + level * 0.8 }]}>
          <Animated.View style={[styles.layer, styles.ring, ring]} />
          <Animated.View style={[styles.layer, styles.ring, lateRing]} />
        </View>
      )}
      <View style={[styles.layer, styles.disc]} />
      {isBusy ? (
        <ActivityIndicator size="small" color={colors.onBrand} />
      ) : (
        <View style={[styles.bars, { gap: size / 16, height: barMin + barRange }]}>
          {BAR_WEIGHTS.map((weight, index) => (
            <View
              key={index}
              style={[
                styles.bar,
                { width: barWidth, height: barMin + barRange * level * weight },
              ]}
            />
          ))}
        </View>
      )}
    </View>
  );
}

const ring = {
  animationName: {
    from: { opacity: 0.55, transform: [{ scale: 1 }] },
    to: { opacity: 0, transform: [{ scale: 1.75 }] },
  },
  animationDuration: "1800ms",
  animationIterationCount: "infinite",
  animationTimingFunction: "ease-out",
} as const;

const lateRing = { ...ring, animationDelay: "900ms" } as const;

const useStyles = createStyles((colors) => ({
  orb: { alignItems: "center", justifyContent: "center" },
  layer: {
    position: "absolute",
    top: 0,
    right: 0,
    bottom: 0,
    left: 0,
    borderRadius: radius.pill,
  },
  halo: { backgroundColor: colors.brandSoft },
  // Invisible until its animation starts, so the late ring does not flash.
  ring: { borderWidth: RING_WIDTH, borderColor: colors.brand, opacity: 0 },
  disc: { backgroundColor: colors.brand },
  bars: { flexDirection: "row", alignItems: "center" },
  bar: { borderRadius: radius.pill, backgroundColor: colors.onBrand },
}));
