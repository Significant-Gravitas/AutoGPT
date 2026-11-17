import { Pressable, View } from "react-native";

import { createStyles, radius } from "@/lib/theme";

interface Props {
  value: boolean;
  onValueChange: (value: boolean) => void;
  accessibilityLabel: string;
  disabled?: boolean;
}

export function Switch({ value, onValueChange, accessibilityLabel, disabled }: Props) {
  const styles = useStyles();

  function handlePress() {
    onValueChange(!value);
  }

  return (
    <Pressable
      accessibilityRole="switch"
      accessibilityLabel={accessibilityLabel}
      accessibilityState={{ checked: value, disabled: !!disabled }}
      disabled={disabled}
      hitSlop={8}
      onPress={handlePress}
      style={[styles.track, value ? styles.on : styles.off, disabled && styles.disabled]}
    >
      <View style={[styles.thumb, value && styles.thumbOn]} />
    </Pressable>
  );
}

const useStyles = createStyles((colors) => ({
  track: {
    width: 44,
    height: 26,
    borderRadius: radius.pill,
    padding: 3,
    justifyContent: "center",
  },
  on: { backgroundColor: colors.action },
  off: { backgroundColor: colors.borderStrong },
  disabled: { opacity: 0.5 },
  thumb: {
    width: 20,
    height: 20,
    borderRadius: 10,
    backgroundColor: colors.background,
    shadowColor: "#000000",
    shadowOpacity: 0.15,
    shadowRadius: 2,
    shadowOffset: { width: 0, height: 1 },
    elevation: 2,
  },
  thumbOn: { transform: [{ translateX: 18 }] },
}));
