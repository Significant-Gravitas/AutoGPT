import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { View } from "react-native";
import Animated, { FadeIn, FadeOut } from "react-native-reanimated";

import { IconButton } from "@/components/IconButton";
import { createStyles, spacing } from "@/lib/theme";

const BUTTON = 40;

interface Props {
  onPress: () => void;
}

/** Floats above the composer once the end of the thread is out of view. */
export function ScrollToEndButton({ onPress }: Props) {
  const styles = useStyles();
  return (
    <View pointerEvents="box-none" style={styles.wrap}>
      <Animated.View entering={FadeIn.duration(150)} exiting={FadeOut.duration(150)}>
        <IconButton
          icon={ArrowDown01Icon}
          label="Scroll to latest"
          onPress={onPress}
          variant="icon"
        />
      </Animated.View>
    </View>
  );
}

const useStyles = createStyles(() => ({
  wrap: {
    position: "absolute",
    top: -(BUTTON + spacing.sm),
    left: 0,
    right: 0,
    alignItems: "center",
  },
}));
