import { Image, type ImageSource } from "expo-image";
import { useState } from "react";
import { type LayoutChangeEvent, ScrollView, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { createStyles } from "@/lib/theme";

const MAX_ZOOM = 4;

interface Props {
  source: ImageSource;
  label: string;
  onError: () => void;
}

/** The picture on black, as the Photos app shows it. A pinch zooms it on iOS. */
export function ImageView({ source, label, onError }: Props) {
  const styles = useStyles();
  const insets = useSafeAreaInsets();
  const [size, setSize] = useState<{ width: number; height: number } | null>(null);

  function handleLayout(event: LayoutChangeEvent) {
    const { width, height } = event.nativeEvent.layout;
    setSize({ width, height });
  }

  return (
    <View style={[styles.frame, { paddingBottom: insets.bottom }]} onLayout={handleLayout}>
      {size ? (
        <ScrollView
          maximumZoomScale={MAX_ZOOM}
          minimumZoomScale={1}
          centerContent
          showsHorizontalScrollIndicator={false}
          showsVerticalScrollIndicator={false}
        >
          <Image
            source={source}
            contentFit="contain"
            accessibilityLabel={label}
            onError={onError}
            style={{ width: size.width, height: size.height - insets.bottom }}
          />
        </ScrollView>
      ) : null}
    </View>
  );
}

const useStyles = createStyles(() => ({
  frame: { flex: 1, backgroundColor: "#000000" },
}));
