import { useState } from "react";
import { ActivityIndicator, View } from "react-native";
import { WebView } from "react-native-webview";

import { createStyles, useColors } from "@/lib/theme";

interface Props {
  uri: string;
  headers: Record<string, string>;
  onError: () => void;
}

/** iOS's web view draws a PDF page by page, with its own page count. */
export function PdfView({ uri, headers, onError }: Props) {
  const styles = useStyles();
  const colors = useColors();
  const [isLoading, setIsLoading] = useState(true);

  function handleLoadEnd() {
    setIsLoading(false);
  }

  return (
    <View style={styles.frame}>
      <WebView
        source={{ uri, headers }}
        originWhitelist={["*"]}
        // A PDF needs no scripts, and a file from the workspace gets none.
        javaScriptEnabled={false}
        onLoadEnd={handleLoadEnd}
        onError={onError}
        onHttpError={onError}
        style={styles.web}
      />
      {isLoading ? (
        <View style={styles.loading}>
          <ActivityIndicator color={colors.textTertiary} />
        </View>
      ) : null}
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  frame: { flex: 1, backgroundColor: colors.surfaceRaised },
  web: { flex: 1, backgroundColor: "transparent" },
  loading: { position: "absolute", top: 0, left: 0, right: 0, bottom: 0, alignItems: "center", justifyContent: "center" },
}));
