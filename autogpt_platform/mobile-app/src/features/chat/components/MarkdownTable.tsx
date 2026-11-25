import type { ReactNode } from "react";
import { StyleSheet, useWindowDimensions, View, type ViewStyle } from "react-native";
import { ScrollView } from "react-native-gesture-handler";

interface Props {
  header: ReactNode[][];
  rows: ReactNode[][][];
  tableStyle?: ViewStyle;
  rowStyle?: ViewStyle;
  cellStyle?: ViewStyle;
}

/**
 * Gesture handler's scroll view, not React Native's, so a sideways drag on the
 * table scrolls it instead of going to the message's long press or the drawers.
 */
export function MarkdownTable({ header, rows, tableStyle, rowStyle, cellStyle }: Props) {
  const { width } = useWindowDimensions();
  // As react-native-marked sizes them: a column is just under half the screen.
  const columnWidth = Math.floor(width * 0.43);
  const { borderColor, borderWidth = StyleSheet.hairlineWidth } = tableStyle ?? {};
  // Each cell draws its top and right edge, the table its left and bottom.
  const tableBorder = { borderColor, borderLeftWidth: borderWidth, borderBottomWidth: borderWidth };
  const cellBorder = { borderColor, borderTopWidth: borderWidth, borderRightWidth: borderWidth };

  function renderRow(cells: ReactNode[], key: string) {
    return (
      <View key={key} style={[styles.row, rowStyle]}>
        {cells.map((cell, index) => (
          <View key={index} style={[{ width: columnWidth }, cellBorder, cellStyle]}>
            {cell}
          </View>
        ))}
      </View>
    );
  }

  return (
    <ScrollView horizontal nestedScrollEnabled showsHorizontalScrollIndicator={false}>
      <View style={tableBorder}>
        {renderRow(header, "header")}
        {rows.map((row, index) => renderRow(row, `row-${index}`))}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  row: { flexDirection: "row" },
});
