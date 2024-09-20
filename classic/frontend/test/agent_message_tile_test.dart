import 'package:auto_gpt_flutter_client/views/chat/agent_message_tile.dart';
import 'package:auto_gpt_flutter_client/views/chat/json_code_snippet_view.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  // Test to verify that the AgentMessageTile renders correctly
  testWidgets('Renders AgentMessageTile', (WidgetTester tester) async {
    await tester.pumpWidget(const MaterialApp(
      home: Scaffold(
        body: AgentMessageTile(message: 'Test Message'),
      ),
    ));

    // Verify that the agent title is displayed
    expect(find.text('Agent'), findsOneWidget);
    // Verify that the message text is displayed
    expect(find.text('Test Message'), findsOneWidget);
  });

  // Test to verify that the expand/collapse functionality works
  testWidgets('Toggle Expand/Collapse', (WidgetTester tester) async {
    await tester.pumpWidget(const MaterialApp(
      home: Scaffold(
        body: AgentMessageTile(message: 'Test Message'),
      ),
    ));

    // Verify that the JSON code snippet is not visible initially
    expect(find.byType(JsonCodeSnippetView), findsNothing);

    // Tap the expand/collapse button
    await tester.tap(find.byIcon(Icons.keyboard_arrow_down));
    await tester.pumpAndSettle();

    // Verify that the JSON code snippet is now visible
    expect(find.byType(JsonCodeSnippetView), findsOneWidget);

    // Tap the expand/collapse button again
    await tester.tap(find.byIcon(Icons.keyboard_arrow_up));
    await tester.pumpAndSettle();

    // Verify that the JSON code snippet is hidden again
    expect(find.byType(JsonCodeSnippetView), findsNothing);
  });
}
