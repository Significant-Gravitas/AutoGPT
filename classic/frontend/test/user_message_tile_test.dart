import 'package:auto_gpt_flutter_client/views/chat/user_message_tile.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  // Test group for UserMessageTile widget
  group('UserMessageTile', () {
    // Test to check if the widget renders without error
    testWidgets('renders without error', (WidgetTester tester) async {
      await tester.pumpWidget(const MaterialApp(
        home: Scaffold(
          body: UserMessageTile(message: 'Hello, User!'),
        ),
      ));
      expect(find.byType(UserMessageTile), findsOneWidget);
    });

    // Test to check if the widget displays the correct user message
    testWidgets('displays the correct user message',
        (WidgetTester tester) async {
      const testMessage = 'Test Message';
      await tester.pumpWidget(const MaterialApp(
        home: Scaffold(
          body: UserMessageTile(message: testMessage),
        ),
      ));

      expect(find.text(testMessage), findsOneWidget);
    });

    // Test to check if the widget displays the "User" title
    testWidgets('displays the "User" title', (WidgetTester tester) async {
      await tester.pumpWidget(const MaterialApp(
        home: Scaffold(
          body: UserMessageTile(message: 'Any Message'),
        ),
      ));

      expect(find.text('User'), findsOneWidget);
    });
  });
}
