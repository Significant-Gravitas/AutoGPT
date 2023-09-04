import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:auto_gpt_flutter_client/views/chat/chat_input_field.dart';

void main() {
  // Test if the ChatInputField widget renders correctly
  testWidgets('ChatInputField renders correctly', (WidgetTester tester) async {
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: ChatInputField(
            onSendPressed: () {},
          ),
        ),
      ),
    );

    // Find the TextField widget
    expect(find.byType(TextField), findsOneWidget);
    // Find the send IconButton
    expect(find.byIcon(Icons.send), findsOneWidget);
  });

  // Test if the TextField inside ChatInputField can accept and display input
  testWidgets('ChatInputField text field accepts input',
      (WidgetTester tester) async {
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: ChatInputField(
            onSendPressed: () {},
          ),
        ),
      ),
    );

    // Type 'Hello' into the TextField
    await tester.enterText(find.byType(TextField), 'Hello');
    // Rebuild the widget with the new text
    await tester.pump();

    // Expect to find 'Hello' in the TextField
    expect(find.text('Hello'), findsOneWidget);
  });

  // Test if the send button triggers the provided onSendPressed callback
  testWidgets('ChatInputField send button triggers callback',
      (WidgetTester tester) async {
    bool onPressedCalled = false;

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: ChatInputField(
            onSendPressed: () {
              onPressedCalled = true;
            },
          ),
        ),
      ),
    );

    // Tap the send IconButton
    await tester.tap(find.byIcon(Icons.send));
    // Rebuild the widget after the tap
    await tester.pump();

    // Check if the callback was called
    expect(onPressedCalled, isTrue);
  });
}
