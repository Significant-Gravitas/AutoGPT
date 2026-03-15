import 'package:flutter/material.dart';

class UserMessageTile extends StatelessWidget {
  final String message;

  // Constructor takes the user message as a required parameter
  const UserMessageTile({
    Key? key,
    required this.message,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        // Calculate the width of the chat view based on the constraints provided
        double chatViewWidth = constraints.maxWidth;

        // Determine the width of the message tile based on the chat view width
        double tileWidth = (chatViewWidth >= 1000) ? 900 : chatViewWidth - 40;

        return Align(
          alignment: Alignment.center,
          child: Container(
            width: tileWidth,
            // Minimum height constraint for the container
            constraints: const BoxConstraints(
              minHeight: 50,
            ),
            // Margin and padding for styling
            margin: const EdgeInsets.symmetric(vertical: 8),
            padding: const EdgeInsets.symmetric(horizontal: 20),
            // Decoration to style the container with a white background, thin black border, and small corner radius
            decoration: BoxDecoration(
              color: Colors.white,
              border: Border.all(color: Colors.black, width: 0.5),
              borderRadius: BorderRadius.circular(4),
            ),
            child: Row(
              children: [
                // "User" label with custom styling
                const Text(
                  "User",
                  style: TextStyle(
                    color: Colors.black,
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(width: 20),
                // Expanded widget to accommodate the message text
                Expanded(
                  child: Container(
                    // Padding for the text content
                    padding: const EdgeInsets.fromLTRB(0, 10, 20, 10),
                    // Displaying the user message with no max line limit
                    child: SelectableText(
                      message,
                      maxLines: null,
                    ),
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}
