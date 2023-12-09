import 'package:auto_gpt_flutter_client/models/test_suite.dart';
import 'package:flutter/material.dart';

class TestSuiteListTile extends StatelessWidget {
  final TestSuite testSuite;
  final VoidCallback onTap;

  const TestSuiteListTile({
    Key? key,
    required this.testSuite,
    required this.onTap,
  }) : super(key: key);

  Widget build(BuildContext context) {
    // Determine the width of the TaskView
    double taskViewWidth = MediaQuery.of(context).size.width;
    double tileWidth = taskViewWidth - 20;
    if (tileWidth > 260) {
      tileWidth = 260;
    }

    return GestureDetector(
      onTap: () {
        onTap();
      },
      child: Material(
        // Use a transparent color to avoid any unnecessary color overlay
        color: Colors.transparent,
        child: Padding(
          // Provide a horizontal padding to ensure the tile does not touch the edges
          padding: const EdgeInsets.symmetric(horizontal: 10.0),
          child: Container(
            // Width and height specifications for the tile
            width: tileWidth,
            height: 50,
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(8.0),
            ),
            child: Row(
              children: [
                // Space from the left edge of the tile
                const SizedBox(width: 8),
                // Message bubble icon indicating a test suite
                const Icon(Icons.play_arrow, color: Colors.black),
                const SizedBox(width: 8),
                // Test suite title
                Expanded(
                  child: Text(
                    testSuite.timestamp,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: const TextStyle(color: Colors.black),
                  ),
                ),
                // Disclosure indicator (arrow pointing right)
                const Icon(Icons.chevron_right, color: Colors.grey),
                const SizedBox(width: 8),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
