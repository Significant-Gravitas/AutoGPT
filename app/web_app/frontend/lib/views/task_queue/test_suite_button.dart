import 'package:auto_gpt_flutter_client/constants/app_colors.dart';
import 'package:auto_gpt_flutter_client/models/test_option.dart';
import 'package:flutter/material.dart';

class TestSuiteButton extends StatefulWidget {
  final bool isDisabled;
  final Function(String) onOptionSelected;
  final Function(String) onPlayPressed;
  String selectedOptionString;

  TestSuiteButton({
    this.isDisabled = false,
    required this.onOptionSelected,
    required this.onPlayPressed,
    required this.selectedOptionString,
  });

  @override
  _TestSuiteButtonState createState() => _TestSuiteButtonState();
}

class _TestSuiteButtonState extends State<TestSuiteButton> {
  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        // Dropdown button with test options
        Expanded(
          // Added Expanded to make sure it takes the available space
          child: PopupMenuButton<String>(
            enabled: !widget.isDisabled,
            onSelected: (value) {
              setState(() {
                widget.selectedOptionString = value;
              });
              widget.onOptionSelected(widget.selectedOptionString);
            },
            itemBuilder: (BuildContext context) {
              return [
                PopupMenuItem(
                  value: TestOption.runSingleTest.description,
                  child: Text(TestOption.runSingleTest.description),
                ),
                PopupMenuItem(
                  value: TestOption
                      .runTestSuiteIncludingSelectedNodeAndAncestors
                      .description,
                  child: Text(TestOption
                      .runTestSuiteIncludingSelectedNodeAndAncestors
                      .description),
                ),
                PopupMenuItem(
                  value: TestOption.runAllTestsInCategory.description,
                  child: Text(TestOption.runAllTestsInCategory.description),
                ),
              ];
            },
            child: Container(
              height: 50,
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              decoration: BoxDecoration(
                color: widget.isDisabled ? Colors.grey : AppColors.primaryLight,
                borderRadius: BorderRadius.circular(8.0),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Flexible(
                    child: Text(
                      widget.selectedOptionString,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 12.50,
                        fontFamily: 'Archivo',
                        fontWeight: FontWeight.w400,
                      ),
                      overflow: TextOverflow.ellipsis,
                      maxLines: 2,
                    ),
                  ),
                  const Icon(
                    Icons.arrow_drop_down,
                    color: Colors.white,
                  )
                ],
              ),
            ),
          ),
        ),
        // Play button
        const SizedBox(width: 10),
        SizedBox(
          height: 50,
          child: ElevatedButton(
            style: ElevatedButton.styleFrom(
              backgroundColor:
                  widget.isDisabled ? Colors.grey : AppColors.primaryLight,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(8.0),
              ),
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              elevation: 5.0,
            ),
            onPressed: widget.isDisabled
                ? null
                : () {
                    widget.onPlayPressed(widget.selectedOptionString);
                  },
            child: const Icon(
              Icons.play_arrow,
              color: Colors.white,
              size: 24,
            ),
          ),
        ),
      ],
    );
  }
}
