import 'package:flutter/material.dart';

class NewTaskButton extends StatelessWidget {
  final VoidCallback onPressed;

  const NewTaskButton({Key? key, required this.onPressed}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    // Determine the width of the TaskView
    double taskViewWidth = MediaQuery.of(context).size.width;
    double buttonWidth = taskViewWidth - 20;
    if (buttonWidth > 260) {
      buttonWidth = 260;
    }

    return ElevatedButton(
      onPressed: onPressed,
      style: ButtonStyle(
        // Set the button's background color
        backgroundColor: MaterialStateProperty.all<Color>(Colors.white),
        // Set the button's edge
        side: MaterialStateProperty.all<BorderSide>(
            const BorderSide(color: Colors.black, width: 0.5)),
        // Set the button's shape with rounded corners
        shape: MaterialStateProperty.all<RoundedRectangleBorder>(
          RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(8.0),
          ),
        ),
      ),
      child: SizedBox(
        width: buttonWidth,
        height: 50,
        child: const Row(
          children: [
            // Black plus icon
            Icon(Icons.add, color: Colors.black),
            SizedBox(width: 8),
            // "New Task" label
            Text('New Task', style: TextStyle(color: Colors.black)),
          ],
        ),
      ),
    );
  }
}
