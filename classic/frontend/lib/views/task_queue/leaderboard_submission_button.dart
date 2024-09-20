import 'package:auto_gpt_flutter_client/constants/app_colors.dart';
import 'package:flutter/material.dart';

class LeaderboardSubmissionButton extends StatelessWidget {
  final VoidCallback? onPressed;
  final bool isDisabled;

  LeaderboardSubmissionButton(
      {required this.onPressed, this.isDisabled = false});

  @override
  Widget build(BuildContext context) {
    final button = SizedBox(
      height: 50,
      child: ElevatedButton(
        style: ElevatedButton.styleFrom(
          backgroundColor: isDisabled ? Colors.grey : AppColors.primaryLight,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(8.0),
          ),
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          elevation: 5.0,
        ),
        onPressed: isDisabled ? null : onPressed,
        child: const Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              'Submit to leaderboard',
              style: TextStyle(
                color: Colors.white,
                fontSize: 12.50,
                fontFamily: 'Archivo',
                fontWeight: FontWeight.w400,
              ),
            ),
            SizedBox(width: 10),
            Icon(
              Icons.emoji_events,
              color: Colors.white,
              size: 24,
            ),
          ],
        ),
      ),
    );

    return isDisabled
        ? Tooltip(
            message:
                "You must complete a test suite before submitting to the leaderboard.",
            child: button,
          )
        : button;
  }
}
