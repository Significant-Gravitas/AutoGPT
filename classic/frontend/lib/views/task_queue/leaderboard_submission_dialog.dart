import 'package:auto_gpt_flutter_client/constants/app_colors.dart';
import 'package:auto_gpt_flutter_client/utils/uri_utility.dart';
import 'package:auto_gpt_flutter_client/viewmodels/task_queue_viewmodel.dart';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

class LeaderboardSubmissionDialog extends StatefulWidget {
  final Function(String, String, String)? onSubmit;
  // TODO: Create a view model for this class and remove the TaskQueueViewModel
  final TaskQueueViewModel viewModel;

  const LeaderboardSubmissionDialog({
    Key? key,
    this.onSubmit,
    required this.viewModel,
  }) : super(key: key);

  @override
  _LeaderboardSubmissionDialogState createState() =>
      _LeaderboardSubmissionDialogState();
}

class _LeaderboardSubmissionDialogState
    extends State<LeaderboardSubmissionDialog> {
  final TextEditingController _teamNameController = TextEditingController();
  final TextEditingController _repoUrlController = TextEditingController();
  final TextEditingController _commitShaController = TextEditingController();

  String? _teamNameError;
  String? _repoUrlError;
  String? _commitShaError;

  @override
  void initState() {
    super.initState();
    _initSharedPreferences();
  }

  Future<void> _initSharedPreferences() async {
    // Using the SharedPreferencesService from the viewModel to get stored values
    _teamNameController.text =
        await widget.viewModel.prefsService.getString('teamName') ?? '';
    _repoUrlController.text =
        await widget.viewModel.prefsService.getString('repoUrl') ?? '';
    _commitShaController.text =
        await widget.viewModel.prefsService.getString('commitSha') ?? '';
  }

  void _validateAndSubmit() async {
    setState(() {
      _teamNameError = null;
      _repoUrlError = null;
      _commitShaError = null;
    });

    bool isValid = true;

    if (_teamNameController.text.isEmpty) {
      isValid = false;
      _teamNameError = 'Team Name is required';
    }

    if (_repoUrlController.text.isEmpty) {
      isValid = false;
      _repoUrlError = 'Repo URL is required';
    } else if (!UriUtility.isURL(_repoUrlController.text)) {
      isValid = false;
      _repoUrlError = 'Invalid URL format';
    } else if (!(await UriUtility()
        .isValidGitHubRepo(_repoUrlController.text))) {
      isValid = false;
      _repoUrlError = 'Not a valid GitHub repository';
    }

    if (_commitShaController.text.isEmpty) {
      isValid = false;
      _commitShaError = 'Commit SHA is required';
    }

    if (isValid) {
      print('Valid leaderboard submission parameters!');
      await _saveToSharedPreferences();
      widget.onSubmit?.call(_teamNameController.text, _repoUrlController.text,
          _commitShaController.text);
      Navigator.of(context).pop();
    } else {
      setState(() {});
    }
  }

  Future<void> _saveToSharedPreferences() async {
    // Using the prefsService to save the values
    await widget.viewModel.prefsService
        .setString('teamName', _teamNameController.text);
    await widget.viewModel.prefsService
        .setString('repoUrl', _repoUrlController.text);
    await widget.viewModel.prefsService
        .setString('commitSha', _commitShaController.text);
  }

  @override
  Widget build(BuildContext context) {
    final containerHeight = 324.0 +
        (_teamNameError == null ? 0 : 22) +
        (_repoUrlError == null ? 0 : 22) +
        (_commitShaError == null ? 0 : 22);
    return Dialog(
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(8.0),
      ),
      child: Container(
        width: 260,
        height: containerHeight,
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Title
            const Text(
              'Leaderboard Submission',
              textAlign: TextAlign.center,
              style: TextStyle(
                color: Colors.black,
                fontSize: 16,
                fontFamily: 'Archivo',
                fontWeight: FontWeight.w600,
              ),
            ),
            const SizedBox(height: 14),
            // Team Name Field
            const Text('Team Name'),
            TextField(
              controller: _teamNameController,
              decoration: InputDecoration(
                hintText: 'Keyboard Warriors',
                errorText: _teamNameError,
                border: OutlineInputBorder(
                  borderSide: BorderSide(
                    color: _teamNameError != null ? Colors.red : Colors.grey,
                  ),
                ),
              ),
            ),
            const SizedBox(height: 8),
            // Github Repo URL Field
            const Text('Github Repo URL'),
            TextField(
              controller: _repoUrlController,
              decoration: InputDecoration(
                hintText: 'https://github.com/KeyboardWarriors/BestAgentEver',
                errorText: _repoUrlError,
                border: OutlineInputBorder(
                  borderSide: BorderSide(
                    color: _repoUrlError != null ? Colors.red : Colors.grey,
                  ),
                ),
              ),
            ),
            const SizedBox(height: 8),
            // Commit SHA Field
            const Text('Commit SHA'),
            TextField(
              controller: _commitShaController,
              decoration: InputDecoration(
                hintText: '389131f2ab78c2cc5bdd2ec257be2d18b3a63da3',
                errorText: _commitShaError,
                border: OutlineInputBorder(
                  borderSide: BorderSide(
                    color: _commitShaError != null ? Colors.red : Colors.grey,
                  ),
                ),
              ),
            ),
            const SizedBox(height: 14),
            // Buttons
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                // Cancel Button
                SizedBox(
                  width: 106,
                  height: 28,
                  child: ElevatedButton(
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.grey,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8.0),
                      ),
                    ),
                    onPressed: () => Navigator.of(context).pop(),
                    child: const Text(
                      'Cancel',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 12.50,
                        fontFamily: 'Archivo',
                        fontWeight: FontWeight.w400,
                      ),
                    ),
                  ),
                ),
                SizedBox(width: 8),
                // Submit Button
                SizedBox(
                  width: 106,
                  height: 28,
                  child: ElevatedButton(
                    style: ElevatedButton.styleFrom(
                      backgroundColor: AppColors.primaryLight,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8.0),
                      ),
                    ),
                    onPressed: _validateAndSubmit,
                    child: const Text(
                      'Submit',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 12.50,
                        fontFamily: 'Archivo',
                        fontWeight: FontWeight.w400,
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    _teamNameController.dispose();
    _repoUrlController.dispose();
    _commitShaController.dispose();
    super.dispose();
  }
}
