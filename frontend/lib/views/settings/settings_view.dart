import 'package:auto_gpt_flutter_client/viewmodels/settings_viewmodel.dart';
import 'package:auto_gpt_flutter_client/views/settings/api_base_url_field.dart';
import 'package:flutter/material.dart';

/// [SettingsView] displays a list of settings that the user can configure.
/// It uses [SettingsViewModel] for state management and logic.
class SettingsView extends StatelessWidget {
  final SettingsViewModel viewModel;

  /// Constructor for [SettingsView], requiring an instance of [SettingsViewModel].
  const SettingsView({Key? key, required this.viewModel}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.grey,
        foregroundColor: Colors.black,
        title: const Text('Settings'),
      ),
      body: Column(
        children: [
          // All settings in a scrollable list
          Expanded(
            child: ListView(
              children: [
                // TODO: Add back dark mode toggle
                // Dark Mode Toggle
                // SwitchListTile(
                //   title: const Text('Dark Mode'),
                //   value: viewModel.isDarkModeEnabled,
                //   onChanged: viewModel.toggleDarkMode,
                // ),
                // const Divider(),
                // Developer Mode Toggle
                SwitchListTile(
                  title: const Text('Developer Mode'),
                  value: viewModel.isDeveloperModeEnabled,
                  onChanged: viewModel.toggleDeveloperMode,
                ),
                const Divider(),
                // Base URL Configuration
                const ListTile(
                  title: Center(child: Text('Agent Base URL')),
                ),
                ApiBaseUrlField(),
                const Divider(),
                // Continuous Mode Steps Configuration
                ListTile(
                  title: const Center(child: Text('Continuous Mode Steps')),
                  // User can increment or decrement the number of steps using '+' and '-' buttons.
                  subtitle: Row(
                    mainAxisAlignment:
                        MainAxisAlignment.center, // Centers the Row's content
                    children: [
                      IconButton(
                        icon: const Icon(Icons.remove),
                        onPressed: viewModel
                            .decrementContinuousModeSteps, // Decrement the number of steps.
                      ),
                      Text('${viewModel.continuousModeSteps} Steps'),
                      IconButton(
                        icon: const Icon(Icons.add),
                        onPressed: viewModel
                            .incrementContinuousModeSteps, // Increment the number of steps.
                      ),
                    ],
                  ),
                ),
                const Divider(),
              ],
            ),
          ),
          // Sign out button fixed at the bottom
          Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            child: ElevatedButton.icon(
              icon: const Icon(Icons.logout, color: Colors.black),
              label:
                  const Text('Sign Out', style: TextStyle(color: Colors.black)),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.white,
              ),
              onPressed: viewModel.signOut,
            ),
          ),
        ],
      ),
    );
  }
}
