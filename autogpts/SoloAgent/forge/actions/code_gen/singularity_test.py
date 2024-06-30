import os
import re
from forge.sdk import Agent, LocalWorkspace


# Mock GPT response for testing
mock_response_content = """
// anchor-lib.rs
use anchor_lang::prelude::*;

pub mod instructions;
pub mod errors;

use instructions::*;

declare_id!("Fg6PaFpoGXkYsidMpWxTWt8sWAb2uZ7AcfQkwJDrsVwC");

#[program]
mod my_anchor_program {
    use super::*;

    pub fn initialize(ctx: Context<Initialize>) -> Result<()> {
        instructions::initialize(ctx)
    }
}

// anchor-instructions.rs
use anchor_lang::prelude::*;
use crate::errors::MyProgramError;

#[derive(Accounts)]
pub struct Initialize<'info> {
    #[account(init, payer = user, space = 8 + 8)]
    pub my_account: Account<'info, MyAccount>,
    #[account(mut)]
    pub user: Signer<'info>,
    pub system_program: Program<'info, System>,
}

pub fn initialize(ctx: Context<Initialize>) -> Result<()> {
    let my_account = &mut ctx.accounts.my_account;
    my_account.data = 0;
    Ok(())
}

// errors.rs
use anchor_lang::prelude::*;

#[error_code]
pub enum MyProgramError {
    #[msg("An error occurred.")]
    GeneralError,
}
"""

# Hardcoded Cargo.toml content
cargo_toml_content = """
[package]
name = "my_anchor_program"
version = "0.1.0"
edition = "2018"

[dependencies]
anchor-lang = "0.30.1"
"""



# Function to parse the response content
def parse_response_content(response_content: str) -> dict:
    parts = {}
    current_filename = None

    # Regular expression to match filenames
    filename_pattern = re.compile(r'^//\s*(.*\.(rs|html|js|css|json|config\.js))$')

    for line in response_content.split('\n'):
        # Check if the line matches the filename pattern
        match = filename_pattern.match(line)
        if match:
            current_filename = match.group(1).strip()
            parts[current_filename] = ''
        elif current_filename:
            parts[current_filename] += line + '\n'

    # Clean up the content by removing unwanted characters and trimming whitespace
    for key in parts:
        parts[key] = re.sub(r'```|rust|html|js|css|json|config\.js', '', parts[key]).strip()

    return parts

# Function to write files based on parsed content
def write_files(parts: dict, base_path: str):
    os.makedirs(base_path, exist_ok=True)

    # Write hardcoded Cargo.toml and Anchor.toml
    with open(os.path.join(base_path, 'Cargo.toml'), 'w') as f:
        f.write(cargo_toml_content)
        print("Written Cargo.toml")

    with open(os.path.join(base_path, 'Anchor.toml'), 'w') as f:
        f.write(anchor_toml_content)
        print("Written Anchor.toml")

    # Write other files based on parsed content
    for filename, content in parts.items():
        file_path = os.path.join(base_path, filename)
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Written {filename}")

# Test the functionality
def test_generate_files():
    parsed_parts = parse_response_content(mock_response_content)
    base_path = Agent.workspace.base_path if isinstance(
        Agent.workspace, LocalWorkspace) else str(Agent.workspace.base_path)
    project_path = os.path.join(base_path, task_id)
    write_files(parsed_parts, base_path)
    print("All files written successfully.")

# Run the test
if __name__ == "__main__":
    test_generate_files()
