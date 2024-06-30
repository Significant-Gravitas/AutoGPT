from __future__ import annotations
from ..registry import action
from forge.sdk import ForgeLogger, Agent
import os
import subprocess
from typing import Dict

LOG = ForgeLogger(__name__)


@action(
    name="deploy_code",
    description="Deploy the Anchor project to Solana Devnet",
    parameters=[
        {
            "name": "project_path",
            "description": "Path to the Anchor project directory",
            "type": "string",
            "required": True,
        }
    ],
    output_type="str",
)
async def deploy_code(agent: Agent, task_id: str, project_path: str) -> str:
    try:
        # Ensure that the Solana CLI is installed and configured to use the devnet
        subprocess.run(['solana', 'config', 'set', '--url',
                       'https://api.devnet.solana.com'], check=True)
        subprocess.run(['solana', 'config', 'set', '--keypair',
                       os.path.expanduser('~/.config/solana/id.json')], check=True)

        # Build the Anchor project
        result = subprocess.run(
            ['anchor', 'build'], cwd=project_path, capture_output=True, text=True)
        if result.returncode != 0:
            LOG.error(f"Build failed with errors: {result.stderr}")
            return f"Build failed: {result.stderr}"

        # Deploy the Anchor project to Devnet
        result = subprocess.run(
            ['anchor', 'deploy'], cwd=project_path, capture_output=True, text=True)
        if result.returncode != 0:
            LOG.error(f"Deployment failed with errors: {result.stderr}")
            return f"Deployment failed: {result.stderr}"

        LOG.info(f"Deployment successful: {result.stdout}")
        return f"Deployment successful: {result.stdout}"

    except subprocess.CalledProcessError as e:
        LOG.error(f"Error during deployment: {e}")
        return f"Deployment process failed: {e}"
    except Exception as e:
        LOG.error(f"Unexpected error during deployment: {e}")
        return f"Unexpected error: {e}"
