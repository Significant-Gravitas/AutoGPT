"""
Block verification script to check that all blocks can be instantiated and have valid schemas.
This script can be run to verify blocks without making actual API calls.
"""

import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Type

from pydantic import ValidationError

from backend.data.model import APIKeyCredentials
from backend.sdk import Block


@dataclass
class BlockVerificationResult:
    """Result of block verification."""

    block_name: str
    success: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class BlockVerifier:
    """Verify blocks without making API calls."""

    def __init__(self):
        self.results: List[BlockVerificationResult] = []

    def verify_block_class(self, block_class: Type[Block]) -> BlockVerificationResult:
        """Verify a single block class."""
        result = BlockVerificationResult(block_name=block_class.__name__, success=True)

        try:
            # 1. Check if block can be instantiated
            block = block_class()

            # 2. Verify block has required attributes
            required_attrs = ["id", "description", "input_schema", "output_schema"]
            for attr in required_attrs:
                if not hasattr(block, attr):
                    result.errors.append(f"Missing required attribute: {attr}")
                    result.success = False

            # 3. Verify input schema
            if hasattr(block, "Input"):
                try:
                    # Try to create an instance with empty data to check required fields
                    input_class = getattr(block, "Input")
                    _ = input_class()
                except ValidationError as e:
                    # This is expected if there are required fields
                    required_fields = [
                        str(err["loc"][0])
                        for err in e.errors()
                        if err["type"] == "missing"
                    ]
                    if required_fields:
                        result.warnings.append(
                            f"Required input fields: {', '.join(required_fields)}"
                        )

                # Check for credentials field
                input_class = getattr(block, "Input")
                if hasattr(input_class, "__fields__"):
                    fields_dict = getattr(input_class, "__fields__")
                    cred_fields = [
                        name
                        for name in fields_dict.keys()
                        if "credentials" in name.lower()
                    ]
                    if cred_fields:
                        result.warnings.append(
                            f"Credential fields found: {', '.join(cred_fields)}"
                        )

            # 4. Verify output schema
            if hasattr(block, "Output"):
                output_fields = []
                output_class = getattr(block, "Output", None)
                if output_class and hasattr(output_class, "__fields__"):
                    output_fields = list(getattr(output_class, "__fields__").keys())
                    if output_fields:
                        result.warnings.append(
                            f"Output fields: {', '.join(output_fields)}"
                        )

            # 5. Verify run method
            if not hasattr(block, "run"):
                result.errors.append("Missing run method")
                result.success = False
            else:
                # Check if run method is async
                if not inspect.iscoroutinefunction(block.run):
                    result.errors.append("run method must be async")
                    result.success = False

            # 6. Check block ID format
            if hasattr(block, "id"):
                block_id = block.id
                if not isinstance(block_id, str) or len(block_id) != 36:
                    result.warnings.append(
                        f"Block ID might not be a valid UUID: {block_id}"
                    )

        except Exception as e:
            result.errors.append(f"Failed to instantiate block: {str(e)}")
            result.success = False

        return result

    def verify_provider_blocks(
        self, provider_name: str
    ) -> List[BlockVerificationResult]:
        """Verify all blocks from a specific provider."""
        results = []

        # Import provider module dynamically
        try:
            if provider_name == "airtable":
                from backend.blocks import airtable

                module = airtable
            elif provider_name == "baas":
                from backend.blocks import baas

                module = baas
            elif provider_name == "elevenlabs":
                from backend.blocks import elevenlabs

                module = elevenlabs
            else:
                return results

            # Get all block classes from the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    inspect.isclass(attr)
                    and issubclass(attr, Block)
                    and attr is not Block
                    and "Block" in attr_name
                ):
                    result = self.verify_block_class(attr)
                    results.append(result)
                    self.results.append(result)

        except ImportError as e:
            error_result = BlockVerificationResult(
                block_name=f"{provider_name}_import",
                success=False,
                errors=[f"Failed to import provider: {str(e)}"],
            )
            results.append(error_result)
            self.results.append(error_result)

        return results

    def generate_report(self) -> str:
        """Generate a verification report."""
        report_lines = ["Block Verification Report", "=" * 50, ""]

        # Summary
        total = len(self.results)
        successful = len([r for r in self.results if r.success])
        failed = total - successful

        report_lines.extend(
            [
                f"Total blocks verified: {total}",
                f"Successful: {successful}",
                f"Failed: {failed}",
                "",
                "Detailed Results:",
                "-" * 50,
                "",
            ]
        )

        # Group by success/failure
        for result in sorted(self.results, key=lambda r: (not r.success, r.block_name)):
            status = "✓" if result.success else "✗"
            report_lines.append(f"{status} {result.block_name}")

            if result.errors:
                for error in result.errors:
                    report_lines.append(f"  ERROR: {error}")

            if result.warnings:
                for warning in result.warnings:
                    report_lines.append(f"  WARNING: {warning}")

            report_lines.append("")

        return "\n".join(report_lines)

    async def test_block_execution(
        self, block_class: Type[Block], test_inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Test block execution with mock inputs (no API calls)."""
        try:
            block = block_class()

            # Create mock credentials if needed
            from pydantic import SecretStr

            from backend.sdk import ProviderName

            mock_creds = APIKeyCredentials(
                provider=ProviderName("airtable"), api_key=SecretStr("test-key")
            )

            # Create input instance
            input_class = getattr(block, "Input")
            input_data = input_class(**test_inputs)

            # Attempt to run the block (will fail at API call, but validates structure)
            outputs = []
            try:
                async for output in block.run(input_data, credentials=mock_creds):
                    outputs.append(output)
            except Exception as e:
                # Expected to fail at API call
                return {
                    "status": "execution_attempted",
                    "error": str(e),
                    "validates_structure": True,
                }

            return {"status": "unexpected_success", "outputs": outputs}

        except ValidationError as e:
            return {
                "status": "validation_error",
                "errors": e.errors(),
                "validates_structure": False,
            }
        except Exception as e:
            return {"status": "error", "error": str(e), "validates_structure": False}


def main():
    """Run block verification."""
    verifier = BlockVerifier()

    # Verify all providers
    providers = ["airtable", "baas", "elevenlabs"]

    print("Starting block verification...\n")

    for provider in providers:
        print(f"Verifying {provider} blocks...")
        results = verifier.verify_provider_blocks(provider)
        print(f"  Found {len(results)} blocks")

    # Generate and print report
    report = verifier.generate_report()
    print("\n" + report)

    # Save report to file
    with open("block_verification_report.txt", "w") as f:
        f.write(report)

    print("Report saved to block_verification_report.txt")

    # Return success if all blocks passed
    failed_count = len([r for r in verifier.results if not r.success])
    return failed_count == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
