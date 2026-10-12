from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class OnboardingWizardConflict(RuntimeError):
    def __init__(self) -> None:
        super().__init__(
            "Onboarding progress changed in another session. Reload before saving."
        )


WizardStep = Literal[
    "team",
    "autopilot",
    "role",
    "painPoints",
    "hire",
    "subscription",
    "connect",
    "preparing",
]
WizardText = Annotated[str, Field(pattern=r"^[^\x00-\x08\x0b\x0c\x0e-\x1f\x7f]*$")]


class OnboardingWizardProgress(BaseModel):
    """Saved wizard answers and navigation, independent of completion or access."""

    model_config = ConfigDict(extra="forbid", strict=True)

    version: Literal[1]
    currentStep: WizardStep
    completedSteps: list[WizardStep] = Field(max_length=8)
    role: WizardText = Field(max_length=100)
    otherRole: WizardText = Field(max_length=100)
    painPoints: list[Annotated[WizardText, Field(min_length=1, max_length=200)]] = (
        Field(max_length=20)
    )
    otherPainPoint: WizardText = Field(max_length=2000)
    selectedBilling: Literal["monthly", "yearly"]
    hasUserSelectedBilling: bool
    selectedCountryCode: str = Field(min_length=2, max_length=2, pattern="^[A-Z]{2}$")
    hiredTemplateIds: list[
        Annotated[WizardText, Field(min_length=1, max_length=128)]
    ] = Field(max_length=100)
