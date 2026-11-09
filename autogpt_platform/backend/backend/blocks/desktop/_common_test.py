"""Volume naming and mount maps for the E2B desktop/shell surfaces."""

from backend.blocks.desktop._common import (
    SHARED_PATH,
    WORKSPACE_PATH,
    expert_volume_name,
    user_volume_name,
    workspace_volume_mounts,
)


class TestWorkspaceVolumeMounts:
    def test_plain_session_mounts_user_volume_as_home(self):
        assert workspace_volume_mounts("u1") == {WORKSPACE_PATH: user_volume_name("u1")}

    def test_expert_gets_own_home_plus_users_shared_workspace(self):
        mounts = workspace_volume_mounts("u1", "exp-1")
        assert mounts == {
            WORKSPACE_PATH: expert_volume_name("exp-1"),
            SHARED_PATH: user_volume_name("u1"),
        }
        # The expert's home is its own, never the user's volume.
        assert mounts[WORKSPACE_PATH] != user_volume_name("u1")

    def test_expert_without_user_has_nothing_to_share(self):
        assert workspace_volume_mounts(None, "exp-1") == {
            WORKSPACE_PATH: expert_volume_name("exp-1")
        }

    def test_no_owner_means_no_mounts(self):
        assert workspace_volume_mounts(None) == {}
        assert workspace_volume_mounts("") == {}

    def test_volume_names_are_scoped_by_kind(self):
        assert expert_volume_name("abc") == "autogpt-expert-abc"
        assert user_volume_name("abc") == "autogpt-user-abc"
        assert expert_volume_name("abc") != user_volume_name("abc")
