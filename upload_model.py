from skops import hub_utils

token = "XXXX"

hub_utils.push(
    repo_id="AndyJamesTurner/suicideDetector",
    source="suicide-detector",
    commit_message="Improved documentation",
    token=token
)