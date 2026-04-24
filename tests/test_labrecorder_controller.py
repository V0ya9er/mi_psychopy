"""Quick test for LabRecorderCLIController path resolution logic."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LabRecorderConfig:
    cli_path: Path
    study_root: str
    path_template: str
    auto_record: bool = True
    stream_queries: tuple[str, ...] = ('name="obci_eeg1"', 'name="obci_eeg2"')


class LabRecorderCLIController:
    def __init__(self, config: LabRecorderConfig) -> None:
        self.config = config
        self.process = None
        self._xdf_path = ""

    def resolve_xdf_path(self, participant: str, session: str,
                         run: int, task: str, class_mode: str = "binary") -> str:
        path = self.config.path_template
        path = path.replace("%p", participant)
        path = path.replace("%s", session)
        path = path.replace("%c", class_mode)
        path = path.replace("%b", task)
        path = path.replace("%n", str(run))
        study_root = self.config.study_root.rstrip("/")
        full_path = f"{study_root}/{path}"
        self._xdf_path = full_path
        return full_path

    @property
    def is_recording(self) -> bool:
        return self.process is not None and self.process.poll() is None


def test_path_resolution():
    cfg = LabRecorderConfig(
        cli_path=Path("LabRecorder/LabRecorderCLI.exe"),
        study_root="D:/CSDIY/EEG/datasets/customs",
        path_template="%p/%s/%c/%b/run-%n.xdf",
    )
    lr = LabRecorderCLIController(cfg)

    xdf = lr.resolve_xdf_path("P001", "S001", 1, "pure_mi", "binary")
    expected = "D:/CSDIY/EEG/datasets/customs/P001/S001/binary/pure_mi/run-1.xdf"
    assert xdf == expected, f"Expected {expected}, got {xdf}"
    print(f"Test 1 OK: {xdf}")

    xdf2 = lr.resolve_xdf_path("P003", "S002", 5, "mi_p300", "ternary")
    expected2 = "D:/CSDIY/EEG/datasets/customs/P003/S002/ternary/mi_p300/run-5.xdf"
    assert xdf2 == expected2, f"Expected {expected2}, got {xdf2}"
    print(f"Test 2 OK: {xdf2}")

    assert not lr.is_recording
    print("Test 3 OK: is_recording=False when no process")

    print("\nAll LabRecorderCLIController tests passed!")


if __name__ == "__main__":
    test_path_resolution()
