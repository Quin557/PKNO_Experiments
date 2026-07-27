from __future__ import annotations


class _NoKoopmanLabPlots:
    """Falsey sentinel for KoopmanLab releases that use ``loc == 0 & flag``."""

    def __bool__(self) -> bool:
        return False

    def __rand__(self, other: object) -> int:
        return -1


_NO_KOOPMANLAB_OPTIONAL_OUTPUTS = _NoKoopmanLabPlots()


def koopmanlab_optional_output_flag(save_outputs: bool) -> bool | _NoKoopmanLabPlots:
    return True if save_outputs else _NO_KOOPMANLAB_OPTIONAL_OUTPUTS
