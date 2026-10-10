"""Builder #2 implements. ``python -m model.train``

Reads prepared loans (dev window, default_flag not NA), fits the PD model, freezes score-band
cut-points, writes the model under ``paths.model_dir()`` and the dev reference scoring to
``paths.scored_path("dev")`` (out-of-time dev sample, schema = SCORED_COLUMNS).
"""


def main() -> None:
    raise NotImplementedError("builder #2: model/train.py")


if __name__ == "__main__":
    main()
