def pytest_addoption(parser):
    parser.addoption(
        "--dataset",
        action="store",
        default=None,
        help="Path to a generated ECSFM dataset directory for QC tests",
    )
