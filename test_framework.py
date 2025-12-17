from framework import import_pdfs, import_sqlite, test_rag


def run_all():
    """
    Convenience runner for local debugging.
    """
    import_pdfs()
    import_sqlite()
    test_rag()


if __name__ == "__main__":
    run_all()
