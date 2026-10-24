_log_file = None


def set_log_file(file_obj):
    global _log_file
    _log_file = file_obj


def get_log_file():
    return _log_file


def close_log_file():
    global _log_file
    if _log_file is not None:
        _log_file.close()
        _log_file = None


def log_print(*args, **kwargs):
    print(*args, **kwargs)
    if _log_file is not None:
        message = " ".join(str(arg) for arg in args)
        end = kwargs.get("end", "\n")
        _log_file.write(message + end)
        _log_file.flush()
