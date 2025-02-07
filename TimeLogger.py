import datetime

log_message = ''  # More descriptive variable name
time_marks = {}  # More descriptive variable name
save_default = False  # More descriptive variable name


def log(msg, save=None, oneline=False):
    global log_message
    global save_default
    time = datetime.datetime.now()
    formatted_msg = f"{time}: {msg}"  # Use f-string for cleaner formatting

    if save is not None:
        if save:
            log_message += formatted_msg + '\n'
    elif save_default:
        log_message += formatted_msg + '\n'

    if oneline:
        print(formatted_msg, end='\r', flush=True)  # Add flush=True for immediate output
    else:
        print(formatted_msg)


def mark_time(marker):  # More descriptive function name
    global time_marks
    time_marks[marker] = datetime.datetime.now()


def spent_time(marker):  # More descriptive function name
    global time_marks
    if marker not in time_marks:
        error_msg = f"LOGGER ERROR, marker {marker} not found"  # Use f-string
        time = datetime.datetime.now()
        formatted_msg = f"{time}: {error_msg}"
        print(formatted_msg)
        return None  # Return None to indicate failure

    return datetime.datetime.now() - time_marks[marker]


def spent_too_long(marker, days=0, hours=0, minutes=0, seconds=0):  # More descriptive function name, clearer argument names
    delta = datetime.timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds) # Calculate delta once
    time_spent = spent_time(marker)
    if time_spent is None: # Handle marker not found
        return False
    return time_spent >= delta


if __name__ == '__main__':
    log("This is a test message.")
    mark_time("start")
    # ... some code that takes time ...
    time.sleep(2)  # Simulate some work
    elapsed = spent_time("start")
    if elapsed:
        log(f"Time spent: {elapsed}")
    log("Another message.")

    if spent_too_long("start", seconds=1):
        log("Time limit exceeded!")

    print(log_message) # Print the accumulated log message
