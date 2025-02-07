import sys

def Read(filename, lines=20):
    try:
        with open(filename, 'r', encoding='utf-8') as fs:
            for i, line in enumerate(fs):  # Use enumerate for cleaner counting
                if i >= lines:
                    break
                print(line.strip())  # Print the stripped line
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.", file=sys.stderr)  # Print to stderr for errors
        return []  # Return empty list to indicate failure
    except Exception as e: # Catch any other exceptions
        print(f"An error occurred: {e}", file=sys.stderr)
        return []
    return []  # Return an empty list (as in the original code)


if __name__ == "__main__":
    arg_len = len(sys.argv)  # Use more descriptive variable name
    if arg_len < 2 or arg_len > 3:  # Simplified condition
        print('Usage: python SampleReader.py $filename [$lines=20]', file=sys.stderr)  # Print usage to stderr
        sys.exit(1) # Exit with a non-zero code to indicate failure
    else:
        filename = sys.argv[1]
        if arg_len == 3:
            try:
                lines = int(sys.argv[2])
                Read(filename, lines)
            except ValueError:
                print("Error: Number of lines must be an integer.", file=sys.stderr)
                sys.exit(1)
        else:
            Read(filename)
