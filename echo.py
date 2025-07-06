while True:
    try:
        line = input()
    except EOFError:
        break
    if not line:
        break
    print(line)
