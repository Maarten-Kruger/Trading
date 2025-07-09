import random


def main():
    print("Welcome to the Guess the Number Game!")
    number = random.randint(1, 100)
    attempts = 0
    while True:
        try:
            guess = int(input("Guess a number between 1 and 100: "))
        except ValueError:
            print("Please enter a valid integer.")
            continue
        attempts += 1
        if guess < number:
            print("Too low! Try again.")
        elif guess > number:
            print("Too high! Try again.")
        else:
            print(f"Congratulations! You guessed the number in {attempts} attempts.")
            break


if __name__ == "__main__":
    main()
