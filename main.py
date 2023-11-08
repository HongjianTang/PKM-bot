import sys, select, os
import process
import prompt.prompt
from dataReceive import DataReceiver
from replit import db


def processInput():
  # Create an instance of DataReceiver
  receiver = DataReceiver()

  while True:
    # Get input from the user
    user_input = input("Enter a message (or type 'quit' to exit): ")

    # Check if the user wants to quit
    if user_input.lower() == 'quit':
      process.train()
      return  # Exit the function, returning to the main menu

    # Call the receiveData method with the user input
    receiver.receiveData(user_input)


def cleanupDatabase():
  for key in db.keys():
    del (db[key])


def main():
  if "API_SECRET" not in os.environ:
    print("You must set an API_SECRET using the Secrets tool", file=sys.stderr)
  elif "OPENAI_API_KEY" not in os.environ:
    print("You must set an OPENAI_API_KEY using the Secrets tool",
          file=sys.stderr)
  else:
    print("== START ==")
    while True:  # Keep showing the menu until the program is exited
      print()
      print(
        "1: Receive Message \n2: Show View \n3: Exit \n4: Cleanup Database \n",
        end="")

      input_ready, output_ready, error_ready = select.select([sys.stdin], [],
                                                             [])
      if input_ready:
        choice = sys.stdin.readline().strip()
        if choice == "1":
          print("Please input message")
          processInput()
          # process.train()
        elif choice == "2":
          print("BOT CONVERSATION MODE")
          prompt.prompt.runPrompt()
        elif choice == "3":
          print("Exiting...")
          break  # Exit the while loop and end the program
        elif choice == "4":
          cleanupDatabase()
        else:
          print("Invalid choice. Please select a valid option.")


if __name__ == "__main__":
  main()
