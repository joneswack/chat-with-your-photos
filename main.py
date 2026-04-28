import agent
import setup
from utils import indexed_folders, pick

_WELCOME = "Welcome to Chat with Your Photos!"


def main():
    print(_WELCOME)
    while True:
        folders = indexed_folders()
        count = len(folders)

        if count == 0:
            print("You currently have 0 folders indexed.\n")
            options = ["Add a new folder to your collection", "Quit"]
        else:
            folder_word = "folder" if count == 1 else "folders"
            print(
                f"You currently have {count} {folder_word} indexed. "
                f"Either chat with them or add a new folder to your collection.\n"
            )
            folder_labels = {f"Chat with {f}": f for f in folders}
            options = list(folder_labels) + ["Add a new folder", "Quit"]

        try:
            choice = pick("Select an option:", options)
        except (KeyboardInterrupt, EOFError):
            break

        if choice == "Quit":
            break
        elif choice in ("Add a new folder", "Add a new folder to your collection"):
            setup.main()
            print()
        else:
            agent.main(folder_labels[choice])
            print()


if __name__ == "__main__":
    main()
