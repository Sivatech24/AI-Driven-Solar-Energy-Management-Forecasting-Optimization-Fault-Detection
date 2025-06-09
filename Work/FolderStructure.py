import os

def display_tree(start_path, indent=""):
    for item in os.listdir(start_path):
        path = os.path.join(start_path, item)
        if os.path.isdir(path):
            print(indent + "📁 " + item)
            display_tree(path, indent + "    ")
        else:
            print(indent + "📄 " + item)

# Example usage
if __name__ == "__main__":
    root_dir = "Deepseek"  # Change this to your target directory
    print(f"Structure of '{root_dir}':\n")
    display_tree(root_dir)
