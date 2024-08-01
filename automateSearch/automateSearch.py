import tkinter as tk
from tkinter import Entry, Label, Button
import webbrowser

# Define the main window
root = tk.Tk()
root.title("YOUR AI ASSISTANT")

# Adding Background color
root.configure(bg='steelblue')

# Define the function to automate youtube search
def search_youtube():
    query = entry.get()
    url = f"https://www.youtube.com/results?search_query={query}"
    webbrowser.open(url)

# Define the function to automate google search
def search_google():
    query = entry.get()
    url = f"https://www.google.com/search?q={query}"
    webbrowser.open(url)

# Define the function to automate instagram search
def search_instagram():
    Username = entry.get().replace('@', "")
    url = f"https://www.instagram.com/{Username}/"
    webbrowser.open(url)

# Create Input Field
Label(root, text="Enter your command :").pack(pady=10)
entry = Entry(root, width=50)
entry.pack(pady=10)

# Create Buttons
Button(root, text="Search on Youtube", command=search_youtube).pack(pady=5)
Button(root, text="Search on Google", command=search_google).pack(pady=5)
Button(root, text="Search on Instagram", command=search_instagram).pack(pady=5)

# Run the GUI event loop
root.mainloop()